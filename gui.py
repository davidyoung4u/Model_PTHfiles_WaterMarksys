# _*_ coding: utf-8 _*_
# @Time : 2025/4/9 14:27
# @Author : yang_4_ever
# @File : gui
# @Project : FinalWMsys

from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
                            QGroupBox, QDoubleSpinBox, QDialog, QDialogButtonBox, QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from torch.utils.data import DataLoader
import torch
import torchvision
from torchvision import transforms

from watermark_system import ModelWatermarkSystem, WatermarkError, set_global_seed
from utils import LANGUAGES

# 统一预处理：与 ResNet50(IMAGENET1K_V1) 权重期望的输入分布保持一致
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EVAL_BATCH_SIZE = 16


def project_root():
    """项目根目录：基于本文件位置解析，而非当前工作目录。"""
    return Path(__file__).resolve().parent


def resources_dir():
    return project_root() / "resources"


def build_eval_transform():
    """评估用 transform：Resize → CenterCrop → ToTensor → Normalize。"""
    return transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_image_dataset(data_dir):
    """构建 ImageFolder 数据集并做基本校验（空目录直接报错，不静默造数据）。"""
    dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=build_eval_transform())
    if len(dataset) == 0:
        raise WatermarkError("选择的目录中没有有效的图像数据")
    return dataset


def summarize_dataset(dataset):
    """输出"类别数 / 总数 / 每类数量"，便于判断类别不均衡。"""
    counts = {}
    for _, index in dataset.samples:
        name = dataset.classes[index]
        counts[name] = counts.get(name, 0) + 1
    detail = ", ".join(f"{name}:{counts[name]}" for name in dataset.classes)
    return f"{len(dataset.classes)} 类 / {len(dataset)} 张（{detail}）"


class HelpDialog(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(400, 300)

        layout = QVBoxLayout()
        self.setLayout(layout)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(message)
        layout.addWidget(text_edit)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)


class ProgressDialog(QDialog):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.setFixedSize(400, 150)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.progress_label = QLabel("准备开始测试...")
        layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

        self._cancelled = False

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
        QApplication.processEvents()  # 强制刷新UI

    def is_cancelled(self):
        return self._cancelled

    def reject(self):
        self._cancelled = True
        super().reject()
class WatermarkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_lang = "zh"
        self.watermark_system = ModelWatermarkSystem()
        self.set_window_icon()
        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # 左侧面板
        self.left_panel = QGroupBox()
        self.left_panel.setObjectName("operations_group")
        self.left_layout = QVBoxLayout()
        self.left_panel.setLayout(self.left_layout)

        # 右侧面板
        self.right_panel = QGroupBox()
        self.right_panel.setObjectName("output_group")
        right_layout = QVBoxLayout()
        self.right_panel.setLayout(right_layout)

        self.create_verification_section(right_layout)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        right_layout.addWidget(self.output_text)

        self.lang_btn = QPushButton("中文/EN")
        self.lang_btn.clicked.connect(self.toggle_language)
        right_layout.insertWidget(0, self.lang_btn)

        main_layout.addWidget(self.left_panel, 1)
        main_layout.addWidget(self.right_panel, 2)

        self.create_operation_buttons()
        self.update_ui_language()

    def toggle_language(self):
        self.current_lang = "zh" if self.current_lang == "en" else "en"
        self.update_ui_language()

    def update_ui_language(self):
        self.setWindowTitle(self.tr("window_title"))
        groups = {
            "operations_group": self.tr("operations"),
            "output_group": self.tr("output"),
            "model_ops_group": self.tr("model_ops"),
            "watermark_gen_group": self.tr("watermark_gen"),
            "whitebox_group": self.tr("whitebox"),
            "verify_group": self.tr("watermark_verify"),
            "test_group": self.tr("performance_test")
        }

        for name, text in groups.items():
            widget = self.findChild(QGroupBox, name)
            if widget:
                widget.setTitle(text)

        # 分组标题标签：统一登记在 self.labels，避免切换语言后残留旧文案
        for key, label in getattr(self, "labels", {}).items():
            label.setText(f"{self.tr(key)}:")

        if hasattr(self, 'load_pretrained_btn'):
            self.load_pretrained_btn.setText(self.tr("load_pretrained"))
            self.save_btn.setText(self.tr("save_model"))
            self.load_btn.setText(self.tr("load_model"))
            self.gen_btn.setText(self.tr("gen_watermark"))
            self.wb_embed_btn.setText(self.tr("embed_watermark"))
            self.wb_extract_btn.setText(self.tr("extract_watermark"))
            self.verify_btn.setText(self.tr("verify_watermark"))
            self.run_tests_btn.setText(self.tr("run_tests"))
            self.load_test_btn.setText(self.tr("load_test_data"))

        self.lang_btn.setText("中文" if self.current_lang == "en" else "EN")
        self.owner_info_edit.setPlaceholderText(self.tr("owner_placeholder"))
        self.verify_owner_edit.setPlaceholderText(self.tr("owner_placeholder"))
        if hasattr(self, "owner_info_label"):
            self.owner_info_label.setText(self.tr("owner_info"))
        if hasattr(self, "strength_label"):
            self.strength_label.setText(self.tr("embed_strength"))
        if hasattr(self, "verify_hint_label"):
            self.verify_hint_label.setText(self.tr("enter_owner_verify"))

    def tr(self, key):
        return LANGUAGES[self.current_lang].get(key, key)

    def create_operation_buttons(self):
        # 清空布局
        while self.left_layout.count():
            item = self.left_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 分组标题标签登记表：切换语言时统一刷新
        self.labels = {}

        # 模型操作组
        model_group = QGroupBox()
        model_group.setObjectName("model_ops_group")
        model_layout = QVBoxLayout()

        help_btn = QPushButton("ℹ️")
        help_btn.setFixedSize(30, 30)
        help_btn.clicked.connect(lambda: self.show_help(
            self.tr("model_ops"),
            self.tr("model_ops_help")
        ))

        self.load_pretrained_btn = QPushButton(self.tr("load_pretrained"))
        self.load_pretrained_btn.clicked.connect(self.load_pretrained_model)

        self.save_btn = QPushButton(self.tr("save_model"))
        self.save_btn.clicked.connect(self.save_model)

        self.load_btn = QPushButton(self.tr("load_model"))
        self.load_btn.clicked.connect(self.load_model)

        self.labels["model_ops"] = QLabel(f"{self.tr('model_ops')}:")
        help_layout = QHBoxLayout()
        help_layout.addWidget(self.labels["model_ops"])
        help_layout.addWidget(help_btn)
        help_layout.addStretch()
        model_layout.addLayout(help_layout)

        model_layout.addWidget(self.load_pretrained_btn)
        model_layout.addWidget(self.save_btn)
        model_layout.addWidget(self.load_btn)
        model_group.setLayout(model_layout)
        self.left_layout.addWidget(model_group)

        # 水印生成组
        gen_group = QGroupBox()
        gen_group.setObjectName("watermark_gen_group")
        gen_layout = QVBoxLayout()

        gen_help_btn = QPushButton("ℹ️")
        gen_help_btn.setFixedSize(30, 30)
        gen_help_btn.clicked.connect(lambda: self.show_help(
            self.tr("watermark_gen"),
            self.tr("watermark_gen_help")
        ))

        self.gen_btn = QPushButton(self.tr("gen_watermark"))
        self.gen_btn.clicked.connect(self.generate_watermark)

        self.owner_info_edit = QTextEdit()
        self.owner_info_edit.setPlaceholderText(self.tr("owner_placeholder"))
        self.owner_info_edit.setMaximumHeight(100)

        self.labels["watermark_gen"] = QLabel(f"{self.tr('watermark_gen')}:")
        gen_help_layout = QHBoxLayout()
        gen_help_layout.addWidget(self.labels["watermark_gen"])
        gen_help_layout.addWidget(gen_help_btn)
        gen_help_layout.addStretch()
        gen_layout.addLayout(gen_help_layout)

        self.owner_info_label = QLabel(self.tr("owner_info"))
        gen_layout.addWidget(self.owner_info_label)
        gen_layout.addWidget(self.owner_info_edit)
        gen_layout.addWidget(self.gen_btn)
        gen_group.setLayout(gen_layout)
        self.left_layout.addWidget(gen_group)

        # 白盒水印组
        wb_group = QGroupBox()
        wb_group.setObjectName("whitebox_group")
        wb_layout = QVBoxLayout()

        wb_help_btn = QPushButton("ℹ️")
        wb_help_btn.setFixedSize(30, 30)
        wb_help_btn.clicked.connect(lambda: self.show_help(
            self.tr("whitebox"),
            self.tr("whitebox_help")
        ))

        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setRange(0.01, 1.0)
        self.strength_spin.setValue(0.1)
        self.strength_spin.setSingleStep(0.01)

        self.wb_embed_btn = QPushButton(self.tr("embed_watermark"))
        self.wb_embed_btn.clicked.connect(self.embed_whitebox)

        self.wb_extract_btn = QPushButton(self.tr("extract_watermark"))
        self.wb_extract_btn.clicked.connect(self.extract_whitebox)

        self.labels["whitebox"] = QLabel(f"{self.tr('whitebox')}:")
        wb_help_layout = QHBoxLayout()
        wb_help_layout.addWidget(self.labels["whitebox"])
        wb_help_layout.addWidget(wb_help_btn)
        wb_help_layout.addStretch()
        wb_layout.addLayout(wb_help_layout)

        self.strength_label = QLabel(self.tr("embed_strength"))
        wb_layout.addWidget(self.strength_label)
        wb_layout.addWidget(self.strength_spin)
        wb_layout.addWidget(self.wb_embed_btn)
        wb_layout.addWidget(self.wb_extract_btn)
        wb_group.setLayout(wb_layout)
        self.left_layout.addWidget(wb_group)

        self.left_layout.addStretch()

        test_group = QGroupBox(self.tr("performance_test"))
        test_group.setObjectName("test_group")
        test_layout = QVBoxLayout()

        test_help_btn = QPushButton("ℹ️")
        test_help_btn.setFixedSize(30, 30)
        test_help_btn.clicked.connect(lambda: self.show_help(
            self.tr("performance_test"),
            self.tr("performance_test_help")
        ))

        self.run_tests_btn = QPushButton(self.tr("run_tests"))
        self.run_tests_btn.clicked.connect(self.run_performance_tests)

        self.test_result = QTextEdit()
        self.test_result.setReadOnly(True)

        self.labels["performance_test"] = QLabel(f"{self.tr('performance_test')}:")
        test_help_layout = QHBoxLayout()
        test_help_layout.addWidget(self.labels["performance_test"])
        test_help_layout.addWidget(test_help_btn)
        test_help_layout.addStretch()
        test_layout.addLayout(test_help_layout)

        test_layout.addWidget(self.run_tests_btn)
        test_layout.addWidget(self.test_result)
        test_group.setLayout(test_layout)
        self.left_layout.addWidget(test_group)

        self.load_test_btn = QPushButton(self.tr("load_test_data"))
        self.load_test_btn.clicked.connect(self.load_test_dataset)
        test_layout.addWidget(self.load_test_btn)  # 添加到测试组的布局中

    def create_verification_section(self, right_layout):
        verify_group = QGroupBox(self.tr("watermark_verify"))
        verify_group.setObjectName("verify_group")
        verify_layout = QVBoxLayout()

        self.verify_owner_edit = QTextEdit()
        self.verify_owner_edit.setPlaceholderText(self.tr("owner_placeholder"))
        self.verify_owner_edit.setMaximumHeight(100)

        self.verify_btn = QPushButton(self.tr("verify_watermark"))
        self.verify_btn.clicked.connect(self.verify_watermark)

        self.verify_result = QLabel()
        self.verify_result.setWordWrap(True)
        self.verify_result.setAlignment(Qt.AlignCenter)

        self.verify_hint_label = QLabel(self.tr("enter_owner_verify"))
        verify_layout.addWidget(self.verify_hint_label)
        verify_layout.addWidget(self.verify_owner_edit)
        verify_layout.addWidget(self.verify_btn)
        verify_layout.addWidget(self.verify_result)
        verify_group.setLayout(verify_layout)

        right_layout.addWidget(verify_group)

    def show_help(self, title, message):
        dialog = HelpDialog(title, message, self)
        dialog.exec_()

    def log_message(self, message):
        self.output_text.append(message)
        self.output_text.ensureCursorVisible()

    def load_pretrained_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, self.tr("load_pretrained"), "",
                                                   "PyTorch Model Files (*.pth);;All Files (*)",
                                                   options=options)
        if file_name:
            try:
                result = self.watermark_system.load_pretrained_model(file_name)
                self.log_message(result)
                self.log_message("")
            except Exception as e:
                self.log_message(f"Error loading pretrained model: {str(e)}")

    def generate_watermark(self):
        owner_info = self.owner_info_edit.toPlainText()
        if not owner_info:
            QMessageBox.warning(self, "Warning", self.tr("no_owner_info"))
            return

        try:
            key = self.watermark_system.generate_watermark(owner_info)
            self.log_message(self.tr("watermark_generated"))
            self.log_message(self.tr("watermark_key").format(key[:16]))
            self.log_message("")
        except Exception as e:
            self.log_message(f"Error generating watermark: {str(e)}")

    def embed_whitebox(self):
        strength = self.strength_spin.value()
        try:
            result = self.watermark_system.embed_watermark_whitebox(strength)
            self.log_message(result)
            self.log_message("")
        except Exception as e:
            self.log_message(f"Error embedding watermark: {str(e)}")

    def extract_whitebox(self):
        try:
            if not self.watermark_system.watermark_key:
                self.log_message(self.tr("no_key"))
                return

            extracted_key = self.watermark_system.extract_watermark_whitebox()
            salt, original_key = self.watermark_system.watermark_key
            match = self.watermark_system.verify_watermark(extracted_key)
            accuracy = self.watermark_system.extraction_bit_accuracy(original_key)

            self.log_message(self.tr("original_key").format(original_key[:16]))
            self.log_message(self.tr("extracted_key").format(extracted_key[:16]))
            self.log_message(self.tr("watermark_match").format(match))
            self.log_message(self.tr("extraction_bits").format(accuracy * 100))
            self.log_message("")
        except Exception as e:
            self.log_message(f"Error extracting watermark: {str(e)}")

    def verify_watermark(self):
        owner_info = self.verify_owner_edit.toPlainText()
        if not owner_info:
            QMessageBox.warning(self, "Warning", self.tr("no_owner_info"))
            return

        try:
            match, message = self.watermark_system.verify_watermark_with_owner(owner_info)

            if match:
                result_msg = self.tr("verify_success") + "\n" + message
            else:
                result_msg = self.tr("verify_fail") + "\n" + message

            self.verify_result.setText(result_msg)
            self.log_message(result_msg)
            self.log_message("")

        except Exception as e:
            error_msg = f"Error verifying watermark: {str(e)}"
            self.log_message(error_msg)
            self.verify_result.setText(error_msg)

    def save_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("save_model"),
            "",
            "PyTorch Model Files (*.pth);;All Files (*)",
            options=options
        )

        if not file_name:
            return

        if not file_name.lower().endswith('.pth'):
            file_name += '.pth'

        try:
            result = self.watermark_system.save_model(file_name)
            self.log_message(result)
            self.log_message("")
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self, "Save Error", error_msg)

    def load_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, self.tr("load_model"), "",
                                                   "PyTorch Model Files (*.pth);;All Files (*)",
                                                   options=options)
        if file_name:
            try:
                result = self.watermark_system.load_model(file_name)
                self.log_message(result)

                if self.watermark_system.watermark_key:
                    self.log_message(self.tr("watermark_key").format(self.watermark_system.watermark_key[1][:16]))
                else:
                    self.log_message(self.tr("no_key"))
                self.log_message("")
            except Exception as e:
                self.log_message(f"Error loading model: {str(e)}")

    def set_window_icon(self):
        icon_path = resources_dir() / "wm.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def run_performance_tests(self):
        """执行性能测试（P0：测试数据不可用即终止，指标语义改为比特一致率）。"""
        if not self.watermark_system.model:
            QMessageBox.warning(self, "Warning", self.tr("no_model_loaded"))
            return
        if not self.watermark_system.watermark_key:
            QMessageBox.warning(self, "Warning", self.tr("no_key"))
            return

        set_global_seed(self.watermark_system.seed)

        progress_dialog = ProgressDialog(self.tr("running_tests"), self)
        progress_dialog.show()
        QApplication.processEvents()  # 立即显示对话框

        self.test_result.clear()
        self.test_result.append("=== 开始性能测试 ===")
        original_acc = None
        ext_rate = None

        try:
            progress_dialog.update_progress(5, "正在加载测试数据...")
            test_loader = self.get_test_loader()
            if test_loader is None:
                self.test_result.append("❌ 测试数据不可用，测试终止（不会用随机数据代替）")
                QMessageBox.critical(self, "错误", self.tr("no_test_data"))
                return
            self.test_result.append("✓ 测试数据加载成功")

            try:
                progress_dialog.update_progress(20, "测试原始准确率...")
                original_acc = self.watermark_system.evaluate_model(test_loader)
                self.test_result.append(f"\n✓ 原始准确率: {original_acc:.2f}%")
            except Exception as e:
                self.test_result.append(f"\n❌ 原始准确率测试失败: {str(e)}")
                if progress_dialog.is_cancelled():
                    return

            try:
                progress_dialog.update_progress(40, "测试水印比特一致率...")
                ext_rate = self.watermark_system.extraction_bit_accuracy()
                self.test_result.append(f"✓ 水印比特一致率: {ext_rate * 100:.2f}%")
            except Exception as e:
                self.test_result.append(f"❌ 水印提取测试失败: {str(e)}")
                if progress_dialog.is_cancelled():
                    return

            # 鲁棒性测试（robustness_test 内部逐项捕获异常并返回 error 字段）
            robustness_results = []
            attack_types = ['prune', 'noise', 'finetune']

            for i, attack in enumerate(attack_types):
                if progress_dialog.is_cancelled():
                    return

                progress_dialog.update_progress(50 + (i * 15), f"正在进行{attack}测试...")

                result = self.watermark_system.robustness_test(
                    test_loader,
                    attack_types=[attack]
                )[0]
                robustness_results.append(result)

                if 'error' in result:
                    self.test_result.append(f"\n❌ {attack}测试失败: {result['error']}")
                else:
                    self.test_result.append(
                        f"\n✓ {attack}测试完成: "
                        f"准确率={result['accuracy']:.2f}%, "
                        f"比特一致率={result['bit_accuracy'] * 100:.2f}%"
                    )

            progress_dialog.update_progress(95, "生成测试报告...")
            self.test_result.append("\n=== 测试结果汇总 ===")
            self.test_result.append(
                "1. 原始准确率: 测试失败" if original_acc is None
                else f"1. 原始准确率: {original_acc:.2f}%"
            )
            self.test_result.append(
                "2. 水印比特一致率: 测试失败" if ext_rate is None
                else f"2. 水印比特一致率: {ext_rate * 100:.2f}%"
            )

            for result in robustness_results:
                if 'error' in result:
                    self.test_result.append(
                        f"3. {result['attack_type']}: 测试失败 ({result['error']})"
                    )
                else:
                    self.test_result.append(
                        f"3. {result['attack_type']}: "
                        f"准确率={result['accuracy']:.2f}%, "
                        f"比特一致率={result['bit_accuracy'] * 100:.2f}%"
                    )

            progress_dialog.update_progress(100, "测试完成！")
            QMessageBox.information(self, "完成", "性能测试执行完毕")

        except Exception as e:
            self.test_result.append(f"\n❌ 测试过程中发生严重错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"测试失败: {str(e)}")

        finally:
            # 保留已加载的测试数据缓存（避免每次测试都要重新选择目录），只释放显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            progress_dialog.close()

    def get_test_loader(self):
        """获取测试数据加载器：优先复用已加载的数据，否则提示选择目录。

        P0 修复：加载失败不再静默回退到随机数据（那会让所有指标失去意义）。
        """
        if self.watermark_system.test_loader is not None:
            return self.watermark_system.test_loader

        options = QFileDialog.Options()
        data_dir = QFileDialog.getExistingDirectory(
            self,
            "选择测试数据集目录",
            options=options
        )
        if not data_dir:
            self.log_message(self.tr("no_dataset_dir"))
            return None

        try:
            dataset = build_image_dataset(data_dir)
        except Exception as e:
            self.log_message(f"{self.tr('test_data_error')}: {str(e)}")
            return None

        info = summarize_dataset(dataset)
        self.log_message(f"{self.tr('test_data_loaded')}: {data_dir}")
        self.log_message(self.tr("test_dataset_info").format(info))
        if len(dataset.classes) != self.watermark_system.num_classes:
            warning = self.tr("class_mismatch").format(
                len(dataset.classes), self.watermark_system.num_classes
            )
            self.log_message(warning)
            QMessageBox.warning(self, self.tr("warning_title"), warning)

        # 评估集必须 shuffle=False，否则结果不可复现
        loader = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
        self.watermark_system.test_loader = loader
        return loader

    def load_test_dataset(self):
        options = QFileDialog.Options()
        data_dir = QFileDialog.getExistingDirectory(self, "选择测试数据集目录", options=options)
        if not data_dir:
            return
        try:
            dataset = build_image_dataset(data_dir)
            self.watermark_system.test_loader = DataLoader(
                dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False
            )
            self.log_message(f"{self.tr('test_data_loaded')}: {data_dir}")
            self.log_message(self.tr("test_dataset_info").format(summarize_dataset(dataset)))
            if len(dataset.classes) != self.watermark_system.num_classes:
                self.log_message(self.tr("class_mismatch").format(
                    len(dataset.classes), self.watermark_system.num_classes
                ))
        except Exception as e:
            self.log_message(f"{self.tr('test_data_error')}: {str(e)}")