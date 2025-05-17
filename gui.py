# _*_ coding: utf-8 _*_
# @Time : 2025/4/9 14:27
# @Author : yang_4_ever
# @File : gui
# @Project : FinalWMsys

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                            QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox,
                            QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox, QDialog, QDialogButtonBox,QProgressBar)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from watermark_system import ModelWatermarkSystem
from utils import LANGUAGES
import sys
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torchvision


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
            "verify_group": self.tr("watermark_verify")
        }

        for name, text in groups.items():
            widget = self.findChild(QGroupBox, name)
            if widget:
                widget.setTitle(text)

        if hasattr(self, 'load_pretrained_btn'):
            self.load_pretrained_btn.setText(self.tr("load_pretrained"))
            self.save_btn.setText(self.tr("save_model"))
            self.load_btn.setText(self.tr("load_model"))
            self.gen_btn.setText(self.tr("gen_watermark"))
            self.wb_embed_btn.setText(self.tr("embed_watermark"))
            self.wb_extract_btn.setText(self.tr("extract_watermark"))
            self.verify_btn.setText(self.tr("verify_watermark"))

        self.lang_btn.setText("中文" if self.current_lang == "en" else "EN")
        self.owner_info_edit.setPlaceholderText(self.tr("owner_placeholder"))
        self.verify_owner_edit.setPlaceholderText(self.tr("owner_placeholder"))

    def tr(self, key):
        return LANGUAGES[self.current_lang].get(key, key)

    def create_operation_buttons(self):
        # 清空布局
        while self.left_layout.count():
            item = self.left_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

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

        help_layout = QHBoxLayout()
        help_layout.addWidget(QLabel(f"{self.tr('model_ops')}:"))
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

        gen_help_layout = QHBoxLayout()
        gen_help_layout.addWidget(QLabel(f"{self.tr('watermark_gen')}:"))
        gen_help_layout.addWidget(gen_help_btn)
        gen_help_layout.addStretch()
        gen_layout.addLayout(gen_help_layout)

        gen_layout.addWidget(QLabel(self.tr("owner_info")))
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

        wb_help_layout = QHBoxLayout()
        wb_help_layout.addWidget(QLabel(f"{self.tr('whitebox')}:"))
        wb_help_layout.addWidget(wb_help_btn)
        wb_help_layout.addStretch()
        wb_layout.addLayout(wb_help_layout)

        wb_layout.addWidget(QLabel(self.tr("embed_strength")))
        wb_layout.addWidget(self.strength_spin)
        wb_layout.addWidget(self.wb_embed_btn)
        wb_layout.addWidget(self.wb_extract_btn)
        wb_group.setLayout(wb_layout)
        self.left_layout.addWidget(wb_group)

        self.left_layout.addStretch()

        test_group = QGroupBox(self.tr("performance_test"))
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

        test_help_layout = QHBoxLayout()
        test_help_layout.addWidget(QLabel(f"{self.tr('performance_test')}:"))
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

        verify_layout.addWidget(QLabel(self.tr("enter_owner_verify")))
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

            self.log_message(self.tr("original_key").format(original_key[:16]))
            self.log_message(self.tr("extracted_key").format(extracted_key[:16]))
            self.log_message(self.tr("watermark_match").format(match))
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
        icon_path = os.path.join("resources", "wm.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

    def run_performance_tests(self):
        """执行性能测试的完整修复版本"""
        try:
            # 1. 前置检查
            if not self.watermark_system.model:
                QMessageBox.warning(self, "Warning", self.tr("no_model_loaded"))
                return

            # 2. 创建并显示进度对话框
            progress_dialog = ProgressDialog(self.tr("running_tests"), self)
            progress_dialog.show()
            QApplication.processEvents()  # 立即显示对话框

            try:
                # 3. 初始化测试结果区域
                self.test_result.clear()
                self.test_result.append("=== 开始性能测试 ===")

                # 4. 获取测试数据加载器（带错误处理）
                test_loader = None
                try:
                    progress_dialog.update_progress(5, "正在加载测试数据...")
                    test_loader = self.get_test_loader()
                    if test_loader is None:
                        raise ValueError("无法获取有效的测试数据加载器")
                    self.test_result.append("✓ 测试数据加载成功")
                except Exception as e:
                    self.test_result.append(f"❌ 测试数据加载失败: {str(e)}")
                    QMessageBox.critical(self, "错误", f"加载测试数据失败: {str(e)}")
                    return

                # 5. 测试原始准确率
                try:
                    progress_dialog.update_progress(20, "测试原始准确率...")
                    original_acc = self.watermark_system.evaluate_model(test_loader)
                    self.test_result.append(f"\n✓ 原始准确率: {original_acc:.2f}%")
                except Exception as e:
                    self.test_result.append(f"\n❌ 原始准确率测试失败: {str(e)}")
                    if progress_dialog.is_cancelled():
                        return

                # 6. 测试水印提取率
                try:
                    progress_dialog.update_progress(40, "测试水印提取率...")
                    ext_rate = self.watermark_system.test_watermark_extraction() * 100
                    self.test_result.append(f"✓ 水印提取成功率: {ext_rate:.2f}%")
                except Exception as e:
                    self.test_result.append(f"❌ 水印提取测试失败: {str(e)}")
                    if progress_dialog.is_cancelled():
                        return

                # 7. 鲁棒性测试
                robustness_results = []
                attack_types = ['prune', 'noise', 'finetune']

                for i, attack in enumerate(attack_types):
                    if progress_dialog.is_cancelled():
                        return

                    progress = 50 + (i * 15)
                    progress_dialog.update_progress(progress, f"正在进行{attack}测试...")

                    try:
                        result = self.watermark_system.robustness_test(
                            test_loader,
                            attack_types=[attack]
                        )[0]  # 获取第一个结果

                        robustness_results.append(result)
                        self.test_result.append(
                            f"\n✓ {attack}测试完成: "
                            f"准确率={result['accuracy']:.2f}%, "
                            f"提取率={result['extraction_rate'] * 100:.2f}%"
                        )
                    except Exception as e:
                        self.test_result.append(f"\n❌ {attack}测试失败: {str(e)}")
                        robustness_results.append({
                            'attack_type': attack,
                            'error': str(e)
                        })

                # 8. 显示最终结果
                progress_dialog.update_progress(95, "生成测试报告...")
                self.test_result.append("\n=== 测试结果汇总 ===")
                self.test_result.append(f"1. 原始准确率: {original_acc:.2f}%")
                self.test_result.append(f"2. 水印提取率: {ext_rate:.2f}%")

                for result in robustness_results:
                    if 'error' in result:
                        self.test_result.append(
                            f"3. {result['attack_type']}: 测试失败 ({result['error']})"
                        )
                    else:
                        self.test_result.append(
                            f"3. {result['attack_type']}: "
                            f"准确率={result['accuracy']:.2f}%, "
                            f"提取率={result['extraction_rate'] * 100:.2f}%"
                        )

                progress_dialog.update_progress(100, "测试完成！")
                QMessageBox.information(self, "完成", "性能测试执行完毕")

            except Exception as e:
                self.test_result.append(f"\n❌ 测试过程中发生严重错误: {str(e)}")
                QMessageBox.critical(self, "错误", f"测试失败: {str(e)}")

            finally:
                self.watermark_system.test_loader = None  # 释放测试数据
                torch.cuda.empty_cache()  # 清理GPU缓存

        except Exception as e:
            self.test_result.append(f"\n❌ 无法初始化测试: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法开始测试: {str(e)}")

    def get_test_loader(self):
        """改进的测试数据加载方法"""
        try:
            # 1. 尝试使用已加载的数据
            if hasattr(self.watermark_system, 'test_loader') and self.watermark_system.test_loader:
                return self.watermark_system.test_loader

            # 2. 提示用户选择数据集目录
            options = QFileDialog.Options()
            data_dir = QFileDialog.getExistingDirectory(
                self,
                "选择测试数据集目录",
                options=options
            )

            if not data_dir:
                raise ValueError("用户取消了数据目录选择")

            if not os.path.exists(data_dir):
                raise ValueError(f"目录不存在: {data_dir}")

            # 3. 创建数据加载器
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])

            dataset = torchvision.datasets.ImageFolder(
                root=data_dir,
                transform=transform
            )

            if len(dataset) == 0:
                raise ValueError("选择的目录中没有有效的图像数据")

            loader = DataLoader(dataset, batch_size=8, shuffle=True)
            self.watermark_system.test_loader = loader  # 缓存加载器
            return loader

        except Exception as e:
            self.log_message(f"加载测试数据错误: {str(e)}")

            # 4. 提供后备的随机数据
            class RandomDataset(Dataset):
                def __init__(self, size=100):
                    self.data = torch.randn(size, 3, 224, 224)
                    self.targets = torch.randint(0, 4, (size,))

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    return self.data[idx], self.targets[idx]

            self.log_message("⚠️ 使用随机生成的数据作为后备")
            return DataLoader(RandomDataset(), batch_size=8, shuffle=True)

    def load_test_dataset(self):
        options = QFileDialog.Options()
        data_dir = QFileDialog.getExistingDirectory(self, "选择测试数据集目录", options=options)
        if data_dir:
            try:
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])

                dataset = torchvision.datasets.ImageFolder(
                    root=data_dir,
                    transform=transform
                )
                self.watermark_system.test_loader = DataLoader(dataset, batch_size=16, shuffle=True)
                self.log_message(f"测试数据集已加载: {data_dir}")
            except Exception as e:
                self.log_message(f"加载测试数据集错误: {str(e)}")

    def load_custom_dataset(self, data_dir):
        # 实现您的数据集加载逻辑（需自定义）
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
        return DataLoader(dataset, batch_size=16, shuffle=True)