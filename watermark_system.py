# _*_ coding: utf-8 _*_
# @Time : 2025/4/9 14:27
# @Author : yang_4_ever
# @File : watermark_system
# @Project : FinalWMsys
"""DNN 模型水印系统的核心逻辑（P0 止血版）。

对应重构方案 docs/REFACTOR_FRAMEWORK.md 第 5 节 P0 阶段，已修复：

* 嵌入层定位错误：原实现取"第一个 4D 参数"，实际写入 ``net.conv1.weight``，
  而 ``watermark_layer = 'layer4'`` 从未生效。现按算法版本显式定位。
* 通道不足分支引用未定义变量导致的 ``NameError``。
* ``save_model`` 在无水印密钥时下标越界；``load_model`` 缺少 ``map_location``
  与 ``weights_only``。
* 模型重复构建（``__init__`` 与 ``load_pretrained_model`` 各建一次）。
* "微调攻击"对已含 softmax 的输出再次使用 ``CrossEntropyLoss``，且未切换
  train/eval 模式。
* "水印提取成功率"重复同一确定性提取（恒为 0/1），现改为比特一致率。
* 剪枝攻击口径与注释不符（注释 10%，代码 70%），且会剪掉 BN 参数。
* 全流程无随机种子，结果不可复现。

兼容性红线（不要修改）：
* 密钥派生参数：``sha256(owner)[:32]`` 作盐 + ``PBKDF2-HMAC-SHA256/100000``，
  截取前 32 位 hex。
* 旧 checkpoint（无 ``watermark_algorithm_version`` 字段）按算法版本 1
  （``net.conv1.weight``）提取，并在返回信息中提示重新嵌入。
"""

import copy
import hashlib
import os
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# --------------------------------------------------------------------------- #
# 常量（重构期冻结，改动会破坏既有模型的验证）
# --------------------------------------------------------------------------- #

DEFAULT_SEED = 42
KEY_HEX_CHARS = 16                      # 参与嵌入的 hex 字符数 → 64 bit
KEY_BITS = KEY_HEX_CHARS * 4            # 64
PBKDF2_ITERATIONS = 100000              # 冻结

ALGO_VERSION_LATEST = 2
ALGO_VERSION_LEGACY_CONV1 = 1           # 历史版本（错误地写入 conv1）
LAYER_BY_ALGO_VERSION: Dict[int, str] = {
    ALGO_VERSION_LEGACY_CONV1: "net.conv1.weight",
    ALGO_VERSION_LATEST: "net.layer4.2.conv2.weight",
}

CHECKPOINT_FORMAT_VERSION = "1.1"
LEGACY_CHECKPOINT_FORMAT_VERSION = "1.0"

DEFAULT_ATTACK_PARAMS: Dict[str, Dict[str, float]] = {
    "prune": {"ratio": 0.10},
    "noise": {"sigma": 0.05},
    "finetune": {"epochs": 2, "lr": 0.001},
}


class WatermarkError(Exception):
    """带用户可读信息的水印流程异常。"""


class LayerNotFoundError(WatermarkError):
    """目标参数层不存在。"""


class CapacityError(WatermarkError):
    """目标层容量不足以承载水印。"""


# --------------------------------------------------------------------------- #
# 纯函数：种子与密钥派生
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """固定 python / numpy / torch 的随机种子，保证评测可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_watermark_key(owner_info: str) -> Tuple[str, str]:
    """由所有者信息派生水印密钥（参数冻结，保证旧模型可验证）。"""
    if owner_info is None or not owner_info.strip():
        raise WatermarkError("所有者信息不能为空")
    raw = owner_info.encode("utf-8")
    salt = hashlib.sha256(raw).hexdigest()[:32]
    key = hashlib.pbkdf2_hmac("sha256", raw, salt.encode("utf-8"), PBKDF2_ITERATIONS).hex()[:32]
    return salt, key


def key_to_bits(key: str, bits: int = KEY_BITS) -> str:
    """hex 字符串 → 定长二进制位串，长度不足直接报错。"""
    hex_chars = bits // 4
    if len(key) < hex_chars:
        raise WatermarkError(f"密钥长度不足：需要 {hex_chars} 位 hex，实际 {len(key)}")
    return "".join(f"{int(c, 16):04b}" for c in key[:hex_chars])


def bits_to_key(bits: str) -> str:
    """二进制位串 → hex 字符串（每 4 bit 一位 hex）。"""
    if len(bits) % 4:
        raise WatermarkError("位串长度必须是 4 的倍数")
    return "".join(f"{int(bits[i:i + 4], 2):x}" for i in range(0, len(bits), 4))


def bit_accuracy(expected: str, actual: str) -> float:
    """两个 hex 密钥之间的比特一致率（0.0 ~ 1.0）。"""
    expected_bits = key_to_bits(expected)
    actual_bits = key_to_bits(actual)
    total = min(len(expected_bits), len(actual_bits))
    if total == 0:
        return 0.0
    matches = sum(1 for a, b in zip(expected_bits, actual_bits) if a == b)
    return matches / total


# --------------------------------------------------------------------------- #
# 核心系统
# --------------------------------------------------------------------------- #

class ModelWatermarkSystem:
    """ResNet50 白盒水印：生成密钥 → 嵌入 → 提取 → 验证 → 评测。"""

    def __init__(self,
                 num_classes: int = 4,
                 device: Optional[str] = None,
                 seed: int = DEFAULT_SEED):
        set_global_seed(seed)
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.model: Optional[nn.Module] = None
        self.watermark_key: Optional[Tuple[str, str]] = None
        self.test_loader = None
        self.watermark_layer: Optional[str] = None
        # 尚未嵌入时为 None；load_model 会依据 checkpoint 还原
        self.watermark_algorithm_version: Optional[int] = None
        self.initialize_model()

    # ---------------------------------------------------------------- 模型
    def initialize_model(self) -> None:
        """构建骨干网络并替换分类头（只在这里构建模型）。"""
        try:
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:                      # torchvision < 0.13
            try:
                self.base_model = models.resnet50(pretrained=True)
            except Exception:                       # 无网络 / 无缓存权重
                self.base_model = models.resnet50()
                print("Warning: Could not load pretrained weights")
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = self._create_model().to(self.device)

    def _create_model(self) -> nn.Module:
        class WatermarkModel(nn.Module):
            def __init__(self, base_model: nn.Module):
                super(WatermarkModel, self).__init__()
                self.net = base_model
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                return self.softmax(self.net(x))

        return WatermarkModel(self.base_model)

    def logits(self, inputs: torch.Tensor) -> torch.Tensor:
        """返回未过 softmax 的 logits（训练/攻击必须用它，避免双重 softmax）。"""
        return self.model.net(inputs)

    # ----------------------------------------------------------- 参数层定位
    def _find_param(self, name: str) -> Optional[torch.nn.Parameter]:
        for param_name, param in self.model.named_parameters():
            if param_name == name:
                return param
        return None

    def _four_d_param_names(self) -> List[str]:
        return [n for n, p in self.model.named_parameters() if p.dim() == 4]

    def locate_target(self, algorithm_version: Optional[int] = None) -> Tuple[str, torch.nn.Parameter]:
        """按算法版本定位嵌入/提取使用的参数层（显式命名，不再"取第一个 4D"）。"""
        version = algorithm_version or self.watermark_algorithm_version or ALGO_VERSION_LATEST
        layer_name = LAYER_BY_ALGO_VERSION.get(version)
        if layer_name is None:
            raise WatermarkError(f"未知的水印算法版本: {version}")
        param = self._find_param(layer_name)
        if param is None:
            raise LayerNotFoundError(
                f"找不到目标层 {layer_name}；当前可用的 4D 参数层: {self._four_d_param_names()}"
            )
        return layer_name, param

    def layer_capacity(self, algorithm_version: Optional[int] = None) -> int:
        """目标层可承载的比特数（每输出通道 1 bit）。"""
        _, param = self.locate_target(algorithm_version)
        return int(param.shape[0])

    # ------------------------------------------------------------- 模型存取
    @staticmethod
    def _extract_state_dict(checkpoint) -> Tuple[dict, dict]:
        """兼容三种 checkpoint：新格式 / 旧格式 / 裸 state_dict。"""
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            return checkpoint["model_state"], checkpoint
        if isinstance(checkpoint, dict):
            if all(torch.is_tensor(v) for v in checkpoint.values()):
                return checkpoint, {}
        raise WatermarkError("无法识别的模型文件格式")

    @staticmethod
    def _torch_load(path: str, map_location):
        """优先以 weights_only=True 读取，失败再回退（并告知调用方降级了）。"""
        try:
            return torch.load(path, map_location=map_location, weights_only=True), True
        except Exception:
            return torch.load(path, map_location=map_location, weights_only=False), False

    def load_pretrained_model(self, path: str) -> str:
        """把权重合并进当前模型（不再重建网络，避免重复加载与显存翻倍）。"""
        try:
            checkpoint, safe = self._torch_load(path, self.device)
            state_dict, _ = self._extract_state_dict(checkpoint)
            model_dict = self.model.state_dict()
            matched = {k: v for k, v in state_dict.items()
                       if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(matched)
            self.model.load_state_dict(model_dict, strict=False)
            skipped = len(state_dict) - len(matched)
            note = "" if safe else "（该文件需要非安全反序列化，已降级读取）"
            return (f"Pretrained model loaded from {path}{note}；"
                    f"匹配 {len(matched)} 个参数，跳过 {skipped} 个不匹配参数")
        except Exception as e:
            return f"Error loading pretrained model: {str(e)}"

    def save_model(self, file_path: Optional[str] = None) -> str:
        """保存模型与水印元数据；无水印密钥时明确报错而不是崩溃。"""
        if file_path is None:
            return "No file path provided"
        if self.watermark_key is None:
            return "Error saving model: 尚未生成水印密钥，请先生成水印（Generate Watermark）"
        if not file_path.lower().endswith(".pth"):
            file_path += ".pth"
        if os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            file_path = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"

        version = self.watermark_algorithm_version or ALGO_VERSION_LATEST
        save_data = {
            "model_state": self.model.state_dict(),
            "watermark_salt": self.watermark_key[0],
            "watermark_hash": self.watermark_key[1],
            # 旧字段名保持兼容
            "system_version": LEGACY_CHECKPOINT_FORMAT_VERSION,
            # 新增元数据（层定位与格式迁移用）
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "watermark_algorithm_version": version,
            "watermark_layer": LAYER_BY_ALGO_VERSION.get(version),
            "num_classes": self.num_classes,
            "seed": self.seed,
        }
        torch.save(save_data, file_path)
        return (f"Model saved to new file: {file_path} "
                f"(算法版本={version}, 层={save_data['watermark_layer']})")

    def load_model(self, file_path: str) -> str:
        checkpoint, safe = self._torch_load(file_path, "cpu")
        state_dict, meta = self._extract_state_dict(checkpoint)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        warnings: List[str] = []
        if not safe:
            warnings.append("该文件需要非安全反序列化，已降级读取（建议重新保存）")
        if meta:
            salt = meta.get("watermark_salt")
            digest = meta.get("watermark_hash")
            if salt and digest:
                self.watermark_key = (salt, digest)
                warnings.append("已恢复水印盐值与密钥")
            version = meta.get("watermark_algorithm_version")
            if version is None:
                version = ALGO_VERSION_LEGACY_CONV1
                warnings.append(f"旧版模型（无算法版本字段），按 v{version}"
                                f"（{LAYER_BY_ALGO_VERSION[version]}）兼容提取，建议重新嵌入水印")
            self.watermark_algorithm_version = int(version)
            self.watermark_layer = meta.get("watermark_layer") or LAYER_BY_ALGO_VERSION.get(int(version))
            model_classes = meta.get("num_classes")
            if model_classes and int(model_classes) != self.num_classes:
                warnings.append(f"类别数不一致：文件 {model_classes}，当前 {self.num_classes}")
        else:
            self.watermark_key = None
            warnings.append("该文件不含水印元数据（裸 state_dict）")

        if missing:
            warnings.append(f"{len(missing)} 个参数缺失（已按当前网络结构初始化）")
        if unexpected:
            warnings.append(f"{len(unexpected)} 个参数未使用")

        detail = "；".join(warnings) if warnings else "无水印元数据"
        return f"Model loaded from {file_path}（{detail}）"

    # --------------------------------------------------------------- 水印
    def generate_watermark(self, owner_info: str) -> str:
        """生成固定水印密钥（同一 owner_info 始终得到同一 key）。"""
        self.watermark_key = derive_watermark_key(owner_info)
        self.watermark_algorithm_version = None      # 尚未嵌入
        self.watermark_layer = None
        return self.watermark_key[1]

    def embed_watermark_whitebox(self, strength: float = 0.1,
                                 algorithm_version: int = ALGO_VERSION_LATEST) -> str:
        """把密钥比特写入目标层的每个输出通道（每通道 1 bit）。"""
        if self.watermark_key is None:
            raise WatermarkError("请先生成水印密钥（调用 generate_watermark）")

        layer_name, param = self.locate_target(algorithm_version)
        capacity = int(param.shape[0])
        bits = key_to_bits(self.watermark_key[1], KEY_BITS)
        if capacity < KEY_BITS:
            raise CapacityError(
                f"目标层 {layer_name} 容量不足：需要 {KEY_BITS} bit，实际仅 {capacity} 个输出通道"
            )

        with torch.no_grad():
            for i, bit in enumerate(bits):
                param[i, 0, 0, 0] = strength if bit == "1" else -strength

        self.watermark_algorithm_version = int(algorithm_version)
        self.watermark_layer = layer_name
        return (f"✅ 水印已嵌入到 [{layer_name}] 层"
                f"（算法版本={algorithm_version}, 强度={strength}, "
                f"嵌入 {KEY_BITS} bit / 容量 {capacity} bit）")

    def extract_watermark_whitebox(self, algorithm_version: Optional[int] = None) -> str:
        """从目标层读回 16 位 hex 水印（与嵌入时同一层定位规则）。"""
        layer_name, param = self.locate_target(algorithm_version)
        capacity = int(param.shape[0])
        if capacity < KEY_BITS:
            raise CapacityError(
                f"参数层 {layer_name} 通道数不足（需≥{KEY_BITS}，实际为 {capacity}）"
            )

        bits: List[str] = []
        with torch.no_grad():
            for i in range(KEY_BITS):
                value = float(param[i, 0, 0, 0].item())
                bits.append("1" if value > 0 else "0")
        return bits_to_key("".join(bits))

    def verify_watermark(self, extracted_key: str) -> bool:
        """逐位严格比对（16 位 hex 全等）。"""
        if not self.watermark_key:
            return False
        return extracted_key.startswith(self.watermark_key[1][:KEY_HEX_CHARS])

    def extraction_bit_accuracy(self, expected_key: Optional[str] = None,
                                algorithm_version: Optional[int] = None) -> float:
        """比特一致率（0.0~1.0）：取代原先恒为 0/1 的"提取成功率"。"""
        if expected_key is None:
            if not self.watermark_key:
                raise WatermarkError("没有可用的水印密钥")
            expected_key = self.watermark_key[1]
        extracted = self.extract_watermark_whitebox(algorithm_version)
        return bit_accuracy(expected_key, extracted)

    def test_watermark_extraction(self, trials: int = 1) -> float:
        """[DEPRECATED] 兼容旧调用：返回值语义已改为比特一致率。

        ``trials`` 仅为保留签名；白盒提取是确定性的，重复提取不产生新信息。
        """
        return self.extraction_bit_accuracy()

    def verify_watermark_with_owner(self, owner_info: str) -> Tuple[bool, str]:
        """增强版水印验证：提取值 + 存储值 + 由所有者信息重新生成值三方比对。"""
        try:
            if not self.watermark_key:
                return False, "No watermark key available"

            salt, stored_key = self.watermark_key
            extracted_key = self.extract_watermark_whitebox()
            match = self.verify_watermark(extracted_key)
            regenerated_key = derive_watermark_key(owner_info)[1]
            accuracy = bit_accuracy(stored_key, extracted_key)

            verification_msg = (
                f"存储密钥: {stored_key[:KEY_HEX_CHARS]}...\n"
                f"提取密钥: {extracted_key[:KEY_HEX_CHARS]}...\n"
                f"重新生成: {regenerated_key[:KEY_HEX_CHARS]}...\n"
                f"比特一致率: {accuracy * 100:.2f}%"
            )
            if not match:
                return False, verification_msg
            if not extracted_key.startswith(regenerated_key[:KEY_HEX_CHARS]):
                return False, verification_msg + "\n⚠️ 提取密钥与重新生成密钥不匹配"
            return True, verification_msg + "\n✅ 所有验证通过"
        except Exception as e:
            return False, f"验证过程中出错: {str(e)}"

    # --------------------------------------------------------------- 评测
    def _require_loader(self, test_loader) -> None:
        if test_loader is None:
            raise WatermarkError("请先加载测试数据集")

    def evaluate_model(self, test_loader, sample_limit: Optional[int] = None) -> float:
        """分类准确率；``sample_limit`` 为空表示全量评估。"""
        self._require_loader(test_loader)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if sample_limit and total >= sample_limit:
                    break
        if total == 0:
            raise WatermarkError("测试数据集为空，无法评估")
        return 100.0 * correct / total

    def _quick_evaluate(self, test_loader, samples: int = 200) -> float:
        return self.evaluate_model(test_loader, sample_limit=samples)

    def _apply_prune(self, ratio: float) -> None:
        """只剪枝二维以上权重（跳过 BN 参数），按 ratio 比例置零。"""
        with torch.no_grad():
            for _, param in self.model.named_parameters():
                if param.dim() < 2:
                    continue
                mask = torch.rand_like(param) > ratio
                param.mul_(mask.float())

    def _apply_noise(self, sigma: float) -> None:
        with torch.no_grad():
            for _, param in self.model.named_parameters():
                if param.dim() < 2:
                    continue
                param.add_(torch.randn_like(param) * sigma)

    def _apply_finetune(self, test_loader, epochs: int, lr: float) -> None:
        """用 logits + CrossEntropyLoss 微调（避免对 softmax 输出再算 CE）。"""
        self._require_loader(test_loader)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(epochs):
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.logits(inputs), labels)
                loss.backward()
                optimizer.step()

    def robustness_test(self, test_loader,
                        attack_types: Sequence[str] = ("prune", "noise", "finetune"),
                        attack_params: Optional[Dict[str, Dict[str, float]]] = None) -> List[dict]:
        """鲁棒性测试：备份参数 → 施加攻击 → 评估 → 恢复参数。

        每项包含 ``accuracy``（攻击后准确率）、``bit_accuracy`` 与兼容字段
        ``extraction_rate``（比特一致率）、``attack_type``、``params``。
        """
        self._require_loader(test_loader)
        params = {k: dict(v) for k, v in DEFAULT_ATTACK_PARAMS.items()}
        for name, override in (attack_params or {}).items():
            params.setdefault(name, {}).update(override)

        original_state = copy.deepcopy(self.model.state_dict())
        results: List[dict] = []
        try:
            for attack in attack_types:
                self.model.load_state_dict(original_state)
                attack_param = dict(params.get(attack, {}))
                try:
                    if attack == "prune":
                        self._apply_prune(attack_param.get("ratio", 0.10))
                    elif attack == "noise":
                        self._apply_noise(attack_param.get("sigma", 0.05))
                    elif attack == "finetune":
                        self._apply_finetune(test_loader,
                                             int(attack_param.get("epochs", 2)),
                                             float(attack_param.get("lr", 0.001)))
                    else:
                        raise WatermarkError(f"未知攻击类型: {attack}")

                    accuracy = self._quick_evaluate(test_loader, samples=200)
                    rate = self.extraction_bit_accuracy()
                    results.append({
                        "attack_type": attack,
                        "params": attack_param,
                        "accuracy": accuracy,
                        "bit_accuracy": rate,
                        "extraction_rate": rate,        # 兼容旧字段名
                    })
                except Exception as e:
                    results.append({
                        "attack_type": attack,
                        "params": attack_param,
                        "error": str(e),
                    })
        finally:
            self.model.load_state_dict(original_state)
            self.model.eval()
        return results
