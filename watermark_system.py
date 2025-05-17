
# _*_ conding : utf-8 _*_
# @Time : 2025/4/9 14:27
# @Author : yang_4_ever
# @File : watermark_system
# @Project : FinalWMsys
import torch
import torch.nn as nn
from torchvision import models
import hashlib
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import random
import os
import time
import threading
import json
import copy


class ModelWatermarkSystem:
    def __init__(self, num_classes=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        self.model = None
        self.watermark = None
        self.trigger_set = None
        self.watermark_layer = 'layer4'
        self.watermark_key = None
        self.test_loader = None
        self.initialize_model()

    def initialize_model(self):
        try:
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        except AttributeError:
            try:
                self.base_model = models.resnet50(pretrained=True)
            except:
                self.base_model = models.resnet50()
                print("Warning: Could not load pretrained weights")

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.model = self._create_model()
        self.model.to(self.device)

        # 模型参数层打印
        # print("\n=== 模型参数层 ===")
        # for name, param in self.model.named_parameters():
        #     if len(param.shape) == 4:  # 只打印卷积层
        #         print(f"{name}: {param.shape}")
        # print("=================\n")

    def _create_model(self):
        class WatermarkModel(nn.Module):
            def __init__(self, base_model):
                super(WatermarkModel, self).__init__()
                self.net = base_model
                self.softmax = nn.Softmax(dim=1)

            def forward(self, x):
                x = self.net(x)
                x = self.softmax(x)
                return x

        return WatermarkModel(self.base_model)

    def load_pretrained_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.initialize_model()
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            return f"Pretrained model loaded from {path}"
        except Exception as e:
            return f"Error loading pretrained model: {str(e)}"

    def save_model(self, file_path=None):
        if file_path is None:
            return "No file path provided"

        if not file_path.lower().endswith('.pth'):
            file_path += '.pth'

        if os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = f"{base}_{timestamp}{ext}"

        save_data = {
            'model_state': self.model.state_dict(),
            'watermark_salt': self.watermark_key[0],
            'watermark_hash': self.watermark_key[1],
            'system_version': '1.0'
        }
        torch.save(save_data, file_path)
        return f"Model saved to new file: {file_path} (with salt)"

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.watermark_key = (
            checkpoint['watermark_salt'],
            checkpoint['watermark_hash']
        )
        self.model.load_state_dict(checkpoint['model_state'])
        return f"Model loaded from {file_path} (salt restored)"

    def generate_watermark(self, owner_info):
        """生成固定水印密钥"""
        salt = hashlib.sha256(owner_info.encode()).hexdigest()[:32]
        key = hashlib.pbkdf2_hmac(
            'sha256',
            owner_info.encode(),
            salt.encode(),
            100000
        ).hex()[:32]

        self.watermark_key = (salt, key)
        return key

    def embed_watermark_whitebox(self, strength=0.5):
        """更健壮的嵌入方法，自动查找目标层"""
        if self.watermark_key is None:
            raise ValueError("请先生成水印密钥（调用generate_watermark）")

        # 自动查找第一个4D卷积层的参数
        target_param, layer_name = None, ""
        for name, param in self.model.named_parameters():  # 注意去掉.net前缀
            if len(param.shape) == 4:  # 找到第一个4D卷积核
                target_param = param
                layer_name = name
                break

        if target_param is None:
            raise ValueError("❌ 找不到可用的卷积参数层！请检查模型结构")

        salt, key = self.watermark_key
#！        # binary_key = ''.join([f"{int(c, 16):04b}" for c in key])
        binary_key = ''.join([f"{int(c, 16):04b}" for c in key[:16]])  # 仅用前16位hex → 64位binary

#！       # 嵌入水印
        # with torch.no_grad():
        #     for i in range(min(32, target_param.shape[0])):  # 防止超出参数范围
        #         bit = int(binary_key[i])
        #         target_param[i, 0, 0, 0] = strength if bit else -strength

        with torch.no_grad():
            for i in range(64):  # 嵌入64位binary
                bit = int(binary_key[i])
                target_param[i, 0, 0, 0] = strength if bit else -strength

        return f"✅ 水印已嵌入到 [{layer_name}] 层（强度={strength}）"

    def extract_watermark_whitebox(self):
        """从模型参数中提取16位十六进制水印（64位二进制）"""
        # 查找目标参数层（与嵌入时相同）
        target_param = None
        for name, param in self.model.named_parameters():
            if len(param.shape) == 4:
                target_param = param
                break

        if target_param is None:
            raise ValueError("❌ 提取失败：找不到卷积参数层")

        # 检查参数层容量是否匹配
        if target_param.shape[0] < 64:
            raise ValueError(f"❌ 参数层 {name} 的通道数不足（需≥64，实际为{target_param.shape[0]}）")

        # 提取二进制（64位）
        binary_str = []
        with torch.no_grad():
            for i in range(64):  # 提取64位binary
                val = target_param[i, 0, 0, 0].item()
                binary_str.append('1' if val > 0 else '0')

        # 二进制转16位hex（每4位binary转1位hex）
        hex_key = ""
        for i in range(0, 64, 4):
            nibble = ''.join(binary_str[i:i + 4])
            hex_key += f"{int(nibble, 2):x}"

        return hex_key  # 返回完整的16位hex
    def verify_watermark(self, extracted_key):
        """验证提取的水印是否匹配"""
        if not self.watermark_key:
            return False
        salt, original_key = self.watermark_key
        # 比较前16位字符
        return extracted_key.startswith(original_key[:16])

    def verify_watermark_with_owner(self, owner_info):
        """增强版水印验证"""
        try:
            if not self.watermark_key:
                return False, "No watermark key available"

            # 提取当前水印
            extracted_key = self.extract_watermark_whitebox()

            # 验证提取的水印
            match = self.verify_watermark(extracted_key)

            # 重新生成密钥用于验证
            salt, stored_key = self.watermark_key
            regenerated_key = hashlib.pbkdf2_hmac(
                'sha256',
                owner_info.encode(),
                salt.encode(),
                100000
            ).hex()[:32]

            # 构建详细验证信息
            verification_msg = (
                f"存储密钥: {stored_key[:16]}...\n"
                f"提取密钥: {extracted_key[:16]}...\n"
                f"重新生成: {regenerated_key[:16]}..."
            )

            # 检查提取的密钥是否匹配存储的密钥
            if not match:
                return False, verification_msg

            # 检查重新生成的密钥是否匹配
            if not extracted_key.startswith(regenerated_key[:16]):
                return False, verification_msg + "\n⚠️ 提取密钥与重新生成密钥不匹配"

            return True, verification_msg + "\n✅ 所有验证通过"

        except Exception as e:
            return False, f"验证过程中出错: {str(e)}"

    def evaluate_model(self, test_loader):
        """评估模型分类性能"""
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

        accuracy = 100 * correct / total
        return accuracy

    def test_watermark_extraction(self, trials=5):  # 原10次 → 5次
        if not self.watermark_key:
            return 0.0
        success = 0
        original_key = self.watermark_key[1][:16]
        for _ in range(trials):
            try:
                extracted = self.extract_watermark_whitebox()[:16]
                if extracted == original_key:
                    success += 1
            except:
                continue
        return success / trials

    def robustness_test(self, test_loader, attack_types=['prune', 'noise', 'finetune']):
        """
        鲁棒性测试
        返回: {
            'accuracy': 攻击后准确率,
            'extraction_rate': 水印提取成功率,
            'attack_type': 攻击类型
        }
        """

        original_state = copy.deepcopy(self.model.state_dict())  # 深度拷贝
        results = []

        try:
            for attack in attack_types:
                # 备份原始参数
                self.model.load_state_dict(original_state)

                if attack == 'prune':
                    # 随机剪枝10%的权重
                    with torch.no_grad():
                        for param in self.model.parameters():
                            mask = torch.rand_like(param) > 0.3
                            param.mul_(mask.float())

                elif attack == 'noise':
                    # 添加高斯噪声
                    with torch.no_grad():
                        for param in self.model.parameters():
                            param.add_(torch.randn_like(param) * 0.05)

                elif attack == 'finetune':
                    # 模拟5个epoch的微调
                    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
                    criterion = nn.CrossEntropyLoss()

                    for _ in range(2):
                        for inputs, labels in test_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            optimizer.zero_grad()
                            outputs = self.model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()

                # 评估攻击后的性能
                acc = self._quick_evaluate(test_loader, samples=200)  # 新增快速评估方法
                ext_rate = self.test_watermark_extraction(trials=5)  # 原10次 → 5次

                results.append({'accuracy': acc, 'extraction_rate': ext_rate, 'attack_type': attack})
        finally:
            self.model.load_state_dict(original_state)  # 最终确保恢复
        return results

    def _quick_evaluate(self, test_loader, samples=100):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):
                if i * inputs.shape[0] >= samples:  # 限制评估样本数
                    break
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total