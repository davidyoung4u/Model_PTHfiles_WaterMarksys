# WMsys_enhanced — DNN 模型白盒水印系统

基于 **PyQt5 + PyTorch** 的 ResNet50 模型白盒水印工具：生成所有者密钥 → 嵌入白盒水印 →
提取/验证 → 性能与鲁棒性评测，全部通过图形界面完成。

> 本仓库只托管**代码与文档**，不包含数据集。`weather_photos/` 等数据需自行准备（见下方「数据集」）。

---

## 1. 功能概览

| 模块 | 能力 |
| --- | --- |
| 模型操作 | 加载 ResNet50 预训练权重（合并进当前网络，不重建模型）、保存带水印模型、加载水印模型 |
| 水印生成 | 由所有者信息派生密钥，`sha256(owner)[:32]` 作盐 + PBKDF2-HMAC-SHA256/100000 |
| 白盒水印 | 把 64 bit 写入 `net.layer4.2.conv2.weight` 的每个输出通道（每通道 1 bit），强度可调（0.01–1.0） |
| 水印验证 | 提取值、存储值、由所有者信息重新生成值三方比对，输出比特一致率 |
| 性能测试 | 原始准确率 + 水印比特一致率 + 鲁棒性评测（剪枝 / 噪声 / 微调） |

---

## 2. 环境要求

| 组件 | 版本 |
| --- | --- |
| Python | 3.9.21 |
| torch / torchvision | 2.2.2+cu118 / 0.17.2+cu118 |
| PyQt5 | 5.15.11 |
| numpy / pillow | 1.26.3 / 11.0.0 |

**GPU 环境（CUDA 11.8）**

```powershell
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu118
```

**CPU 环境**

```powershell
pip install torch==2.2.2 torchvision==0.17.2 PyQt5==5.15.11 numpy==1.26.3 pillow==11.0.0
```

---

## 3. 快速开始

```powershell
python main.py
```

- 首次运行会在 `~/.cache/torch/hub/checkpoints/` 自动下载 **ResNet50 (IMAGENET1K_V1)** 预训练权重（约 100 MB）。
- 程序启动时加载模型，加载期间如需下载权重会稍慢，请耐心等待。

---

## 4. 核心算法流程

```
所有者信息 owner
    │
    ▼
密钥派生：salt = sha256(owner)[:32]
          key  = PBKDF2-HMAC-SHA256(owner, salt, 100000 iter) 取前 32 位 hex
          其中前 16 位 hex（64 bit）参与嵌入
    │
    ▼
水印嵌入：64 bit 写入 net.layer4.2.conv2.weight
          每个输出通道 1 bit，强度可调（0.01–1.0）
          嵌入前校验通道容量，不足抛 CapacityError
    │
    ▼
水印提取/验证：提取值 ↔ 存储值 ↔ 由 owner 重新生成值
              三方比对，输出比特一致率（extraction_bit_accuracy）
    │
    ▼
鲁棒性评测：prune / noise / finetune 攻击后再提取，统计一致率
```

---

## 5. 数据集

本仓库**不包含**测试数据集。性能测试所需的目录需为 `ImageFolder` 结构：

```
数据集根目录/
├── cloudy/*.jpg
├── rain/*.jpg
├── shine/*.jpg
└── sunrise/*.jpg
```

开发时使用的样例数据集为 `weather_photos/`：**cloudy 300 张、rain 215 张、shine 253 张、
sunrise 357 张，共 1125 张 / 4 类**（类别不均衡，`sunrise : rain ≈ 1.66 : 1`）。

> ⚠️ 数据加载失败时程序会**直接报错并终止测试**，不会静默使用随机数据替代——避免产生无意义的指标。

---

## 6. 目录结构

```
WMsys_enhanced/
├── main.py                  # 程序入口（高 DPI 适配、全局异常钩子、随机种子）
├── gui.py                   # PyQt5 主界面与交互编排
├── watermark_system.py      # 水印算法核心：密钥派生/嵌入/提取/评估/攻击
├── utils.py                 # 中英文文案字典
├── resources/
│   └── wm.png               # 窗口图标（运行时资源，必须保留）
├── docs/
│   └── REFACTOR_FRAMEWORK.md  # 重构与功能优化框架
├── requirements.txt         # 依赖清单
└── .gitignore
```

以下内容**不纳入版本库**（已在 `.gitignore` 中排除）：`weather_photos/`、`*.jpg`、
`logi.txt`、`resources/el_wm.txt`、`环境及包版本.txt`、`.idea/`、`__pycache__/`、模型权重 `*.pth`。 

---

## 7. P0 修复要点

| 问题 | 处理 |
| --- | --- |
| 嵌入层定位错误（原实现写入 `net.conv1.weight`） | 按算法版本显式定位 `net.layer4.2.conv2.weight`，嵌入前校验容量 |
| 通道不足分支 `NameError` | 改为显式异常 `CapacityError` / `LayerNotFoundError`，报错带层名 |
| `save_model` 无水印时崩溃、`load_model` 缺 `map_location`/`weights_only` | 前置校验 + 安全反序列化（失败降级并提示）+ CPU 映射 |
| 数据加载失败静默使用随机数据 | 直接报错并终止测试，绝不伪造指标 |
| 预处理缺 `Normalize`、评估集 `shuffle=True` | 统一 IMAGENET1K 预处理；评估集固定 `shuffle=False` |
| "提取成功率"恒为 0/1 | 改为**比特一致率**（`extraction_bit_accuracy`） |
| 微调攻击对 softmax 输出再算交叉熵 | 改用 logits 训练，并正确切换 train/eval |
| 剪枝口径与注释不符、会剪掉 BN 参数 | 参数化 `ratio`（默认 0.10），仅剪二维以上权重 |
| 无随机种子 | 启动与评测前固定种子（默认 42） |
| 资源路径依赖工作目录、语言切换残留 | `Path(__file__)` 定位 `resources/`；标签统一登记并随语言刷新 |
| 无依赖清单与忽略规则 | 新增 `requirements.txt`、`.gitignore`、本 README |

---

## 8. 兼容性说明

- 密钥派生参数已冻结，**同一所有者信息在旧版本与新版本得到相同密钥**。
- checkpoint 新增 `watermark_algorithm_version` 字段：
  - 旧文件（无该字段）按**算法版本 1**（`net.conv1.weight`）提取，加载时提示“建议重新嵌入水印”；
  - 用新版本重新嵌入的模型按**算法版本 2**（`net.layer4.2.conv2.weight`）提取。
- 结论：**旧模型仍可加载与验证，但建议重新嵌入**，以使用更深的层获得更强鲁棒性。

---

## 9. 后续计划

分阶段重构方案（分层、并发、算法插件化、测试与打包）见
[docs/REFACTOR_FRAMEWORK.md](docs/REFACTOR_FRAMEWORK.md)。当前已完成其中的 P0 止血阶段。

---

## 10. 许可

本项目未附带开源许可证文件。如需公开分发或复用，请先与作者确认授权方式。
