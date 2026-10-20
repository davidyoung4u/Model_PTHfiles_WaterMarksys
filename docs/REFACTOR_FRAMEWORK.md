# WMsys_enhanced 重构与功能优化框架

> 目标读者：项目维护者（yang_4_ever）
> 范围：`E:\Python_PROJECT\WMsys_enhanced`（PyQt5 + PyTorch 的 DNN 模型水印工具）
> 本文既是"现状诊断书"，也是"目标架构 + 分阶段落地路线"。

---

## 0. 一页结论

| 维度 | 现状 | 目标 |
| --- | --- | --- |
| 代码规模 | 4 个文件，约 1150 行，全部耦合在一起 | 分层包结构，单文件 < 300 行，职责单一 |
| 分层 | UI 里直接跑 torch 训练/剪枝/评估 | UI → Service → Domain ← Adapter（单向依赖） |
| 并发 | 主线程跑长任务 + `processEvents()` 打补丁 | `QThreadPool` 作业 + 信号量进度/取消，UI 永不阻塞 |
| 算法 | 硬编码在 `ModelWatermarkSystem` 里，层定位靠"第一个 4D 参数" | 算法/攻击 **插件注册表**（Strategy + Registry），可扩展黑盒水印 |
| 错误处理 | `except` 后返回 `"Error ..."` 字符串，靠日志文本判断状态 | 领域异常 + `OperationResult` DTO，UI 按 code 决定提示级别 |
| 评测可信度 | 静默回退随机数据、无 train/test 划分、无 Normalize、无随机种子 | 数据集校验失败即报错，统一预处理，固定种子，报告可导出 |
| 工程质量 | 无 requirements/README/测试/日志/配置 | `pyproject.toml` + pytest + 结构化日志 + 配置对象 + CI |

**一句话方案**：先做"止血补丁"（P0），再把 `ModelWatermarkSystem` 拆成 `algorithms` + `torch_adapter` + `services` 三层，最后把耗时操作搬进 `QThreadPool` 作业，UI 只保留视图与翻译。

---

## 1. 现状结构画像

### 1.1 文件清单

| 文件 | 行数 | 实际职责 | 评价 |
| --- | --- | --- | --- |
| `main.py` | 15 | 入口 | 无高 DPI、无异常钩子、无日志 |
| `gui.py` | 647 | 视图 + 对话框 + 业务编排 + 数据加载 + 评测流程 + 线程补丁 | **上帝类**，17 个职责 |
| `watermark_system.py` | 336 | 模型工厂 + 密钥派生 + 嵌入/提取 + 评估 + 攻击 + 存取 | 领域/基础设施混在一起 |
| `utils.py` | 138 | 中英文文案字典 | 唯一的"配置" |
| `logi.txt` | 57 | 旧版验证逻辑草稿（未接入） | 死代码 |
| `resources/el_wm.txt` | 382 | 旧版 `watermark_system` 备份（注释形式） | 死代码，但是**兼容性证据**（旧版嵌入在 `net.layer4.2.conv2.weight`） |
| `环境及包版本.txt` | 144 | 手工记录的包版本 | 应改为 `requirements.txt` |
| `weather_photos/` | 1125 张 | 4 类图像数据集 | 见 1.3 |

### 1.2 依赖与调用关系（现状）

```
main.py
  └── gui.WatermarkApp ──────────────► utils.LANGUAGES (文案)
            │
            └── watermark_system.ModelWatermarkSystem
                      ├── torchvision.models.resnet50 (构造时即加载权重)
                      ├── hashlib (PBKDF2 密钥派生)
                      ├── torch.save / torch.load (.pth 存取)
                      └── 攻击/评估 (剪枝/噪声/微调/准确率)

gui.py 另外直接使用：torch / torchvision.transforms / DataLoader / Dataset  ← 视图层直接碰 ML 栈
```

运行时链路（现状）：`QApplication → WatermarkApp.__init__ → ModelWatermarkSystem.__init__ → 加载 ResNet50 预训练权重（阻塞、可能触发下载）` → 用户点击 → **主线程**同步执行推理/微调/评估。

### 1.3 数据与环境事实

- 数据集：`weather_photos/{cloudy 300, rain 215, shine 253, sunrise 357}` = **1125 张，4 类，类别不均衡**（`sunrise : rain ≈ 1.66 : 1`）。
- `num_classes=4` 是硬编码默认值（`watermark_system.py:24`），与数据集恰好一致，但**没有从数据推断**，换数据集即静默错位。
- 环境：Python 3.9.21、PyQt5 5.15.11、torch 2.2.2+cu118、torchvision 0.17.2+cu118（见 `环境及包版本.txt`）。
- 无 `.pth` 样例模型、无 README、无 requirements、无测试、无 git 忽略规则（`.idea/`、`__pycache__/` 会进版本库）。

---

## 2. 问题清单（按严重度）

### P0 — 功能性缺陷（会产生错误结论或直接崩溃）

| # | 位置 | 问题 | 影响 |
| --- | --- | --- | --- |
| 1 | `watermark_system.py:132-158` | 嵌入时取"第一个 4D 参数"，实际是 `net.conv1.weight`，而 `self.watermark_layer='layer4'` 从未被使用；旧版本（`resources/el_wm.txt:169`）明确写 `net.layer4.2.conv2.weight` | 水印嵌在最浅、最脆弱的层；**旧模型用新代码提取必失败**（层不一致） |
| 2 | `watermark_system.py:174` | 通道不足分支里引用 `name`，而该变量只在上一个循环中存在 | `NameError`（错误处理路径自身崩溃） |
| 3 | `gui.py:451-563` | 性能测试整段跑在 UI 线程，靠 `processEvents()`（`gui.py:68`）刷新 | 界面假死；取消按钮在长耗时 torch 调用中无响应 |
| 4 | `gui.py:565-621` | 数据加载失败**静默回退**到 `RandomDataset`（`randint(0,4)` 假标签） | 输出的准确率/鲁棒性数字无意义，且用户不知道 |
| 5 | `gui.py:587-601`, `628-638` | 预处理只有 `Resize/CenterCrop/ToTensor`，缺 `Normalize(IMAGENET1K 均值方差)` | 与预训练权重期望分布不符，准确率系统性偏低 |
| 6 | `watermark_system.py:84-112` | `save_model` 在 `watermark_key is None` 时下标越界；`load_model` 缺 `map_location`、未用 `weights_only` | 崩溃 / 跨设备加载失败 / 反序列化安全风险 |
| 7 | `watermark_system.py:256-268` | `test_watermark_extraction` 重复 5 次**同一确定性提取** | "提取成功率"恒为 0% 或 100%，无信息量；应为 BER（比特错误率） |
| 8 | `watermark_system.py:301-313` | "微调攻击"对**已含 softmax 的输出**再用 `CrossEntropyLoss`，且未 `train()`（BN/Dropout 仍为 eval） | 双重 softmax + 未真正训练，鲁棒性结论失真 |
| 9 | `gui.py:123-153` | 语言切换只刷新部分控件（`owner_info`、`embed_strength` 等 `QLabel` 不更新；`performance_test` 组未纳入 `groups`） | 中英混杂 |
| 10 | 全流程 | 无随机种子（`torch.randn`、`shuffle=True`） | 结果不可复现 |
| 11 | `watermark_system.py:114-125` | `salt = sha256(owner)[:32]`，盐值由 owner 推导而非随机；密钥明文写入 checkpoint | 可离线爆破的验证强度；密钥不保密 |
| 12 | `watermark_system.py:35-48`, `71-80` | `__init__` 与 `load_pretrained_model` 各建一次 ResNet50 | 启动阻塞、显存翻倍、可能触发权重下载 |
| 13 | `gui.py:446-449` | 图标路径 `os.path.join("resources", ...)` 依赖当前工作目录 | 从别处启动即丢失图标 |

### P1 — 结构性/维护性问题

- **上帝类**：`WatermarkApp` 同时是视图、控制器、服务定位器；`ModelWatermarkSystem` 同时是模型工厂、密码学、算法、评测框架、序列化层。
- **状态以字符串传递**：服务层 `return f"Error loading ..."`，UI 只能打印；无法区分"未加载模型"和"文件损坏"。
- **UI 与 ML 栈耦合**：`gui.py` 直接 `import torch` 并构造 `DataLoader`，换框架（TensorFlow/ONNX）需要改 UI。
- **重复加载模型**、**隐含全局状态**（`self.watermark_system.test_loader` 被 `finally` 置 None，导致每次测试都要重选目录）。
- **死代码/未用导入**：`load_custom_dataset`、`logi.txt`、`el_wm.txt`、`random/threading/json/numpy/PIL` 导入未使用。
- **零测试**、零类型注解、无 lint/format 配置、无打包脚本。
- 无日志系统（只有 `print` 与 `QTextEdit`），无配置文件，常量散落。

### P2 — 功能优化机会（重构后可低成本获得）

| 现有能力 | 可升级为 | 归属模块 |
| --- | --- | --- |
| 固定白盒符号水印 | 多算法插件：白盒符号 / 触发集黑盒 / 权重正则化，可对比 | `algorithms/` |
| 3 种攻击 | 攻击库 + 参数化（剪枝率、噪声 σ、微调 epoch/学习率、量化位宽），可跑矩阵 | `attacks/` |
| 单次评测 | 批量实验 + 强度-准确率曲线 + BER 曲线，结果落盘可复现 | `services/` + `report_service` |
| 手抄结果 | 报告导出 JSON/CSV/Markdown | `report_service.py` |
| 硬编码 ResNet50 | 骨干注册表（ResNet 系列 / ViT / 自定义 `.pth`），类别数自动推断 | `torch_adapter/backbones.py` |
| 无容量提示 | 嵌入前显示"目标层容量 N bit / 需要 M bit" | `layer_locator.py` |
| 无模型对比 | 模型库：加载多个 `.pth` 并排比较水印状态与准确率 | `services/` + `ui/views/` |

---

## 3. 目标架构

### 3.1 分层与依赖规则

```
┌──────────────────────────── presentation (wmsys/ui) ────────────────────────────┐
│ MainWindow / Panels / Dialogs / Workers(QThreadPool) / I18n / AppState          │
└───────────────▲───────────────────────────────────────────────▲────────────────┘
                │ 只依赖 DTO + 信号                              │ 只读配置
┌───────────────┴──────────────── application (wmsys/services) ──┴────────────────┐
│ WatermarkService / EvaluationService / ReportService                            │
│ 编排 + 进度回调 + 取消令牌 + 事务式备份/恢复，不 import Qt                        │
└───────────────▲───────────────────────────────────────────────▲────────────────┘
                │ 只依赖抽象端口                                │
┌───────────────┴─────────────────── domain (wmsys/core) ────────┴────────────────┐
│ entities(DTO) / errors / ports(Protocol) / seed                                 │
│ 纯 Python，禁止 import torch、PyQt5、torchvision                                 │
└───────────────▲────────────────────────────────────────────────────────────────┘
                │ 实现端口
┌───────────────┴──────── infrastructure (torch_adapter / data / attacks / infra) ┐
│ backbones / layer_locator / checkpoint_io / evaluator / transforms / config /   │
│ logging / paths                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**依赖铁律**（可用 `import-linter` 或 CI 正则校验）：

1. `core` 只 import 标准库。
2. `services` 只 import `core`（端口/实体/异常），不 import torch 与 Qt。
3. `ui` 只 import `services` 的 DTO 与 `infra.config/i18n`，**不 import torch**。
4. `torch_adapter`/`attacks`/`data` 实现 `core.ports`，可 import torch。
5. 唯一组合根是 `app/container.py`。

### 3.2 目标目录树

```
WMsys_enhanced/
├── app/
│   ├── main.py                     # QApplication、高DPI、异常钩子、主题、装配
│   └── container.py                # 组合根：config→logger→adapters→services
├── wmsys/
│   ├── core/
│   │   ├── entities.py             # WatermarkKey / ModelRef / CheckpointMeta / MetricResult / EvaluationReport / OperationResult
│   │   ├── errors.py               # WMsysError 体系 + 退出码 + i18n key
│   │   ├── ports.py                # WatermarkAlgorithm / Attack / BackboneFactory / DatasetProvider / CheckpointRepository / Progress / CancelToken
│   │   └── seed.py                 # set_global_seed(42)
│   ├── algorithms/
│   │   ├── registry.py             # @register_algorithm("whitebox-sign")
│   │   ├── whitebox_sign.py        # 现有白盒符号水印（修正层定位 + 容量校验 + 算法版本）
│   │   └── blackbox_trigger.py     # 预留：触发集黑盒水印
│   ├── attacks/
│   │   ├── registry.py
│   │   ├── pruning.py  noise.py  finetune.py  quantization.py
│   ├── torch_adapter/
│   │   ├── backbones.py            # 骨干工厂 + 本地权重缓存 + 类别数注入
│   │   ├── layer_locator.py        # 定位策略：first-4d / by-name / max-capacity
│   │   ├── checkpoint_io.py        # v1.0 旧格式读兼容 + v1.1 写 + map_location/weights_only
│   │   ├── evaluator.py            # 全量/采样准确率、混淆矩阵、BER
│   │   └── device.py
│   ├── data/
│   │   ├── transforms.py           # 训练/评估两套 transform（含 Normalize）
│   │   ├── imagefolder.py          # 类别发现 + 校验（空目录/单类即报错）
│   │   └── split.py                # 固定种子分层划分 train/val/test
│   ├── services/
│   │   ├── watermark_service.py    # generate → embed → extract → verify → save/load
│   │   ├── evaluation_service.py   # baseline→extraction→attacks 矩阵→报告
│   │   └── report_service.py       # JSON / CSV / Markdown 导出
│   ├── infra/
│   │   ├── config.py               # @dataclass Config + YAML/环境变量覆盖
│   │   ├── logging.py              # 文件轮转 + Qt 日志桥
│   │   └── paths.py                # Path(__file__) 定位 resources/，不依赖 cwd
│   └── ui/
│       ├── main_window.py          # 只做布局/装配/信号连接
│       ├── state.py                # AppState(QObject)：current_model/watermark/dataset + 信号
│       ├── workers.py              # Job(QRunnable) + JobRunner(QThreadPool) + CancelToken
│       ├── i18n/i18n.py            # I18n(QObject).language_changed + retranslate 协议
│       ├── panels/                 # operation_panel / output_panel / verify_panel / test_panel
│       └── dialogs/                # help_dialog / progress_dialog / about_dialog
├── tests/
│   ├── unit/                       # 密钥确定性、层定位、checkpoint 往返、i18n 完整性
│   ├── integration/                # 服务编排（用 fake 端口，不下载 ResNet50）
│   └── fixtures/legacy_v1.0.pth    # 旧格式兼容基准
├── resources/{wm.png, strings/}
├── docs/REFACTOR_FRAMEWORK.md
├── pyproject.toml  requirements.txt  .gitignore  README.md
└── scripts/{run.ps1, package.ps1}
```

### 3.3 关键设计模式

| 模式 | 用途 |
| --- | --- |
| **端口-适配器（Hexagonal）** | `core.ports` 定义 `WatermarkAlgorithm/Attack/BackboneFactory`，torch 实现放外层，UI 与 torch 解耦且可 mock |
| **策略 + 注册表** | 水印算法、攻击方式均按名字注册，新增不改编排代码 |
| **命令 / 作业** | 每个用户操作封装为 `Job`，统一进度、取消、异常、日志 |
| **观察者（Qt 信号）** | `AppState` + service 进度回调 → UI 更新，取代 `processEvents()` |
| **DTO / Result 对象** | 服务返回 `OperationResult(ok, code, message_key, payload)`，UI 决定弹窗级别，不再靠字符串判断 |
| **组合根（DI）** | `container.build()` 注入 config/logger/adapters，测试时替换为 fake |

---

## 4. 关键接口契约（可直接落地的骨架）

### 4.1 领域实体 `wmsys/core/entities.py`

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class WatermarkKey:
    salt: str
    digest: str
    algorithm: str = "whitebox-sign"
    version: int = 1
    @property
    def short(self) -> str:            # UI 展示用
        return self.digest[:16]

@dataclass
class EmbedReport:
    algorithm: str
    target_layer: str
    bits_embedded: int
    strength: float
    capacity_bits: int
    algorithm_version: int = 1

@dataclass
class VerifyResult:
    matched: bool
    extracted: str
    ber: float | None = None          # 比特错误率，替代"提取成功率"
    detail: dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricResult:
    name: str
    value: float
    unit: str = "%"
    detail: dict[str, Any] = field(default_factory=dict)

@dataclass
class RobustnessRow:
    attack: str
    params: dict[str, Any]
    accuracy: MetricResult
    ber: float
    matched: bool
    error: str | None = None

@dataclass
class EvaluationReport:
    created_at: str
    dataset_dir: str
    classes: list[str]
    split: dict[str, int]
    baseline: MetricResult
    extraction: MetricResult
    robustness: list[RobustnessRow]
    notes: list[str] = field(default_factory=list)
    seed: int = 42

@dataclass
class OperationResult:
    ok: bool
    code: str                          # "ok" / "no_model" / "no_watermark" / "io_error" / "cancelled"
    message_key: str                   # i18n key，例如 "log.model_loaded"
    message_args: dict[str, Any] = field(default_factory=dict)
    payload: Any = None
```

### 4.2 端口 `wmsys/core/ports.py`

```python
from typing import Protocol, Iterable, Callable

ProgressFn = Callable[[int, str], None]      # (百分比, i18n key 或文本)

class CancelToken(Protocol):
    def raise_if_cancelled(self) -> None: ...
    def is_cancelled(self) -> bool: ...

class WatermarkAlgorithm(Protocol):
    name: str
    algorithm_version: int
    def capacity_bits(self, model) -> int: ...
    def embed(self, model, key, strength: float) -> EmbedReport: ...
    def extract(self, model) -> str: ...
    def verify(self, extracted: str, key) -> VerifyResult: ...

class Attack(Protocol):
    name: str
    def params_schema(self) -> dict: ...                       # UI 据此生成参数控件
    def apply(self, model, data_loader, ctx) -> None: ...

class BackboneFactory(Protocol):
    def create(self, name: str, num_classes: int) -> object: ...
    def available(self) -> list[str]: ...

class CheckpointRepository(Protocol):
    def save(self, path: str, model, key, meta: dict) -> str: ...
    def load(self, path: str) -> tuple[object, object, dict]:  # (model, key, meta)
        ...

class DatasetProvider(Protocol):
    def describe(self, path: str) -> dict: ...                 # classes/counts/是否可分层
    def loaders(self, path: str, batch_size: int, seed: int) -> dict: ...
```

### 4.3 白盒算法（修正版关键逻辑）`wmsys/algorithms/whitebox_sign.py`

```python
LAYER_CANDIDATES = ("net.layer4.2.conv2.weight", "net.layer4.2.conv2.bias")

class WhiteboxSignAlgorithm:
    name = "whitebox-sign"
    algorithm_version = 2      # v1 = 旧版 conv1 方案（见 core/ports 兼容说明）

    def __init__(self, locator: LayerLocator, bits: int = 64):
        self.locator, self.bits = locator, bits

    def capacity_bits(self, model) -> int:
        name, param = self.locator.locate(model, strategy="by-name", candidates=LAYER_CANDIDATES)
        return param.shape[0]                       # 每通道 1 bit

    def embed(self, model, key: WatermarkKey, strength: float) -> EmbedReport:
        name, param = self.locator.locate(model, strategy="by-name", candidates=LAYER_CANDIDATES)
        capacity = param.shape[0]
        if capacity < self.bits:
            raise CapacityError(layer=name, need=self.bits, got=capacity)
        bits = bits_of(key.digest, self.bits)        # hex → 二进制，长度严格校验
        with torch.no_grad():
            for i, bit in enumerate(bits):
                param[i, 0, 0, 0] = strength if bit else -strength
        return EmbedReport(self.name, name, self.bits, strength, capacity, self.algorithm_version)
```

要点：**先定位再校验再写入**，错误信息带上真实层名；算法版本写进 checkpoint 以便兼容旧模型。

### 4.4 应用服务 `wmsys/services/evaluation_service.py`

```python
class EvaluationService:
    def __init__(self, evaluator, attacks: AttackRegistry, algo_registry, clock=time.time):
        ...

    def run(self, req: EvaluationRequest, progress: ProgressFn, cancel: CancelToken) -> EvaluationReport:
        progress(5, "log.loading_dataset")
        ds = self.datasets.describe(req.dataset_dir)          # 空目录/单类 → 直接抛异常，绝不造数据
        loaders = self.datasets.loaders(req.dataset_dir, req.batch_size, req.seed)

        backup = self.evaluator.snapshot(model)
        try:
            baseline = self.evaluator.accuracy(model, loaders["test"], sample=req.sample_limit)
            cancel.raise_if_cancelled()
            extraction = self.evaluator.ber(model, algo, key)  # BER 取代"成功率"
            rows = []
            for i, attack in enumerate(req.attacks):
                cancel.raise_if_cancelled()
                progress(50 + i * 15, f"log.attack.{attack.name}")
                self.evaluator.restore(model, backup)
                attack.apply(model, loaders["train"], ctx=AttackContext(seed=req.seed, **attack.params))
                rows.append(RobustnessRow(...))
            return EvaluationReport(...)
        finally:
            self.evaluator.restore(model, backup)              # 事务式恢复，异常也不污染模型
            torch.cuda.empty_cache()
```

### 4.5 并发与进度 `wmsys/ui/workers.py`

```python
class JobSignals(QObject):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)      # OperationResult / EvaluationReport
    failed = pyqtSignal(str, str)      # code, message_key

class Job(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.signals, self._token = JobSignals(), CancelTokenImpl()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        try:
            result = self._fn(*self._args, progress=self.signals.progress.emit,
                              cancel=self._token, **self._kwargs)
            self.signals.finished.emit(result)
        except Cancelled:
            self.signals.finished.emit(OperationResult(False, "cancelled", "log.cancelled"))
        except WMsysError as e:
            self.signals.failed.emit(e.code, e.message_key)
        except Exception as e:                      # 兜底：记录堆栈，不让线程静默死亡
            log.exception("job failed")
            self.signals.failed.emit("internal_error", str(e))

    def cancel(self): self._token.cancel()
```

`ProgressDialog` 改为**被动视图**：只接收 `progress` 信号，取消时调用 `job.cancel()`，不再需要 `processEvents()`。

### 4.6 国际化 `wmsys/ui/i18n/i18n.py`

```python
class I18n(QObject):
    language_changed = pyqtSignal(str)

    def __init__(self, catalogs: dict[str, dict[str, str]], lang: str = "zh"):
        super().__init__()
        self._catalogs, self._lang, self._sinks = catalogs, lang, []

    def text(self, key: str, **fmt) -> str:
        raw = self._catalogs.get(self._lang, {}).get(key) or self._catalogs["en"].get(key, key)
        return raw.format(**fmt) if fmt else raw

    def bind(self, apply_fn) -> None:
        """控件注册自己的 retranslate 回调，切换语言时统一刷新，取代 findChild(objectName)。"""
        self._sinks.append(apply_fn)
        apply_fn()

    def set_language(self, lang: str) -> None:
        self._lang = lang
        for fn in self._sinks:
            fn()
        self.language_changed.emit(lang)
```

配套约束：**每个含文案的 widget 类都实现 `retranslate(self, i18n)`**，`MainWindow.retranslate` 向下递归调用；新增测试 `tests/unit/test_i18n.py` 断言 `zh`/`en` 键集合完全一致。

---

## 5. 分阶段落地路线

### P0 止血（0.5–1 天，不改架构）

1. 层定位修正：显式定位 `net.layer4.2.conv2.weight`，容量不足抛带层名的异常；checkpoint 增加 `algorithm_version`。
2. 修 `watermark_system.py:174` 的 `NameError`。
3. `save_model` 校验密钥；`load_model` 加 `map_location="cpu"` + `weights_only=True`。
4. 删除随机数据回退：改为 `raise DatasetNotConfigured`，UI 弹错误提示。
5. 预处理补 `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`；测试集 `shuffle=False`。
6. 固定随机种子（torch/numpy/random）。
7. 资源路径改 `Path(__file__).resolve().parent / "resources"`。
8. 产出 `requirements.txt`、`.gitignore`（`.idea/`、`__pycache__/`、`*.pth`）、README 骨架。

**DoD**：能加载旧 `.pth` 并给出"算法版本不一致，使用 legacy 定位器"的明确提示；不再出现静默假指标；启动时只加载一次模型。

### P1 分层剥离（2–3 天）

- 建包骨架与 `core` 三件套（entities/errors/ports），把 `ModelWatermarkSystem` 按职责切开：
  `initialize_model → torch_adapter/backbones`、密钥 → `core/crypto`（仍用 PBKDF2，参数不变）、嵌入/提取 → `algorithms/whitebox_sign`、评估/攻击 → `torch_adapter/evaluator` + `attacks/*`、存取 → `torch_adapter/checkpoint_io`。
- `gui.py` 只保留视图，业务调用改为 `WatermarkService`，返回值改为 `OperationResult`。
- 引入 `infra/config.py`（device、batch_size、strength 默认值、sample_limit、seed、语言）。

**DoD**：`wmsys/core` 与 `wmsys/services` 里 `grep -r "torch\|PyQt5"` 为空；`gui` 拆分后单文件 < 300 行；功能与 P0 完全等价。

### P2 并发与体验（1–2 天）

- 所有耗时操作走 `JobRunner`；进度条/日志改为信号驱动；取消真正生效（`CancelToken` 在采集循环与攻击循环中检查）。
- 启动不加载权重，改为"加载模型"时才构建骨干，并在状态栏显示设备与显存。
- 统一日志面板：`logging` → `QtLogBridge` 信号 → `output_panel`，文件日志轮转。
- 语言切换走 `I18n.bind`；窗口尺寸/位置、最近使用目录持久化（`QSettings`）。

**DoD**：任何操作期间 UI 可拖动、可取消；无 `processEvents()` 调用；取消后模型参数与加载器状态全部复原。

### P3 算法与评测插件化（2–3 天）

- `algorithms/registry` + `attacks/registry`，攻击参数化并暴露 `params_schema` 自动生成控件。
- 数据划分（分层 train/val/test，固定种子）+ 两套 transform + 类别自动发现（`num_classes = len(classes)`）。
- BER 取代"提取成功率"；报告导出 JSON/CSV/Markdown；批量实验（强度 × 攻击）与曲线导出。
- 预留黑盒触发集水印实现，验证"插件式"扩展成本 < 100 行。

**DoD**：新增一种攻击只需新增一个文件 + 注册一行；评测报告可一键导出并在另一台机器复现（同种子同结果）。

### P4 工程质量（持续）

- pytest：`core` 与 `services` 覆盖率 ≥ 70%；用 fake 端口做集成测试（**不下载 ResNet50**）。
- ruff + black + mypy（`core`/`services` 严格，`ui` 宽松）；CI 跑 lint + test。
- 打包：PyInstaller spec（含 `resources/`）或 `pip install -e .` + 快捷方式；版本号写入 checkpoint。
- 文档：README（安装/使用/截图）、`docs/architecture.md`、CHANGELOG。

---

## 6. 兼容性与风险控制

| 风险 | 说明 | 对策 |
| --- | --- | --- |
| 旧 `.pth` 无法提取 | 旧版水印在 `net.layer4.2.conv2.weight`（`resources/el_wm.txt`），当前代码读 `net.conv1.weight` | `checkpoint_io` 写 `watermark.algorithm_version`；读到无版本号的旧文件时按 legacy 定位器提取，并提示"建议重新嵌入" |
| 密钥派生变更破坏验证 | 现有 `salt=sha256(owner)[:32]` + `PBKDF2-HMAC-SHA256/100000` | 派生参数冻结为常量并加单元测试锁死；若引入随机盐，走新 `version` 并在密钥结构里同时保存旧字段 |
| 层定位从 conv1 改为 layer4 后准确率波动 | 单点权重改写量：conv1 每通道 1 个权重 / 9408；layer4 需实测 | P0 后先用 `weather_photos` 跑基线，记录"嵌入前后准确率差 ≤ 1%"作为门槛 |
| 内存/显存 | `copy.deepcopy(state_dict)` 在 ResNet50 上约 100 MB 且攻击循环内多次 | 改为一次快照 + `restore` 复用；微调用小学习率少量 batch；提供 `sample_limit` |
| 启动耗时 | 权重建构/下载 | 懒加载 + 本地权重缓存目录 + 无网络时给出明确错误 |
| 评测可信度 | 之前存在假数据回退 | 任何指标必须能追溯到 `dataset_dir + classes + split + seed`，写入报告 |

---

## 7. 验收指标（重构完成的判定）

1. `wmsys/core` 与 `wmsys/services` 中不出现 `torch`、`PyQt5`（静态检查）。
2. 启动到可交互 < 1.5 s（不含权重建构）；任何长任务期间 UI 无 > 100 ms 卡顿，取消 1 s 内生效。
3. 所有用户可见错误都有 `code + i18n key`，不再靠字符串匹配；`zh/en` 文案键集合一致。
4. 单元+集成测试覆盖 `core`/`services` ≥ 70%，其中含：密钥确定性、层定位与容量、checkpoint v1.0 往返、报告复现性。
5. 同种子下两次评测报告逐字段一致。
6. 新增算法/攻击的改动量为"1 文件 + 1 注册行"。

---

## 8. 老 → 新 映射速查

| 现状 | 目标 |
| --- | --- |
| `gui.WatermarkApp.init_ui/create_operation_buttons` | `ui/main_window.py` + `ui/panels/*`（每面板独立类，实现 `retranslate`） |
| `gui.HelpDialog/ProgressDialog` | `ui/dialogs/*`，`ProgressDialog` 改为信号驱动的被动视图 |
| `gui.run_performance_tests` | `services/evaluation_service.py` + `ui/workers.py` |
| `gui.get_test_loader/load_test_dataset/load_custom_dataset` | `data/imagefolder.py` + `data/transforms.py` + `data/split.py` |
| `WatermarkApp.tr` + `utils.LANGUAGES` | `ui/i18n/i18n.py` + `resources/strings/{zh,en}.json` |
| `ModelWatermarkSystem.initialize_model/_create_model` | `torch_adapter/backbones.py` |
| `generate_watermark` | `core/crypto.py`（参数冻结）+ `WatermarkKey` |
| `embed/extract_watermark_whitebox` | `algorithms/whitebox_sign.py` + `torch_adapter/layer_locator.py` |
| `save_model/load_model` | `torch_adapter/checkpoint_io.py`（含版本迁移） |
| `evaluate_model/_quick_evaluate` | `torch_adapter/evaluator.py` |
| `robustness_test/test_watermark_extraction` | `attacks/*` + `services/evaluation_service.py`（BER 度量） |
| `logi.txt`、`resources/el_wm.txt` | 删除，迁移为 `docs/legacy-notes.md` + `tests/fixtures/` |

---

## 9. 建议的第一步（可立即执行）

1. 先做 P0 的 8 项止血补丁（半天内可见效果：不崩、不假指标、准确率可信）。
2. 同时冻结"密钥派生参数 + legacy 层定位器"这两个兼容点，作为后续重构的不变式。
3. 再按 P1→P4 推进；每一阶段结束跑一次 `weather_photos` 全流程回归，记录基线报告作为对比。
