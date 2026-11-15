# KAtts

> 在 Android 手机上实时运行 GPT-SoVITS v2ProPlus 的完整语音合成链路。
> 纯 Python + ONNX Runtime（**不含 PyTorch**），声码器走 C++ + MNN，可选 NPU 加速。

**English**: A mobile port of GPT-SoVITS that runs entirely on an Android phone CPU/NPU.
No PyTorch, no cloud. ~899 MB of models (~768 MB ONNX + 129 MB vocoder segments), first-chunk latency ~1.5 s,
~5x faster vocoder on MTK MDLA NPU vs CPU.

---

## 实测性能

设备：Unihertz 8849 Tank 4 Pro（天玑 8300 / MT6897，Android 15，已 root）

| 指标 | 数值 |
| --- | --- |
| 首包延迟（会话热） | **1.3 ~ 1.8 s** |
| 首包延迟（冷启动） | ~8 s |
| 6.19 s 音频总耗时 | **4.78 s**（流水线并行后） |
| 声码器（dec）NPU vs CPU | **4.68×**（关键层 5.5×） |
| 模型总体积 | **~899 MB**（int8 + 声码器分段 + 音色参考） |
| 常驻内存 | ~1.2 GB |

## 特性

- **完全离线**：文本到波形全程本地，不联网、不调 API
- **不需要 PyTorch**：整个自回归推理用 numpy 手写，模型走 onnxruntime
- **流式输出**：边合成边吐 PCM，长句不等整段
- **流水线并行**：`s1+s2` 与声码器解码重叠执行（实测省 21%）
- **三轴情感控制**：`a`/`s`/`g` 分别控制音色融合强度、语速、气口，**不改音高**
- **NPU 加速**：声码器 11 段交替 CPU/NPU，关键层快 5 倍
- **codes 坍缩检测**：自动识别自回归退化并换种子重试，避免整句嗡鸣

## 架构

```
文本
 ↓ text/ 文本前端（清洗、数字转写、中英混排切分）
音素序列 + word2ph 对齐
 ↓ bert_int8.onnx（287 MB）
语义特征
 ↓ s1 自回归（prefill 一次 + step 循环，73 MB）
semantic codes
 ↓ s2_pre.onnx（111 MB）
latent z [1, 192, T]
 ↓ bin/dec_server（C++ + MNN，11 段交替 CPU/NPU）
32 kHz PCM（流式输出）
```

各阶段耗时（串行基线 6.03 s，处理 6.19 s 音频；流水线后总时间降到 4.78 s）：

```
s1 + s2     ≈ 3.4 s   (56%)
声码器       ≈ 2.6 s   (43%)   → NPU 加速后
```

## 快速开始

### 1. 环境

```bash
# Termux（Android）
pkg install python clang
pip install -r requirements.txt
# 中文默认就够用；多语言依赖在 requirements 里是注释状态，需要时取消
```

Python 3.10+，仅支持 **ARM64 Linux/Android**（`bin/dec_server` 是 aarch64 二进制）。

### 2. 模型

模型约 899 MB，超过 GitHub 单文件限制，走 **Release** 分发（两个压缩包）：

```bash
# 一条命令拉齐（会自动解压到 models/ 和 segs/）
export KATTS_RELEASE_URL=<本仓库 Releases 里 v0.1.0 的下载前缀>
bash scripts/get_models.sh
```

| 包 | 解开后 | 内容 |
| --- | --- | --- |
| `models.tar.gz` | `models/` | 4 个 ONNX + 1 个外部权重 + 1 个 npz |
| `segs.tar.gz` | `segs/` | 声码器 11 段 .mnn |

参考音色（`refs/`）**已经随仓库自带**，不用下载。

**必须先准备模型，否则启动会直接报错并提示缺哪个文件。**

### 3. 运行

```bash
python app.py
# [app] loading engine...
# [app] engine ready
# [app] warming up...
# [app] listening http://127.0.0.1:8080
```

打开 `http://127.0.0.1:8080` 就是测试页面，输入文字直接播放。

### 4. 调用 API

```bash
# 纯文本
curl -X POST http://127.0.0.1:8080 -d "你好，今天天气不错" --output out.pcm

# 带参数
curl -X POST http://127.0.0.1:8080 \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","ref":"warm","s":1.05,"a":3,"g":1.1}' \
  --output out.pcm
```

返回 **32 kHz / 16-bit / 单声道 PCM**，`Transfer-Encoding: chunked` 流式传输。

| 参数 | 含义 | 默认 |
| --- | --- | --- |
| `text` | 要合成的文本 | 必填 |
| `ref` | 参考音色名（对应 `refs/<name>.npz`） | 第一个 |
| `s` | 语速倍率 | 1.0 |
| `a` | 音色融合强度 0~7（0 = 纯锚点音色） | 0.0 |
| `g` | 气口（停顿）倍率 | 1.0 |

转成 wav：

```bash
ffmpeg -f s16le -ar 32000 -ac 1 -i out.pcm out.wav
```

## 三轴情感控制

不用换模型、不改音高，靠三个独立轴控制语气：

| 轴 | 参数 | 范围 | 作用 |
| --- | --- | --- | --- |
| **音色融合** | `a` | 0 ~ 7 | 锚点固定 1 票、目标参考 `a` 票，按票数平均 ge。**a=0 = 纯锚点音色**（锚点默认等于默认音色，可用 `KATTS_ANCHOR_REF` 改） |
| **语速** | `s` | 0.8 ~ 1.2 | 改的是 **code 表长度**，不是波形重采样，所以音高不变 |
| **气口** | `g` | 0.5 ~ 1.5 | 段间停顿时长；句末标点不同，基线不同 |

三个轴互相独立，可以反向组合（比如"语速快 + 气口大"）。

仓库自带 **52 个参考音色**，按情绪 × 语言分成两组：

```
refs/
├── zh_normal.npz     完整参考  中文发音（音素 + BERT + semantic + 音色向量）  50~100 KB
├── zh_joy.npz        同上
├── ja_normal.npz     同样 4 个 key，只是参考音频更短                          4 KB
└── ja_joy.npz        同上
```

13 种情绪 × 2 语言（每种还有个 `_2` 变体）：
normal / joy / sad / excited / warm / shy / surprised /
concern / firm / guilty / longing / nervous / questioning

请求时用 `"ref": "zh_joy"` 选音色，`"a": 3` 控制往该情绪的默认语气融合多少。

> **这些参考音色是 AI 生成的（日语女声合成，再转成中文发音），不是真人录音**，
> 所以可以随仓库分发。如果你要用自己的声音，见 `refs/README.md`。

## 环境变量

| 变量 | 作用 |
| --- | --- |
| `KATTS_DEFAULT_REF` | 默认音色名（缺省按 zh_normal → zh_warm → default → ja_warm 挑） |
| `KATTS_ANCHOR_REF` | a 轴的锚点音色（a=0 时用的就是它，缺省等于默认音色） |
| `KATTS_REF_NPZ` | 额外的单个音色文件，作为名为 `default` 的音色加入；只有 refs/ 下没有 default.npz 时才生效 |
| `KATTS_RELEASE_URL` | 模型 Release 地址（`scripts/get_models.sh` 用） |
| `KATTS_DUMP_AUDIO` | 设为 1 时把合成音频和文本存到 `tts_dumps/`（调试，默认关） |

## 已知限制

- **仅 aarch64**：`dec_server` 是为 Android arm64 编的
- **模型要自己准备**：本项目只分发代码，不分发权重
- **语音前端是 GPT-SoVITS 的**：直接复用上游 `text/`，多语言依赖较重
- **声码器分段固定**：`segs/` 的 11 段是按 T=64 切分的，改分辨率/长度要重新切
- **NPU 段需要 MNN 编译时开启 NNAPI**：官方 `libMNN.so` 默认带，自己编要注意

## 目录结构

```
KAtts/
├── app.py              HTTP 服务 + 合成流程（分句、流式、流水线并行）
├── tts_mobile.py       推理引擎（s1 自回归 / s2 / BERT）
├── bert_tok.py         BERT tokenizer
├── bert_vocab.json     词表
├── text/               GPT-SoVITS 文本前端
├── bin/
│   ├── dec_server      声码器服务（aarch64）
│   └── libMNN.so       MNN 运行时
├── tools/
│   ├── dec_server.cpp   声码器服务版源码（`--serve` 常驻模式）
│   └── dec_pipeline.cpp 声码器一次性版本
├── models/             模型（Release 下载，见 scripts/get_models.sh）
├── segs/               声码器分段（同上）
└── refs/               参考音色（仓库自带 52 个，AI 合成）
```

## 关于 `bin/dec_server`

声码器服务：把 latent z 解码成 32 kHz 波形。

- 推理框架是 **MNN**（阿里开源，Apache-2.0），不是 PyTorch
- 加载 `segs/` 下的 11 段 `.mnn` 模型
- `--serve` 模式：走 stdin/stdout 二进制协议，Python 每次喂一个 z，它回一批音频
- 11 段交替 CPU / NPU（NNAPI → MTK MDLA）

源码见 `tools/dec_server.cpp`（含 stdin/stdout 二进制协议与 11 段 SEGS 表）。
编译需要 Android NDK + MNN 头文件，见 `docs/BUILD.md`。

## 致谢与许可

- **GPT-SoVITS**（MIT）—— 模型结构与文本前端来自上游
- **MNN**（Apache-2.0）—— 声码器推理
- **ONNX Runtime**（MIT）

本仓库的移植代码（`app.py` / `tts_mobile.py` / `bert_tok.py` / `tools/`）为 MIT。
模型权重、参考音色**不在**本仓库，其许可请咨询各自来源。
