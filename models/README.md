# 模型文件（需自行准备）

模型权重约 **899 MB**。因为超过 GitHub 单文件 100 MB 限制，走 Release 分发，
用 `scripts/get_models.sh` 一键下载。

## 需要的文件

| 文件 | 位置 | 体积 | 说明 |
| --- | --- | --- | --- |
| `bert_int8.onnx` | `models/` | 287 MB | 中文 BERT，文本 → 语义特征 |
| `s1_step_int8.onnx` | `models/` | 73 MB | 自回归单步（带 KV cache） |
| `s1_prefill_mask18.onnx` | `models/` | 0.8 MB | 自回归预填充（图结构） |
| `s1_prefill_mask18.onnx.data` | `models/` | **289 MB** | 上者的外部权重（大头在这） |
| `s2_pre.onnx` | `models/` | 111 MB | codes → latent z |
| `s1_params.npz` | `models/` | 7.4 MB | s1 的 embedding / 投影权重 |
| `s{0..10}_{cpu,apu}.mnn` | `segs/` | 共 129 MB | 声码器 11 段（6 个 CPU 段 + 5 个 NPU 段交替） |

## 接口规格

从 `tts_mobile.py` 反推出的输入输出名，导出 ONNX 时必须对上：

```
BERT:    输入  input_ids / attention_mask（见 bert_tok.py）
         输出  最后一层 hidden [1, T, 768]

prefill: 输入  x [1, L, 512], attn_mask
         输出  out, k, v

step:    输入  x [1, 1, 512], k_cache, v_cache
         输出  out, k, v

s2:      输入  codes [1, 1, T], text [1, L], ge [1, 1024]
         输出  z [1, 192, T]
```

## 怎么来

模型从 **GPT-SoVITS v2ProPlus** 的官方权重导出：

1. 下载 GPT-SoVITS，用它的 `api.py` 确认能正常合成
2. 用 `torch.onnx.export` 分别导出 Bert / AR(prefill + step) / s2
3. `s1_params.npz` 是 AR 里几个不走计算图的权重（embedding、投影矩阵）
4. 声码器把 GPT-SoVITS 的 `s2G` 按时间轴切成 11 段，各转一份 MNN 模型

> ⚠️ 完整的导出脚本还没随仓库提供。
> 按上表的接口规格，用你自己的 GPT-SoVITS 环境导出即可。

## 检查是否齐全

```bash
ls -lh models/*.onnx models/*.npz segs/*.mnn
```
