# 实现细节

## 为什么不用 PyTorch

手机上装 PyTorch 要 1 GB 以上，还要为 aarch64 编译。
所以整个推理重新用 **numpy 手写**，模型转成 ONNX 交给 onnxruntime（< 10 MB）。

## s1 自回归怎么实现

GPT 类模型的核心是「每次吐一个 token，然后把它拼回输入」。
没有 torch 之后，这套循环要自己写：

```python
# 第一步：整段上下文一次性跑完，拿到 KV cache
out, k, v = prefill.run(None, {"x": xy, "attn_mask": mask})

# 之后：每步只喂一个新 token，KV 用上一步的
for _ in range(max_steps):
    out, k, v = step.run(None, {"x": ye, "k_cache": k, "v_cache": v})
    code = sample(out)          # top_k 采样
    ye = embed(code)            # 拼回输入
```

KV cache 在 ONNX 里是普通张量，只要在导出时把它声明成输入/输出即可。

## codes 为什么会坍缩（以及怎么防）

自回归模型偶尔会进入「复读」状态：输出的 semantic code 几乎不变，
解码出来就是一段平坦的嗡鸣，整句没起伏。

**检测**（`tts_mobile.py::_degenerate`）：

```python
不同取值 < 30%     → 判定坍缩
最大连续重复 > 50% → 判定坍缩
```

**处理**：

1. 换随机种子重生成（最多 3 次）
2. 三次都坍缩 → 切掉尾部坍缩段

句尾特别容易塌缩，所以还有一个 `_tail_bad()` 只看最后 1/4。

## 分段与流式

长句先切段（`plan_segments`），每段独立合成、独立解码，合成完就吐出去。

- ≤ 24 字：整句一次合成（太短的句子切开会让语气断掉）
- 超过：在标点处切，段尾补静音（长度按标点类型变化）

## 流水线并行

串行时 s1s2 和声码器解码没法重叠，实测 6.19 s 音频要 6.03 s。

改成生产者-消费者后：

```python
# 生产者线程：算下一段的 s1s2
def producer():
    for seg in segs:
        pipe.put(engine.synth(seg))

# 主线程：取出来解码、发送（保证顺序）
while item := pipe.get():
    dec_stream(item.z) → emit()
```

实测 6.03 s → 4.78 s（**快 21%**），首包时间不变，输出音频逐字节一致。

## 声码器的 CPU/NPU 分段

`tools/dec_pipeline.cpp` 里 11 段交替：

```
s0 CPU  s1 NPU  s2 CPU  s3 NPU  s4 CPU  s5 NPU
s6 CPU  s7 NPU  s8 CPU  s9 NPU  s10 CPU
```

规律是：**计算密集的 resblock 放 NPU，上采样（ConvTranspose）留 CPU**。

实测对比（同一份模型，只改后端）：

| 段 | CPU | NPU | 加速 |
| --- | --- | --- | --- |
| s1 | 226.0 ms | 50.7 ms | 4.5× |
| s3 | 460.8 ms | 94.3 ms | 4.9× |
| s5 | 265.0 ms | 29.4 ms | 9.0× |
| s7 | 131.7 ms | 22.0 ms | 6.0× |
| s9 | 81.7 ms | 15.5 ms | 5.3× |
| **合计** | **1220.6 ms** | **260.7 ms** | **4.68×** |

## 踩过的坑

1. **解码器是严格时序协议**：不能在 `for chunk in dec_stream()` 里做额外缓冲，
   多缓冲一步管道就错位（报 "could not convert string to float"）
2. **语速不能重采样**：会连带改变音高。改法是在 code 表上拉伸（`stretch_codes`）
3. **onnxruntime 内存池**：只跑一次的模型（BERT）要关掉 `enable_cpu_mem_arena`，
   否则一上来就预留一大块
