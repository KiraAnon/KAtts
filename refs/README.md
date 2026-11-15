# 参考音色（需自行准备）

TTS 需要一段参考音频来决定“音色”。本项目用的是 GPT-SoVITS 的提取格式：
每个 `.npz` 里包含该参考音频的语义 code、BERT 特征和 speaker embedding。

## 目录约定

```
refs/
├── zh_warm.npz
├── zh_joy.npz
└── ja_sad_2.npz
```

文件名就是音色名，请求时用 `"ref": "zh_warm"` 指定。
**未知的音色名不会报错，会静默回落到默认音色。**
启动时默认用排序后的第一个，也可以用环境变量 `KATTS_DEFAULT_REF` 指定。

## npz 里需要哪些 key

| key | shape | 说明 |
| --- | --- | --- |
| `p_phones` | [L] | 参考音频的音素序列 |
| `p_bert_features` | [L, 768] | 对应的 BERT 特征 |
| `p_prompt_semantic` | [T] | 参考音频的 semantic codes |
| `ge` | [1024] | speaker embedding |

## ⚠️ 版权提醒

**参考音频决定音色，音色本身可能涉及版权和人格权。**

- 用自己的声音，或明确授权的素材
- 不要把别人的声优音频直接放进公开仓库
- 本目录里的 52 个音色是 **AI 合成的**（非真人录音），随本仓库以 MIT 分发

## 怎么生成

1. 准备 3~10 秒、干净无噪、情绪明确的音频（16 kHz 以上）
2. 用 GPT-SoVITS 的 `get_spepc.py` / `extract_features.py` 提取 ge 与 semantic
3. 按上表的 key 存成 npz
