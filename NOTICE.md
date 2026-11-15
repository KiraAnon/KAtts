# 第三方组件与许可

| 组件 | 来源 | 许可 | 用途 |
| --- | --- | --- | --- |
| GPT-SoVITS | https://github.com/RVC-Boss/GPT-SoVITS | MIT | 模型结构、文本前端（`text/`） |
| MNN | https://github.com/alibaba/MNN | Apache-2.0 | 声码器推理（`bin/libMNN.so`） |
| ONNX Runtime | https://github.com/microsoft/onnxruntime | MIT | 主推理引擎 |
| pypinyin / jieba / cn2an | PyPI | MIT/Apache | 中文文本处理 |
| WeTextProcessing / PaddleSpeech | https://github.com/wenet-e2e/WeTextProcessing | Apache-2.0 | `text/zh_normalization/` 文本正则化 |

## 关于模型权重

本仓库**不分发**模型权重（走 Release）；
`refs/` 里的 52 个参考音色是 AI 合成的，随本仓库以 MIT 分发。

用户自行准备的权重，其许可（包括是否允许商用）由权重来源决定，
与本仓库的 MIT 许可无关。
