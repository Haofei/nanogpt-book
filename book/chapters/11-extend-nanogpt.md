# 11 如何扩展 nanoGPT

## 本章目标

讨论如何在不破坏项目简洁性的前提下扩展 nanoGPT，包括新数据集、新 tokenizer、新模型结构、新日志和新评估。

## 增加新数据集

推荐复制现有目录结构：

```text
data/my_dataset/
  prepare.py
  readme.md
```

`prepare.py` 的职责应该保持单一：下载或读取原始文本，切分 train/val，编码成 token id，写出 `train.bin`、`val.bin`，必要时写出 `meta.pkl`。

## 增加新配置

新增配置文件时，放在 `config/` 下，例如：

```text
config/train_my_dataset.py
```

配置文件只覆盖和该实验相关的变量，不要复制 `train.py` 的全部默认配置。这样更容易看出实验真正改变了什么。

## 改模型结构

常见扩展方向：

- 替换位置编码，例如 RoPE 或 ALiBi。
- 修改 MLP 激活函数或宽度。
- 增加 RMSNorm。
- 调整 attention 实现。
- 加入 dropout 或正则化策略。

改模型时要同时检查 checkpoint 兼容性。只要参数名或 shape 变化，旧 checkpoint 就可能无法直接加载。

## 增加评估

当前训练主要依赖 train/val loss。可以增加：

- 固定 prompt 的生成样例。
- 困惑度评估。
- 下游任务零样本评估。
- 训练速度和显存记录。

评估代码可以先独立成脚本，稳定后再决定是否接入训练循环。

## 增加日志

`wandb_log` 已经提供基础接入。扩展日志时建议记录：

- train loss 和 val loss。
- learning rate。
- tokens per second。
- gradient norm。
- checkpoint 路径。
- 采样样例。

日志越多不一定越好。关键是能解释实验结果。

## 保持项目可读性

扩展 nanoGPT 时最容易犯的错误是把它改成复杂框架。建议遵守：

- 新功能先用独立脚本验证。
- 配置变化放到 `config/`。
- 数据处理放到 `data/`。
- 不把所有实验开关都塞进 `train.py`。
- 修改核心模型前先写清楚 checkpoint 兼容性影响。

## 小结

好的扩展不是堆功能，而是让每个新能力都有清晰边界。nanoGPT 的价值正在于小，扩展时要保护这种可读性。

