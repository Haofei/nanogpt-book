# 01 nanoGPT 是什么

## 本章目标

建立项目的整体地图：`nanoGPT` 解决什么问题，哪些文件负责哪些职责，以及学习时应该先看哪里。

## 项目定位

`nanoGPT` 是一个用于训练和微调 GPT 类语言模型的极简 PyTorch 项目。它不是教学玩具，因为它包含真实训练所需的配置、checkpoint、DDP、混合精度、GPT-2 权重加载和采样逻辑；它也不是完整工业系统，因为它没有复杂的数据平台、模型服务、监控系统和大规模容错能力。

它适合回答一个核心问题：一个 GPT 训练项目最小但完整的形态是什么？

## 核心文件

- `model.py`：定义 GPT 模型，包括 `LayerNorm`、`CausalSelfAttention`、`MLP`、`Block`、`GPTConfig` 和 `GPT`。
- `train.py`：训练入口，包括配置覆盖、DDP 初始化、batch 读取、loss 评估、优化器、checkpoint 和训练循环。
- `sample.py`：推理入口，包括加载 checkpoint 或 GPT-2 权重、编码 prompt、调用 `generate`。
- `config/`：保存不同训练任务的参数覆盖。
- `data/`：保存数据准备脚本和数据集说明。
- `bench.py`：用于隔离训练核心计算，方便做性能观察。

## 一条完整链路

训练时，数据从文本开始：

```text
raw text -> token ids -> train.bin/val.bin -> batch x/y -> GPT -> logits -> cross entropy loss -> backward -> AdamW
```

生成时，流程反过来看：

```text
prompt text -> token ids -> GPT -> next-token logits -> sampling -> append token -> repeat -> generated text
```

理解这两条链路，就有了阅读全仓库的主线。

## 第一处源码入口

建议先读 `README.md` 的 quick start，然后打开 `config/train_shakespeare_char.py`。这个配置比默认 OpenWebText 训练小得多，更适合建立直觉。接下来再读 `train.py` 的配置区和 `model.py` 的 `GPT.forward`。

## 小结

`nanoGPT` 的优势不是功能多，而是把 GPT 训练项目压缩到少数几个文件里。学习它时要抓住数据、模型、训练、采样四条主线。

