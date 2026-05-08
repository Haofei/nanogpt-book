# 00 前言

## 本章目标

说明为什么用 `nanoGPT` 写一本书，以及这本书应该怎样读。

## 为什么选择 nanoGPT

`nanoGPT` 的价值在于它足够小，但又没有把关键工程细节藏起来。读者能在一个仓库里看到完整链路：

- 文本如何变成 token id。
- token id 如何组成 batch。
- GPT 模型如何做前向计算。
- loss 如何驱动反向传播。
- optimizer 如何更新参数。
- checkpoint 如何保存和恢复。
- 采样时模型如何一个 token 一个 token 地生成文本。

这比只读论文更接近工程实践，也比直接阅读大型训练框架更容易建立全局理解。

## 本书不是什么

本书不是 Transformer 论文的逐段注释，也不是 PyTorch API 百科。它的核心线索是源码。所有数学、框架和训练知识都围绕一个问题展开：这段代码为什么能训练一个 GPT？

## 学习环境

最低要求：

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

如果只想阅读源码，不需要准备 GPU。如果想跑第一个实验，可以先使用 CPU 或 Apple Silicon 的 `mps`，再根据条件切换到 CUDA。

## 推荐节奏

第一遍阅读时，不要急着改代码。先运行最小 Shakespeare 字符级实验，获得对输入、loss 和输出的直觉。第二遍再从 `model.py` 和 `train.py` 开始逐层追踪。

