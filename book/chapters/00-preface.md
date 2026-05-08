# 00 前言

## 本章目标

本章说明为什么要围绕 `nanoGPT` 写一本书、这本书适合谁、应该如何阅读，以及读者在开始前需要准备什么。

## 0.1 为什么选择 nanoGPT

学习 GPT 有三种常见路径。

第一种是读论文。论文能提供理论框架，但不会告诉你训练脚本如何组织、checkpoint 保存什么、数据如何喂给模型，也不会暴露很多工程细节。

第二种是读大型训练框架。大型框架更接近工业实践，但抽象层多，配置复杂，初学者很容易在分布式、数据平台和工程封装里迷路。

第三种是读一个小而完整的项目。`nanoGPT` 正好处在这个位置。它的文件数量少，但包含训练 GPT 所需的核心链路：

```text
文本数据
-> tokenization
-> 二进制训练文件
-> batch 构造
-> GPT 模型
-> loss
-> optimizer
-> checkpoint
-> sampling
```

这使它非常适合写成一本源码导读书。

## 0.2 本书的核心问题

本书围绕一个问题展开：

```text
一个最小但完整的 GPT 训练项目是怎样工作的？
```

所有章节都服务于这个问题。数学公式、PyTorch API、训练技巧和性能优化都不会孤立出现，而是落回到 nanoGPT 的具体代码。

## 0.3 适合读者

本书适合：

- 会 Python，正在学习 PyTorch 的读者。
- 看过 Transformer 或 GPT 介绍，但没有读过完整训练代码的读者。
- 想用 nanoGPT 训练小模型、微调模型或做实验的工程师。
- 想把语言模型从“概念”落实到“代码路径”的学习者。

本书不假设读者已经能从零实现 Transformer，但假设读者愿意跟着源码追踪变量和 shape。

## 0.4 不适合的期待

本书不是现代大模型训练平台全景图。它不会覆盖数据治理、RLHF、大规模推理服务、模型压缩、在线评估和安全对齐等完整工业系统问题。

本书也不是数学教材。我们会解释必要公式，但目标是帮助读者读懂代码，而不是从理论上推导 Transformer。

## 0.5 推荐学习方式

第一遍阅读：

```text
跑通 Shakespeare 字符级训练
-> 看懂 model.py 的 forward
-> 看懂 train.py 的 get_batch 和训练循环
-> 用 sample.py 生成文本
```

第二遍阅读：

```text
追踪 attention shape
-> 调整训练参数
-> 比较 loss 和采样输出
-> 写自己的数据 prepare.py
```

第三遍阅读：

```text
修改模型结构
-> 增加评估脚本
-> 设计成体系的实验
```

不要第一天就试图复现 GPT-2。先用小实验建立直觉。

## 0.6 环境准备

基础依赖：

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

最小验证：

```sh
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=1 --eval_iters=1 --batch_size=2 --block_size=8 --n_layer=1 --n_head=2 --n_embd=16
```

这个命令不会训练出好模型，但能验证数据、模型和训练循环能跑通。

## 0.7 阅读源码时的习惯

阅读深度学习源码时，最重要的习惯是追踪 shape。看到每个张量，都问：

```text
它的 shape 是什么？
这个维度代表什么？
这一步有没有混合不同 token 的信息？
这一步有没有改变通道维度？
训练和推理是否走同一条路径？
```

本书会反复使用 `B x T x C`：

- `B`：batch size。
- `T`：sequence length。
- `C`：embedding dimension。

## 本章小结

`nanoGPT` 的价值在于完整但不庞大。本书会用它作为主线，把 GPT 的模型结构、数据处理、训练循环、采样和性能优化连起来。读者应该带着“代码如何工作”的问题阅读，而不是只记概念。

