# 附录 C 术语与背景知识

## 本附录目标

本附录补充读者在阅读 nanoGPT 时容易默认缺失的背景知识。很多教程会直接使用这些词，但不解释它们在训练代码里具体对应什么。

## Step、Iteration 和 Epoch

在 nanoGPT 中，`iter_num` 基本可以理解为 optimizer 更新次数，也就是训练 step。

一次 iteration 通常包含：

```text
取 batch
-> forward
-> loss
-> backward
-> optimizer step
```

如果使用梯度累积，一个 optimizer step 会包含多个 micro step。

Epoch 是“完整看过一遍训练集”。nanoGPT 默认把数据看成一条很长的 token 流，并随机切片训练，所以它不强调 epoch，而更强调 iteration 和 token 数。

## Batch、Micro Batch 和 Effective Batch

`batch_size` 是一次 forward 中有多少条序列。每条序列长度是 `block_size`。

如果：

```text
batch_size = 12
block_size = 1024
```

一次 micro batch 包含：

```text
12 * 1024 = 12,288 tokens
```

如果 `gradient_accumulation_steps = 40`，一次 optimizer step 的有效 token 数是：

```text
12 * 1024 * 40 = 491,520 tokens
```

多卡时还要乘以 `world_size`。

## Loss 和 Perplexity

语言模型常用 cross entropy loss。直觉上，loss 衡量模型给真实下一个 token 的概率有多高。

如果模型对真实 token 很有信心，loss 低。如果模型把概率分散到很多错误 token 上，loss 高。

Perplexity 可以由 loss 转换：

```text
perplexity = exp(loss)
```

loss 越低，perplexity 越低。

## Logits、Probability 和 Sampling

模型输出的是 logits，不是概率。logits 是未归一化分数，可以是任意实数。

softmax 把 logits 转成概率：

```text
logits -> softmax -> probability distribution
```

训练时 `F.cross_entropy` 内部会处理 logits，不需要手动 softmax。生成时要显式 softmax，然后采样。

## Tokenizer

Tokenizer 决定文本如何变成 token id。

字符级 tokenizer：

```text
每个字符一个 token
```

BPE tokenizer：

```text
常见文本片段一个 token
```

模型的 embedding 和 `vocab_size` 与 tokenizer 强绑定。换 tokenizer 通常不能直接复用旧 checkpoint 的 embedding 和输出层。

## Checkpoint

checkpoint 是训练状态快照，不只是模型权重。

nanoGPT 的 checkpoint 包含：

```text
model
optimizer
model_args
iter_num
best_val_loss
config
```

如果只保存模型权重，可以做推理，但不能完整恢复训练现场。

## CUDA、MPS 和 CPU

`device=cpu` 使用 CPU。它最通用，但训练慢。

`device=cuda` 使用 NVIDIA GPU。多数深度学习训练优化都优先支持 CUDA。

`device=mps` 使用 Apple Silicon 的 Metal 后端。它能加速部分训练，但算子支持和性能特征与 CUDA 不完全一致。

## dtype、float16、bfloat16

`dtype` 决定计算中使用的数据类型。

`float32` 稳定但显存占用大、速度慢。

`float16` 更快更省显存，但数值范围小，训练时通常需要 `GradScaler`。

`bfloat16` 数值范围更接近 `float32`，在支持的硬件上通常更稳定。

## DDP

DDP 是 Distributed Data Parallel。它的基本方式是每张 GPU 一个进程，每个进程处理不同 batch，反向传播后同步梯度。

DDP 不是把一个模型拆到多张卡上，而是每张卡都有一份完整模型副本。

## Overfitting

过拟合是模型记住训练数据，但不能很好泛化到验证数据。

常见现象：

```text
train loss 继续下降
val loss 停止下降甚至上升
生成内容更像背诵
```

处理方法包括增加数据、减小模型、增加 dropout、提前停止训练。
