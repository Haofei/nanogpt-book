# 00b PyTorch 和张量最小基础

## 本章目标

本章只讲阅读 nanoGPT 必须知道的 PyTorch 基础。目标不是系统学习 PyTorch，而是让你看到 `model.py`、`train.py` 时不至于被最基本的概念挡住。

读完后，你应该理解：

- Tensor 是什么。
- Shape 为什么重要。
- `dtype` 和 `device` 是什么。
- `nn.Module`、`forward`、`Parameter` 是什么。
- `Embedding`、`Linear`、`LayerNorm` 大致做什么。
- 为什么训练需要 forward、loss、backward、optimizer step。

## 00b.1 Tensor 是什么

Tensor 可以先理解为多维数组。

一个数字是 0 维：

```text
3.14
```

一串数字是 1 维：

```text
[1, 2, 3]
```

表格是 2 维：

```text
[
  [1, 2, 3],
  [4, 5, 6]
]
```

语言模型里经常会看到 3 维张量：

```text
B x T x C
```

其中：

- `B`：batch size，一次处理多少条序列。
- `T`：sequence length，每条序列有多少 token。
- `C`：channel 或 embedding dimension，每个 token 用多少维向量表示。

## 00b.2 Shape 是读源码的地图

阅读 nanoGPT 时，最重要的习惯是追踪 shape。

例如输入 token id：

```text
idx: B x T
```

经过 embedding：

```text
tok_emb: B x T x C
```

经过模型输出：

```text
logits: B x T x vocab_size
```

如果你能说清楚每一步 shape 为什么这样变，源码就已经读懂了一半。

## 00b.3 dtype 是什么

`dtype` 表示张量里的数字类型。

常见类型：

- `torch.long`：整数，常用于 token id。
- `torch.float32`：32 位浮点数，稳定但占显存。
- `torch.float16`：16 位浮点数，更省显存但数值范围小。
- `torch.bfloat16`：另一种 16 位浮点数，深度学习训练中常见。

Embedding 的输入必须是整数 token id，所以 `idx` 通常是 `torch.long`。模型内部的权重和激活通常是浮点数。

## 00b.4 device 是什么

`device` 表示张量在哪个计算设备上：

```text
cpu
cuda
mps
```

一个常见错误是：模型在 GPU 上，输入张量在 CPU 上。PyTorch 会报 device mismatch。nanoGPT 里会把 batch 移到配置指定的 `device`。

你可以把 device 理解为“这块数据现在放在哪里算”。

## 00b.5 nn.Module 是什么

PyTorch 里，模型通常继承 `nn.Module`。

在 nanoGPT 中：

```python
class GPT(nn.Module):
    ...
```

`nn.Module` 的作用是帮你管理：

- 子模块。
- 可训练参数。
- `train()` / `eval()` 模式。
- `state_dict()` 保存和加载。

当你调用：

```python
logits, loss = model(X, Y)
```

PyTorch 实际会调用模型的 `forward` 方法。

## 00b.6 Parameter 是什么

`Parameter` 是模型会学习的张量。

例如 Linear 层里的权重就是 Parameter。训练时，反向传播会计算这些参数的梯度，optimizer 会更新它们。

不是所有张量都是 Parameter。比如 causal mask 是固定规则，不需要学习，所以 nanoGPT 用 `register_buffer` 保存它。

## 00b.7 Embedding 是什么

Embedding 是查表。

假设词表大小是 65，每个 token 用 128 维向量表示，那么 embedding 表是：

```text
65 x 128
```

输入 token id：

```text
B x T
```

输出向量：

```text
B x T x 128
```

在 nanoGPT 里：

```python
wte = nn.Embedding(config.vocab_size, config.n_embd)
```

`wte` 是 token embedding，`wpe` 是 position embedding。

## 00b.8 Linear 是什么

Linear 层做线性变换：

```text
输入维度 -> 输出维度
```

例如：

```python
nn.Linear(128, 512)
```

表示把最后一维从 128 变成 512。

如果输入是：

```text
B x T x 128
```

输出是：

```text
B x T x 512
```

Linear 不会改变 `B` 和 `T`，只改变最后一维。

## 00b.9 forward、loss、backward、step

训练神经网络通常分四步：

```text
forward：模型根据输入算输出
loss：比较输出和正确答案
backward：计算每个参数应该怎么改
step：optimizer 更新参数
```

nanoGPT 训练循环里也遵循这个结构，只是加入了混合精度、梯度累积、DDP 和 checkpoint。

入门阶段先记住这条主线：

```text
X, Y -> model(X, Y) -> loss -> loss.backward() -> optimizer.step()
```

## 00b.10 train 模式和 eval 模式

PyTorch 模型有训练模式和评估模式。

训练模式：

```python
model.train()
```

评估模式：

```python
model.eval()
```

区别之一是 dropout。训练时 dropout 会随机丢弃部分激活；评估时不会。nanoGPT 在估计 loss 时会切到 `eval()`，评估结束再切回 `train()`。

## 00b.11 no_grad 是什么

评估或生成时不需要训练参数，所以不需要保存反向传播图。

PyTorch 写法：

```python
with torch.no_grad():
    ...
```

这样可以节省显存和计算。`sample.py` 生成文本时就会使用它。

## 00b.12 本章检查点

继续阅读源码前，确认你能回答：

```text
Tensor 和 shape 是什么？
B x T x C 分别代表什么？
token id 为什么是整数？
Embedding 为什么会把 B x T 变成 B x T x C？
Linear 通常改变哪个维度？
forward、loss、backward、optimizer step 分别做什么？
model.train() 和 model.eval() 有什么区别？
```

如果这些问题基本清楚，就可以第一次完整跑通 nanoGPT。

