# 02 Transformer 必备知识

## 本章目标

本章补齐阅读 `model.py` 前必须掌握的 Transformer 概念。读者不需要先完整推导论文，但需要理解 token embedding、position embedding、self-attention、causal mask、LayerNorm、残差连接和 MLP。

对应源码位置：`code/nanogpt/model.py`。

## 2.1 语言模型在做什么

GPT 类语言模型的训练目标是预测下一个 token。

给定序列：

```text
x0, x1, x2, x3
```

训练目标是：

```text
用 x0 预测 x1
用 x0, x1 预测 x2
用 x0, x1, x2 预测 x3
```

在 nanoGPT 中，`get_batch` 构造的 `x` 和 `y` 正好相差一个位置。模型看到 `x`，目标是预测 `y`。

## 2.2 Token 与 Token ID

模型不能直接处理字符串。文本要先被 tokenizer 编码成 token id。

字符级例子：

```text
"abc" -> [0, 1, 2]
```

BPE 例子可能是：

```text
"hello world" -> [31373, 995]
```

模型看到的是整数，不是文字。tokenizer 决定了整数和文本片段之间的关系。

## 2.3 Token Embedding

Embedding 是查表。假设词表大小是 `vocab_size`，embedding 维度是 `n_embd`，则 token embedding 表 shape 是：

```text
vocab_size x n_embd
```

输入 `idx` 的 shape：

```text
B x T
```

经过 token embedding：

```text
B x T x C
```

其中 `C = n_embd`。

在源码中：

```python
tok_emb = self.transformer.wte(idx)
```

## 2.4 Position Embedding

Transformer 本身没有循环结构。如果只给 token embedding，模型不知道 token 的顺序。所以 GPT 需要位置信息。

nanoGPT 使用可学习的绝对位置 embedding：

```python
pos = torch.arange(0, t, dtype=torch.long, device=device)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

`pos_emb` 的 shape 是：

```text
T x C
```

加到 `tok_emb` 上后，模型输入仍然是：

```text
B x T x C
```

## 2.5 Self-Attention 的直觉

Self-attention 让序列中的每个位置读取其他位置的信息。

对 GPT 来说，第 `t` 个位置只能读取自己和左侧历史位置，不能读取右侧未来位置。

可以把 attention 理解成每个 token 都在问：

```text
我应该从前面的哪些 token 中取信息？
每个 token 应该取多少？
取到的信息如何汇总？
```

## 2.6 Q、K、V

Attention 使用 Query、Key、Value。

直觉解释：

- Query：当前位置想找什么。
- Key：每个位置提供什么索引。
- Value：每个位置携带什么内容。

源码中：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

同一个输入 `x` 经过线性层，一次性生成 Q、K、V。

## 2.7 Scaled Dot-Product Attention

核心公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

`QK^T` 计算匹配分数。除以 `sqrt(d_k)` 是为了控制数值尺度。softmax 把分数变成权重。最后乘以 `V` 得到上下文表示。

在 nanoGPT 手写路径中：

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = F.softmax(att, dim=-1)
y = att @ v
```

## 2.8 Multi-Head Attention

多头注意力把通道维度拆成多个 head。每个 head 可以学习不同类型的关系。

如果：

```text
n_embd = 384
n_head = 6
```

则：

```text
head_size = 64
```

shape 从：

```text
B x T x C
```

变成：

```text
B x n_head x T x head_size
```

每个 head 独立计算 attention，最后再拼回 `B x T x C`。

## 2.9 Causal Mask

GPT 是自回归模型。训练时不能让当前位置看到未来 token，否则模型会作弊。

下三角 mask：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

`1` 表示可见，`0` 表示不可见。

nanoGPT 中有两种实现：

- PyTorch 2.0 的 `scaled_dot_product_attention(..., is_causal=True)`。
- 手写路径中的下三角 `bias` buffer。

## 2.10 LayerNorm 和残差连接

Transformer Block 中：

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

LayerNorm 让每个 token 的隐藏向量分布更稳定。残差连接让每层学习一个增量，而不是完全重写表示。

这对深层模型训练很重要。没有残差连接和归一化，深层 Transformer 很难稳定训练。

## 2.11 MLP

Attention 混合不同位置的信息。MLP 则在每个位置内部做非线性变换。

nanoGPT 的 MLP：

```text
C -> 4C -> GELU -> C
```

它不改变序列长度，也不混合不同 token 位置。输入输出都是：

```text
B x T x C
```

## 2.12 一个 Block 的完整流程

一个 Block 可以理解成：

```text
输入 x
-> LayerNorm
-> Causal Self-Attention
-> 残差相加
-> LayerNorm
-> MLP
-> 残差相加
-> 输出 x
```

由于输入输出 shape 相同，GPT 可以把多个 Block 直接堆叠。

## 2.13 阅读源码时最重要的 shape

请记住：

```text
idx: B x T
token embedding: B x T x C
q/k/v: B x n_head x T x head_size
attention score: B x n_head x T x T
block output: B x T x C
logits: B x T x vocab_size
```

后面读 `model.py` 时，大部分困惑都可以通过追踪这些 shape 解决。

## 本章小结

Transformer 的核心不是神秘公式，而是一组可追踪的张量变换。token embedding 把 id 变成向量，position embedding 加入顺序信息，causal self-attention 读取历史上下文，MLP 增强每个位置的表达，残差和 LayerNorm 保持训练稳定。
