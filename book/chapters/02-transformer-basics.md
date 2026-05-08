# 02 Transformer 必备知识

## 本章目标

在阅读 `model.py` 前，补齐必须的 Transformer 概念：embedding、位置编码、自注意力、残差连接、LayerNorm、MLP 和因果 mask。

## Token 与 Embedding

语言模型不能直接处理字符串。文本会先被编码成整数序列，例如：

```text
"hello" -> [31373]
```

模型中的 `wte` 是 token embedding 表。给定 token id 后，它会查表得到向量。`wpe` 是位置 embedding 表，用来告诉模型每个 token 位于序列中的哪个位置。

在 `GPT.forward` 中：

```python
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

这里的核心思想是：每个位置的输入向量等于 token 含义和位置信息的相加。

## 自注意力

自注意力让每个 token 读取它左侧上下文的信息。`nanoGPT` 中一次线性层同时产生 Q、K、V：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

可以把三者理解为：

- Q：当前位置想查询什么。
- K：每个历史位置提供什么索引。
- V：每个历史位置真正提供的信息内容。

注意力分数来自 `q @ k.transpose(-2, -1)`。分数经过 softmax 后变成权重，再和 `v` 相乘得到上下文向量。

## 因果 Mask

GPT 是自回归模型。第 `t` 个位置只能看见 `0..t` 的 token，不能偷看未来。这个限制叫 causal attention。

在 PyTorch 2.0 路径中，`scaled_dot_product_attention(..., is_causal=True)` 负责这件事。旧路径中，代码显式注册下三角 mask：

```python
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))
```

## 残差与 LayerNorm

每个 Block 的前向逻辑是：

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

这是一种 pre-norm Transformer 结构。LayerNorm 先稳定输入分布，Attention 或 MLP 负责变换，残差连接保留原始路径，降低深层网络训练难度。

## MLP

`MLP` 是逐 token 的前馈网络：

```text
n_embd -> 4 * n_embd -> GELU -> n_embd
```

Attention 混合不同位置的信息，MLP 则在每个位置内部增加非线性表达能力。

## 小结

阅读 `model.py` 时，最重要的不是记住所有类名，而是看懂每个 Block 如何把 `B x T x C` 的张量变成同样形状的更有信息量的张量。

