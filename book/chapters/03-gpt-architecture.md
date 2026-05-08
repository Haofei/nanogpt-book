# 03 GPT 架构总览

## 本章目标

把 GPT 模型从整体上拆开，为下一章逐段阅读 `model.py` 做准备。

## GPT 的输入输出

`nanoGPT` 的模型输入是 `idx`，形状为：

```text
B x T
```

其中 `B` 是 batch size，`T` 是上下文长度。每个元素是一个 token id。

模型输出是 `logits`，训练时形状通常为：

```text
B x T x vocab_size
```

每个位置都会预测下一个 token 的概率分布。训练目标是让第 `t` 个位置预测真实的第 `t+1` 个 token。

## 模型层次

`GPT` 包含：

- `wte`：token embedding。
- `wpe`：position embedding。
- `drop`：输入 dropout。
- `h`：由多个 `Block` 组成的层列表。
- `ln_f`：最后的 LayerNorm。
- `lm_head`：把隐藏向量投影到词表大小。

结构可以写成：

```text
token ids
  -> token embedding + position embedding
  -> Block x n_layer
  -> final LayerNorm
  -> lm_head
  -> logits
```

## GPTConfig

`GPTConfig` 是模型大小的集中定义：

- `block_size`：最大上下文长度。
- `vocab_size`：词表大小。
- `n_layer`：Transformer Block 层数。
- `n_head`：注意力头数。
- `n_embd`：隐藏维度。
- `dropout`：dropout 比例。
- `bias`：Linear 和 LayerNorm 是否使用 bias。

这几个参数共同决定模型容量、显存占用和训练速度。

## 权重共享

`nanoGPT` 使用 token embedding 与输出层权重共享：

```python
self.transformer.wte.weight = self.lm_head.weight
```

直觉上，输入 token 的向量空间和输出 token 的分类空间可以共享一套表示。这能减少参数，也常见于语言模型。

## 小结

GPT 是一个很规整的堆叠结构。理解它的关键是追踪 shape：`B x T` 进入模型，变成 `B x T x C`，最后变成 `B x T x vocab_size`。

