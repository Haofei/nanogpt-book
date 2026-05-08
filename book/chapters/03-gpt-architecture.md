# 03 GPT 架构总览

## 本章目标

本章从整体上理解 GPT 架构，为下一章精读 `model.py` 做准备。读者应该能画出 nanoGPT 的模型结构，并说明训练时每个位置如何预测下一个 token。

对应源码位置：`code/nanogpt/model.py` 的 `GPT` 类。

## 3.1 GPT 是 Decoder-only Transformer

GPT 属于 decoder-only Transformer。它只使用 Transformer 解码器风格的堆叠结构，并通过 causal mask 保证只能看左侧上下文。

和 encoder-decoder 模型不同，GPT 没有单独的编码器，也没有跨 attention。它的任务很直接：

```text
给定已有 token，预测下一个 token
```

这个简单目标可以扩展到预训练、微调和文本生成。

## 3.2 输入和输出

输入 `idx`：

```text
B x T
```

每个元素是 token id。

训练输出 `logits`：

```text
B x T x vocab_size
```

每个位置都有一个词表大小的分类分数。训练时，位置 `t` 的 logits 用来预测真实的下一个 token。

推理输出：

```text
B x 1 x vocab_size
```

生成时只需要最后位置的 logits，所以 nanoGPT 推理路径只对最后位置做输出投影。

## 3.3 模型主干

`GPT.__init__` 中的主干：

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

对应结构：

```text
token ids
-> token embedding
-> position embedding
-> dropout
-> Transformer Block x n_layer
-> final LayerNorm
-> lm_head
-> logits
```

## 3.4 Embedding 层

`wte` 把 token id 映射到向量。`wpe` 把位置 id 映射到向量。

二者相加：

```python
x = tok_emb + pos_emb
```

这表示每个位置的初始表示同时包含“这个 token 是什么”和“它在序列中的哪个位置”。

## 3.5 Transformer Block 堆叠

每个 Block 包含：

```text
LayerNorm
CausalSelfAttention
LayerNorm
MLP
```

Block 的输入输出 shape 都是：

```text
B x T x C
```

因此可以堆叠任意层数。`n_layer` 越大，模型越深。

## 3.6 输出层 lm_head

`lm_head` 把隐藏向量从 `C` 维投影到 `vocab_size` 维：

```text
B x T x C -> B x T x vocab_size
```

每个位置的 logits 都表示下一个 token 的候选分数。

softmax 不在 `GPT.forward` 里显式调用，因为训练时 `F.cross_entropy` 会内部处理 logits。生成时则需要手动 softmax 变成概率。

## 3.7 权重共享

nanoGPT 使用：

```python
self.transformer.wte.weight = self.lm_head.weight
```

这让输入 embedding 和输出分类权重共享参数。

直觉上，token 作为输入和作为输出候选时，可以使用同一套语义空间。这个技巧能减少参数量，也常见于语言模型。

## 3.8 训练目标

假设输入：

```text
x = [10, 20, 30, 40]
y = [20, 30, 40, 50]
```

模型在位置 0 看到 token 10，预测 20。位置 1 看到 10 和 20，预测 30。位置 2 预测 40。位置 3 预测 50。

这就是 teacher forcing：训练时真实历史 token 已经给定，模型可以并行预测所有位置的下一个 token。

## 3.9 生成过程

生成时没有完整 `y`。模型先根据 prompt 预测一个 token，采样后拼接到 prompt 后面，再预测下一个。

```text
[prompt]
-> 预测 token A
-> [prompt, A]
-> 预测 token B
-> [prompt, A, B]
```

这个循环由 `GPT.generate` 实现。

## 3.10 模型大小如何变化

主要由三组参数控制：

```text
n_layer：深度
n_head：注意力头数
n_embd：宽度
```

还有两个重要维度：

```text
block_size：最大上下文
vocab_size：词表大小
```

扩大模型通常提升拟合能力，但会增加显存和训练时间。扩大 `block_size` 尤其会增加 attention 成本。

## 3.11 从架构图到源码

读者可以把 `model.py` 想成三层：

```text
底层积木：LayerNorm, CausalSelfAttention, MLP
中层模块：Block
顶层模型：GPT
```

下一章会按这个顺序阅读源码。

## 本章小结

GPT 是一个自回归 decoder-only Transformer。它把 token id 转成向量，经过多层 causal Transformer Block，再投影到词表空间。训练时并行预测每个位置的下一个 token，生成时循环预测最后一个位置的下一个 token。
