# 附录 B 数学符号速查

## Cross Entropy

语言模型训练中，每个位置都是一个分类问题。给定真实下一个 token `y` 和模型输出分布 `p`，交叉熵可以理解为：

```text
loss = -log p(y)
```

如果模型给真实 token 的概率越高，loss 越低。

## Softmax

Softmax 把 logits 转成概率分布：

```text
softmax(z_i) = exp(z_i) / sum_j exp(z_j)
```

logits 的相对大小决定概率，而不是绝对大小。

## Scaled Dot-Product Attention

Attention 的核心公式：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

除以 `sqrt(d_k)` 是为了避免维度增大后点积数值过大，导致 softmax 太尖锐。

## Causal Mask

GPT 使用下三角 mask，保证第 `t` 个位置不能关注未来位置。矩阵直觉如下：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

`1` 表示可见，`0` 表示不可见。

## Perplexity

困惑度常由平均交叉熵转换得到：

```text
perplexity = exp(loss)
```

loss 越低，perplexity 越低。它可以粗略理解为模型在每个位置面对的平均候选不确定性。

