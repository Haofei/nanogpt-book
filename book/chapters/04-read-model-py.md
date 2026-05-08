# 04 读懂 model.py

## 本章目标

逐块阅读 `model.py`，理解 nanoGPT 的模型定义、参数初始化、优化器分组、GPT-2 权重加载和自回归生成。

## LayerNorm

源码中的 `LayerNorm` 主要解决一个小差异：PyTorch 的 `nn.LayerNorm` 不方便只关闭 bias，所以项目自己包了一层。

```python
self.weight = nn.Parameter(torch.ones(ndim))
self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
```

当 `bias=False` 时，LayerNorm 只保留缩放参数，不保留平移参数。这个选项会和模型配置中的 `bias` 保持一致。

## CausalSelfAttention

Attention 初始化中有两个关键线性层：

- `c_attn`：一次性生成 Q、K、V。
- `c_proj`：把多头拼接后的结果投影回 `n_embd`。

前向传播的核心 shape 是：

```text
x: B x T x C
q/k/v: B x n_head x T x head_size
attention score: B x n_head x T x T
y: B x T x C
```

如果当前 PyTorch 支持 `scaled_dot_product_attention`，代码走 Flash Attention 路径；否则走手写 attention 路径。两条路径的语义相同，区别主要在性能和内存。

## MLP 与 Block

`MLP` 是两层线性变换加 GELU：

```text
C -> 4C -> C
```

`Block` 则把 Attention 与 MLP 串起来：

```python
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))
```

这说明每一层都由两次“归一化、变换、残差相加”组成。

## GPT.forward

`GPT.forward` 有两个模式：

- 训练模式：传入 `targets`，返回完整序列 logits 和 cross entropy loss。
- 推理模式：不传 `targets`，只计算最后一个位置的 logits，节省计算。

训练时的 loss：

```python
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

这里把 `B x T x vocab_size` 压成二维分类问题，每个位置都预测下一个 token。

## from_pretrained

`from_pretrained` 用 Hugging Face 的 GPT-2 权重初始化 nanoGPT 模型。一个重要细节是 OpenAI GPT-2 原实现中部分权重来自 Conv1D 表达，在 PyTorch Linear 中需要转置：

```python
if any(k.endswith(w) for w in transposed):
    sd[k].copy_(sd_hf[k].t())
```

这类代码体现了工程迁移时的关键问题：数学结构一样，不代表框架中的张量布局一样。

## configure_optimizers

优化器分组把参数分成两类：

- 二维及以上参数使用 weight decay，例如 Linear 和 Embedding 权重。
- 一维参数不使用 weight decay，例如 bias 和 LayerNorm 权重。

这是训练 Transformer 时常见的 AdamW 设置。

## generate

`generate` 是自回归生成循环：

```text
裁剪上下文 -> 模型前向 -> 取最后位置 logits -> temperature/top-k -> 采样 -> 拼接新 token
```

这个函数很短，但它展示了语言模型生成文本的本质：反复预测下一个 token。

## 小结

`model.py` 是整本书的核心。读懂它后，GPT 不再是抽象架构图，而是一系列清晰的张量变换和训练约束。

