# 04 读懂 model.py

## 本章目标

本章是全书的第一个源码精读样章。读完以后，读者应该能回答这些问题：

- `nanoGPT` 的模型由哪些模块组成。
- 一个 batch 的 token id 如何变成 logits。
- Q、K、V 的 shape 如何变化。
- causal mask 为什么是 GPT 的必要约束。
- 训练时 loss 如何计算。
- 推理时为什么只取最后一个位置的 logits。
- GPT-2 预训练权重为什么有些张量需要转置。
- AdamW 参数分组为什么要区分 decay 和 no decay。
- `generate` 为什么能不断生成新 token。

对应源码位置：`code/nanogpt/model.py`。

## 4.1 model.py 在项目中的位置

`model.py` 是 nanoGPT 的模型定义文件。它不负责读数据，不负责训练循环，也不负责解析命令行。它只回答一个问题：给定 token id，GPT 如何计算 logits 和 loss？

在训练链路中，`train.py` 会导入：

```python
from model import GPTConfig, GPT
```

然后根据配置构造模型：

```python
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
```

这说明 `model.py` 暴露给外部的核心对象只有两个：

- `GPTConfig`：描述模型结构大小。
- `GPT`：真正的模型类。

其他类如 `LayerNorm`、`CausalSelfAttention`、`MLP`、`Block` 都是内部积木。

## 4.2 GPTConfig：模型大小的集中入口

`GPTConfig` 是一个 dataclass：

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
```

这些参数基本决定了模型规模。

`block_size` 是最大上下文长度。训练时 `idx` 的序列长度不能超过它。生成时如果上下文越来越长，`generate` 会裁剪到最近的 `block_size` 个 token。

`vocab_size` 是输出分类数量。语言模型在每个位置都要从词表里选下一个 token，所以最后一层 `lm_head` 的输出维度就是 `vocab_size`。

`n_layer` 是 Transformer Block 的层数。层数越多，模型深度越大，能表达更复杂的函数，但训练更慢，显存占用也更高。

`n_head` 是 attention head 数。`n_embd` 必须能被 `n_head` 整除，因为每个 head 会分到 `n_embd // n_head` 个通道。

`n_embd` 是隐藏维度，也就是常见记号里的 `C`。大多数张量在模型内部都会变成 `B x T x C`。

`dropout` 是随机丢弃比例，常用于减少过拟合。nanoGPT 默认预训练时设为 0，微调时可以提高。

`bias` 控制 Linear 和 LayerNorm 是否使用 bias。它让模型可以更接近 GPT-2，也可以选择更轻量的无 bias 版本。

## 4.3 LayerNorm：为什么自己包一层

源码：

```python
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

LayerNorm 的作用是对每个 token 的隐藏向量做归一化。假设输入 `x` 的 shape 是：

```text
B x T x C
```

LayerNorm 会在最后一个维度 `C` 上做归一化，所以输出 shape 仍然是：

```text
B x T x C
```

它不会混合 batch，也不会混合不同时间位置。每个 token 位置独立归一化。

nanoGPT 自己写这个类的原因很具体：它希望在配置中控制 bias 是否存在。PyTorch 的标准 `nn.LayerNorm` 有 `elementwise_affine`，但不方便只保留 weight 而关闭 bias。所以这里用函数式 API `F.layer_norm`，手动传入 `weight` 和可选的 `bias`。

理解这段代码时要注意一点：LayerNorm 不是为了改变模型表达维度，而是为了让深层网络更稳定。后面的 `Block` 使用 pre-norm 结构，也就是先 LayerNorm，再进入 Attention 或 MLP。

## 4.4 CausalSelfAttention：初始化阶段

`CausalSelfAttention` 是 GPT 的核心。先看初始化：

```python
assert config.n_embd % config.n_head == 0
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
```

第一行断言非常重要。多头注意力会把 `n_embd` 拆成 `n_head` 份，每个 head 的维度是：

```text
head_size = n_embd // n_head
```

如果不能整除，后面的 reshape 就没有明确含义。

`c_attn` 一次性输出 `3 * n_embd`，因为它同时产生 query、key、value。很多教材会把 Q、K、V 写成三个独立矩阵，但工程实现常常合并成一个 Linear。这样可以减少 kernel 调用，代码也更紧凑。

`c_proj` 是 attention 后的输出投影。多个 head 的输出会拼回 `n_embd`，再经过这个线性层混合通道。

## 4.5 Flash Attention 与手写 Attention

初始化里还有这段：

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if not self.flash:
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                .view(1, 1, config.block_size, config.block_size))
```

如果 PyTorch 提供 `scaled_dot_product_attention`，nanoGPT 会使用内置高效实现。否则，它会注册一个下三角 mask，后面用手写 attention。

`register_buffer` 的含义是：这个张量属于模块状态，会跟着模型移动到 GPU，也会进 `state_dict`，但它不是可训练参数。

下三角矩阵直觉如下：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

第 0 个位置只能看第 0 个 token。第 1 个位置能看第 0 和第 1 个 token。任何位置都不能看未来。这就是 causal attention。

## 4.6 Attention 前向传播：shape 逐步追踪

前向入口：

```python
def forward(self, x):
    B, T, C = x.size()
```

这里：

- `B`：batch size。
- `T`：序列长度。
- `C`：embedding 维度，也就是 `n_embd`。

例如 Shakespeare 小模型可能是：

```text
B = 64
T = 256
C = 384
```

第一步生成 Q、K、V：

```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

`self.c_attn(x)` 的 shape 是：

```text
B x T x 3C
```

`split(self.n_embd, dim=2)` 后得到三个张量，每个 shape 是：

```text
B x T x C
```

接着改成多头形式：

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```

先 `view`：

```text
B x T x n_head x head_size
```

再 `transpose(1, 2)`：

```text
B x n_head x T x head_size
```

这样每个 head 都可以独立做 attention。

## 4.7 手写 Attention 路径

如果没有 Flash Attention，代码走手写路径：

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
att = self.attn_dropout(att)
y = att @ v
```

逐步看 shape：

```text
q: B x nh x T x hs
k.transpose(-2, -1): B x nh x hs x T
att: B x nh x T x T
v: B x nh x T x hs
y: B x nh x T x hs
```

`q @ k^T` 表示每个位置和每个历史位置的匹配程度。除以 `sqrt(head_size)` 是为了避免点积值随维度增大而过大。如果 logits 太大，softmax 会变得极端尖锐，训练更不稳定。

`masked_fill(..., -inf)` 把未来位置的注意力分数设为负无穷。softmax 后，这些位置概率变成 0。

最后 `att @ v` 表示按注意力权重加权汇总 value。

## 4.8 多头输出如何拼回去

attention 得到的 `y` 是：

```text
B x n_head x T x head_size
```

源码：

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.resid_dropout(self.c_proj(y))
```

先把 head 维度换回去：

```text
B x T x n_head x head_size
```

再 `view(B, T, C)`，把所有 head 拼成一个通道维度。`contiguous()` 是因为 `transpose` 后张量内存布局可能不连续，而 `view` 需要兼容的内存布局。

最后 `c_proj` 把拼接结果再投影一次。注意输出 shape 仍然是：

```text
B x T x C
```

这很关键。Block 的输入输出 shape 不变，多个 Block 才能堆叠。

## 4.9 MLP：逐 token 的非线性变换

源码：

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
```

MLP 对每个 token 位置独立生效。它不负责混合不同位置的信息，位置之间的信息混合已经由 attention 完成。

维度变化：

```text
B x T x C
-> B x T x 4C
-> B x T x 4C
-> B x T x C
```

为什么中间是 `4C`？这是 Transformer 中常见的宽度设计。attention 负责从上下文取信息，MLP 负责在每个位置上做更强的非线性变换。扩宽再压回去可以增加表达能力。

## 4.10 Block：pre-norm 与残差连接

源码：

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

这是一个 pre-norm Transformer Block。所谓 pre-norm，是指进入 attention 或 MLP 前先做 LayerNorm。

第一行：

```python
x = x + self.attn(self.ln_1(x))
```

可以拆成：

```text
归一化 x -> attention 读取上下文 -> 和原始 x 相加
```

第二行：

```python
x = x + self.mlp(self.ln_2(x))
```

可以拆成：

```text
归一化 x -> MLP 做非线性变换 -> 和原始 x 相加
```

残差连接的好处是保留一条容易传播梯度的路径。深层模型如果每层都完全覆盖输入，训练会困难得多。残差让每一层更像是在原有表示上增加一个修正项。

## 4.11 GPT.__init__：把积木组装成模型

核心结构：

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

`wte` 是 token embedding。输入 token id 后查表得到 token 向量。

`wpe` 是 position embedding。GPT 没有循环结构，必须显式注入位置信息。

`h` 是 Block 列表。`n_layer=12` 就是 12 个 Block 顺序堆叠。

`ln_f` 是最后的 LayerNorm。

`lm_head` 把隐藏向量投影到词表大小。也就是说，每个位置最终都会得到一个长度为 `vocab_size` 的 logits 向量。

## 4.12 权重共享

源码：

```python
self.transformer.wte.weight = self.lm_head.weight
```

这叫 weight tying。输入 embedding 和输出分类层共享同一套权重。

直觉上，输入 token embedding 表示“某个 token 作为输入时的语义向量”，输出层权重表示“某个 token 作为候选输出时的分类向量”。这两者可以共享同一个空间。

好处：

- 减少参数量。
- 常常改善语言模型效果。
- 让输入和输出 token 表示保持一致。

需要注意的是，这行代码不是复制权重，而是让两个模块引用同一个参数。

## 4.13 参数初始化

源码：

```python
self.apply(self._init_weights)
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

`self.apply` 会递归访问所有子模块，对 Linear 和 Embedding 初始化：

```python
torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

后面又对所有以 `c_proj.weight` 结尾的残差投影做特殊缩放。原因是残差路径会跨很多层累积，如果每层输出尺度不控制，深层网络的激活可能变得不稳定。

这里可以看出一个工程事实：模型结构不只是层的排列，初始化策略也是训练能否稳定的重要组成部分。

## 4.14 GPT.forward：训练路径

源码入口：

```python
def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()
    assert t <= self.config.block_size
    pos = torch.arange(0, t, dtype=torch.long, device=device)
```

`idx` 是 token id，shape 是：

```text
B x T
```

`pos` 是位置 id：

```text
0, 1, 2, ..., T-1
```

接着做 embedding：

```python
tok_emb = self.transformer.wte(idx)
pos_emb = self.transformer.wpe(pos)
x = self.transformer.drop(tok_emb + pos_emb)
```

shape 变化：

```text
idx: B x T
tok_emb: B x T x C
pos_emb: T x C
tok_emb + pos_emb: B x T x C
```

这里 `pos_emb` 会通过 broadcasting 加到每个 batch 样本上。

然后经过所有 Block：

```python
for block in self.transformer.h:
    x = block(x)
x = self.transformer.ln_f(x)
```

shape 全程保持：

```text
B x T x C
```

如果训练时传入 `targets`：

```python
logits = self.lm_head(x)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
```

`logits` shape：

```text
B x T x vocab_size
```

`targets` shape：

```text
B x T
```

`view(-1, vocab_size)` 把所有位置展平成独立分类样本：

```text
(B*T) x vocab_size
```

`targets.view(-1)` 变成：

```text
B*T
```

所以训练目标就是：每个位置都预测它的下一个 token。

## 4.15 GPT.forward：推理路径

如果 `targets is None`：

```python
logits = self.lm_head(x[:, [-1], :])
loss = None
```

为什么只取最后一个位置？因为生成时我们只需要下一个 token。假设 prompt 有 100 个 token，模型会为每个位置都算出 logits，但真正用于采样的只有最后一个位置的 logits。

`x[:, [-1], :]` 保留时间维度，所以 shape 是：

```text
B x 1 x C
```

经过 `lm_head` 后：

```text
B x 1 x vocab_size
```

这比对所有 `T` 个位置都做输出投影更省计算。

## 4.16 crop_block_size：缩短上下文

源码：

```python
def crop_block_size(self, block_size):
    assert block_size <= self.config.block_size
    self.config.block_size = block_size
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
```

这个方法用于把模型的最大上下文长度变短。典型场景是加载 GPT-2 权重后，想用更短上下文做微调，减少显存开销。

它会裁剪 position embedding。如果存在手写 attention 的 mask，也会裁剪 mask。

注意：它只能缩短，不能变长。因为变长需要新 position embedding，而旧 checkpoint 没有这些位置的权重。

## 4.17 from_pretrained：加载 GPT-2 权重

`from_pretrained` 支持：

```python
{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
```

不同模型对应不同层数、头数和 embedding 维度：

```python
'gpt2':         dict(n_layer=12, n_head=12, n_embd=768)
'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024)
'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280)
'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600)
```

加载时强制：

```python
vocab_size = 50257
block_size = 1024
bias = True
```

这是因为 GPT-2 原始 checkpoint 的这些结构是固定的。想加载权重，模型结构必须匹配。

一个关键工程细节：

```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
```

这些权重需要转置，因为 OpenAI GPT-2 的 TensorFlow/Conv1D 表达和 PyTorch Linear 的权重布局不同。数学上是同一个线性变换，存储时 shape 方向不同。

这也是阅读模型迁移代码时必须注意的点：不能只看层名字，还要看张量 shape。

## 4.18 configure_optimizers：AdamW 参数分组

源码逻辑：

```python
param_dict = {pn: p for pn, p in self.named_parameters()}
param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
```

规则是：

- 二维及以上参数使用 weight decay。
- 一维参数不使用 weight decay。

一般来说，Linear 权重和 Embedding 权重是二维矩阵，适合 weight decay。bias、LayerNorm weight 是一维向量，不做 decay。

为什么要这样分？AdamW 的 weight decay 是一种正则化，但对归一化参数和 bias 做 decay 往往没有必要，甚至可能影响训练稳定性。

后面还有 fused AdamW 检测：

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
```

如果当前 PyTorch 和 CUDA 支持 fused AdamW，就使用更快实现。

## 4.19 estimate_mfu：估算训练效率

`estimate_mfu` 估算模型实际使用了 A100 峰值算力的多少比例。它不是训练正确性的指标，而是性能观察工具。

核心思路：

```text
每 token FLOPs -> 每 forward/backward FLOPs -> 每 iteration FLOPs -> 每秒 FLOPs -> 除以 A100 峰值
```

其中 attention 的成本和 `T` 有强关系。上下文越长，attention 的 `T x T` 计算越重。

阅读这段代码的意义不是记公式，而是理解性能优化要看 tokens per second、batch、上下文长度、模型大小和硬件利用率。

## 4.20 generate：自回归生成

源码：

```python
for _ in range(max_new_tokens):
    idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
    logits, _ = self(idx_cond)
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    idx = torch.cat((idx, idx_next), dim=1)
```

生成过程可以翻译成：

```text
拿当前上下文
-> 如果太长就裁剪
-> 计算最后一个位置的 logits
-> 用 temperature 调整分布
-> 可选 top-k 截断
-> softmax 得到概率
-> 采样一个 token
-> 拼到序列末尾
-> 重复
```

`temperature` 越低，模型越保守。`temperature` 越高，模型越随机。`top_k` 限制候选范围，能减少极低概率 token 被采样到。

这里使用 `torch.multinomial`，所以生成不是简单取最大概率 token。它是从概率分布中采样。

## 4.21 一个建议实验：打印 shape

为了让读者真正理解 attention，可以临时在 `CausalSelfAttention.forward` 中加入：

```python
print("x", x.shape)
print("q", q.shape)
print("k", k.shape)
print("v", v.shape)
```

然后用极小配置跑一次：

```sh
cd code/nanogpt
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=1 --eval_iters=1 --batch_size=2 --block_size=8 --n_layer=1 --n_head=2 --n_embd=16
```

预期能看到类似：

```text
x torch.Size([2, 8, 16])
q torch.Size([2, 2, 8, 8])
k torch.Size([2, 2, 8, 8])
v torch.Size([2, 2, 8, 8])
```

这个实验比单纯看公式更有效，因为它把抽象概念落到了真实张量。

## 4.22 常见问题

为什么训练时每个位置都预测，而生成时只用最后一个位置？

训练时已知整段序列，所以所有位置都能并行计算 loss。生成时每次只需要新 token，因此只取最后位置的 logits。

为什么 causal mask 不允许看未来？

如果训练时能看未来，模型会学到作弊路径。生成时未来 token 并不存在，这会导致训练和推理不一致。

为什么 Q、K、V 用一个 Linear 生成？

这是工程优化。数学上等价于三个 Linear，但一次性计算更紧凑。

为什么 `block_size` 不能随便超过配置？

position embedding 和 attention mask 都按 `block_size` 初始化。超过后没有对应的位置参数。

## 本章小结

`model.py` 把 GPT 的主要思想压缩进一个文件：embedding 加位置信息，多个 pre-norm Transformer Block 堆叠，最后用 `lm_head` 预测下一个 token。训练时它输出完整序列 loss，生成时它反复预测最后一个位置的下一个 token。

真正读懂这一章，不是背下类名，而是能从 `B x T` 的 token id 一路追踪到 `B x T x vocab_size` 的 logits，并解释每一步为什么存在。
