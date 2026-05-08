# 08 采样与文本生成

## 本章目标

本章解释 `sample.py` 如何加载模型并生成文本。读者应该能理解 prompt 如何编码、logits 如何变成概率、temperature 和 top-k 如何影响输出，以及为什么生成过程必须一个 token 一个 token 地进行。

对应源码位置：`code/nanogpt/sample.py`、`code/nanogpt/model.py` 的 `GPT.generate`。

## 8.1 训练和生成的差异

训练时，模型一次处理完整序列，并为每个位置计算 loss：

```text
输入 x: B x T
目标 y: B x T
输出 logits: B x T x vocab_size
```

生成时，未来 token 不存在。模型只能根据已有 prompt 预测下一个 token，采样后再把新 token 拼回上下文。

生成链路：

```text
prompt text
-> encode
-> token ids
-> model.generate
-> decode
-> generated text
```

训练和生成还有一个关键差异：训练时模型看到的历史 token 都来自真实数据，生成时后续历史 token 来自模型自己采样的结果。如果模型前面生成错了，后面会继续基于错误上下文生成。这也是为什么小模型经常越写越跑偏。

## 8.2 sample.py 的配置

`sample.py` 顶部定义：

```python
init_from = 'resume'
out_dir = 'out'
start = "\n"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
device = 'cuda'
```

这些都可以用命令行覆盖：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu --num_samples=3 --max_new_tokens=200
```

采样脚本也使用 `configurator.py`，所以覆盖方式和训练脚本一致。

## 8.3 加载本地 checkpoint

默认路径：

```python
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
model.load_state_dict(state_dict)
```

这里先从 checkpoint 读取 `model_args`，再构造同样结构的模型，最后加载权重。

这说明采样不能只拿到权重。模型结构也必须和训练时一致。

## 8.4 加载 GPT-2 权重

如果：

```sh
python sample.py --init_from=gpt2 --start="Once upon a time"
```

代码会走：

```python
model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```

这种模式不需要本地 checkpoint，直接从 Hugging Face 加载 GPT-2 系列权重。

## 8.5 Prompt 如何编码

如果 `start` 以 `FILE:` 开头：

```python
with open(start[5:], 'r', encoding='utf-8') as f:
    start = f.read()
```

否则直接使用命令行里的字符串。

编码器选择：

```text
如果 checkpoint 指向的数据目录有 meta.pkl -> 使用 meta.pkl 中的 stoi/itos
否则 -> 默认 GPT-2 tokenizer
```

字符级模型必须使用自己的 `meta.pkl`。GPT-2 模型或 BPE 数据通常使用 `tiktoken`。

## 8.6 generate 的核心循环

`GPT.generate` 每次循环生成一个 token：

```python
idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
logits, _ = self(idx_cond)
logits = logits[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
idx = torch.cat((idx, idx_next), dim=1)
```

注意两点。

第一，`idx_cond` 会裁剪上下文。模型不能处理超过 `block_size` 的序列，所以长文本生成时只能使用最近的一段上下文。

第二，采样只使用最后位置的 logits。因为我们只需要预测下一个 token。

## 8.7 解码策略的背景

如果每一步都取最大概率 token，这叫贪心解码。它稳定，但容易重复，也可能错过更自然的表达。

如果从概率分布中采样，输出更多样，但也更不可控。temperature 和 top-k 的作用就是在稳定性和多样性之间调节。

实际使用中，生成质量不只取决于模型本身，也取决于解码策略。同一个 checkpoint，在不同 temperature、top-k、prompt 下可能表现差异很大。

## 8.8 Temperature

temperature 作用在 logits 上：

```python
logits = logits[:, -1, :] / temperature
```

当 `temperature < 1`，logits 差距被放大，概率分布更尖锐，输出更稳定但可能更重复。

当 `temperature > 1`，logits 差距被缩小，概率分布更平，输出更多样但可能更混乱。

常用实验范围：

```text
0.6, 0.8, 1.0, 1.2
```

## 8.9 Top-k

top-k 会保留最高的 k 个候选：

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
```

被设为负无穷的 logits 在 softmax 后概率为 0。

top-k 的直觉是：不让模型从极低概率 token 中采样。它能提升输出稳定性，但太小会降低多样性。

## 8.10 贪心、采样和可重复性

nanoGPT 使用：

```python
torch.multinomial(probs, num_samples=1)
```

这意味着它是采样，不是每次取最大概率 token。相同 prompt 也可能生成不同结果。

为了提高可重复性，可以固定：

```sh
--seed=1337
```

但不同硬件和不同 PyTorch 后端仍可能存在细微差异。

## 8.11 采样实验

固定模型和 prompt，比较参数：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="KING:" --temperature=0.6 --top_k=50
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="KING:" --temperature=0.8 --top_k=200
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="KING:" --temperature=1.2 --top_k=500
```

记录：

- 是否重复。
- 是否语法更稳定。
- 是否出现更多新词或奇怪字符。
- 是否保持 Shakespeare 风格。

## 8.12 常见问题

为什么 prompt 很长时前面的内容好像被忘了？

模型最多只能看到 `block_size` 长度的上下文。超过后，`generate` 会裁剪掉最早的 token。

为什么输出乱码？

优先检查 tokenizer 是否匹配。字符级 checkpoint 需要对应数据目录的 `meta.pkl`。

为什么输出每次不同？

因为使用概率采样。降低 temperature、降低 top-k 或固定 seed 可以让输出更稳定。

## 本章小结

文本生成的本质是反复执行 next-token prediction。`sample.py` 负责加载模型和编码 prompt，`generate` 负责循环采样。temperature 和 top-k 是最基础也最重要的生成控制参数。
