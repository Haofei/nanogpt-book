# 08 采样与文本生成

## 本章目标

理解 `sample.py` 和 `GPT.generate` 如何把模型输出的 logits 变成文本。

## 加载模型

`sample.py` 支持两种来源：

- `--init_from=resume`：从本地 checkpoint 加载自己训练的模型。
- `--init_from=gpt2`、`gpt2-medium`、`gpt2-large`、`gpt2-xl`：加载 GPT-2 权重。

本地模型命令示例：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

GPT-2 权重示例：

```sh
python sample.py --init_from=gpt2 --start="Once upon a time" --max_new_tokens=100
```

## Prompt 编码

如果 prompt 以 `FILE:` 开头，`sample.py` 会从文件读取内容。否则直接把命令行字符串作为 prompt。

编码器选择逻辑：

- 找到训练数据的 `meta.pkl`：使用该数据集自己的编码器。
- 找不到：默认使用 GPT-2 tokenizer。

## Temperature

temperature 控制 logits 的缩放：

```python
logits = logits[:, -1, :] / temperature
```

较低 temperature 会让分布更尖锐，输出更保守。较高 temperature 会让分布更平，输出更多样，但也更容易跑偏。

## Top-k

top-k 会只保留概率最高的 k 个候选 token：

```python
v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
logits[logits < v[:, [-1]]] = -float('Inf')
```

这能减少极低概率 token 被采样到的机会，让生成更稳定。

## 自回归循环

`generate` 每次只生成一个 token：

```text
输入当前序列 -> 得到最后位置 logits -> 采样一个 token -> 拼到序列末尾 -> 重复
```

当序列超过 `block_size` 时，只保留最近的上下文。这是因为模型的 position embedding 和 attention mask 都有最大长度限制。

## 小结

生成文本不是一次性输出整段文章，而是不断重复 next-token prediction。temperature 和 top-k 是控制生成风格的两个基础旋钮。

