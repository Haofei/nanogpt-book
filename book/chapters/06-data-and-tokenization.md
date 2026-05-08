# 06 数据与 Tokenization

## 本章目标

理解 nanoGPT 如何把文本数据变成训练文件，以及字符级 tokenization 和 GPT-2 BPE tokenization 的区别。

## 为什么需要 Tokenization

神经网络只能处理数字。语言模型训练前，文本必须被编码成整数 token id。不同 tokenizer 会产生不同粒度的 token。

## Shakespeare 字符级数据

`data/shakespeare_char/prepare.py` 使用字符级编码：

```python
chars = sorted(list(set(data)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
```

这种方式非常适合教学：

- 词表小。
- 数据准备快。
- 不依赖复杂 tokenizer。
- 生成结果容易观察。

缺点是序列更长，模型需要用更多 token 表达同样的文本。

## GPT-2 BPE 数据

`data/openwebtext/prepare.py` 使用 `tiktoken` 的 GPT-2 BPE：

```python
enc = tiktoken.get_encoding("gpt2")
ids = enc.encode_ordinary(example['text'])
ids.append(enc.eot_token)
```

BPE 的 token 粒度介于字符和单词之间，常见片段会合并为一个 token。这更接近 GPT-2 的训练方式。

## train.bin 与 val.bin

准备脚本最终会把 token id 写成二进制文件：

```python
train_ids.tofile(...)
val_ids.tofile(...)
```

训练时再用 `np.memmap` 读取。这样做的好处是大文件不必一次性读入内存，训练循环可以快速随机切片。

## 数据切分

Shakespeare 字符级脚本按 90/10 切分训练和验证数据。OpenWebText 脚本则从训练集中抽出很小比例作为验证集。验证集不参与参数更新，只用于估计模型泛化效果。

## 常见问题

如果 `sample.py` 找到 `meta.pkl`，它会用训练数据对应的字符级编码器解码。否则默认使用 GPT-2 tokenizer。这解释了为什么字符级模型采样时需要保留 `meta.pkl`。

## 小结

数据准备决定了模型看见的世界。训练同一个 GPT 结构时，字符级数据和 BPE 数据会产生非常不同的学习难度和生成风格。

