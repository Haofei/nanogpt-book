# 06 数据与 Tokenization

## 本章目标

本章解释 nanoGPT 如何把原始文本变成训练需要的二进制 token 文件。读完后，读者应该能理解：

- 字符级 tokenizer 和 BPE tokenizer 的区别。
- `train.bin`、`val.bin` 和 `meta.pkl` 分别保存什么。
- 为什么训练时使用 `np.memmap`。
- 如何为自己的文本数据创建新数据目录。

对应源码位置：`code/nanogpt/data/shakespeare_char/prepare.py`、`code/nanogpt/data/shakespeare/prepare.py`、`code/nanogpt/data/openwebtext/prepare.py`、`code/nanogpt/train.py` 的 `get_batch`。

## 6.1 为什么不能直接训练字符串

神经网络处理的是数字张量。文本必须先变成整数序列：

```text
文本 -> token -> token id -> Tensor
```

不同 tokenizer 会决定“一个 token 是什么”。字符级 tokenizer 把每个字符当成 token。BPE tokenizer 会把常见字符片段、词片段或单词编码成 token。

这一步非常重要，因为模型看到的不是原始文字，而是 tokenizer 之后的 id 序列。

## 6.2 Shakespeare 字符级数据

`data/shakespeare_char/prepare.py` 是最适合学习的入口。它先下载 `input.txt`，然后找出全部字符：

```python
chars = sorted(list(set(data)))
vocab_size = len(chars)
```

接着建立双向映射：

```python
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
```

`stoi` 是 string-to-index，负责编码。`itos` 是 index-to-string，负责解码。

编码函数：

```python
def encode(s):
    return [stoi[c] for c in s]
```

解码函数：

```python
def decode(l):
    return ''.join([itos[i] for i in l])
```

字符级方案的优点是透明。读者可以直接看到每个字符如何映射成整数。缺点是效率不高，因为表达同样一句话需要更多 token。

## 6.3 train/val 切分

脚本按 90/10 切分：

```python
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
```

训练集用于参数更新，验证集用于评估模型是否泛化。验证集不能参与训练，否则 val loss 就失去意义。

对于时间顺序或文体连续的文本，这种简单切分可能导致训练和验证分布略有差异。作为教学实验可以接受，正式实验则需要更严谨地设计切分方法。

## 6.4 写出 train.bin 和 val.bin

脚本把 token id 转成 `uint16`：

```python
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
```

为什么用 `uint16`？因为字符级词表很小，GPT-2 BPE 的最大 token id 也小于 `2**16`。用 `uint16` 比 `int64` 更省磁盘。

训练时再转回 `int64`，因为 PyTorch embedding 层需要 long 类型索引。

## 6.5 meta.pkl 的作用

字符级脚本会保存：

```python
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
```

`train.py` 会读取 `meta.pkl` 中的 `vocab_size`，这样模型输出维度和数据词表一致。

`sample.py` 会读取 `stoi` 和 `itos`，这样模型生成的 id 能被正确解码回字符。

如果没有 `meta.pkl`，`sample.py` 会默认使用 GPT-2 tokenizer。这对字符级模型是错误的，所以字符级数据目录必须保留 `meta.pkl`。

## 6.6 GPT-2 BPE 数据

`data/openwebtext/prepare.py` 使用：

```python
enc = tiktoken.get_encoding("gpt2")
```

处理每篇文本：

```python
ids = enc.encode_ordinary(example['text'])
ids.append(enc.eot_token)
```

`encode_ordinary` 不处理特殊 token。脚本手动追加 `eot_token`，表示文档结束。

BPE 的优势：

- 比字符级序列短。
- 更接近 GPT-2 的预训练设置。
- 常见词片段能被一个 token 表示。

BPE 的代价：

- 不像字符级那样直观。
- tokenizer 本身成为模型行为的一部分。
- 换 tokenizer 往往意味着不能直接复用旧模型的 embedding。

## 6.7 OpenWebText 的大文件写入

OpenWebText 数据很大。脚本用 memmap 写入：

```python
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
```

然后分 shard 写入：

```python
for batch_idx in tqdm(range(total_batches)):
    batch = dset.shard(...)
    arr_batch = np.concatenate(batch['ids'])
    arr[idx : idx + len(arr_batch)] = arr_batch
```

这样不需要把所有 token 一次性放进内存。

## 6.8 训练时为什么用 np.memmap

`train.py` 的 `get_batch` 每次读取：

```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
```

memmap 让大文件像数组一样被切片，但底层由操作系统按需加载。对于 OpenWebText 这种十几 GB 的文件，这比一次性读入内存更实际。

`get_batch` 的随机切片方式假设数据已经是一条连续 token 流。它不关心文档边界，只负责随机抽取长度为 `block_size` 的片段。

## 6.9 为自己的数据创建目录

建议复制这个结构：

```text
data/my_dataset/
  input.txt
  prepare.py
  readme.md
```

最小字符级 prepare 流程：

```text
读取 input.txt
-> 构建字符词表
-> 切分 train/val
-> 编码成 id
-> 写出 train.bin 和 val.bin
-> 写出 meta.pkl
```

训练时指定：

```sh
python train.py --dataset=my_dataset
```

如果使用配置文件，则在 `config/train_my_dataset.py` 中设置：

```python
dataset = 'my_dataset'
```

## 6.10 常见错误

`FileNotFoundError: train.bin`

说明还没有运行对应数据目录下的 `prepare.py`。

`vocab_size` 不匹配

通常是 checkpoint 的模型结构和当前数据 tokenizer 不一致。字符级模型不能直接换成 GPT-2 BPE 数据继续训练，除非重新设计 embedding 和输出层。

生成乱码

可能是 `sample.py` 使用了错误的 tokenizer。检查 checkpoint 里的 dataset 配置，以及对应目录下是否有 `meta.pkl`。

## 本章小结

数据准备不是训练前的杂活，而是语言模型系统的一部分。tokenizer 决定了模型预测的基本单位，二进制文件决定了训练读取方式，`meta.pkl` 决定了训练和采样是否能使用同一套编码规则。
