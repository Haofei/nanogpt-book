# 00c 第一次跑通 nanoGPT：逐条命令解释

## 本章目标

本章带你跑通一次最小 nanoGPT 训练。重点不是训练出好模型，而是让你知道每条命令做了什么、会生成哪些文件、看到哪些输出算正常。

本章默认你已经安装依赖，并且在仓库根目录。

## 00c.1 进入源码目录

运行代码前先进入 nanoGPT 源码目录：

```sh
cd code/nanogpt
```

确认当前目录里有这些文件：

```sh
ls
```

你应该能看到：

```text
model.py
train.py
sample.py
data
config
```

如果看不到，说明你不在 `code/nanogpt` 目录。

## 00c.2 准备 Shakespeare 字符级数据

运行：

```sh
python data/shakespeare_char/prepare.py
```

这条命令会做几件事：

```text
下载 tiny shakespeare 文本
-> 找出文本中出现过的所有字符
-> 建立字符到整数的映射
-> 切分 train/val
-> 写出 train.bin 和 val.bin
-> 写出 meta.pkl
```

正常情况下会看到类似输出：

```text
length of dataset in characters: ...
all the unique characters: ...
vocab size: ...
train has ... tokens
val has ... tokens
```

运行后检查文件：

```sh
ls data/shakespeare_char
```

你应该看到：

```text
input.txt
train.bin
val.bin
meta.pkl
prepare.py
```

其中：

- `input.txt` 是原始文本。
- `train.bin` 是训练 token id。
- `val.bin` 是验证 token id。
- `meta.pkl` 保存字符表和 `vocab_size`。

## 00c.3 先跑 1 step 最小训练

不要一开始跑完整训练。先用极小配置验证链路：

```sh
python train.py config/train_shakespeare_char.py \
  --device=cpu \
  --compile=False \
  --max_iters=1 \
  --eval_iters=1 \
  --batch_size=2 \
  --block_size=8 \
  --n_layer=1 \
  --n_head=2 \
  --n_embd=16
```

这条命令的意思：

- `config/train_shakespeare_char.py`：使用 Shakespeare 字符级训练配置。
- `--device=cpu`：先用 CPU，最稳。
- `--compile=False`：关闭 PyTorch 编译，减少入门阶段的不确定性。
- `--max_iters=1`：只训练 1 次迭代。
- `--eval_iters=1`：评估时只取 1 个 batch。
- `--batch_size=2`：每次只取 2 条序列。
- `--block_size=8`：每条序列只有 8 个字符 token。
- `--n_layer=1`：只用 1 层 Transformer Block。
- `--n_head=2`：2 个 attention head。
- `--n_embd=16`：每个 token 向量只有 16 维。

这个模型非常小，生成效果会很差，但很适合验证代码。

## 00c.4 训练输出怎么看

你会看到类似：

```text
tokens per iteration will be: ...
found vocab_size = ...
Initializing a new model from scratch
number of parameters: ...
step 0: train loss ..., val loss ...
iter 0: loss ..., time ..., mfu ...
```

逐行理解：

`tokens per iteration` 表示每次参数更新处理多少 token。

`found vocab_size` 表示 `train.py` 从 `meta.pkl` 里读到了字符词表大小。

`Initializing a new model from scratch` 表示模型从随机权重开始。

`number of parameters` 表示模型有多少可训练参数。

`train loss` 和 `val loss` 是训练集和验证集上的损失。

`iter 0` 是第 0 次训练迭代的日志。

## 00c.5 检查 checkpoint

训练会写到配置里的 `out_dir`。Shakespeare 字符级默认通常是：

```text
out-shakespeare-char
```

检查：

```sh
ls out-shakespeare-char
```

如果看到：

```text
ckpt.pt
```

说明 checkpoint 已保存。

如果没有看到，可能是因为训练步数太少或配置里的保存条件还没触发。可以把 `--max_iters=10` 再跑一次。

## 00c.6 尝试采样

如果已经有 checkpoint，可以运行：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu --max_new_tokens=100 --num_samples=1
```

这个命令会：

```text
读取 out-shakespeare-char/ckpt.pt
-> 重建模型结构
-> 加载权重
-> 找到 data/shakespeare_char/meta.pkl
-> 把 prompt 编码成 token id
-> 生成新 token
-> 解码回字符
```

如果输出很乱，这是正常的。因为我们只训练了 1 step 或很少 step。这个阶段的目标是确认采样链路能跑通。

## 00c.7 常见错误排查

`No such file or directory: train.py`

你可能不在 `code/nanogpt` 目录。先执行：

```sh
cd code/nanogpt
```

`No such file or directory: train.bin`

还没有运行数据准备脚本：

```sh
python data/shakespeare_char/prepare.py
```

`ModuleNotFoundError`

依赖没有安装到当前环境。确认虚拟环境已激活，再执行：

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

`CUDA out of memory`

显存不够。入门阶段先用：

```sh
--device=cpu
```

或者减小 `batch_size`、`block_size`、`n_layer`、`n_embd`。

采样输出乱码

如果模型训练很少，这是正常的。如果字符完全不对，检查是否使用了正确的 `out_dir`，以及对应数据目录里是否有 `meta.pkl`。

## 00c.8 从 1 step 到真正训练

1 step 只验证链路。想看到模型真的学到一点 Shakespeare 风格，可以把训练步数调大：

```sh
python train.py config/train_shakespeare_char.py \
  --device=cpu \
  --compile=False \
  --eval_iters=20 \
  --block_size=64 \
  --batch_size=12 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000
```

这会慢一些，但更可能生成带有 Shakespeare 风格的文本。

## 00c.9 本章检查点

完成本章后，你应该知道：

```text
为什么要先进入 code/nanogpt
prepare.py 生成了哪些文件
train.py 的最小命令每个参数是什么意思
训练日志里每行大概表示什么
checkpoint 在哪里
sample.py 如何加载 checkpoint 生成文本
遇到常见错误先查什么
```

如果这些都清楚，再读后面的 `model.py` 和 `train.py` 源码会轻松很多。下一章会先不急着读代码，而是从项目地图开始，把你刚刚运行过的命令放回 nanoGPT 的整体结构里。
