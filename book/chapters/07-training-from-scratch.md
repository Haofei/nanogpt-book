# 07 从零训练与微调

## 本章目标

本章把 `README.md` 中的命令扩展成完整训练指南。读完后，读者应该知道如何选择训练路径、如何调整小模型参数、如何判断训练是否正常，以及从零训练和微调有什么本质区别。

对应源码位置：`code/nanogpt/train.py`、`code/nanogpt/config/train_shakespeare_char.py`、`code/nanogpt/config/finetune_shakespeare.py`、`code/nanogpt/config/train_gpt2.py`。

## 本章在全书中的位置

到这里，你已经分别理解了模型、训练脚本和数据。本章把它们重新合成一个可操作流程：继续以 Shakespeare 字符级模型为主线，解释如何从最小 1 step 验证走向真正训练，以及什么时候该尝试微调 GPT-2。

## 7.1 三种训练任务

nanoGPT 常见任务可以分成三类：

```text
从零训练小模型
从 GPT-2 权重微调
复现 GPT-2 规模预训练
```

初学者应该从第一类开始。它需要的数据和算力都最少，失败成本低。微调适合已有预训练模型、想让模型适应新文本风格的场景。复现 GPT-2 预训练需要多卡 GPU 和大数据，不适合作为第一步。

## 7.2 第一个实验：Shakespeare 字符级模型

准备数据：

```sh
cd code/nanogpt
python data/shakespeare_char/prepare.py
```

训练：

```sh
python train.py config/train_shakespeare_char.py
```

这个配置会训练一个较小的字符级 GPT。它的目标不是生成真正高质量文本，而是帮助读者跑通完整链路：

```text
prepare.py -> train.bin/val.bin -> train.py -> ckpt.pt -> sample.py
```

## 7.3 没有 GPU 时怎么跑

CPU 训练可以使用更小配置：

```sh
python train.py config/train_shakespeare_char.py \
  --device=cpu \
  --compile=False \
  --eval_iters=20 \
  --log_interval=1 \
  --block_size=64 \
  --batch_size=12 \
  --n_layer=4 \
  --n_head=4 \
  --n_embd=128 \
  --max_iters=2000 \
  --lr_decay_iters=2000 \
  --dropout=0.0
```

这里每个参数都有明确目的：

- `--device=cpu`：不用 CUDA。
- `--compile=False`：避免 CPU 调试时的编译开销。
- `--eval_iters=20`：减少评估成本。
- `--block_size=64`：缩短上下文。
- `--n_layer=4 --n_head=4 --n_embd=128`：缩小模型。
- `--max_iters=2000`：减少训练步数。

这类命令不是最佳效果配置，而是“让实验能在普通机器上完成”的配置。

## 7.4 Apple Silicon 的 mps

如果是 Apple Silicon，可以尝试：

```sh
python train.py config/train_shakespeare_char.py --device=mps --compile=False
```

`mps` 可以使用 Apple 的 Metal 后端。它通常比 CPU 快，但算子支持和性能不一定等同 CUDA。遇到异常时，先用 CPU 跑最小配置确认代码链路，再切回 `mps`。

## 7.5 关键超参数怎么理解

`block_size` 决定模型能看到多长上下文。它越大，attention 成本越高，因为 attention score 是 `T x T`。

`batch_size` 是每个 micro batch 的样本数。它越大，显存占用越高。

`gradient_accumulation_steps` 用多个 micro batch 模拟更大 batch。显存不够时，可以降低 `batch_size`，提高累积步数。

`n_layer`、`n_head`、`n_embd` 决定模型容量。扩大它们通常能降低训练 loss，但也更慢、更容易过拟合小数据。

`learning_rate` 决定每次参数更新幅度。过大可能发散，过小则训练很慢。

`dropout` 是正则化。小数据微调时可以适当增加，预训练时常设为 0。

## 7.6 读者容易忽略的训练背景

很多初学者会把“训练一次模型”理解成模型完整读完一遍数据。nanoGPT 不是这样组织的。它把 `train.bin` 看成一条连续 token 流，每次随机取很多长度为 `block_size` 的片段。训练进度主要用 iteration 和 token 数衡量，而不是 epoch。

另一个容易混淆的概念是 batch。`batch_size=12` 不代表一次更新只看 12 个 token，而是 12 条序列。每条序列长度是 `block_size`。如果 `block_size=1024`，一个 micro batch 就有 12,288 个 token。再乘以 `gradient_accumulation_steps`，才是一次 optimizer step 的有效 token 数。

还要区分“模型变大”和“上下文变长”。增大 `n_layer`、`n_head`、`n_embd` 主要增加模型容量。增大 `block_size` 主要让模型看见更长历史，但 attention 的计算和显存会明显增加。两者都会变慢，但原因不同。

## 7.7 最小实验应该观察什么

第一次跑最小训练时，不要期待生成文本变好。更合理的观察目标是：

```text
数据脚本能否生成 train.bin / val.bin / meta.pkl
模型参数量是否符合配置变化
step 0 是否能算出 train/val loss
iter 日志是否持续输出
out_dir 是否生成 checkpoint
sample.py 是否能加载 checkpoint
```

这些是工程链路验证。只有链路稳定后，才值得讨论模型效果。

## 7.8 什么时候该扩大模型

如果 train loss 和 val loss 都很高，且训练时间足够，模型可能欠拟合，可以尝试增加模型容量或训练更久。

如果 train loss 明显低于 val loss，模型已经开始记训练集，这时扩大模型通常会加重过拟合。更合理的方向是增加数据、提高 dropout、缩短训练或减小模型。

如果训练很慢但 loss 正常下降，优先检查硬件和配置，不要盲目调学习率。速度问题和收敛问题要分开处理。

## 7.9 如何看日志

训练中常见日志：

```text
tokens per iteration will be: ...
number of parameters: ...
step 0: train loss ..., val loss ...
iter 0: loss ..., time ..., mfu ...
```

`tokens per iteration` 告诉你每次 optimizer step 处理多少 token。`number of parameters` 告诉你模型规模。`train loss` 和 `val loss` 是判断训练是否有效的主要指标。

早期 loss 快速下降是正常的。后期下降变慢也是正常的。只要 val loss 总体下降，训练大概率在正常进行。

## 7.10 从零训练和微调的区别

从零训练：

```python
init_from = 'scratch'
```

模型从随机权重开始。它必须从当前数据中学习所有语言规律。

微调：

```python
init_from = 'gpt2'
```

模型先加载 GPT-2 权重，再在新数据上继续训练。它已经具备通用语言能力，微调主要改变风格、领域和局部行为。

微调通常使用更小学习率，因为大模型已有有用表示，过大学习率可能破坏预训练知识。

## 7.11 微调 Shakespeare

准备 BPE 版 Shakespeare 数据：

```sh
python data/shakespeare/prepare.py
```

运行微调：

```sh
python train.py config/finetune_shakespeare.py
```

字符级 Shakespeare 是从零训练小词表模型；这个微调实验则是用 GPT-2 tokenizer 和 GPT-2 权重。两者生成风格、训练速度和数据需求都不同。

## 7.12 保存与采样

训练会在 `out_dir` 中保存 checkpoint。训练完后采样：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

采样结果不只受模型影响，也受 `temperature` 和 `top_k` 影响。评估模型时要固定采样参数，否则不同输出之间不容易比较。

## 7.13 训练异常排查

loss 变成 `nan`：

可能是学习率太大、混合精度不稳定或梯度爆炸。先降低学习率，关闭 compile，缩小模型，用 CPU 跑最小配置复现。

训练极慢：

检查是否误用 CPU，是否开启了不合适的 `compile`，以及 `eval_interval` 和 `eval_iters` 是否太大。

val loss 上升：

可能是过拟合。可以减少训练步数、增加 dropout、缩小模型或增加数据。

生成结果重复：

可能是采样 temperature 太低，top-k 太小，或模型训练不足。

## 7.14 本章实验

建议记录三组训练：

```text
小模型：n_layer=2, n_head=2, n_embd=64
中模型：n_layer=4, n_head=4, n_embd=128
较大模型：n_layer=6, n_head=6, n_embd=384
```

固定数据和训练步数，比较参数量、训练速度、best val loss 和采样效果。这会帮助读者建立模型容量和训练成本之间的直觉。

## 本章小结

训练 nanoGPT 的关键不是记住某条命令，而是理解命令背后的资源约束和实验目标。小实验用于学习链路，微调用于迁移能力，大规模预训练用于复现模型能力。三者的参数选择逻辑不同。下一章会使用训练得到的 checkpoint，解释文本生成和采样参数。
