# 09 性能、显存与分布式训练

## 本章目标

本章解释 nanoGPT 中主要性能设计。读者应该能理解训练速度和显存受哪些因素影响，以及 `torch.compile`、混合精度、Flash Attention、梯度累积和 DDP 各自解决什么问题。

对应源码位置：`code/nanogpt/train.py`、`code/nanogpt/model.py`、`code/nanogpt/bench.py`。

## 本章在全书中的位置

前面章节已经跑通了完整训练和采样闭环。本章不改变模型目标，而是解释同一套训练为什么在 CPU、MPS、CUDA、单卡和多卡上表现不同。你可以把它看成“当 Shakespeare 小实验跑通后，如何理解速度和显存”的章节。

## 9.1 性能不是单一指标

训练性能至少包含四类问题：

```text
能不能放进显存
每秒能处理多少 token
多卡通信是否成为瓶颈
训练结果是否仍然稳定
```

一个配置可能速度很快但 loss 不稳定，也可能 loss 正常但吞吐很低。优化时要同时记录训练质量和运行效率。

## 9.2 显存主要花在哪里

Transformer 训练显存主要来自：

- 模型参数。
- optimizer 状态。
- 激活值。
- attention 中间结果。
- batch 数据。

扩大 `n_layer`、`n_embd` 会增加参数和激活。扩大 `block_size` 会显著增加 attention 成本。扩大 `batch_size` 会线性增加激活显存。

因此显存不够时，常见处理顺序是：

```text
降低 batch_size
-> 增加 gradient_accumulation_steps 保持有效 batch
-> 降低 block_size
-> 缩小模型
-> 使用混合精度
```

## 9.3 读者需要先知道的硬件背景

CPU 擅长通用计算，但深度学习训练中的大矩阵乘法更适合 GPU。GPU 有大量并行计算单元，能同时处理许多矩阵操作。

CUDA 是 NVIDIA GPU 的计算平台。大多数 PyTorch 深度学习性能优化最先围绕 CUDA 成熟。MPS 是 Apple Silicon 的 Metal 后端，可以使用 Apple 芯片上的 GPU，但算子覆盖、性能和数值行为不完全等同 CUDA。

显存和内存也要区分。CPU 内存不足会影响数据加载或系统稳定；GPU 显存不足会直接导致训练 batch 放不下。报错里如果出现 CUDA out of memory，优先调小 `batch_size` 或 `block_size`。

吞吐通常用 tokens per second 或 iteration time 衡量。显存占用和吞吐不是同一个指标：有些设置更省显存但更慢，有些设置更快但更吃显存。

## 9.4 混合精度

`train.py` 根据设备选择 dtype：

```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

然后用 autocast：

```python
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

混合精度让部分矩阵计算使用更低精度，通常能提升速度并减少显存。

`bfloat16` 动态范围更大，通常比 `float16` 更稳定。`float16` 路径使用 `GradScaler` 防止梯度下溢。

## 9.5 TF32

CUDA 路径开启：

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

TF32 是 NVIDIA Ampere 及以后 GPU 上的矩阵计算格式。它在很多深度学习任务中能用较小精度代价换取更高吞吐。

读者不需要手动管理每个算子。理解重点是：同样的 PyTorch 代码，在不同硬件和后端上可能有不同性能表现。

## 9.6 Flash Attention

`model.py` 中：

```python
self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
```

如果可用，attention 前向使用：

```python
torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)
```

Flash Attention 的价值在于更高效地计算 attention，尤其减少显存读写。对于长上下文，attention 的 `T x T` 结构会成为瓶颈，因此高效 attention 实现非常重要。

## 9.7 torch.compile

`train.py` 中：

```python
if compile:
    model = torch.compile(model)
```

`torch.compile` 会尝试编译模型图，减少 Python 开销并优化算子执行。它常在 GPU 上带来收益，但有首次编译成本。

调试、小模型、CPU 环境中，建议先设：

```sh
--compile=False
```

等链路稳定后再打开。

## 9.8 梯度累积

有效 token 数：

```text
gradient_accumulation_steps * world_size * batch_size * block_size
```

显存限制的是单个 micro batch，不一定限制有效 batch。通过梯度累积，可以用多个小 batch 模拟一个大 batch。

代价是每次 optimizer step 需要更多前向和反向，所以 wall-clock 时间会增加。

## 9.9 DDP

DDP 的基本思想：

```text
每个 GPU 一个进程
每个进程处理不同数据
反向传播后同步梯度
每个进程更新同样的模型参数
```

nanoGPT 通过 `torchrun` 启动。单机 4 卡示例：

```sh
torchrun --standalone --nproc_per_node=4 train.py
```

多机训练还需要指定 master 地址和节点 rank。

DDP 的瓶颈常常是通信。如果网络慢，多卡效率会很差。不是 GPU 数量翻倍，训练速度就一定翻倍。

## 9.10 MFU

`estimate_mfu` 估算 Model FLOPs Utilization。它把理论 FLOPs 和实际每秒迭代速度结合起来，估计 A100 峰值算力利用率。

MFU 的作用是比较不同配置或优化的相对效率。它不是模型质量指标，也不是跨硬件绝对公平的指标。

## 9.11 bench.py 的作用

`bench.py` 把训练核心计算抽出来，减少日志、checkpoint、评估等干扰。它适合回答：

- 当前模型前向/反向有多快。
- `torch.compile` 是否有效。
- 不同 dtype 性能差异如何。
- batch size 改变后吞吐如何变化。

性能实验应该尽量隔离变量。不要同时改模型大小、dtype、compile 和 batch size，否则很难判断是哪一个改变带来了效果。

## 9.12 性能实验记录模板

```text
硬件：
PyTorch 版本：
device：
dtype：
compile：
模型配置：
batch_size：
block_size：
gradient_accumulation_steps：
tokens_per_iter：
iter time：
MFU：
best val loss：
结论：
```

把训练质量和速度一起记录，才能避免只优化吞吐而损害模型效果。

## 9.13 性能排查顺序

如果训练慢，先确认设备：

```sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.backends.mps.is_available())"
```

然后检查日志里的 `time`、`mfu` 和 `tokens per iteration`。如果 `tokens per iteration` 很小，GPU 可能吃不饱。如果 `block_size` 很大，attention 可能成为瓶颈。如果每次评估耗时很久，可能是 `eval_iters` 太大。

如果显存不够，优先按这个顺序试：

```text
减小 batch_size
关闭不必要的 eval
减小 block_size
减小 n_layer / n_embd
使用更低精度
```

不要第一反应就改学习率。学习率影响收敛，通常不能解决显存问题。

## 本章小结

nanoGPT 的性能设计并不复杂，但覆盖了现代训练的关键主题：混合精度、编译、高效 attention、梯度累积和分布式同步。理解这些机制后，读者才能根据自己的硬件条件合理缩放模型。下一章会把这些配置变化放进实验框架里，避免凭感觉乱调。
