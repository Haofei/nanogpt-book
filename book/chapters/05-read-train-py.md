# 05 读懂 train.py

## 本章目标

`train.py` 是 nanoGPT 的训练系统。读完本章，读者应该能理解：

- 配置如何从默认值、配置文件和命令行合并。
- `get_batch` 如何从二进制 token 文件构造训练样本。
- `init_from` 三种路径分别适合什么场景。
- 梯度累积、混合精度、梯度裁剪和 AdamW 如何协作。
- checkpoint 保存了哪些内容，为什么不只保存模型权重。
- DDP 模式下每个进程如何分工。

对应源码位置：`train.py`。

## 5.1 train.py 的主线

`train.py` 可以分成七段：

```text
默认配置
-> 读取覆盖配置
-> 初始化设备和 DDP
-> 定义 get_batch
-> 初始化模型和优化器
-> 定义评估与学习率函数
-> 进入训练循环
```

阅读时不要从 while 循环直接开始。文件顶部的配置决定了后面几乎所有行为。

## 5.2 配置系统

文件开头定义大量变量：

```python
out_dir = 'out'
eval_interval = 2000
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
device = 'cuda'
compile = True
```

然后执行：

```python
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
```

这段代码的效果是：先收集所有可配置变量，再让 `configurator.py` 根据命令行或配置文件覆盖它们。

例如：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

执行顺序是：

```text
加载 train.py 默认配置
-> 执行 config/train_shakespeare_char.py 覆盖部分变量
-> 命令行参数再次覆盖 device 和 compile
```

这种配置方式很轻量，代价是变量都在脚本全局作用域里。读者调试时要清楚最终配置不是只来自一个文件。

## 5.3 DDP 初始化

源码通过环境变量判断是否是分布式运行：

```python
ddp = int(os.environ.get('RANK', -1)) != -1
```

普通运行时没有 `RANK`，`ddp=False`。用 `torchrun` 启动时，每个进程都会有自己的 `RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`。

DDP 路径中：

```python
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)
master_process = ddp_rank == 0
seed_offset = ddp_rank
```

每个进程绑定一张 GPU。只有 rank 0 是 master process，负责日志和 checkpoint。这样可以避免多个进程同时写同一个文件。

还有一行很关键：

```python
gradient_accumulation_steps //= ddp_world_size
```

如果有更多 GPU 并行训练，每个进程需要累积的 micro step 数可以减少，从而保持全局有效 batch 大小一致。

## 5.4 tokens_per_iter

源码打印：

```python
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
```

这是每次 optimizer step 看到的 token 数量。它比单独看 `batch_size` 更重要。

例如：

```text
gradient_accumulation_steps = 40
world_size = 1
batch_size = 12
block_size = 1024
tokens_per_iter = 491,520
```

语言模型训练通常用 token 数衡量训练规模。调参时，不要只问 batch size 多大，而要问每次更新包含多少 token。

## 5.5 get_batch：从连续 token 流构造样本

核心代码：

```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
```

`train.bin` 是一条很长的 token id 序列。`get_batch` 随机选 `batch_size` 个起点，每个起点截取一段长度为 `block_size` 的输入。

如果输入是：

```text
data[i : i+block_size]
```

目标就是：

```text
data[i+1 : i+1+block_size]
```

这正是 next-token prediction。

shape：

```text
x: B x T
y: B x T
```

其中 `T = block_size`。

如果在 CUDA 上训练，代码会使用 pinned memory 和 non-blocking transfer：

```python
x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
```

这有助于减少 CPU 到 GPU 数据传输的等待。

## 5.6 模型初始化三条路径

`init_from` 决定模型从哪里开始。

### scratch

```python
if init_from == 'scratch':
    model = GPT(gptconf)
```

这是从随机权重开始训练。适合小实验、预训练复现或完全新 tokenizer。

如果数据目录里有 `meta.pkl`，会读取 `vocab_size`。字符级 Shakespeare 数据就是这种情况。否则默认使用 GPT-2 词表大小附近的 `50304`。

### resume

```python
elif init_from == 'resume':
    checkpoint = torch.load(ckpt_path, map_location=device)
```

恢复训练时不仅加载模型权重，还会加载 optimizer 状态、当前迭代数和最佳验证 loss。

这很重要。AdamW 有动量状态，如果只加载模型权重，训练会像“拿着旧模型重新开始优化”，而不是严格接着上次训练。

### gpt2*

```python
elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, override_args)
```

这条路径用于加载 OpenAI GPT-2 权重。适合微调或评估预训练模型。

## 5.7 crop_block_size

加载模型后，代码检查：

```python
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
```

这允许用更短上下文训练或微调。例如 GPT-2 原生 `block_size=1024`，但小数据微调可能只想用 256。缩短上下文可以减少显存和计算。

## 5.8 混合精度和 autocast

训练前会构造上下文管理器：

```python
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

前向计算被包在：

```python
with ctx:
    logits, loss = model(X, Y)
```

这让 GPU 上的部分计算使用低精度，提升吞吐并降低显存占用。

如果是 `float16`，还会使用：

```python
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

`GradScaler` 用来降低梯度下溢风险。

## 5.9 estimate_loss：评估不是训练

评估函数：

```python
@torch.no_grad()
def estimate_loss():
    model.eval()
    ...
    model.train()
```

这里有三个重点：

- `torch.no_grad()`：不保存反向传播图。
- `model.eval()`：关闭 dropout 等训练行为。
- `model.train()`：评估结束后恢复训练模式。

评估会在 train 和 val split 上各采样多个 batch，取平均 loss。`eval_iters` 越大估计越稳定，但越耗时。

## 5.10 学习率调度

`get_lr` 实现 warmup + cosine decay：

```text
warmup 阶段：学习率线性升高
中间阶段：余弦衰减
结束后：保持 min_lr
```

训练一开始模型权重随机，直接使用大学习率可能不稳定。warmup 让优化过程先平缓启动。后期学习率降低，有助于收敛。

## 5.11 训练循环详解

训练循环最重要的片段是梯度累积：

```python
for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train')
    scaler.scale(loss).backward()
```

把 loss 除以 `gradient_accumulation_steps` 是为了让多个 micro batch 的梯度平均，而不是简单放大。

注意 `X, Y = get_batch('train')` 放在 backward 前。这是一个性能细节：当前 batch 前向完成后，尽快预取下一个 batch，让数据准备和 GPU 工作更好地重叠。

随后：

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

梯度裁剪防止梯度爆炸。`set_to_none=True` 通常比把梯度清零更高效。

## 5.12 Checkpoint 保存什么

checkpoint 内容：

```python
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
```

这里不只保存 `model`。因为真正恢复训练需要知道：

- 模型结构是什么。
- 优化器状态是什么。
- 训练到第几步。
- 最好验证 loss 是多少。
- 当时使用的配置是什么。

如果只保存权重，最多只能做推理或手动继续训练，不能完整恢复训练现场。

## 5.13 DDP 中的梯度同步优化

DDP 训练中，代码有一行：

```python
model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
```

含义是：只有最后一个 micro step 才同步梯度。前面的 micro step 只在本地累积。否则每个 micro step 都同步一次，会浪费通信成本。

这是一个短小但很重要的分布式训练优化点。

## 5.14 建议实验：跑一个 1 step 训练

为了理解训练循环，可以跑最小命令：

```sh
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=1 --eval_iters=1 --batch_size=2 --block_size=8 --n_layer=1 --n_head=2 --n_embd=16
```

观察输出：

- `tokens per iteration`
- `number of parameters`
- `step 0: train loss ..., val loss ...`
- `iter 0: loss ..., time ...`

这个命令不追求效果，只用来验证训练链路。

## 5.15 常见问题

为什么 CPU 上要加 `--compile=False`？

`torch.compile` 在小模型或 CPU 上可能首次编译开销很大，也可能因为环境不适配导致问题。调试时关闭更直接。

为什么 loss 一开始接近 `log(vocab_size)`？

随机模型对每个 token 的预测近似均匀，交叉熵约等于词表大小的自然对数。字符级 Shakespeare 词表约 65，因此随机 loss 大约在 `log(65)` 附近。

为什么 train loss 低于 val loss？

模型直接在 train split 上更新参数，更容易拟合训练数据。val loss 更能反映泛化。

## 本章小结

`train.py` 把训练系统压缩在一个脚本里。它的核心不是 while 循环本身，而是配置、数据、模型初始化、评估、优化和 checkpoint 之间的协作。读懂它以后，读者就能把 nanoGPT 从“能跑的代码”变成“可调试、可实验的训练平台”。

