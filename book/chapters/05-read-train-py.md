# 05 读懂 train.py

## 本章目标

理解 `train.py` 如何把配置、数据、模型、优化器、评估和 checkpoint 串成一个完整训练系统。

## 配置覆盖

`train.py` 先定义默认配置，再执行：

```python
exec(open('configurator.py').read())
```

这允许通过配置文件和命令行覆盖变量。例如：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

这种方式很轻量，但也意味着配置变量是脚本级全局变量。阅读时要从文件顶部开始追踪。

## DDP 初始化

代码通过环境变量判断是否运行在分布式模式：

```python
ddp = int(os.environ.get('RANK', -1)) != -1
```

如果是 DDP，每个进程绑定一张 GPU，并按 `WORLD_SIZE` 调整 `gradient_accumulation_steps`。这样全局 batch 规模保持一致。

## get_batch

`get_batch` 从 `train.bin` 或 `val.bin` 中随机截取连续 token：

```text
x = data[i : i + block_size]
y = data[i + 1 : i + 1 + block_size]
```

这正是语言模型的训练目标：给定当前位置及之前的 token，预测下一个 token。

## 模型初始化

`init_from` 有三种路径：

- `scratch`：从随机权重开始训练。
- `resume`：从 checkpoint 恢复训练。
- `gpt2*`：加载 GPT-2 系列预训练权重。

这三条路径覆盖了从零训练、断点续训和微调三类常见工作流。

## 训练循环

训练循环的关键步骤：

1. 根据 step 设置学习率。
2. 定期估计 train/val loss。
3. 保存 checkpoint。
4. 做多次 micro step 实现梯度累积。
5. 反向传播、梯度裁剪、optimizer step。
6. 打印 loss、耗时和 MFU。

梯度累积的核心代码是：

```python
loss = loss / gradient_accumulation_steps
scaler.scale(loss).backward()
```

这让多个小 batch 的梯度合并成一个更大的有效 batch。

## Checkpoint

checkpoint 保存：

- 模型权重。
- 优化器状态。
- 模型结构参数。
- 当前迭代数。
- 最佳验证 loss。
- 当前配置。

因此恢复训练时不只恢复权重，还能恢复优化器动量和训练进度。

## 小结

`train.py` 是一个压缩版训练系统。它没有复杂框架，但保留了训练 GPT 所需的大多数关键机制。

