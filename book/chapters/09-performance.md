# 09 性能、显存与分布式训练

## 本章目标

理解 nanoGPT 中和性能相关的主要设计：混合精度、TF32、Flash Attention、梯度累积、DDP、`torch.compile` 和 MFU。

## 混合精度

`train.py` 根据设备能力选择 `bfloat16` 或 `float16`，并用 autocast 包裹前向计算：

```python
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```

混合精度能减少显存占用并提升吞吐，但需要注意数值稳定性。`float16` 路径还会使用 `GradScaler`。

## TF32

代码开启：

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

TF32 是 NVIDIA Ampere 及之后 GPU 上常见的加速选项，通常能在可接受精度下提高矩阵乘法速度。

## Flash Attention

`model.py` 会检测 PyTorch 是否有 `scaled_dot_product_attention`。如果有，就使用更高效的 attention 实现。attention 的显存压力通常随 `T x T` 增长，因此长上下文训练时这类优化非常重要。

## 梯度累积

当显存放不下大 batch 时，可以用多个 micro batch 累积梯度。有效 token 数为：

```text
gradient_accumulation_steps * world_size * batch_size * block_size
```

这也是 `train.py` 打印 `tokens per iteration` 的来源。

## DDP

DDP 通过多个进程并行训练。每个进程处理不同 batch，反向传播时同步梯度。`nanoGPT` 还在 micro step 中控制 `require_backward_grad_sync`，只在最后一次 micro step 同步，避免不必要通信。

## torch.compile

`torch.compile(model)` 会尝试编译模型图，提高运行速度。它可能带来首次编译开销，也可能因为环境问题失败。因此 CPU 或调试环境中常用 `--compile=False`。

## MFU

`estimate_mfu` 估算模型实际 FLOPS 与 A100 理论峰值的比例。它不是绝对精确的性能指标，但能帮助比较不同配置的吞吐效率。

## 小结

性能优化不是单一技巧，而是由数据读取、矩阵计算、attention 实现、精度、batch 组织和多卡通信共同决定。

