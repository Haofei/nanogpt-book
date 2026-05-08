# 附录 A PyTorch 速查

## Tensor 形状

阅读 nanoGPT 时最常见的形状：

| 符号 | 含义 |
| --- | --- |
| `B` | batch size |
| `T` | sequence length |
| `C` | embedding dimension |
| `nh` | number of heads |
| `hs` | head size, usually `C // nh` |

## 常见操作

`view` 改变张量形状，要求内存布局兼容。`transpose` 交换维度。`contiguous` 让张量内存连续，常在 `transpose` 后配合 `view` 使用。

在 attention 中常见：

```python
k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
```

这会把 `B x T x C` 变成 `B x n_head x T x head_size`。

## ModuleList 与 ModuleDict

`ModuleList` 用于保存一组子模块，PyTorch 会正确注册参数。`ModuleDict` 用于按名字组织子模块。nanoGPT 用它们组织 Transformer：

```python
self.transformer = nn.ModuleDict(dict(
    wte=nn.Embedding(...),
    h=nn.ModuleList([...]),
))
```

## no_grad

评估和生成时常用 `torch.no_grad()`，因为不需要保存反向传播图，能减少显存和计算开销。

## autocast 与 GradScaler

`autocast` 让部分算子使用低精度执行。`GradScaler` 用于 float16 训练，减少梯度下溢风险。bfloat16 通常不需要 scaler。

