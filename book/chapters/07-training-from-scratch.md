# 07 从零训练与微调

## 本章目标

把 `README.md` 中的命令转化成可解释的训练工作流，理解从零训练、恢复训练和微调的差异。

## 最小实验

先准备 Shakespeare 字符级数据：

```sh
python data/shakespeare_char/prepare.py
```

再用小配置训练：

```sh
python train.py config/train_shakespeare_char.py
```

如果没有 GPU，可以降低配置：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000
```

## 关键超参数

- `block_size`：上下文长度。越大，模型能看见越长历史，但 attention 成本更高。
- `batch_size`：每个 micro batch 的样本数。
- `gradient_accumulation_steps`：累积多少个 micro batch 再更新参数。
- `n_layer`、`n_head`、`n_embd`：决定模型容量。
- `learning_rate`：学习率过大容易发散，过小训练慢。
- `max_iters`：训练总步数。
- `dropout`：正则化强度，微调时通常比预训练更有用。

## 从零训练

`init_from='scratch'` 时，模型权重随机初始化。模型必须从数据中学习所有统计规律。这适合小数据实验和预训练复现。

## 恢复训练

`init_from='resume'` 会从 `out_dir/ckpt.pt` 恢复模型和优化器。恢复训练时，模型结构参数必须和 checkpoint 匹配，否则无法正确加载。

## 微调

`init_from='gpt2'` 或其他 GPT-2 变体时，模型先加载预训练权重，再在新数据上继续训练。微调通常需要更小学习率和更少训练步数。

## 如何判断训练是否正常

正常现象：

- train loss 和 val loss 总体下降。
- 早期 loss 下降较快，后期变慢。
- 小模型生成文本可能语法混乱，但逐渐出现训练集风格。

异常信号：

- loss 变成 `nan`。
- train loss 降低但 val loss 上升明显，可能过拟合。
- 训练极慢，可能设备配置或 `compile` 设置不合适。

## 小结

训练不是单个命令，而是一组约束的平衡：数据规模、模型大小、上下文长度、batch、学习率和设备能力共同决定结果。

