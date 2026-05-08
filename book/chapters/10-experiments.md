# 10 实验设计

## 本章目标

本章把 nanoGPT 变成实验平台。重点不是“跑更多命令”，而是训练读者建立实验纪律：一次只改少数变量，记录现象，根据证据调整下一步。

## 10.1 为什么需要实验设计

语言模型训练结果受很多因素影响：

```text
数据
tokenizer
模型规模
上下文长度
batch
学习率
训练步数
采样参数
硬件后端
```

如果一次改动太多，即使结果变好，也不知道原因是什么。好的实验应该有明确假设、固定对照和可复现记录。

## 10.2 实验记录模板

每次实验建议写成：

```text
实验名称：
问题/假设：
代码提交：
数据集：
训练命令：
配置变更：
硬件：
开始时间：
结束时间：
最好 val loss：
采样命令：
生成样例：
观察：
结论：
下一步：
```

`代码提交` 很重要。没有版本记录，过几天就很难知道当时跑的到底是哪版代码。

## 10.3 读者需要补上的实验常识

实验不是把命令跑完就结束。一个有效实验至少包含：

```text
假设
对照
变量
指标
结论
```

假设是你预期会发生什么。对照是没有改动的 baseline。变量是本次唯一或主要改变的配置。指标是判断结果的证据。结论要说明假设是否成立。

例如“把 `block_size` 从 64 改到 128 会降低 val loss，但会增加 iter time”就是一个可测试假设。只写“试试更大的 block_size”不是一个好实验。

还要区分训练指标和生成指标。loss 更稳定、更容易比较；生成文本更直观，但受采样随机性影响。正式比较生成效果时，prompt、seed、temperature 和 top-k 都应该固定。

## 10.4 实验一：模型规模

目标：观察模型变大后 loss、速度和生成质量如何变化。

固定数据和训练步数，比较：

```text
small:  n_layer=2, n_head=2, n_embd=64
medium: n_layer=4, n_head=4, n_embd=128
large:  n_layer=6, n_head=6, n_embd=384
```

示例命令：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=500 --eval_iters=20 --n_layer=2 --n_head=2 --n_embd=64
```

观察：

- 参数量如何变化。
- 每次迭代时间如何变化。
- train loss 和 val loss 差距是否变大。
- 生成文本是否更像训练集风格。

## 10.5 实验二：上下文长度

目标：理解 `block_size` 的影响。

比较：

```text
block_size=32
block_size=64
block_size=128
block_size=256
```

上下文越长，模型能看到更长历史，但 attention 成本更高。字符级模型尤其依赖较长上下文来维持单词和句子结构。

记录每组：

```text
tokens_per_iter
iter time
best val loss
生成样例中的连贯性
```

## 10.6 实验三：学习率

目标：找到稳定且下降较快的学习率范围。

比较：

```text
1e-4
3e-4
6e-4
1e-3
```

学习率太低，loss 下降慢。学习率太高，loss 可能震荡、发散或出现 `nan`。

学习率实验要固定模型、数据、batch 和训练步数，否则结论不清楚。

## 10.7 实验四：采样参数

目标：理解 temperature 和 top-k 如何影响生成。

固定同一个 checkpoint 和 prompt：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="KING:" --temperature=0.6 --top_k=50
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="KING:" --temperature=0.8 --top_k=200
python sample.py --out_dir=out-shakespeare-char --device=cpu --start="KING:" --temperature=1.2 --top_k=500
```

比较：

- 重复程度。
- 语法稳定性。
- 新颖程度。
- 是否出现异常 token。

采样实验不改变模型，只改变解码策略。它能帮助读者区分“模型能力”和“采样策略”的影响。

## 10.8 实验五：换自己的数据

目标：验证 nanoGPT 是否能学习新文本风格。

步骤：

```text
复制 data/shakespeare_char
-> 替换 input.txt
-> 运行 prepare.py
-> 新建 config/train_my_dataset.py
-> 训练
-> 采样
```

注意事项：

- 数据太少容易过拟合。
- 文本清洗会影响生成质量。
- 如果使用字符级 tokenizer，特殊字符和换行都会进入词表。
- 如果换成 BPE，需要重新考虑 `meta.pkl` 和解码逻辑。

## 10.9 实验六：过拟合观察

目标：故意用小数据训练较大模型，观察 train loss 和 val loss 分离。

现象：

```text
train loss 持续下降
val loss 下降后上升
生成内容更像背诵训练集
```

处理方法：

- 增加数据。
- 减小模型。
- 增加 dropout。
- 提前停止训练。

这个实验能帮助读者理解为什么验证集重要。

## 10.10 实验七：checkpoint 恢复

目标：确认 checkpoint 不只是保存权重，而是能恢复训练状态。

步骤：

```text
训练到少量 iter 并保存 checkpoint
-> 记录 iter_num 和 best_val_loss
-> 使用 init_from=resume 继续训练
-> 检查日志是否从旧 iter 附近继续
```

这个实验能帮助读者理解为什么 checkpoint 里要保存 optimizer、config 和 iter_num。

## 10.11 实验八：tokenizer 影响

目标：比较字符级 Shakespeare 和 BPE Shakespeare 的差异。

观察：

```text
vocab_size
同一段文本的 token 数
训练速度
生成文本风格
sample.py 是否需要 meta.pkl
```

这个实验能说明 tokenizer 不是无关紧要的预处理步骤，而是模型定义的一部分。

## 10.12 实验结论怎么写

不要只写“效果更好”。应该写具体证据：

```text
block_size 从 64 增加到 128 后，best val loss 从 X 降到 Y，但 iter time 从 A 增加到 B。采样中长句结构更稳定，但训练成本增加。
```

好的结论应该同时包含收益、代价和下一步。

## 本章小结

实验设计是把源码学习转化为真实能力的关键。nanoGPT 足够小，适合做可控实验。每次实验都应该明确变量、记录命令和保留结论。
