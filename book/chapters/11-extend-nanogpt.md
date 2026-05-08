# 11 如何扩展 nanoGPT

## 本章目标

本章讨论如何在保持项目清晰的前提下扩展 nanoGPT。读者应该能判断哪些改动应该放在数据目录、配置文件、模型文件或独立脚本中，以及修改后会影响哪些训练和 checkpoint 行为。

## 11.1 扩展的原则

nanoGPT 的价值在于小而完整。扩展时最重要的是保持边界清晰：

```text
数据处理放 data/
实验参数放 config/
模型结构放 model.py
训练流程放 train.py
评估工具优先独立成脚本
```

不要把所有开关都堆进 `train.py`。一旦训练脚本变成复杂框架，nanoGPT 的可读性就会下降。

## 11.2 增加新数据集

推荐结构：

```text
data/my_dataset/
  input.txt
  prepare.py
  readme.md
```

`prepare.py` 的职责：

```text
读取或下载原始数据
-> 清洗文本
-> 切分 train/val
-> tokenization
-> 写 train.bin/val.bin
-> 写 meta.pkl
```

`readme.md` 应该说明数据来源、清洗规则、tokenizer 和训练命令。

训练时：

```sh
python train.py --dataset=my_dataset
```

## 11.3 增加新配置

配置文件应该只写相对默认值的差异。例如：

```python
out_dir = 'out-my-dataset'
dataset = 'my_dataset'
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
max_iters = 5000
```

不要把 `train.py` 的全部默认配置复制一遍。只保留实验真正改变的变量，后续读者才能快速看懂配置意图。

## 11.4 修改模型结构

常见扩展：

- 把绝对位置 embedding 改成 RoPE。
- 把 LayerNorm 改成 RMSNorm。
- 修改 MLP 激活函数。
- 增加 SwiGLU。
- 替换 attention 实现。
- 加入新的 dropout 或正则化。

修改 `model.py` 时要特别注意：

```text
参数名是否变化
参数 shape 是否变化
checkpoint 是否还能加载
from_pretrained 是否还能工作
```

如果参数结构变了，旧 checkpoint 可能无法直接恢复。需要在文档中说明兼容性。

## 11.5 增加 RoPE 的思路

绝对位置 embedding 当前来自：

```python
wpe = nn.Embedding(config.block_size, config.n_embd)
```

RoPE 会把位置信息注入 Q/K，而不是加到 token embedding 上。改动大致涉及：

```text
移除或绕过 wpe
-> 在 attention 中对 q/k 应用旋转位置编码
-> 检查 GPT-2 权重加载兼容性
```

这类改动不建议初学者一开始做。它会影响模型定义、权重加载和训练行为。

## 11.6 增加评估脚本

当前训练主要看 train/val loss。可以新增：

```text
eval_perplexity.py
eval_prompts.py
eval_generation_samples.py
```

建议先独立脚本实现，而不是立刻塞进训练循环。

例如固定 prompt 评估可以记录：

```text
prompt
temperature
top_k
seed
generated text
checkpoint path
```

这样每次训练后都能比较同一组 prompt 的生成变化。

## 11.7 增加日志

`train.py` 已有 wandb 支持：

```python
wandb_log = False
```

可以扩展记录：

- tokens per second。
- gradient norm。
- parameter norm。
- best checkpoint path。
- 固定 prompt 的采样样例。

但日志不是越多越好。每个指标都应该能回答一个实验问题。

## 11.8 增加测试

nanoGPT 原项目很轻量，没有完整测试套件。如果要长期扩展，可以先加小测试：

```text
测试 GPT.forward 输出 shape
测试 generate 输出长度
测试 prepare.py 能生成 train.bin/val.bin/meta.pkl
测试 checkpoint 能保存和加载
```

这些测试不需要训练出好模型，只需要验证代码链路不坏。

## 11.9 扩展时的风险清单

每次改动前问：

```text
是否改变 tokenizer？
是否改变 vocab_size？
是否改变 block_size？
是否改变参数名或 shape？
是否影响 checkpoint resume？
是否影响 GPT-2 from_pretrained？
是否影响 sample.py 解码？
```

这些问题能提前发现大多数破坏性变更。

## 11.10 一个推荐扩展示例

最适合初学者的扩展不是改 attention，而是增加自己的数据集和配置。

步骤：

```text
新增 data/my_dataset/prepare.py
新增 data/my_dataset/readme.md
新增 config/train_my_dataset.py
运行 prepare
运行训练
运行采样
记录实验
```

这条路径能练到数据、配置、训练和采样，但不会过早破坏模型结构。

## 本章小结

扩展 nanoGPT 的关键是控制复杂度。先扩数据和配置，再扩评估和日志，最后再改模型结构。每次改动都要明确它影响的是数据、模型、训练、采样还是 checkpoint。

