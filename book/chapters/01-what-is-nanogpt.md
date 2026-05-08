# 01 nanoGPT 是什么

## 本章目标

本章建立 nanoGPT 的项目地图。读者应该知道每个主要文件负责什么、训练和生成的完整链路是什么，以及应该按什么顺序进入源码。

## 1.1 项目定位

`nanoGPT` 是一个用于训练和微调 GPT 类语言模型的极简 PyTorch 项目。它不是只为教学而写的伪代码，因为它包含真实训练中会遇到的功能：

- GPT 模型定义。
- GPT-2 权重加载。
- 数据准备脚本。
- 单卡训练。
- 分布式训练。
- 混合精度。
- checkpoint。
- 文本采样。
- wandb 日志。

但它也不是完整工业系统。它没有复杂数据平台，没有模型服务框架，没有在线评估系统，也没有大规模容错机制。

它最适合回答：

```text
如果把 GPT 训练系统压缩到最少文件，核心代码会长什么样？
```

## 1.2 仓库结构

主要文件：

```text
code/nanogpt/model.py
code/nanogpt/train.py
code/nanogpt/sample.py
code/nanogpt/configurator.py
code/nanogpt/bench.py
code/nanogpt/config/
code/nanogpt/data/
code/nanogpt/assets/
```

`code/nanogpt/model.py` 定义 GPT。里面有 LayerNorm、CausalSelfAttention、MLP、Block、GPTConfig 和 GPT。

`code/nanogpt/train.py` 是训练入口。它负责读取配置、构造 batch、初始化模型、运行训练循环、评估 loss 和保存 checkpoint。

`code/nanogpt/sample.py` 是生成入口。它负责加载模型、编码 prompt、调用 `generate` 并解码输出。

`configurator.py` 负责让配置文件和命令行参数覆盖默认变量。

`bench.py` 用来做性能基准，减少训练脚本中评估和 checkpoint 的干扰。

`code/nanogpt/config/` 保存不同任务的配置，例如训练 Shakespeare、微调 Shakespeare、评估 GPT-2。

`code/nanogpt/data/` 保存数据准备脚本。

## 1.3 训练链路

训练链路可以写成：

```text
原始文本
-> prepare.py
-> train.bin / val.bin
-> train.py 的 get_batch
-> x/y batch
-> GPT.forward
-> logits 和 loss
-> backward
-> AdamW step
-> checkpoint
```

其中最容易忽略的是 `x/y batch` 的关系：

```text
x = token[i : i + block_size]
y = token[i + 1 : i + 1 + block_size]
```

模型不是“理解整段文本后写作文”，而是在每个位置预测下一个 token。

## 1.4 生成链路

生成链路：

```text
prompt
-> tokenizer encode
-> token ids
-> GPT.generate
-> next token
-> 拼回上下文
-> 重复
-> tokenizer decode
```

这说明语言模型生成文本并不是一次性输出完整段落，而是循环执行 next-token prediction。

## 1.5 四条学习主线

读 nanoGPT 时可以抓四条线。

第一条是数据线：

```text
data/*/prepare.py -> train.bin/val.bin -> get_batch
```

第二条是模型线：

```text
GPTConfig -> GPT -> Block -> Attention/MLP -> logits
```

第三条是训练线：

```text
config -> optimizer -> loss -> backward -> checkpoint
```

第四条是生成线：

```text
sample.py -> encode -> generate -> decode
```

这四条线交叉在一起，就是完整项目。

## 1.6 建议阅读顺序

不要一开始就从 `train.py` 第 1 行读到最后。更好的顺序是：

1. 根目录 `README.md` 和 `book/README.md`。
2. `code/nanogpt/config/train_shakespeare_char.py`。
3. `code/nanogpt/data/shakespeare_char/prepare.py`。
4. `code/nanogpt/model.py` 的 `GPT.forward`。
5. `code/nanogpt/train.py` 的 `get_batch`。
6. `code/nanogpt/train.py` 的训练循环。
7. `code/nanogpt/sample.py` 和 `GPT.generate`。

这个顺序符合数据实际流动。

## 1.7 nanoGPT 的学习价值

nanoGPT 的代码少，所以每个设计都暴露在读者面前。读者能直接看到：

- 为什么 checkpoint 需要保存 optimizer。
- 为什么 DDP 中只有 master process 写文件。
- 为什么生成时只取最后一个 token 的 logits。
- 为什么 `block_size` 影响 position embedding 和 attention。
- 为什么 tokenizer 和 checkpoint 必须匹配。

这些知识很难只靠架构图学会。

## 本章小结

nanoGPT 是一个小型 GPT 训练系统。它的核心价值是让读者在一个仓库中看到数据、模型、训练、生成四条链路如何闭合。下一章会补齐阅读模型源码前必须知道的 Transformer 基础。
