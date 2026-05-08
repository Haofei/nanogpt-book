# 从零读懂 nanoGPT

这是一本围绕 `nanoGPT` 源码写作的中文技术书初版。目标不是把代码逐行翻译成自然语言，而是把一个小而完整的 GPT 训练项目拆成可学习、可运行、可改造的知识体系。

当前版本已经从目录大纲扩展为教程草稿：每章包含目标、源码位置、概念解释、关键代码路径、实验建议或常见问题。其中第 4 章 `model.py` 是源码精读样章，后续章节可以按同样粒度继续加深。

## 读者定位

本书面向三类读者：

- 已经会 Python 和一点 PyTorch，希望系统理解 GPT 训练流程的人。
- 想通过真实源码学习 Transformer、训练循环、tokenizer、采样和性能优化的工程师。
- 准备用 `nanoGPT` 做实验、改模型、换数据或复现实验结果的学习者。

## 阅读方式

建议按顺序阅读：

1. 先读第 00a-00c 章，把环境、PyTorch 最小基础和第一次运行打通。
2. 再读第 1-3 章，建立项目和 Transformer 的整体地图。
3. 接着读第 4-8 章，对应源码中的 `model.py`、`train.py`、数据脚本和 `sample.py`。
4. 最后读第 9-11 章，把项目扩展为自己的训练实验。

每章都遵循相同结构：本章目标、源码位置、核心概念、代码导读、实验建议、常见问题和小结。

本书有一条贯穿案例线：Shakespeare 字符级模型。前面三章先让你把这个最小实验跑起来；后面讲 `model.py`、`train.py`、数据、采样和实验设计时，都会回到这个案例，解释你已经运行过的命令背后到底发生了什么。

本书中的源码都位于仓库的 `code/nanogpt/` 目录。除非章节另有说明，所有训练和采样命令都默认先进入源码目录：

```sh
cd code/nanogpt
```

如果阅读过程中遇到 `step`、`batch`、`logits`、`checkpoint`、`CUDA`、`MPS`、`DDP` 等术语，可以先查 [附录 C 术语与背景知识](appendix/c-glossary.md)。

## 项目源码地图

| 文件或目录 | 本书章节 | 作用 |
| --- | --- | --- |
| `code/nanogpt/model.py` | 第 4 章 | GPT 模型、Attention、MLP、Block、参数初始化、生成 |
| `code/nanogpt/train.py` | 第 5 章 | 配置、数据读取、训练循环、评估、checkpoint、DDP |
| `code/nanogpt/sample.py` | 第 8 章 | 加载模型、编码 prompt、自回归生成 |
| `code/nanogpt/data/*/prepare.py` | 第 6 章 | 数据下载、tokenization、二进制训练文件 |
| `code/nanogpt/config/*.py` | 第 7 章 | 不同训练/微调/评估配置 |
| `code/nanogpt/bench.py` | 第 9 章 | 简化性能测试入口 |

## 目录

- [00 前言](chapters/00-preface.md)
- [00a 环境准备与项目运行](chapters/00a-environment-setup.md)
- [00b PyTorch 和张量最小基础](chapters/00b-pytorch-tensor-basics.md)
- [00c 第一次跑通 nanoGPT](chapters/00c-first-run-walkthrough.md)
- [01 nanoGPT 是什么](chapters/01-what-is-nanogpt.md)
- [02 Transformer 必备知识](chapters/02-transformer-basics.md)
- [03 GPT 架构总览](chapters/03-gpt-architecture.md)
- [04 读懂 model.py](chapters/04-read-model-py.md)
- [05 读懂 train.py](chapters/05-read-train-py.md)
- [06 数据与 Tokenization](chapters/06-data-and-tokenization.md)
- [07 从零训练与微调](chapters/07-training-from-scratch.md)
- [08 采样与文本生成](chapters/08-sampling-and-generation.md)
- [09 性能、显存与分布式训练](chapters/09-performance.md)
- [10 实验设计](chapters/10-experiments.md)
- [11 如何扩展 nanoGPT](chapters/11-extend-nanogpt.md)
- [附录 A PyTorch 速查](appendix/a-pytorch.md)
- [附录 B 数学符号速查](appendix/b-math.md)
- [附录 C 术语与背景知识](appendix/c-glossary.md)

## 写作原则

- 每个概念都尽量落到源码中的具体变量、shape 或命令。
- 每个工程选择都回答两个问题：为什么这么做，以及改错会发生什么。
- 先帮助读者跑通最小实验，再讨论大规模训练。
- 不把 `nanoGPT` 神化。它是一个非常适合学习的最小工程样本，但不是现代 LLM 训练系统的全部。
