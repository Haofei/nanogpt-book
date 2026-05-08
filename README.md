# 从零读懂 nanoGPT

这是一个围绕 `nanoGPT` 源码写作的中文书稿仓库。仓库现在按“书”和“代码”分离组织，根目录不再作为 nanoGPT 原项目入口。

## 仓库结构

```text
book/
  README.md
  SUMMARY.md
  chapters/
  appendix/
  book.toml

code/
  nanogpt/
    README.md
    model.py
    train.py
    sample.py
    data/
    config/
```

- `book/`：中文书稿，主入口是 [book/README.md](book/README.md)。
- `code/nanogpt/`：原始 nanoGPT 源码副本，用于配合书稿阅读和实验。

## 阅读入口

从这里开始：

- [从零读懂 nanoGPT](book/README.md)
- [章节目录](book/SUMMARY.md)
- [源码目录](code/nanogpt/README.md)

## 当前版本

当前版本已经从目录大纲扩展为教程草稿。第 4 章 `model.py` 是源码精读样章，第 5-8 章覆盖训练、数据、训练实践和采样生成，第 9-11 章覆盖性能、实验设计和扩展方案。

运行书中的 nanoGPT 命令前，先进入源码目录：

```sh
cd code/nanogpt
```

例如最小训练链路：

```sh
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=1 --eval_iters=1 --batch_size=2 --block_size=8 --n_layer=1 --n_head=2 --n_embd=16
```

## 来源说明

`code/nanogpt/` 来自 Andrej Karpathy 的 nanoGPT 项目，本仓库在其基础上增加中文书稿和目录整理。原项目许可证保留在 [code/nanogpt/LICENSE](code/nanogpt/LICENSE)。
