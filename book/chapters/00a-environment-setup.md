# 00a 环境准备与项目运行

## 本章目标

本章面向第一次接触深度学习项目的读者。读完后，你应该知道：

- 这个仓库的书稿和源码分别在哪里。
- 为什么建议使用虚拟环境。
- 如何安装 nanoGPT 需要的依赖。
- 如何判断当前机器能用 CPU、Apple MPS 还是 NVIDIA CUDA。
- 常见安装问题应该先检查什么。

本章不要求你已经会 PyTorch。这里只先把项目跑起来。

## 00a.1 先确认仓库结构

本仓库不是原始 nanoGPT 仓库，而是一本书稿仓库。目录分成两块：

```text
book/
  中文书稿

code/nanogpt/
  nanoGPT 源码
```

阅读书稿时，你通常在仓库根目录看 `book/`。运行 nanoGPT 代码时，要先进入源码目录：

```sh
cd code/nanogpt
```

如果你在仓库根目录直接运行：

```sh
python train.py
```

会找不到文件，因为 `train.py` 已经放在 `code/nanogpt/` 下面。

## 00a.2 Python 版本

建议使用 Python 3.10 或 3.11。先查看版本：

```sh
python --version
```

有些系统里命令叫 `python3`：

```sh
python3 --version
```

如果你看到的是 Python 2，或者系统找不到 `python`，需要先安装现代 Python。macOS 可以用 Homebrew，Windows 可以用 Python 官网安装包，Linux 通常可以用系统包管理器。

本书后面统一写 `python`。如果你的机器上只能用 `python3`，把命令里的 `python` 替换成 `python3` 即可。

## 00a.3 为什么要用虚拟环境

Python 项目的依赖很多。不同项目可能需要不同版本的包。如果全部装到系统 Python 里，时间久了容易互相污染。

虚拟环境可以给当前项目单独创建一套依赖目录。

在仓库根目录执行：

```sh
python -m venv .venv
```

macOS 或 Linux 激活：

```sh
source .venv/bin/activate
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

激活后，命令行前面通常会出现 `(.venv)`。这表示你当前使用的是项目虚拟环境。

升级 pip：

```sh
python -m pip install --upgrade pip
```

## 00a.4 安装依赖

nanoGPT 的基础依赖：

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

这些包大致负责：

- `torch`：PyTorch，模型、张量、训练都依赖它。
- `numpy`：数据处理和二进制数组读写。
- `transformers`：加载 GPT-2 预训练权重。
- `datasets`：下载和处理 OpenWebText 等数据集。
- `tiktoken`：GPT-2 BPE tokenizer。
- `wandb`：可选实验日志。
- `tqdm`：进度条。

入门阶段不需要马上配置 wandb。即使安装了它，默认训练配置里也不会强制使用。

## 00a.5 选择 CPU、MPS 或 CUDA

训练命令里常见：

```sh
--device=cpu
--device=mps
--device=cuda
```

三者含义：

- `cpu`：使用 CPU。最稳，但慢。
- `mps`：Apple Silicon Mac 的 GPU 后端。
- `cuda`：NVIDIA GPU 后端。

检查 PyTorch 是否能看到 CUDA 或 MPS：

```sh
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("mps:", torch.backends.mps.is_available())
PY
```

如果 `cuda: True`，可以尝试 CUDA。  
如果 `mps: True`，Apple Silicon 可以尝试 MPS。  
如果都不是 True，就先用 CPU。

入门读者建议第一轮用 CPU 最小配置跑通。CPU 慢，但错误更容易排查。

## 00a.6 安装是否成功

进入源码目录：

```sh
cd code/nanogpt
```

检查能否导入依赖：

```sh
python - <<'PY'
import torch
import numpy
import tiktoken
print("ok")
PY
```

如果输出：

```text
ok
```

说明最核心依赖可用。

## 00a.7 常见安装问题

`ModuleNotFoundError`

说明某个包没装到当前 Python 环境里。先确认虚拟环境已激活，再重新安装依赖。

`pip install torch` 很慢

PyTorch 包较大，网络慢时需要等待。也可以去 PyTorch 官网选择与你系统匹配的安装命令。

`torch.cuda.is_available()` 是 False

可能原因很多：没有 NVIDIA GPU、驱动没装好、安装的是 CPU 版 PyTorch、CUDA 版本不匹配。入门阶段先用 `--device=cpu` 跑通，不要把第一天都耗在 GPU 环境上。

`torch.backends.mps.is_available()` 是 False

只有 Apple Silicon Mac 和合适版本的 PyTorch 才支持 MPS。如果不可用，先用 CPU。

## 00a.8 本章检查点

继续下一章前，确认你能做到：

```text
知道 book/ 和 code/nanogpt/ 的区别
能进入 code/nanogpt/
能创建并激活虚拟环境
能安装基础依赖
能运行 import torch / import tiktoken
知道自己的机器该先用 cpu、mps 还是 cuda
```

如果这些都完成了，就可以进入 PyTorch 和张量最小基础。

