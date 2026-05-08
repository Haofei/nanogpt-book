# nanoGPT Source Snapshot

This directory contains the nanoGPT source code used by the Chinese book in `../../book/`.

From the repository root, enter this directory before running nanoGPT commands:

```sh
cd code/nanogpt
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --max_iters=1 --eval_iters=1 --batch_size=2 --block_size=8 --n_layer=1 --n_head=2 --n_embd=16
```

The original project is Andrej Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT

The original license is preserved in `LICENSE`.
