# 10 实验设计

## 本章目标

把 nanoGPT 从阅读对象变成实验平台。通过小实验理解模型规模、数据、上下文长度和采样参数的影响。

## 实验一：缩小模型

从 Shakespeare 字符级训练开始，固定数据和训练步数，只修改：

- `n_layer`
- `n_head`
- `n_embd`

观察参数量、训练速度、train loss 和 val loss。这个实验能帮助你建立模型容量和过拟合之间的直觉。

## 实验二：修改上下文长度

比较 `block_size=32`、`64`、`128`、`256`。上下文越长，模型能利用的信息越多，但 attention 成本也更高。对于字符级 Shakespeare，较长上下文通常能改善局部一致性。

## 实验三：学习率扫描

在同一配置下尝试几个学习率：

```text
1e-4, 3e-4, 6e-4, 1e-3
```

记录 loss 曲线。学习率过高时训练可能震荡或发散；过低时下降缓慢。

## 实验四：采样参数

训练同一个模型后，用不同参数采样：

```sh
python sample.py --out_dir=out-shakespeare-char --temperature=0.6 --top_k=50
python sample.py --out_dir=out-shakespeare-char --temperature=1.0 --top_k=200
python sample.py --out_dir=out-shakespeare-char --temperature=1.2 --top_k=500
```

比较输出的稳定性、多样性和错误率。

## 实验五：换自己的数据

复制一个 `data/shakespeare_char` 风格的数据目录，把 `input.txt` 换成自己的文本，再复用 prepare 逻辑生成 `train.bin`、`val.bin` 和 `meta.pkl`。

## 实验记录模板

每次实验建议记录：

```text
目标：
代码版本：
数据集：
配置变更：
硬件：
训练命令：
关键日志：
最好 val loss：
生成样例：
结论：
下一步：
```

## 小结

学习模型训练不能只看最终 loss。更重要的是建立实验纪律：一次只改少数变量，记录现象，再根据证据调整配置。

