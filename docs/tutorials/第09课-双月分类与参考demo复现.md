# 第 09 课：双月分类与参考 demo 复现

第 08 课已经走通“前向 → 清梯度 → 反传 → 更新”。本课把同一流程用到非线性分类：使用自己的 `MLP` 区分两组弯月形数据，最后画出决策边界。

参考本机的 [demo.ipynb](/data/ghs/projects/micrograd/demo.ipynb)。先做全量训练，不加入小批量采样。两个测试检查损失的手算结果、数据和网络结构，不把随机训练的准确率写成硬性通过条件。

## 1. 本课创建哪些文件，使用哪些依赖

由你手动创建：

| 文件 | 内容 |
| --- | --- |
| `/data/ghs/projects/tianyi-micrograd/examples/train_moons.py` | 数据、分类损失、训练和绘图 |
| `/data/ghs/projects/tianyi-micrograd/test/tianyi_micrograd/test_moons.py` | 两个测试 |

沿用第 08 课创建的空文件 `examples/__init__.py`。模型继续使用现有 `nn.py`，引擎和工程配置不需要修改。

| 工具 | 本课用途 |
| --- | --- |
| 自己的 `Value`、`MLP` | 前向、损失、反向传播、参数更新 |
| 标准库 `random` | 固定模型初始化 |
| scikit-learn 的 `make_moons` | 生成双月数据 |
| NumPy | 整理绘图网格 |
| Matplotlib | 保存决策边界图片 |

PyTorch 不参与本课。外部库只负责数据与展示，不替代自己的模型计算。

编写教程时，`agent` 环境已有 NumPy、scikit-learn、Matplotlib。若你更换环境，可以先查看安装情况；缺少时再由你手动安装演示依赖，不需要把它们加入核心运行依赖：

```bash
conda activate agent
cd /data/ghs/projects/tianyi-micrograd
python -m pip show numpy scikit-learn matplotlib
```

确有缺失时，可手动执行 `python -m pip install numpy scikit-learn matplotlib`。包的安装名是 `scikit-learn`，Python 导入名是 `sklearn`。

## 2. 输出是分数，分类看正负

本课使用 `MLP(2,[16,16,1])`：输入是二维坐标，两个隐藏层各有 16 个 ReLU 神经元，最后输出一个线性分数 `score`。

```text
score > 0  →  预测标签 +1
score <= 0 →  预测标签 -1
```

这里的分数不是概率。最后一层关闭 ReLU，才能同时输出正负值。参数总数为：

```text
(2+1)*16 + (16+1)*16 + (16+1)*1 = 48+272+17 = 337
```

数据集原来的标签是 `0/1`，需要转换为 `-1/+1`，才能使用下面的分类损失。

## 3. 第一步：准备固定数据和模型

手动创建 `train_moons.py`，先写入：

```python
import random
from pathlib import Path

from tianyi_micrograd.nn import MLP


def make_data():
    from sklearn.datasets import make_moons

    xs, ys = make_moons(n_samples=100, noise=0.1, random_state=1337)
    return xs.tolist(), (2 * ys - 1).tolist()


def make_model():
    random.seed(1337)
    return MLP(2, [16, 16, 1])
```

`random_state=1337` 固定数据生成，`random.seed(1337)` 固定模型初始化。这是两个不同的随机数来源，不能只设置其中一个。

`.tolist()` 将数据转为普通 Python 数值列表。每一行是 `[x0,x1]`，可以直接交给自己的模型；`Value` 的混合运算会包装常量，不需要再为所有输入显式创建节点。

把数据库导入放在 `make_data()` 内，让后面的损失函数测试可以在没有 scikit-learn 时独立运行。绘图库也只在绘图函数里导入。

## 4. 第二步：理解间隔损失与 L2

单个样本使用间隔损失，也称 hinge loss：

```text
sample_loss = max(0, 1-y*score)
```

当 `y*score>=1`，分类方向正确且间隔足够，样本损失为零；当它小于 1，损失会推动分数朝正确方向移动。

| y | score | y*score | 样本损失 | 是否分类正确 |
| ---: | ---: | ---: | ---: | --- |
| +1 | 2 | 2 | 0 | 是 |
| +1 | 0.2 | 0.2 | 0.8 | 是 |
| -1 | 1 | -1 | 2 | 否 |

第二行说明：准确率已经正确，损失仍可能继续下降。准确率只看正负，间隔损失还关心离分界线有多远。

所有样本取平均，再加入 L2 正则：

```text
data_loss = 平均样本损失
reg_loss = alpha * Σ(parameter**2)
total_loss = data_loss + reg_loss
```

本课和参考 demo 一样，对全部参数（包括偏置）做正则，取 `alpha=1e-4`。它为参数梯度增加 `2*alpha*parameter`，抑制参数值过大；这里没有额外的 `1/2` 系数。

ReLU 在输入为零时沿用项目约定，梯度取零。因此样本恰好达到间隔 `y*score=1` 时，hinge 这一项也取零梯度。

## 5. 第三步：写损失和训练准确率

在同一个文件顶层追加：

```python
def loss_and_accuracy(model, xs, ys, alpha=1e-4):
    scores = [model(x) for x in xs]
    losses = [(1 - y * score).relu() for y, score in zip(ys, scores)]
    data_loss = sum(losses) / len(losses)
    reg_loss = alpha * sum(p * p for p in model.parameters())

    correct = sum((y > 0) == (score.data > 0) for y, score in zip(ys, scores))
    return data_loss + reg_loss, correct / len(ys)
```

本课约定 `xs`、`ys` 非空、长度相等，每个输入有两个数，标签为 `-1/+1`。不要依赖 `zip()` 静默截断错误长度。

损失计算全程保留 `Value`，让数据项与正则项一起反传。准确率只是统计指标，可以读取 `score.data`；不对准确率本身求导。

Python 中 `True`、`False` 在求和时分别相当于 `1`、`0`，因此 `sum(...)` 能统计分类正确的样本数。对零分数，当前规则归入负类。

## 6. 第四步：完成全量训练循环

继续在文件顶层追加：

```python
def train(steps=100):
    xs, ys = make_data()
    model = make_model()
    history = []

    for step in range(steps):
        total_loss, accuracy = loss_and_accuracy(model, xs, ys)
        history.append((total_loss.data, accuracy))

        model.zero_grad()
        total_loss.backward()
        learning_rate = 1.0 - 0.9 * step / steps
        for p in model.parameters():
            p.data -= learning_rate * p.grad

        if step % 10 == 0 or step == steps - 1:
            print(f"before update {step:03d}: loss={total_loss.data:.6f}, "
                  f"train_acc={accuracy:.1%}, lr={learning_rate:.3f}")

    final_loss, final_accuracy = loss_and_accuracy(model, xs, ys)
    history.append((final_loss.data, final_accuracy))
    print(f"after {steps} updates: loss={final_loss.data:.6f}, "
          f"train_acc={final_accuracy:.1%}")
    return model, xs, ys, history
```

约定 `steps` 为正整数。默认 100 步时，学习率从 `1.0` 逐渐降到最后一步的 `0.109`；公式趋近 `0.1`，最后一次并非恰好取 `0.1`。

这和参考 demo 的默认学习率公式一致。每一步都使用全部 100 个样本，因此这里是全量梯度下降，没有随机选小批量的过程。

日志里的 `before update` 使用本次更新前的损失和准确率，结尾重新前向得到最终指标。`history[0]`、`history[-1]` 分别对应初始模型、最终模型，总共 101 条记录。

这个标量引擎会为每次加乘创建节点，比 Tensor 库慢很多。先保持默认小数据和网络，不急着增加样本、层宽或绘图网格密度。

## 7. 第五步：画出决策边界

在同一文件顶层追加：

```python
def plot_boundary(model, xs, ys, path="outputs/moons_boundary.png"):
    import numpy as np
    import matplotlib.pyplot as plt

    points = np.asarray(xs)
    h = 0.25
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    predicted = np.array([model(row.tolist()).data > 0 for row in grid])
    predicted = predicted.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, predicted, levels=[-0.5, 0.5, 1.5],
                cmap="Spectral", alpha=0.6)
    ax.scatter(points[:, 0], points[:, 1], c=[int(y > 0) for y in ys],
               cmap="Spectral", vmin=0, vmax=1, edgecolors="k", s=30)
    ax.set(xlabel="x0", ylabel="x1", title="Training data and decision boundary")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output
```

逐段理解：

1. `points[:,0]`、`points[:,1]` 分别取横纵坐标，确定覆盖范围。
2. `meshgrid()` 生成规则网格，`ravel()` 把坐标矩阵展开，`column_stack()` 将同一位置的横纵坐标配对。
3. 每个网格点交给自己的模型，分数大于零记为正类。
4. `reshape(xx.shape)` 把预测恢复成网格形状，供 `contourf()` 画背景区域。
5. 散点表示训练数据，颜色映射同样使用 `0/1`，与背景保持一致。

这里不调用 `backward()`，也不更新参数。当前引擎没有 `no_grad()` 模式，预测仍会建立临时图；列表只保存布尔结果，避免一直持有全部网格点的输出图。

`mkdir(...,exist_ok=True)` 在你运行时创建输出目录；`close()` 释放绘图资源，不弹出窗口。

## 8. 第六步：添加入口，手动运行

在文件末尾追加：

```python
def main():
    model, xs, ys, history = train()
    print(f"parameters: {len(model.parameters())}")
    print(f"initial: loss={history[0][0]:.6f}, train_acc={history[0][1]:.1%}")
    print(f"final:   loss={history[-1][0]:.6f}, train_acc={history[-1][1]:.1%}")
    output = plot_boundary(model, xs, ys)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
```

在服务器上使用不依赖图形桌面的后端：

```bash
MPLBACKEND=Agg python -m examples.train_moons
```

运行后应报告 337 个参数，并生成 `/data/ghs/projects/tianyi-micrograd/outputs/moons_boundary.png`。观察初始与最终损失、准确率及弯曲的决策边界，不要求日志逐字复现参考 notebook。

请在自己的学习记录里填入实际结果：

| 项目 | 记录内容 |
| --- | --- |
| 数据 | 100 个样本，噪声 0.1，随机状态 1337 |
| 模型 | `MLP(2,[16,16,1])`，初始化种子 1337 |
| 训练 | 100 步，全量数据，`alpha=1e-4`，递减学习率 |
| 初始与最终 | 实际的总损失、训练准确率 |
| 图像 | 决策边界文件路径及观察 |

这里没有单独留出测试集，打印的都是**训练集准确率**。它说明模型对已见数据的拟合，不能据此断言对新数据的效果。

## 9. 第七步：保持两个简洁测试

手动创建 `test_moons.py`，写入：

```python
from math import isclose

import pytest

from examples.train_moons import loss_and_accuracy, make_data, make_model
from tianyi_micrograd.engine import Value
from tianyi_micrograd.nn import Neuron


def test_classification_loss():
    model = Neuron(1, nonlin=False)
    model.w[0].data = 1.0
    model.b.data = 0.0
    loss, accuracy = loss_and_accuracy(model, [[2.0], [1.0]], [1, -1], alpha=0.1)
    loss.backward()

    assert isclose(loss.data, 1.1)
    assert accuracy == 0.5
    assert isclose(model.w[0].grad, 0.7)
    assert isclose(model.b.grad, 0.5)


def test_moons_data_and_model():
    pytest.importorskip("sklearn")
    xs, ys = make_data()
    assert len(xs) == len(ys) == 100
    assert all(len(x) == 2 for x in xs)
    assert set(ys) == {-1, 1}

    model = make_model()
    assert len(model.parameters()) == 337
    assert isinstance(model(xs[0]), Value)
```

第一个测试故意使用一个可手算的线性神经元。`loss_and_accuracy()` 只需要模型可调用且有 `parameters()`，不要求一定是 MLP：

```text
两个分数为 2、1，标签为 +1、-1
两个样本损失为 0、2，平均为 1
正则为 0.1*(1**2+0**2)=0.1，总损失为 1.1

第二个样本贡献 dw=1/2、db=1/2
正则贡献 dw=0.2、db=0
所以 dw=0.7、db=0.5
```

第二个测试检查演示数据和完整网络的接口。没有 scikit-learn 时只跳过它，损失测试仍可执行。测试不运行完整训练，也不生成图片。

```bash
python -m pytest test/tianyi_micrograd/test_moons.py -q
python -m pytest test -q
```

演示依赖齐全时，新文件应为 `2 passed`；缺少 scikit-learn 时应为 `1 passed, 1 skipped`。绘图依赖是否齐全由前面的环境检查及实际绘图确认。

## 10. 与参考 demo 对齐什么

| 内容 | 本课做法 |
| --- | --- |
| 数据和网络 | 双月、100 个样本、噪声 0.1、两层 16 宽隐藏层 |
| 标签和损失 | `-1/+1`、平均 hinge loss、全部参数的 L2 |
| 更新 | 清参数梯度、反传、更新 `.data`，默认学习率公式一致 |
| 数据输入 | 使用普通浮点列表，由引擎包装常量 |
| 演示形式 | 普通 Python 模块，保存图片，不要求 notebook |
| 暂不加入 | 小批量采样、独立测试集、自动优化器 |

对齐的是计算逻辑与可解释的行为。初始化、依赖版本、浮点累加顺序等都可能影响具体训练轨迹，不把某个固定准确率当作引擎正确性的证明。

## 11. 常见问题与验收

| 现象 | 优先检查 |
| --- | --- |
| 负类几乎不参与学习 | 标签是否仍为 0/1，最后一层是否启用 ReLU |
| 损失是 float 或参数没有梯度 | 数据项、正则项是否提前取了 `.data` |
| 参数数目不是 337 | 每层宽度、偏置、参数收集是否正确 |
| 准确率与边界颜色对不上 | 正负类阈值、散点与背景是否用同一颜色映射 |
| 同一脚本每次结果不同 | 数据随机状态与模型随机种子是否都固定 |
| 图像保存时出现桌面相关错误 | 是否使用 `MPLBACKEND=Agg` |

验收时解释：为何正确分类后仍有损失；L2 怎样改变参数梯度；为什么准确率可以取 `.data` 而损失不可以；为什么这里必须称为训练集准确率。

完成后，自动求导、模型、训练和分类演示已经连成一条主线。进入 [第 10 课：计算图可视化与项目收尾](./第10课-计算图可视化与项目收尾.md)，用小图回看梯度如何传播。
