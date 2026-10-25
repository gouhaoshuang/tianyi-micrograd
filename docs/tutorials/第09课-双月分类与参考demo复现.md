# 第 09 课：双月分类与参考 demo 复现

第 08 课已经走通“前向 → 清梯度 → 反传 → 更新”。本课在 **Jupyter Notebook** 里用自己的 `MLP` 完成双月分类，依次观察数据分布、训练前后的决策边界，以及损失和准确率曲线。

参考本机的 [demo.ipynb](/data/ghs/projects/micrograd/demo.ipynb)。采用全量训练，保留两个简洁的 pytest 测试。Notebook 和辅助代码都由你手写，本文只给出操作步骤与单元格内容。

## 1. 本课创建哪些文件

沿用项目已有的 `examples/` 目录，以下文件由你手动创建：

| 文件                                                                        | 内容                                           |
| --------------------------------------------------------------------------- | ---------------------------------------------- |
| `/data/ghs/projects/tianyi-micrograd/examples/train_moons.ipynb`          | 本课实验：数据图、训练、曲线和决策边界         |
| `/data/ghs/projects/tianyi-micrograd/examples/moons_utils.py`             | 数据生成、模型创建、损失函数，供实验和测试共用 |
| `/data/ghs/projects/tianyi-micrograd/test/tianyi_micrograd/test_moons.py` | 两个测试                                       |

沿用第 08 课已有的 `examples/__init__.py`、`nn.py` 和可编辑安装。原教程中的训练脚本入口改为 Notebook 单元格；辅助模块只保存可复用函数，导入时不会启动训练。

### 环境与 Notebook 内核

默认使用 conda `agent`。数据与绘图依赖仍是 NumPy、scikit-learn、Matplotlib，PyTorch 不参与实验。

```bash
conda activate agent
cd /data/ghs/projects/tianyi-micrograd
python -m pip show ipykernel numpy scikit-learn matplotlib
```

编写本次文档时，后三项已安装，`ipykernel` 尚未安装。由你按需补齐 Notebook 内核：

```bash
python -m pip install ipykernel
```

如果换环境后缺少数据绘图库，再手动安装 `numpy scikit-learn matplotlib`。这些是实验依赖，不需要因此修改核心包的依赖声明。

在支持 `.ipynb` 的编辑器中，新建并保存 `examples/train_moons.ipynb`，选择 `agent` 对应的 Python 内核，解释器应为：

```text
/data/ghs/miniconda3/envs/agent/bin/python
```

若使用浏览器版 JupyterLab，可另行手动安装 `jupyterlab`，再从项目根目录执行 `python -m jupyter lab`，打开同一个 Notebook。

每个“单元格”下面的代码块对应一个代码单元格，可以在前面加 Markdown 单元格记录目的和观察。最后用“重启内核并运行全部单元格”检查执行顺序。

## 2. 先理解输出和分类损失

模型使用 `MLP(2,[16,16,1])`：输入二维坐标，两个隐藏层各有 16 个 ReLU 神经元，最后一层线性输出一个分数。

```text
score > 0  →  预测标签 +1
score <= 0 →  预测标签 -1
参数数目 = (2+1)*16 + (16+1)*16 + (16+1)*1 = 337
```

分数不是概率。最后一层关闭 ReLU，才能输出负类所需的负分数。数据原来的 `0/1` 标签转换为 `-1/+1`。

单个样本采用间隔损失（hinge loss）：

```text
sample_loss = max(0, 1-y*score)
```

|  y | score | y*score | 样本损失 | 是否分类正确 |
| -: | ----: | ------: | -------: | ------------ |
| +1 |     2 |       2 |        0 | 是           |
| +1 |   0.2 |     0.2 |      0.8 | 是           |
| -1 |     1 |      -1 |        2 | 否           |

准确率只看正负；损失还要求正确方向上的间隔足够大，所以分类正确后损失仍可能下降。

所有样本取平均，加上 L2 正则：

```text
data_loss = 平均样本损失
reg_loss = alpha * Σ(parameter**2)
total_loss = data_loss + reg_loss
```

与参考 demo 一样，`alpha=1e-4`，对全部参数（含偏置）做正则，为每个参数增加 `2*alpha*parameter` 的梯度。本项目 ReLU 在零点取零梯度，因此恰好达到间隔 `y*score=1` 时，样本损失的梯度也取零。

## 3. 第一步：手写三个共用函数

先在 `examples/moons_utils.py` 写入：

```python
import random

from tianyi_micrograd.nn import MLP


def make_data():
    from sklearn.datasets import make_moons

    xs, ys = make_moons(n_samples=100, noise=0.1, random_state=1337)
    return xs.tolist(), (2 * ys - 1).tolist()


def make_model():
    random.seed(1337)
    return MLP(2, [16, 16, 1])
```

数据生成与模型初始化是两套随机数来源，分别固定 `random_state`、`random.seed`。`.tolist()` 把数据转成普通 Python 数值列表，交给自己的模型后，由 `Value` 运算包装常量。

随后在同一文件末尾追加：

```python
def loss_and_accuracy(model, xs, ys, alpha=1e-4):
    scores = [model(x) for x in xs]
    losses = [(1 - y * score).relu() for y, score in zip(ys, scores)]
    data_loss = sum(losses) / len(losses)
    reg_loss = alpha * sum(p * p for p in model.parameters())

    correct = sum((y > 0) == (score.data > 0) for y, score in zip(ys, scores))
    return data_loss + reg_loss, correct / len(ys)
```

约定数据非空、输入与标签数量相等，标签为 `-1/+1`，输入长度匹配模型。损失始终保留 `Value`；准确率只作统计，可以取 `.data`。`True/False` 求和相当于 `1/0`，零分数归入负类。

`sklearn` 在 `make_data()` 内才导入，保证损失函数测试不需要数据生成库。测试直接导入这个小模块，无需读取或执行整个 Notebook。

## 4. 第二步：准备 Notebook 内核和导入

### 单元格 1：确认解释器，固定工作目录

Notebook 内核可能从 `examples/` 启动，先统一到项目根目录，避免后续导入和保存路径随启动位置改变。

```python
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/data/ghs/projects/tianyi-micrograd")
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Python:", sys.executable)
print("工作目录:", Path.cwd())
```

解释器应指向 `agent`，目录应为项目根目录。这里补入根目录是为了导入本地 `examples`；核心包继续通过已有可编辑安装导入。

### 单元格 2：导入与内联绘图

下面的 `%matplotlib inline` 是 Notebook 的 IPython 指令，用于在单元格下方显示图片，不写入普通 `.py` 文件。

```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from examples.moons_utils import make_data, make_model, loss_and_accuracy
```

修改了 `engine.py`、`nn.py` 或辅助模块后，重启内核并从头运行，避免继续使用内存中旧版本的类和函数。

## 5. 第三步：先看数据，再看初始模型

### 单元格 3：画出双月数据

```python
xs, ys = make_data()
points = np.asarray(xs)
colors = np.array([int(y > 0) for y in ys])

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(points[:, 0], points[:, 1], c=colors, cmap="Spectral",
           vmin=0, vmax=1, edgecolors="k", s=30)
ax.set(xlabel="x0", ylabel="x1", title="Two moons: training data")
plt.show()
```

应看到两组有噪声的弯月形点。`points[:,0]`、`points[:,1]` 是横纵坐标，`colors` 把负正标签映射成 `0/1`，后面边界图沿用同一套颜色。

### 单元格 4：查看模型和初始指标

```python
model = make_model()
initial_loss, initial_accuracy = loss_and_accuracy(model, xs, ys)
print(model)
print("参数数目:", len(model.parameters()))
print(f"初始总损失: {initial_loss.data:.6f}, 训练准确率: {initial_accuracy:.1%}")
```

应有 337 个参数。记录自己的初始指标，不要求随机模型达到某个固定准确率。

## 6. 第四步：画训练前的决策边界

### 单元格 5：定义绘图函数，立即显示初始边界

```python
def plot_boundary(model, xs, ys, title):
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
    ax.set(xlabel="x0", ylabel="x1", title=title)
    return fig


initial_boundary_fig = plot_boundary(model, xs, ys, "Before training")
plt.show()
```

`meshgrid()` 生成规则网格，`ravel()` 将坐标展开，`column_stack()` 把横纵坐标配对。每个网格点交给自己的模型，正负预测再 `reshape()` 回原来的网格形状，供 `contourf()` 画背景。

绘图只读取分数，不反传、不改参数。当前引擎没有 `no_grad()`，预测仍会创建临时图；这里只保存布尔预测，避免持有全部网格点的输出图。先保持 `h=0.25`，网格太密会明显变慢。

## 7. 第五步：在单元格里训练

### 单元格 6：重置模型，完成 100 次更新

```python
model = make_model()
steps = 100
history = []

for step in range(steps):
    total_loss, accuracy = loss_and_accuracy(model, xs, ys)
    history.append((step, total_loss.data, accuracy))

    model.zero_grad()
    total_loss.backward()
    learning_rate = 1.0 - 0.9 * step / steps
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if step % 10 == 0 or step == steps - 1:
        print(f"更新前 {step:03d}: loss={total_loss.data:.6f}, "
              f"train_acc={accuracy:.1%}, lr={learning_rate:.3f}")

final_loss, final_accuracy = loss_and_accuracy(model, xs, ys)
history.append((steps, final_loss.data, final_accuracy))
print(f"完成 {steps} 次更新: loss={final_loss.data:.6f}, "
      f"train_acc={final_accuracy:.1%}")
```

这里保留第 08 课的顺序：重新前向、清参数梯度、反传、更新 `.data`。每一步都使用全部 100 个样本。

单元格开头重新创建固定种子的模型，并清空 `history`，因此重复运行本单元格会从初始参数重新训练。若想复现实验，从头执行各单元格；不要一边换数据，一边保留旧图和指标。

学习率与参考 demo 的默认公式一致，从 `1.0` 递减，最后一步为 `0.109`。循环内记录更新前的指标，结尾重新前向得到最终指标；`history` 共 101 条，横坐标表示已经完成的更新次数。

历史只存普通数值，避免持有每轮旧图。损失应使用当前图反传，不能对旧损失反复调用 `backward()`。

## 8. 第六步：并排看曲线，再看最终边界

### 单元格 7：总损失与训练准确率曲线

```python
updates, losses, accuracies = zip(*history)

curves_fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(updates, losses)
axes[0].set(xlabel="Updates", ylabel="Total loss", title="Loss")
axes[1].plot(updates, [100 * acc for acc in accuracies])
axes[1].set(xlabel="Updates", ylabel="Training accuracy (%)",
            title="Accuracy", ylim=(0, 105))
curves_fig.tight_layout()
plt.show()
```

`zip(*history)` 把每条记录的三项拆成三组序列。总损失包含平均 hinge loss 与 L2，不要求每一步都下降；准确率达到高值后，间隔与正则仍可能让损失变化。

### 单元格 8：显示最终决策边界和指标

```python
boundary_fig = plot_boundary(model, xs, ys, "After training")
plt.show()

print(f"初始: loss={history[0][1]:.6f}, train_acc={history[0][2]:.1%}")
print(f"最终: loss={history[-1][1]:.6f}, train_acc={history[-1][2]:.1%}")
```

与第 5 个单元格的图对比：背景区域如何改变，能否沿两组弯月弯曲，哪些点仍落在另一类区域？这里只使用训练数据，图和指标都不能代替独立测试集上的评估。

### 可选单元格 9：保存图片

Notebook 已经能内联显示图片；需要导出分享时，再手动运行：

```python
output_dir = PROJECT_ROOT / "outputs"
output_dir.mkdir(exist_ok=True)
curves_fig.savefig(output_dir / "moons_training_curves.png", dpi=150, bbox_inches="tight")
boundary_fig.savefig(output_dir / "moons_boundary.png", dpi=150, bbox_inches="tight")
```

保存 Notebook 时可以保留图表输出，方便回顾。修改数据或训练设置后，重新运行下游单元格，避免展示的曲线与边界来自不同实验。

## 9. 第七步：仍然只写两个 pytest 测试

手动创建 `test_moons.py`，写入：

```python
from math import isclose

import pytest

from examples.moons_utils import loss_and_accuracy, make_data, make_model
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

第一个测试用线性神经元手算：分数 `2、1`，标签 `+1、-1`，平均样本损失为 `1`，正则为 `0.1`。第二个样本贡献 `dw=db=0.5`，正则增加 `dw=0.2`，所以总梯度为 `0.7、0.5`。

第二个检查数据与完整网络的接口。缺少 scikit-learn 时只跳过它，损失测试仍可运行。测试不需要 Notebook 内核，不启动训练，也不生成图片。

在终端、项目根目录手动运行：

```bash
python -m pytest test/tianyi_micrograd/test_moons.py -q
python -m pytest test -q
```

依赖齐全时，新文件应为 `2 passed`；缺少 scikit-learn 时应为 `1 passed, 1 skipped`。Notebook 的可视化效果由你从头运行实验确认，不把固定训练准确率写成测试门槛。

## 10. 实验记录、常见问题与验收

在 Notebook 最后加一个 Markdown 单元格，记录：

| 项目 | 记录内容                                          |
| ---- | ------------------------------------------------- |
| 数据 | 100 个样本，噪声 0.1，随机状态 1337               |
| 模型 | `MLP(2,[16,16,1])`，初始化种子 1337，337 个参数 |
| 训练 | 100 步，全量数据，`alpha=1e-4`，递减学习率      |
| 指标 | 自己实际运行得到的初始和最终总损失、训练准确率    |
| 图表 | 数据分布、训练曲线、前后决策边界的观察            |

与原版对齐的是数据规模、网络、标签、损失和更新逻辑。初始化、库版本及浮点累加顺序可能影响具体轨迹，不要求逐字复现原 notebook 日志。

| 现象                         | 优先检查                                         |
| ---------------------------- | ------------------------------------------------ |
| 内核无法启动或解释器不对     | 是否安装`ipykernel`，是否选中 `agent`        |
| 不能导入 examples 或自己的包 | 是否运行单元格 1，核心包是否完成可编辑安装       |
| 改了模块但实验行为没变       | 重启内核，重新导入并从头运行                     |
| 图片没有出现在单元格下方     | 是否运行`%matplotlib inline` 与 `plt.show()` |
| 负类几乎不参与学习           | 标签是否为 -1/+1，最后一层是否关闭 ReLU          |
| 曲线与最终边界不对应         | 是否在重新训练后执行了曲线和边界单元格           |
| 参数没有梯度                 | 损失是否提前取了`.data`，是否每轮重新建图      |

验收时解释：为何分类正确仍可能有损失；L2 怎样改变梯度；为什么准确率可以取 `.data` 而损失不可以；为什么这里要称为训练集准确率；为什么 Notebook 需要从头运行来确认实验可复现。

完成后进入 [第 10 课：计算图可视化与项目收尾](./第10课-计算图可视化与项目收尾.md)。
