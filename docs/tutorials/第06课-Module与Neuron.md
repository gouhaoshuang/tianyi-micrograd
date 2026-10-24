# 第 06 课：Module 与 Neuron

前面的 `Value` 已经能对标量表达式求导。完成第 05 课的主线检查后，本课把这些标量运算组织成一个神经元，并明确哪些节点是模型参数、怎样清空参数梯度。

本课只实现 `Module` 和 `Neuron`，不写 `Layer`、`MLP` 或训练循环。验证全部写成 pytest 测试，共两个测试函数，不使用 PyTorch。

## 1. 本课创建哪些文件

由你手动创建：

| 文件                                                                     | 内容                     |
| ------------------------------------------------------------------------ | ------------------------ |
| `/data/ghs/projects/tianyi-micrograd/src/tianyi_micrograd/nn.py`       | `Module` 和 `Neuron` |
| `/data/ghs/projects/tianyi-micrograd/test/tianyi_micrograd/test_nn.py` | 两个神经元测试           |

现有 `engine.py`、`__init__.py`、`pyproject.toml` 都不需要修改。在已有的可编辑安装下，新增包内模块后可直接导入：

```python
from tianyi_micrograd.nn import Neuron
```

后面的命令都在 conda `agent` 环境、项目根目录执行：

```bash
conda activate agent
cd /data/ghs/projects/tianyi-micrograd
```

## 2. 一个神经元到底做什么

对于三个输入，先计算加权和，再按需经过 ReLU：

```text
z = w0*x0 + w1*x1 + w2*x2 + b
output = relu(z)   或直接返回 z
```

这里有两类数：

- 输入 `x`：本次送给模型的数据。
- 参数 `w`、`b`：模型持有、以后会根据梯度更新的权重与偏置。

`Neuron(3)` 中的 `3` 是输入个数，不是参数个数。它有三个权重、一个偏置，共四个参数，每个参数都是 `Value`。

神经元不需要重新实现反向传播。只要用这些参数参与 `Value` 运算，已有计算图就能把梯度传回原来的参数对象。

## 3. 第一步：导入工具，写 Module

先在 `nn.py` 顶部写入：

```python
import random

from tianyi_micrograd.engine import Value
```

`random` 是 Python 标准库，只用于初始化权重。

随后在文件顶层添加：

```python
class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0
```

`Module` 提供两个约定：`parameters()` 返回当前模块的参数，`zero_grad()` 清空这些参数的梯度。

基类本身没有参数，所以默认返回空列表。之后的神经元、层、网络分别实现自己的参数收集，清梯度就能复用同一个方法。

注意：清梯度只修改 `.grad`，不改变 `.data`，也不创建新的权重对象。

## 4. 第二步：创建 Neuron 及其参数

在 `Module` 类之后、文件顶层添加下面的类。后面的步骤继续向这个类增加方法，不要重复定义同名类：

```python
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin
```

`Neuron(Module)` 表示继承：`Neuron` 可以使用基类提供的 `zero_grad()`。

- `nin` 是输入数量，本课使用正整数。
- `range(nin)` 迭代 `nin` 次；列表推导每次创建一个新的权重节点。
- `_` 是没有进一步使用的循环变量，表示这里只关心执行次数。
- `random.uniform(-1,1)` 在指定区间产生随机数，初始化方式沿用参考项目。
- `self.b` 是一个独立的偏置节点，初始为零。
- `nonlin=True` 默认启用 ReLU；传 `False` 时返回线性加权和。

基类没有需要初始化的实例状态，这里不必额外调用 `super().__init__()`。

测试时我们会在前向计算前指定参数值，避免随机初始化影响预期结果。

## 5. 第三步：收集参数

在 `Neuron` 类中添加：

```python
    def parameters(self):
        return self.w + [self.b]
```

这覆盖了基类的同名方法。调用 `neuron.zero_grad()` 时，基类方法中的 `self` 仍是这个神经元，`self.parameters()` 会调用这里的版本。

`self.w+[self.b]` 新建一个列表，里面保留原参数对象的引用：

```text
参数列表：[w0, w1, w2, b]
           ↓   ↓   ↓   ↓
        模型实际使用的四个 Value
```

列表可以是新的，里面的参数不能是复制出来的新 `Value`。后面通过参数列表清梯度或更新 `.data`，必须能作用到模型实际使用的对象。

也不要用 `self.w.append(self.b)` 收集参数，否则每调用一次就把偏置塞进权重列表，改变了模型结构。

## 6. 第四步：让神经元可以被调用

在同一个 `Neuron` 类中添加：

```python
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act
```

定义了 `__call__` 后，`neuron([1.0,2.0,-2.0])` 就会执行这个方法，不需要再写一个额外的 `forward()`。

逐段理解计算过程：

1. `zip(self.w,x)` 依次配对权重与输入：`(w0,x0)`、`(w1,x1)`、`(w2,x2)`。
2. `(wi*xi for ...)` 是生成器表达式，逐项产生乘积节点。
3. `sum(...,self.b)` 从偏置节点开始累加乘积，因此偏置也进入计算图。
4. 根据 `nonlin`，返回 ReLU 的结果或加权和本身。

输入可以是普通数值，已有运算会把它们包装成节点；也可以是上一层输出的 `Value`。

本课约定输入长度恰好等于 `nin`。`zip()` 会按较短的一侧结束，并不自动检查维度；暂时不要把长度不匹配的输入作为合法用法。

### 为什么这里不再写 _backward

乘法、加法、ReLU 已经各自建立计算历史并保存局部规则。`Neuron` 负责组合它们，最终返回的仍是一个 `Value`。

对输出调用 `backward()`，梯度自然沿图回到 `self.w`、`self.b`，不需要另写一套神经元反传。

## 7. 第五步：增加便于观察的显示

在 `Neuron` 类中添加：

```python
    def __repr__(self):
        kind = "ReLU" if self.nonlin else "Linear"
        return f"{kind}Neuron({len(self.w)})"
```

例如 `Neuron(3,nonlin=False)` 显示为 `LinearNeuron(3)`。这里显示输入规模与激活类型，不把随机参数全部展开。

至此，`nn.py` 只有两个导入、一个 `Module` 类和一个 `Neuron` 类。

## 8. 第六步：固定参数，检查前向与梯度

先选择一份可以手算的配置：

```text
w = [2, -3, 0.5]
b = 1
x = [1, 2, -2]

z = 2*1 + (-3)*2 + 0.5*(-2) + 1 = -4
```

关闭 ReLU 时，输出就是 `-4`。以这个输出为目标，其梯度为：

| 参数   | 输出对该参数的导数 |
| ------ | -----------------: |
| `w0` |           `x0=1` |
| `w1` |           `x1=2` |
| `w2` |          `x2=-2` |
| `b`  |                  1 |

如果启用 ReLU，同样的加权和落在负区，输出与这一路的参数梯度都为零。下面两个测试分别检查线性路径、清梯度与 ReLU 路径。

### 手写第一个测试

手动创建 `test_nn.py`，先写入：

```python
from tianyi_micrograd.nn import Neuron


def test_linear_neuron_and_zero_grad():
    neuron = Neuron(3, nonlin=False)
    params = neuron.parameters()
    assert len(params) == 4
    assert params[0] is neuron.w[0]
    assert params[-1] is neuron.b

    values = [2.0, -3.0, 0.5, 1.0]
    for parameter, value in zip(params, values):
        parameter.data = value

    out = neuron([1.0, 2.0, -2.0])
    out.backward()
    assert out.data == -4.0
    assert [p.grad for p in params] == [1.0, 2.0, -2.0, 1.0]

    neuron.zero_grad()
    assert [p.grad for p in params] == [0, 0, 0, 0]
    assert [p.data for p in params] == values
```

这里在前向之前设置参数 `.data`，之后直到反传结束都不改变它们。测试不依赖随机种子，也不需要先写训练程序。

`is` 检查对象身份，确保参数列表指向模型的实际参数。最后两个断言一起确认：清梯度只清 `.grad`，保留参数值。

## 9. 第七步：用一个测试检查 ReLU 开关

在同一文件末尾追加，不重复添加导入：

```python
def test_relu_neuron():
    for bias, expected_data, expected_grad in [
        (-3.0, 0.0, 0.0),
        (-2.0, 0.0, 0.0),
        (1.0, 3.0, 1.0),
    ]:
        neuron = Neuron(1)
        neuron.w[0].data = 1.0
        neuron.b.data = bias
        out = neuron([2.0])
        out.backward()

        assert out.data == expected_data
        assert (neuron.w[0].grad, neuron.b.grad) == (2 * expected_grad, expected_grad)
```

这次只用一个输入：加权和为 `2+bias`，三轮依次落在负区、零点、正区。`expected_grad` 是 ReLU 的局部导数；权重梯度还要乘输入 `2`，偏置梯度则直接等于它。

每轮重新创建神经元，因此参数梯度互不干扰。线性测试又验证了 `nonlin=False`，两个测试合起来覆盖激活开关。

运行：

```bash
python -m pytest test/tianyi_micrograd/test_nn.py -q
python -m pytest test -q
```

本课文件应显示 `2 passed`。结果都能精确表示，直接比较即可；全部测试的数量以实际文件为准。

## 10. 清梯度的边界与本课验收

`Module.zero_grad()` 只清 `parameters()` 返回的参数，不负责清空旧计算图的所有中间节点。

下一次计算仍需要重新调用神经元，建立新的中间图；后面每轮训练也会在反传前清空被复用的参数梯度。不要把清参数梯度理解为可以反复对旧输出调用 `backward()`。

| 现象                             | 优先检查                                         |
| -------------------------------- | ------------------------------------------------ |
| 参数个数少一个                   | 是否遗漏偏置                                     |
| 参数列表中的值改了，模型输出没变 | 是否复制了参数节点，而非返回原对象               |
| 偏置梯度为零                     | 加权和是否包含`self.b`，是否中途取了 `.data` |
| 线性负输出被变成零               | 是否无条件调用了 ReLU                            |
| 调用 zero_grad 后权重变了        | 是否把参数值或参数对象也重置了                   |
| 清梯度方法没有作用               | `Neuron.parameters()` 是否覆盖了基类的空实现   |

验收时解释：为什么三个输入对应四个参数；为什么 `Neuron` 不需要自己的 `_backward`；为什么基类的 `zero_grad()` 可以清空子类参数。

参考阅读：[原版 Module 与 Neuron](/data/ghs/projects/micrograd/micrograd/nn.py:4)。下一课：[第 07 课：Layer 与 MLP](./第07课-Layer与MLP.md)，将多个神经元组成层，再把层组成网络。
