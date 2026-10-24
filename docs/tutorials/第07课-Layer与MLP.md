# 第 07 课：Layer 与 MLP

完成 [第 06 课](./第06课-Module与Neuron.md) 后，`nn.py` 应当已经有 `Module` 和 `Neuron`。本课继续在这个文件里增加 `Layer`、`MLP`，让多个神经元共同完成一次前向计算。

本课沿用原版的接口：多输出层返回列表，单输出层返回一个 `Value`；隐藏层使用 ReLU，最后一层输出线性分数。不训练模型，只用两个测试检查组合关系、参数和梯度。

## 1. 本课修改哪些文件

以下步骤都由你手动完成：

| 文件 | 操作 |
| --- | --- |
| `/data/ghs/projects/tianyi-micrograd/src/tianyi_micrograd/nn.py` | 在 `Neuron` 后追加两个类 |
| `/data/ghs/projects/tianyi-micrograd/test/tianyi_micrograd/test_mlp.py` | 新建，写两个测试 |

保留第 06 课的 `test_nn.py`。本课无需修改 `engine.py`、包初始化文件或 `pyproject.toml`。

后面的命令都从项目根目录执行：

```bash
conda activate agent
cd /data/ghs/projects/tianyi-micrograd
```

## 2. 一层和一个网络分别做什么

`Layer(3,4)` 创建四个神经元，每个神经元都读取同一份三个输入，输出四个数。

`MLP(3,[4,4,1])` 则依次执行三层：

```mermaid
flowchart LR
    X["3 个输入"] --> A["Layer(3,4)：4 个 ReLU 输出"]
    A --> B["Layer(4,4)：4 个 ReLU 输出"]
    B --> C["Layer(4,1)：1 个线性输出"]
```

MLP 是多层感知机。这里的列表 `[4,4,1]` 表示各层的输出宽度，输入宽度 `3` 单独传入。

每个神经元有 `输入数+1` 个参数，多出来的一个是偏置：

| 层 | 神经元数量 | 每个神经元的参数数 | 本层参数数 |
| --- | ---: | ---: | ---: |
| `3 → 4` | 4 | 4 | 16 |
| `4 → 4` | 4 | 5 | 20 |
| `4 → 1` | 1 | 5 | 5 |
| 合计 | | | **41** |

参数列表里应当恰好包含这 41 个原始对象，不能漏掉偏置，也不能把中间输出收进去。

## 3. 第一步：创建 Layer，完成前向

在 `nn.py` 的 `Neuron` 类之后，顶层追加：

```python
class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out
```

`nin` 是每个神经元的输入数，`nout` 是本层神经元数。所有神经元接收同一份 `x`，各自持有不同的参数。

### **kwargs 在这里做什么

`**kwargs` 收集额外的关键字参数，再原样传给每个神经元。例如：

```python
layer = Layer(2, 3, nonlin=False)
```

这等价于创建三个 `Neuron(2,nonlin=False)`。不传 `nonlin` 时，使用 `Neuron` 的默认值 `True`。

这里要写列表推导式，不能用 `[Neuron(nin)]*nout`。后者会让多个位置引用同一个神经元，意外共享全部参数。

### 单输出的约定

`Layer(2,3)` 返回长度为 3 的列表，`Layer(2,1)` 返回单个 `Value`。这是与本机参考实现对齐的选择，调用方需要知道它的返回类型。

这个约定也有边界：如果中间层宽度是 `1`，它返回的标量不能直接交给下一层的 `zip(self.w,x)`。因此本系列先约定：输入维度匹配、层宽为正、`nouts` 是非空列表，且只有最后一层可以宽度为 `1`。

比如 `MLP(3,[4,4,1])` 合法；不要把 `MLP(3,[1,4,1])` 当作已经支持的结构。统一列表返回或增加输入检查，可以在主线完成后再单独设计。

## 4. 第二步：收集一层的参数

在同一个 `Layer` 类里继续添加：

```python
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        items = ", ".join(str(neuron) for neuron in self.neurons)
        return f"Layer of [{items}]"
```

两层 `for` 的顺序是：先选一个神经元，再取出它的全部参数，最后继续下一个神经元。结果是一个扁平列表：

```text
[第一个神经元的 w 和 b，第二个神经元的 w 和 b，……]
```

不能返回 `[neuron.parameters() for neuron in self.neurons]`，那样得到的是列表套列表，基类的 `zero_grad()` 无法对内层列表访问 `.grad`。

`join()` 把多个神经元的字符串用逗号连接，只影响显示。

## 5. 第三步：按尺寸创建 MLP

在 `Layer` 类之后，顶层追加：

```python
class MLP(Module):
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]
```

以 `MLP(3,[4,4,1])` 为例，`sizes=[3,4,4,1]`，相邻两项依次组成 `(3,4)`、`(4,4)`、`(4,1)`。

`i != len(nouts)-1` 在最后一层为 `False`，其余层为 `True`。所以最后一层不经过 ReLU，可以输出负数；这对后面用正负分数做分类很重要。

不要把这个判断写反，也不要给最后一层无条件加 ReLU。

## 6. 第四步：串联各层，收集整个网络的参数

在同一个 `MLP` 类里添加：

```python
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        items = ", ".join(str(layer) for layer in self.layers)
        return f"MLP of [{items}]"
```

循环中的 `x` 每次绑定到当前层的输出，作为下一层的输入。这里没有复制或重新包装上一层的节点，计算图会自然连接起来。

参数收集再次展开一层列表。`model.zero_grad()` 继承自 `Module`，会通过这里的 `parameters()` 找到所有层的权重和偏置。

`Layer`、`MLP` 都不需要新的 `_backward()`。它们只组织已有的 `Value` 运算。

## 7. 第五步：用固定参数检查一层

手动创建 `test_mlp.py`，先写入：

```python
from tianyi_micrograd.engine import Value
from tianyi_micrograd.nn import Layer, MLP


def test_layer():
    layer = Layer(2, 2, nonlin=False)
    for neuron, values in zip(layer.neurons, [(1, 2, 0.5), (-1, 0.5, -1)]):
        for parameter, value in zip(neuron.parameters(), values):
            parameter.data = value

    out = layer([2.0, -1.0])
    assert isinstance(out, list)
    assert [v.data for v in out] == [0.5, -3.5]
    assert len(layer.parameters()) == 6

    sum(out).backward()
    assert [p.grad for p in layer.parameters()] == [2, -1, 1, 2, -1, 1]

    single = Layer(2, 1, nonlin=False)
    for p in single.parameters():
        p.data = -1.0
    scalar = single([1.0, 2.0])
    assert isinstance(scalar, Value)
    assert scalar.data == -4.0
```

前两个输出可以手算：`1*2+2*(-1)+0.5=0.5`，`(-1)*2+0.5*(-1)-1=-3.5`。

目标是两个输出之和，因此每个神经元的参数梯度都是 `[x0,x1,1]=[2,-1,1]`。最后几行单独检查单输出返回标量，且线性输出允许负值。

## 8. 第六步：检查整个网络

在同一个测试文件末尾追加：

```python
def test_mlp():
    model = MLP(3, [4, 4, 1])
    params = model.parameters()
    assert len(params) == len({id(p) for p in params}) == 41
    assert params[-1] is model.layers[-1].neurons[0].b
    assert all(n.nonlin for layer in model.layers[:-1] for n in layer.neurons)
    assert model.layers[-1].neurons[0].nonlin is False

    for p in params:
        p.data = 0.5
    params[-1].data = -10.0

    out = model([1.0, 2.0, -1.0])
    assert isinstance(out, Value)
    assert out.data == -3.0
    out.backward()
    assert params[-1].grad == 1.0
    assert params[-2].grad == 3.5
    assert params[0].grad == 1.0

    model.zero_grad()
    assert all(p.grad == 0 for p in params)
```

这里先把参数全部设为 `0.5`，再通过参数列表把最后的偏置设为 `-10`，然后才做前向。预期结果如下：

| 位置 | 每个输出的值 |
| --- | ---: |
| 第一层 | `0.5*(1+2-1)+0.5=1.5` |
| 第二层 | `4*0.5*1.5+0.5=3.5` |
| 最后一层 | `4*0.5*3.5-10=-3` |

两个隐藏层都位于 ReLU 正区。最后一个权重的梯度是其输入 `3.5`，最后偏置的梯度是 `1`。第一个权重经过四个第二层神经元到达输出，梯度为 `1*4*0.5*0.5=1`。

`{id(p) for p in params}` 收集对象身份，检查 41 个参数没有被重复引用。通过参数列表设置偏置能改变前向结果，也说明列表指向模型实际使用的对象。

运行：

```bash
python -m pytest test/tianyi_micrograd/test_mlp.py -q
python -m pytest test -q
```

本课新文件应为 `2 passed`。本例中的数值都可以精确表示，不需要浮点误差容限。整个项目的测试数量以你实际创建的文件为准。

## 9. 常见问题与验收

| 现象 | 优先检查 |
| --- | --- |
| 参数数目不是 41 | 是否漏偏置、遗漏一层，或返回了嵌套列表 |
| 多个神经元参数完全相同且同时变化 | 是否用列表乘法重复了同一个对象 |
| 输出意外变为零 | 最后一层是否错误启用了 ReLU |
| 标量不能被遍历 | 是否把宽度为 1 的层放到了中间 |
| 最后一层有梯度，第一层没有 | 层间是否取了 `.data` 或重新创建了 `Value` |
| 清梯度只影响最后一层 | `MLP.parameters()` 是否收集了所有层 |

验收时解释：41 个参数怎样算出；为什么最后一层关闭 ReLU；为什么网络能直接复用 `Module.zero_grad()`；为什么返回同一个数值和返回同一个参数对象是两件事。

参考：[原版 Layer 与 MLP](/data/ghs/projects/micrograd/micrograd/nn.py:30)。通过后进入 [第 08 课：第一次完整训练](./第08课-第一次完整训练.md)。
