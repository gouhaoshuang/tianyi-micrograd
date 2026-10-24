# 第 04 课：幂运算、ReLU 与 Python 运算协议

先完成 [第 03 课](./第03课-拓扑排序与自动反向传播.md) 的 `backward()`。本课补齐常用表达式，让你可以直接写 `2*x+1`、`x**2`、`x.relu()` 和 `sum(...)`，同时保留完整的求导路径。

学习顺序是：普通数字包装 → 幂运算 → ReLU → 负号与左右运算协议 → 减法与除法。源码中的每段方法只手写一次，不需要重建整个 `Value` 类。

## 1. 本课修改哪里

以下操作由你手动完成：

| 文件                                                                      | 操作                                      |
| ------------------------------------------------------------------------- | ----------------------------------------- |
| `/data/ghs/projects/tianyi-micrograd/src/tianyi_micrograd/engine.py`    | 修改加乘方法，增加幂、ReLU 和运算协议方法 |
| `/data/ghs/projects/tianyi-micrograd/test/tianyi_micrograd/test_ops.py` | 新建文件，只写两个测试                    |

保留构造函数、`__repr__`、`backward()` 和已有测试。继续使用 `agent` 环境与现有 `pyproject.toml`，不增加依赖。

## 2. 第一步：让加法和乘法接受普通数字

之前 `x+2` 会失败，因为普通数字没有 `.data`。处理方式是在运算入口把数字转换成常数节点。

用下面两个方法替换原来的加乘方法；和第 02 课相比，各自只多了开头的包装语句：

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
```

`isinstance(other, Value)` 判断对象是否是 `Value`。条件表达式 `A if 条件 else B` 表示根据条件选择结果：已经是节点就保留，否则调用 `Value(other)` 包装。

本课的普通数字使用 Python `int`、`float`，不扩展其他类型。包装后，局部反传只处理节点，不必再为数字写一份求导逻辑。

此时 `x+2`、`x*2` 可以工作，`2+x`、`2*x` 还需要后面的反向运算协议。

## 3. 第二步：增加常数指数的幂运算

对于 `out=x**p`，指数 `p` 是固定数值：

```text
局部导数：dout/dx = p*x**(p-1)
梯度贡献：x.grad += p*x.data**(p-1)*out.grad
```

例如 `x=2`、`p=3` 时，输出为 `8`，局部导数为 `12`。

在 `Value` 类中新增：

```python
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "only int/float exponents are supported"
        out = Value(self.data ** exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out
```

`x**3` 会调用 `x.__pow__(3)`。`(self,)` 是只有一个元素的元组，末尾逗号不能省略；这个运算只有一个输入节点。

指数直接保存在闭包中，不是待求导的节点。本课不支持 `x**Value(3)`，也不实现 `2**x`。

| 表达式，取`x=2` | 输出值 | 输出对 x 的导数 |
| ----------------- | -----: | --------------: |
| `x**3`          |      8 |              12 |
| `x**2`          |      4 |               4 |
| `x**-1`         |    0.5 |           -0.25 |

负一次幂稍后用于实现除法。负幂要求底数非零；非整数幂先使用正底数。本课沿用原版幂公式，零次幂先在非零底数上验证，暂不把 `Value(0)**0` 的反传列为支持用例，因为公式会先计算 `0**-1`。

## 4. 第三步：增加 ReLU

ReLU 是一个分段函数：

```text
relu(x) = max(0, x)

x > 0 时，局部导数为 1
x < 0 时，局部导数为 0
x = 0 时，本项目约定反传贡献为 0
```

零点在数学上不可导，这里选择 `0`，与参考实现一致。它不是通过有限差分推导出的唯一答案。

在 `Value` 类中新增：

```python
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out
```

`out.data > 0` 得到布尔值。Python 在这里做乘法时，`True` 按 `1`、`False` 按 `0` 参与计算，因此它充当局部导数的开关。

ReLU 不会修改输入，而是产生新节点。负区间传回的贡献为零，不代表把输入之前从其他路径收到的梯度清掉，所以仍然使用 `+=`。

## 5. 第四步：负号与反向加乘

### 5.1 负号可以由乘法组成

在类中新增：

```python
    def __neg__(self):
        return self * -1
```

`-x` 会调用这个方法。乘法已经知道怎样建立图和反传，所以这里不用再写一份反传闭包。

### 5.2 为什么 x+2 和 2+x 需要分别处理

对于 `x+2`，左边的 `Value` 可以处理 `__add__`。对于 `2+x`，整数的加法无法直接处理这个对象时，Python 会尝试右边对象的 `__radd__`。

这里的 `r` 表示反向运算协议，与反向传播不是同一个概念。

新增：

```python
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
```

加法和乘法可以交换操作数，因此转回已有的方法即可。

| 表达式  | 对 Value 使用的方法 |
| ------- | ------------------- |
| `x+2` | `x.__add__(2)`    |
| `2+x` | `x.__radd__(2)`   |
| `x*2` | `x.__mul__(2)`    |
| `2*x` | `x.__rmul__(2)`   |

### 5.3 为什么现在 sum 也能用了

`sum([a,b,c])` 默认从整数 `0` 开始累加，第一步就是 `0+a`。有了 `__radd__` 和数字包装，就能继续得到 `Value`，保留计算图。

## 6. 第五步：用已有运算组成减法和除法

新增下面四个方法：

```python
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1
```

对应关系是：

```text
x-y = x+(-y)
y-x = y+(-x)
x/y = x*(y**-1)
y/x = y*(x**-1)
```

减法和除法不能随便交换顺序，反向协议要保留原表达式的方向。特别是 `5-x` 不能写成 `x-5`，`8/x` 也不能写成 `x/8`。

当除数是普通数字时，Python 先计算数值倒数，再由乘法包装；当除数是 `Value` 时，幂运算会建立倒数节点。两种情况都复用已有的求导规则，不额外写除法闭包。

### c += ... 会不会原地改掉旧节点？

当前没有实现 `__iadd__`，所以这里的 `c += ...` 会使用加法结果重新绑定名字，不会直接修改旧节点的 `.data`。

下面是理解用的小实验，可在 Python 交互窗口执行：

```python
from tianyi_micrograd.engine import Value

c = Value(2.0)
old = c
c += c + 1
c.backward()

print(c.data, old.data, old.grad)
```

预期是 `5.0 2.0 2`：`old` 仍指向原节点，新表达式两次使用它，因此收到两份梯度贡献。不要为模拟 `+=` 而直接改变 `.data`，那样会破坏计算历史。

## 7. 第六步：用一个混合表达式检查结果

手动创建 `/data/ghs/projects/tianyi-micrograd/test/tianyi_micrograd/test_ops.py`，将混合表达式写成测试：

```python
from tianyi_micrograd.engine import Value


def test_mixed_ops():
    x = Value(2.0)
    loss = sum([(2*x + 1)**2 / 2, 5-x, -x, 8/x])
    loss += x - 1
    loss.backward()

    assert loss.data == 18.5
    assert x.grad == 7.0
```

在 `agent` 环境、项目根目录运行这个测试：

```bash
python -m pytest test/tianyi_micrograd/test_ops.py::test_mixed_ops -q
```

通过时应显示 `1 passed`。两个断言分别检查前向结果和输入梯度，同时覆盖正幂、倒数、左右混合运算、`sum` 与 `+=`。

可以逐项核对，而不依赖另一个求导库：

| 项，取`x=2`    | 数值 | 对 x 的导数 |
| ---------------- | ---: | ----------: |
| `(2*x+1)**2/2` | 12.5 |          10 |
| `5-x`          |    3 |          -1 |
| `-x`           |   -2 |          -1 |
| `8/x`          |    4 |          -2 |
| `x-1`          |    1 |           1 |
| 合计             | 18.5 |           7 |

第一项的导数由链式法则得到：`2*(2*x+1)*2/2`。第四项为 `-8/x**2`。它们都在 `x=2` 处求值。

## 8. 第七步：补充 ReLU 测试

在同一个测试文件末尾追加下面的函数，不需要重复添加导入或混合运算测试：

```python
def test_relu():
    for data, expected_data, expected_grad in [
        (-2.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (2.0, 4.0, 2.0),
    ]:
        x = Value(data)
        out = 2 * x.relu()
        out.backward()

        assert out.data == expected_data
        assert x.grad == expected_grad
```

这样本课仍然只有两个测试：第 7 节检查混合运算，本节用一个循环检查 ReLU 的负区、零点和正区，并确认反传乘上了上游梯度 `2`。

循环每次创建新节点，不会复用前一轮的梯度。这些例子的结果能被精确表示，因此直接比较；后续一般浮点和有限差分检查再使用误差容限。

运行本课测试和已有测试：

```bash
python -m pytest test/tianyi_micrograd/test_ops.py -q
python -m pytest test -q
```

本课文件应显示 `2 passed`，完整数量以你实际保留的测试为准。

## 9. 常见问题与验收

| 现象                         | 优先检查                                              |
| ---------------------------- | ----------------------------------------------------- |
| `x+2` 可以，`2+x` 不行   | 是否增加了`__radd__`                                |
| `sum()` 在第一项失败       | 是否能处理`0+Value`                                 |
| 减法、除法的数值方向相反     | 反向协议是否保留`other-self`、`other/self` 的方向 |
| 幂或 ReLU 的梯度少了倍数     | 是否乘了`out.grad`，并用 `+=` 累加                |
| ReLU 零点与差分结果不同      | 零点不可导，本项目选择反传贡献为零                    |
| 表达式数值对，但部分梯度丢失 | 是否中途取出`.data` 做运算，绕过了计算图            |

完成后回答：为什么减法和除法不用再写局部反传？为什么 `__radd__` 与反向传播无关？为什么 `c += ...` 之后旧节点仍可参与求导？

参考阅读：[原版幂和 ReLU](/data/ghs/projects/micrograd/micrograd/engine.py:35)、[运算协议](/data/ghs/projects/micrograd/micrograd/engine.py:72)。下一课再用有限差分系统检查组合表达式的梯度，并按需加入 PyTorch 对照。
