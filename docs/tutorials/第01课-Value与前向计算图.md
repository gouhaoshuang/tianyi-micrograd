# 第 01 课：Value 与前向计算图

第 00 课用普通浮点数完成了前向与有限差分。本课开始手写项目：让一个对象同时保存数值和“它是怎样算出来的”。

本课只完成前向计算图。结束时还没有 `backward()`，梯度保持初始值；后两课会补上反向传播。

## 1. 你需要手动创建哪些文件

请在编辑器中手动创建：

| 文件                                                                     | 内容                       |
| ------------------------------------------------------------------------ | -------------------------- |
| `/data/ghs/projects/tianyi-micrograd/src/tianyi_micrograd/__init__.py` | 留空即可                   |
| `/data/ghs/projects/tianyi-micrograd/src/tianyi_micrograd/engine.py`   | 本课逐步写出的`Value` 类 |
| `/data/ghs/projects/tianyi-micrograd/test/test_engine.py`              | 本课末尾的测试             |

需要先创建 `src/tianyi_micrograd/` 目录。不修改参考仓库。

`__init__.py` 用来标识普通 Python 包，本课不在里面重导出任何类。`engine.py` 是模块，以后通过下面的路径导入：

```python
from tianyi_micrograd.engine import Value
```

包名使用下划线。项目目录 `tianyi-micrograd` 可以有连字符，但 Python 导入语句不能把它当成包名。

## 2. 先理解节点之间的关系

沿用第 00 课：

```text
a=2 ──┐
      ├── e=a*b=-6 ──┐
b=-3 ─┘              ├── d=e+c=4 ──┐
c=10 ────────────────┘             ├── L=d*f=-8
f=-2 ──────────────────────────────┘
```

每个数值对应一个 `Value` 对象。`e` 保存对输入对象 `a`、`b` 的引用，`d` 保存 `e`、`c` 的引用，以此类推。

图上画的是输入流向输出；实现时，输出通过 `_prev` 找回输入。这样以后从最终输出出发，就可以找到所有参与运算的节点。

这里的“父节点”统一指当前结果的**直接输入**。参考实现把构造参数叫 `_children`，字段叫 `_prev`；不要被不同命名误导。

## 3. 第一步：保存数值与图信息

在 `engine.py` 写入下面这段。后续步骤只向这个类添加方法，不要重新定义另一个同名类。

```python
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
```

### 3.1 类、实例与 self

`class Value:` 定义类，`a = Value(2.0)` 创建一个实例，并调用初始化方法 `__init__`。`self` 指向正在初始化的那个对象。

### 3.2 这些字段各自表示什么

| 字段      | 本课的作用                         |
| --------- | ---------------------------------- |
| `data`  | 保存前向数值，示例中主要使用浮点数 |
| `grad`  | 为后续反传预留，初始为`0`        |
| `_prev` | 保存当前结果的直接输入对象         |
| `_op`   | 保存`+`、`*` 等名称，方便调试  |

`_children=()` 的默认值是空元组，表示直接构造的叶节点没有计算输入。`set(_children)` 创建一个集合；集合会去重，不保证遍历顺序。

单下划线是“主要供内部使用”的命名约定，不是强制访问限制。调试和本课测试可以读取这些字段。

`Value(2.0)` 和另一个 `Value(2.0)` 是两个不同对象。当前没有定义 `__eq__`，它们不会仅因为数值相同就在集合中合并。不要为了比较数值而提前增加 `__eq__`。

### 3.3 先不写 _backward

参考代码的构造函数还包含 `_backward`。那属于下一课的局部反传规则，现在先省略。这个阶段的 `grad` 只是一个初始为零的字段，不会自己计算。

## 4. 第二步：让节点能被看见

在同一个 `Value` 类中，添加：

```python
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
```

`__repr__` 是 Python 的特殊方法，决定 `repr(a)` 的表示形式。在这里，`print(a)` 也会使用这个表示。

例如 `repr(Value(2.0))` 得到：

```text
Value(data=2.0, grad=0)
```

`f"...{self.data}..."` 会把属性值插入字符串。显示梯度是为了以后方便检查，不表示现在已经实现求导。

## 5. 第三步：写加法，得到新的结果节点

继续在 `Value` 类中添加：

```python
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out
```

本课暂时要求两边都是 `Value`：

```python
a = Value(2.0)
b = Value(-3.0)
c = a + b
```

对于这里的同类对象，`a+b` 调用 `a.__add__(b)`，其中 `self` 是 `a`，`other` 是 `b`。

构造结果时的三个参数分别是：

```text
self.data + other.data   输出数值，这里是 -1
(self, other)            两个直接输入对象
"+"                      运算名称
```

这里必须新建一个节点，不能把 `self.data` 改掉。若改变输入，原来的计算历史就被破坏了。

`(self, other)` 是元组，集合构造会保留其中对象的引用，不会复制两份 `Value`。所以 `c._prev` 中找得到原来的 `a`、`b`。

此时 `a+1.0` 和 `1.0+a` 尚未支持；先显式写 `a+Value(1.0)`。数字包装和左右运算协议留到第 04 课。

## 6. 第四步：写乘法

继续在同一个类中添加：

```python
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out
```

它与加法的结构相同，这次只更改数值计算和运算名称。局部导数的差异下一课再讲。

现在可以表达：

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(-2.0)

e = a * b
d = e + c
loss = d * f
```

`e.data=-6.0`，`d.data=4.0`，`loss.data=-8.0`。最后的节点沿 `_prev` 可以追到所有输入。

## 7. 第五步：做一次终端检查

确认前面的方法都放在一个 `Value` 类中后，在终端执行：

```bash
conda activate agent
cd /data/ghs/projects/tianyi-micrograd
PYTHONPATH="$PWD/src" python - <<'PY'
from tianyi_micrograd.engine import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = Value(-2.0)

e = a * b
d = e + c
loss = d * f

print(e)
print(d)
print(loss)
print("loss 的输入数量:", len(loss._prev))
print("d 是 loss 的直接输入:", d in loss._prev)
print("f 是 loss 的直接输入:", f in loss._prev)
PY
```

预期输出：

```text
Value(data=-6.0, grad=0)
Value(data=4.0, grad=0)
Value(data=-8.0, grad=0)
loss 的输入数量: 2
d 是 loss 的直接输入: True
f 是 loss 的直接输入: True
```

`PYTHONPATH="$PWD/src"` 为这次运行增加导入搜索路径，`$PWD` 是当前目录。不需要修改工程配置或安装自己的包。

不要依赖直接打印 `_prev` 时的元素顺序，集合顺序不是接口约定。

## 8. 第六步：手写第一份测试

本课只写两个测试：检查完整前向计算，以及同一个节点重复参与运算的情况。

手动将下面内容写入 `/data/ghs/projects/tianyi-micrograd/test/test_engine.py`；如果已经写了旧版测试，可以替换为这一版。

```python
from tianyi_micrograd.engine import Value


def test_forward():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(-2.0)
    e = a * b
    d = e + c
    loss = d * f

    assert (e.data, d.data, loss.data) == (-6.0, 4.0, -8.0)
    assert e._prev == {a, b}
    assert d._prev == {e, c}
    assert loss._prev == {d, f}
    assert (a.data, b.data, c.data, f.data) == (2.0, -3.0, 10.0, -2.0)
    assert loss.grad == 0


def test_repeated_operand():
    x = Value(3.0)
    out = x + x
    assert out.data == 6.0
    assert out._prev == {x}
```

运行：

```bash
cd /data/ghs/projects/tianyi-micrograd
PYTHONPATH="$PWD/src" python -m pytest test/test_engine.py -q
```

通过时应显示 `2 passed`，以你实际运行的结果为准。

第一个测试同时检查加乘结果、节点连接和输入数值未被修改；第二个测试确认集合会将重复的 `x` 去重。虽然输入集合只有一个节点，下一课反传时仍需要计算两个操作数的梯度贡献。

## 9. 常见问题如何定位

| 现象                                   | 先检查哪里                                                                          |
| -------------------------------------- | ----------------------------------------------------------------------------------- |
| `No module named 'tianyi_micrograd'` | 是否在项目根目录运行、是否传入`PYTHONPATH`、目录和包名是否一致                    |
| `cannot import name 'Value'`         | `engine.py` 是否保存、类名是否正确、是否误导入另一个文件                          |
| `IndentationError`                   | 类方法与方法体的缩进，是否混用 Tab 和空格                                           |
| `unsupported operand type(s)`        | `__add__`、`__mul__` 是否拼写正确、是否放在类内部；是否提前用了未支持的数字混算 |
| 数值对，但`_prev` 测试失败           | 结果构造时是否传入`(self, other)`，是否存了数值而非对象                           |
| 输入数值变了                           | 是否在加乘运算里修改了`self.data` 或 `other.data`                               |
| 看到的`grad` 一直为零                | 这是本课的预期；反传尚未实现                                                        |

要确认到底导入了哪个模块，可以运行：

```bash
PYTHONPATH="$PWD/src" python -c 'import tianyi_micrograd.engine as engine; print(engine.__file__)'
```

预期路径为 `/data/ghs/projects/tianyi-micrograd/src/tianyi_micrograd/engine.py`。先确认运行的是自己的代码，再排查计算逻辑。

## 10. 本课验收与参考阅读

通过测试后，用自己的话解释：

1. `Value(2.0)` 与另一个 `Value(2.0)` 为什么是两个节点？
2. `a+b` 为什么需要新建一个 `Value`？
3. `_prev` 保存的是数值副本，还是输入对象的引用？
4. 为什么结果节点可以通过 `_prev` 找回整个计算历史？
5. `x+x` 的输入集合只有一个对象，为什么后面的梯度仍应是 `2`？

最后再打开参考 [Value 的构造、加法和乘法](/data/ghs/projects/micrograd/micrograd/engine.py:2)，只对照前向部分。你会发现参考版额外做了数字包装，并为每种运算保存了反传函数；这些正是后面的增量。

本课完成后，保留当前代码和测试。下一课在此基础上添加 `_backward`，先手动传递梯度，再讨论如何自动安排执行顺序。
