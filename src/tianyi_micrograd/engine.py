class Value:

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value=(data={self.data} , grad={self.grad})"

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

    def __pow__(self, exponent):
        assert isinstance(
            exponent, (int, float)
        ), "only int/float exponents are supported"

        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward():
            self.grad += out.grad * (exponent * self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward():
            self.grad += out.grad * (out.data > 0)

        out._backward = _backward
        return out

    # -x
    def __neg__(self):
        return self * -1

    # 反向加乘
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    # 用已有运算组成减法和除法
    def __sub__(self, other):   
        return self + (-other)   # x-y = x+(-y)

    def __rsub__(self, other):
        return other + (-self)   # y-x = y+(-x)

    def __truediv__(self, other):
        return self * other**-1   # x/y = x*(y**-1)

    def __rtruediv__(self, other):
        return other * self**-1    # y/x = y*(x**-1)

    def backward(self):
        topo = []  # ​​topo​ 是有顺序的列表，保存最终执行所需的节点排列。
        visited = set()

        def build_topo(node):
            if node in visited:
                return

            visited.add(node)

            for parent in node._prev:
                build_topo(parent)

            topo.append(node)

        build_topo(self)
        self.grad = 1

        for node in reversed(topo):
            node._backward()
