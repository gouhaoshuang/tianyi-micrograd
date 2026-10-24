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
        
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward


        return out

    def backward(self):
        topo = []    # ​​topo​ 是有顺序的列表，保存最终执行所需的节点排列。
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
            





