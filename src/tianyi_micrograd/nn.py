import random

from tianyi_micrograd.engine import Value


class Module:
    def parameters(self):
        return []

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0


class Neuron(Module):
    def __init__(self, nin, nonlin=True):  # nin 输入数量
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin  # 是否启用ReLU

    def parameters(self):
        return self.w + [self.b]

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def __repr__(self):
        kind = "ReLU" if self.nonlin else "Linear"
        return f"{kind}Neuron ({len(self.w)})"

