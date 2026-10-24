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

class Layer(Module):
    def __init__(self , nin , nout , **kwargs):
        self.neurons = [  Neuron(nin , **kwargs) for _ in range(nout)]

    def __call__(self,x):
        out = [ neuron(x) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [ p for neuron in self.neurons  for p in neuron.parameters()]

    def __repr__(self):
        items = ", ".join(str(neuron) for neuron in self.neurons)
        return f"Layer of [{items}]"

class MLP(Module):
    def __init__(self, nin , nouts):
        sizes = [nin] + nouts
        self.layers = [
            Layer(sizes[i] ,sizes[i + 1] , nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [ p for layer in self.layers  for p in layer.parameters()]

    def  __repr__(self):
        items = ", ".join( (str(layer)) for layer in self.layers)
        return f"MLP of [{items}]"