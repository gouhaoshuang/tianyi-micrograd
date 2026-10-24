from tianyi_micrograd.nn import MLP, Layer
from tianyi_micrograd.engine import Value


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