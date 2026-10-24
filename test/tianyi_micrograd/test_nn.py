from tianyi_micrograd.engine import Value
from tianyi_micrograd.nn import Neuron


def test_linear_neuron_and_zero_grad():
    neuron = Neuron(3, nonlin=False)

    params = neuron.parameters()

    assert len(params) == 4
    assert params[-1] == neuron.b

    values = [2.0, -3.0, 0.5, 1.0]
    for parameter, value in zip(params, values):
        parameter.data = value

    out = neuron([1.0, 2.0, -2.0])
    out.backward()
    assert out.data == -4
    assert [p.grad for p in params] == [1.0, 2.0, -2.0, 1]

    neuron.zero_grad()
    assert [p.grad for p in params] == [0, 0, 0, 0]
    assert [p.data for p in params] == values


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
