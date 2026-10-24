from tianyi_micrograd.engine import Value


def test_mixed_ops():
    x = Value(2.0)
    loss = sum([(2*x + 1)**2 / 2, 5-x, -x, 8/x])
    loss += x - 1
    loss.backward()

    assert loss.data == 18.5
    assert x.grad == 7.0



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