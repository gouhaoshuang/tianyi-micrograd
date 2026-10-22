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
    