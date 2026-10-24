from tianyi_micrograd.engine import Value

def test_backward_shared_node():
    x = Value(3.0)
    k = Value(2.0)

    u = x * k 
    q = u * u

    loss = u + q

    loss.backward()

    assert loss.data == 42.0
    assert u.grad == 13.0
    assert x.grad == 26.0

def test_backward_chain():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(-2.0)

    loss = (a * b + c) * f
    loss.backward()

    assert loss.data == -8.0

    assert (a.grad, b.grad, c.grad, f.grad) == (6.0, -4.0, -2.0, 4.0)
    