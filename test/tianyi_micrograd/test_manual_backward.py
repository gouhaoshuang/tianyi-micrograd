from tianyi_micrograd.engine import Value

def test_backgrad_hand():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(-2.0)

    e = a * b
    d = e + c
    loss = d * f


    loss.grad = 1

    loss._backward()
    print("loss 反向传播 ， d = 4 , f = -2 , loss = -8")
    print(f"loss.grad = {loss.grad}")
    print(f"d.grad = {d.grad}")
    print(f"f.grad = {f.grad}")
    print()

    d._backward()

    print("d 反向传播 ， d = e + c , 4 = -6 + 10")
    print(f"loss.grad = {loss.grad}")
    print(f"d.grad = {d.grad}")
    print(f"e.grad = {e.grad}")
    print(f"c.grad = {c.grad}")
    print()


    e._backward()

    print("e 反向传播 ， e = a * b , -6 = 2 * -3")
    print(f"loss.grad = {loss.grad}")
    print(f"e.grad = {e.grad}")
    print(f"a.grad = {a.grad}")
    print(f"b.grad = {b.grad}")
    print()


    print("L =", loss.data)

    for name, node in (("loss", loss),("a", a), ("b", b), ("c", c), ("f", f)):
        print(f"{name}.grad = {node.grad}")



def test_manual_backward():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    f = Value(-2.0)
    e = a * b
    d = e + c
    loss = d * f

    loss.grad = 1
    loss._backward()
    d._backward()
    e._backward()

    assert (a.grad, b.grad, c.grad, f.grad) == (6.0, -4.0, -2.0, 4.0)


def test_repeated_operand_backward():
    x = Value(3.0)
    out = x + x
    out.grad = 1
    out._backward()
    assert x.grad == 2.0

    x = Value(3.0)
    out = x * x
    out.grad = 1
    out._backward()
    assert x.grad == 6.0