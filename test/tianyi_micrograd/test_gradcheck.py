from math import isclose


from tianyi_micrograd.engine import Value


def test_reference_expression():
    x = Value(-4.0)
    z = 2 * x + 2 + x

    q = z.relu() + z * x

    h = (z * z).relu()

    result: Value = h + q + q * x
    result.backward()

    assert result.data == -20.0

    assert x.grad == 46.0


def objective(a, b):
    return (max(0.0, a * b) + a**3) / (b + 2)


def test_finite_diference():
    h = 1e-6

    for a0, b0 in [(1.5, 0.5), (-1.5, 0.5)]:
        a = Value(a0)
        b = Value(b0)
        loss: Value = ((a * b).relu() + a**3) / (b + 2)

        loss.backward()

        da = (objective(a0 + h , b0) - objective(a0 - h , b0)) / (2 * h) 
        db = (objective(a0 , b0 + h) - objective(a0 , b0 - h)) / (2 * h)

        assert isclose((loss.data) , objective(a0 , b0) , rel_tol=1e-12 , abs_tol=1e-12)

        assert isclose(a.grad , da  , rel_tol=1e-5 , abs_tol=1e-6 )
        assert isclose(b.grad , db  , rel_tol=1e-5 , abs_tol=1e-6 )

