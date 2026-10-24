from math import isclose

import pytest
# import torch 
from tianyi_micrograd.engine import Value

def reference_expression(a, b):
    c = a + b
    d = a * b + b**3
    c = c + (c + 1)
    c = c + (1 + c + (-a))
    d = d + (d * 2 + (b + a).relu())
    d = d + (3 * d + (b - a).relu())
    f = (c - d)**2
    return f / 2.0 + 10.0 / f

def test_torch_reference():
    torch = pytest.importorskip("torch")

    a, b = Value(-4.0) , Value(2.0)

    result:Value = reference_expression(a , b)
    result.backward()

    ta = torch.tensor(-4.0 ,dtype=torch.float64 , requires_grad=True)
    tb = torch.tensor(2.0 ,dtype=torch.float64 , requires_grad=True)

    expected = reference_expression(ta , tb)
    expected.backward()

    for actual, target in [(result.data, expected.item()),
                           (a.grad, ta.grad.item()),
                           (b.grad, tb.grad.item())]:
        assert isclose(actual, target, rel_tol=1e-9, abs_tol=1e-6)