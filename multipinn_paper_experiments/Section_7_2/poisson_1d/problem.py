import torch

from multipinn.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import Hypercube, Shell


def exact_solution(arg):
    """Manufactured solution on the interval [-1, 1]: u = cos(pi x / 2)."""
    x = arg[:, 0]
    return torch.cos(0.5 * torch.pi * x)


def poisson_1d():
    input_dim = 1
    output_dim = 1

    def inner(model, arg):
        (u,) = unpack(model(arg))
        u_x = unpack(grad(u, arg))[0]
        u_xx = unpack(grad(u_x, arg))[0]
        rhs = (0.5 * torch.pi) ** 2 * exact_solution(arg)
        return [-u_xx - rhs]

    def boundary(model, arg):
        (u,) = unpack(model(arg))
        return [u - exact_solution(arg)]

    domain = Hypercube([-1.0], [1.0])
    conditions = [
        Condition(inner, domain),
        Condition(boundary, Shell(domain)),
    ]
    return conditions, input_dim, output_dim
