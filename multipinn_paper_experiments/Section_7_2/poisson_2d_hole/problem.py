import torch

from multipinn.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import Hypercube, Hypersphere, Shell


def exact_solution(arg):
    """Manufactured solution on the square with a circular cutout.

    u = cos(pi x / 2) cos(pi y / 2).
    """
    x, y = unpack(arg)
    return torch.cos(0.5 * torch.pi * x) * torch.cos(0.5 * torch.pi * y)


def poisson_2d_hole():
    input_dim = 2
    output_dim = 1

    def inner(model, arg):
        (u,) = unpack(model(arg))
        u_x, u_y = unpack(grad(u, arg))
        u_xx = unpack(grad(u_x, arg))[0]
        u_yy = unpack(grad(u_y, arg))[1]
        rhs = 2.0 * (0.5 * torch.pi) ** 2 * exact_solution(arg)
        return [-(u_xx + u_yy) - rhs]

    def boundary(model, arg):
        (u,) = unpack(model(arg))
        return [u - exact_solution(arg)]

    square = Hypercube([-1.0, -1.0], [1.0, 1.0])
    hole = Hypersphere([0.0, 0.0], 0.35)
    domain = square - hole
    conditions = [
        Condition(inner, domain),
        Condition(boundary, Shell(domain)),
    ]
    return conditions, input_dim, output_dim
