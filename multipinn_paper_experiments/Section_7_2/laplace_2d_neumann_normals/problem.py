import torch

from multipinn.condition import Condition, ConditionExtra
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import Hypercube, Shell


def exact_solution(arg):
    """Harmonic field u = x^2 - y^2 on the square [-1, 1]^2."""
    x, y = unpack(arg)
    return x**2 - y**2


def laplace_2d_neumann_normals():
    input_dim = 2
    output_dim = 1

    domain = Hypercube([-1.0, -1.0], [1.0, 1.0])
    boundary = Shell(domain)
    top = Hypercube([-1.0, 1.0], [1.0, 1.0])
    neumann_boundary = boundary - top
    dirichlet_boundary = boundary & top

    def inner(model, arg):
        (u,) = unpack(model(arg))
        u_x, u_y = unpack(grad(u, arg))
        u_xx = unpack(grad(u_x, arg))[0]
        u_yy = unpack(grad(u_y, arg))[1]
        return [u_xx + u_yy]

    def neumann(model, arg, data):
        (u,) = unpack(model(arg))
        x, y = unpack(arg)
        normals = data[0].to(device=arg.device, dtype=arg.dtype)
        u_x, u_y = unpack(grad(u, arg))
        predicted_flux = u_x * normals[:, 0] + u_y * normals[:, 1]
        exact_flux = 2.0 * x * normals[:, 0] - 2.0 * y * normals[:, 1]
        return [predicted_flux - exact_flux]

    def dirichlet(model, arg):
        (u,) = unpack(model(arg))
        return [u - exact_solution(arg)]

    conditions = [
        Condition(inner, domain),
        ConditionExtra(neumann, neumann_boundary, ["normals"]),
        Condition(dirichlet, dirichlet_boundary),
    ]
    return conditions, input_dim, output_dim
