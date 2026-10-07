import torch

from multipinn.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import (
    DomainAxisymmetricExtension,
    DomainPermute,
    Hypercube,
    Shell,
)


def exact_solution(arg):
    """Manufactured solution on the cube with a cylindrical cutout.

    u = cos(pi x / 2) cos(pi y / 2) cos(pi z / 2).
    """
    x, y, z = unpack(arg)
    return (
        torch.cos(0.5 * torch.pi * x)
        * torch.cos(0.5 * torch.pi * y)
        * torch.cos(0.5 * torch.pi * z)
    )


def _cylinder_along_z(radius=0.35):
    # Revolve a rectangle in the (z, r) plane, then align the axis with z.
    cylinder_z_r = Hypercube([-1.0, 0.0], [1.0, radius])
    cylinder_z_y_x = DomainAxisymmetricExtension(cylinder_z_r, axis=1)
    return DomainPermute(cylinder_z_y_x, (1, 2, 0))


def poisson_3d_cylinder():
    input_dim = 3
    output_dim = 1

    def inner(model, arg):
        (u,) = unpack(model(arg))
        u_x, u_y, u_z = unpack(grad(u, arg))
        u_xx = unpack(grad(u_x, arg))[0]
        u_yy = unpack(grad(u_y, arg))[1]
        u_zz = unpack(grad(u_z, arg))[2]
        rhs = 3.0 * (0.5 * torch.pi) ** 2 * exact_solution(arg)
        return [-(u_xx + u_yy + u_zz) - rhs]

    def boundary(model, arg):
        (u,) = unpack(model(arg))
        return [u - exact_solution(arg)]

    cube = Hypercube([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    domain = cube - _cylinder_along_z(radius=0.35)
    conditions = [
        Condition(inner, domain),
        Condition(boundary, Shell(domain)),
    ]
    return conditions, input_dim, output_dim
