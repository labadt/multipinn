import torch

from mesh_zones import geometries_from_grid, mesh_path, require_wall_zones
from multipinn.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.mesh.grid_reader import GridReader


def exact_solution(arg):
    x, y = unpack(arg)
    return torch.cos(0.5 * torch.pi * x) * torch.cos(0.5 * torch.pi * y)


def poisson_2d_hole_mesh(msh=None):
    """Poisson problem on the imported Fluent mesh of [-1, 1]^2 minus a hole.

    Zone ids: 2 interior, 3 outer wall, 4 hole wall. Collocation points are
    face midpoints of those zones.
    """
    input_dim = 2
    output_dim = 1
    path = mesh_path() if msh is None else msh
    grid = GridReader().read(str(path))
    require_wall_zones(grid)
    zones = geometries_from_grid(grid)

    def inner(model, arg):
        (u,) = unpack(model(arg))
        u_x, u_y = unpack(grad(u, arg))
        u_xx = unpack(grad(u_x, arg))[0]
        u_yy = unpack(grad(u_y, arg))[1]
        return [u_xx + u_yy + (0.5 * torch.pi**2) * exact_solution(arg)]

    def dirichlet(model, arg):
        (u,) = unpack(model(arg))
        return [u - exact_solution(arg)]

    conditions = [
        Condition(inner, zones["interior"]),
        Condition(dirichlet, zones["outer_wall"]),
        Condition(dirichlet, zones["hole_wall"]),
    ]
    return conditions, input_dim, output_dim
