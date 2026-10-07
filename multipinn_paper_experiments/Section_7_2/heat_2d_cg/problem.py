import torch

from multipinn.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import Hypercube, Hypersphere, Shell

KX = torch.pi / 16.0
KY = torch.pi / 24.0
DECAY = KX**2 + KY**2


def exact_solution(arg):
    """Manufactured heat solution on the perforated plate times t in [0, 3].

    u = exp(-(kx^2 + ky^2) t) cos(kx x) cos(ky y),
    kx = pi / 16, ky = pi / 24.
    """
    x, y, t = unpack(arg)
    return torch.exp(-DECAY * t) * torch.cos(KX * x) * torch.cos(KY * y)


def _union(geometries):
    result = geometries[0]
    for geom in geometries[1:]:
        result = result | geom
    return result


def heat_2d_cg():
    input_dim = 3
    output_dim = 1

    def inner(model, arg):
        (u,) = unpack(model(arg))
        u_x, u_y, u_t = unpack(grad(u, arg))
        u_xx = unpack(grad(u_x, arg))[0]
        u_yy = unpack(grad(u_y, arg))[1]
        return [u_t - u_xx - u_yy]

    def exact_condition(model, arg):
        (u,) = unpack(model(arg))
        return [u - exact_solution(arg)]

    large_centers = [
        (4.0, 3.0),
        (-4.0, -3.0),
        (-4.0, 3.0),
        (4.0, -3.0),
        (4.0, 9.0),
        (-4.0, -9.0),
        (4.0, -9.0),
        (-4.0, 9.0),
        (0.0, 0.0),
        (0.0, 6.0),
        (0.0, -6.0),
    ]
    small_centers = [
        (3.2, 6.0),
        (-3.2, -6.0),
        (3.2, -6.0),
        (-3.2, 6.0),
        (3.2, 0.0),
        (-3.2, 0.0),
    ]
    rect = Hypercube([-8.0, -12.0], [8.0, 12.0])
    large_holes = _union([Hypersphere(center, 1.0) for center in large_centers])
    small_holes = _union([Hypersphere(center, 0.4) for center in small_centers])
    spatial = rect - (large_holes | small_holes)

    domain = spatial * Hypercube([0.0], [3.0])
    initial = spatial * Hypercube([0.0], [0.0])
    bottom = Hypercube([-8.0, -12.0, 0.0], [8.0, 12.0, 0.0])
    top = Hypercube([-8.0, -12.0, 3.0], [8.0, 12.0, 3.0])
    lateral = Shell(domain) - (bottom | top)

    conditions = [
        Condition(inner, domain),
        Condition(exact_condition, initial),
        Condition(exact_condition, lateral),
    ]
    return conditions, input_dim, output_dim
