import torch
import numpy as np
from multipinn.condition import Condition, Symbols
from multipinn.geometry import *


def sphere():
    symbols = Symbols(variables="x, y, z", functions="u", mode="diff")

    def solution(points):
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        return torch.sin(x) * torch.sin(y) * torch.sin(z)

    inner_symbols = symbols(
        "u, u_x, u_y, u_z, u_xx, u_yy, u_zz, u_xy, u_xz, u_yz")
    basic_symbols = symbols("x, y, z")

    pols_base = torch.tensor(
    [
        [0., 0., 1.],
        [0., 0., -1.],
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [np.sqrt(0.5), 0.5, 0.5],
        [0.5, np.sqrt(0.5), 0.5],
        [0.5, 0.5, np.sqrt(0.5)],
        [-np.sqrt(0.5), 0.5, 0.5],
        [0.5, -np.sqrt(0.5), 0.5],
        [0.5, 0.5, -np.sqrt(0.5)],
    ],
    dtype=torch.float32)

    def surface(model, arg):
        u, u_x, u_y, u_z, u_xx, u_yy, u_zz, u_xy, u_xz, u_yz = inner_symbols(
            model, arg)
        x, y, z = basic_symbols(model, arg)
        eq1 = (1-x**2)*u_xx + (1-y**2)*u_yy + (1-z**2)*u_zz - 2*x*y*u_xy - 2*x*z*u_xz - 2*y*z*u_yz - 2*x*u_x - 2*y*u_y - 2*z*u_z - ((1-x**2)*u_x - x*y*u_y - x*z*u_z - x*y*u_x + (1-y**2)*u_y - y*z*u_z - x*z*u_x - y*z*u_y + (1-z**2)*u_z) + 5*u \
            - ((1-x**2)*torch.sin(x)*torch.sin(y)*torch.sin(z)*(-1) + (1-y**2)*torch.sin(x)*torch.sin(y)*torch.sin(z)*(-1) + (1-z**2)*torch.sin(x)*torch.sin(y)*torch.sin(z)*(-1) - 2*x*y*torch.cos(x)*torch.cos(y)*torch.sin(z) - 2*x*z*torch.cos(x)*torch.sin(y)*torch.cos(z) - 2*y*z*torch.sin(x)*torch.cos(y)*torch.cos(z) - 2*x*torch.cos(x)*torch.sin(y) * torch.sin(z) - 2*y*torch.sin(x)*torch.cos(y)*torch.sin(z) - 2*z*torch.sin(x)*torch.sin(y)*torch.cos(z) - ((1-x**2)
               * torch.cos(x)*torch.sin(y)*torch.sin(z) - x*y*torch.sin(x)*torch.cos(y)*torch.sin(z) - x*z*torch.sin(x)*torch.sin(y)*torch.cos(z) - x*y*torch.cos(x)*torch.sin(y)*torch.sin(z) + (1-y**2)*torch.sin(x)*torch.cos(y)*torch.sin(z) - y*z*torch.sin(x)*torch.sin(y)*torch.cos(z) - x*z*torch.cos(x)*torch.sin(y)*torch.sin(z) - y*z*torch.sin(x)*torch.cos(y)*torch.sin(z) + (1-z**2)*torch.sin(x)*torch.sin(y)*torch.cos(z)) + 5*torch.sin(x)*torch.sin(y)*torch.sin(z))
        return [eq1,]

    def points_c(model, arg):
        pols = pols_base.to(device=arg.device, dtype=arg.dtype)
        u = model(pols)
        return [abs(u.flatten() - solution(pols).flatten())]

    domain = Hypersphere(center=[0.0, 0.0, 0.0], radius=1)

    pde = [
        Condition(surface, Shell(domain)),
        Condition(points_c, domain)
    ]
    return pde, symbols.input_dim, symbols.output_dim
