import torch

from multipinn.condition.condition import Condition
from multipinn.condition.diff import grad, unpack
from multipinn.geometry import *


def problem_2D1C_Allen_Cahn():
    input_dim = 2
    output_dim = 1

    def basic_symbols(model, arg):
        f = model(arg)
        (u,) = unpack(f)
        x, t = unpack(arg)
        return f, u, x, t

    def inner(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        u_x, u_t = unpack(grad(u, arg))
        u_xx, u_xt = unpack(grad(u_x, arg))
        # uox, = _unpack(_num_diff_random(model, arg, f, [1, 0]))
        # uot, = _unpack(_num_diff_random(model, arg, f, [0, 1]))
        eq1 = u_t - 1e-4 * u_xx + 5 * (u**3 - u)
        return [eq1]
        # e = 1e-2
        # step = _random_spherical(2) * e
        # aprox = u + step[0] * u_x + step[1] * u_t + step[0] * step[0] * 0.5 * u_xx
        # aprox_res = (aprox - model(arg + step[np.newaxis, :])[:, 0]) * (1./e)
        # sim = u - model(arg * torch.Tensor([-1, 1])[np.newaxis, :])[:, 0]
        # return [eq1, aprox_res, sim]

    def bc1(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x) * -1))
        u_x, u_t = unpack(grad(u, arg))
        return [u + 1]

    def bc2(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(x, torch.ones_like(x)))
        u_x, u_t = unpack(grad(u, arg))
        return [u + 1]

    def ic(model, arg):
        f, u, x, t = basic_symbols(model, arg)
        # assert torch.all(torch.isclose(t, torch.zeros_like(t)))
        return [x**2 * torch.cos(torch.pi * x) - u]
        # return [(x ** 2 * torch.cos(torch.pi * x) - u) * 10]

    domain = Hypercube(low=[-1, 0], high=[1, 1])
    x_min = Hypercube(low=[-1, 0], high=[-1, 1])
    x_max = Hypercube(low=[1, 0], high=[1, 1])
    t_0 = Hypercube(low=[-1, 0], high=[1, 0])
    pde = [
        Condition(inner, domain),
        Condition(bc1, x_min),
        Condition(bc2, x_max),
        Condition(ic, t_0),
    ]
    return pde, input_dim, output_dim
