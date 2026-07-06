from typing import List

import torch
import numpy as np
from multipinn.condition import Condition, Symbols, ConditionExtra
from multipinn.geometry import *


def fisher_problem():

    symbols = Symbols(variables="x, y, z, t", functions="u", mode="diff")
    inner_symbols = symbols("u, u_xx, u_yy, u_zz, u_x, u_y, u_z, u_t")
    basic_symbols = symbols("x, y, z, t")

    def equation(model, arg):
        u, u_xx, u_yy, u_zz, _, _, _, u_t = inner_symbols(model, arg)
        eq = u_t - 0.5*u_xx - 0.5*u_yy - 0.5*u_zz - 0.05*u*(1 - u/50)
        return [eq,]

    def initial_func_c(model, arg):
        u, _, _, _, _, _, _, _ = inner_symbols(model, arg)
        x, y, z, t = basic_symbols(model, arg)
        init_func = u - 5*torch.exp((-1)*((x-50)**2+(y-50)**2+(z-50)**2)/50)
        return [init_func]

    def bc(model, arg, data):
        u, _, _, _, u_x, u_y, u_z, u_t = inner_symbols(model, arg)
        normal = data[0]
        norm_der = u_x * normal[:, 0] + u_y * normal[:, 1] + u_z * normal[:, 2]
        return [norm_der]

    domain = Hypercube(low=[0, 0, 0], high=[100, 100, 100])
    time_axis = Hypercube(low=[0], high=[100])
    t_0 = Hypercube(low=[0], high=[0])
    domain_time = domain * time_axis
    border_time = Shell(domain).product_back(time_axis)
    initial_domain = domain * t_0

    pde = [
        Condition(equation, domain_time),
        Condition(initial_func_c, initial_domain),
        ConditionExtra(bc, border_time, ["normals"]),
    ]
    return pde, symbols.input_dim, symbols.output_dim
