from typing import List

import torch
import numpy as np
from multipinn.condition import Condition, Symbols, ConditionExtra
from multipinn.geometry import *


def clein_gordon_problem():

    symbols = Symbols(variables="x, y, z, t", functions="u", mode="diff")
    inner_symbols = symbols("u, u_xx, u_yy, u_zz, u_t, u_tt")
    basic_symbols = symbols("x, y, z, t")
    
    def equation(model, arg):
        u, u_xx, u_yy, u_zz, u_t, u_tt = inner_symbols(model, arg)
        eq = u_tt - u_xx - u_yy - u_zz + 4*u
        return [eq,]

    def initial_func_c(model, arg):
        u, _, _, _, _, _ = inner_symbols(model, arg)
        x, y, z, t =  basic_symbols(model, arg)
        init_func = u - torch.sin(torch.pi * x / 100)*torch.sin(torch.pi * y / 100)*torch.sin(torch.pi * z / 100)
        return[init_func]

    def initial_der_c(model, arg):
        _, _, _, _, u_t, _ = inner_symbols(model, arg)
        return[u_t]

    def border_c(model, arg):
        u, _, _, _, _, _ = inner_symbols(model, arg)
        return [u]

    domain = Hypercube(low=[0, 0, 0], high=[100, 100, 100])
    time_axis = Hypercube(low=[0], high=[5])
    t_0 = Hypercube(low=[0], high=[0])
    domain_time = domain * time_axis
    border_time = Shell(domain).product_back(time_axis)
    initial_domain = domain * t_0

    pde = [
        Condition(equation, domain_time),
        Condition(initial_func_c, initial_domain),
        Condition(initial_der_c, initial_domain),
        Condition(border_c, border_time)
    ]
    return pde, symbols.input_dim, symbols.output_dim
