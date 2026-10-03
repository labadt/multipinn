import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multipinn.generation import Generator


def convection():
    from examples.convection_1D.problem import convection_problem

    return convection_problem(betta=1.0, t_max=1.0)


def heat():
    from examples.heat_2D_1C.problem import problem_2D1C_heat_equation

    return problem_2D1C_heat_equation(a=1, b=0.5, alpha=0.5, beta=10, gamma=0.7)


def allen_cahn():
    from examples.allen_cahn_2D_1C.problem import problem_2D1C_Allen_Cahn

    return problem_2D1C_Allen_Cahn()


def ns_block():
    from examples.navier_stokes_2D_obstacle.problem import problem

    return problem(re=50)


def ns_curved():
    from examples.navier_stokes_2D_pipe.problem import navier_stokes_2D_pipe

    return navier_stokes_2D_pipe(re=100)


def ns_pipe_3d():
    from examples.navier_stokes_3D_pipe.problem import navier_stokes_3D_pipe

    return navier_stokes_3D_pipe(re=100)

TASKS = {
    "convection": (convection, 20_000, 2_000),
    "heat": (heat, 20_000, 2_000),
    "allen_cahn": (allen_cahn, 20_000, 2_000),
    "ns_block": (ns_block, 21_000, 3_000),
    "ns_curved": (ns_curved, 40_000, 2_000),
    "ns_pipe_3d": (ns_pipe_3d, 10_000, 2_000),
}


def build(task, scale=1.0):
    factory, n_domain, n_bound = TASKS[task]
    conditions, input_dim, output_dim = factory()
    Generator(max(1, int(n_domain * scale)), "pseudo").use_for(conditions[0])
    Generator(max(1, int(n_bound * scale)), "pseudo").use_for(conditions[1:])
    return conditions, input_dim, output_dim
