import torch

def _to_arg_tensor(value, arg):
    return torch.as_tensor(value, device=arg.device, dtype=arg.dtype)
    
def unpack(batch: torch.Tensor) -> tuple:
    """Unpack a batched tensor into individual components along dimension 1.

    Provides a convenient way to split tensors representing vectors, derivatives,
    or other multi-component quantities into their individual components.

    Args:
        batch (torch.Tensor): Input tensor of shape (batch_size, num_components)

    Returns:
        tuple: Individual components as separate tensors

    Examples:
        >>> x, y, z = unpack(arg)  # Equivalent to arg[:, 0], arg[:, 1], arg[:, 2]
        >>> u, v, w = unpack(f)    # Split solution components
        >>> u_xx, u_xy, u_xz = unpack(grad(u_x, arg))  # Split derivatives
        >>> u_y, v_y, w_y = unpack(num_diff(model, f, arg, direction))  # Numerical derivatives
    """
    return torch.unbind(batch, dim=1)


def grad(u: torch.Tensor, arg: torch.Tensor) -> torch.Tensor:
    """Compute gradient of a scalar function with respect to input coordinates.

    Calculates derivatives using PyTorch's autograd functionality. Returns gradients
    suitable for use in differential equations.

    Args:
        u (torch.Tensor): Scalar function values, shape (batch_size,)
        arg (torch.Tensor): Points at which to evaluate gradient, shape (batch_size, dim)

    Returns:
        torch.Tensor: Gradient values, shape (batch_size, dim)

    Notes:
        IMPORTANT: Always use complete arg tensor - never slice it (e.g., arg[:, 0]).
        This ensures proper gradient computation and doesn't save memory.

    Examples:
        >>> x, y, z = unpack(arg)
        >>> u_x, u_y, u_z = unpack(grad(u, arg))  # First derivatives
        >>> u_yx, u_yy, u_yz = unpack(grad(u_y, arg))  # Second derivatives
    """
    return torch.autograd.grad(u, arg, torch.ones_like(u), create_graph=True)[0]


def num_diff(
    model: torch.nn.Module,
    arg: torch.Tensor,
    fns: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 2**-13,
) -> torch.Tensor:
    """Compute directional derivative using finite differences.

    Calculates numerical approximation of the derivative along a specified direction
    using forward differences.

    Args:
        model (torch.nn.Module): Neural network model
        arg (torch.Tensor): Input points, shape (batch_size, input_dim)
        fns (torch.Tensor): Model output at arg, shape (batch_size, output_dim)
        direction (torch.Tensor): Direction vectors, shape (batch_size, input_dim) or (1, input_dim)
        eps (float, optional): Step size. Defaults to 2**-13.

    Returns:
        torch.Tensor: Directional derivatives, shape (batch_size, output_dim)

    Notes:
        Requires fns == model(arg) for efficiency. Direction tensor can be broadcasted.

    Examples:
        >>> normal = torch.tensor([[0, 1, 0]])  # y direction
        >>> u_y, v_y, w_y = unpack(num_diff(model, arg, fns, normal))
    """
    direction = _to_arg_tensor(direction, arg)
    eps = _to_arg_tensor(eps, arg).reshape(-1, 1)
    second = arg + (direction * eps).detach()
    return (model(second) - fns) / eps


def num_diff_random(
    model: torch.nn.Module,
    arg: torch.Tensor,
    fns: torch.Tensor,
    direction: torch.Tensor,
    max_eps: float = 1e-2,
    min_eps: float = 1e-4,
) -> torch.Tensor:
    """Compute directional derivative with randomized step sizes.

    Like num_diff, but uses random step sizes and directions for better
    numerical stability.

    Args:
        model (torch.nn.Module): Neural network model
        arg (torch.Tensor): Input points
        fns (torch.Tensor): Model output at arg
        direction (torch.Tensor): Base direction vectors
        max_eps (float, optional): Maximum step size. Defaults to 1e-2.
        min_eps (float, optional): Minimum step size. Defaults to 1e-4.

    Returns:
        torch.Tensor: Directional derivatives
    """
    n = len(arg)
    sign = 1 - 2 * torch.randint(0, 2, (n,), device=arg.device, dtype=torch.int64)
    eps = torch.empty(n, device=arg.device, dtype=arg.dtype).uniform_(min_eps, max_eps)
    return num_diff(model, arg, fns, direction, sign.to(arg.dtype) * eps)


def num_diff_second_same(
    model: torch.nn.Module,
    arg: torch.Tensor,
    fns: torch.Tensor,
    direction: torch.Tensor,
    eps: float = 2**-7,
) -> torch.Tensor:
    """Compute second derivative along a single direction.

    Uses a four-point scheme for accurate approximation of pure second derivatives.

    Args:
        model (torch.nn.Module): Neural network model
        arg (torch.Tensor): Input points
        fns (torch.Tensor): Model output at arg
        direction (torch.Tensor): Direction vectors
        eps (float, optional): Step size. Defaults to 2**-7.

    Returns:
        torch.Tensor: Second derivatives
    """
    direction = _to_arg_tensor(direction, arg)
    eps = _to_arg_tensor(eps, arg).reshape(-1, 1)
    step = (direction * eps).detach()
    f_plus = model(arg + step)
    f_minus = model(arg - step)
    return __four_point_scheme(f_plus, fns, fns, f_minus, eps**2)


def num_diff_second_cross(
    model: torch.nn.Module,
    arg: torch.Tensor,
    direction1: torch.Tensor,
    direction2: torch.Tensor,
    eps1: float = 2**-7,
    eps2: float = 2**-7,
) -> torch.Tensor:
    """Compute mixed second derivatives along two directions.

    Uses a four-point scheme for accurate approximation of mixed derivatives.

    Args:
        model (torch.nn.Module): Neural network model
        arg (torch.Tensor): Input points
        direction1 (torch.Tensor): First direction vectors
        direction2 (torch.Tensor): Second direction vectors
        eps1 (float, optional): Step size for first direction. Defaults to 2**-7.
        eps2 (float, optional): Step size for second direction. Defaults to 2**-7.

    Returns:
        torch.Tensor: Mixed second derivatives
    """
    direction1 = _to_arg_tensor(direction1, arg)
    direction2 = _to_arg_tensor(direction2, arg)
    eps1 = _to_arg_tensor(eps1, arg).reshape(-1, 1)
    eps2 = _to_arg_tensor(eps2, arg).reshape(-1, 1)

    step1 = (direction1 * eps1).detach()
    step2 = (direction2 * eps2).detach()

    f_plus_plus = model(arg + step1 + step2)
    f_plus_minus = model(arg + step1 - step2)
    f_minus_plus = model(arg - step1 + step2)
    f_minus_minus = model(arg - step1 - step2)

    return __four_point_scheme(
        f_plus_plus, f_plus_minus, f_minus_plus, f_minus_minus, eps1 * eps2
    )

def num_laplace(
    model: torch.nn.Module, arg: torch.Tensor, fns: torch.Tensor, eps: float = 2**-7
) -> torch.Tensor:
    """Compute Laplacian using spherical averaging method.

    Implements method from:
    https://isis2.cc.oberlin.edu/physics/dstyer/Electrodynamics/Laplacian.pdf

    Args:
        model (torch.nn.Module): Neural network model
        arg (torch.Tensor): Input points
        fns (torch.Tensor): Model output at arg
        eps (float, optional): Step size. Defaults to 2**-7.

    Returns:
        torch.Tensor: Laplacian values

    Examples:
        >>> u, v, p = unpack(fns)
        >>> laplace_u, laplace_v, laplace_p = unpack(num_laplace(model, arg, fns))
    """
    dim = arg.shape[1]
    vectors, _ = torch.linalg.qr(torch.randn((dim, dim), device=arg.device, dtype=arg.dtype))
    eps = torch.as_tensor(eps, device=arg.device, dtype=arg.dtype)
    sum_on_sphere = torch.zeros_like(fns)
    for i in range(dim):
        step = (vectors[i : i + 1, :] * eps).detach()
        sum_on_sphere += model(arg - step) - 2 * fns + model(arg + step)
    return sum_on_sphere / (eps * eps)


def _diff_residual(
    model: torch.nn.Module, arg: torch.Tensor, eps: float = 1e-4
) -> list:
    """Compare analytical and numerical derivatives for testing.

    Generates residuals between automatic differentiation and finite difference
    approximations to verify derivative computation accuracy.

    Args:
        model (torch.nn.Module): Neural network model
        arg (torch.Tensor): Input points
        eps (float, optional): Step size. Defaults to 1e-4.

    Returns:
        list: Residuals for each input-output derivative pair
    """
    f = model(arg)
    analytical = []
    for out_dim in range(f.shape[1]):
        analytical.append(grad(f[:, out_dim], arg))

    numerical = []
    for in_dim in range(arg.shape[1]):
        one_hot = torch.nn.functional.one_hot(
            torch.Tensor([in_dim]).type(torch.int64), num_classes=arg.shape[1]
        )
        left = arg - one_hot * eps
        right = arg + one_hot * eps
        numerical.append((model(right) - model(left)) * (0.5 / eps))

    result = []
    for out_dim in range(f.shape[1]):
        for in_dim in range(arg.shape[1]):
            result.append(
                analytical[out_dim][:, in_dim] - numerical[in_dim][:, out_dim]
            )
    return result


def __four_point_scheme(
    f_plus_plus: torch.Tensor,
    f_plus_minus: torch.Tensor,
    f_minus_plus: torch.Tensor,
    f_minus_minus: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    """Helper function implementing four-point finite difference scheme.

    Args:
        f_plus_plus: f(x+h1, y+h2)
        f_plus_minus: f(x+h1, y-h2)
        f_minus_plus: f(x-h1, y+h2)
        f_minus_minus: f(x-h1, y-h2)
        denom: Denominator for scaling (typically h1*h2)

    Returns:
        torch.Tensor: Finite difference approximation
    """
    return (f_plus_plus - f_plus_minus - f_minus_plus + f_minus_minus) / denom
