from typing import Callable, List, Sequence, Union

import torch


class Condition:
    """Base class for representing Physics-Informed Neural Network (PINN) conditions.

    Handles boundary conditions, initial conditions, and governing equations in PINN models.
    Manages point generation, batching, and residual computation for training.

    Attributes:
        function (Callable): Function that computes residuals
        geometry (Geometry): Geometric domain for the condition
        generator (Generator): Point generator for sampling domain
        points (torch.Tensor): Current set of evaluation points
        batch_size (int): Size of batches for training
        batch_points (torch.Tensor): Current batch of points
        output_len (int): Length of function output

    Example:
        >>> def boundary_condition(model, x):
        ...     return model(x) - exact_solution(x)
        >>> condition = Condition(boundary_condition, domain)
    """

    def __init__(self, function: Callable, geometry: "Geometry") -> None:
        """Initialize a condition.

        Args:
            function (Callable): Function computing residuals, with signature:
                function(model: torch.nn.Module, arg: torch.Tensor) -> List[torch.Tensor]
            geometry (Geometry): Geometric domain where condition applies
        """
        self.function = function
        self.geometry = geometry
        self.generator = None

        # Runtime attributes
        self.points = None
        self.batch_size = None
        self.batch_points = None
        self.output_len = None

    def update_points(self, model=None) -> None:
        """Update evaluation points using the generator.

        Args:
            model (torch.nn.Module, optional): Current model state for adaptive sampling
        """
        self.points = self.generator.generate(self, model)

    def select_batch(self, i: int) -> None:
        """Select batch i from current points.

        Args:
            i (int): Batch index
        """
        first_element = i * self.batch_size
        self.batch_points = self.points[first_element : first_element + self.batch_size]

    def get_residual(self, model) -> List[torch.Tensor]:
        """Compute residuals for current batch.

        Args:
            model (torch.nn.Module): Current model state

        Returns:
            List[torch.Tensor]: List of residual tensors
        """
        return self.function(model, self.batch_points)

    def get_residual_fn(self, model) -> Callable:
        """Create a residual function for arbitrary points.

        Args:
            model (torch.nn.Module): Current model state

        Returns:
            Callable: Function that computes residuals for given points
        """

        def residual(arg):
            return self.function(model, arg)

        return residual

    def set_batching(self, num_batches: int) -> None:
        """Configure batch size based on number of batches.

        Args:
            num_batches (int): Number of batches to create

        Raises:
            AssertionError: If points cannot be evenly divided into batches
        """
        assert (
            self.generator.n_points % num_batches == 0
        ), "Points must divide evenly into batches"
        self.batch_size = self.generator.n_points // num_batches

    def init_output_len(self, model) -> None:
        """Initialize output length by evaluating function at domain center.

        Args:
            model (torch.nn.Module): Current model state
        """
        # Get device from model parameters
        device = next(model.parameters()).device
        low = torch.as_tensor(self.geometry.bbox[0], dtype=torch.float32, device=device)
        high = torch.as_tensor(self.geometry.bbox[1], dtype=torch.float32, device=device)
        arg_point = ((low + high) * 0.5).reshape(1, -1).requires_grad_()
        self.output_len = len(self.get_residual_fn(model)(arg_point))


class ConditionExtra(Condition):
    """Extended condition class supporting additional data generators.

    Allows conditions to use auxiliary data (e.g., normal vectors) in residual
    computation. Useful for complex boundary conditions or constraints that
    depend on geometric properties.

    Example:
        >>> condition = ConditionExtra(neumann_bc, domain, ["normals"])
        >>> # Will compute normal vectors along with residuals
    """

    def __init__(
        self,
        function: Callable,
        geometry: "Geometry",
        data_gen: List[Union[str, Callable]],
    ):
        """Initialize an extended condition.

        Args:
            function (Callable): Function computing residuals, with signature:
                function(model: torch.nn.Module, arg: torch.Tensor, data: tuple) -> List[torch.Tensor]
            geometry (Geometry): Geometric domain where condition applies
            data_gen (List[Union[str, Callable]]): List of data generators or special keywords
                Currently supports "normals" as a special keyword for boundary normals
        """
        super().__init__(function, geometry)
        if "normals" in data_gen:  # Special handling for normal vectors
            assert hasattr(
                geometry, "boundary_normal"
            ), "Geometry must support normal vectors"
            data_gen[data_gen.index("normals")] = ConditionExtra.generator_for_normals(
                geometry
            )

        self.data_gen = lambda points: tuple(fn(points) for fn in data_gen)
        self.data = None
        self.batch_data = None

    @staticmethod
    def generator_for_normals(geometry) -> Callable:
        """Create a generator for boundary normal vectors.

        Args:
            geometry (Geometry): Geometry to compute normals for

        Returns:
            Callable: Function that computes normal vectors for given points
        """
        return lambda points: torch.tensor(
            geometry.boundary_normal(points.detach().cpu().numpy()),
            device=points.device
        )

    def update_points(self, model=None) -> None:
        """Update points and compute associated auxiliary data.

        Args:
            model (torch.nn.Module, optional): Current model state for adaptive sampling
        """
        self.points = self.generator.generate(self, model)
        self.data = self.data_gen(self.points)

    def select_batch(self, i: int) -> None:
        """Select batch i from current points and auxiliary data.

        Args:
            i (int): Batch index
        """
        first_element = i * self.batch_size
        self.batch_points = self.points[first_element : first_element + self.batch_size]
        self.batch_data = self.data[first_element : first_element + self.batch_size]

    def get_residual(self, model) -> List[torch.Tensor]:
        """Compute residuals using points and auxiliary data.

        Args:
            model (torch.nn.Module): Current model state

        Returns:
            List[torch.Tensor]: List of residual tensors
        """
        return self.function(model, self.batch_points, self.batch_data)

    def get_residual_fn(self, model) -> Callable:
        """Create a residual function that includes auxiliary data computation.

        Args:
            model (torch.nn.Module): Current model state

        Returns:
            Callable: Function that computes residuals with auxiliary data
        """

        def residual(arg):
            return self.function(model, arg, self.data_gen(arg))

        return residual
