import abc
from typing import Union

import numpy as np

from .domain import Domain
from .geometry import Geometry


class BaseShell(Geometry):
    """Abstract base class for shell geometries.

    A shell represents the boundary of a geometric domain. This class provides
    the foundation for creating and manipulating shell geometries.

    Args:
        dim (int): Dimension of the geometry
        bbox (tuple): Bounding box as ((min_x, min_y, ...), (max_x, max_y, ...))
        diam (float): Characteristic diameter/size of the geometry
    """

    def __init__(self, dim, bbox, diam):
        super().__init__(dim, bbox, diam)

    @abc.abstractmethod
    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        """Check if points are on the shell boundary.

        Args:
            x (np.ndarray): Points to check with shape (n, dim)

        Returns:
            Union[np.ndarray, bool]: Boolean mask indicating which points are on boundary
        """
        raise NotImplementedError

    @abc.abstractmethod
    def boundary_normal(self, x):
        """Compute unit normal vectors at given boundary points.

        Args:
            x (np.ndarray): Points on boundary with shape (n, dim)

        Returns:
            np.ndarray: Normal vectors at each point with shape (n, dim)
        """
        raise NotImplementedError

    def union(self, other: "BaseShell"):
        """Create union of this shell with another shell.

        Args:
            other (BaseShell): Shell to unite with

        Returns:
            ShellUnion: New shell representing the union
        """
        return ShellUnion(self, other)

    def __or__(self, other: "BaseShell"):
        return self.union(other)

    def difference(self, other: Domain):
        return ShellDifference(self, other)

    def __sub__(self, other: Domain):
        return self.difference(other)

    def intersection(self, other: Domain):
        return ShellIntersection(self, other)

    def __and__(self, other: Domain):
        return self.intersection(other)

    def product_back(self, other: Domain):
        return ProductShellDomain(self, other)

    def product_front(self, other: Domain):
        return ProductDomainShell(other, self)


class Shell(BaseShell):
    """Shell representing the boundary of a given domain.

    This class wraps a Domain object and represents its boundary as a shell.

    Args:
        geometry (Domain): The domain whose boundary forms this shell
    """

    def __init__(self, geometry: Domain):
        super().__init__(geometry.dim, geometry.bbox, geometry.diam)
        self.geom = geometry

    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        """Check if points are on the shell boundary.

        Args:
            x (np.ndarray): Points to check with shape (n, dim)

        Returns:
            Union[np.ndarray, bool]: Boolean mask indicating which points are on boundary
        """
        return self.geom.on_boundary(x)

    def random_points(self, n, random="pseudo"):
        """Generate random points on the shell boundary.

        Args:
            n (int): Number of points to generate
            random (str, optional): Random number generation method. Options are:
                - "pseudo": Pseudorandom sampling (default)
                - "LHS": Latin Hypercube Sampling
                - "Halton": Halton sequence
                - "Hammersley": Hammersley sequence
                - "Sobol": Sobol sequence

        Returns:
            np.ndarray: Array of random boundary points with shape (n, dim)
        """
        return self.geom.random_boundary_points(n, random)

    def boundary_normal(self, x):
        """Compute unit normal vectors at given boundary points.

        Args:
            x (np.ndarray): Points on boundary with shape (n, dim)

        Returns:
            np.ndarray: Normal vectors at each point with shape (n, dim)
        """
        return self.geom.boundary_normal(x)


class ShellUnion(BaseShell):
    """Union of two shells.

    Represents the union of two shell geometries, preserving their boundaries
    where they don't overlap.

    Args:
        shell (BaseShell): First shell
        shell_other (BaseShell): Second shell

    Raises:
        ValueError: If shells have different dimensions
    """

    def __init__(self, shell: BaseShell, shell_other: BaseShell):
        if shell.dim != shell_other.dim:
            raise ValueError("Dimensions do not match")
        super().__init__(
            shell.dim,
            (
                np.minimum(shell.bbox[0], shell_other.bbox[0]),
                np.maximum(shell.bbox[1], shell_other.bbox[1]),
            ),
            shell.diam + shell_other.diam,
        )
        self.s1 = shell
        self.s2 = shell_other

    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        """Check if points are on either shell's boundary.

        Args:
            x (np.ndarray): Points to check with shape (n, dim)

        Returns:
            Union[np.ndarray, bool]: Boolean mask indicating which points are on boundary
        """
        return np.logical_or(self.s1.on_boundary(x), self.s2.on_boundary(x))

    def random_points(self, n, random="pseudo"):
        """Generate random points on the union of shell boundaries.

        Points are distributed proportionally to the relative sizes of the shells.

        Args:
            n (int): Number of points to generate
            random (str, optional): Random number generation method

        Returns:
            np.ndarray: Array of random boundary points with shape (n, dim)
        """
        n1 = round(n * 1 / (1 + (self.s2.diam / self.s1.diam) ** (self.dim - 1)))
        n2 = n - n1
        return np.concatenate((self.s1.random_points(n1), self.s2.random_points(n2)))

    def boundary_normal(self, x):
        """Compute unit normal vectors at given boundary points.

        For points on both shells' boundaries, randomly selects which normal to use.

        Args:
            x (np.ndarray): Points on boundary with shape (n, dim)

        Returns:
            np.ndarray: Normal vectors at each point with shape (n, dim)
        """
        normals1 = self.s1.boundary_normal(x)
        normals2 = self.s2.boundary_normal(x)
        first = np.logical_and(self.s1.on_boundary(x), ~self.s2.on_boundary(x))
        second = np.logical_and(self.s2.on_boundary(x), ~self.s1.on_boundary(x))
        choice = np.where(
            first,
            True,
            np.where(second, False, np.random.choice([True, False], size=x.shape[0])),
        )
        return np.where(choice[:, np.newaxis], normals1, normals2)


class ShellDifference(BaseShell):
    """Difference between a shell and a domain.

    Represents the part of a shell that does not intersect with a given domain.

    Args:
        shell (BaseShell): The shell to subtract from
        geom (Domain): The domain to subtract

    Raises:
        ValueError: If shell and domain have different dimensions
    """

    def __init__(self, shell: BaseShell, geom: Domain):
        if shell.dim != geom.dim:
            raise ValueError("Dimensions do not match")
        super().__init__(shell.dim, shell.bbox, shell.diam)
        self.shell = shell
        self.geom2 = geom

    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        """Check if points are on the difference boundary.

        Points are on the boundary if they are on the shell boundary
        and outside the subtracted domain.

        Args:
            x (np.ndarray): Points to check with shape (n, dim)

        Returns:
            Union[np.ndarray, bool]: Boolean mask indicating which points are on boundary
        """
        return np.logical_and(self.shell.on_boundary(x), ~self.geom2.inside(x))

    def random_points(self, n, random="pseudo"):
        """Generate random points on the difference boundary.

        Args:
            n (int): Number of points to generate
            random (str, optional): Random number generation method

        Returns:
            np.ndarray: Array of random boundary points with shape (n, dim)
        """
        x = np.empty(shape=(n, self.geom2.dim), dtype="float32")
        i = 0
        while i < n:
            tmp = self.shell.random_points(n, random=random)
            tmp = tmp[~self.geom2.inside(tmp)]
            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def boundary_normal(self, x):
        """Compute unit normal vectors at given boundary points.

        Args:
            x (np.ndarray): Points on boundary with shape (n, dim)

        Returns:
            np.ndarray: Normal vectors at each point with shape (n, dim)
        """
        return self.shell.boundary_normal(x)


class ShellIntersection(BaseShell):
    """Intersection between a shell and a domain.

    Represents the part of a shell that lies within a given domain.

    Args:
        shell (BaseShell): The shell to intersect
        geom (Domain): The domain to intersect with

    Raises:
        ValueError: If shell and domain have different dimensions
    """

    def __init__(self, shell: BaseShell, geom: Domain):
        if shell.dim != geom.dim:
            raise ValueError("Dimensions do not match")
        super().__init__(
            shell.dim,
            (
                np.maximum(shell.bbox[0], geom.bbox[0]),
                np.minimum(shell.bbox[1], geom.bbox[1]),
            ),
            min(shell.diam, geom.diam),
        )
        self.shell = shell
        self.geom2 = geom

    def on_boundary(self, x) -> Union[np.ndarray, bool]:
        """Check if points are on the intersection boundary.

        Points are on the boundary if they are on the shell boundary
        and inside the intersecting domain.

        Args:
            x (np.ndarray): Points to check with shape (n, dim)

        Returns:
            Union[np.ndarray, bool]: Boolean mask indicating which points are on boundary
        """
        return np.logical_and(self.shell.on_boundary(x), self.geom2.inside(x))

    def random_points(self, n, random="pseudo"):
        """Generate random points on the intersection boundary.

        Args:
            n (int): Number of points to generate
            random (str, optional): Random number generation method

        Returns:
            np.ndarray: Array of random boundary points with shape (n, dim)
        """
        x = np.empty(shape=(n, self.geom2.dim), dtype="float32")
        i = 0
        while i < n:
            tmp = self.shell.random_points(n, random=random)
            tmp = tmp[self.geom2.inside(tmp)]
            if len(tmp) > n - i:
                tmp = tmp[: n - i]
            x[i : i + len(tmp)] = tmp
            i += len(tmp)
        return x

    def boundary_normal(self, x):
        """Compute unit normal vectors at given boundary points.

        Args:
            x (np.ndarray): Points on boundary with shape (n, dim)

        Returns:
            np.ndarray: Normal vectors at each point with shape (n, dim)
        """
        return self.shell.boundary_normal(x)


class ProductShellDomain(BaseShell):
    def __init__(self, shell: BaseShell, geom: Domain):
        super().__init__(
            shell.dim + geom.dim,
            (
                np.concatenate((shell.bbox[0], geom.bbox[0])),
                np.concatenate((shell.bbox[1], geom.bbox[1])),
            ),
            np.hypot(shell.diam, geom.diam),
        )
        self.shell = shell
        self.geom = geom

    def __shell_part(self, x):
        return x[:, : self.shell.dim]

    def on_boundary(self, x):
        return self.shell.on_boundary(self.__shell_part(x))

    def random_points(self, n, random="pseudo"):
        return np.concatenate(
            (self.shell.random_points(n, random), self.geom.random_points(n, random)),
            axis=1,
        )

    def boundary_normal(self, x):
        return np.concatenate(
            (
                self.shell.boundary_normal(self.__shell_part(x)),
                np.zeros_like(x[:, self.shell.dim :]),
            ),
            axis=1,
        )


class ProductDomainShell(BaseShell):
    def __init__(self, geom: Domain, shell: BaseShell):
        super().__init__(
            geom.dim + shell.dim,
            (
                np.concatenate((geom.bbox[0], shell.bbox[0])),
                np.concatenate((geom.bbox[1], shell.bbox[1])),
            ),
            np.hypot(geom.diam, shell.diam),
        )
        self.geom = geom
        self.shell = shell

    def __shell_part(self, x):
        return x[:, self.geom.dim:]

    def on_boundary(self, x):
        return self.shell.on_boundary(self.__shell_part(x))

    def random_points(self, n, random="pseudo"):
        return np.concatenate(
            (self.geom.random_points(n, random), self.shell.random_points(n, random)),
            axis=1,
        )

    def boundary_normal(self, x):
        return np.concatenate(
            (
                np.zeros_like(x[:, : self.geom.dim]),
                self.shell.boundary_normal(self.__shell_part(x)),
            ),
            axis=1,
        )
