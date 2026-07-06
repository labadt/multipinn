import re
from typing import Callable, List

import torch

from multipinn.condition.diff import *


class Symbols:
    """A symbolic differentiation code generator for PINN models.

    Generates code for computing derivatives using either automatic differentiation
    or numerical methods. Supports variable parsing, validation, and code generation
    for first and second derivatives.

    The class follows a specific syntax for derivative notation:
    - Single letters for variables (x, y, z, etc.)
    - Single letters for functions (u, v, p, etc.)
    - Underscore notation for derivatives (u_x, u_xy, etc.)

    Attributes:
        variables (List[str]): Input variable names
        functions (List[str]): Output function names
        input_dim (int): Number of input dimensions
        output_dim (int): Number of output dimensions
        mode (str): Differentiation mode - "grad" or "diff"

    Example:
        >>> symbols = Symbols("x,y", "u,v", mode="grad")
        >>> compute_grad = symbols("u_x,u_y,v_x,v_y")
        >>> # Generated function computes du/dx, du/dy, dv/dx, dv/dy
    """

    def __init__(self, variables: str, functions: str, mode: str = "grad"):
        """Initialize symbol generator.

        Args:
            variables (str): Comma-separated input variable names
            functions (str): Comma-separated output function names
            mode (str, optional): Differentiation mode - "grad" or "diff". Defaults to "grad".
                "grad" uses automatic differentiation
                "diff" uses numerical approximations

        Raises:
            ValueError: If variable/function names are invalid or duplicated
        """
        self.variables = variables.replace(" ", "").rstrip(",").split(",")
        self.functions = functions.replace(" ", "").rstrip(",").split(",")
        self.check_names()
        self.input_dim = len(self.variables)
        self.output_dim = len(self.functions)
        self.mode = mode

    def check_names(self):
        """Validate variable and function names.

        Names must be single letters [a-zA-Z] without duplicates.

        Raises:
            ValueError: If names don't match requirements
        """
        all_names = self.variables + self.functions
        for name in all_names:
            pattern = r"^[a-zA-Z]$"
            if not re.match(pattern, name):
                raise ValueError(
                    f"Names must be a single letter [a-zA-Z], but found {name}"
                )
        if len(all_names) != len(set(all_names)):
            raise ValueError("Duplicate names in variables or functions")

    def check_format(self, derivatives: List[str]):
        if derivatives == [""]:
            raise ValueError(f"No reason to call this function with zero symbols")
        for symbol in derivatives:
            if len(symbol) == 1:
                if symbol not in self.variables and symbol not in self.functions:
                    raise ValueError(
                        f"Symbol '{symbol}' is not a defined function or variable."
                    )
            else:
                if symbol.count("_") != 1 or symbol[1] != "_":
                    raise ValueError(
                        f"Invalid symbol format: '{symbol}'. Expected format is 'u_xyz' or 'u' or 'x'."
                    )
                f, var = symbol.split("_")
                if not var:
                    raise ValueError(
                        f"Invalid symbol format: '{symbol}'. Variable part cannot be empty."
                    )
                if f not in self.functions:
                    raise ValueError(
                        f"Invalid symbol format: '{symbol}'. Function '{f}' is not defined."
                    )
                if any(c not in self.variables for c in var):
                    raise ValueError(
                        f"Invalid symbol format: '{symbol}'. Variable '{var}' contains undefined variables."
                    )

    def parsing(self, output_format: str) -> List[str]:
        derivatives = output_format.replace(" ", "").rstrip(",").split(",")
        self.check_format(derivatives)
        return derivatives

    @staticmethod
    def __join_as_tuple(variables: List[str]) -> str:
        if len(variables) == 1:
            return "(" + variables[0] + ",)"
        else:
            return ", ".join(variables)

    def __children_grad(self, symbol: str) -> List[str]:
        if "_" in symbol:
            return [f"{symbol}{var}" for var in self.variables]
        else:
            return [f"{symbol}_{var}" for var in self.variables]

    def __gen__line_grad(self, symbol: str) -> str:
        """
        Generate a line of code (with tab) for calculating grad from symbol.
        """
        new_variables = self.__children_grad(symbol)
        return f"\t{self.__join_as_tuple(new_variables)} = unpack(grad({symbol}, arg))"

    def gen_str_using_grad(self, derivatives: List[str]) -> str:
        requested_symbols = sorted(
            derivatives, key=lambda s: len(s.split("_")[1]) if "_" in s else 0
        )

        func_code = []
        func_code.append("def compute_gradients(model, arg):")
        if any(symbol[0] in self.functions for symbol in requested_symbols):
            func_code.append(
                "\t" + self.__join_as_tuple(self.functions) + " = unpack(model(arg))"
            )
        if any(symbol in self.variables for symbol in requested_symbols):
            func_code.append(
                "\t" + self.__join_as_tuple(self.variables) + " = unpack(arg)"
            )
        existing_symbols = set(self.variables + self.functions)

        def add_symbol(current: str):
            if current in existing_symbols:
                return
            parent = current[:-1]
            if parent.endswith("_"):
                parent = current[:-2]
            if parent not in existing_symbols:
                add_symbol(parent)
            func_code.append(self.__gen__line_grad(parent))
            new_variables = self.__children_grad(parent)
            existing_symbols.update(new_variables)

        for s in requested_symbols:
            add_symbol(s)

        func_code.append("\treturn " + self.__join_as_tuple(derivatives))
        return "\n".join(func_code)

    def __hot_one_vector(self, var: str) -> str:
        hot_one = ["0"] * self.input_dim
        hot_one[self.variables.index(var)] = "1"
        return ("torch.tensor([["
    + self.__join_as_tuple(hot_one)
    + "]], device=arg.device, dtype=arg.dtype)")

    def __children_diff_first(self, var: str) -> List[str]:
        return [f"{fun}_{var}" for fun in self.functions]

    def __gen__line_diff_first(self, var: str) -> str:
        """
        Generate a line of code (with tab) for calculating all derivatives at dir.
        """
        new_variables = self.__children_diff_first(var)
        vector = self.__hot_one_vector(var)
        return f"\t{self.__join_as_tuple(new_variables)} = unpack(num_diff_random(model, arg, fns, {vector}))"

    def __children_diff_second(self, var1: str, var2: str) -> List[str]:
        return [f"{fun}_{var1}{var2}" for fun in self.functions]

    def __gen__line_diff_second(self, var1: str, var2: str) -> List[str]:
        new_variables = self.__children_diff_second(var1, var2)
        vector1 = self.__hot_one_vector(var1)
        if var1 == var2:
            return f"\t{self.__join_as_tuple(new_variables)} = unpack(num_diff_second_same(model, arg, fns, {vector1}))"
        else:
            vector2 = self.__hot_one_vector(var2)
            return f"\t{self.__join_as_tuple(new_variables)} = unpack(num_diff_second_cross(model, arg, {vector1}, {vector2}))"

    def gen_str_using_diff(self, derivatives: List[str]) -> str:
        requested_symbols = sorted(
            derivatives, key=lambda s: len(s.split("_")[1]) if "_" in s else 0
        )

        func_code = []
        func_code.append("def compute_gradients(model, arg):")
        if any(symbol[0] in self.functions for symbol in requested_symbols):
            have_pure_model_output = any(
                symbol in self.functions for symbol in requested_symbols
            )
            have_non_cross_diff = any(
                len(symbol) == 3 or len(symbol) == 4 and symbol[-2] == symbol[-1]
                for symbol in requested_symbols
            )
            if have_pure_model_output or have_non_cross_diff:
                func_code.append("\tfns = model(arg)")
            if have_pure_model_output:
                func_code.append(
                    "\t" + self.__join_as_tuple(self.functions) + " = unpack(fns)"
                )
        if any(symbol in self.variables for symbol in requested_symbols):
            func_code.append(
                "\t" + self.__join_as_tuple(self.variables) + " = unpack(arg)"
            )
        existing_symbols = set(self.variables + self.functions)

        for current in requested_symbols:
            if current not in existing_symbols:
                if len(current) == 3:  # u_x
                    var = current[-1]
                    func_code.append(self.__gen__line_diff_first(var))
                    new_variables = self.__children_diff_first(var)
                    existing_symbols.update(new_variables)
                elif len(current) == 4:  # u_xy
                    var1 = current[-2]
                    var2 = current[-1]
                    func_code.append(self.__gen__line_diff_second(var1, var2))
                    new_variables = self.__children_diff_second(var1, var2)
                    existing_symbols.update(new_variables)
                else:
                    raise ValueError(
                        f"'diff' mode doesn't support derivatives of order > 2, but {current} found"
                    )

        func_code.append("\treturn " + self.__join_as_tuple(derivatives))
        return "\n".join(func_code)

    def generate_str(self, derivatives: List[str]) -> str:
        if self.mode == "grad":
            return self.gen_str_using_grad(derivatives)
        elif self.mode == "diff":
            return self.gen_str_using_diff(derivatives)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

    def __call__(self, output_format: str) -> Callable:
        derivatives = self.parsing(output_format)
        func_code_str = self.generate_str(derivatives)
        print(f"Autogenerated from {output_format}:\n{func_code_str}\n")
        namespace = {
            "torch": torch,
            "unpack": unpack,
            "grad": grad,
            "num_diff_random": num_diff_random,
            "num_diff_second_cross": num_diff_second_cross,
            "num_diff_second_same": num_diff_second_same,
        }
        exec(func_code_str, namespace)
        return namespace["compute_gradients"]
