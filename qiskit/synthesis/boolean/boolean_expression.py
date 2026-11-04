# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class for parsing and synthesizing boolean expressions"""

import ast
import itertools
import re
from os.path import isfile

from collections.abc import Callable
import numpy as np

from .boolean_expression_visitor import (
    BooleanExpressionEvalVisitor,
    BooleanExpressionArgsCollectorVisitor,
)


class TruthTable:
    """A simple implementation of a truth table for a boolean function

    The truth table is built from a callable which takes an assignment, which
    is a tuple of boolean values of a fixed given length (the number of bits
    of the truth table) and returns a boolean value.

    For a number of bits at most `EXPLICIT_REP_THRESHOLD` the values of the table
    are explicitly computed and stored. Otherwise, the values are computed on the fly
    and stored in a dictionary."""

    EXPLICIT_REP_THRESHOLD = (
        12  # above this number of bits, do not explicitly save the values in a list
    )

    def __init__(self, func: Callable, num_bits: int):
        self.func = func
        self.num_bits = num_bits
        self.explicit_storage = self.num_bits <= self.EXPLICIT_REP_THRESHOLD
        if self.explicit_storage:
            self.values = np.array(
                [self.func(assignment) for assignment in self.all_assignments()],
                dtype=bool,
            )
        else:
            self.implicit_values = {}

    def all_assignments(self) -> list[tuple[bool]]:
        """Return an ordered list of all assignments, ordered from right to left
        i.e. 000, 100, 010, 110, 001, 101, 011, 111"""
        return [
            tuple(reversed(assignment))
            for assignment in itertools.product([False, True], repeat=self.num_bits)
        ]

    def __getitem__(self, key: str | tuple[bool]) -> bool:
        if isinstance(key, str):
            key = (bit != "0" for bit in key)
        if self.explicit_storage:
            key = sum(2**n for n, bit in enumerate(key) if bit)
            return self.values[key]
        else:
            return self.implicit_values.setdefault(key, self.func(key))

    def __str__(self) -> str:
        if self.explicit_storage:
            return "".join(
                "1" if self[assignment] else "0" for assignment in self.all_assignments()
            )
        else:
            return f"Truth table on {self.num_bits} bits (implicit representation)"


class BooleanExpression:
    """A Boolean Expression"""

    def __init__(self, expression: str, var_order: list | None = None) -> None:
        """
        Args:
            expression (str): The logical expression string.
            name (str): Optional. Instruction gate name. Otherwise part of the expression is
               going to be used.
            var_order(list): A list with the order in which variables will be created.
               (default: by appearance)
        """
        self.expression = expression
        self.expression_ast = ast.parse(expression)
        args_collector = BooleanExpressionArgsCollectorVisitor()
        args_collector.visit(self.expression_ast)
        self.args = args_collector.get_sorted_args()
        self.num_bits = len(self.args)
        if var_order is not None:
            missing_args = set(self.args) - set(var_order)
            if len(missing_args) > 0:
                raise ValueError(f"var_order missing the variable(s) {', '.join(missing_args)}")
            self.args.sort(key=var_order.index)
        self.truth_table_ = None

    def simulate(self, bitstring: str | tuple) -> bool:
        """Evaluate the expression on a bitstring.

        This evaluation is done classically.

        Args:
            bitstring: The bitstring for which to evaluate,
            either as a string of 0 and 1 or a tuple of booleans.

        Returns:
            bool: result of the evaluation.
        """
        if self.num_bits != len(bitstring):
            raise ValueError(
                f"bitstring length differs from the number of variables "
                f"({len(bitstring)} != {self.num_bits}"
            )
        if isinstance(bitstring, str):
            bitstring = (bit != "0" for bit in bitstring)
        eval_visitor = BooleanExpressionEvalVisitor()
        eval_visitor.arg_values = dict(zip(self.args, bitstring))
        return eval_visitor.visit(self.expression_ast)

    @property
    def truth_table(self) -> dict:
        """Generates the full truth table for the expression
        Returns:
            dict: A dictionary mapping boolean assignments to the boolean result
        """
        if self.truth_table_ is None:
            self.truth_table_ = TruthTable(self.simulate, self.num_bits)
        return self.truth_table_

    def synth(self, circuit_type: str = "bit"):
        r"""Synthesize the logic network into a :class:`~qiskit.circuit.QuantumCircuit`.
        There are two common types of circuits for a boolean function :math:`f(x)`:

        1. **Bit-flip oracles** which compute:

         .. math::

            |x\rangle|y\rangle |-> |x\rangle|f(x)\oplusy\rangle

        2. **Phase-flip** oracles which compute:

         .. math::

            |x\rangle |-> (-1)^{f(x)}|x\rangle

        By default the bit-flip oracle is generated.

        Args:
            circuit_type: which type of oracle to create, 'bit' or 'phase' flip oracle.
        Returns:
            QuantumCircuit: A circuit implementing the logic network.
        Raises:
            ValueError: If ``circuit_type`` is not either 'bit' or 'phase'.
        """

        from .boolean_expression_synth import (
            synth_bit_oracle_from_esop,
            synth_phase_oracle_from_esop,
            EsopGenerator,
        )  # import here to avoid cyclic import

        # generating the esop currently requires generating the full truth table
        # there are many optimizations that can be done to improve this step
        esop = EsopGenerator(self.truth_table).esop
        if circuit_type == "bit":
            return synth_bit_oracle_from_esop(esop, self.num_bits + 1)
        if circuit_type == "phase":
            return synth_phase_oracle_from_esop(esop, self.num_bits)
        raise ValueError("'circuit_type' must be either 'bit' or 'phase'")

    @staticmethod
    def from_dimacs(dimacs: str):
        """Create a BooleanExpression from a string in the DIMACS format.
        Args:
            dimacs : A string in DIMACS format.

        Returns:
            BooleanExpression: A gate for the input string

        Raises:
            ValueError: If the string is not formatted according to DIMACS rules
        """
        header_regex = re.compile(r"p\s+cnf\s+(\d+)\s+(\d+)")
        clause_regex = re.compile(r"(-?\d+)")
        lines = [line.strip() for line in dimacs.split("\n")]
        lines = [
            line for line in lines if not line.startswith("c") and line != ""
        ]  # DIMACS comment line start with c
        header_match = header_regex.match(lines[0])
        if not header_match:
            raise ValueError("First line must start with 'p cnf'")
        num_vars, _ = map(int, header_match.groups())
        variables = [f"x{i+1}" for i in range(num_vars)]
        clauses = []
        for line in lines[1:]:
            literals = clause_regex.findall(line)
            if len(literals) == 0 or literals[-1] != "0":
                continue
            clauses.append([int(c) for c in literals[:-1]])
        clause_strings = [
            " | ".join([f'{"~" if lit < 0 else ""}{variables[abs(lit)-1]}' for lit in clause])
            for clause in clauses
        ]
        expr = " & ".join([f"({c})" for c in clause_strings])
        return BooleanExpression(expr, var_order=variables)

    @staticmethod
    def from_dimacs_file(filename: str):
        """Create a BooleanExpression from a file in the DIMACS format.
        Args:
            filename: A file in DIMACS format.

        Returns:
            BooleanExpression: A gate for the input string

        Raises:
            FileNotFoundError: If filename is not found.
        """
        if not isfile(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        with open(filename) as dimacs_file:
            dimacs = dimacs_file.read()
        return BooleanExpression.from_dimacs(dimacs)
