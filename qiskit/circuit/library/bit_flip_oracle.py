# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Bit-flip Oracle object."""

from __future__ import annotations

import typing

from qiskit.circuit import Gate

if typing.TYPE_CHECKING:
    from qiskit.synthesis.boolean.boolean_expression import BooleanExpression


class BitFlipOracleGate(Gate):
    r"""Implements a bit-flip oracle

    The Bit-flip Oracle Gate object constructs circuits for any arbitrary
    input logical expressions. A logical expression is composed of logical operators
    `&` (logical `AND`), `|` (logical  `OR`),
    `~` (logical  `NOT`), and `^` (logical  `XOR`).
    as well as symbols for literals (variables).
    For example, `'a & b'`, and `(v0 | ~v1) & (~v2 & v3)`
    are both valid string representation of boolean logical expressions.

    A bit-flip oracle for a boolean function `f(x)` performs the following
    quantum operation:

    .. math::

            |x\rangle|y\rangle \mapsto |x\rangle|f(x)\oplus y\rangle

    For convenience, this oracle, in addition to parsing arbitrary logical expressions,
    also supports input strings in the `DIMACS CNF format
    <https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
    which is the standard format for specifying SATisfiability (SAT) problem instances in
    `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
    which is a conjunction of one or more clauses, where a clause is a disjunction of one
    or more literals.
    See :meth:`qiskit.circuit.library.bit_flip_oracle.BitFlipOracleGate.from_dimacs_file`.

    From 16 variables on, possible performance issues should be expected when using the
    default synthesizer.
    """

    def __init__(
        self,
        expression: str | BooleanExpression,
        var_order: list[str] | None = None,
        label: str | None = None,
    ) -> None:
        """
        Args:
            expression: A Python-like boolean expression string or a `BooleanExpression` object.
            var_order: A list with the order in which variables will be created.
               (default: by appearance)
            label: A label for the gate to display in visualizations. Per default, the label is
                set to display the textual representation of the boolean expression (truncated if needed)
        """
        from qiskit.synthesis.boolean.boolean_expression import BooleanExpression

        if label is None:
            if isinstance(expression, str):
                label = (expression[:15] + "...") if len(expression) > 15 else expression
            else:
                label = "Boolean Expression"

        if isinstance(expression, str):
            expression = BooleanExpression(expression, var_order=var_order)
        self.boolean_expression = expression

        super().__init__(
            name="Bit-flip Oracle",
            num_qubits=self.boolean_expression.num_bits + 1,
            params=[],
            label=label,
        )

    def _define(self):
        """Defined by the synthesized bit-flip oracle"""
        self.definition = self.boolean_expression.synth(circuit_type="bit")

    @classmethod
    def from_dimacs_file(cls, filename: str) -> BitFlipOracleGate:
        r"""Create a BitFlipOracleGate from the string in the DIMACS format.

        It is possible to build a BitFlipOracleGate from a file in `DIMACS CNF format
        <https://web.archive.org/web/20190325181937/https://www.satcompetition.org/2009/format-benchmarks2009.html>`__,
        which is the standard format for specifying SATisfiability (SAT) problem instances in
        `Conjunctive Normal Form (CNF) <https://en.wikipedia.org/wiki/Conjunctive_normal_form>`__,
        which is a conjunction of one or more clauses, where a clause is a disjunction of one
        or more literals.

        The following is an example of a CNF expressed in the DIMACS format:

        .. code:: text

          c DIMACS CNF file with 3 satisfying assignments: 1 -2 3, -1 -2 -3, 1 2 -3.
          p cnf 3 5
          -1 -2 -3 0
          1 -2 3 0
          1 2 -3 0
          1 -2 -3 0
          -1 2 3 0

        The first line, following the `c` character, is a comment. The second line specifies that
        the CNF is over three boolean variables --- let us call them  :math:`x_1, x_2, x_3`, and
        contains five clauses.  The five clauses, listed afterwards, are implicitly joined by the
        logical `AND` operator, :math:`\land`, while the variables in each clause, represented by
        their indices, are implicitly disjoined by the logical `OR` operator, :math:`\lor`. The
        :math:`-` symbol preceding a boolean variable index corresponds to the logical `NOT`
        operator, :math:`\lnot`. Character `0` (zero) marks the end of each clause.  Essentially,
        the code above corresponds to the following CNF:

        :math:`(\lnot x_1 \lor \lnot x_2 \lor \lnot x_3)
        \land (x_1 \lor \lnot x_2 \lor x_3)
        \land (x_1 \lor x_2 \lor \lnot x_3)
        \land (x_1 \lor \lnot x_2 \lor \lnot x_3)
        \land (\lnot x_1 \lor x_2 \lor x_3)`.


        Args:
            filename: A file in DIMACS format.

        Returns:
            BitFlipOracleGate: A quantum gate with a bit-flip oracle.
        """
        from qiskit.synthesis.boolean.boolean_expression import BooleanExpression

        expr = BooleanExpression.from_dimacs_file(filename)
        return cls(expr)
