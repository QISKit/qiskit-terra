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

"""
A generic InverseCancellation pass for any set of gate-inverse pairs.
"""
from __future__ import annotations


from qiskit.circuit import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.utils import control_flow

from qiskit._accelerate.inverse_cancellation import (
    inverse_cancellation,
    run_inverse_cancellation_standard_gates,
)


class InverseCancellation(TransformationPass):
    """Cancel specific Gates which are inverses of each other when they occur back-to-
    back."""

    def __init__(
        self,
        gates_to_cancel: list[Gate | tuple[Gate, Gate]] | None = None,
        run_default: bool = False,
    ):
        """Initialize InverseCancellation pass.

        Args:
            gates_to_cancel: List describing the gates to cancel. Each element of the
                list is either a single gate or a pair of gates. If a single gate, then
                it should be self-inverse. If a pair of gates, then the gates in the
                pair should be inverses of each other. If ``None`` a default list of
                self-inverse gates and a default list of inverse gate pairs will be used.
                The current default list of self-inverse gates is:

                  * :class:`.CXGate`
                  * :class:`.ECRGate`
                  * :class:`.CYGate`
                  * :class:`.CZGate`
                  * :class:`.XGate`
                  * :class:`.YGate`
                  * :class:`.ZGate`
                  * :class:`.HGate`
                  * :class:`.SwapGate`
                  * :class:`.CHGate`
                  * :class:`.CCXGate`
                  * :class:`.CCZGate`
                  * :class:`.RCCXGate`
                  * :class:`.CSwapGate`
                  * :class:`.C3XGate`

                and the default list of inverse gate pairs is:

                  * :class:`.TGate` and :class:`.TdgGate`
                  * :class:`.SGate` and :class:`.SdgGate`
                  * :class:`.SXGate` and :class:`.SXdgGate`
                  * :class:`.CSGate` and :class:`.CSdgGate`

            run_default: If set to true and ``gates_to_cancel`` is set to a list then in
                addition to the gates listed in ``gates_to_cancel`` the default list of gate
                inverses (the same as when ``gates_to_cancel`` is set to ``None``) will be
                run. The order of evaluation is significant in how sequences of gates are
                cancelled and the default gates will be evaluated after the provided gates
                in ``gates_to_cancel``. If ``gates_to_cancel`` is ``None`` this option has
                no impact.

        Raises:
            TranspilerError: Input is not a self-inverse gate or a pair of inverse gates.
        """
        self.self_inverse_gates = []
        self.inverse_gate_pairs = []
        self.self_inverse_gate_names = set()
        self.inverse_gate_pairs_names = set()
        self._also_default = run_default
        if gates_to_cancel is None:
            self._use_standard_gates = True
        else:
            self._use_standard_gates = False
            for gates in gates_to_cancel:
                if isinstance(gates, Gate):
                    if gates != gates.inverse():
                        raise TranspilerError(f"Gate {gates.name} is not self-inverse")
                elif isinstance(gates, tuple):
                    if len(gates) != 2:
                        raise TranspilerError(
                            f"Too many or too few inputs: {gates}. Only two are allowed."
                        )
                    if gates[0] != gates[1].inverse():
                        raise TranspilerError(
                            f"Gate {gates[0].name} and {gates[1].name} are not inverse."
                        )
                else:
                    raise TranspilerError(
                        f"InverseCancellation pass does not take input type {type(gates)}. Input must be"
                        " a Gate."
                    )

            for gates in gates_to_cancel:
                if isinstance(gates, Gate):
                    self.self_inverse_gates.append(gates)
                    self.self_inverse_gate_names.add(gates.name)
                else:
                    self.inverse_gate_pairs.append(gates)
                    self.inverse_gate_pairs_names.update(x.name for x in gates)
        super().__init__()

    @control_flow.trivial_recurse
    def run(self, dag: DAGCircuit):
        """Run the InverseCancellation pass on `dag`.

        Args:
            dag: the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        if self._use_standard_gates:
            run_inverse_cancellation_standard_gates(dag)
        else:
            inverse_cancellation(
                dag,
                self.inverse_gate_pairs,
                self.self_inverse_gates,
                self.inverse_gate_pairs_names,
                self.self_inverse_gate_names,
                self._also_default,
            )
        return dag
