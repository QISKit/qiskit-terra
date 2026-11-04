# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Recursively expands 3q+ gates until the circuit only contains 2q or 1q gates."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.utils import control_flow
from qiskit._accelerate.unroll_3q_or_more import unroll_3q_or_more
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES


class Unroll3qOrMore(TransformationPass):
    """Recursively expands 3q+ gates until the circuit only contains 2q or 1q gates."""

    def __init__(self, target=None, basis_gates=None):
        """Initialize the Unroll3qOrMore pass

        Args:
            target (Target): The target object representing the compilation
                target. If specified any multi-qubit instructions in the
                circuit when the pass is run that are supported by the target
                device will be left in place. If both this and ``basis_gates``
                are specified only the target will be checked.
            basis_gates (list): A list of basis gate names that the target
                device supports. If specified any gate names in the circuit
                which are present in this list will not be unrolled. If both
                this and ``target`` are specified only the target will be used
                for checking which gates are supported.
        """
        super().__init__()
        self.target = target
        self.basis_gates = None
        if basis_gates is not None:
            self.basis_gates = set(basis_gates)
            if target is None:
                self.target = Target.from_configuration(
                    [x for x in basis_gates if x not in CONTROL_FLOW_OP_NAMES]
                )

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the Unroll3qOrMore pass on `dag`.

        Args:
            dag(DAGCircuit): input dag
        Returns:
            DAGCircuit: output dag with maximum node degrees of 2
        Raises:
            QiskitError: if a 3q+ gate is not decomposable
        """
        # In Rust unitary gates don't have a definition and we always
        # run UnitarySynthesis first in a pass manager. But for backwards
        # compatibility we need to unroll any unitary gates in the circuit.
        # The simplest way to do this is to just run UnitarySynthesis with
        # the default universal basis of U-CX
        if "unitary" in dag.count_ops(recurse=False) and (
            not self.target or "unitary" not in self.target
        ):
            dag = UnitarySynthesis(["cx", "u"], min_qubits=3).run(dag)

        unroll_3q_or_more(dag, self.target)
        return dag
