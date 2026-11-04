# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Extend the dag with virtual qubits that are in layout but not in the circuit yet."""

import itertools

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError


class EnlargeWithAncilla(TransformationPass):
    """Extend the dag with virtual qubits that are in layout but not in the circuit yet.

    Extend the DAG circuit with new virtual qubits (ancilla) that are specified
    in the layout, but not present in the circuit. Which qubits to add are
    previously allocated in the ``layout`` property, by a previous pass.
    """

    def run(self, dag):
        """Run the EnlargeWithAncilla pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to extend.

        Returns:
            DAGCircuit: An extended DAG.

        Raises:
            TranspilerError: If there is no layout in the property set or not set at init time.
        """
        layout = self.property_set["layout"]
        qubit_indices = self.property_set["original_qubit_indices"]

        if layout is None:
            raise TranspilerError('EnlargeWithAncilla requires property_set["layout"]')

        new_qregs = {reg for reg in layout.get_registers() if reg not in dag.qregs.values()}

        for qreg in new_qregs:
            if qubit_indices is not None:
                qubit_indices.update(zip(qreg, itertools.count(dag.num_qubits())))
            dag.add_qreg(qreg)

        self.property_set["original_qubit_indices"] = qubit_indices
        return dag
