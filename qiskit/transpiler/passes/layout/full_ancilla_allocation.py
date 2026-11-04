# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Allocate all idle nodes from the coupling map as ancilla on the layout."""

from qiskit.circuit import QuantumRegister
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target


class FullAncillaAllocation(AnalysisPass):
    """Allocate all idle nodes from the coupling map or target as ancilla on the layout.

    A pass for allocating all idle physical qubits (those that exist in coupling
    map or target but not the dag circuit) as ancilla. It will also choose new
    virtual qubits to correspond to those physical ancilla.

    Note:
        This is an analysis pass, and only responsible for choosing physical
        ancilla locations and their corresponding virtual qubits.
        A separate transformation pass must add those virtual qubits to the
        circuit.
    """

    def __init__(self, coupling_map):
        """FullAncillaAllocation initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): directed graph representing a coupling map.
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.ancilla_name = "ancilla"

    def run(self, dag):
        """Run the FullAncillaAllocation pass on `dag`.

        Extend the layout with new (physical qubit, virtual qubit) pairs.
        The dag signals which virtual qubits are already in the circuit.
        This pass will allocate new virtual qubits such that no collision occurs
        (i.e. Layout bijectivity is preserved)

        The coupling_map and layout together determine which physical qubits are free.

        Args:
            dag (DAGCircuit): circuit to analyze

        Returns:
            DAGCircuit: returns the same dag circuit, unmodified

        Raises:
            TranspilerError: If there is no layout in the property set or not set at init time.
        """
        layout = self.property_set.get("layout")

        if layout is None:
            raise TranspilerError('FullAncillaAllocation pass requires property_set["layout"].')

        virtual_bits = layout.get_virtual_bits()
        physical_bits = layout.get_physical_bits()
        if layout:
            FullAncillaAllocation.validate_layout(virtual_bits, set(dag.qubits))
            layout_physical_qubits = list(range(max(physical_bits) + 1))
        else:
            layout_physical_qubits = []

        idle_physical_qubits = [q for q in layout_physical_qubits if q not in physical_bits]

        if self.target is not None and self.target.num_qubits is not None:
            idle_physical_qubits = [
                q for q in range(self.target.num_qubits) if q not in physical_bits
            ]
        elif self.coupling_map:
            idle_physical_qubits = [
                q for q in self.coupling_map.physical_qubits if q not in physical_bits
            ]

        if idle_physical_qubits:
            if self.ancilla_name in dag.qregs:
                qreg = QuantumRegister._new_with_prefix(
                    len(idle_physical_qubits), self.ancilla_name
                )
            else:
                qreg = QuantumRegister(len(idle_physical_qubits), name=self.ancilla_name)

            for idx, idle_q in enumerate(idle_physical_qubits):
                self.property_set["layout"][idle_q] = qreg[idx]
            self.property_set["layout"].add_register(qreg)
        return dag

    @staticmethod
    def validate_layout(layout_qubits, dag_qubits):
        """
        Checks if all the qregs in ``layout_qregs`` already exist in ``dag_qregs``. Otherwise, raise.
        """
        for qreg in layout_qubits:
            if qreg not in dag_qubits:
                raise TranspilerError(
                    "FullAncillaAllocation: The layout refers to a qubit "
                    "that does not exist in circuit."
                )
