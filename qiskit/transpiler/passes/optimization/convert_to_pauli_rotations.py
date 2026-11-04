# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Convert standard gates into Pauli product rotation gates for Pauli Based Computation"""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.utils import control_flow
from qiskit._accelerate.convert_to_pauli_rotations import convert_to_pauli_rotations


class ConvertToPauliRotations(TransformationPass):
    r"""
    Convert a quantum circuit into an equivalent circuit composed of
    :class:`.PauliProductRotationGate` gates and :class:`.PauliProductMeasurement`
    instructions.

    The pass converts all single-qubit, two-qubit and three-qubit standard gates into
    Pauli product rotations, converts measures to Pauli product measurements,
    and leaves barriers, delays, resets, Pauli product rotations, and
    Pauli product measurements unchanged.

    For example::

      from qiskit.circuit import QuantumCircuit
      from qiskit.transpiler.passes import ConvertToPauliRotations
      from qiskit.quantum_info import Operator

      qc = QuantumCircuit(3)
      qc.h(0)
      qc.cx(0, 1)
      qc.ry(0.123, 0)
      qc.t(2)
      qc.rzz(pi/4, 0, 2)

      # The transformed circuit consists of PauliProductRotationGate gates
      qct = ConvertToPauliRotations()(qc)
      ops_names = set(qct.count_ops().keys())
      self.assertEqual(ops_names, {"pauli_product_rotation"})

      # The circuits before and after the transformation are equivalent
      assert Operator(qc) == Operator(qct)
    """

    @control_flow.trivial_recurse
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ConvertToPauliRotations optimization pass on ``dag``.

        Args:
            dag: the input DAG.

        Returns:
            The output DAG.

        Raises:
            TranspilerError: if the circuit contains instructions not supported by the pass.
        """
        dag = convert_to_pauli_rotations(dag)

        return dag
