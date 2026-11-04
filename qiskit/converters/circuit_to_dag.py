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

"""Helper function for converting a circuit to a dag"""

from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit._accelerate.converters import circuit_to_dag as core_circuit_to_dag


def circuit_to_dag(circuit, copy_operations=True, *, qubit_order=None, clbit_order=None):
    """Build a :class:`.DAGCircuit` object from a :class:`.QuantumCircuit`.

    This is also accessible as :meth:`.QuantumCircuit.to_dag`.

    Args:
        circuit (QuantumCircuit): the input circuit.
        copy_operations (bool): Deep copy the operation objects
            in the :class:`~.QuantumCircuit` for the output :class:`~.DAGCircuit`
            and shallow copy the metadata.
            This should only be set to ``False`` if the input :class:`~.QuantumCircuit`
            will not be used anymore as the operations in the output
            :class:`~.DAGCircuit` will be shared instances and modifications to
            operations in the :class:`~.DAGCircuit` will be reflected in the
            :class:`~.QuantumCircuit` (and vice versa).
        qubit_order (Iterable[~qiskit.circuit.Qubit] or None): the order that the qubits should be
            indexed in the output DAG.  Defaults to the same order as in the circuit.
        clbit_order (Iterable[Clbit] or None): the order that the clbits should be indexed in the
            output DAG.  Defaults to the same order as in the circuit.

    Return:
        DAGCircuit: the DAG representing the input circuit.

    Raises:
        ValueError: if the ``qubit_order`` or ``clbit_order`` parameters do not match the bits in
            the circuit.

    Example:
        .. plot::
            :include-source:
            :nofigs:

            from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
            from qiskit.dagcircuit import DAGCircuit
            from qiskit.converters import circuit_to_dag

            q = QuantumRegister(3, 'q')
            c = ClassicalRegister(3, 'c')
            circ = QuantumCircuit(q, c)
            circ.h(q[0])
            circ.cx(q[0], q[1])
            circ.measure(q[0], c[0])
            circ.rz(0.5, q[1])
            dag = circuit_to_dag(circ)
    """
    # If we have an instance of BluePrintCircuit, make sure it is built by calling ._build()
    if isinstance(circuit, BlueprintCircuit):
        if not circuit._is_built:
            circuit._build()

    if qubit_order is not None and (
        len(qubit_order) != circuit.num_qubits or set(qubit_order) != set(circuit.qubits)
    ):
        raise ValueError("'qubit_order' does not contain exactly the same qubits as the circuit")

    if clbit_order is not None and (
        len(clbit_order) != circuit.num_clbits or set(clbit_order) != set(circuit.clbits)
    ):
        raise ValueError("'clbit_order' does not contain exactly the same clbits as the circuit")

    dagcircuit = core_circuit_to_dag(circuit, copy_operations, qubit_order, clbit_order)

    dagcircuit._duration = circuit._duration
    dagcircuit._unit = circuit._unit
    return dagcircuit
