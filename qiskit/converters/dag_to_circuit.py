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

"""Helper function for converting a dag to a circuit."""

from qiskit.circuit import QuantumCircuit
from qiskit._accelerate.converters import dag_to_circuit as dag_to_circuit_rs


def dag_to_circuit(dag, copy_operations=True):
    """Build a ``QuantumCircuit`` object from a ``DAGCircuit``.

    This is also accessible as :meth:`.DAGCircuit.to_circuit`.

    Args:
        dag (DAGCircuit): the input dag.
        copy_operations (bool): Deep copy the operation objects
            in the :class:`~.DAGCircuit` for the output :class:`~.QuantumCircuit`,
            and shallow copy the metadata.
            This should only be set to ``False`` if the input :class:`~.DAGCircuit`
            will not be used anymore as the operations in the output
            :class:`~.QuantumCircuit` will be shared instances and
            modifications to operations in the :class:`~.DAGCircuit` will
            be reflected in the :class:`~.QuantumCircuit` (and vice versa).

    Return:
        QuantumCircuit: the circuit representing the input dag.

    Example:
        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
           from qiskit.dagcircuit import DAGCircuit
           from qiskit.converters import circuit_to_dag
           from qiskit.circuit.library.standard_gates import CHGate, U2Gate, CXGate
           from qiskit.converters import dag_to_circuit

           q = QuantumRegister(3, 'q')
           c = ClassicalRegister(3, 'c')
           circ = QuantumCircuit(q, c)
           circ.h(q[0])
           circ.cx(q[0], q[1])
           circ.measure(q[0], c[0])
           circ.rz(0.5, q[1])
           dag = circuit_to_dag(circ)
           circuit = dag_to_circuit(dag)
           circuit.draw('mpl')
    """

    name = dag.name or None

    circuit_data = dag_to_circuit_rs(dag, copy_operations)
    circuit = QuantumCircuit(
        dag.qubits,
        dag.clbits,
        *dag.qregs.values(),
        *dag.cregs.values(),
        name=name,
        global_phase=dag.global_phase,
    )
    metadata = dag.metadata or {}
    circuit.metadata = metadata.copy() if copy_operations else metadata
    circuit._data = circuit_data
    circuit._duration = dag._duration
    circuit._unit = dag._unit
    return circuit
