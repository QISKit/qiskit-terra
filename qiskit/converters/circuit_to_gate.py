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


"""Helper function for converting a circuit to a gate"""
from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.gate import Gate
from qiskit.circuit import QuantumRegister
from qiskit.exceptions import QiskitError


def _check_is_gate(op):
    """Checks whether op can be converted to Gate."""
    if isinstance(op, Gate):
        return True
    elif isinstance(op, AnnotatedOperation):
        return _check_is_gate(op.base_op)
    return False


def circuit_to_gate(circuit, parameter_map=None, equivalence_library=None, label=None):
    """Build a :class:`.Gate` object from a :class:`.QuantumCircuit`.

    The gate is anonymous (not tied to a named quantum register),
    and so can be inserted into another circuit. The gate will
    have the same string name as the circuit.

    Args:
        circuit (QuantumCircuit): the input circuit.
        parameter_map (dict): For parameterized circuits, a mapping from
           parameters in the circuit to parameters to be used in the gate.
           If None, existing circuit parameters will also parameterize the
           Gate.
        equivalence_library (EquivalenceLibrary): Optional equivalence library
           where the converted gate will be registered.
        label (str): Optional gate label.

    Raises:
        QiskitError: if circuit is non-unitary or if
            parameter_map is not compatible with circuit

    Return:
        Gate: a Gate equivalent to the action of the
        input circuit. Upon decomposition, this gate will
        yield the components comprising the original circuit.
    """

    from qiskit.circuit.quantumcircuit import QuantumCircuit

    if circuit.clbits:
        raise QiskitError("Circuit with classical bits cannot be converted to gate.")
    if circuit.num_vars:
        raise QiskitError("circuits with realtime classical variables cannot be converted to gates")

    for instruction in circuit.data:
        if not _check_is_gate(instruction.operation):
            raise QiskitError(
                "One or more instructions cannot be converted to"
                f' a gate. "{instruction.operation.name}" is not a gate instruction'
            )

    if parameter_map is None:
        parameter_dict = {p: p for p in circuit.parameters}
    else:
        parameter_dict = circuit._unroll_param_dict(parameter_map)

    if parameter_dict.keys() != circuit.parameters:
        raise QiskitError(
            "parameter_map should map all circuit parameters. "
            f"Circuit parameters: {circuit.parameters}, parameter_map: {parameter_dict}"
        )

    gate = Gate(
        name=circuit.name,
        num_qubits=circuit.num_qubits,
        params=[*parameter_dict.values()],
        label=label,
    )
    gate.condition = None

    target = circuit.assign_parameters(parameter_dict, inplace=False)

    if equivalence_library is not None:
        equivalence_library.add_equivalence(gate, target)

    qc = QuantumCircuit(name=gate.name, global_phase=target.global_phase)
    if gate.num_qubits > 0:
        q = QuantumRegister(gate.num_qubits, "q")
        qc.add_register(q)
    qubit_map = {bit: q[idx] for idx, bit in enumerate(circuit.qubits)}

    # The 3rd parameter in the output tuple) is hard coded to [] because
    # Gate objects do not have cregs set and we've verified that all
    # instructions are gates
    for instruction in target.data:
        qc._append(instruction.replace(qubits=tuple(qubit_map[y] for y in instruction.qubits)))
    gate.definition = qc
    return gate
