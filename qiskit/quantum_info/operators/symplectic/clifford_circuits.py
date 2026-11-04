# This code is part of Qiskit.
#
# (C) Copyright IBM 2017--2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit simulation for the Clifford class.
"""

from __future__ import annotations

import numpy as np

from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError


def _append_circuit(clifford, circuit, qargs=None):
    """Update Clifford inplace by applying a Clifford circuit.

    Args:
        clifford (Clifford): The Clifford to update.
        circuit (QuantumCircuit): The circuit to apply.
        qargs (list or None): The qubits to apply circuit to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input circuit cannot be decomposed into Clifford operations.
    """
    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    for instruction in circuit:
        if instruction.clbits:
            raise QiskitError(
                f"Cannot apply Instruction with classical bits: {instruction.operation.name}"
            )
        # Get the integer position of the flat register
        new_qubits = [qargs[circuit.find_bit(bit).index] for bit in instruction.qubits]
        clifford = _append_operation(clifford, instruction.operation, new_qubits)
    return clifford


def _prepend_circuit(clifford, circuit, qargs=None):
    """Update Clifford inplace by prepending a Clifford circuit.

    Args:
        clifford (Clifford): The Clifford to update.
        circuit (QuantumCircuit): The circuit to apply before the clifford.
        qargs (list or None): The qubits to apply circuit to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input circuit cannot be decomposed into Clifford operations.
    """
    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    # reverse the order of instructions when prepending a circuit
    for instruction in circuit.reverse_ops():
        if instruction.clbits:
            raise QiskitError(
                f"Cannot apply Instruction with classical bits: {instruction.operation.name}"
            )
        # Get the integer position of the flat register
        new_qubits = [qargs[circuit.find_bit(bit).index] for bit in instruction.qubits]
        clifford = _prepend_operation(clifford, instruction.operation, new_qubits)
    return clifford


def _append_operation(clifford, operation, qargs=None):
    """Update Clifford inplace by applying a Clifford operation.

    Args:
        clifford (Clifford): The Clifford to update.
        operation (Instruction or Clifford or str): The operation or composite operation to apply.
        qargs (list or None): The qubits to apply operation to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input operation cannot be converted into Clifford operations.
    """

    if isinstance(operation, (Barrier, Delay)):
        return clifford

    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    gate = operation

    if isinstance(gate, str):
        # Check if gate is a valid Clifford basis gate string
        if gate not in _BASIS_1Q and gate not in _BASIS_2Q:
            raise QiskitError(f"Invalid Clifford gate name string {gate}")
        name = gate
    else:
        # assert isinstance(gate, Instruction)
        name = gate.name

    # Apply gate if it is a Clifford basis gate
    if name in _NON_CLIFFORD:
        raise QiskitError(f"Cannot update Clifford with non-Clifford gate {name}")
    if name in _BASIS_1Q:
        if len(qargs) != 1:
            raise QiskitError("Invalid qubits for 1-qubit gate.")
        return _BASIS_1Q[name](clifford, qargs[0])
    if name in _BASIS_2Q:
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate.")
        return _BASIS_2Q[name](clifford, qargs[0], qargs[1])

    # If u gate, check if it is a Clifford, and if so, apply it
    if isinstance(gate, Gate) and name == "u" and len(qargs) == 1:
        try:
            theta, phi, lam = tuple(_n_half_pis(par) for par in gate.params)
        except ValueError as err:
            raise QiskitError("U gate angles must be multiples of pi/2 to be a Clifford") from err
        if theta == 0:
            clifford = _append_rz(clifford, qargs[0], lam + phi)
        elif theta == 1:
            clifford = _append_rz(clifford, qargs[0], lam - 2)
            clifford = _append_h(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi)
        elif theta == 2:
            clifford = _append_rz(clifford, qargs[0], lam - 1)
            clifford = _append_x(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi + 1)
        elif theta == 3:
            clifford = _append_rz(clifford, qargs[0], lam)
            clifford = _append_h(clifford, qargs[0])
            clifford = _append_rz(clifford, qargs[0], phi + 2)
        return clifford

    # If gate is a Clifford, we can either unroll the gate using the "to_circuit"
    # method, or we can compose the Cliffords directly. Experimentally, for large
    # cliffords the second method is considerably faster.

    from qiskit.quantum_info import Clifford

    if isinstance(gate, Clifford):
        composed_clifford = clifford.compose(gate, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford

    from qiskit.circuit.library import LinearFunction

    if isinstance(gate, LinearFunction):
        gate_as_clifford = Clifford.from_linear_function(gate)
        composed_clifford = clifford.compose(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford

    from qiskit.circuit.library import PermutationGate

    if isinstance(gate, PermutationGate):
        gate_as_clifford = Clifford.from_permutation(gate)
        composed_clifford = clifford.compose(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford

    # If the gate is not directly appendable, we try to unroll the gate with its definition.
    # This succeeds only if the gate has all-Clifford definition (decomposition).
    # If fails, we need to restore the clifford that was before attempting to unroll and append.
    if gate.definition is not None:
        try:
            return _append_circuit(clifford.copy(), gate.definition, qargs)
        except QiskitError:
            pass

    # As a final attempt, if the gate is up to 3 qubits,
    # we try to construct a Clifford to be appended from its matrix representation.
    if isinstance(gate, Gate) and len(qargs) <= 3:
        try:
            matrix = gate.to_matrix()
            gate_cliff = Clifford.from_matrix(matrix)
            return _append_operation(clifford, gate_cliff, qargs=qargs)
        except TypeError as err:
            raise QiskitError(f"Cannot apply {gate.name} gate with unbounded parameters") from err
        except CircuitError as err:
            raise QiskitError(f"Cannot apply {gate.name} gate without to_matrix defined") from err
        except QiskitError as err:
            raise QiskitError(f"Cannot apply non-Clifford gate: {gate.name}") from err

    raise QiskitError(f"Cannot apply {gate}")


def _prepend_operation(clifford, operation, qargs=None):
    """Update Clifford inplace by prepending a Clifford operation.

    Args:
        clifford (Clifford): The Clifford to update.
        operation (Instruction or Clifford or str): The operation or composite operation to apply
           before the clifford.
        qargs (list or None): The qubits to apply operation to.

    Returns:
        Clifford: the updated Clifford.

    Raises:
        QiskitError: if input operation cannot be converted into Clifford operations.
    """

    if isinstance(operation, (Barrier, Delay)):
        return clifford

    if qargs is None:
        qargs = list(range(clifford.num_qubits))

    gate = operation

    if isinstance(gate, str):
        # Check if gate is a valid Clifford basis gate string
        if gate not in _BASIS_PREP_1Q and gate not in _BASIS_PREP_2Q:
            raise QiskitError(f"Invalid Clifford gate name string {gate}")
        name = gate
    else:
        # assert isinstance(gate, Instruction)
        name = gate.name

    # Apply gate if it is a Clifford basis gate
    if name in _NON_CLIFFORD:
        raise QiskitError(f"Cannot update Clifford with non-Clifford gate {name}")
    if name in _BASIS_PREP_1Q:
        if len(qargs) != 1:
            raise QiskitError("Invalid qubits for 1-qubit gate.")
        return _BASIS_PREP_1Q[name](clifford, qargs[0])
    if name in _BASIS_PREP_2Q:
        if len(qargs) != 2:
            raise QiskitError("Invalid qubits for 2-qubit gate.")
        return _BASIS_PREP_2Q[name](clifford, qargs[0], qargs[1])

    # If u gate, check if it is a Clifford, and if so, apply it
    if isinstance(gate, Gate) and name == "u" and len(qargs) == 1:
        try:
            theta, phi, lam = tuple(_n_half_pis(par) for par in gate.params)
        except ValueError as err:
            raise QiskitError("U gate angles must be multiples of pi/2 to be a Clifford") from err
        if theta == 0:
            clifford = _prepend_rz(clifford, qargs[0], lam + phi)
        elif theta == 1:
            clifford = _prepend_rz(clifford, qargs[0], phi)
            clifford = _prepend_h(clifford, qargs[0])
            clifford = _prepend_rz(clifford, qargs[0], lam - 2)
        elif theta == 2:
            clifford = _prepend_rz(clifford, qargs[0], phi + 1)
            clifford = _prepend_x(clifford, qargs[0])
            clifford = _prepend_rz(clifford, qargs[0], lam - 1)
        elif theta == 3:
            clifford = _prepend_rz(clifford, qargs[0], phi + 2)
            clifford = _prepend_h(clifford, qargs[0])
            clifford = _prepend_rz(clifford, qargs[0], lam)
        return clifford

    # If gate is a Clifford, we can either unroll the gate using the "to_circuit"
    # method, or we can compose (dot) the Cliffords directly. Experimentally, for large
    # cliffords the second method is considerably faster.

    from qiskit.quantum_info import Clifford

    if isinstance(gate, Clifford):
        composed_clifford = clifford.dot(gate, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford

    from qiskit.circuit.library import LinearFunction

    if isinstance(gate, LinearFunction):
        gate_as_clifford = Clifford.from_linear_function(gate)
        composed_clifford = clifford.dot(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford

    from qiskit.circuit.library import PermutationGate

    if isinstance(gate, PermutationGate):
        gate_as_clifford = Clifford.from_permutation(gate)
        composed_clifford = clifford.dot(gate_as_clifford, qargs=qargs, front=False)
        clifford.tableau = composed_clifford.tableau
        return clifford

    # If the gate is not directly prependable, we try to unroll the gate with its definition.
    # This succeeds only if the gate has all-Clifford definition (decomposition).
    # If fails, we need to restore the clifford that was before attempting to unroll and append.
    if gate.definition is not None:
        try:
            return _prepend_circuit(clifford.copy(), gate.definition, qargs)
        except QiskitError:
            pass

    # As a final attempt, if the gate is up to 3 qubits,
    # we try to construct a Clifford to be prepended from its matrix representation.
    if isinstance(gate, Gate) and len(qargs) <= 3:
        try:
            matrix = gate.to_matrix()
            gate_cliff = Clifford.from_matrix(matrix)
            return _prepend_operation(clifford, gate_cliff, qargs=qargs)
        except TypeError as err:
            raise QiskitError(f"Cannot apply {gate.name} gate with unbounded parameters") from err
        except CircuitError as err:
            raise QiskitError(f"Cannot apply {gate.name} gate without to_matrix defined") from err
        except QiskitError as err:
            raise QiskitError(f"Cannot apply non-Clifford gate: {gate.name}") from err

    raise QiskitError(f"Cannot apply {gate}")


def _n_half_pis(param) -> int:
    try:
        param = float(param)
        epsilon = (abs(param) + 0.5 * 1e-10) % (np.pi / 2)
        if epsilon > 1e-10:
            raise ValueError(f"{param} is not to a multiple of pi/2")
        multiple = int(np.round(param / (np.pi / 2)))
        return multiple % 4
    except TypeError as err:
        raise ValueError(f"{param} is not bounded") from err


def _count_y(x, z, dtype=None):
    """Count the number of Y Paulis"""
    return (x & z).sum(axis=0, dtype=dtype)


def _calculate_composed_phase(x1, z1, x2, z2):
    """Direct calculation of the phase of Pauli((x1, z1)).compose(Pauli(x2, z2))"""
    cnt_phase = 2 * _count_y(x2, z1)
    cnt_y1 = _count_y(x1, z1)
    cnt_y2 = _count_y(x2, z2)
    cnt_y = _count_y(x1 ^ x2, z1 ^ z2)
    phase = (cnt_phase + cnt_y - cnt_y1 - cnt_y2) % 4
    return phase


# ---------------------------------------------------------------------
# Helper functions for applying basis gates
# ---------------------------------------------------------------------
def _append_rz(clifford, qubit, multiple):
    """Apply an Rz gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.
        multiple (int): z-rotation angle in a multiple of pi/2

    Returns:
        Clifford: the updated Clifford.
    """
    if multiple % 4 == 1:
        return _append_s(clifford, qubit)
    if multiple % 4 == 2:
        return _append_z(clifford, qubit)
    if multiple % 4 == 3:
        return _append_sdg(clifford, qubit)

    return clifford


def _prepend_rz(clifford, qubit, multiple):
    """Apply an Rz gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.
        multiple (int): z-rotation angle in a multiple of pi/2

    Returns:
        Clifford: the updated Clifford.
    """
    if multiple % 4 == 1:
        return _prepend_s(clifford, qubit)
    if multiple % 4 == 2:
        return _prepend_z(clifford, qubit)
    if multiple % 4 == 3:
        return _prepend_sdg(clifford, qubit)

    return clifford


def _append_i(clifford, qubit):
    """Apply an I gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """

    return clifford


def _prepend_i(clifford, qubit):
    """Apply an I gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """

    return clifford


def _append_x(clifford, qubit):
    """Apply an X gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.phase ^= clifford.z[:, qubit]
    return clifford


def _prepend_x(clifford, qubit):
    """Apply an X gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.stab_phase[qubit] ^= True
    return clifford


def _append_y(clifford, qubit):
    """Apply a Y gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x ^ z
    return clifford


def _prepend_y(clifford, qubit):
    """Apply a Y gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.stab_phase[qubit] ^= True
    clifford.destab_phase[qubit] ^= True
    return clifford


def _append_z(clifford, qubit):
    """Apply a Z gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.phase ^= clifford.x[:, qubit]
    return clifford


def _prepend_z(clifford, qubit):
    """Apply a Z gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.destab_phase[qubit] ^= True
    return clifford


def _append_h(clifford, qubit):
    """Apply a H gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & z
    tmp = x.copy()
    x[:] = z
    z[:] = tmp
    return clifford


def _prepend_h(clifford, qubit):
    """Apply a H gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab = clifford.destab[qubit, :]
    stab = clifford.stab[qubit, :]

    tmp = destab.copy()
    destab[:] = stab
    stab[:] = tmp
    return clifford


def _append_s(clifford, qubit):
    """Apply an S gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]

    clifford.phase ^= x & z
    z ^= x
    return clifford


def _prepend_s(clifford, qubit):
    """Apply an S gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab_x = clifford.destab_x[qubit, :]
    stab_x = clifford.stab_x[qubit, :]
    destab_z = clifford.destab_z[qubit, :]
    stab_z = clifford.stab_z[qubit, :]

    destab_x ^= stab_x
    destab_z ^= stab_z
    clifford.destab_phase[qubit] ^= clifford.stab_phase[qubit]

    phase = _calculate_composed_phase(destab_x, destab_z, stab_x, stab_z)
    if phase == 1:
        clifford.destab_phase[qubit] ^= True

    return clifford


def _append_sdg(clifford, qubit):
    """Apply an Sdg gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    clifford.phase ^= x & ~z
    z ^= x
    return clifford


def _prepend_sdg(clifford, qubit):
    """Apply an Sdg gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab_x = clifford.destab_x[qubit, :]
    stab_x = clifford.stab_x[qubit, :]
    destab_z = clifford.destab_z[qubit, :]
    stab_z = clifford.stab_z[qubit, :]

    destab_x ^= stab_x
    destab_z ^= stab_z
    clifford.destab_phase[qubit] ^= clifford.stab_phase[qubit]

    phase = _calculate_composed_phase(destab_x, destab_z, stab_x, stab_z)
    if phase == 3:
        clifford.destab_phase[qubit] ^= True

    return clifford


def _append_sx(clifford, qubit):
    """Apply an SX gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]

    clifford.phase ^= ~x & z
    x ^= z
    return clifford


def _prepend_sx(clifford, qubit):
    """Apply an SX gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab_x = clifford.destab_x[qubit, :]
    stab_x = clifford.stab_x[qubit, :]
    destab_z = clifford.destab_z[qubit, :]
    stab_z = clifford.stab_z[qubit, :]

    stab_x ^= destab_x
    stab_z ^= destab_z
    clifford.stab_phase[qubit] ^= clifford.destab_phase[qubit]

    phase = _calculate_composed_phase(stab_x, stab_z, destab_x, destab_z)
    if phase == 1:
        clifford.stab_phase[qubit] ^= True

    return clifford


def _append_sxdg(clifford, qubit):
    """Apply an SXdg gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]

    clifford.phase ^= x & z
    x ^= z
    return clifford


def _prepend_sxdg(clifford, qubit):
    """Apply an SXdg gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab_x = clifford.destab_x[qubit, :]
    stab_x = clifford.stab_x[qubit, :]
    destab_z = clifford.destab_z[qubit, :]
    stab_z = clifford.stab_z[qubit, :]

    stab_x ^= destab_x
    stab_z ^= destab_z
    clifford.stab_phase[qubit] ^= clifford.destab_phase[qubit]

    phase = _calculate_composed_phase(stab_x, stab_z, destab_x, destab_z)
    if phase == 3:
        clifford.stab_phase[qubit] ^= True

    return clifford


def _append_v(clifford, qubit):
    """Apply a V gate to a Clifford.

    This is equivalent to an Sdg gate followed by a H gate.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = x.copy()
    x ^= z
    z[:] = tmp
    return clifford


def _prepend_v(clifford, qubit):
    """Apply a V gate before a Clifford.

    This is equivalent to an Sdg gate followed by a H gate.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab = clifford.destab[qubit, :]
    stab = clifford.stab[qubit, :]

    tmp = destab.copy()
    destab ^= stab
    clifford.stab[qubit, :] = tmp

    destab_x = clifford.destab_x[qubit, :]
    stab_x = clifford.stab_x[qubit, :]
    destab_z = clifford.destab_z[qubit, :]
    stab_z = clifford.stab_z[qubit, :]

    phase = _calculate_composed_phase(destab_x, destab_z, stab_x, stab_z)
    if phase == 3:
        clifford.destab_phase[qubit] ^= True

    return clifford


def _append_w(clifford, qubit):
    """Apply a W gate to a Clifford.

    This is equivalent to two V gates.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x = clifford.x[:, qubit]
    z = clifford.z[:, qubit]
    tmp = z.copy()
    z ^= x
    x[:] = tmp
    return clifford


def _prepend_w(clifford, qubit):
    """Apply a W gate before a Clifford.

    This is equivalent to an H gate followed by an S gate.

    Args:
        clifford (Clifford): a Clifford.
        qubit (int): gate qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab = clifford.destab[qubit, :]
    stab = clifford.stab[qubit, :]

    tmp = stab.copy()
    stab ^= destab
    clifford.destab[qubit, :] = tmp

    destab_x = clifford.destab_x[qubit, :]
    stab_x = clifford.stab_x[qubit, :]
    destab_z = clifford.destab_z[qubit, :]
    stab_z = clifford.stab_z[qubit, :]

    phase = _calculate_composed_phase(stab_x, stab_z, destab_x, destab_z)
    if phase == 1:
        clifford.stab_phase[qubit] ^= True

    return clifford


def _append_cx(clifford, control, target):
    """Apply a CX gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= (x1 ^ z0 ^ True) & z1 & x0
    x1 ^= x0
    z0 ^= z1
    return clifford


def _prepend_cx(clifford, control, target):
    """Apply a CX gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab_x_c = clifford.destab_x[control]
    destab_z_c = clifford.destab_z[control]
    stab_x_c = clifford.stab_x[control]
    stab_z_c = clifford.stab_z[control]
    destab_x_t = clifford.destab_x[target]
    destab_z_t = clifford.destab_z[target]
    stab_x_t = clifford.stab_x[target]
    stab_z_t = clifford.stab_z[target]

    destab_x_c ^= destab_x_t
    destab_z_c ^= destab_z_t
    stab_x_t ^= stab_x_c
    stab_z_t ^= stab_z_c
    clifford.destab_phase[control] ^= clifford.destab_phase[target]
    clifford.stab_phase[target] ^= clifford.stab_phase[control]

    phase_control = _calculate_composed_phase(destab_x_c, destab_z_c, destab_x_t, destab_z_t)
    phase_target = _calculate_composed_phase(stab_x_c, stab_z_c, stab_x_t, stab_z_t)
    clifford.destab_phase[control] ^= phase_control != 0
    clifford.stab_phase[target] ^= phase_target != 0
    return clifford


def _append_cz(clifford, control, target):
    """Apply a CZ gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    x0 = clifford.x[:, control]
    z0 = clifford.z[:, control]
    x1 = clifford.x[:, target]
    z1 = clifford.z[:, target]
    clifford.phase ^= x0 & x1 & (z0 ^ z1)
    z1 ^= x0
    z0 ^= x1
    return clifford


def _prepend_cz(clifford, control, target):
    """Apply a CZ gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    destab_x_c = clifford.destab_x[control]
    destab_z_c = clifford.destab_z[control]
    stab_x_c = clifford.stab_x[control]
    stab_z_c = clifford.stab_z[control]
    destab_x_t = clifford.destab_x[target]
    destab_z_t = clifford.destab_z[target]
    stab_x_t = clifford.stab_x[target]
    stab_z_t = clifford.stab_z[target]

    destab_x_c ^= stab_x_t
    destab_z_c ^= stab_z_t
    destab_x_t ^= stab_x_c
    destab_z_t ^= stab_z_c
    clifford.destab_phase[control] ^= clifford.stab_phase[target]
    clifford.destab_phase[target] ^= clifford.stab_phase[control]

    phase_control = _calculate_composed_phase(destab_x_c, destab_z_c, stab_x_t, stab_z_t)
    phase_target = _calculate_composed_phase(destab_x_t, destab_z_t, stab_x_c, stab_z_c)
    clifford.destab_phase[control] ^= phase_control != 0
    clifford.destab_phase[target] ^= phase_target != 0
    return clifford


def _append_cy(clifford, control, target):
    """Apply a CY gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _append_sdg(clifford, target)
    clifford = _append_cx(clifford, control, target)
    clifford = _append_s(clifford, target)
    return clifford


def _prepend_cy(clifford, control, target):
    """Apply a CY gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        control (int): gate control qubit index.
        target (int): gate target qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _prepend_s(clifford, target)
    clifford = _prepend_cx(clifford, control, target)
    clifford = _prepend_sdg(clifford, target)
    return clifford


def _append_swap(clifford, qubit0, qubit1):
    """Apply a Swap gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.x[:, [qubit0, qubit1]] = clifford.x[:, [qubit1, qubit0]]
    clifford.z[:, [qubit0, qubit1]] = clifford.z[:, [qubit1, qubit0]]
    return clifford


def _prepend_swap(clifford, qubit0, qubit1):
    """Apply a Swap gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford.stab[[qubit0, qubit1], :] = clifford.stab[[qubit1, qubit0], :]
    clifford.destab[[qubit0, qubit1], :] = clifford.destab[[qubit1, qubit0], :]
    return clifford


def _append_iswap(clifford, qubit0, qubit1):
    """Apply a iSwap gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _append_s(clifford, qubit0)
    clifford = _append_h(clifford, qubit0)
    clifford = _append_s(clifford, qubit1)
    clifford = _append_cx(clifford, qubit0, qubit1)
    clifford = _append_cx(clifford, qubit1, qubit0)
    clifford = _append_h(clifford, qubit1)
    return clifford


def _prepend_iswap(clifford, qubit0, qubit1):
    """Apply a iSwap gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _prepend_s(clifford, qubit0)
    clifford = _prepend_h(clifford, qubit0)
    clifford = _prepend_s(clifford, qubit1)
    clifford = _prepend_cx(clifford, qubit0, qubit1)
    clifford = _prepend_cx(clifford, qubit1, qubit0)
    clifford = _prepend_h(clifford, qubit1)
    return clifford


def _append_dcx(clifford, qubit0, qubit1):
    """Apply a DCX gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _append_cx(clifford, qubit0, qubit1)
    clifford = _append_cx(clifford, qubit1, qubit0)
    return clifford


def _prepend_dcx(clifford, qubit0, qubit1):
    """Apply a DCX gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _prepend_cx(clifford, qubit1, qubit0)
    clifford = _prepend_cx(clifford, qubit0, qubit1)
    return clifford


def _append_ecr(clifford, qubit0, qubit1):
    """Apply an ECR gate to a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _append_s(clifford, qubit0)
    clifford = _append_sx(clifford, qubit1)
    clifford = _append_cx(clifford, qubit0, qubit1)
    clifford = _append_x(clifford, qubit0)

    return clifford


def _prepend_ecr(clifford, qubit0, qubit1):
    """Apply an ECR gate before a Clifford.

    Args:
        clifford (Clifford): a Clifford.
        qubit0 (int): first qubit index.
        qubit1 (int): second  qubit index.

    Returns:
        Clifford: the updated Clifford.
    """
    clifford = _prepend_x(clifford, qubit0)
    clifford = _prepend_cx(clifford, qubit0, qubit1)
    clifford = _prepend_sx(clifford, qubit1)
    clifford = _prepend_s(clifford, qubit0)

    return clifford


# Basis Clifford Gates
_BASIS_1Q = {
    "i": _append_i,
    "id": _append_i,
    "iden": _append_i,
    "x": _append_x,
    "y": _append_y,
    "z": _append_z,
    "h": _append_h,
    "s": _append_s,
    "sdg": _append_sdg,
    "sinv": _append_sdg,
    "sx": _append_sx,
    "sxdg": _append_sxdg,
    "v": _append_v,
    "w": _append_w,
}
_BASIS_2Q = {
    "cx": _append_cx,
    "cz": _append_cz,
    "cy": _append_cy,
    "swap": _append_swap,
    "iswap": _append_iswap,
    "ecr": _append_ecr,
    "dcx": _append_dcx,
}


# Basis Clifford Gates
_BASIS_PREP_1Q = {
    "i": _prepend_i,
    "id": _prepend_i,
    "iden": _prepend_i,
    "x": _prepend_x,
    "y": _prepend_y,
    "z": _prepend_z,
    "h": _prepend_h,
    "s": _prepend_s,
    "sdg": _prepend_sdg,
    "sinv": _prepend_sdg,
    "sx": _prepend_sx,
    "sxdg": _prepend_sxdg,
    "v": _prepend_v,
    "w": _prepend_w,
}
_BASIS_PREP_2Q = {
    "cx": _prepend_cx,
    "cz": _prepend_cz,
    "cy": _prepend_cy,
    "swap": _prepend_swap,
    "iswap": _prepend_iswap,
    "ecr": _prepend_ecr,
    "dcx": _prepend_dcx,
}


# Clifford gate names
_CLIFFORD_GATE_NAMES = [
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "sx",
    "sxdg",
    "cx",
    "cz",
    "cy",
    "swap",
    "iswap",
    "ecr",
    "dcx",
]

# Non-clifford gates
_NON_CLIFFORD = {"t", "tdg", "ccx", "ccz"}


def get_clifford_gate_names() -> list:
    """Returns the list of Clifford gate names."""
    return _CLIFFORD_GATE_NAMES
