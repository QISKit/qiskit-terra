# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from qiskit.circuit.library import CXGate


def generate_gray_code(num_bits):
    """Generate the gray code for ``num_bits`` bits."""
    if num_bits <= 0:
        raise ValueError("Cannot generate the gray code for less than 1 bit.")
    result = [0]
    for i in range(num_bits):
        result += [x + 2**i for x in reversed(result)]
    return [format(x, f"0{num_bits}b") for x in result]


def gray_code_chain(q, num_ctrl_qubits, gate):
    """Apply the gate to the last qubit in the register ``q``, controlled on all
    preceding qubits. This function uses the gray code to propagate down to the last qubit.

    Ported and adapted from Aqua (github.com/Qiskit/qiskit-aqua),
    commit 769ca8d, file qiskit/aqua/circuits/gates/multi_control_u1_gate.py.
    """
    rule = []
    q_controls, q_target = q[:num_ctrl_qubits], q[num_ctrl_qubits]
    gray_code = generate_gray_code(num_ctrl_qubits)
    last_pattern = None

    for pattern in gray_code:
        if "1" not in pattern:
            continue
        if last_pattern is None:
            last_pattern = pattern
        # find left most set bit
        lm_pos = list(pattern).index("1")

        # find changed bit
        comp = [i != j for i, j in zip(pattern, last_pattern)]
        if True in comp:
            pos = comp.index(True)
        else:
            pos = None
        if pos is not None:
            if pos != lm_pos:
                rule.append((CXGate(), [q_controls[pos], q_controls[lm_pos]], []))
            else:
                indices = [i for i, x in enumerate(pattern) if x == "1"]
                for idx in indices[1:]:
                    rule.append((CXGate(), [q_controls[idx], q_controls[lm_pos]], []))
        # check parity
        if pattern.count("1") % 2 == 0:
            # inverse
            rule.append((gate.inverse(), [q_controls[lm_pos], q_target], []))
        else:
            rule.append((gate, [q_controls[lm_pos], q_target], []))
        last_pattern = pattern

    return rule
