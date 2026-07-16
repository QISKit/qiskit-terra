# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute the sum of two qubit registers using Classical Addition."""

from __future__ import annotations

from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit, Bit


def adder_ripple_v95(num_state_qubits: int, kind: str = "half") -> QuantumCircuit:
    r"""The VBE ripple carry adder [1].

    This method uses :math:`4n + O(1)` CCX gates and :math:`4n + 1` CX gates at a depth
    of :math:`6n - 2` [2].

    This circuit performs inplace addition of two equally-sized quantum registers.
    As an example, a classical adder circuit that performs full addition (i.e. including
    a carry-in bit) on two 2-qubit sized registers is as follows:

    .. parsed-literal::

                  ┌────────┐                       ┌───────────┐┌──────┐
           cin_0: ┤0       ├───────────────────────┤0          ├┤0     ├
                  │        │                       │           ││      │
             a_0: ┤1       ├───────────────────────┤1          ├┤1     ├
                  │        │┌────────┐     ┌──────┐│           ││  Sum │
             a_1: ┤        ├┤1       ├──■──┤1     ├┤           ├┤      ├
                  │        ││        │  │  │      ││           ││      │
             b_0: ┤2 Carry ├┤        ├──┼──┤      ├┤2 Carry_dg ├┤2     ├
                  │        ││        │┌─┴─┐│      ││           │└──────┘
             b_1: ┤        ├┤2 Carry ├┤ X ├┤2 Sum ├┤           ├────────
                  │        ││        │└───┘│      ││           │
          cout_0: ┤        ├┤3       ├─────┤      ├┤           ├────────
                  │        ││        │     │      ││           │
        helper_0: ┤3       ├┤0       ├─────┤0     ├┤3          ├────────
                  └────────┘└────────┘     └──────┘└───────────┘


    Here *Carry* and *Sum* gates correspond to the gates introduced in [1].
    *Carry_dg* correspond to the inverse of the *Carry* gate. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.
    This is different ordering as compared to Figure 2 in [1], which leads to a different
    drawing of the circuit.

    Args:
        num_state_qubits: The size of the register.
        kind: The kind of adder, can be ``"full"`` for a full adder, ``"half"`` for a half
            adder, or ``"fixed"`` for a fixed-sized adder. A full adder includes both carry-in
            and carry-out, a half only carry-out, and a fixed-sized adder neither carry-in
            nor carry-out.

    Raises:
        ValueError: If ``num_state_qubits`` is lower than 1.

    References:

    [1] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    [2] Cuccaro et al., A new quantum ripple-carry addition circuit, 2004.
    `arXiv:quant-ph/0410184 <https://arxiv.org/pdf/quant-ph/0410184.pdf>`_

    """
    if num_state_qubits < 1:
        raise ValueError("The number of qubits must be at least 1.")

    # define the input registers
    registers: list[QuantumRegister | list[Bit]] = []
    if kind == "full":
        qr_cin = QuantumRegister(1, name="cin")
        registers.append(qr_cin)
    else:
        qr_cin = QuantumRegister(0)

    qr_a = QuantumRegister(num_state_qubits, name="a")
    qr_b = QuantumRegister(num_state_qubits, name="b")

    registers += [qr_a, qr_b]

    if kind in ["half", "full"]:
        qr_cout = QuantumRegister(1, name="cout")
        registers.append(qr_cout)
    else:
        qr_cout = QuantumRegister(0)

    if num_state_qubits > 1:
        qr_help = AncillaRegister(num_state_qubits - 1, name="helper")
        registers.append(qr_help)
    else:
        qr_help = AncillaRegister(0)

    circuit = QuantumCircuit(*registers)

    # the code is simplified a lot if we create a list of all carries and helpers
    carries = qr_cin[:] + qr_help[:] + qr_cout[:]

    # corresponds to Carry gate in [1]
    qc_carry = QuantumCircuit(4, name="Carry")
    qc_carry.ccx(1, 2, 3)
    qc_carry.cx(1, 2)
    qc_carry.ccx(0, 2, 3)
    carry_gate = qc_carry.to_gate()
    carry_gate_dg = carry_gate.inverse()

    # corresponds to Sum gate in [1]
    qc_sum = QuantumCircuit(3, name="Sum")
    qc_sum.cx(1, 2)
    qc_sum.cx(0, 2)
    sum_gate = qc_sum.to_gate()

    # handle all cases for the first qubits, depending on whether cin is available
    i = 0
    if kind == "half":
        i += 1
        circuit.ccx(qr_a[0], qr_b[0], carries[0])
    elif kind == "fixed":
        i += 1
        if num_state_qubits == 1:
            circuit.cx(qr_a[0], qr_b[0])
        else:
            circuit.ccx(qr_a[0], qr_b[0], carries[0])

    for inp, out in zip(carries[:-1], carries[1:]):
        circuit.append(carry_gate, [inp, qr_a[i], qr_b[i], out])
        i += 1

    if kind in ["full", "half"]:  # final CX (cancels for the 'fixed' case)
        circuit.cx(qr_a[-1], qr_b[-1])

    if len(carries) > 1:
        circuit.append(sum_gate, [carries[-2], qr_a[-1], qr_b[-1]])

    i -= 2
    for j, (inp, out) in enumerate(zip(reversed(carries[:-1]), reversed(carries[1:]))):
        if j == 0:
            if kind == "fixed":
                i += 1
            else:
                continue
        circuit.append(carry_gate_dg, [inp, qr_a[i], qr_b[i], out])
        circuit.append(sum_gate, [inp, qr_a[i], qr_b[i]])
        i -= 1

    if kind in ["half", "fixed"] and num_state_qubits > 1:
        circuit.ccx(qr_a[0], qr_b[0], carries[0])
        circuit.cx(qr_a[0], qr_b[0])

    return circuit
