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

"""Compute the sum of two qubit registers using ripple-carry approach."""

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister


def adder_ripple_c04(num_state_qubits: int, kind: str = "half") -> QuantumCircuit:
    r"""A ripple-carry circuit to perform in-place addition on two qubit registers.

    This circuit uses :math:`2n + O(1)` RCCX gates and :math:`5n + O(1)` CX gates,
    at a depth of :math:`2n + O(1)` [1].
    The relative phases introduced during majority computation cancel during
    uncomputation. The constant depends on the kind of adder implemented.

    As an example, a ripple-carry adder circuit that performs addition on two 3-qubit sized
    registers with a carry-in bit (``kind="full"``) is as follows:

    .. parsed-literal::

                ┌──────┐                                     ┌──────┐
         cin_0: ┤2     ├─────────────────────────────────────┤2     ├
                │      │┌──────┐                     ┌──────┐│      │
           a_0: ┤0     ├┤2     ├─────────────────────┤2     ├┤0     ├
                │      ││      │┌──────┐     ┌──────┐│      ││      │
           a_1: ┤  MAJ ├┤0     ├┤2     ├─────┤2     ├┤0     ├┤  UMA ├
                │      ││      ││      │     │      ││      ││      │
           a_2: ┤      ├┤  MAJ ├┤0     ├──■──┤0     ├┤  UMA ├┤      ├
                │      ││      ││      │  │  │      ││      ││      │
           b_0: ┤1     ├┤      ├┤  MAJ ├──┼──┤  UMA ├┤      ├┤1     ├
                └──────┘│      ││      │  │  │      ││      │└──────┘
           b_1: ────────┤1     ├┤      ├──┼──┤      ├┤1     ├────────
                        └──────┘│      │  │  │      │└──────┘
           b_2: ────────────────┤1     ├──┼──┤1     ├────────────────
                                └──────┘┌─┴─┐└──────┘
        cout_0: ────────────────────────┤ X ├────────────────────────
                                        └───┘

    Here *MAJ* and *UMA* gates correspond to the gates introduced in [1]. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.

    Two different kinds of adders are supported. By setting the ``kind`` argument, you can also
    choose a half-adder, which doesn't have a carry-in, and a fixed-sized-adder, which has neither
    carry-in nor carry-out, and thus acts on fixed register sizes. Unlike the full-adder,
    these circuits need one additional helper qubit.

    The circuit diagram for the fixed-point adder (``kind="fixed"``) on 3-qubit sized inputs is

    .. parsed-literal::

                ┌──────┐┌──────┐                ┌──────┐┌──────┐
           a_0: ┤0     ├┤2     ├────────────────┤2     ├┤0     ├
                │      ││      │┌──────┐┌──────┐│      ││      │
           a_1: ┤      ├┤0     ├┤2     ├┤2     ├┤0     ├┤      ├
                │      ││      ││      ││      ││      ││      │
           a_2: ┤      ├┤  MAJ ├┤0     ├┤0     ├┤  UMA ├┤      ├
                │      ││      ││      ││      ││      ││      │
           b_0: ┤1 MAJ ├┤      ├┤  MAJ ├┤  UMA ├┤      ├┤1 UMA ├
                │      ││      ││      ││      ││      ││      │
           b_1: ┤      ├┤1     ├┤      ├┤      ├┤1     ├┤      ├
                │      │└──────┘│      ││      │└──────┘│      │
           b_2: ┤      ├────────┤1     ├┤1     ├────────┤      ├
                │      │        └──────┘└──────┘        │      │
        help_0: ┤2     ├────────────────────────────────┤2     ├
                └──────┘                                └──────┘

    It has one less qubit than the full-adder since it doesn't have the carry-out, but uses
    a helper qubit instead of the carry-in, so it only has one less qubit, not two.

    Args:
        num_state_qubits: The number of qubits in either input register for
            state :math:`|a\rangle` or :math:`|b\rangle`. The two input
            registers must have the same number of qubits.
        kind: The kind of adder, can be ``"full"`` for a full adder, ``"half"`` for a half
            adder, or ``"fixed"`` for a fixed-sized adder. A full adder includes both carry-in
            and carry-out, a half only carry-out, and a fixed-sized adder neither carry-in
            nor carry-out.

    Raises:
        ValueError: If ``num_state_qubits`` is lower than 1.

    References:

    [1] Cuccaro et al., A new quantum ripple-carry addition circuit, 2004.
    `arXiv:quant-ph/0410184 <https://arxiv.org/pdf/quant-ph/0410184.pdf>`_

    [2] Vedral et al., Quantum Networks for Elementary Arithmetic Operations, 1995.
    `arXiv:quant-ph/9511018 <https://arxiv.org/pdf/quant-ph/9511018.pdf>`_

    """
    if num_state_qubits < 1:
        raise ValueError("The number of qubits must be at least 1.")

    circuit = QuantumCircuit()

    if kind == "full":
        qr_c = QuantumRegister(1, name="cin")
        circuit.add_register(qr_c)
    else:
        qr_c = AncillaRegister(1, name="help")

    qr_a = QuantumRegister(num_state_qubits, name="a")
    qr_b = QuantumRegister(num_state_qubits, name="b")
    circuit.add_register(qr_a, qr_b)

    if kind in ["full", "half"]:
        qr_z = QuantumRegister(1, name="cout")
        circuit.add_register(qr_z)

    if kind != "full":
        circuit.add_register(qr_c)

    # build carry circuit for majority of 3 bits in-place
    # corresponds to MAJ gate in [1]
    qc_maj = QuantumCircuit(3, name="MAJ")
    qc_maj.cx(0, 1)
    qc_maj.cx(0, 2)
    qc_maj.rccx(2, 1, 0)
    maj_gate = qc_maj.to_gate()

    # build circuit for reversing carry operation
    # corresponds to UMA gate in [1]
    qc_uma = QuantumCircuit(3, name="UMA")
    qc_uma.rccx(2, 1, 0)
    qc_uma.cx(0, 2)
    qc_uma.cx(2, 1)
    uma_gate = qc_uma.to_gate()

    # build ripple-carry adder circuit
    circuit.append(maj_gate, [qr_a[0], qr_b[0], qr_c[0]])

    for i in range(num_state_qubits - 1):
        circuit.append(maj_gate, [qr_a[i + 1], qr_b[i + 1], qr_a[i]])

    if kind in ["full", "half"]:
        circuit.cx(qr_a[-1], qr_z[0])

    for i in reversed(range(num_state_qubits - 1)):
        circuit.append(uma_gate, [qr_a[i + 1], qr_b[i + 1], qr_a[i]])

    circuit.append(uma_gate, [qr_a[0], qr_b[0], qr_c[0]])

    return circuit
