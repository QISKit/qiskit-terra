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

"""Compute the product of two qubit registers using classical multiplication approach."""


from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit

from .multiplier import Multiplier


class HRSCumulativeMultiplier(Multiplier):
    r"""A multiplication circuit to store product of two input registers out-of-place.

    Circuit uses the approach from [1]. As an example, a multiplier circuit that
    performs a non-modular multiplication on two 3-qubit sized registers with
    the default adder is as follows (where ``Adder`` denotes the
    ``CDKMRippleCarryAdder``):

    .. code-block:: text

          a_0: ────■─────────────────────────
                   │
          a_1: ────┼─────────■───────────────
                   │         │
          a_2: ────┼─────────┼─────────■─────
               ┌───┴────┐┌───┴────┐┌───┴────┐
          b_0: ┤0       ├┤0       ├┤0       ├
               │        ││        ││        │
          b_1: ┤1       ├┤1       ├┤1       ├
               │        ││        ││        │
          b_2: ┤2       ├┤2       ├┤2       ├
               │        ││        ││        │
        out_0: ┤3       ├┤        ├┤        ├
               │        ││        ││        │
        out_1: ┤4       ├┤3       ├┤        ├
               │  Adder ││  Adder ││  Adder │
        out_2: ┤5       ├┤4       ├┤3       ├
               │        ││        ││        │
        out_3: ┤6       ├┤5       ├┤4       ├
               │        ││        ││        │
        out_4: ┤        ├┤6       ├┤5       ├
               │        ││        ││        │
        out_5: ┤        ├┤        ├┤6       ├
               │        ││        ││        │
        aux_0: ┤7       ├┤7       ├┤7       ├
               └────────┘└────────┘└────────┘

    Multiplication in this circuit is implemented in a classical approach by performing
    a series of shifted additions using one of the input registers while the qubits
    from the other input register act as control qubits for the adders.

    .. seealso::

        The :class:`.MultiplierGate` object represents a multiplication, like this circuit class,
        but allows the compiler to select the optimal decomposition based on the context.
        Specific implementations can be set via the :class:`.HLSConfig`, e.g. this circuit
        can be chosen via ``Multiplier=["cumulative_h18"]``.

    References:

    [1] Häner et al., Optimizing Quantum Circuits for Arithmetic, 2018.
    `arXiv:1805.12445 <https://arxiv.org/pdf/1805.12445.pdf>`_

    """

    def __init__(
        self,
        num_state_qubits: int,
        num_result_qubits: int | None = None,
        adder: QuantumCircuit | None = None,
        name: str = "HRSCumulativeMultiplier",
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits in either input register for
                state :math:`|a\rangle` or :math:`|b\rangle`. The two input
                registers must have the same number of qubits.
            num_result_qubits: The number of result qubits to limit the output to.
                If number of result qubits is :math:`n`, multiplication modulo :math:`2^n` is performed
                to limit the output to the specified number of qubits. Default
                value is ``2 * num_state_qubits`` to represent any possible
                result from the multiplication of the two inputs.
            adder: Half adder circuit to be used for performing multiplication. The
                CDKMRippleCarryAdder is used as default if no adder is provided.
            name: The name of the circuit object.
        Raises:
            NotImplementedError: If ``num_result_qubits`` is not default and a custom adder is provided.
        """
        super().__init__(num_state_qubits, num_result_qubits, name=name)

        if self.num_result_qubits != 2 * num_state_qubits and adder is not None:
            raise NotImplementedError("Only default adder is supported for modular multiplication.")

        # define the registers
        qr_a = QuantumRegister(num_state_qubits, name="a")
        qr_b = QuantumRegister(num_state_qubits, name="b")
        qr_out = QuantumRegister(self.num_result_qubits, name="out")
        self.add_register(qr_a, qr_b, qr_out)

        # prepare adder as controlled gate
        if adder is None:
            from qiskit.circuit.library.arithmetic.adders import CDKMRippleCarryAdder

            adder = CDKMRippleCarryAdder(num_state_qubits, kind="half")

        # get the number of helper qubits needed
        num_helper_qubits = adder.num_ancillas

        # add helper qubits if required
        if num_helper_qubits > 0:
            qr_h = AncillaRegister(num_helper_qubits, name="helper")  # helper/ancilla qubits
            self.add_register(qr_h)

        # build multiplication circuit
        circuit = QuantumCircuit(*self.qregs, name=name)

        for i in range(num_state_qubits):
            excess_qubits = max(0, num_state_qubits + i + 1 - self.num_result_qubits)
            if excess_qubits == 0:
                num_adder_qubits = num_state_qubits
                adder_for_current_step = adder
            else:
                num_adder_qubits = num_state_qubits - excess_qubits + 1
                adder_for_current_step = CDKMRippleCarryAdder(num_adder_qubits, kind="fixed")
            controlled_adder = adder_for_current_step.to_gate().control(1, annotated=False)
            qr_list = (
                [qr_a[i]]
                + qr_b[:num_adder_qubits]
                + qr_out[i : num_state_qubits + i + 1 - excess_qubits]
            )
            if num_helper_qubits > 0:
                qr_list.extend(qr_h[:])
            circuit.append(controlled_adder, qr_list)

        self.append(circuit.to_gate(), self.qubits)
