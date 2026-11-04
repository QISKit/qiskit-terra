# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multiple-Control, Multiple-Target Gate."""

from __future__ import annotations

import warnings
from collections.abc import Callable

from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumCircuit
from qiskit.circuit._utils import _ctrl_state_to_int
from qiskit.utils.deprecation import deprecate_func
from ..standard_gates import get_standard_gate_name_mapping


class MCMT(QuantumCircuit):
    """The multi-controlled multi-target gate, for an arbitrary singly controlled target gate.

    For example, the H gate controlled on 3 qubits and acting on 2 target qubits is represented as:

    .. code-block:: text

        ───■────
           │
        ───■────
           │
        ───■────
        ┌──┴───┐
        ┤0     ├
        │  2-H │
        ┤1     ├
        └──────┘

    This default implementation requires no ancilla qubits, by broadcasting the target gate
    to the number of target qubits and using Qiskit's generic control routine to control the
    broadcasted target on the control qubits. If ancilla qubits are available, a more efficient
    variant using the so-called V-chain decomposition can be used. This is implemented in
    :class:`~qiskit.circuit.library.MCMTVChain`.
    """

    @deprecate_func(since="1.4", additional_msg="Use MCMTGate instead.")
    def __init__(
        self,
        gate: Gate | Callable[[QuantumCircuit, circuit.Qubit, circuit.Qubit], circuit.Instruction],
        num_ctrl_qubits: int,
        num_target_qubits: int,
    ) -> None:
        """Create a new multi-control multi-target gate.

        Args:
            gate: The gate to be applied controlled on the control qubits and applied to the target
                qubits. Can be either a Gate or a circuit method.
                If it is a callable, it will be cast to a Gate.
            num_ctrl_qubits: The number of control qubits.
            num_target_qubits: The number of target qubits.

        Raises:
            AttributeError: If the gate cannot be cast to a controlled gate.
            AttributeError: If the number of controls or targets is 0.
        """
        if num_ctrl_qubits == 0 or num_target_qubits == 0:
            raise AttributeError("Need at least one control and one target qubit.")

        if callable(gate):
            warnings.warn(
                "Passing a callable to MCMT is pending deprecation since Qiskit 1.3. Pass a "
                "gate instance or the gate name instead, e.g. pass 'h' instead of QuantumCircuit.h.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            gate = gate.__name__
        elif isinstance(gate, QuantumCircuit):
            warnings.warn(
                "Passing a QuantumCircuit is pending deprecation since Qiskit 1.3. Pass a gate "
                "or turn the circuit into a gate using the ``to_gate`` method, instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            gate = gate.to_gate()

        self.gate = MCMTGate._identify_base_gate(gate)
        self.num_ctrl_qubits = num_ctrl_qubits
        self.num_target_qubits = num_target_qubits

        # initialize the circuit object
        num_qubits = num_ctrl_qubits + num_target_qubits + self.num_ancilla_qubits
        super().__init__(num_qubits, name="mcmt")
        self._build()

    def _build(self):
        gate = MCMTGate(self.gate, self.num_ctrl_qubits, self.num_target_qubits)
        self.append(gate, self.qubits)

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancillas."""
        return 0

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None, annotated=False):
        """Return the controlled version of the MCMT circuit."""
        if not annotated and ctrl_state is None:
            gate = MCMT(self.gate, self.num_ctrl_qubits + num_ctrl_qubits, self.num_target_qubits)
        else:
            gate = super().control(num_ctrl_qubits, label, ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool = False):
        """Return the inverse MCMT circuit, which is itself."""
        return MCMT(self.gate, self.num_ctrl_qubits, self.num_target_qubits)


class MCMTVChain(MCMT):
    """The MCMT implementation using the CCX V-chain.

    This implementation requires ancillas but is decomposed into a much shallower circuit
    than the default implementation in :class:`~qiskit.circuit.library.MCMT`.

    Expanded circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import MCMTVChain, ZGate
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = MCMTVChain(ZGate(), 2, 2)
       _generate_circuit_library_visualization(circuit.decompose())

    Examples:

        >>> from qiskit.circuit.library import HGate
        >>> MCMTVChain(HGate(), 3, 2).draw()

        q_0: ──■────────────────────────■──
               │                        │
        q_1: ──■────────────────────────■──
               │                        │
        q_2: ──┼────■──────────────■────┼──
               │    │  ┌───┐       │    │
        q_3: ──┼────┼──┤ H ├───────┼────┼──
               │    │  └─┬─┘┌───┐  │    │
        q_4: ──┼────┼────┼──┤ H ├──┼────┼──
             ┌─┴─┐  │    │  └─┬─┘  │  ┌─┴─┐
        q_5: ┤ X ├──■────┼────┼────■──┤ X ├
             └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
        q_6: ─────┤ X ├──■────■──┤ X ├─────
                  └───┘          └───┘
    """

    @deprecate_func(
        since="1.4",
        additional_msg="Use MCMTGate with the V-chain synthesis plugin instead.",
    )
    def __init__(
        self,
        gate: Gate | Callable[[QuantumCircuit, circuit.Qubit, circuit.Qubit], circuit.Instruction],
        num_ctrl_qubits: int,
        num_target_qubits: int,
    ) -> None:
        super().__init__(gate, num_ctrl_qubits, num_target_qubits)

    def _build(self):

        from qiskit.synthesis.multi_controlled import synth_mcmt_vchain

        synthesized = synth_mcmt_vchain(self.gate, self.num_ctrl_qubits, self.num_target_qubits)
        self.compose(synthesized, inplace=True, copy=False)

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancilla qubits required."""
        return max(0, self.num_ctrl_qubits - 1)

    def inverse(self, annotated: bool = False):
        return MCMTVChain(self.gate, self.num_ctrl_qubits, self.num_target_qubits)


class MCMTGate(ControlledGate):
    """The multi-controlled multi-target gate, for an arbitrary singly controlled target gate.

    For example, the H gate controlled on 3 qubits and acting on 2 target qubits is represented as:

    .. parsed-literal::

        ───■────
           │
        ───■────
           │
        ───■────
        ┌──┴───┐
        ┤0     ├
        │  2-H │
        ┤1     ├
        └──────┘

    Depending on the number of available auxiliary qubits, this operation can be synthesized
    using different methods. For example, if :math:`n - 1` clean auxiliary qubits are available
    (where :math:`n` is the number of control qubits), a V-chain decomposition can be used whose
    depth is linear in :math:`n`. See also :func:`.synth_mcmt_chain`.
    """

    def __init__(
        self,
        gate: Gate,
        num_ctrl_qubits: int,
        num_target_qubits: int,
        ctrl_state: int | str | None = None,
        label: str | None = None,
    ) -> None:
        """
        Args:
            gate: The base gate to apply on multiple target qubits, controlled by other qubits.
                This must be a single-qubit gate or a controlled single-qubit gate.
            num_ctrl_qubits: The number of control qubits.
            num_target_qubits: The number of target qubits.
            ctrl_state: The control state of the control qubits. Defaults to all closed controls.
            label: The gate label.
        """
        if num_target_qubits < 1:
            raise ValueError("Need at least one target qubit.")

        if num_ctrl_qubits < 1:
            raise ValueError("Need at least one control qubit.")

        self.num_target_qubits = num_target_qubits

        base_gate = self._identify_base_gate(gate)
        num_qubits = num_ctrl_qubits + num_target_qubits

        if label is None:
            label = f"{num_target_qubits}-{gate.name.capitalize()}"

        super().__init__(
            "mcmt",
            base_gate=base_gate,
            num_qubits=num_qubits,
            params=gate.params,
            num_ctrl_qubits=num_ctrl_qubits,
            ctrl_state=ctrl_state,
            label=label,
        )

    def _define(self):
        """Default definition relying on gate.control. Control state is handled by superclass."""

        from qiskit.transpiler.passes.synthesis.hls_plugins import MCMTSynthesisDefault

        self.definition = MCMTSynthesisDefault().run(self)

    @staticmethod
    def _identify_base_gate(gate):
        """Get the control base gate. Note this must be a single qubit gate."""

        # try getting the standard name from the string
        if isinstance(gate, str):
            standard_gates = get_standard_gate_name_mapping()
            if gate in standard_gates:
                gate = standard_gates[gate]
            else:
                raise AttributeError(
                    f"Unknown gate {gate}. Available: {list(get_standard_gate_name_mapping.keys())}"
                )

        # extract the base gate
        if isinstance(gate, ControlledGate):
            warnings.warn(
                "Passing a controlled gate to MCMT is pending deprecation since Qiskit 1.3. Pass a "
                "single-qubit gate instance or the gate name instead, e.g. pass 'h' instead of 'ch'.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            base_gate = gate.base_gate
        elif isinstance(gate, Gate):
            base_gate = gate
        else:
            raise TypeError(f"Invalid gate type {type(gate)}.")

        if base_gate.num_qubits != 1:
            raise ValueError(
                f"MCMTGate requires a base gate with a single qubit, but got {base_gate.num_qubits}."
            )

        return base_gate

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the MCMT gate.

        The controlled gate is implemented as :class:`.MCMTGate`, regardless of the
        value of ``annotated``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state

        gate = MCMTGate(
            self.base_gate,
            self.num_ctrl_qubits + num_ctrl_qubits,
            self.num_target_qubits,
            ctrl_state=new_ctrl_state,
            label=label,
        )

        return gate

    def inverse(self, annotated: bool = False):
        """Return the inverse MCMT circuit."""
        return MCMTGate(
            self.base_gate.inverse(), self.num_ctrl_qubits, self.num_target_qubits, self.ctrl_state
        )
