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

"""Controlled unitary gate."""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING
from qiskit.circuit.exceptions import CircuitError


from . import QuantumRegister
from .quantumcircuit import QuantumCircuit
from .gate import Gate
from ._utils import _ctrl_state_to_int


if TYPE_CHECKING:
    from qiskit.circuit.annotated_operation import AnnotatedOperation


class ControlledGate(Gate):
    """Controlled unitary gate."""

    def __init__(
        self,
        name: str,
        num_qubits: int,
        params: list,
        label: str | None = None,
        num_ctrl_qubits: int | None = 1,
        definition: QuantumCircuit | None = None,
        ctrl_state: int | str | None = None,
        base_gate: Gate | None = None,
        *,
        _base_label=None,
    ):
        """Create a new ControlledGate. In the new gate the first ``num_ctrl_qubits``
        of the gate are the controls.

        Args:
            name: The name of the gate.
            num_qubits: The number of qubits the gate acts on.
            params: A list of parameters for the gate.
            label: An optional label for the gate.
            num_ctrl_qubits: Number of control qubits.
            definition: A list of gate rules for implementing this gate. The
                elements of the list are tuples of (:meth:`~qiskit.circuit.Gate`, [qubit_list],
                [clbit_list]).
            ctrl_state: The control state in decimal or as
                a bitstring (e.g. '111'). If specified as a bitstring the length
                must equal num_ctrl_qubits, MSB on left. If None, use
                2**num_ctrl_qubits-1.
            base_gate: Gate object to be controlled.

        Raises:
            CircuitError: If ``num_ctrl_qubits`` >= ``num_qubits``.
            CircuitError: ctrl_state < 0 or ctrl_state > 2**num_ctrl_qubits.

        Examples:

        Create a controlled standard gate and apply it to a circuit.

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit import QuantumCircuit, QuantumRegister
           from qiskit.circuit.library.standard_gates import HGate

           qr = QuantumRegister(3)
           qc = QuantumCircuit(qr)
           c3h_gate = HGate().control(2)
           qc.append(c3h_gate, qr)
           qc.draw('mpl')

        Create a controlled custom gate and apply it to a circuit.

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit import QuantumCircuit, QuantumRegister
           from qiskit.circuit.library.standard_gates import HGate

           qc1 = QuantumCircuit(2)
           qc1.x(0)
           qc1.h(1)
           custom = qc1.to_gate().control(2)

           qc2 = QuantumCircuit(4)
           qc2.append(custom, [0, 3, 1, 2])
           qc2.draw('mpl')
        """
        self.base_gate = None if base_gate is None else base_gate.copy()
        super().__init__(name, num_qubits, params, label=label)
        self._num_ctrl_qubits = 1
        self.num_ctrl_qubits = num_ctrl_qubits
        self.definition = copy.deepcopy(definition)
        self._ctrl_state = None
        self._open_ctrl = None
        self.ctrl_state = ctrl_state
        self._name = name

    @property
    def definition(self) -> QuantumCircuit:
        """Return definition in terms of other basic gates. If the gate has
        open controls, as determined from :attr:`ctrl_state`, the returned
        definition is conjugated with X without changing the internal
        ``_definition``.
        """
        if self._open_ctrl:
            closed_gate = self.to_mutable()
            closed_gate.ctrl_state = None
            bit_ctrl_state = bin(self.ctrl_state)[2:].zfill(self.num_ctrl_qubits)
            qreg = QuantumRegister(self.num_qubits, "q")
            qc_open_ctrl = QuantumCircuit(qreg)
            for qind, val in enumerate(bit_ctrl_state[::-1]):
                if val == "0":
                    qc_open_ctrl.x(qind)
            qc_open_ctrl.append(closed_gate, qargs=qreg[:])
            for qind, val in enumerate(bit_ctrl_state[::-1]):
                if val == "0":
                    qc_open_ctrl.x(qind)
            return qc_open_ctrl
        else:
            return super().definition

    @definition.setter
    def definition(self, excited_def: QuantumCircuit):
        """Set controlled gate definition with closed controls.

        Args:
            excited_def: The circuit with all closed controls.
        """
        self._definition = excited_def

    @property
    def name(self) -> str:
        """Get name of gate. If the gate has open controls the gate name
        will become:

           <original_name_o<ctrl_state>

        where <original_name> is the gate name for the default case of
        closed control qubits and <ctrl_state> is the integer value of
        the control state for the gate.
        """
        if self._open_ctrl:
            return f"{self._name}_o{self.ctrl_state}"
        else:
            return self._name

    @name.setter
    def name(self, name_str: str) -> None:
        """Set the name of the gate. Note the reported name may differ
        from the set name if the gate has open controls.
        """
        self._name = name_str

    @property
    def num_ctrl_qubits(self) -> int:
        """Get number of control qubits.

        Returns:
            int: The number of control qubits for the gate.
        """
        return self._num_ctrl_qubits

    @num_ctrl_qubits.setter
    def num_ctrl_qubits(self, num_ctrl_qubits: int) -> None:
        """Set the number of control qubits.

        Args:
            num_ctrl_qubits (int): The number of control qubits.

        Raises:
            CircuitError: ``num_ctrl_qubits`` is not an integer in ``[0, num_qubits]``.
        """
        if num_ctrl_qubits != int(num_ctrl_qubits):
            raise CircuitError("The number of control qubits must be an integer.")
        num_ctrl_qubits = int(num_ctrl_qubits)
        # This is a range rather than an equality limit because some controlled gates represent a
        # controlled version of the base gate whose definition also uses auxiliary qubits.
        upper_limit = self.num_qubits - getattr(self.base_gate, "num_qubits", 0)
        if num_ctrl_qubits < 0:
            raise CircuitError("The number of control qubits must be non-negative.")
        if num_ctrl_qubits > upper_limit:
            limit = "num_qubits" if self.base_gate is None else "num_qubits - base_gate.num_qubits"
            raise CircuitError(f"The number of control qubits must be in `[0, {limit}]`.")
        self._num_ctrl_qubits = num_ctrl_qubits

    @property
    def ctrl_state(self) -> int:
        """Return the control state of the gate as a decimal integer."""
        return self._ctrl_state

    @ctrl_state.setter
    def ctrl_state(self, ctrl_state: int | str | None):
        """Set the control state of this gate.

        Args:
            ctrl_state: The control state of the gate.

        Raises:
            CircuitError: ctrl_state is invalid.
        """
        self._ctrl_state = _ctrl_state_to_int(ctrl_state, self.num_ctrl_qubits)
        self._open_ctrl = self.ctrl_state < 2**self.num_ctrl_qubits - 1

    @property
    def params(self):
        """Get parameters from base_gate.

        Returns:
            list: List of gate parameters.

        Raises:
            CircuitError: Controlled gate does not define a base gate
        """
        if self.base_gate:
            return self.base_gate.params
        else:
            raise CircuitError("Controlled gate does not define base gate for extracting params")

    @params.setter
    def params(self, parameters):
        """Set base gate parameters.

        Args:
            parameters (list): The list of parameters to set.

        Raises:
            CircuitError: If controlled gate does not define a base gate.
        """
        if self.base_gate:
            if self.base_gate.mutable:
                self.base_gate.params = parameters
            elif parameters:
                raise CircuitError("cannot set parameters on immutable base gate")
        else:
            raise CircuitError("Controlled gate does not define base gate for extracting params")

    def __deepcopy__(self, memo=None):
        cpy = copy.copy(self)
        cpy.base_gate = self.base_gate.copy()
        if self._definition:
            cpy._definition = copy.deepcopy(self._definition, memo)
        return cpy

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, ControlledGate)
            and self.num_ctrl_qubits == other.num_ctrl_qubits
            and self.ctrl_state == other.ctrl_state
            and self.base_gate == other.base_gate
            and self.num_qubits == other.num_qubits
            and self.num_clbits == other.num_clbits
            and self.definition == other.definition
        )

    def inverse(self, annotated: bool = False) -> ControlledGate | AnnotatedOperation:
        """Invert this gate by calling inverse on the base gate."""
        if not annotated:
            inverse_gate = self.base_gate.inverse().control(
                self.num_ctrl_qubits, ctrl_state=self.ctrl_state, annotated=annotated
            )
        else:
            inverse_gate = super().inverse(annotated=annotated)
        return inverse_gate
