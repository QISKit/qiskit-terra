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

"""Blueprint circuit object."""

from __future__ import annotations
from abc import ABC, abstractmethod
import copy as _copy

from qiskit._accelerate.circuit import CircuitData
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit.circuit.parametertable import ParameterView
from qiskit.circuit.quantumcircuit import QuantumCircuit, _copy_metadata
from qiskit.utils.deprecation import deprecate_func


class BlueprintCircuit(QuantumCircuit, ABC):
    """Blueprint circuit object.

    In many applications it is necessary to pass around the structure a circuit will have without
    explicitly knowing e.g. its number of qubits, or other missing information. This can be solved
    by having a circuit that knows how to construct itself, once all information is available.

    This class provides an interface for such circuits. Before internal data of the circuit is
    accessed, the ``_build`` method is called. There the configuration of the circuit is checked.
    """

    @deprecate_func(
        since="2.1",
        additional_msg="There is no direct replacement other than the QuantumCircuit class.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, *regs, name: str | None = None) -> None:
        """Create a new blueprint circuit."""
        self._is_initialized = False
        super().__init__(*regs, name=name)
        self._qregs: list[QuantumRegister] = []
        self._cregs: list[ClassicalRegister] = []
        self._is_built = False
        self._is_initialized = True

    @abstractmethod
    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration allows the circuit to be built.

        Args:
            raise_on_failure: If True, raise if the configuration is invalid. If False, return
                False if the configuration is invalid.

        Returns:
            True, if the configuration is valid. Otherwise, depending on the value of
            ``raise_on_failure`` an error is raised or False is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def _build(self) -> None:
        """Build the circuit."""
        if self._is_built:
            return

        # check whether the configuration is valid
        self._check_configuration()
        self._is_built = True

    def _invalidate(self) -> None:
        """Invalidate the current circuit build."""
        # Take out the registers before invalidating
        qregs = self._data.qregs
        cregs = self._data.cregs
        self._data = CircuitData(self._data.qubits, self._data.clbits)
        # Re-add the registers
        for qreg in qregs:
            self._data.add_qreg(qreg)
        for creg in cregs:
            self._data.add_creg(creg)
        self.global_phase = 0
        self._is_built = False

    @property
    def qregs(self):
        """A list of the quantum registers associated with the circuit."""
        if not self._is_initialized:
            return self._qregs
        return super().qregs

    @qregs.setter
    def qregs(self, qregs):
        """Set the quantum registers associated with the circuit."""
        if not self._is_initialized:
            # Workaround to ignore calls from QuantumCircuit.__init__() which
            # doesn't expect 'qregs' to be an overridden property!
            return
        self._qregs = []
        self._ancillas = []
        self._data = CircuitData(clbits=self._data.clbits)
        self.global_phase = 0
        self._is_built = False

        self.add_register(*qregs)

    @property
    def data(self):
        """The circuit data (instructions and context).

        Returns:
            QuantumCircuitData: a list-like object containing the :class:`.CircuitInstruction`\\ s
            for each instruction.
        """
        if not self._is_built:
            self._build()
        return super().data

    def decompose(self, gates_to_decompose=None, reps=1):
        if not self._is_built:
            self._build()
        return super().decompose(gates_to_decompose, reps)

    def draw(self, *args, **kwargs):
        if not self._is_built:
            self._build()
        return super().draw(*args, **kwargs)

    @property
    def num_parameters(self) -> int:
        """The number of parameter objects in the circuit."""
        if not self._is_built:
            self._build()
        return super().num_parameters

    @property
    def parameters(self) -> ParameterView:
        """The parameters defined in the circuit.

        This attribute returns the :class:`.Parameter` objects in the circuit sorted
        alphabetically. Note that parameters instantiated with a :class:`.ParameterVector`
        are still sorted numerically.

        Examples:

            The snippet below shows that insertion order of parameters does not matter.

            .. code-block:: python

                >>> from qiskit.circuit import QuantumCircuit, Parameter
                >>> a, b, elephant = Parameter("a"), Parameter("b"), Parameter("elephant")
                >>> circuit = QuantumCircuit(1)
                >>> circuit.rx(b, 0)
                >>> circuit.rz(elephant, 0)
                >>> circuit.ry(a, 0)
                >>> circuit.parameters  # sorted alphabetically!
                ParameterView([Parameter(a), Parameter(b), Parameter(elephant)])

            Bear in mind that alphabetical sorting might be unintuitive when it comes to numbers.
            The literal "10" comes before "2" in strict alphabetical sorting.

            .. code-block:: python

                >>> from qiskit.circuit import QuantumCircuit, Parameter
                >>> angles = [Parameter("angle_1"), Parameter("angle_2"), Parameter("angle_10")]
                >>> circuit = QuantumCircuit(1)
                >>> circuit.u(*angles, 0)
                >>> circuit.draw()
                   ┌─────────────────────────────┐
                q: ┤ U(angle_1,angle_2,angle_10) ├
                   └─────────────────────────────┘
                >>> circuit.parameters
                ParameterView([Parameter(angle_1), Parameter(angle_10), Parameter(angle_2)])

            To respect numerical sorting, a :class:`.ParameterVector` can be used.

            .. code-block:: python

                >>> from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
                >>> x = ParameterVector("x", 12)
                >>> circuit = QuantumCircuit(1)
                >>> for x_i in x:
                ...     circuit.rx(x_i, 0)
                >>> circuit.parameters
                ParameterView([
                    ParameterVectorElement(x[0]), ParameterVectorElement(x[1]),
                    ParameterVectorElement(x[2]), ParameterVectorElement(x[3]),
                    ..., ParameterVectorElement(x[11])
                ])


        Returns:
            The sorted :class:`.Parameter` objects in the circuit.
        """
        if not self._is_built:
            self._build()
        return super().parameters

    def _append(self, instruction, _qargs=None, _cargs=None, *, _standard_gate=False):
        if not self._is_built:
            self._build()
        return super()._append(instruction, _qargs, _cargs, _standard_gate=_standard_gate)

    def compose(
        self,
        other,
        qubits=None,
        clbits=None,
        front=False,
        inplace=False,
        wrap=False,
        *,
        copy=True,
        var_remap=None,
        inline_captures=False,
    ):
        if not self._is_built:
            self._build()
        return super().compose(
            other,
            qubits,
            clbits,
            front,
            inplace,
            wrap,
            copy=copy,
            var_remap=var_remap,
            inline_captures=inline_captures,
        )

    def inverse(self, annotated: bool = False):
        if not self._is_built:
            self._build()
        return super().inverse(annotated=annotated)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def size(self, *args, **kwargs):
        if not self._is_built:
            self._build()
        return super().size(*args, **kwargs)

    def to_instruction(self, parameter_map=None, label=None):
        if not self._is_built:
            self._build()
        return super().to_instruction(parameter_map, label=label)

    def to_gate(self, parameter_map=None, label=None):
        if not self._is_built:
            self._build()
        return super().to_gate(parameter_map, label=label)

    def depth(self, *args, **kwargs):
        if not self._is_built:
            self._build()
        return super().depth(*args, **kwargs)

    def count_ops(self):
        if not self._is_built:
            self._build()
        return super().count_ops()

    def num_nonlocal_gates(self):
        if not self._is_built:
            self._build()
        return super().num_nonlocal_gates()

    def num_connected_components(self, unitary_only=False):
        if not self._is_built:
            self._build()
        return super().num_connected_components(unitary_only=unitary_only)

    def copy_empty_like(
        self, name: str | None = None, *, vars_mode: str = "alike"
    ) -> QuantumCircuit:
        """Return an empty :class:`.QuantumCircuit` of same size and metadata.

        See also :meth:`.QuantumCircuit.copy_empty_like` for more details on copied metadata.

        Args:
            name: Name for the copied circuit. If None, then the name stays the same.
            vars_mode: The mode to handle realtime variables in.

        Returns:
            An empty circuit of same dimensions. Note that the result is no longer a
            :class:`.BlueprintCircuit`.
        """

        cpy = QuantumCircuit(*self.qregs, *self.cregs, name=name, global_phase=self.global_phase)
        _copy_metadata(self, cpy)
        cpy._data = self._data.copy_empty_like(vars_mode=vars_mode)
        return cpy

    def copy(self, name: str | None = None) -> BlueprintCircuit:
        """Copy the blueprint circuit.

        Args:
            name: Name to be given to the copied circuit. If None, then the name stays the same.

        Returns:
            A deepcopy of the current blueprint circuit, with the specified name.
        """
        if not self._is_built:
            self._build()

        cpy = _copy.copy(self)
        _copy_metadata(self, cpy)

        cpy._is_built = self._is_built
        cpy._data = self._data.copy()

        if name is not None:
            cpy.name = name

        return cpy
