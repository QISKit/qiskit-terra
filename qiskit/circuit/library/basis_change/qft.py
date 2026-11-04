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

"""Define a Quantum Fourier Transform circuit (QFT) and a native gate (QFTGate)."""

from __future__ import annotations
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumRegister, CircuitInstruction, Gate
from qiskit.utils.deprecation import deprecate_func
from ..blueprintcircuit import BlueprintCircuit


class QFT(BlueprintCircuit):
    r"""Quantum Fourier Transform Circuit.

    The Quantum Fourier Transform (QFT) on :math:`n` qubits is the operation

    .. math::

        |j\rangle \mapsto \frac{1}{2^{n/2}} \sum_{k=0}^{2^n - 1} e^{2\pi ijk / 2^n} |k\rangle

    The circuit that implements this transformation can be implemented using Hadamard gates
    on each qubit, a series of controlled-U1 (or Z, depending on the phase) gates and a
    layer of Swap gates. The layer of Swap gates can in principle be dropped if the QFT appears
    at the end of the circuit, since then the re-ordering can be done classically. They
    can be turned off using the ``do_swaps`` attribute.

    For 4 qubits, the circuit that implements this transformation is:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import QFT
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = QFT(4)
       _generate_circuit_library_visualization(circuit)

    The inverse QFT can be obtained by calling the ``inverse`` method on this class.
    The respective circuit diagram is:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import QFT
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = QFT(4).inverse()
       _generate_circuit_library_visualization(circuit)

    One method to reduce circuit depth is to implement the QFT approximately by ignoring
    controlled-phase rotations where the angle is beneath a threshold. This is discussed
    in more detail in https://arxiv.org/abs/quant-ph/9601018 or
    https://arxiv.org/abs/quant-ph/0403071.

    Here, this can be adjusted using the ``approximation_degree`` attribute: the smallest
    ``approximation_degree`` rotation angles are dropped from the QFT. For instance, a QFT
    on 5 qubits with approximation degree 2 yields (the barriers are dropped in this example):

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import QFT
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = QFT(5, approximation_degree=2)
       _generate_circuit_library_visualization(circuit)

    """

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "Use qiskit.circuit.library.QFTGate or qiskit.synthesis.qft.synth_qft_full instead, "
            "for access to all previous arguments.",
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_qubits: int | None = None,
        approximation_degree: int = 0,
        do_swaps: bool = True,
        inverse: bool = False,
        insert_barriers: bool = False,
        name: str | None = None,
    ) -> None:
        """
        Args:
            num_qubits: The number of qubits on which the QFT acts.
            approximation_degree: The degree of approximation (0 for no approximation).
            do_swaps: Whether to include the final swaps in the QFT.
            inverse: If True, the inverse Fourier transform is constructed.
            insert_barriers: If True, barriers are inserted as visualization improvement.
            name: The name of the circuit.

        """
        if name is None:
            name = "IQFT" if inverse else "QFT"

        super().__init__(name=name)
        self._approximation_degree = approximation_degree
        self._do_swaps = do_swaps
        self._insert_barriers = insert_barriers
        self._inverse = inverse
        self.num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """The number of qubits in the QFT circuit.

        Returns:
            The number of qubits in the circuit.
        """
        # This method needs to be overwritten to allow adding the setter for num_qubits while still
        # complying to pylint.
        return super().num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits.

        Note that this changes the registers of the circuit.

        Args:
            num_qubits: The new number of qubits.
        """
        if num_qubits != self.num_qubits:
            self._invalidate()

            self.qregs = []
            if num_qubits is not None and num_qubits > 0:
                self.qregs = [QuantumRegister(num_qubits, name="q")]

    @property
    def approximation_degree(self) -> int:
        """The approximation degree of the QFT.

        Returns:
            The currently set approximation degree.
        """
        return self._approximation_degree

    @approximation_degree.setter
    def approximation_degree(self, approximation_degree: int) -> None:
        """Set the approximation degree of the QFT.

        Args:
            approximation_degree: The new approximation degree.

        Raises:
            ValueError: If the approximation degree is smaller than 0.
        """
        if approximation_degree < 0:
            raise ValueError("Approximation degree cannot be smaller than 0.")

        if approximation_degree != self._approximation_degree:
            self._invalidate()
            self._approximation_degree = approximation_degree

    @property
    def insert_barriers(self) -> bool:
        """Whether barriers are inserted for better visualization or not.

        Returns:
            True, if barriers are inserted, False if not.
        """
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        """Specify whether barriers are inserted for better visualization or not.

        Args:
            insert_barriers: If True, barriers are inserted, if False not.
        """
        if insert_barriers != self._insert_barriers:
            self._invalidate()
            self._insert_barriers = insert_barriers

    @property
    def do_swaps(self) -> bool:
        """Whether the final swaps of the QFT are applied or not.

        Returns:
            True, if the final swaps are applied, False if not.
        """
        return self._do_swaps

    @do_swaps.setter
    def do_swaps(self, do_swaps: bool) -> None:
        """Specify whether to do the final swaps of the QFT circuit or not.

        Args:
            do_swaps: If True, the final swaps are applied, if False not.
        """
        if do_swaps != self._do_swaps:
            self._invalidate()
            self._do_swaps = do_swaps

    def is_inverse(self) -> bool:
        """Whether the inverse Fourier transform is implemented.

        Returns:
            True, if the inverse Fourier transform is implemented, False otherwise.
        """
        return self._inverse

    def inverse(self, annotated: bool = False) -> QFT:
        """Invert this circuit.

        Args:
            annotated: indicates whether the inverse gate can be implemented
                as an annotated gate. The value of this argument is ignored as the
                inverse of a QFT is an IQFT which is just another instance of
                :class:`.QFT`.

        Returns:
            The inverted circuit.
        """

        if self.name in ("QFT", "IQFT"):
            name = "QFT" if self._inverse else "IQFT"
        else:
            name = self.name + "_dg"

        inverted = self.copy(name=name)

        # data consists of the QFT gate only
        iqft = self.data[0].operation.inverse()
        iqft.name = name

        inverted.data.clear()
        inverted._append(CircuitInstruction(iqft, inverted.qubits, []))

        inverted._inverse = not self._inverse
        return inverted

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits has not been set.")
        return valid

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        num_qubits = self.num_qubits

        if num_qubits == 0:
            return

        from qiskit.synthesis.qft import synth_qft_full

        circuit = synth_qft_full(
            num_qubits,
            do_swaps=self._do_swaps,
            insert_barriers=self._insert_barriers,
            approximation_degree=self._approximation_degree,
            inverse=self._inverse,
            name=self.name,
        )

        wrapped = circuit.to_instruction() if self.insert_barriers else circuit.to_gate()
        self.compose(wrapped, qubits=self.qubits, inplace=True)


class QFTGate(Gate):
    r"""Quantum Fourier Transform Gate.

    The Quantum Fourier Transform (QFT) on :math:`n` qubits is the operation

    .. math::

        |j\rangle \mapsto \frac{1}{2^{n/2}} \sum_{k=0}^{2^n - 1} e^{2\pi ijk / 2^n} |k\rangle

    """

    def __init__(
        self,
        num_qubits: int,
    ):
        """
        Args:
            num_qubits: The number of qubits on which the QFT acts.
        """
        super().__init__(name="qft", num_qubits=num_qubits, params=[])

    def __array__(self, dtype=complex, copy=None):
        """Return a numpy array for the QFTGate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        n = self.num_qubits
        nums = np.arange(2**n)
        outer = np.outer(nums, nums)
        return np.exp(2j * np.pi * outer * (0.5**n), dtype=dtype) * (0.5 ** (n / 2))

    def _define(self):
        """Provide a specific decomposition of the QFTGate into a quantum circuit."""
        from qiskit.synthesis.qft import synth_qft_full

        self.definition = synth_qft_full(num_qubits=self.num_qubits)
