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

"""Synthesize a single qubit gate to a discrete basis set."""

from __future__ import annotations

import typing
import warnings
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import get_standard_gate_name_mapping, IGate
from qiskit.utils.deprecation import deprecate_func
from qiskit._accelerate.synthesis.discrete_basis import (
    SolovayKitaevSynthesis as RustSolovayKitaevSynthesis,
    GateSequence,
)

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit


class SolovayKitaevDecomposition:
    """The Solovay Kitaev discrete decomposition algorithm.

    This class is called recursively by the transpiler pass, which is why it is separated.
    See :class:`~qiskit.transpiler.passes.SolovayKitaev` for more information.
    """

    def __init__(
        self,
        basic_approximations: str | dict[str, np.ndarray] | list[GateSequence] | None = None,
        *,
        basis_gates: list[str | Gate] | None = None,
        depth: int = 12,
        check_input: bool = False,
    ) -> None:
        """

        .. note::

            If ``basic_approximations`` is passed as ``.npy`` file, pickle is used internally
            to load the data. This is a potential security vulnerability and only trusted files
            should be loaded.

        Args:
            basic_approximations: A specification of the basic SO(3) approximations in terms
                of discrete gates. At each iteration of this algorithm, the remaining error is
                approximated with the closest sequence of gates in this set.
                If a ``str``, this specifies a filename from which to load the
                approximation. If a ``dict``, then this contains
                ``{gates: effective_SO3_matrix}`` pairs,
                e.g. ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``.
                If a list, this contains the same information as the dict, but already converted to
                :class:`.GateSequence` objects, which contain the SO(3) matrix and gates.

                Either this parameter, or ``basis_gates`` and ``depth`` can be specified.
            basis_gates: A list of discrete (i.e., non-parameterized) standard gates.
                Defaults to ``["h", "t", "tdg"]``.
            depth: The number of basis gate combinations to consider in the basis set. This
                determines how fast (and if) the algorithm converges and should be chosen
                sufficiently high.
            check_input: If ``True``, perform intermediate steps checking whether the matrices
                are of expected form.
        """
        if basic_approximations is None:
            if basis_gates is not None:
                basis_gates = normalize_gates(basis_gates)
            self._sk = RustSolovayKitaevSynthesis(basis_gates, depth, None, check_input)

        elif basis_gates is not None:
            raise ValueError(
                "Either basic_approximations or basis_gates + depth can be specified, not both."
            )

        # Fast Rust path to load the file
        elif isinstance(basic_approximations, str) and basic_approximations[~3:] != ".npy":
            self._sk = RustSolovayKitaevSynthesis.from_basic_approximations(
                basic_approximations, True
            )
        else:
            sequences = self.load_basic_approximations(basic_approximations)
            self._sk = RustSolovayKitaevSynthesis.from_sequences(sequences, True)

        self._depth = depth
        self._check_input = check_input
        self._basis_gates = basis_gates

    @property
    def depth(self) -> int:
        """The maximum gate depth of the basic approximations."""
        return self._depth

    @property
    def check_input(self) -> bool:
        """Whether to perform runtime checks on the internal data."""
        return self._check_input

    @property
    def basis_gates(self) -> list[str] | None:
        """The basis gate set of the basic approximations.

        If ``None``, defaults to ``["h", "t", "tdg"]``.
        """
        return self._basis_gates

    @staticmethod
    def load_basic_approximations(data: list | str | dict) -> list[GateSequence]:
        """Load basic approximations.

        .. note::

            If ``data`` is given as string, this method internally relies on pickle to load
            the file. This is a potential security vulnerability and only trusted files should be
            loaded.

        Args:
            data: If a string, specifies the path to the file from where to load the data.
                If a dictionary, directly specifies the decompositions as ``{gates: matrix}``
                or ``{gates: (matrix, global_phase)}``. There, ``gates`` are the names of the gates
                producing the SO(3) matrix ``matrix``, e.g.
                ``{"h t": np.array([[0, 0.7071, -0.7071], [0, -0.7071, -0.7071], [-1, 0, 0]]}``
                and the ``global_phase`` can be given to account for a global phase difference
                between the U(2) matrix of the quantum gates and the stored SO(3) matrix.
                If not given, the ``global_phase`` will be assumed to be 0.

        Returns:
            A list of basic approximations as type ``GateSequence``.

        Raises:
            ValueError: If the number of gate combinations and associated matrices does not match.
        """
        # new data format stored by the Rust internal class
        if isinstance(data, str) and data[-4:] != ".npy":
            sk = SolovayKitaevDecomposition(data)
            return sk._sk.get_gate_sequences()

        warnings.warn(
            "It is suggested to pass basic_approximations in the binary format produced "
            "by SolovayKitaevDecomposition.save_basic_approximations, which is more "
            "performant than other formats. Passing a .npy format is deprecated since Qiskit 2.3 "
            "and support will be removed no sooner than 3 months after the release date.",
            category=DeprecationWarning,
        )

        # is already a list of GateSequences
        if isinstance(data, list):
            return data

        # file is ``.npy``, load the dictionary it contains
        if isinstance(data, str):
            data = np.load(data, allow_pickle=True).item()

        # parse the dictionary
        sequences = []
        for gatestring, matrix_and_phase in data.items():
            if isinstance(matrix_and_phase, tuple):
                matrix, global_phase = matrix_and_phase
            else:
                matrix, global_phase = matrix_and_phase, 0

            gates = normalize_gates(gatestring.split())
            sequence = GateSequence.from_gates_and_matrix(gates, matrix, global_phase)
            sequences.append(sequence)

        return sequences

    def save_basic_approximations(self, filename: str):
        """Save the basic approximations into a file.

        This can then be loaded again via the class initializer (preferred) or
        via :meth:`load_basic_approximations`::

            filename = "approximations.bin"
            sk.save_basic_approximations(filename)

            new_sk = SolovayKitaevDecomposition(filename)

        Args:
            filename: The filename to store the approximations in.

        Raises:
            ValueError: If the filename has a `.npy` extension. The format is not `.npy`,
                and storing as such can cause errors when loading the file again.
        """
        # Safety guard: previously, we serialized via npy, but this format is incompatible
        # with the current serialization, using Rust's binrw. While we can still load
        # .npy files in legacy format, the new format should not be stored as .npy.
        if filename[~3:] == ".npy":
            raise ValueError(
                "The basic approximations are not stored in npy format. "
                "Choose a different file extension (e.g. .bin)."
            )
        self._sk.save_basic_approximations(filename)

    def run(
        self,
        gate_matrix: np.ndarray | Gate,
        recursion_degree: int,
        return_dag: bool = False,
        check_input: bool = True,
    ) -> QuantumCircuit | DAGCircuit:
        r"""Run the algorithm.

        Args:
            gate_matrix: The single-qubit gate to approximate. Can either be a :class:`.Gate`, where
                :meth:`.Gate.to_matrix` returns the matrix, or a :math:`2\times 2` unitary matrix
                representing the gate.
            recursion_degree: The recursion degree, called :math:`n` in the paper.
            return_dag: If ``True`` return a :class:`.DAGCircuit`, else a :class:`.QuantumCircuit`.
            check_input: If ``True`` check that the input matrix is valid for the decomposition.
                Overrides the class attribute with the same name, but only for this function call.

        Returns:
            A one-qubit circuit approximating the ``gate_matrix`` in the specified discrete basis.
        """
        # handle overriding the check_input setting
        self_check_input = self.check_input
        if check_input != self_check_input:
            self._sk.do_checks = check_input

        if isinstance(gate_matrix, Gate):
            data = self._sk.synthesize(gate_matrix, recursion_degree)
        else:
            data = self._sk.synthesize_matrix(gate_matrix, recursion_degree)

        if check_input != self_check_input:
            self._sk.do_checks = self_check_input

        circuit = QuantumCircuit._from_circuit_data(data, legacy_qubits=True)

        if return_dag:
            return circuit.to_dag()
        return circuit

    def query_basic_approximation(self, gate: np.ndarray | Gate) -> QuantumCircuit:
        """Query a basic approximation of a matrix."""
        if isinstance(gate, Gate):
            data = self._sk.query_basic_approximation(gate)
        else:
            data = self._sk.query_basic_approximation_matrix(gate)

        circuit = QuantumCircuit._from_circuit_data(data, legacy_qubits=True)
        return circuit

    @deprecate_func(
        since="2.3",
        additional_msg="Use query_basic_approximation instead, which takes a Gate or matrix "
        "as input and returns a QuantumCircuit object.",
    )
    def find_basic_approximation(self, sequence: GateSequence) -> GateSequence:
        """Find ``GateSequence`` in ``self._basic_approximations`` that approximates ``sequence``.

        Args:
            sequence: ``GateSequence`` to find the approximation to.

        Returns:
            ``GateSequence`` in that approximates ``sequence``.
        """
        return self._sk.find_basic_approximation(sequence)


def normalize_gates(gates: list[Gate | str]) -> list[Gate]:
    """Normalize a list[Gate | str] into list[Gate]."""
    name_to_gate = get_standard_gate_name_mapping()
    # special case: we used to support "i" as IGate, but the official name is "id", so
    # we add it manually here
    name_to_gate["i"] = IGate()

    def normalize(gate: Gate | str) -> Gate:
        if isinstance(gate, Gate):
            return gate
        if gate in name_to_gate:
            return name_to_gate[gate]
        raise ValueError(f"Unsupported gate: {gate}")

    return list(map(normalize, gates))
