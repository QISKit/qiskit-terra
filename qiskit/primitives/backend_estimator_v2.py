# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Estimator V2 implementation for an arbitrary Backend object."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from qiskit.circuit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.providers import BackendV2
from qiskit.quantum_info import Pauli, PauliList
from qiskit.result import Counts, Result
from qiskit.transpiler import PassManager, PassManagerConfig
from qiskit.transpiler.passes import Optimize1qGatesDecomposition

from .base import BaseEstimatorV2
from .containers import DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from .containers.bindings_array import BindingsArray
from .containers.estimator_pub import EstimatorPub
from .primitive_job import PrimitiveJob


def _run_circuits(
    circuits: QuantumCircuit | list[QuantumCircuit],
    backend: BackendV2,
    clear_metadata: bool = True,
    **run_options,
) -> tuple[list[Result], list[dict]]:
    """Remove metadata of circuits and run the circuits on a backend.
    Args:
        circuits: The circuits
        backend: The backend
        clear_metadata: Clear circuit metadata before passing to backend.run if
            True.
        **run_options: run_options
    Returns:
        The result and the metadata of the circuits
    """
    if isinstance(circuits, QuantumCircuit):
        circuits = [circuits]
    metadata = []
    for circ in circuits:
        metadata.append(circ.metadata)
        if clear_metadata:
            circ.metadata = {}
    if isinstance(backend, BackendV2):
        max_circuits = backend.max_circuits
    else:
        raise RuntimeError("Backend version not supported")
    if max_circuits:
        jobs = [
            backend.run(circuits[pos : pos + max_circuits], **run_options)
            for pos in range(0, len(circuits), max_circuits)
        ]
        result = [x.result() for x in jobs]
    else:
        result = [backend.run(circuits, **run_options).result()]
    return result, metadata


def _prepare_counts(results: list[Result]):
    counts = []
    for res in results:
        count = res.get_counts()
        if not isinstance(count, list):
            count = [count]
        counts.extend(count)
    return counts


def _pauli_expval_with_variance(counts: Counts, paulis: PauliList) -> tuple[np.ndarray, np.ndarray]:
    """Return array of expval and variance pairs for input Paulis.
    Note: All non-identity Paulis are treated as Z-Paulis, assuming
    that basis rotations have been applied to convert them to the
    diagonal basis.
    """
    # Diag indices
    size = len(paulis)
    diag_inds = _paulis2inds(paulis)

    expvals = np.zeros(size, dtype=float)
    denom = 0  # Total shots for counts dict
    for bin_outcome, freq in counts.items():
        split_outcome = bin_outcome.split(" ", 1)[0] if " " in bin_outcome else bin_outcome
        outcome = int(split_outcome, 2)
        denom += freq
        for k in range(size):
            coeff = (-1) ** _parity(diag_inds[k] & outcome)
            expvals[k] += freq * coeff

    # Divide by total shots
    expvals /= denom

    # Compute variance
    variances = 1 - expvals**2
    return expvals, variances


def _paulis2inds(paulis: PauliList) -> list[int]:
    """Convert PauliList to diagonal integers.
    These are integer representations of the binary string with a
    1 where there are Paulis, and 0 where there are identities.
    """
    # Treat Z, X, Y the same
    nonid = paulis.z | paulis.x

    # bits are packed into uint8 in little endian
    # e.g., i-th bit corresponds to coefficient 2^i
    packed_vals = np.packbits(nonid, axis=1, bitorder="little")
    power_uint8 = 1 << (8 * np.arange(packed_vals.shape[1], dtype=object))
    inds = packed_vals @ power_uint8
    return inds.tolist()


def _parity(integer: int) -> int:
    """Return the parity of an integer"""
    return bin(integer).count("1") % 2


@dataclass
class Options:
    """Options for :class:`~.BackendEstimatorV2`."""

    default_precision: float = 0.015625
    """The default precision to use if none are specified in :meth:`~run`.
    Default: 0.015625 (1 / sqrt(4096)).
    """

    abelian_grouping: bool = True
    """Whether the observables should be grouped into sets of qubit-wise commuting observables.
    Default: True.
    """

    seed_simulator: int | None = None
    """The seed to use in the simulator. If None, a random seed will be used.
    Default: None.
    """


@dataclass
class _PreprocessedData:
    """Internal data structure to store the results of the preprocessing of a pub."""

    circuits: list[QuantumCircuit]
    """The quantum circuits generated by binding parameters of the pub's circuit."""

    parameter_indices: np.ndarray
    """The indices of the pub's bindings array broadcast to the shape of the pub."""

    observables: np.ndarray
    """The pub's observable array broadcast to the shape of the pub."""


class BackendEstimatorV2(BaseEstimatorV2):
    r"""Evaluates expectation values for provided quantum circuit and observable combinations.

    The :class:`~.BackendEstimatorV2` class is a generic implementation of the
    :class:`~.BaseEstimatorV2` interface that is used to wrap a :class:`~.BackendV2`
    object in the :class:`~.BaseEstimatorV2` API. It
    facilitates using backends that do not provide a native
    :class:`~.BaseEstimatorV2` implementation in places that work with
    :class:`~.BaseEstimatorV2`. However,
    if you're using a provider that has a native implementation of
    :class:`~.BaseEstimatorV2`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.

    This class does not perform any measurement or gate mitigation, and, presently, is only
    compatible with Pauli-based observables. More formally, given an observable of the type
    :math:`O=\sum_{i=1}^Na_iP_i`, where :math:`a_i` is a complex number and :math:`P_i` is a
    Pauli operator, the estimator calculates the expectation :math:`\mathbb{E}(P_i)` of each
    :math:`P_i` and finally calculates the expectation value of :math:`O` as
    :math:`\mathbb{E}(O)=\sum_{i=1}^Na_i\mathbb{E}(P_i)`. The reported ``std`` is calculated
    as

    .. math::

        \frac{\sum_{i=1}^{n}|a_i|\sqrt{\textrm{Var}\big(P_i\big)}}{\sqrt{N}}\:,

    where :math:`\textrm{Var}(P_i)` is the variance of :math:`P_i`, :math:`N=O(\epsilon^{-2})` is
    the number of shots, and :math:`\epsilon` is the target precision [1].

    Each tuple of ``(circuit, observables, <optional> parameter values, <optional> precision)``,
    called an estimator primitive unified bloc (PUB), produces its own array-based result. The
    :meth:`~.BackendEstimatorV2.run` method can be given a sequence of pubs to run in one call.

    The options for :class:`~.BackendEstimatorV2` consist of the following items.

    * ``default_precision``: The default precision to use if none are specified in :meth:`~run`.
      Default: 0.015625 (1 / sqrt(4096)).

    * ``abelian_grouping``: Whether the observables should be grouped into sets of qubit-wise
      commuting observables.
      Default: True.

    * ``seed_simulator``: The seed to use in the simulator. If None, a random seed will be used.
      Default: None.

    **Reference:**

    [1] O. Crawford, B. van Straaten, D. Wang, T. Parks, E. Campbell, St. Brierley,
    Efficient quantum measurement of Pauli operators in the presence of finite sampling error.
    `Quantum 5, 385 <https://doi.org/10.22331/q-2021-01-20-385>`_
    """

    def __init__(
        self,
        *,
        backend: BackendV2,
        options: dict | None = None,
    ):
        """
        Args:
            backend: The backend to run the primitive on.
            options: The options to control the default precision (``default_precision``),
                the operator grouping (``abelian_grouping``), and
                the random seed for the simulator (``seed_simulator``).
        """
        self._backend = backend
        self._options = Options(**options) if options else Options()

        basis = PassManagerConfig.from_backend(backend).basis_gates
        if isinstance(backend, BackendV2):
            opt1q = Optimize1qGatesDecomposition(basis=basis, target=backend.target)
        else:
            opt1q = Optimize1qGatesDecomposition(basis=basis)
        self._passmanager = PassManager([opt1q])

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    @property
    def backend(self) -> BackendV2:
        """Returns the backend which this estimator object is based on."""
        return self._backend

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._options.default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[EstimatorPub]):
        for i, pub in enumerate(pubs):
            if pub.precision <= 0.0:
                raise ValueError(
                    f"The {i}-th pub has precision less than or equal to 0 ({pub.precision}). ",
                    "But precision should be larger than 0.",
                )

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            shots = math.ceil(1.0 / pub.precision**2)
            pub_dict[shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs([pubs[i] for i in lst], shots)
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pubs(self, pubs: list[EstimatorPub], shots: int) -> list[PubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        preprocessed_data = []
        flat_circuits = []
        for pub in pubs:
            data = self._preprocess_pub(pub)
            preprocessed_data.append(data)
            flat_circuits.extend(data.circuits)

        run_result, metadata = _run_circuits(
            flat_circuits, self._backend, shots=shots, seed_simulator=self._options.seed_simulator
        )
        counts = _prepare_counts(run_result)

        results = []
        start = 0
        for pub, data in zip(pubs, preprocessed_data):
            end = start + len(data.circuits)
            expval_map = self._calc_expval_map(counts[start:end], metadata[start:end])
            start = end
            results.append(self._postprocess_pub(pub, expval_map, data, shots))
        return results

    def _preprocess_pub(self, pub: EstimatorPub) -> _PreprocessedData:
        """Converts a pub into a list of bound circuits necessary to estimate all its observables.

        The circuits contain metadata explaining which bindings array index they are with respect to,
        and which measurement basis they are measuring.

        Args:
            pub: The pub to preprocess.

        Returns:
            The values ``(circuits, bc_param_ind, bc_obs)`` where ``circuits`` are the circuits to
            execute on the backend, ``bc_param_ind`` are indices of the pub's bindings array and
            ``bc_obs`` is the observables array, both broadcast to the shape of the pub.
        """
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values

        # calculate broadcasting of parameters and observables
        param_shape = parameter_values.shape
        param_indices = np.fromiter(np.ndindex(param_shape), dtype=object).reshape(param_shape)
        bc_param_ind, bc_obs = np.broadcast_arrays(param_indices, observables)

        param_obs_map = defaultdict(set)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            param_obs_map[param_index].update(bc_obs[index])

        bound_circuits = self._bind_and_add_measurements(circuit, parameter_values, param_obs_map)
        return _PreprocessedData(bound_circuits, bc_param_ind, bc_obs)

    def _postprocess_pub(
        self, pub: EstimatorPub, expval_map: dict, data: _PreprocessedData, shots: int
    ) -> PubResult:
        """Computes expectation values (evs) and standard errors (stds).

        The values are stored in arrays broadcast to the shape of the pub.

        Args:
            pub: The pub to postprocess.
            expval_map: The map
            data: The result data of the preprocessing.
            shots: The number of shots.

        Returns:
            The pub result.
        """
        bc_param_ind = data.parameter_indices
        bc_obs = data.observables
        evs = np.zeros_like(bc_param_ind, dtype=float)
        variances = np.zeros_like(bc_param_ind, dtype=float)
        for index in np.ndindex(*bc_param_ind.shape):
            param_index = bc_param_ind[index]
            for pauli, coeff in bc_obs[index].items():
                expval, variance = expval_map[param_index, pauli]
                evs[index] += expval * coeff
                variances[index] += np.abs(coeff) * variance**0.5
        stds = variances / np.sqrt(shots)
        data_bin = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data_bin,
            metadata={
                "target_precision": pub.precision,
                "shots": shots,
                "circuit_metadata": pub.circuit.metadata,
            },
        )

    def _bind_and_add_measurements(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray,
        param_obs_map: dict[tuple[int, ...], set[str]],
    ) -> list[QuantumCircuit]:
        """Bind the given circuit against each parameter value set, and add necessary measurements
        to each.

        Args:
            circuit: The (possibly parametric) circuit of interest.
            parameter_values: An array of parameter value sets that can be applied to the circuit.
            param_obs_map: A mapping from locations in ``parameter_values`` to a sets of
                Pauli terms whose expectation values are required in those locations.

        Returns:
            A flat list of circuits sufficient to measure all Pauli terms in the ``param_obs_map``
            values at the corresponding ``parameter_values`` location, where requisite
            book-keeping is stored as circuit metadata.
        """
        circuits = []
        for param_index, pauli_strings in param_obs_map.items():
            bound_circuit = parameter_values.bind(circuit, param_index)
            # sort pauli_strings so that the order is deterministic
            meas_paulis = PauliList(sorted(pauli_strings))
            new_circuits = self._create_measurement_circuits(
                bound_circuit, meas_paulis, param_index
            )
            circuits.extend(new_circuits)
        return circuits

    def _calc_expval_map(
        self,
        counts: list[Counts],
        metadata: dict,
    ) -> dict[tuple[tuple[int, ...], str], tuple[float, float]]:
        """Computes the map of expectation values.

        Args:
            counts: The counts data.
            metadata: The metadata.

        Returns:
            The map of expectation values takes a pair of an index of the bindings array and
            a pauli string as a key and returns the expectation value of the pauli string
            with the pub's circuit bound against the parameter value set in the index of
            the bindings array.
        """
        expval_map: dict[tuple[tuple[int, ...], str], tuple[float, float]] = {}
        for count, meta in zip(counts, metadata):
            orig_paulis = meta["orig_paulis"]
            meas_paulis = meta["meas_paulis"]
            param_index = meta["param_index"]
            expvals, variances = _pauli_expval_with_variance(count, meas_paulis)
            for pauli, expval, variance in zip(orig_paulis, expvals, variances):
                expval_map[param_index, pauli.to_label()] = (expval, variance)
        return expval_map

    def _create_measurement_circuits(
        self, circuit: QuantumCircuit, observable: PauliList, param_index: tuple[int, ...]
    ) -> list[QuantumCircuit]:
        """Generate a list of circuits sufficient to estimate each of the given Paulis.

        Paulis are divided into qubitwise-commuting subsets to reduce the total circuit count.
        Metadata is attached to circuits in order to remember what each one measures, and
        where it belongs in the output.

        Args:
            circuit: The circuit of interest.
            observable: Which Pauli terms we would like to observe.
            param_index: Where to put the data we estimate (only passed to metadata).

        Returns:
            A list of circuits sufficient to estimate each of the given Paulis.
        """
        meas_circuits: list[QuantumCircuit] = []
        if self._options.abelian_grouping:
            for obs in observable.group_commuting(qubit_wise=True):
                basis = Pauli((np.logical_or.reduce(obs.z), np.logical_or.reduce(obs.x)))
                meas_circuit, indices = _measurement_circuit(circuit.num_qubits, basis)
                paulis = PauliList.from_symplectic(
                    obs.z[:, indices],
                    obs.x[:, indices],
                    obs.phase,
                )
                meas_circuit.metadata = {
                    "orig_paulis": obs,
                    "meas_paulis": paulis,
                    "param_index": param_index,
                }
                meas_circuits.append(meas_circuit)
        else:
            for basis in observable:
                meas_circuit, indices = _measurement_circuit(circuit.num_qubits, basis)
                obs = PauliList(basis)
                paulis = PauliList.from_symplectic(
                    obs.z[:, indices],
                    obs.x[:, indices],
                    obs.phase,
                )
                meas_circuit.metadata = {
                    "orig_paulis": obs,
                    "meas_paulis": paulis,
                    "param_index": param_index,
                }
                meas_circuits.append(meas_circuit)

        # unroll basis gates
        meas_circuits = self._passmanager.run(meas_circuits)

        # combine measurement circuits
        preprocessed_circuits = []
        for meas_circuit in meas_circuits:
            circuit_copy = circuit.copy()
            # meas_circuit is supposed to have a classical register whose name is different from
            # those of the transpiled_circuit
            clbits = meas_circuit.cregs[0]
            for creg in circuit_copy.cregs:
                if clbits.name == creg.name:
                    raise QiskitError(
                        "Classical register for measurements conflict with those of the input "
                        f"circuit: {clbits}. "
                        "Recommended to avoid register names starting with '__'."
                    )
            circuit_copy.add_register(clbits)
            circuit_copy.compose(meas_circuit, clbits=clbits, inplace=True)
            circuit_copy.metadata = meas_circuit.metadata
            preprocessed_circuits.append(circuit_copy)
        return preprocessed_circuits


def _measurement_circuit(num_qubits: int, pauli: Pauli):
    # Note: if pauli is I for all qubits, this function generates a circuit to measure only
    # the first qubit.
    # Although such an operator can be optimized out by interpreting it as a constant (1),
    # this optimization requires changes in various methods. So it is left as future work.
    qubit_indices = np.arange(pauli.num_qubits)[pauli.z | pauli.x]
    if not np.any(qubit_indices):
        qubit_indices = [0]
    meas_circuit = QuantumCircuit(
        QuantumRegister(num_qubits, "q"), ClassicalRegister(len(qubit_indices), f"__c_{pauli}")
    )
    for clbit, i in enumerate(qubit_indices):
        if pauli.x[i]:
            if pauli.z[i]:
                meas_circuit.sdg(i)
            meas_circuit.h(i)
        meas_circuit.measure(i, clbit)
    return meas_circuit, qubit_indices
