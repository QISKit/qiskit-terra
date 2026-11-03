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

"""Sampler V2 implementation for an arbitrary Backend object."""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.primitives.backend_estimator_v2 import _run_circuits
from qiskit.primitives.base import BaseSamplerV2
from qiskit.primitives.containers import (
    BitArray,
    DataBin,
    PrimitiveResult,
    SamplerPubLike,
    SamplerPubResult,
)
from qiskit.primitives.containers.bit_array import _min_num_bytes
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.primitive_job import PrimitiveJob
from qiskit.providers.backend import BackendV2
from qiskit.result import Result


@dataclass
class Options:
    """Options for :class:`~.BackendSamplerV2`"""

    default_shots: int = 1024
    """The default shots to use if none are specified in :meth:`~.run`.
    Default: 1024.
    """

    seed_simulator: int | None = None
    """The seed to use in the simulator. If None, a random seed will be used.
    Default: None.
    """

    run_options: dict[str, Any] | None = None
    """A dictionary of options to pass to the backend's ``run()`` method.
    Default: None (no option passed to backend's ``run`` method)
    """


@dataclass
class _MeasureInfo:
    creg_name: str
    num_bits: int
    num_bytes: int
    start: int


ResultMemory = list[str] | list[list[float]] | list[list[list[float]]]
"""Type alias for possible level 2 and level 1 result memory formats. For level
2, the format is a list of bit strings. For level 1, format can be either a
list of I/Q pairs (list with two floats) for each memory slot if using
``meas_return=avg`` or a list of lists of I/Q pairs if using
``meas_return=single`` with the outer list indexing shot number and the inner
list indexing memory slot.
"""


class BackendSamplerV2(BaseSamplerV2):
    """Evaluates bitstrings for provided quantum circuits

    The :class:`~.BackendSamplerV2` class is a generic implementation of the
    :class:`~.BaseSamplerV2` interface that is used to wrap a :class:`~.BackendV2`
    object in the class :class:`~.BaseSamplerV2` API. It
    facilitates using backends that do not provide a native
    :class:`~.BaseSamplerV2` implementation in places that work with
    :class:`~.BaseSamplerV2`. However,
    if you're using a provider that has a native implementation of
    :class:`~.BaseSamplerV2`, it is a better choice to leverage that native
    implementation as it will likely include additional optimizations and be
    a more efficient implementation. The generic nature of this class
    precludes doing any provider- or backend-specific optimizations.

    This class does not perform any measurement or gate mitigation.

    Each tuple of ``(circuit, <optional> parameter values, <optional> shots)``, called a sampler
    primitive unified bloc (PUB), produces its own array-valued result. The :meth:`~run` method can
    be given many pubs at once.

    The options for :class:`~.BackendSamplerV2` consist of the following items.

    * ``default_shots``: The default shots to use if none are specified in :meth:`~run`.
      Default: 1024.

    * ``seed_simulator``: The seed to use in the simulator. If None, a random seed will be used.
      Default: None.

    * ``run_options``: A dictionary of options to pass through to the ``run()``
      method of the wrapped :class:`~.BackendV2` instance.

    .. note::

        When the backend does not support the ``memory`` run option, per-shot
        samples are derived from the returned counts. This preserves outcome
        frequencies, but the order of samples does not reflect backend shot
        order.

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
            options: The options to control the default shots (``default_shots``) and
                the random seed for the simulator (``seed_simulator``).
        """
        self._backend = backend
        self._options = Options(**options) if options else Options()

    @property
    def backend(self) -> BackendV2:
        """Returns the backend which this sampler object is based on."""
        return self._backend

    @property
    def options(self) -> Options:
        """Return the options"""
        return self._options

    def run(
        self, pubs: Iterable[SamplerPubLike], *, shots: int | None = None
    ) -> PrimitiveJob[PrimitiveResult[SamplerPubResult]]:
        if shots is None:
            shots = self._options.default_shots
        coerced_pubs = [SamplerPub.coerce(pub, shots) for pub in pubs]
        self._validate_pubs(coerced_pubs)
        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _validate_pubs(self, pubs: list[SamplerPub]):
        for i, pub in enumerate(pubs):
            if len(pub.circuit.cregs) == 0:
                warnings.warn(
                    f"The {i}-th pub's circuit has no output classical registers and so the result "
                    "will be empty. Did you mean to add measurement instructions?",
                    UserWarning,
                )

    def _run(self, pubs: list[SamplerPub]) -> PrimitiveResult[SamplerPubResult]:
        pub_dict = defaultdict(list)
        # consolidate pubs with the same number of shots
        for i, pub in enumerate(pubs):
            pub_dict[pub.shots].append(i)

        results = [None] * len(pubs)
        for shots, lst in pub_dict.items():
            # run pubs with the same number of shots at once
            pub_results = self._run_pubs([pubs[i] for i in lst], shots)
            # reconstruct the result of pubs
            for i, pub_result in zip(lst, pub_results):
                results[i] = pub_result
        return PrimitiveResult(results, metadata={"version": 2})

    def _run_pubs(self, pubs: list[SamplerPub], shots: int) -> list[SamplerPubResult]:
        """Compute results for pubs that all require the same value of ``shots``."""
        # prepare circuits
        bound_circuits = [pub.parameter_values.bind_all(pub.circuit) for pub in pubs]
        flatten_circuits = []
        for circuits in bound_circuits:
            flatten_circuits.extend(np.ravel(circuits).tolist())

        run_opts = dict(self._options.run_options or {})
        # Keep the existing memory=True path for backends that support it,
        # but avoid passing it to minimal BackendV2 implementations that do
        # not advertise a memory run option.
        if "memory" in self._backend.options:
            run_opts.setdefault("memory", True)
        else:
            run_opts.pop("memory", None)
        run_opts.pop("seed_simulator", None)
        run_opts.setdefault("shots", shots)
        if self._options.seed_simulator is not None:
            run_opts["seed_simulator"] = self._options.seed_simulator
        # run circuits
        results, _ = _run_circuits(
            flatten_circuits,
            self._backend,
            clear_metadata=False,
            **run_opts,
        )
        result_memory = _prepare_memory(results)

        # pack memory to an ndarray of uint8
        results = []
        start = 0
        meas_level = (
            None
            if self._options.run_options is None
            else self._options.run_options.get("meas_level")
        )
        for pub, bound in zip(pubs, bound_circuits):
            meas_info, max_num_bytes = _analyze_circuit(pub.circuit)
            end = start + bound.size
            results.append(
                self._postprocess_pub(
                    result_memory[start:end],
                    shots,
                    bound.shape,
                    meas_info,
                    max_num_bytes,
                    pub.circuit.metadata,
                    meas_level,
                )
            )
            start = end

        return results

    def _postprocess_pub(
        self,
        result_memory: list[ResultMemory],
        shots: int,
        shape: tuple[int, ...],
        meas_info: list[_MeasureInfo],
        max_num_bytes: int,
        circuit_metadata: dict,
        meas_level: int | None,
    ) -> SamplerPubResult:
        """Converts the memory data into a sampler pub result

        For level 2 data, the memory data are stored in an array of bit arrays
        with the shape of the pub. For level 1 data, the data are stored in a
        complex numpy array.
        """
        if meas_level == 2 or meas_level is None:
            arrays = {
                item.creg_name: np.zeros(shape + (shots, item.num_bytes), dtype=np.uint8)
                for item in meas_info
            }
            memory_array = _memory_array(result_memory, max_num_bytes)

            for samples, index in zip(memory_array, np.ndindex(*shape)):
                for item in meas_info:
                    ary = _samples_to_packed_array(samples, item.num_bits, item.start)
                    arrays[item.creg_name][index] = ary

            meas = {
                item.creg_name: BitArray(arrays[item.creg_name], item.num_bits)
                for item in meas_info
            }
        elif meas_level == 1:
            raw = np.array(result_memory)
            cplx = raw[..., 0] + 1j * raw[..., 1]
            cplx = np.reshape(cplx, (*shape, *cplx.shape[1:]))
            meas = {item.creg_name: cplx for item in meas_info}
        else:
            raise QiskitError(f"Unsupported meas_level: {meas_level}")
        return SamplerPubResult(
            DataBin(**meas, shape=shape),
            metadata={"shots": shots, "circuit_metadata": circuit_metadata},
        )


def _analyze_circuit(circuit: QuantumCircuit) -> tuple[list[_MeasureInfo], int]:
    """Analyzes the information for each creg in a circuit."""
    meas_info = []
    max_num_bits = 0
    for creg in circuit.cregs:
        name = creg.name
        num_bits = creg.size
        if num_bits != 0:
            start = circuit.find_bit(creg[0]).index
        else:
            start = 0
        meas_info.append(
            _MeasureInfo(
                creg_name=name,
                num_bits=num_bits,
                num_bytes=_min_num_bytes(num_bits),
                start=start,
            )
        )
        max_num_bits = max(max_num_bits, start + num_bits)
    return meas_info, _min_num_bytes(max_num_bits)


def _prepare_memory(results: list[Result]) -> list[ResultMemory]:
    """Joins split results if exceeding max_experiments.

    When the backend does not support the ``memory`` run option (not part of
    BackendV2 abstract interface), per-shot data is derived from counts so
    that BackendSamplerV2 works with any BackendV2.
    """
    lst = []
    for res in results:
        for exp in res.results:
            if hasattr(exp.data, "memory") and exp.data.memory:
                lst.append(exp.data.memory)
            elif hasattr(exp.data, "counts") and exp.data.counts:
                # Backend did not return memory; expand counts to per-shot list
                lst.append(_counts_to_memory(exp.data.counts, exp.shots))
            else:
                # no measure in a circuit
                lst.append(["0x0"] * exp.shots)
    return lst


def _counts_to_memory(counts: dict, shots: int) -> list[str]:
    """Expand a counts dict (outcome -> count) to a list of hex strings.

    Keys may be hex strings (e.g. "0x0"), binary (e.g. "0b011"), bit strings
    (e.g. "011"), or ints. They are normalized to hex strings for downstream
    use, following the same semantics as :class:`qiskit.result.Counts`.
    Produces a list of length shots with deterministic ordering (sorted
    numerically) so that results are reproducible.
    """
    # Match Counts: bit strings are [01] plus optional spaces/underscores
    _bitstring_re = re.compile(r"^[01\s_]+$")

    def _key_to_int(outcome: str | int) -> int:
        if isinstance(outcome, int):
            return outcome
        if outcome.startswith("0x"):
            return int(outcome, 0)
        if outcome.startswith("0b"):
            return int(outcome, 0)
        if _bitstring_re.match(outcome):
            return int(outcome.replace(" ", "").replace("_", ""), 2)
        return int(outcome, 16)

    memory = []
    for outcome in sorted(counts.keys(), key=_key_to_int):
        hex_outcome = hex(_key_to_int(outcome))
        memory.extend([hex_outcome] * counts[outcome])
    if len(memory) < shots:
        memory.extend(["0x0"] * (shots - len(memory)))
    return memory[:shots]


def _memory_array(results: list[list[str]], num_bytes: int) -> NDArray[np.uint8]:
    """Converts the memory data into an array in an unpacked way.

    The ``num_bytes`` argument is the *minimum* number of bytes required to
    represent the classical bits in the circuit (derived from the cregs).
    Some backends, however, may include additional classical memory slots in
    their returned memory values (for example, extra bits in ``memory_slots``),
    which means the values can require more bytes than ``num_bytes``.

    To avoid ``OverflowError`` when converting these values to bytes, this
    function computes the number of bytes actually required by the data and
    uses ``max(num_bytes, required_bytes)`` as the byte width. Extra high-order
    bits are ignored later when we slice out the bits corresponding to the
    circuit's classical registers in :func:`_samples_to_packed_array`.
    """
    lst = []
    for memory in results:
        if num_bytes > 0:
            parsed_values: list[int] = []
            required_bytes = 0

            for value in memory:
                if isinstance(value, int):
                    as_int = value
                else:
                    v = str(value).strip().replace(" ", "").replace("_", "")
                    if not v:
                        as_int = 0
                    elif v.startswith(("0x", "0X", "0b", "0B")):
                        as_int = int(v, 0)
                    elif re.fullmatch(r"[01]+", v):
                        as_int = int(v, 2)
                    elif re.fullmatch(r"[0-9]+", v):
                        as_int = int(v, 10)
                    else:
                        as_int = int(v, 16)

                parsed_values.append(as_int)
                bits = max(1, as_int.bit_length())
                needed = (bits + 7) // 8
                required_bytes = max(required_bytes, needed)

            width = max(num_bytes, required_bytes)
            data = b"".join(v.to_bytes(width, "big") for v in parsed_values)
            data = np.frombuffer(data, dtype=np.uint8).reshape(-1, width)
        else:
            # no measure in a circuit
            data = np.zeros((len(memory), num_bytes), dtype=np.uint8)
        lst.append(data)
    ary = np.asarray(lst)
    return np.unpackbits(ary, axis=-1, bitorder="big")


def _samples_to_packed_array(
    samples: NDArray[np.uint8], num_bits: int, start: int
) -> NDArray[np.uint8]:
    """Converts an unpacked array of the memory data into a packed array."""
    # samples of `Backend.run(memory=True)` will be the order of
    # clbit_last, ..., clbit_1, clbit_0
    # place samples in the order of clbit_start+num_bits-1, ..., clbit_start+1, clbit_start
    if start == 0:
        ary = samples[:, -start - num_bits :]
    else:
        ary = samples[:, -start - num_bits : -start]
    # pad 0 in the left to align the number to be mod 8
    # since np.packbits(bitorder='big') pads 0 to the right.
    pad_size = -num_bits % 8
    ary = np.pad(ary, ((0, 0), (pad_size, 0)), constant_values=0)
    # pack bits in big endian order
    ary = np.packbits(ary, axis=-1, bitorder="big")
    return ary
