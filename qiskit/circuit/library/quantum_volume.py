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

"""Quantum Volume model circuit."""

from __future__ import annotations


import numpy as np
from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library.generalized_gates import PermutationGate, UnitaryGate
from qiskit._accelerate.circuit_library import quantum_volume as qv_rs
from qiskit.utils.deprecation import deprecate_func


class QuantumVolume(QuantumCircuit):
    """A quantum volume model circuit.

    The model circuits are random instances of circuits used to measure
    the Quantum Volume metric, as introduced in [1].

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    Reference Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import QuantumVolume
       circuit = QuantumVolume(5, 6, seed=10)
       circuit.draw('mpl')

    Expanded Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import QuantumVolume
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = QuantumVolume(5, 6, seed=10, classical_permutation=False)
       _generate_circuit_library_visualization(circuit.decompose())

    References:

    [1] A. Cross et al. Validating quantum computers using
    randomized model circuits, Phys. Rev. A 100, 032328 (2019).
    `arXiv:1811.12926 <https://arxiv.org/abs/1811.12926>`__
    """

    @deprecate_func(
        since="2.2",
        additional_msg="Use the function qiskit.circuit.library.quantum_volume instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_qubits: int,
        depth: int | None = None,
        seed: int | np.random.Generator | None = None,
        classical_permutation: bool = True,
        *,
        flatten: bool = False,
    ) -> None:
        """
        Args:
            num_qubits: number of active qubits in model circuit.
            depth: layers of SU(4) operations in model circuit.
            seed: Random number generator or generator seed.
            classical_permutation: use classical permutations at every layer,
                rather than quantum.
            flatten: If ``False`` (the default), construct a circuit that contains a single
                instruction, which in turn has the actual volume structure.  If ``True``, construct
                the volume structure directly.

        """
        import scipy.stats

        # Parameters
        depth = depth or num_qubits  # how many layers of SU(4)
        width = num_qubits // 2  # how many SU(4)s fit in each layer
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        seed_name = seed
        if seed_name is None:
            # Get the internal entropy used to seed the default RNG, if no seed was given.  This
            # stays in the output name, so effectively stores a way of regenerating the circuit.
            # This is just best-effort only, for backwards compatibility, and isn't critical (if
            # someone needs full reproducibility, they should be manually controlling the seeding).
            seed_name = getattr(getattr(rng.bit_generator, "seed_seq", None), "entropy", None)

        super().__init__(
            num_qubits,
            name="quantum_volume_" + str([num_qubits, depth, seed_name]).replace(" ", ""),
        )
        if classical_permutation:
            if seed is not None:
                max_value = np.iinfo(np.int64).max
                seed = rng.integers(max_value, dtype=np.int64)
            qv_circ = quantum_volume(num_qubits, depth, seed)
            qv_circ.name = self.name
            if flatten:
                self.compose(qv_circ, inplace=True)
            else:
                self._append(CircuitInstruction(qv_circ.to_instruction(), tuple(self.qubits)))
        else:
            if seed is None:
                seed = seed_name

            base = self if flatten else QuantumCircuit(num_qubits, name=self.name)

            # For each layer, generate a permutation of qubits
            # Then generate and apply a Haar-random SU(4) to each pair
            unitaries = scipy.stats.unitary_group.rvs(4, depth * width, rng).reshape(
                depth, width, 4, 4
            )
            qubits = tuple(base.qubits)
            for row in unitaries:
                perm = rng.permutation(num_qubits)
                base._append(CircuitInstruction(PermutationGate(perm), qubits))
                for w, unitary in enumerate(row):
                    gate = UnitaryGate(unitary, check_input=False, num_qubits=2)
                    qubit = 2 * w
                    base._append(CircuitInstruction(gate, qubits[qubit : qubit + 2]))
            if not flatten:
                self._append(CircuitInstruction(base.to_instruction(), tuple(self.qubits)))


def quantum_volume(
    num_qubits: int,
    depth: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> QuantumCircuit:
    """A quantum volume model circuit.

    The model circuits are random instances of circuits used to measure
    the Quantum Volume metric, as introduced in [1].

    The model circuits consist of layers of Haar random
    elements of SU(4) applied between corresponding pairs
    of qubits in a random bipartition.

    This function is multithreaded and will launch a thread pool with threads equal to the number
    of CPUs by default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
    environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
    to 4 threads.

    Args:
        num_qubits: The number qubits to use for the generated circuit.
        depth: The number of layers for the generated circuit. If this
            is not specified it will default to ``num_qubits`` layers.
        seed: An optional RNG seed used for generating the random SU(4)
            matrices used in the output circuit. This can be either an
            integer or a numpy generator. If an integer is specified it must
            be an value between 0 and 2**64 - 1.

    Reference Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import quantum_volume
       circuit = quantum_volume(5, 6, seed=10)
       circuit.draw('mpl')

    References:

    [1] A. Cross et al. Validating quantum computers using
    randomized model circuits, Phys. Rev. A 100, 032328 (2019).
    `arXiv:1811.12926 <https://arxiv.org/abs/1811.12926>`__
    """
    if isinstance(seed, np.random.Generator):
        max_value = np.iinfo(np.int64).max
        seed = seed.integers(max_value, dtype=np.int64)
    depth = depth or num_qubits
    return QuantumCircuit._from_circuit_data(qv_rs(num_qubits, depth, seed))
