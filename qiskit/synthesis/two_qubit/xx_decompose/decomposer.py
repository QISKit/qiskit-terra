# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Driver for a synthesis routine which emits optimal XX-based circuits.
"""
from __future__ import annotations
import heapq
import math
from operator import itemgetter
from collections.abc import Callable

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import RXXGate, RZXGate, UnitaryGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, average_gate_fidelity
from qiskit.synthesis.one_qubit.one_qubit_decompose import ONE_QUBIT_EULER_BASIS_GATES
from qiskit.synthesis.two_qubit.two_qubit_decompose import TwoQubitWeylDecomposition

from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from .utilities import EPSILON
from .polytopes import XXPolytope


def _average_infidelity(p, q):
    """
    Computes the infidelity distance between two points p, q expressed in positive canonical
    coordinates.
    """

    a0, b0, c0 = p
    a1, b1, c1 = q

    return 1 - 1 / 20 * (
        4
        + 16
        * (
            math.cos(a0 - a1) ** 2 * math.cos(b0 - b1) ** 2 * math.cos(c0 - c1) ** 2
            + math.sin(a0 - a1) ** 2 * math.sin(b0 - b1) ** 2 * math.sin(c0 - c1) ** 2
        )
    )


class XXDecomposer:
    r"""
    A class for optimal decomposition of 2-qubit unitaries into 2-qubit basis gates of ``XX`` type
    (i.e., each locally equivalent to :math:`CAN(\alpha, 0, 0)` for a possibly varying :math:`\alpha`).

    Args:
        basis_fidelity: available strengths and fidelity of each.
            Can be either (1) a dictionary mapping ``XX`` angle values to fidelity at that angle; or
            (2) a single float ``f``, interpreted as ``{pi: f, pi/2: f/2, pi/3: f/3}``.
        euler_basis: Basis string provided to :class:`.OneQubitEulerDecomposer` for 1Q synthesis.
            Defaults to ``"U"``.
        embodiments: A dictionary mapping interaction strengths alpha to native circuits which
            embody the gate :math:`CAN(\alpha, 0, 0)`. Strengths are taken so that :math:`\pi/2`
            represents the class of a full :class:`.CXGate`.
        backup_optimizer: If supplied, defers synthesis to this callable when :class:`.XXDecomposer`
            has no efficient decomposition of its own. Useful for special cases involving 2 or 3
            applications of :math:`XX(\pi/2)`, in which case standard synthesis methods provide lower
            1Q gate count.

    .. note::
        If ``embodiments`` is not passed, or if an entry is missing, it will be populated as needed
        using the method ``_default_embodiment``.

    .. automethod:: __call__
    """

    def __init__(
        self,
        basis_fidelity: dict | float = 1.0,
        euler_basis: str = "U",
        embodiments: dict[float, QuantumCircuit] | None = None,
        backup_optimizer: Callable[..., QuantumCircuit] | None = None,
    ):
        from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
            Optimize1qGatesDecomposition,
        )

        self._decomposer1q = Optimize1qGatesDecomposition(ONE_QUBIT_EULER_BASIS_GATES[euler_basis])
        self.embodiments = embodiments if embodiments is not None else {}
        self.backup_optimizer = backup_optimizer
        self.basis_fidelity = basis_fidelity

        # expose one of the basis gates so others can know what this decomposer targets
        embodiment_circuit = next(iter(self.embodiments.values()), QuantumCircuit())
        for instruction in embodiment_circuit:
            if len(instruction.qubits) == 2:
                self.gate = instruction.operation
                break
        else:
            self.gate = RZXGate(np.pi / 2)

        self._check_embodiments()

    @staticmethod
    def _default_embodiment(strength):
        """
        If the user does not provide a custom implementation of XX(strength), then this routine
        defines a default implementation using RZX.
        """
        xx_circuit = QuantumCircuit(2)

        # NOTE: One could branch here on `strength == np.pi / 2` and decide to use a CX-based
        #       circuit in this one case where it's available.
        xx_circuit.h(0)
        xx_circuit.rzx(strength, 0, 1)
        xx_circuit.h(0)

        return xx_circuit

    def _check_embodiments(self):
        """
        Checks that `self.embodiments` is populated with legal circuit embodiments: the key-value
        pair (angle, circuit) satisfies Operator(circuit) approx RXX(angle).to_matrix().
        """

        for angle, embodiment in self.embodiments.items():
            actual = Operator(RXXGate(angle))
            purported = Operator(embodiment)
            if average_gate_fidelity(actual, purported) < 1 - EPSILON:
                raise QiskitError(
                    f"RXX embodiment provided for angle {angle} disagrees with RXXGate({angle})"
                )

    @staticmethod
    def _best_decomposition(canonical_coordinate, available_strengths):
        """
        Finds the cheapest sequence of `available_strengths` which supports the best approximation
        to `canonical_coordinate`. Returns a dictionary with keys "cost", "point", and "operations".

        NOTE: `canonical_coordinate` is a positive canonical coordinate. `strengths` is a dictionary
              mapping the available strengths to their (infidelity) costs, with the strengths
              themselves normalized so that pi/2 represents CX = RZX(pi/2).
        """
        best_point, best_cost, best_sequence = [0, 0, 0], 1.0, []
        priority_queue = []
        heapq.heappush(priority_queue, (0, []))
        canonical_coordinate = np.array(canonical_coordinate)

        while True:
            if len(priority_queue) == 0:
                if len(available_strengths) == 0:
                    raise QiskitError(
                        "Attempting to synthesize entangling gate with no controlled gates in basis set."
                    )
                raise QiskitError("Unable to synthesize a 2q unitary with the supplied basis set.")

            sequence_cost, sequence = heapq.heappop(priority_queue)

            strength_polytope = XXPolytope.from_strengths(*[x / 2 for x in sequence])
            candidate_point = strength_polytope.nearest(canonical_coordinate)
            candidate_cost = sequence_cost + _average_infidelity(
                canonical_coordinate, candidate_point
            )

            if candidate_cost < best_cost:
                best_point, best_cost, best_sequence = candidate_point, candidate_cost, sequence

            if strength_polytope.member(canonical_coordinate):
                break

            for strength, extra_cost in available_strengths.items():
                if len(sequence) == 0 or strength <= sequence[-1]:
                    heapq.heappush(
                        priority_queue, (sequence_cost + extra_cost, sequence + [strength])
                    )

        return {"point": best_point, "cost": best_cost, "sequence": best_sequence}

    def num_basis_gates(self, unitary: Operator | np.ndarray):
        """
        Counts the number of gates that would be emitted during re-synthesis.

        .. note::
            This method is used by :class:`.ConsolidateBlocks`.
        """
        strengths = self._strength_to_infidelity(1.0)

        # get the associated _positive_ canonical coordinate
        weyl_decomposition = TwoQubitWeylDecomposition(unitary)
        target = [getattr(weyl_decomposition, x) for x in ("a", "b", "c")]
        if target[-1] < -EPSILON:
            target = [np.pi / 2 - target[0], target[1], -target[2]]

        best_sequence = self._best_decomposition(target, strengths)["sequence"]
        return len(best_sequence)

    @staticmethod
    def _strength_to_infidelity(basis_fidelity, approximate=False):
        """
        Converts a dictionary mapping XX strengths to fidelities to a dictionary mapping XX
        strengths to infidelities. Also supports one of the other formats Qiskit uses: if only a
        lone float is supplied, it extends it from CX over CX/2 and CX/3 by linear decay.
        """

        if isinstance(basis_fidelity, float):
            if not approximate:
                slope, offset = 1e-10, 1e-12
            else:
                slope, offset = (1 - basis_fidelity) / 2, (1 - basis_fidelity) / 2
            return {
                strength: slope * strength / (np.pi / 2) + offset
                for strength in [np.pi / 2, np.pi / 4, np.pi / 6]
            }
        elif isinstance(basis_fidelity, dict):
            return {
                strength: (1 - fidelity if approximate else 1e-12 + 1e-10 * strength / (np.pi / 2))
                for (strength, fidelity) in basis_fidelity.items()
            }

        raise TypeError("Unknown basis_fidelity payload.")

    def __call__(
        self,
        unitary: Operator | np.ndarray,
        basis_fidelity: dict | float | None = None,
        approximate: bool = True,
        use_dag: bool = False,
    ) -> QuantumCircuit:
        r"""
        Fashions a circuit which (perhaps approximately) models the special unitary operation
        ``unitary``, using the circuit templates supplied at initialization as ``embodiments``.  The
        routine uses ``basis_fidelity`` to select the optimal circuit template, including when
        performing exact synthesis; the contents of ``basis_fidelity`` is a dictionary mapping
        interaction strengths (scaled so that :math:`CX = RZX(\pi/2)` corresponds to :math:`\pi/2`)
        to circuit fidelities.

        Args:
            unitary (Operator or ndarray): :math:`4 \times 4` unitary to synthesize.
            basis_fidelity (dict or float): Fidelity of basis gates. Can be either (1) a dictionary
                mapping ``XX`` angle values to fidelity at that angle; or (2) a single float ``f``,
                interpreted as ``{pi: f, pi/2: f/2, pi/3: f/3}``.
                If given, overrides the basis_fidelity given at init.
            approximate (bool): Approximates if basis fidelities are less than 1.0 .
            use_dag (bool): If true a :class:`.DAGCircuit` is returned instead of a
                :class:`QuantumCircuit` when this class is called.

        Returns:
            QuantumCircuit: Synthesized circuit.
        """
        basis_fidelity = basis_fidelity or self.basis_fidelity
        strength_to_infidelity = self._strength_to_infidelity(
            basis_fidelity, approximate=approximate
        )

        # get the associated _positive_ canonical coordinate
        weyl_decomposition = TwoQubitWeylDecomposition(unitary)
        target = [getattr(weyl_decomposition, x) for x in ("a", "b", "c")]
        if target[-1] < -EPSILON:
            target = [np.pi / 2 - target[0], target[1], -target[2]]

        # scan for the best point
        best_point, best_sequence = itemgetter("point", "sequence")(
            self._best_decomposition(target, strength_to_infidelity)
        )
        # build the circuit building this canonical gate
        embodiments = {
            k: self.embodiments.get(k, self._default_embodiment(k))
            for k, v in strength_to_infidelity.items()
        }
        circuit = canonical_xx_circuit(best_point, best_sequence, embodiments)

        if (
            best_sequence in ([np.pi / 2, np.pi / 2, np.pi / 2], [np.pi / 2, np.pi / 2])
            and self.backup_optimizer is not None
        ):
            pi2_fidelity = 1 - strength_to_infidelity[np.pi / 2]
            return self.backup_optimizer(unitary, basis_fidelity=pi2_fidelity, use_dag=use_dag)

        # change to positive canonical coordinates
        if weyl_decomposition.c >= -EPSILON:
            # if they're the same...
            corrected_circuit = QuantumCircuit(2)
            corrected_circuit.rz(np.pi, [0])
            corrected_circuit.compose(circuit, [0, 1], inplace=True)
            corrected_circuit.rz(-np.pi, [0])
            circuit = corrected_circuit
        else:
            # else they're in the "positive" scissors part...
            corrected_circuit = QuantumCircuit(2)
            _, source_reflection, _ = apply_reflection("reflect XX, ZZ", [0, 0, 0])
            _, source_shift, _ = apply_shift("X shift", [0, 0, 0])

            corrected_circuit.compose(source_reflection.inverse(), inplace=True)
            corrected_circuit.rz(np.pi, [0])
            corrected_circuit.compose(circuit, [0, 1], inplace=True)
            corrected_circuit.rz(-np.pi, [0])
            corrected_circuit.compose(source_shift.inverse(), inplace=True)
            corrected_circuit.compose(source_reflection, inplace=True)
            corrected_circuit.global_phase += np.pi / 2

            circuit = corrected_circuit

        circ = QuantumCircuit(2, global_phase=weyl_decomposition.global_phase)

        circ.append(UnitaryGate(weyl_decomposition.K2r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K2l), [1])
        circ.compose(circuit, [0, 1], inplace=True)
        circ.append(UnitaryGate(weyl_decomposition.K1r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K1l), [1])

        circ = self._decomposer1q(circ)
        if use_dag:
            return circ.to_dag(copy_operations=False)
        return circ
