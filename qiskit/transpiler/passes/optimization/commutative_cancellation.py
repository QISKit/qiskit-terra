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

"""Cancel the redundant (self-adjoint) gates through commutation relations."""

from __future__ import annotations

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.commutation_library import StandardGateCommutations

from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.rz import RZGate
from qiskit._accelerate import commutation_cancellation
from qiskit._accelerate.commutation_checker import CommutationChecker

from qiskit.transpiler.passes.utils.control_flow import trivial_recurse

_CUTOFF_PRECISION = 1e-5


class CommutativeCancellation(TransformationPass):
    r"""Cancel self-adjoint gates and merge rotations by exploiting commutation relations.

    This pass uses commutation rules to apply the following optimizations
    to a sequence of gate:

    * **Self-inverse gates** (``h, y, cx, cy, cz``): if an even number of copies
      of the *same* self-inverse gate on the *same* qubit(s) commute together, they
      cancel completely; an odd number leaves a single copy behind.
    * **Same-axis rotations**: consecutive Z-rotations (``z, p, u1, rz, s, sdg, t, tdg``)
      or X-rotations (``x, rx, sx, sxdg``) on a qubit are summed into a single gate.
      A total angle that is a multiple of :math:`2\pi` removes all of them entirely
      (up to global phase), so inverse pairs like ``t`` + ``tdg`` cancel out naturally.

    Merging of Y-rotations is out of scope for this pass. Gates with symbolic
    (:class:`~.Parameter`) angles are also never merged.

    This pass is multithreaded and will potentially launch a thread pool with threads
    equal to the number of CPUs by default. Tune the number of threads with the
    ``RAYON_NUM_THREADS`` environment variable, e.g. ``RAYON_NUM_THREADS=4``.

    Example:
        the two ``cx`` gates below commute past the ``z`` gate (which acts
        only on the control qubit) and cancel each other, leaving just the ``z``::

                      ┌───┐            ┌───┐
            q_0: ──■──┤ Z ├──■──   ->  ┤ Z ├
                 ┌─┴─┐└-──┘┌─┴─┐       └───┘
            q_1: ┤ X ├─────┤ X ├   ->  ──────
                 └───┘     └───┘


        .. code-block:: python

            from qiskit import QuantumCircuit
            from qiskit.transpiler.passes import CommutativeCancellation

            qc = QuantumCircuit(2)
            qc.cx(0, 1)
            qc.z(0)
            qc.cx(0, 1)  # commutes past `z` and cancels the first `cx`

            optimized = CommutativeCancellation()(qc)
            optimized.count_ops()  # {'z': 1}

    See also :class:`.CommutativeOptimization`, which unifies and extends this pass's
    functionality together with :class:`.CommutativeInverseCancellation` — cancelling
    commuting inverse pairs beyond this pass's fixed self-inverse/rotation sets.
    """

    def __init__(
        self,
        basis_gates=None,
        target=None,
        approximation_degree: float = 1.0,
    ):
        """
        CommutativeCancellation initializer.

        Args:
            basis_gates (list[str]): Specifies which gate to use when writing back a
                merged same-axis rotation result, but only when the circuit itself does
                not already contain a suitable gate for that — the circuit always takes
                precedence over this list. For Z-rotations, the pass looks for ``rz``,
                ``p``, or ``u1``; if none of those is found in the circuit or in this
                list, Z-rotation merging is skipped entirely. For X-rotations, the pass
                looks for ``x`` or ``sx``; if neither is found, X-rotation merging still
                happens, just written as ``rx`` instead. Has no effect on which gates are
                eligible for cancellation in the first place; that set is fixed.
            target (Target): The :class:`~.Target` representing the target backend.
                Its operation names are extracted and used exactly like ``basis_gates``
                above — as a source of gate names for choosing the merged-rotation
                output gate. When both ``basis_gates`` and ``target`` are provided,
                ``target`` takes precedence and ``basis_gates`` is ignored entirely.
            approximation_degree: Threshold for treating two gates as commuting or
                cancelling even when they only do so approximately. It sets a
                tolerance of ``max(1e-12, 1 - approximation_degree)`` on the average
                gate fidelity between the two gate orderings; anything within that
                tolerance counts as commuting. The default, ``1.0``, means exact
                commutativity (up to floating-point rounding). Lowering it below
                ``1.0`` lets more gates be grouped and cancelled, at the cost of a
                small unitary error. This doesn't affect the separate, fixed check for
                whether a merged angle is close enough to 2π to drop.

        """
        super().__init__()
        if basis_gates:
            self.basis = set(basis_gates)
        else:
            self.basis = set()
        self.target = target
        self._approximation_degree = approximation_degree
        if target is not None:
            self.basis = set(target.operation_names)

        self._var_z_map = {"rz": RZGate, "p": PhaseGate, "u1": U1Gate}

        self._z_rotations = {"p", "z", "u1", "rz", "t", "s", "tdg", "sdg"}
        self._x_rotations = {"x", "rx", "sx", "sxdg"}
        self._gates = {"cx", "cy", "cz", "h", "y"}  # Now the gates supported are hard-coded

        # build a commutation checker restricted to the gates we cancel -- the others we
        # do not have to investigate, which allows to save time
        self._commutation_checker = CommutationChecker(
            StandardGateCommutations, gates=self._gates | self._z_rotations | self._x_rotations
        )

    @trivial_recurse
    def run(self, dag):
        """Run the CommutativeCancellation pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        commutation_cancellation.cancel_commutations(
            dag,
            self._commutation_checker,
            sorted(self.basis),
            self._approximation_degree,
        )
        return dag
