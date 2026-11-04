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

"""Routing via SWAP insertion using the SABRE method from Li et al."""

import functools
import logging
import time

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.target import Target, _FakeTarget
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passes.layout import disjoint_utils
from qiskit.utils import default_num_processes

from qiskit._accelerate.sabre import sabre_routing, Heuristic, SetScaling, RoutingTarget
from qiskit._accelerate.nlayout import NLayout

LOG = logging.getLogger(__name__)


class SabreSwap(TransformationPass):
    r"""Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of the SWAP-based heuristic search from the SABRE qubit
    mapping paper [2] (Algorithm 1) with the modifications from the LightSABRE
    paper [1]. The heuristic aims to minimize the number of lossy SWAPs inserted
    and the depth of the circuit.

    This algorithm starts from an initial layout of virtual qubits onto physical
    qubits, and iterates over the circuit DAG until all gates are exhausted,
    inserting SWAPs along the way. It only considers 2-qubit gates as only those
    are germane for the mapping problem (it is assumed that 3+ qubit gates are
    already decomposed).

    In each iteration, it will first check if there are any gates in the
    ``front_layer`` that can be directly applied. If so, it will apply them and
    remove them from ``front_layer``, and replenish that layer with new gates
    if possible. Otherwise, it will try to search for SWAPs, insert the SWAPs,
    and update the mapping.

    The search for SWAPs is restricted, in the sense that we only consider
    physical qubits in the neighborhood of those qubits involved in
    ``front_layer``. These give rise to a ``swap_candidate_list`` which is
    scored according to some heuristic cost function. The best SWAP is
    implemented and ``current_layout`` updated.

    This transpiler pass adds onto the SABRE algorithm in that it will run
    multiple trials of the algorithm with different seeds. The best output,
    determined by the trial with the least amount of SWAPs inserted, will
    be selected from the random trials.

    **References:**

    [1] Henry Zou and Matthew Treinish and Kevin Hartman and Alexander Ivrii and Jake Lishman.
    "LightSABRE: A Lightweight and Enhanced SABRE Algorithm"
    `arXiv:2409.08368 <https://doi.org/10.48550/arXiv.2409.08368>`__
    [2] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(self, coupling_map, heuristic="basic", seed=None, fake_run=False, trials=None):
        r"""SabreSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): CouplingMap of the target backend.
            heuristic (str): The type of heuristic to use when deciding best
                swap strategy ('basic' or 'lookahead' or 'decay').
            seed (int): random seed used to tie-break among candidate swaps.
            fake_run (bool): if true, it only pretend to do routing, i.e., no
                swap is effectively added.
            trials (int): The number of seed trials to run sabre with. These will
                be run in parallel (unless the PassManager is already running in
                parallel). If not specified this defaults to the number of physical
                CPUs on the local system. For reproducible results it is recommended
                that you set this explicitly, as the output will be deterministic for
                a fixed number of trials.

        Raises:
            TranspilerError: If the specified heuristic is not valid.

        Additional Information:

            The search space of possible SWAPs on physical qubits is explored
            by assigning a score to the layout that would result from each SWAP.
            The goodness of a layout is evaluated based on how viable it makes
            the remaining virtual gates that must be applied. A few heuristic
            cost functions are supported

            - 'basic':

            The sum of distances for corresponding physical qubits of
            interacting virtual qubits in the front_layer.

            .. math::

                H_{basic} = \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'lookahead':

            This is the sum of two costs: first is the same as the basic cost.
            Second is the basic cost but now evaluated for the
            extended set as well (i.e. :math:`|E|` number of upcoming successors to gates in
            front_layer F). This is weighted by some amount EXTENDED_SET_WEIGHT (W) to
            signify that upcoming gates are less important than the front_layer.

            .. math::

                H_{decay}=\frac{1}{\left|{F}\right|}\sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]
                    + W*\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]

            - 'decay':

            This is the same as 'lookahead', but the whole cost is multiplied by a
            decay factor. This increases the cost if the SWAP that generated the
            trial layout was recently used (i.e. it penalizes increase in depth).

            .. math::

                H_{decay} = max(decay(SWAP.q_1), decay(SWAP.q_2)) {
                    \frac{1}{\left|{F}\right|} \sum_{gate \in F} D[\pi(gate.q_1)][\pi(gate.q2)]\\
                    + W *\frac{1}{\left|{E}\right|} \sum_{gate \in E} D[\pi(gate.q_1)][\pi(gate.q2)]
                    }
        """
        super().__init__()
        self._routing_target = None
        if isinstance(coupling_map, Target) and not isinstance(coupling_map, _FakeTarget):
            self.target = coupling_map
        elif coupling_map is None:
            # This is an invalid state, but we defer the error to runtime to match historical
            # behaviour of Qiskit.
            self.target = None
        else:
            coupling_map = (
                coupling_map.build_coupling_map()
                if isinstance(coupling_map, _FakeTarget)
                else coupling_map
            )
            # A dummy target to represent the same coupling constraints. Basis gates are arbitrary.
            self.target = Target.from_configuration(
                basis_gates=["u", "cx"], coupling_map=coupling_map
            )

        self.heuristic = heuristic
        self.seed = seed
        self.trials = default_num_processes() if trials is None else trials
        self.fake_run = fake_run

    @functools.cached_property
    def dist_matrix(self):
        # This property is not intended to be public API, it just keeps backwards compatibility.
        return None if self._routing_target is None else self._routing_target.distance_matrix()

    @functools.cached_property
    def coupling_map(self):
        # This property is not intended to be public API, it just keeps backwards compatibility.
        return (
            None
            if self._routing_target is None
            else CouplingMap(self._routing_target.coupling_list())
        )

    def run(self, dag):
        """Run the SabreSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped.
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """
        if self.target is None:
            raise TranspilerError("SabreSwap cannot run with coupling_map=None")
        if self._routing_target is None:
            self._routing_target = RoutingTarget.from_target(self.target)
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Sabre swap runs on physical circuits only.")
        num_dag_qubits = len(dag.qubits)
        num_coupling_qubits = self.target.num_qubits
        if num_dag_qubits < num_coupling_qubits:
            raise TranspilerError(
                f"Fewer qubits in the circuit ({num_dag_qubits}) than the coupling map"
                f" ({num_coupling_qubits})."
                " Have you run a layout pass and then expanded your DAG with ancillas?"
                " See `FullAncillaAllocation`, `EnlargeWithAncilla` and `ApplyLayout`."
            )
        if num_dag_qubits > num_coupling_qubits:
            raise TranspilerError(
                f"More qubits in the circuit ({num_dag_qubits}) than available in the coupling map"
                f" ({num_coupling_qubits})."
                " This circuit cannot be routed to this device."
            )

        # In our defaults, the basic heuristic shouldn't scale by size; if it does, it's liable to
        # get the algorithm stuck.  See https://github.com/Qiskit/qiskit/pull/14458 for more.
        if isinstance(self.heuristic, Heuristic):
            heuristic = self.heuristic
        elif self.heuristic == "basic":
            heuristic = Heuristic(attempt_limit=10 * num_dag_qubits).with_basic(
                1.0, SetScaling.Size
            )
        elif self.heuristic == "lookahead":
            heuristic = (
                Heuristic(attempt_limit=10 * num_dag_qubits)
                .with_basic(1.0, SetScaling.Constant)
                .with_lookahead([0.5 / num_coupling_qubits], SetScaling.Constant)
            )
        elif self.heuristic == "decay":
            heuristic = (
                Heuristic(attempt_limit=10 * num_dag_qubits)
                .with_basic(1.0, SetScaling.Constant)
                .with_lookahead([0.5 / num_coupling_qubits], SetScaling.Constant)
                .with_decay(0.001, 5)
            )
        else:
            raise TranspilerError(f"Heuristic {self.heuristic} not recognized.")
        disjoint_utils.require_layout_isolated_to_component(dag, self.target)

        initial_layout = NLayout.generate_trivial_layout(num_dag_qubits)
        sabre_start = time.perf_counter()
        dag, final_layout = sabre_routing(
            dag, self._routing_target, heuristic, initial_layout, self.trials, self.seed
        )
        sabre_stop = time.perf_counter()
        LOG.debug("Sabre swap algorithm execution complete in: %s", sabre_stop - sabre_start)
        permutation = [
            final_layout.virtual_to_physical(initial_layout.physical_to_virtual(i))
            for i in range(num_dag_qubits)
        ]
        layout = Layout(dict(zip(dag.qubits, permutation)))
        self.property_set["final_layout"] = (
            layout
            if (prev := self.property_set["final_layout"]) is None
            # The "final layout" can be thought of as a "comes from" permutation that you apply at
            # the end of the circuit to invert the routing.  So if there's an existing one, what we
            # apply at the end of the circuit needs to set the circuit qubits so they "come from"
            # the previous one, then those "come from" the one we've just added.
            else prev.compose(layout, dag.qubits)
        )
        return dag
