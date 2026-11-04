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

"""Layout selection using the SABRE bidirectional search approach from Li et al."""

import copy
import logging
import time

import numpy as np

from qiskit.converters import dag_to_circuit
from qiskit.circuit import QuantumRegister
from qiskit.transpiler.passes.layout.set_layout import SetLayout
from qiskit.transpiler.passes.layout.full_ancilla_allocation import FullAncillaAllocation
from qiskit.transpiler.passes.layout.enlarge_with_ancilla import EnlargeWithAncilla
from qiskit.transpiler.passes.layout.apply_layout import ApplyLayout
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target, _FakeTarget
from qiskit._accelerate.sabre import sabre_layout_and_routing, Heuristic, SetScaling
from qiskit.utils import default_num_processes

logger = logging.getLogger(__name__)


class SabreLayout(TransformationPass):
    """Choose a Layout via iterative bidirectional routing of the input circuit.

    Starting with a random initial `Layout`, the algorithm does a full routing
    of the circuit (via the `routing_pass` method) to end up with a
    `final_layout`. This final_layout is then used as the initial_layout for
    routing the reverse circuit. The algorithm iterates a number of times until
    it finds an initial_layout that reduces full routing cost.

    This method exploits the reversibility of quantum circuits, and tries to
    include global circuit information in the choice of initial_layout.

    By default, this pass will run both layout and routing and will transform the
    circuit so that the layout is applied to the input dag (meaning that the output
    circuit will have ancilla qubits allocated for unused qubits on the coupling map
    and the qubits will be reordered to match the mapped physical qubits) and then
    routing will be applied (inserting :class:`~.SwapGate` objects to account for limited
    connectivity). This is unlike most other layout passes which are :class:`~.AnalysisPass`
    objects and just find an initial layout and set that on the property set. This is
    done because by default the pass will run parallel seed trials with different random
    seeds for selecting the random initial layout and then selecting the routed output
    which results in the least number of swap gates needed.

    You can use the ``routing_pass`` argument to have this pass operate as a typical
    layout pass. When specified this will use the specified routing pass to select an
    initial layout only and will not run multiple seed trials.

    In addition to starting with a random initial `Layout` the pass can also take in
    an additional list of starting layouts which will be used for additional
    trials. If the ``sabre_starting_layouts`` is present in the property set
    when this pass is run, that will be used for additional trials. There will still
    be ``layout_trials`` of full random starting layouts run and the contents of
    ``sabre_starting_layouts`` will be run in addition to those. The output which results
    in the lowest amount of swap gates (whether from the random trials or the property
    set starting point) will be used. The value for this property set field should be a
    list of :class:`.Layout` objects representing the starting layouts to use. If a
    virtual qubit is missing from an :class:`.Layout` object in the list a random qubit
    will be selected.

    Property Set Fields Read
    ------------------------

    ``sabre_starting_layouts`` (``list[Layout]``)
        An optional list of :class:`~.Layout` objects to use for additional layout trials. This is
        in addition to the full random trials specified with the ``layout_trials`` argument.

    Property Set Values Written
    ---------------------------

    ``layout`` (:class:`.Layout`)
        The chosen initial mapping of virtual to physical qubits, including the ancilla allocation.

    ``final_layout`` (:class:`.Layout`)
        A permutation of how swaps have been applied to the input qubits at the end of the circuit.

    **References:**

    [1] Henry Zou and Matthew Treinish and Kevin Hartman and Alexander Ivrii and Jake Lishman.
    "LightSABRE: A Lightweight and Enhanced SABRE Algorithm"
    `arXiv:2409.08368 <https://doi.org/10.48550/arXiv.2409.08368>`__
    [2] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
    for NISQ-era quantum devices." ASPLOS 2019.
    `arXiv:1809.02573 <https://arxiv.org/pdf/1809.02573.pdf>`_
    """

    def __init__(
        self,
        coupling_map,
        routing_pass=None,
        seed=None,
        max_iterations=3,
        swap_trials=None,
        layout_trials=None,
        skip_routing=False,
    ):
        """SabreLayout initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): directed graph representing a coupling map.
            routing_pass (BasePass): the routing pass to use while iterating.
                If specified this pass operates as an :class:`~.AnalysisPass` and
                will only populate the ``layout`` field in the property set and
                the input dag is returned unmodified. This argument is mutually
                exclusive with the ``swap_trials`` and the ``layout_trials``
                arguments and if this is specified at the same time as either
                argument an error will be raised.
            seed (int): seed for setting a random first trial layout.
            max_iterations (int): number of forward-backward iterations.
            swap_trials (int): The number of trials to run of
                :class:`~.SabreSwap` for each iteration. This is equivalent to
                the ``trials`` argument on :class:`~.SabreSwap`. If this is not
                specified (and ``routing_pass`` isn't set) by default the number
                of physical CPUs on your local system will be used. For
                reproducibility between environments it is best to set this
                to an explicit number because the output will potentially depend
                on the number of trials run. This option is mutually exclusive
                with the ``routing_pass`` argument and an error will be raised
                if both are used.
            layout_trials (int): The number of random seed trials to run
                layout with. When > 1 the trial that results in the output with
                the fewest swap gates will be selected. If this is not specified
                (and ``routing_pass`` is not set) then the number of local
                physical CPUs will be used as the default value. This option is
                mutually exclusive with the ``routing_pass`` argument and an error
                will be raised if both are used. An additional 3 or 4 trials
                depending on the ``coupling_map`` value are run with common layouts
                on top of the random trial count specified by this value.
            skip_routing (bool): If this is set ``True`` and ``routing_pass`` is not used
                then routing will not be applied to the output circuit.  Only the layout
                will be set in the property set. This is a tradeoff to run custom
                routing with multiple layout trials, as using this option will cause
                SabreLayout to run the routing stage internally but not use that result.

        Raises:
            TranspilerError: If both ``routing_pass`` and ``swap_trials`` or
            both ``routing_pass`` and ``layout_trials`` are specified
        """
        super().__init__()
        if isinstance(coupling_map, Target) and not isinstance(coupling_map, _FakeTarget):
            self.target = coupling_map
            self._coupling_map = None
        else:
            coupling_map = (
                coupling_map.build_coupling_map()
                if isinstance(coupling_map, _FakeTarget)
                # We assume we can mutate this to make it symmetric.
                else copy.deepcopy(coupling_map)
            )
            # A dummy target purely to represent a homogeneous coupling graph for layout purposes.
            self.target = Target.from_configuration(
                basis_gates=["u", "cx"], coupling_map=coupling_map
            )
            self._coupling_map = coupling_map
            if self._coupling_map is not None:
                self._coupling_map.make_symmetric()
        if routing_pass is not None and (swap_trials is not None or layout_trials is not None):
            raise TranspilerError(
                "The 'routing_pass' argument cannot be set alongside 'swap_trials' or 'layout_trials'."
            )
        self.routing_pass = routing_pass
        self.seed = seed
        self.max_iterations = max_iterations
        self.swap_trials = default_num_processes() if swap_trials is None else swap_trials
        self.layout_trials = default_num_processes() if layout_trials is None else layout_trials
        self.skip_routing = skip_routing

    @property
    def coupling_map(self):
        # This property is not intended to be public API, it just keeps backwards compatibility.
        if self._coupling_map is None:
            self._coupling_map = self.target.build_coupling_map()
            if self._coupling_map is not None:
                self._coupling_map.make_symmetric()
        return self._coupling_map

    def run(self, dag):
        """Run the SabreLayout pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to find layout for.

        Returns:
           DAGCircuit: The output dag if swap mapping was run
            (otherwise the input dag is returned unmodified).

        Raises:
            TranspilerError: if dag wider than the target.
        """
        if self.target.num_qubits is None:
            raise TranspilerError("given 'Target' was not initialized with a qubit count")
        if len(dag.qubits) > self.target.num_qubits:
            raise TranspilerError(
                f"More virtual qubits ({len(dag.qubits)}) exist"
                f" than physical ({self.target.num_qubits})."
            )

        if self.routing_pass is not None:
            if not self.coupling_map.is_connected():
                raise TranspilerError(
                    "The routing_pass argument cannot be used with disjoint coupling maps."
                )
            if self.seed is None:
                seed = np.random.randint(0, np.iinfo(np.int32).max)
            else:
                seed = self.seed
            rng = np.random.default_rng(seed)

            physical_qubits = rng.choice(self.coupling_map.size(), len(dag.qubits), replace=False)
            physical_qubits = rng.permutation(physical_qubits)
            initial_layout = Layout({q: dag.qubits[i] for i, q in enumerate(physical_qubits)})

            self.routing_pass.fake_run = True

            # Do forward-backward iterations.
            circ = dag_to_circuit(dag)
            rev_circ = circ.reverse_ops()
            for _ in range(self.max_iterations):
                for _ in ("forward", "backward"):
                    pm = self._layout_and_route_passmanager(initial_layout)
                    new_circ = pm.run(circ)

                    # Update initial layout and reverse the unmapped circuit.
                    pass_final_layout = pm.property_set["final_layout"]
                    final_layout = self._compose_layouts(
                        initial_layout, pass_final_layout, new_circ.qregs
                    )
                    initial_layout = final_layout
                    circ, rev_circ = rev_circ, circ
                logger.info("new initial layout: %s", initial_layout)

            for qreg in dag.qregs.values():
                initial_layout.add_register(qreg)
            self.property_set["layout"] = initial_layout
            self.routing_pass.fake_run = False
            return dag

        # Combined
        dag_indices = {qubit: i for i, qubit in enumerate(dag.qubits)}

        def to_partial_layout(layout):
            out = [None] * dag.num_qubits()
            for virt, phys in layout.get_virtual_bits().items():
                if (pos := dag_indices.get(virt, None)) is not None:
                    out[pos] = phys
            return out

        starting_layouts = [
            to_partial_layout(layout)
            for layout in self.property_set.get("sabre_starting_layouts", [])
        ]
        heuristic = (
            Heuristic(attempt_limit=10 * self.target.num_qubits)
            .with_basic(1.0, SetScaling.Constant)
            .with_lookahead([0.5 / self.target.num_qubits], SetScaling.Constant)
            .with_decay(0.001, 5)
        )
        sabre_start = time.perf_counter()
        # If `skip_routing`, then `out_dag` and `final` are meaningless but well-typed.
        out_dag, initial, final = sabre_layout_and_routing(
            dag,
            self.target,
            heuristic,
            max_iterations=self.max_iterations,
            num_swap_trials=self.swap_trials or 1,
            num_random_trials=self.layout_trials,
            seed=self.seed,
            partial_layouts=starting_layouts,
            skip_routing=self.skip_routing,
        )
        sabre_stop = time.perf_counter()
        logger.debug(
            "Sabre layout algorithm execution for all components complete in: %s sec.",
            sabre_stop - sabre_start,
        )

        if self.skip_routing:
            virtuals = list(dag.qubits)
            # In `skip_routing` mode, subsequent passes don't expect ancilla expansion.
            initial_layout = Layout(
                {p: virtuals[v] for v, p in initial.layout_mapping() if v < len(virtuals)}
            )
            for register in dag.qregs.values():
                initial_layout.add_register(register)
            self.property_set["layout"] = initial_layout
            self.property_set["original_qubit_indices"] = {
                qubit: i for i, qubit in enumerate(virtuals)
            }
            return dag

        ancilla_register_name = "ancilla"
        ancilla_suffix = 0
        while ancilla_register_name in dag.qregs:
            ancilla_register_name = f"{ancilla_register_name}{ancilla_suffix}"
            ancilla_suffix += 1

        ancillas = QuantumRegister(self.target.num_qubits - dag.num_qubits(), ancilla_register_name)
        virtuals = list(dag.qubits) + list(ancillas)
        initial_layout = Layout({p: virtuals[v] for v, p in initial.layout_mapping()})
        for register in dag.qregs.values():
            initial_layout.add_register(register)
        if ancillas:
            initial_layout.add_register(ancillas)
        self.property_set["layout"] = initial_layout
        self.property_set["original_qubit_indices"] = {qubit: i for i, qubit in enumerate(virtuals)}

        final_layout = Layout(
            {out_dag.qubits[initial.virtual_to_physical(v)]: p for v, p in final.layout_mapping()}
        )
        if (prev_final_layout := self.property_set.get("final_layout", None)) is None:
            self.property_set["final_layout"] = final_layout
        else:
            # The "final layout" can be thought of as a "comes from" permutation that you apply at
            # the end of the circuit to invert the routing.  So if there's an existing one, what we
            # apply at the end of the circuit needs to set the circuit qubits so they "come from"
            # the previous one, then those "come from" the one we've just added.
            self.property_set["final_layout"] = prev_final_layout.compose(
                final_layout, out_dag.qubits
            )
        return out_dag

    def _layout_and_route_passmanager(self, initial_layout):
        """Return a passmanager for a full layout and routing.

        We use a factory to remove potential statefulness of passes.
        """
        layout_and_route = [
            SetLayout(initial_layout),
            FullAncillaAllocation(self.coupling_map),
            EnlargeWithAncilla(),
            ApplyLayout(),
            self.routing_pass,
        ]
        pm = PassManager(layout_and_route)
        return pm

    def _compose_layouts(self, initial_layout, pass_final_layout, qregs):
        """Return the real final_layout resulting from the composition
        of an initial_layout with the final_layout reported by a pass.

        The routing passes internally start with a trivial layout, as the
        layout gets applied to the circuit prior to running them. So the
        ``"final_layout"`` they report must be amended to account for the actual
        initial_layout that was selected.
        """
        trivial_layout = Layout.generate_trivial_layout(*qregs)
        qubit_map = Layout.combine_into_edge_map(initial_layout, trivial_layout)
        final_layout = {v: pass_final_layout._v2p[qubit_map[v]] for v in initial_layout._v2p}
        return Layout(final_layout)
