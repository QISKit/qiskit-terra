# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Creating Sabre starting layouts."""

from __future__ import annotations

import itertools

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout
from qiskit.transpiler.target import Target
from qiskit._accelerate.error_map import ErrorMap


class SabrePreLayout(AnalysisPass):
    """Choose a starting layout to use for additional Sabre layout trials.

    The pass works by augmenting the coupling map with more and more "extra" edges
    until VF2 succeeds to find a perfect graph isomorphism. More precisely, the
    augmented coupling map contains edges between nodes that are within a given
    distance ``d`` in the original coupling map. The original edges are noise-free
    while the additional edges have noise that scales exponentially with the distance.
    The value of ``d`` is increased until an isomorphism is found.

    Intuitively, a better VF2 layout involves fewer and of shorter-distance extra edges.

    Property Set Values Written
    ---------------------------

    ``sabre_starting_layouts`` (``list[Layout]``)
        An optional list of :class:`~.Layout` objects to use for additional Sabre layout trials.

    **References:**

    [1] Henry Zou and Matthew Treinish and Kevin Hartman and Alexander Ivrii and Jake Lishman.
    "LightSABRE: A Lightweight and Enhanced SABRE Algorithm"
    `arXiv:2409.08368 <https://doi.org/10.48550/arXiv.2409.08368>`__
    """

    def __init__(
        self,
        coupling_map: CouplingMap | Target,
        max_distance: int = 2,
        error_rate: float = 0.1,
        max_trials_vf2: int | None = 100,
        call_limit_vf2: None | int | tuple[int | None, int | None] = None,
        improve_layout: bool = True,
        min_distance: int = 1,
    ):
        """
        Args:
            coupling_map: Directed graph representing the original coupling map or a target modelling
                the backend (including its connectivity).
            max_distance: The maximum distance for running VF2 with the augmented coupling
                map. In particular, this also specifies the maximum distance between the original nodes
                that become connected in the augmented coupling map.
            error_rate: The error rate to assign to the "extra" edges. A non-zero
                error rate prioritizes VF2 to choose original edges over extra edges.
            max_trials_vf2: Specifies the maximum number of VF2 trials. This option remains primarily
                for legacy reasons since the introduction of on-the-fly scoring in VF2, which was
                introduced in Qiskit 2.3. To bound the time for the pass, set parameters ``max_distance``
                and ``call_limit_vf2`` instead.
            call_limit_vf2: The maximum number of times that the inner VF2 isomorphism search will
                attempt to extend the mapping. If ``None``, then no limit.  If a 2-tuple, then the
                limit starts as the first item, and swaps to the second after the first match is found,
                without resetting the number of steps taken.  This can be used to allow a long search
                for any mapping, but still terminate quickly with a small extension budget if one is
                found.
            improve_layout: Unused (the option became obsolete with the introduction of on-the-fly
                scoring in VF2).
            min_distance: The distance for the first VF2 run with the augmented coupling map. Setting
                ``min_distance > 1`` skips all smaller-distance checks, and in particular skips the
                distance-1 check which corresponds to running the ``VF2Layout`` pass.

        Raises:
            TranspilerError: At runtime, if the argument ``coupling_map`` is not provided.
        """

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.error_rate = error_rate
        self.max_trials_vf2 = max_trials_vf2
        self.call_limit_vf2 = call_limit_vf2
        self.improve_layout = improve_layout

        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map

        super().__init__()

    def run(self, dag):
        """Run the SabrePreLayout pass on `dag`.

        The discovered starting layout is written to the property set
        value ``sabre_starting_layouts``.

        Args:
            dag (DAGCircuit): DAG to create starting layout for.
        """

        if self.coupling_map is None:
            raise TranspilerError(
                "SabrePreLayout requires coupling_map to be used with either "
                "CouplingMap or a Target."
            )

        starting_layout = None
        cur_distance = self.min_distance
        while cur_distance <= self.max_distance:
            augmented_map, augmented_error_map = self._add_extra_edges(cur_distance)
            pass_ = VF2Layout(
                augmented_map,
                seed=-1,
                max_trials=self.max_trials_vf2,
                call_limit=self.call_limit_vf2,
            )
            pass_.property_set["vf2_avg_error_map"] = augmented_error_map
            pass_.run(dag)

            if "layout" in pass_.property_set:
                starting_layout = pass_.property_set["layout"]
                break

            cur_distance += 1

        if cur_distance > 1 and starting_layout is not None:
            # write discovered layout into the property set
            if "sabre_starting_layouts" not in self.property_set:
                self.property_set["sabre_starting_layouts"] = [starting_layout]
            else:
                self.property_set["sabre_starting_layouts"].append(starting_layout)

    def _add_extra_edges(self, distance):
        """Augments the coupling map with extra edges that connect nodes ``distance``
        apart in the original graph. The extra edges are assigned errors allowing VF2
        to prioritize real edges over extra edges.
        """
        nq = len(self.coupling_map.graph)
        augmented_coupling_map = CouplingMap()
        augmented_coupling_map.graph = self.coupling_map.graph.copy()
        augmented_error_map = ErrorMap(nq)

        for x, y in itertools.combinations(self.coupling_map.graph.node_indices(), 2):
            d = self.coupling_map.distance(x, y)
            if 1 < d <= distance:
                error_rate = 1 - ((1 - self.error_rate) ** d)
                augmented_coupling_map.add_edge(x, y)
                augmented_error_map.add_error((x, y), error_rate)
                augmented_coupling_map.add_edge(y, x)
                augmented_error_map.add_error((y, x), error_rate)

        return augmented_coupling_map, augmented_error_map
