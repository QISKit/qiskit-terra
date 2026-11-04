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

"""This module contains common utils for disjoint coupling maps."""
from __future__ import annotations


from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.target import Target
from qiskit.transpiler.exceptions import TranspilerError


def require_layout_isolated_to_component(dag: DAGCircuit, components_source: Target | CouplingMap):
    """
    Check that the layout of the dag does not require connectivity across connected components
    in the CouplingMap

    Args:
        dag: DAGCircuit to check.
        components_source: Target to check against.

    Raises:
        TranspilerError: Chosen layout is not valid for the target disjoint connectivity.
    """
    if isinstance(components_source, Target):
        coupling_map = components_source.build_coupling_map(filter_idle_qubits=True)
    else:
        coupling_map = components_source
    component_sets = [set(x.graph.nodes()) for x in coupling_map.connected_components()]
    for inst in dag.two_qubit_ops():
        component_index = None
        for i, component_set in enumerate(component_sets):
            if dag.find_bit(inst.qargs[0]).index in component_set:
                component_index = i
                break
        if dag.find_bit(inst.qargs[1]).index not in component_sets[component_index]:
            raise TranspilerError(
                "The circuit has an invalid layout as two qubits need to interact in disconnected "
                "components of the coupling map. The physical qubit "
                f"{dag.find_bit(inst.qargs[1]).index} needs to interact with the "
                f"qubit {dag.find_bit(inst.qargs[0]).index} and they belong to different components"
            )
