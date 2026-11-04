# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=================================
Clifford Unitary Synthesis Plugin
=================================

.. autosummary::
   :toctree: ../stubs/

   CliffordUnitarySynthesis
"""

from __future__ import annotations

import math

from qiskit.exceptions import QiskitError
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, Clifford
from qiskit.quantum_info.operators.predicates import matrix_equal

from .plugin import UnitarySynthesisPlugin


class CliffordUnitarySynthesis(UnitarySynthesisPlugin):
    """A Clifford unitary synthesis plugin.

    The plugin is invoked by the :class:`.UnitarySynthesis` transpiler pass
    when the parameter ``method`` is set to ``"clifford"``.

    The plugin checks if the given unitary can be represented by a Clifford,
    in which case it returns a circuit implementing this unitary and
    consisting only of Clifford gates.

    In addition, the parameter ``plugin_config`` of :class:`.UnitarySynthesis`
    can be used to pass the following plugin-specific parameters:

    * min_qubits: the minimum number of qubits to consider (the default value is 1).

    * max_qubits: the maximum number of qubits to consider (the default value is 3).

    """

    @property
    def max_qubits(self):
        return None

    @property
    def min_qubits(self):
        return None

    @property
    def supports_basis_gates(self):
        return False

    @property
    def supports_coupling_map(self):
        return False

    @property
    def supports_natural_direction(self):
        return False

    @property
    def supports_pulse_optimize(self):
        return False

    @property
    def supports_gate_lengths(self):
        return False

    @property
    def supports_gate_errors(self):
        return False

    @property
    def supported_bases(self):
        return None

    def run(self, unitary, **options):
        """Run the CliffordUnitarySynthesis plugin on the given unitary."""

        config = options.get("config") or {}

        min_qubits = config.get("min_qubits", 1)
        max_qubits = config.get("max_qubits", 3)

        num_qubits = int(math.log2(unitary.shape[0]))

        dag = None
        if min_qubits <= num_qubits <= max_qubits:
            try:
                # Attempts to construct a Clifford from a unitary
                # (raises a QiskitError if this is not possible)
                cliff = Clifford.from_matrix(unitary)

                circuit = cliff.to_circuit()

                # Constructing Clifford from a unitary discards the global phase,
                # so we need to reconstruct it.
                props = {}
                if not matrix_equal(
                    unitary, Operator(circuit)._data, ignore_phase=True, props=props
                ):
                    raise QiskitError("Clifford synthesis is incorrect")
                circuit.global_phase -= props["phase_difference"]

                dag = circuit_to_dag(circuit)
            except QiskitError:
                pass

        return dag
