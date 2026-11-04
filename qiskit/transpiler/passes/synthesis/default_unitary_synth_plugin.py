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

"""
================================
Default Unitary Synthesis Plugin
================================

The default unitary synthesis plugin defines the default algorithm used by
:class:`.UnitarySynthesis` transpiler pass to synthesize unitary
gates in the circuit. Its behavior depends on the number of qubits in the
target unitary and the target basis.

Continuous basis sets
---------------------

When the target basis is continuous (i.e. not Clifford+T), the plugin uses the
following decompositions:

* 1-qubit gates: Euler angle decompositions (see :class:`.OneQubitEulerDecomposer`).
* 2-qubit gates: KAK/Cartan decompositions (see :class:`.TwoQubitBasisDecomposer`,
  :class:`.TwoQubitControlledUDecomposer`, :class:`.XXDecomposer`).
* 3+ qubit gates: Quantum Shannon Decomposition, see :func:`.qs_decomposition`.

For 1- and 2-qubit gates, the plugin uses the basis error information from the target
to select the decomposition that maximizes the expected fidelity.

Clifford+T basis sets
---------------------

When the target basis is Clifford+T, the plugin uses the Solovay-Kitaev algorithm (see
:class:`.SolovayKitaevDecomposition`) to approximate 1-qubit unitaries. The
settings used to create basic approximations are ``basis_gates=["h", "t", "tdg"]``,
``depth=12`` and ``recursion_degree=5``, and were chosen empirically to balance
approximation quality with computational cost. To use Solovay-Kitaev with custom parameters,
or to avoid recomputing approximations for multiple circuits, you can invoke
:class:`.SolovayKitaevSynthesis` unitary synthesis plugin instead of this default plugin.

At present, Qiskit does not include algorithms for approximating 2+ qubit unitaries
directly in the Clifford+T basis set. Therefore, the preset pass manager first decomposes
such gates into ``["cx", "u"]`` basis, and then applies Solovay-Kitaev to the resulting
1-qubit unitary gates.

.. autosummary::
   :toctree: ../stubs/

   DefaultUnitarySynthesis
"""

from __future__ import annotations

from qiskit.transpiler.passes.synthesis import plugin
from qiskit._accelerate.unitary_synthesis import synthesize_unitary_matrix


class DefaultUnitarySynthesis(plugin.UnitarySynthesisPlugin):
    """The default unitary synthesis plugin."""

    @property
    def supports_basis_gates(self):
        return True

    @property
    def supports_coupling_map(self):
        return True

    @property
    def supports_natural_direction(self):
        return True

    @property
    def supports_pulse_optimize(self):
        return True

    @property
    def supports_gate_lengths(self):
        return False

    @property
    def supports_gate_errors(self):
        return False

    @property
    def supports_gate_lengths_by_qubit(self):
        return True

    @property
    def supports_gate_errors_by_qubit(self):
        return True

    @property
    def max_qubits(self):
        return None

    @property
    def min_qubits(self):
        return None

    @property
    def supported_bases(self):
        return None

    @property
    def supports_target(self):
        return True

    def run(self, unitary, **options):
        # Approximation degree is set directly as an attribute on the
        # instance by the UnitarySynthesis pass here as it's not part of
        # plugin interface. However if for some reason it's not set assume
        # it's 1.
        approximation_degree = getattr(self, "_approximation_degree", 1.0)
        basis_gates = options["basis_gates"]
        coupling_map = options["coupling_map"][0]
        natural_direction = options["natural_direction"]
        pulse_optimize = options["pulse_optimize"]
        qubits = options["coupling_map"][1]
        target = options["target"]

        # The options "gate_lengths_by_qubit" and "gate_errors_by_qubit" should
        # be subsumed by target.

        _coupling_edges = set(coupling_map.get_edges()) if coupling_map is not None else set()

        synth_dag = synthesize_unitary_matrix(
            unitary,
            qubits,
            target,
            basis_gates,
            _coupling_edges,
            approximation_degree,
            natural_direction,
            pulse_optimize,
        )
        return synth_dag
