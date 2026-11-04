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
==============================
Ross-Selinger Synthesis Plugin
==============================

.. autosummary::
   :toctree: ../stubs/

   RossSelingerSynthesis
"""

from __future__ import annotations

from qiskit.converters import circuit_to_dag
from qiskit.synthesis import gridsynth_unitary

from .plugin import UnitarySynthesisPlugin


class RossSelingerSynthesis(UnitarySynthesisPlugin):
    """A Ross-Selinger Qiskit unitary synthesis plugin.

    The algorithm is described in [1]. The source code (in Rust) is available at
    https://github.com/qiskit-community/rsgridsynth.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"gridsynth"``.

    This plugin supports customization and additional parameters can be passed to the plugin
    by passing a dictionary as the ``unitary_synthesis_plugin_config`` parameter of
    the :func:`~qiskit.compiler.transpile` function.

    Supported parameters in the dictionary:

    epsilon (f64):
        The allowed approximation error.

    References:

    [1] Neil J. Ross, Peter Selinger, Optimal ancilla-free Clifford+T approximation of z-rotations,
        `arXiv:1403.2975 <https://arxiv.org/pdf/1403.2975>`_

    """

    @property
    def max_qubits(self):
        """Maximum number of supported qubits is ``1``."""
        return 1

    @property
    def min_qubits(self):
        """Minimum number of supported qubits is ``1``."""
        return 1

    @property
    def supports_natural_direction(self):
        """The plugin does not support natural direction, it does not assume
        bidirectional two qubit gates."""
        return True

    @property
    def supports_pulse_optimize(self):
        """The plugin does not support optimization of pulses."""
        return False

    @property
    def supports_gate_lengths(self):
        """The plugin does not support gate lengths."""
        return False

    @property
    def supports_gate_errors(self):
        """The plugin does not support gate errors."""
        return False

    @property
    def supported_bases(self):
        """The plugin does not support bases for synthesis."""
        return None

    @property
    def supports_basis_gates(self):
        """The plugin does not support basis gates. By default it synthesizes to the
        ``["h", "s", "t", "x"]`` gate basis."""
        return False

    @property
    def supports_coupling_map(self):
        """The plugin does not support coupling maps."""
        return False

    def run(self, unitary, **options):
        """Run the Ross-Selinger synthesis plugin on the given unitary."""
        # ToDo: possibly we should use the approximation_degree instead,
        # and compute epsilon based on that.
        if (config := options.get("config")) is not None:
            epsilon = config.get("epsilon", 1e-10)
        else:
            epsilon = 1e-10

        approximate_circuit = gridsynth_unitary(unitary, epsilon)
        return circuit_to_dag(approximate_circuit)
