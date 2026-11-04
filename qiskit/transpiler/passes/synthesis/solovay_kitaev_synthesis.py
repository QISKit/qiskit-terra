# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
===============================
Solovay-Kitaev Synthesis Plugin
===============================

.. autosummary::
   :toctree: ../stubs/

   SolovayKitaevSynthesis
"""

from __future__ import annotations

import numpy as np

from qiskit.converters import circuit_to_dag
from qiskit.circuit.gate import Gate
from qiskit.dagcircuit import DAGCircuit
from qiskit.synthesis.discrete_basis.solovay_kitaev import SolovayKitaevDecomposition
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils.control_flow import trivial_recurse

from .plugin import UnitarySynthesisPlugin


class SolovayKitaev(TransformationPass):
    r"""Approximately decompose 1q gates to a discrete basis using the Solovay-Kitaev algorithm.

    The Solovay-Kitaev theorem [1] states that any single qubit gate can be approximated to
    arbitrary precision by a set of fixed single-qubit gates, if the set generates a dense
    subset in :math:`SU(2)`. This is an important result, since it means that any single-qubit
    gate can be expressed in terms of a discrete, universal gate set that we know how to implement
    fault-tolerantly. Therefore, the Solovay-Kitaev algorithm allows us to take any
    non-fault tolerant circuit and rephrase it in a fault-tolerant manner.

    This implementation of the Solovay-Kitaev algorithm is based on [2].

    For example, the following circuit

    .. code-block:: text

             ┌─────────┐
        q_0: ┤ RX(0.8) ├
             └─────────┘

    can be decomposed into

    .. code-block:: text

        global phase: 7π/8
             ┌───┐┌───┐┌───┐
        q_0: ┤ H ├┤ T ├┤ H ├
             └───┘└───┘└───┘

    with an L2-error of approximately 0.01.

    Examples:

        Per default, the basis gate set is ``["t", "tdg", "h"]``:

        .. plot::
           :include-source:
           :nofigs:

            import numpy as np
            from qiskit.circuit import QuantumCircuit
            from qiskit.transpiler.passes.synthesis import SolovayKitaev
            from qiskit.quantum_info import Operator

            circuit = QuantumCircuit(1)
            circuit.rx(0.8, 0)

            print("Original circuit:")
            print(circuit.draw())

            skd = SolovayKitaev(recursion_degree=2)

            discretized = skd(circuit)

            print("Discretized circuit:")
            print(discretized.draw())

            print("Error:", np.linalg.norm(Operator(circuit).data - Operator(discretized).data))

        .. code-block:: text

            Original circuit:
               ┌─────────┐
            q: ┤ Rx(0.8) ├
               └─────────┘
            Discretized circuit:
            global phase: 7π/8
               ┌───┐┌───┐┌───┐
            q: ┤ H ├┤ T ├┤ H ├
               └───┘└───┘└───┘
            Error: 2.828408279166474

        Individual basis gate sets can be specified in the initializer.

        .. plot::
           :include-source:
           :nofigs:

            from qiskit.transpiler.passes import SolovayKitaev

            basis = ["s", "sdg", "t", "tdg", "z", "h"]

            skd = SolovayKitaev(recursion_degree=2, basis_gates=basis)

        To generate and store basic approximations in between different instances, the
        :class:`.SolovayKitaevDecomposition` and its
        :meth:`~.SolovayKitaevDecomposition.save_basic_approximations` method can be used.

        .. code-block:: python

            from qiskit.transpiler.passes import SolovayKitaev
            from qiskit.synthesis import SolovayKitaevDecomposition

            # generate basic approximations
            basis = ["s", "sdg", "t", "tdg", "z", "h"]
            decomp = SolovayKitaevDecomposition(basis_gates=basis, depth=5)

            # store them in a local file
            fname = "sk_approx.bin"
            decomp.save_basic_approximations(fname)

            # load them for running Solovay-Kitaev
            skd = SolovayKitaev(recursion_degree=2, basic_approximations=fname)


    References:

    [1] Kitaev, A Yu (1997). Quantum computations: algorithms and error correction.
    Russian Mathematical Surveys. 52 (6): 1191–1249.
    `Online <https://iopscience.iop.org/article/10.1070/RM1997v052n06ABEH002155>`_.

    [2] Dawson, Christopher M.; Nielsen, Michael A. (2005) The Solovay-Kitaev Algorithm.
    `arXiv:quant-ph/0505030 <https://arxiv.org/abs/quant-ph/0505030>`_.

    """

    def __init__(
        self,
        recursion_degree: int = 5,
        basic_approximations: str | dict[str, np.ndarray] | None = None,
        *,
        basis_gates: list[str | Gate] | None = None,
        depth: int = 12,
    ) -> None:
        r"""
        Args:
            recursion_degree: The recursion depth for the Solovay-Kitaev algorithm.
                A larger recursion depth increases the accuracy and length of the
                decomposition.
            basic_approximations: The basic approximations for the finding the best discrete
                decomposition at the root of the recursion. If a string, it specifies the
                file to load the approximations from. If a dictionary, it contains
                ``{label: SO(3)-matrix}`` pairs. If ``None``, a default based on the :math:`H`,
                :math:`T` and :math:`T^\dagger` gates up to depth 16 is generated.
                Note that if ``basic_approximations`` is passed, ``basis_gates`` and
                ``depth`` cannot be set.
            basis_gates: The basis gates used to build the net of basic approximations.
                Defaults to ``["h", "t", "tdg"]``. This argument cannot be set if
                ``basic_approximations`` is provided.
            depth: The maximal gate depth used in basic approximations. This argument cannot be
                set if ``basic_approximations`` is provided.
        """
        super().__init__()
        self.recursion_degree = recursion_degree
        self._sk = SolovayKitaevDecomposition(
            basic_approximations, basis_gates=basis_gates, depth=depth
        )

    @trivial_recurse
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the ``SolovayKitaev`` pass on `dag`.

        Args:
            dag: The input dag.

        Returns:
            Output dag with 1q gates synthesized in the discrete target basis.

        Raises:
            TranspilerError: if a gate does not have to_matrix
        """
        for node in dag.op_nodes():
            # ignore operations on which the algorithm cannot run
            if (
                (node.op.num_qubits != 1)
                or node.is_parameterized()
                or (not hasattr(node.op, "to_matrix"))
            ):
                continue

            # we do not check the input matrix as we know it comes from a Qiskit gate, as this
            # we know it will generate a valid SU(2) matrix
            check_input = not isinstance(node.op, Gate)

            # call solovay kitaev
            approximation = self._sk.run(
                node.op, self.recursion_degree, return_dag=True, check_input=check_input
            )

            # convert to a dag and replace the gate by the approximation
            dag.substitute_node_with_dag(node, approximation)

        return dag


class SolovayKitaevSynthesis(UnitarySynthesisPlugin):
    """A Solovay-Kitaev Qiskit unitary synthesis plugin.

    This plugin is invoked by :func:`~.compiler.transpile` when the ``unitary_synthesis_method``
    parameter is set to ``"sk"``.

    This plugin supports customization and additional parameters can be passed to the plugin
    by passing a dictionary as the ``unitary_synthesis_plugin_config`` parameter of
    the :func:`~qiskit.compiler.transpile` function.

    Supported parameters in the dictionary:

    basic_approximations (str | dict):
        The basic approximations for the finding the best discrete decomposition at the root of the
        recursion. If a string, it specifies the ``.npy`` file to load the approximations from.
        If a dictionary, it contains ``{label: SO(3)-matrix}`` pairs. If None, a default based on
        the specified ``basis_gates`` and ``depth`` is generated.

    basis_gates (list):
        A list of strings specifying the discrete basis gates to decompose to. If None,
        it defaults to ``["h", "t", "tdg"]``. If ``basic_approximations`` is not None,
        ``basis_set`` is required to correspond to the basis set that was used to
        generate it.

    depth (int):
        The gate-depth of the basic approximations. All possible, unique combinations of the
        basis gates up to length ``depth`` are considered. If None, defaults to 12.
        If ``basic_approximations`` is not None, ``depth`` is required to correspond to the
        depth that was used to generate it.

    recursion_degree (int):
        The number of times the decomposition is recursively improved. If None, defaults to 5.
    """

    # Generating basic approximations of single-qubit gates is computationally expensive.
    # We cache the instance of the Solovay-Kitaev class (which contains the approximations),
    # as well as the basis gates and the depth (used to generate it).
    # When the plugin is called again, we check if the specified basis gates and depth are
    # the same as before. If so, the stored basic approximations are reused, and if not, the
    # approximations are re-generated. In practice (when the plugin is run as a part of the
    # UnitarySynthesis transpiler pass), the basis gates and the depth do not change, and
    # basic approximations are not re-generated.
    _sk = None
    _basis_gates = None
    _depth = None

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
        ``["h", "t", "tdg"]`` gate basis."""
        return True

    @property
    def supports_coupling_map(self):
        """The plugin does not support coupling maps."""
        return False

    def run(self, unitary, **options):
        """Run the SolovayKitaevSynthesis synthesis plugin on the given unitary."""

        config = options.get("config") or {}
        basis_gates = options.get("basis_gates", None)
        depth = config.get("depth", 12)
        basic_approximations = config.get("basic_approximations", None)
        recursion_degree = config.get("recursion_degree", 5)

        # Check if we didn't yet construct the Solovay-Kitaev instance (which contains the basic
        # approximations) or if the basic approximations need to be recomputed.
        if (SolovayKitaevSynthesis._sk is None) or (
            (basis_gates != SolovayKitaevSynthesis._basis_gates)
            or (depth != SolovayKitaevSynthesis._depth)
        ):
            SolovayKitaevSynthesis._basis_gates = basis_gates
            SolovayKitaevSynthesis._depth = depth
            SolovayKitaevSynthesis._sk = SolovayKitaevDecomposition(
                basic_approximations, basis_gates=basis_gates, depth=depth
            )
        approximate_circuit = SolovayKitaevSynthesis._sk.run(unitary, recursion_degree)
        return circuit_to_dag(approximate_circuit)
