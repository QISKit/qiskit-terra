# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


r"""

High Level Synthesis Plugins
-----------------------------

Clifford Synthesis
''''''''''''''''''

.. list-table:: Plugins for :class:`qiskit.quantum_info.Clifford` (key = ``"clifford"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"ag"``
      - :class:`~.AGSynthesisClifford`
      - all-to-all
      - greedily optimizes CX-count
    * - ``"bm"``
      - :class:`~.BMSynthesisClifford`
      - all-to-all
      - optimal count for :math:`n \in \{2,3\}`; used in ``"default"`` for :math:`n \in \{2,3\}`
    * - ``"greedy"``
      - :class:`~.GreedySynthesisClifford`
      - all-to-all
      - greedily optimizes CX-count; used in ``"default"`` for :math:`n\geq 4`
    * - ``"layers"``
      - :class:`~.LayerSynthesisClifford`
      - all-to-all
      -
    * - ``"lnn"``
      - :class:`~.LayerLnnSynthesisClifford`
      - linear
      - many CX-gates but guarantees CX-depth of at most :math:`7n+2`
    * - ``"default"``
      - :class:`~.DefaultSynthesisClifford`
      - all-to-all
      - usually best for optimizing CX-count (and optimal CX-count for :math:`n \in \{2,3\}`)

.. autosummary::
   :toctree: ../stubs/

   AGSynthesisClifford
   BMSynthesisClifford
   GreedySynthesisClifford
   LayerSynthesisClifford
   LayerLnnSynthesisClifford
   DefaultSynthesisClifford


Linear Function Synthesis
'''''''''''''''''''''''''

.. list-table:: Plugins for :class:`.LinearFunction` (key = ``"linear"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"kms"``
      - :class:`~.KMSSynthesisLinearFunction`
      - linear
      - many CX-gates but guarantees CX-depth of at most :math:`5n`
    * - ``"pmh"``
      - :class:`~.PMHSynthesisLinearFunction`
      - all-to-all
      - greedily optimizes CX-count; used in ``"default"``
    * - ``"default"``
      - :class:`~.DefaultSynthesisLinearFunction`
      - all-to-all
      - best for optimizing CX-count

.. autosummary::
   :toctree: ../stubs/

   KMSSynthesisLinearFunction
   PMHSynthesisLinearFunction
   DefaultSynthesisLinearFunction


Permutation Synthesis
'''''''''''''''''''''

.. list-table:: Plugins for :class:`.PermutationGate` (key = ``"permutation"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
      - Description
    * - ``"basic"``
      - :class:`~.BasicSynthesisPermutation`
      - all-to-all
      - optimal SWAP-count; used in ``"default"``
    * - ``"acg"``
      - :class:`~.ACGSynthesisPermutation`
      - all-to-all
      - guarantees SWAP-depth of at most :math:`2`
    * - ``"kms"``
      - :class:`~.KMSSynthesisPermutation`
      - linear
      - many SWAP-gates, but guarantees SWAP-depth of at most :math:`n`
    * - ``"token_swapper"``
      - :class:`~.TokenSwapperSynthesisPermutation`
      - any
      - greedily optimizes SWAP-count for arbitrary connectivity
    * - ``"default"``
      - :class:`~.BasicSynthesisPermutation`
      - all-to-all
      - best for optimizing SWAP-count

.. autosummary::
   :toctree: ../stubs/

   BasicSynthesisPermutation
   ACGSynthesisPermutation
   KMSSynthesisPermutation
   TokenSwapperSynthesisPermutation


QFT Synthesis
'''''''''''''

.. list-table:: Plugins for :class:`.QFTGate` (key = ``"qft"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Targeted connectivity
    * - ``"full"``
      - :class:`~.QFTSynthesisFull`
      - all-to-all
    * - ``"line"``
      - :class:`~.QFTSynthesisLine`
      - linear
    * - ``"default"``
      - :class:`~.QFTSynthesisFull`
      - all-to-all

.. autosummary::
   :toctree: ../stubs/

   QFTSynthesisFull
   QFTSynthesisLine


MCX Synthesis
'''''''''''''

The following table lists synthesis plugins available for an :class:`.MCXGate` gate
with `k` control qubits. If the available number of clean/dirty auxiliary qubits is
not sufficient, the corresponding synthesis method will return `None`.

.. list-table:: Plugins for :class:`.MCXGate` (key = ``"mcx"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Number of dirty ancillas
      - Description
    * - ``"gray_code"``
      - :class:`~.MCXSynthesisGrayCode`
      - `0`
      - `0`
      - exponentially many CX gates; use only for small values of :math:`k`
    * - ``"noaux_v24"``
      - :class:`~.MCXSynthesisNoAuxV24`
      - :math:`0`
      - :math:`0`
      - quadratic number of CX gates :math:`8 k^2+O(k)`
    * - ``"noaux_sp22"``
      - :class:`~.MCXSynthesisNoAuxSP22`
      - :math:`0`
      - :math:`0`
      - quadratic number of CX gates :math:`4 k^2+O(k)`
    * - ``"noaux_hp24"``
      - :class:`~.MCXSynthesisNoAuxHP24`
      - :math:`0`
      - :math:`0`
      - linear number of CX gates; use instead of ``"noaux_sp22"`` or ``"noaux_v24"`` or ``"gray_code"`` for :math:`k>32`
    * - ``"n_clean_m15"``
      - :class:`~.MCXSynthesisNCleanM15`
      - :math:`\lceil(k-2)/2\rceil`
      - :math:`0`
      - at most :math:`6k-6` CX gates
    * - ``"n_dirty_i15"``
      - :class:`~.MCXSynthesisNDirtyI15`
      - :math:`0`
      - :math:`k-2`
      - at most :math:`8k-6` CX gates
    * - ``"n_dirty_m15"``
      - :class:`~.MCXSynthesisNDirtyM15`
      - :math:`0`
      - :math:`\lceil(k-2)/2\rceil`
      - :math:`8k-12` CX gates for :math:`k\geq 4`
    * - ``"2_clean_kg24"``
      - :class:`~.MCXSynthesis2CleanKG24`
      - :math:`2`
      - :math:`0`
      - at most :math:`6k-6` CX gates
    * - ``"2_dirty_kg24"``
      - :class:`~.MCXSynthesis2DirtyKG24`
      - `0`
      - `2`
      - at most :math:`12k-18` CX gates
    * - ``"1_clean_kg24"``
      - :class:`~.MCXSynthesis1CleanKG24`
      - `1`
      - `0`
      - at most :math:`6k-6` CX gates
    * - ``"1_dirty_kg24"``
      - :class:`~.MCXSynthesis1DirtyKG24`
      - :math:`0`
      - :math:`1`
      - at most :math:`12k-18` CX gates
    * - ``"1_clean_b95"``
      - :class:`~.MCXSynthesis1CleanB95`
      - :math:`1`
      - :math:`0`
      - at most :math:`16k-8` CX gates
    * - ``"default"``
      - :class:`~.MCXSynthesisDefault`
      - any
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   MCXSynthesisGrayCode
   MCXSynthesisNoAuxV24
   MCXSynthesisNoAuxSP22
   MCXSynthesisNoAuxHP24
   MCXSynthesisNCleanM15
   MCXSynthesisNDirtyI15
   MCXSynthesisNDirtyM15
   MCXSynthesis2CleanKG24
   MCXSynthesis2DirtyKG24
   MCXSynthesis1CleanKG24
   MCXSynthesis1DirtyKG24
   MCXSynthesis1CleanB95
   MCXSynthesisDefault


MCMT Synthesis
''''''''''''''

.. list-table:: Plugins for :class:`.MCMTGate` (key = ``"mcmt"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Number of dirty ancillas
      - Description
    * - ``"vchain"``
      - :class:`.MCMTSynthesisVChain`
      - :math:`k-1`
      - :math:`0`
      - uses a linear number of Toffoli gates
    * - ``"noaux"``
      - :class:`~.MCMTSynthesisNoAux`
      - :math:`0`
      - :math:`0`
      - uses Qiskit's standard control mechanism
    * - ``"xgate"``
      - :class:`.MCMTSynthesisXGate`
      - :math:`0`
      - :math:`0`
      - uses a linear number of Toffoli gates
    * - ``"default"``
      - :class:`~.MCMTSynthesisDefault`
      - any
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   MCMTSynthesisVChain
   MCMTSynthesisNoAux
   MCMTSynthesisXGate
   MCMTSynthesisDefault


Integer comparators
'''''''''''''''''''

.. list-table:: Plugins for :class:`.IntegerComparatorGate` (key = ``"IntComp"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Description
      - Auxiliary qubits
    * - ``"twos"``
      - :class:`~.IntComparatorSynthesis2s`
      - use addition with two's complement
      - :math:`n - 1` clean
    * - ``"noaux"``
      - :class:`~.IntComparatorSynthesisNoAux`
      - flip the target controlled on all :math:`O(2^l)` allowed integer values
      - none
    * - ``"default"``
      - :class:`~.IntComparatorSynthesisDefault`
      - use the best algorithm depending on the available auxiliary qubits
      - any

.. autosummary::
   :toctree: ../stubs/

   IntComparatorSynthesis2s
   IntComparatorSynthesisNoAux
   IntComparatorSynthesisDefault


Sums
''''

.. list-table:: Plugins for :class:`.WeightedSumGate` (key = ``"WeightedSum"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Description
      - Auxiliary qubits
    * - ``"default"``
      - :class:`.WeightedSumSynthesisDefault`
      - use a V-chain based synthesis
      - given :math:`s` sum qubits, used :math:`s - 1_{s \leq 2}` clean auxiliary qubits

.. autosummary::
   :toctree: ../stubs/

   WeightedSumSynthesisDefault


Pauli Evolution Synthesis
'''''''''''''''''''''''''

.. list-table:: Plugins for :class:`.PauliEvolutionGate` (key = ``"PauliEvolution"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Description
      - Targeted connectivity
    * - ``"basic"``
      - :class:`~.PauliEvolutionSynthesisBasic`
      - use a diagonalizing Clifford per Pauli term
      - all-to-all
    * - ``"rustiq"``
      - :class:`~.PauliEvolutionSynthesisRustiq`
      - use the synthesis method from `Rustiq circuit synthesis library
        <https://github.com/smartiel/rustiq-core>`_
      - all-to-all
    * - ``"mcts"``
      - :class:`~.PauliEvolutionSynthesisMcts`
      - use the Monte Carlo Tree Synthesis (MCTS) method
      - all-to-all
    * - ``"default"``
      - :class:`~.PauliEvolutionSynthesisDefault`
      - Uses the best synthesis method available.
      - all-to-all

.. autosummary::
   :toctree: ../stubs/

   PauliEvolutionSynthesisDefault
   PauliEvolutionSynthesisBasic
   PauliEvolutionSynthesisRustiq
   PauliEvolutionSynthesisMcts


Modular Adder Synthesis
'''''''''''''''''''''''

.. list-table:: Plugins for :class:`.ModularAdderGate` (key = ``"ModularAdder"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"modular_v17"``
      - :class:`.ModularAdderSynthesisV17`
      - :math:`0`
      - a modular adder without any ancillary qubits
    * - ``"ripple_cdkm"``
      - :class:`.ModularAdderSynthesisC04`
      - :math:`1`
      - a ripple-carry adder
    * - ``"ripple_vbe"``
      - :class:`.ModularAdderSynthesisV95`
      - :math:`n-1`, for :math:`n`-bit numbers
      - a ripple-carry adder
    * - ``"qft"``
      - :class:`.ModularAdderSynthesisD00`
      - :math:`0`
      - a QFT-based adder
    * - ``"default"``
      - :class:`~.ModularAdderSynthesisDefault`
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   ModularAdderSynthesisV17
   ModularAdderSynthesisC04
   ModularAdderSynthesisD00
   ModularAdderSynthesisV95
   ModularAdderSynthesisDefault

Half Adder Synthesis
''''''''''''''''''''

.. list-table:: Plugins for :class:`.HalfAdderGate` (key = ``"HalfAdder"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"ripple_cdkm"``
      - :class:`.HalfAdderSynthesisC04`
      - :math:`1`
      - a ripple-carry adder
    * - ``"ripple_r25"``
      - :class:`.HalfAdderSynthesisR25`
      - :math:`0`
      - a ripple-carry adder with no ancillas
    * - ``"ripple_vbe"``
      - :class:`.HalfAdderSynthesisV95`
      - :math:`n-1`, for :math:`n`-bit numbers
      - a ripple-carry adder
    * - ``"qft"``
      - :class:`.HalfAdderSynthesisD00`
      - :math:`0`
      - a QFT-based adder
    * - ``"default"``
      - :class:`~.HalfAdderSynthesisDefault`
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   HalfAdderSynthesisC04
   HalfAdderSynthesisD00
   HalfAdderSynthesisV95
   HalfAdderSynthesisR25
   HalfAdderSynthesisDefault

Full Adder Synthesis
''''''''''''''''''''

.. list-table:: Plugins for :class:`.FullAdderGate` (key = ``"FullAdder"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"ripple_cdkm"``
      - :class:`.FullAdderSynthesisC04`
      - :math:`0`
      - a ripple-carry adder
    * - ``"ripple_vbe"``
      - :class:`.FullAdderSynthesisV95`
      - :math:`n-1`, for :math:`n`-bit numbers
      - a ripple-carry adder
    * - ``"default"``
      - :class:`~.FullAdderSynthesisDefault`
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   FullAdderSynthesisC04
   FullAdderSynthesisV95
   FullAdderSynthesisDefault


Multiplier Synthesis
''''''''''''''''''''

.. list-table:: Plugins for :class:`.MultiplierGate` (key = ``"Multiplier"``)
    :header-rows: 1

    * - Plugin name
      - Plugin class
      - Number of clean ancillas
      - Description
    * - ``"cumulative"``
      - :class:`.MultiplierSynthesisH18`
      - depending on the :class:`.AdderGate` used
      - a cumulative adder based on controlled adders
    * - ``"qft"``
      - :class:`.MultiplierSynthesisR17`
      - :math:`0`
      - a QFT-based multiplier
    * - ``"default"``
      - :class:`~.MultiplierSynthesisDefault`
      - any
      - chooses the best algorithm based on the ancillas available

.. autosummary::
   :toctree: ../stubs/

   MultiplierSynthesisH18
   MultiplierSynthesisR17
   MultiplierSynthesisDefault

"""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings
import numpy as np
import rustworkx as rx

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.operation import Operation
from qiskit.circuit.library import (
    LinearFunction,
    QFTGate,
    XGate,
    MCXGate,
    C3XGate,
    C4XGate,
    PauliEvolutionGate,
    PermutationGate,
    MCMTGate,
    ModularAdderGate,
    HalfAdderGate,
    FullAdderGate,
    MultiplierGate,
    WeightedSumGate,
    GlobalPhaseGate,
)
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    Modifier,
    ControlModifier,
    InverseModifier,
    PowerModifier,
    _canonicalize_modifiers,
)
from qiskit.transpiler.coupling import CouplingMap

from qiskit.synthesis.arithmetic import (
    synth_integer_comparator_2s,
    synth_integer_comparator_greedy,
    synth_weighted_sum_carry,
)
from qiskit.synthesis.clifford import (
    synth_clifford_full,
    synth_clifford_layers,
    synth_clifford_depth_lnn,
    synth_clifford_greedy,
    synth_clifford_ag,
    synth_clifford_bm,
)
from qiskit.synthesis.linear import (
    synth_cnot_count_full_pmh,
    synth_cnot_depth_line_kms,
    calc_inverse_matrix,
)
from qiskit.synthesis.linear.linear_circuits_utils import transpose_cx_circ
from qiskit.synthesis.permutation import (
    synth_permutation_basic,
    synth_permutation_acg,
    synth_permutation_depth_lnn_kms,
)
from qiskit.synthesis.qft import (
    synth_qft_full,
    synth_qft_line,
)
from qiskit.synthesis.multi_controlled import (
    synth_mcx_n_dirty_i15,
    synth_mcx_n_dirty_m15,
    synth_mcx_2_dirty_kg24,
    synth_mcx_1_dirty_kg24,
    synth_mcx_n_clean_m15,
    synth_mcx_2_clean_kg24,
    synth_mcx_1_clean_kg24,
    synth_mcx_1_clean_b95,
    synth_mcx_gray_code,
    synth_mcx_noaux_v24,
    synth_mcx_noaux_sp22,
    synth_mcx_noaux_hp24,
    synth_mcmt_vchain,
    synth_mcmt_xgate,
)
from qiskit.synthesis.evolution import (
    ProductFormula,
    synth_pauli_network_rustiq,
    synth_pauli_network_mcts,
)
from qiskit.synthesis.arithmetic import (
    adder_ripple_c04,
    adder_qft_d00,
    adder_ripple_v95,
    adder_ripple_r25,
    adder_modular_v17,
    multiplier_qft_r17,
    multiplier_cumulative_h18,
)
from qiskit.quantum_info import Clifford, SparsePauliOp, SparseObservable
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit._add_control import EFFICIENTLY_CONTROLLED_GATES
from qiskit.transpiler.optimization_metric import OptimizationMetric
from qiskit.circuit.library.pauli_evolution import _contains_projectors

from qiskit._accelerate.high_level_synthesis import synthesize_operation, HighLevelSynthesisData
from .plugin import HighLevelSynthesisPlugin

if TYPE_CHECKING:
    from qiskit.circuit.quantumcircuitdata import CircuitInstruction


def _is_coupling_map_all_to_all(coupling_map: CouplingMap) -> bool:
    """Return whether the coupling map is all-to-all.

    A coupling map is all-to-all if for every pair of distinct qubits
    ``i`` and ``j`` either ``(i, j)`` or ``(j, i)`` or both are present
    in the edge list.
    """
    n = coupling_map.size()
    edges = {(min(a, b), max(a, b)) for a, b in coupling_map}
    return len(edges) == n * (n - 1) // 2


class DefaultSynthesisClifford(HighLevelSynthesisPlugin):
    """The default clifford synthesis plugin.

    For N <= 3 qubits this is the optimal CX cost decomposition by Bravyi, Maslov.
    For N > 3 qubits this is done using the general non-optimal greedy compilation
    routine from reference by Bravyi, Hu, Maslov, Shaydulin.

    This plugin name is :``clifford.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_full(high_level_object)
        return decomposition


class AGSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Aaronson-Gottesman method.

    This plugin name is :``clifford.ag`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_ag(high_level_object)
        return decomposition


class BMSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method.

    The method only works on Cliffords with at most 3 qubits, for which it
    constructs the optimal CX cost decomposition.

    This plugin name is :``clifford.bm`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        if high_level_object.num_qubits <= 3:
            decomposition = synth_clifford_bm(high_level_object)
        else:
            decomposition = None

        return decomposition


class GreedySynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the greedy synthesis
    Bravyi-Hu-Maslov-Shaydulin method.

    This plugin name is :``clifford.greedy`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_greedy(high_level_object)
        return decomposition


class LayerSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers.

    This plugin name is :``clifford.layers`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_layers(high_level_object)
        return decomposition


class LayerLnnSynthesisClifford(HighLevelSynthesisPlugin):
    """Clifford synthesis plugin based on the Bravyi-Maslov method
    to synthesize Cliffords into layers, with each layer synthesized
    adhering to LNN connectivity.

    This plugin name is :``clifford.lnn`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Clifford."""
        if not isinstance(high_level_object, Clifford):
            return None

        decomposition = synth_clifford_depth_lnn(high_level_object)
        return decomposition


class DefaultSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """The default linear function synthesis plugin.

    This plugin name is :``linear_function.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        if not isinstance(high_level_object, LinearFunction):
            return None

        decomposition = synth_cnot_count_full_pmh(high_level_object.linear)
        return decomposition


class KMSSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Kutin-Moulton-Smithline method.

    This plugin name is :``linear_function.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * use_inverted: Indicates whether to run the algorithm on the inverse matrix
        and to invert the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.
    * use_transposed: Indicates whether to run the algorithm on the transposed matrix
        and to invert the order of CX gates in the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        if not isinstance(high_level_object, LinearFunction):
            return None

        use_inverted = options.get("use_inverted", False)
        use_transposed = options.get("use_transposed", False)

        mat = high_level_object.linear.astype(bool, copy=False)

        if use_transposed:
            mat = np.transpose(mat)
        if use_inverted:
            mat = calc_inverse_matrix(mat)

        decomposition = synth_cnot_depth_line_kms(mat)

        if use_transposed:
            decomposition = transpose_cx_circ(decomposition)
        if use_inverted:
            decomposition = decomposition.inverse()

        return decomposition


class PMHSynthesisLinearFunction(HighLevelSynthesisPlugin):
    """Linear function synthesis plugin based on the Patel-Markov-Hayes method.

    This plugin name is :``linear_function.pmh`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * section size: The size of each section used in the Patel–Markov–Hayes algorithm [1].
    * use_inverted: Indicates whether to run the algorithm on the inverse matrix
        and to invert the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.
    * use_transposed: Indicates whether to run the algorithm on the transposed matrix
        and to invert the order of CX gates in the synthesized circuit.
        In certain cases this provides a better decomposition than the direct approach.

    References:
        1. Patel, Ketan N., Igor L. Markov, and John P. Hayes,
           *Optimal synthesis of linear reversible circuits*,
           Quantum Information & Computation 8.3 (2008): 282-294.
           `arXiv:quant-ph/0302002 [quant-ph] <https://arxiv.org/abs/quant-ph/0302002>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given LinearFunction."""
        if not isinstance(high_level_object, LinearFunction):
            return None

        section_size = options.get("section_size", 2)
        use_inverted = options.get("use_inverted", False)
        use_transposed = options.get("use_transposed", False)

        mat = high_level_object.linear.astype(bool, copy=False)

        if use_transposed:
            mat = np.transpose(mat)
        if use_inverted:
            mat = calc_inverse_matrix(mat)

        decomposition = synth_cnot_count_full_pmh(mat, section_size=section_size)

        if use_transposed:
            decomposition = transpose_cx_circ(decomposition)
        if use_inverted:
            decomposition = decomposition.inverse()

        return decomposition


class KMSSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Kutin, Moulton, Smithline method.

    This plugin name is :``permutation.kms`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        if not isinstance(high_level_object, PermutationGate):
            return None

        decomposition = synth_permutation_depth_lnn_kms(high_level_object.pattern)
        return decomposition


class BasicSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on sorting.

    This plugin name is :``permutation.basic`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        if not isinstance(high_level_object, PermutationGate):
            return None

        decomposition = synth_permutation_basic(high_level_object.pattern)
        return decomposition


class ACGSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the Alon, Chung, Graham method.

    This plugin name is :``permutation.acg`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""
        if not isinstance(high_level_object, PermutationGate):
            return None

        decomposition = synth_permutation_acg(high_level_object.pattern)
        return decomposition


class QFTSynthesisFull(HighLevelSynthesisPlugin):
    """Synthesis plugin for QFT gates using all-to-all connectivity.

    This plugin name is :``qft.full`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    Note that the plugin mechanism is not applied if the gate is called ``qft`` but
    is not an instance of ``QFTGate``. This allows users to create custom gates with
    name ``qft``.

    The plugin supports the following additional options:

    * reverse_qubits (bool): Whether to synthesize the "QFT" operation (if ``False``,
        which is the default) or the "QFT-with-reversal" operation (if ``True``).
        Some implementation of the ``QFTGate`` include a layer of swap gates at the end
        of the synthesized circuit, which can in principle be dropped if the ``QFTGate``
        itself is the last gate in the circuit.
    * approximation_degree (int): The degree of approximation (0 for no approximation).
        It is possible to implement the QFT approximately by ignoring
        controlled-phase rotations with the angle beneath a threshold. This is discussed
        in more detail in [1] or [2].
    * insert_barriers (bool): If True, barriers are inserted as visualization improvement.
    * inverse (bool): If True, the inverse Fourier transform is constructed.
    * name (str): The name of the circuit.

    References:
        1. Adriano Barenco, Artur Ekert, Kalle-Antti Suominen, and Päivi Törmä,
           *Approximate Quantum Fourier Transform and Decoherence*,
           Physical Review A (1996).
           `arXiv:quant-ph/9601018 [quant-ph] <https://arxiv.org/abs/quant-ph/9601018>`_
        2. Donny Cheung,
           *Improved Bounds for the Approximate QFT* (2004),
           `arXiv:quant-ph/0403071 [quant-ph] <https://https://arxiv.org/abs/quant-ph/0403071>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given QFTGate."""

        # Even though the gate is called "qft", it's not a QFTGate,
        # and we should not synthesize it using the plugin.
        if not isinstance(high_level_object, QFTGate):
            return None

        reverse_qubits = options.get("reverse_qubits", False)
        approximation_degree = options.get("approximation_degree", 0)
        insert_barriers = options.get("insert_barriers", False)
        inverse = options.get("inverse", False)
        name = options.get("name", None)

        decomposition = synth_qft_full(
            num_qubits=high_level_object.num_qubits,
            do_swaps=not reverse_qubits,
            approximation_degree=approximation_degree,
            insert_barriers=insert_barriers,
            inverse=inverse,
            name=name,
        )
        return decomposition


class QFTSynthesisLine(HighLevelSynthesisPlugin):
    """Synthesis plugin for QFT gates using linear connectivity.

    This plugin name is :``qft.line`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    Note that the plugin mechanism is not applied if the gate is called ``qft`` but
    is not an instance of ``QFTGate``. This allows users to create custom gates with
    name ``qft``.

    The plugin supports the following additional options:

    * reverse_qubits (bool): Whether to synthesize the "QFT" operation (if ``False``,
        which is the default) or the "QFT-with-reversal" operation (if ``True``).
        Some implementation of the ``QFTGate`` include a layer of swap gates at the end
        of the synthesized circuit, which can in principle be dropped if the ``QFTGate``
        itself is the last gate in the circuit.
    * approximation_degree (int): the degree of approximation (0 for no approximation).
        It is possible to implement the QFT approximately by ignoring
        controlled-phase rotations with the angle beneath a threshold. This is discussed
        in more detail in [1] or [2].

    References:
        1. Adriano Barenco, Artur Ekert, Kalle-Antti Suominen, and Päivi Törmä,
           *Approximate Quantum Fourier Transform and Decoherence*,
           Physical Review A (1996).
           `arXiv:quant-ph/9601018 [quant-ph] <https://arxiv.org/abs/quant-ph/9601018>`_
        2. Donny Cheung,
           *Improved Bounds for the Approximate QFT* (2004),
           `arXiv:quant-ph/0403071 [quant-ph] <https://https://arxiv.org/abs/quant-ph/0403071>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given QFTGate."""

        # Even though the gate is called "qft", it's not a QFTGate,
        # and we should not synthesize it using the plugin.
        if not isinstance(high_level_object, QFTGate):
            return None

        reverse_qubits = options.get("reverse_qubits", False)
        approximation_degree = options.get("approximation_degree", 0)

        decomposition = synth_qft_line(
            num_qubits=high_level_object.num_qubits,
            do_swaps=not reverse_qubits,
            approximation_degree=approximation_degree,
        )
        return decomposition


class TokenSwapperSynthesisPermutation(HighLevelSynthesisPlugin):
    """The permutation synthesis plugin based on the token swapper algorithm.

    This plugin name is :``permutation.token_swapper`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    In more detail, this plugin is used to synthesize objects of type `PermutationGate`.
    When synthesis succeeds, the plugin outputs a quantum circuit consisting only of swap
    gates. When synthesis does not succeed, the plugin outputs `None`.

    If either `coupling_map` or `qubits` is None, then the synthesized circuit
    is not required to adhere to connectivity constraints, as is the case
    when the synthesis is done before layout/routing.

    On the other hand, if both `coupling_map` and `qubits` are specified, the synthesized
    circuit is supposed to adhere to connectivity constraints. At the moment, the
    plugin only creates swap gates between qubits in `qubits`, i.e. it does not use
    any other qubits in the coupling map (if such synthesis is not possible, the
    plugin  outputs `None`).

    The plugin supports the following plugin-specific options:

    * trials: The number of trials for the token swapper to perform the mapping. The
      circuit with the smallest number of SWAPs is returned.
    * seed: The argument to the token swapper specifying the seed for random trials.
    * parallel_threshold: The argument to the token swapper specifying the number of nodes
      in the graph beyond which the algorithm will use parallel processing.

    For more details on the token swapper algorithm, see to the paper:
    `arXiv:1902.09102 <https://arxiv.org/abs/1902.09102>`__.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given Permutation."""

        if not isinstance(high_level_object, PermutationGate):
            return None

        trials = options.get("trials", 5)
        seed = options.get("seed", 0)
        parallel_threshold = options.get("parallel_threshold", 50)

        pattern = high_level_object.pattern
        pattern_as_dict = {j: i for i, j in enumerate(pattern)}

        # When the plugin is called from the HighLevelSynthesis transpiler pass,
        # the coupling map already takes target into account.
        if coupling_map is None or qubits is None:
            # The abstract synthesis uses a fully connected coupling map, allowing
            # arbitrary connections between qubits.
            used_coupling_map = CouplingMap.from_full(len(pattern))
        else:
            # The concrete synthesis uses the coupling map restricted to the set of
            # qubits over which the permutation gate is defined. If we allow using other
            # qubits in the coupling map, replacing the node in the DAGCircuit that
            # defines this PermutationGate by the DAG corresponding to the constructed
            # decomposition becomes problematic. Note that we allow the reduced
            # coupling map to be disconnected.
            used_coupling_map = coupling_map.reduce(qubits, check_if_connected=False)

        graph = used_coupling_map.graph.to_undirected()
        swapper = ApproximateTokenSwapper(graph, seed=seed)

        try:
            swapper_result = swapper.map(
                pattern_as_dict, trials, parallel_threshold=parallel_threshold
            )
        except rx.InvalidMapping:
            swapper_result = None

        if swapper_result is not None:
            decomposition = QuantumCircuit(len(graph.node_indices()))
            for swap in swapper_result:
                decomposition.swap(*swap)
            return decomposition

        return None


class MCXSynthesisNDirtyI15(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper
    by Iten et al. (2016).

    See [1] for details.

    This plugin name is :``mcx.n_dirty_i15`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 4` control qubits this synthesis
    method requires :math:`k - 2` additional dirty auxiliary qubits. The synthesized
    circuit consists of :math:`2 * k - 1` qubits and at most :math:`8 * k - 6` CX gates.
    For :math:`k\le 3` explicit efficient circuits are used instead.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean auxiliary qubits available.
    * num_dirty_ancillas: The number of dirty auxiliary qubits available.
    * relative_phase: When set to ``True``, the method applies the optimized multi-controlled
      X gate up to a relative phase, in a way that, by lemma 8 of [1], the relative
      phases of the ``action part`` cancel out with the phases of the ``reset part``.
    * action_only: when set to ``True``, the method applies only the ``action part``
      of lemma 8 of [1].

    References:
        1. Iten et. al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
           `arXiv:1501.06911 <https://arxiv.org/abs/1501.06911>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_clean_ancillas = options.get("num_clean_ancillas", 0)
        num_dirty_ancillas = options.get("num_dirty_ancillas", 0)
        relative_phase = options.get("relative_phase", False)
        action_only = options.get("actions_only", False)

        if num_ctrl_qubits >= 3 and num_dirty_ancillas + num_clean_ancillas < num_ctrl_qubits - 2:
            # This synthesis method is not applicable as there are not enough ancilla qubits
            return None

        decomposition = synth_mcx_n_dirty_i15(num_ctrl_qubits, relative_phase, action_only)
        return decomposition


class MCXSynthesisNDirtyM15(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate using dirty ancillas, following
    circuit (5) for three controls and Proposition 5 for four or more controls in
    Maslov (2016).

    This plugin name is ``mcx.n_dirty_m15``. For :math:`k = 3` control qubits it requires
    one dirty ancilla and produces 16 T gates and 14 CX gates. For :math:`k \ge 4` it requires
    :math:`\lceil(k - 2) / 2\rceil` dirty ancillas and produces :math:`8k - 8` T gates and
    :math:`8k - 12` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean auxiliary qubits available.
    * num_dirty_ancillas: The number of dirty auxiliary qubits available.

    References:
        1. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application
           to multiple control Toffoli optimization*, Phys. Rev. A 93, 022311 (2016),
           `arXiv:1508.03273 <https://arxiv.org/abs/1508.03273>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_ancillas = options.get("num_clean_ancillas", 0) + options.get("num_dirty_ancillas", 0)
        required_ancillas = (num_ctrl_qubits - 1) // 2 if num_ctrl_qubits >= 3 else 0
        if num_ancillas < required_ancillas:
            return None

        return synth_mcx_n_dirty_m15(num_ctrl_qubits)


class MCXSynthesisNCleanM15(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate following Proposition 4 of
    Maslov (2016).

    See [1] for details.

    This plugin name is :``mcx.n_clean_m15`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 3` control qubits this synthesis
    method requires :math:`\lceil(k - 2) / 2\rceil` additional clean auxiliary qubits.
    The synthesized circuit contains :math:`8 * k - 9` T gates and
    :math:`6 * k - 6` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean auxiliary qubits available.

    References:
        1. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application
           to multiple control Toffoli optimization*, Phys. Rev. A 93, 022311 (2016),
           `arXiv:1508.03273 <https://arxiv.org/abs/1508.03273>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_clean_ancillas = options.get("num_clean_ancillas", 0)

        if num_ctrl_qubits >= 3 and num_clean_ancillas < (num_ctrl_qubits - 1) // 2:
            # This synthesis method is not applicable as there are not enough ancilla qubits
            return None

        decomposition = synth_mcx_n_clean_m15(num_ctrl_qubits)
        return decomposition


class MCXSynthesis1CleanB95(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper by
    Barenco et al. (1995).

    See [1] for details.

    This plugin name is :``mcx.1_clean_b95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 5` control qubits this synthesis
    method requires a single additional clean auxiliary qubit. The synthesized
    circuit consists of :math:`k + 2` qubits and at most :math:`16 * k - 24` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean auxiliary qubits available.

    References:
        1. Barenco et. al., *Elementary gates for quantum computation*, Phys.Rev. A52 3457 (1995),
           `arXiv:quant-ph/9503016 <https://arxiv.org/abs/quant-ph/9503016>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits

        if num_ctrl_qubits <= 2:
            # The method requires at least 3 control qubits
            return None

        num_clean_ancillas = options.get("num_clean_ancillas", 0)

        if num_ctrl_qubits >= 5 and num_clean_ancillas == 0:
            # This synthesis method is not applicable as there are not enough ancilla qubits
            return None

        decomposition = synth_mcx_1_clean_b95(num_ctrl_qubits)
        return decomposition


class MCXSynthesis2CleanKG24(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper by Khattar and
    Gidney (2024).

    See [1] for details.

    The plugin name is :``mcx.2_clean_kg24`` which can be used as the key on an :class:`~.HLSConfig`
    object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 3` control qubits this synthesis method requires
    :math:`2` additional clean ancillary qubits. The synthesized circuit consists of :math:`k + 3`
    qubits and at most :math:`6 * k - 6` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean ancillary qubits available.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_clean_ancillas = options.get("num_clean_ancillas", 0)

        if num_clean_ancillas < 2:
            return None

        decomposition = synth_mcx_2_clean_kg24(num_ctrl_qubits)
        return decomposition


class MCXSynthesis2DirtyKG24(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper by Khattar and
    Gidney (2024).

    See [1] for details.

    The plugin name is :``mcx.2_dirty_kg24`` which can be used as the key on an :class:`~.HLSConfig`
    object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 3` control qubits this synthesis method requires
    :math:`2` additional dirty ancillary qubits. The synthesized circuit consists of :math:`k + 3`
    qubits and at most :math:`12 * k - 18` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean ancillary qubits available.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_dirty_ancillas = options.get("num_dirty_ancillas", 0) + options.get(
            "num_clean_ancillas", 0
        )

        if num_dirty_ancillas < 2:
            return None

        decomposition = synth_mcx_2_dirty_kg24(num_ctrl_qubits)
        return decomposition


class MCXSynthesis1CleanKG24(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper by Khattar and
    Gidney (2024).

    See [1] for details.

    The plugin name is :``mcx.1_clean_kg24`` which can be used as the key on an :class:`~.HLSConfig`
    object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 3` control qubits this synthesis method requires
    :math:`1` additional clean ancillary qubit. The synthesized circuit consists of :math:`k + 2`
    qubits and at most :math:`6 * k - 6` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean ancillary qubits available.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_clean_ancillas = options.get("num_clean_ancillas", 0)

        if num_clean_ancillas < 1:
            return None

        decomposition = synth_mcx_1_clean_kg24(num_ctrl_qubits)
        return decomposition


class MCXSynthesis1DirtyKG24(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the paper by Khattar and
    Gidney (2024).

    See [1] for details.

    The plugin name is :``mcx.1_dirty_kg24`` which can be used as the key on an :class:`~.HLSConfig`
    object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k\ge 3` control qubits this synthesis method requires
    :math:`1` additional dirty ancillary qubit. The synthesized circuit consists of :math:`k + 2`
    qubits and at most :math:`12 * k - 18` CX gates.

    The plugin supports the following plugin-specific options:

    * num_clean_ancillas: The number of clean ancillary qubits available.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        num_dirty_ancillas = options.get("num_dirty_ancillas", 0) + options.get(
            "num_clean_ancillas", 0
        )

        if num_dirty_ancillas < 1:
            return None

        decomposition = synth_mcx_1_dirty_kg24(num_ctrl_qubits)
        return decomposition


class MCXSynthesisGrayCode(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the Gray code.

    This plugin name is :``mcx.gray_code`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k` control qubits this synthesis
    method requires no additional clean auxiliary qubits. The synthesized
    circuit consists of :math:`k + 1` qubits.

    It is not recommended to use this method for large values of :math:`k + 1`
    as it produces exponentially many gates.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        decomposition = synth_mcx_gray_code(num_ctrl_qubits)
        return decomposition


class MCXSynthesisNoAuxV24(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the
    implementation for MCPhaseGate, which is in turn based on the
    paper by Vale et al. (2024).

    See [1] for details.

    This plugin name is :``mcx.noaux_v24`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k` control qubits this synthesis
    method requires no additional clean auxiliary qubits. The synthesized
    circuit consists of :math:`k + 1` qubits. The number of CX-gates is quadratic in
    :math:`k`.

    References:
        1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
           Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
           `arXiv:2302.06377 <https://arxiv.org/abs/2302.06377>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        decomposition = synth_mcx_noaux_v24(num_ctrl_qubits)
        return decomposition


class MCXSynthesisNoAuxSP22(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled :class:`.XGate` based on the
    implementation for :class:`.MCPhaseGate`, which is in turn based on the
    paper by da Silva et al. [1] and the implementation in qclib [2].

    See [1, 2] for details.

    This plugin name is ``mcx.noaux_sp22`` which can be used as the key on
    an :class:`.HLSConfig` object to use this method with :class:`.HighLevelSynthesis`.

    For a multi-controlled :class:`.XGate` with :math:`k` control qubits this synthesis
    method requires no additional clean auxiliary qubits. The synthesized
    circuit consists of :math:`k + 1` qubits. The number of CX-gates is quadratic in
    :math:`k`.

    References:

    [1] A. J. da Silva and D. K. Park,
    Linear-depth quantum circuits for multiqubit controlled gates,
    `Phys. Rev. A 106, 042602
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042602>`__.

    [2] https://github.com/qclib/qclib/blob/master/qclib/gates/ldmcu.py
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        decomposition = synth_mcx_noaux_sp22(num_ctrl_qubits)
        return decomposition


class MCXSynthesisNoAuxHP24(HighLevelSynthesisPlugin):
    r"""Synthesis plugin for a multi-controlled X gate based on the
    paper by Huang and Palsberg.

    See [1] for details.

    This plugin name is :``mcx.noaux_hp24`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For a multi-controlled X gate with :math:`k` control qubits this synthesis
    method requires no additional clean auxiliary qubits. The synthesized
    circuit consists of :math:`k + 1` qubits. The number of CX-gates is linear in
    :math:`k`.

    References:
        1. Huang and Palsberg, *Compiling Conditional Quantum Gates without Using
           Helper Qubits*, PLDI (2024),
           <https://dl.acm.org/doi/10.1145/3656436>`_
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        num_ctrl_qubits = high_level_object.num_ctrl_qubits
        decomposition = synth_mcx_noaux_hp24(num_ctrl_qubits)
        return decomposition


class MCXSynthesisDefault(HighLevelSynthesisPlugin):
    r"""The default synthesis plugin for a multi-controlled X gate.

    This plugin name is :``mcx.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.


    The plugin supports the following plugin-specific options:

    * ``optimization_metric``: The optimization metric, indicating the property
      of the output circuit (e.g., the 2-qubit gate count or the T-count) that should
      be minimized. See :class:`.OptimizationMetric`.
    * ``num_clean_ancillas``: The number of clean ancillary qubits available.
    * ``num_dirty_ancillas``: The number of dirty ancillary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        """Run synthesis for the given MCX gate."""

        if not isinstance(high_level_object, (MCXGate, C3XGate, C4XGate)):
            # Unfortunately we occasionally have custom instructions called "mcx"
            # which get wrongly caught by the plugin interface. A simple solution is
            # to return None in this case, since HLS would proceed to examine
            # their definition as it should.
            return None

        metric = options.get("optimization_metric", OptimizationMetric.COUNT_2Q)
        if metric == OptimizationMetric.COUNT_T:
            # The order is optimized towards Clifford+T -friendly synthesis methods.
            methods = [
                MCXSynthesis2CleanKG24,
                MCXSynthesis1CleanKG24,
                MCXSynthesisNCleanM15,
                MCXSynthesisNDirtyM15,
                MCXSynthesis2DirtyKG24,
                MCXSynthesis1DirtyKG24,
                MCXSynthesis1CleanB95,
                MCXSynthesisNDirtyI15,
                (
                    MCXSynthesisNoAuxSP22
                    if high_level_object.num_ctrl_qubits <= 32
                    else MCXSynthesisNoAuxHP24
                ),
            ]
        else:
            # The order is optimized towards CX-count -friendly synthesis methods.
            # For C3X, use NDirtyI15 since it matches NDirtyM15's 14 CX gates without
            # using an ancilla. For larger MCX gates, NDirtyM15 uses fewer CX gates.
            dirty_method = (
                MCXSynthesisNDirtyI15
                if high_level_object.num_ctrl_qubits == 3
                else MCXSynthesisNDirtyM15
            )
            methods = [
                MCXSynthesis2CleanKG24,
                MCXSynthesis1CleanKG24,
                MCXSynthesisNCleanM15,
                dirty_method,
                MCXSynthesis2DirtyKG24,
                MCXSynthesis1DirtyKG24,
                MCXSynthesis1CleanB95,
                (
                    MCXSynthesisNoAuxSP22
                    if high_level_object.num_ctrl_qubits <= 32
                    else MCXSynthesisNoAuxHP24
                ),
            ]

        for method in methods:
            if (
                decomposition := method().run(
                    high_level_object, coupling_map, target, qubits, **options
                )
            ) is not None:
                return decomposition

        return None


class MCMTSynthesisDefault(HighLevelSynthesisPlugin):
    """A default decomposition for MCMT gates."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MCMTGate):
            return None

        for synthesis_method in [
            MCMTSynthesisXGate,
            MCMTSynthesisVChain,
        ]:
            if (
                decomposition := synthesis_method().run(
                    high_level_object, coupling_map, target, qubits, **options
                )
            ) is not None:
                return decomposition

        return MCMTSynthesisNoAux().run(high_level_object, coupling_map, target, qubits, **options)


class MCMTSynthesisNoAux(HighLevelSynthesisPlugin):
    """A V-chain based synthesis for ``MCMTGate``."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MCMTGate):
            return None

        base_gate = high_level_object.base_gate
        ctrl_state = options.get("ctrl_state", None)

        if high_level_object.num_target_qubits == 1:
            # no broadcasting needed (makes for better circuit diagrams)
            circuit = QuantumCircuit(high_level_object.num_qubits)
            circuit.append(
                base_gate.control(
                    high_level_object.num_ctrl_qubits, ctrl_state=ctrl_state, annotated=False
                ),
                circuit.qubits,
            )

        else:
            base = QuantumCircuit(high_level_object.num_target_qubits, name=high_level_object.label)
            for i in range(high_level_object.num_target_qubits):
                base.append(base_gate, [i], [])

            circuit = base.control(
                high_level_object.num_ctrl_qubits, ctrl_state=ctrl_state, annotated=False
            )

        return circuit.decompose()


class MCMTSynthesisVChain(HighLevelSynthesisPlugin):
    """A V-chain based synthesis for ``MCMTGate``."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MCMTGate):
            return None

        if options.get("num_clean_ancillas", 0) < high_level_object.num_ctrl_qubits - 1:
            return None  # insufficient number of auxiliary qubits

        ctrl_state = options.get("ctrl_state", None)

        return synth_mcmt_vchain(
            high_level_object.base_gate,
            high_level_object.num_ctrl_qubits,
            high_level_object.num_target_qubits,
            ctrl_state,
        )


class MCMTSynthesisXGate(HighLevelSynthesisPlugin):
    """A synthesis for ``MCMTGate`` with X gate as the base gate."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MCMTGate):
            return None

        if not isinstance(high_level_object.base_gate, XGate):
            return None  # this plugin only supports X gates

        ctrl_state = options.get("ctrl_state", None)
        return synth_mcmt_xgate(
            high_level_object.num_ctrl_qubits, high_level_object.num_target_qubits, ctrl_state
        )


class IntComparatorSynthesisDefault(HighLevelSynthesisPlugin):
    """The default synthesis for ``IntegerComparatorGate``.

    Currently this is only supporting an ancilla-based decomposition.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        num_state_qubits = high_level_object.num_qubits - 1
        num_aux = num_state_qubits - 1
        if options.get("num_clean_ancillas", 0) < num_aux:
            return synth_integer_comparator_greedy(
                num_state_qubits, high_level_object.value, high_level_object.geq
            )

        return synth_integer_comparator_2s(
            num_state_qubits, high_level_object.value, high_level_object.geq
        )


class IntComparatorSynthesisNoAux(HighLevelSynthesisPlugin):
    """A potentially exponentially expensive comparison w/o auxiliary qubits."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        return synth_integer_comparator_greedy(
            high_level_object.num_state_qubits, high_level_object.value, high_level_object.geq
        )


class IntComparatorSynthesis2s(HighLevelSynthesisPlugin):
    """An integer comparison based on 2s complement."""

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        num_aux = high_level_object.num_state_qubits - 1
        if options.get("num_clean_ancillas", 0) < num_aux:
            return None

        return synth_integer_comparator_2s(
            high_level_object.num_state_qubits, high_level_object.value, high_level_object.geq
        )


class ModularAdderSynthesisDefault(HighLevelSynthesisPlugin):
    """The default modular adder (no carry in, no carry out qubit) synthesis.

    This plugin name is:``ModularAdder.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * ``optimization_metric``: The optimization metric, indicating the property
      of the output circuit (e.g., the 2-qubit gate count or the T-count) that should
      be minimized. See :class:`.OptimizationMetric`.
    * ``num_clean_ancillas``: The number of clean ancillary qubits available.
    * ``num_dirty_ancillas``: The number of dirty ancillary qubits available.

    For two-qubit-gate count, this selects QFT synthesis for two to four state
    qubits, and CDKM synthesis from five state qubits when a clean ancilla is
    available. Otherwise, it uses the ancilla-free V17 synthesis. For T-count,
    it always uses V17 synthesis.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        metric = options.get("optimization_metric", OptimizationMetric.COUNT_2Q)
        if metric == OptimizationMetric.COUNT_2Q:
            # The order is optimized towards CX-friendly synthesis methods.
            methods = []
            if 2 <= high_level_object.num_state_qubits <= 4:
                methods.append(ModularAdderSynthesisD00)
            elif high_level_object.num_state_qubits >= 5:
                methods.append(ModularAdderSynthesisC04)
            methods.append(ModularAdderSynthesisV17)
        else:
            # The order is optimized towards Clifford+T-friendly synthesis methods.
            methods = [ModularAdderSynthesisV17]

        for method in methods:
            if (
                decomposition := method().run(
                    high_level_object, coupling_map, target, qubits, **options
                )
            ) is not None:
                return decomposition

        return None


class ModularAdderSynthesisV17(HighLevelSynthesisPlugin):
    r"""A modular adder (modulo :math:`2^n`) without any ancillary qubits.

    The plugin name is :``ModularAdder.v17`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin requires no auxiliary qubits.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits

        return adder_modular_v17(num_state_qubits)


class ModularAdderSynthesisC04(HighLevelSynthesisPlugin):
    r"""A ripple-carry adder, modulo :math:`2^n`.

    This plugin name is:``ModularAdder.ripple_c04`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin requires at least one clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        # unless we implement the full adder, this implementation needs an ancilla qubit
        if options.get("num_clean_ancillas", 0) < 1:
            return None

        return adder_ripple_c04(high_level_object.num_state_qubits, kind="fixed")


class ModularAdderSynthesisV95(HighLevelSynthesisPlugin):
    r"""A ripple-carry adder, modulo :math:`2^n`.

    This plugin name is:``ModularAdder.ripple_v95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For an adder on 2 registers with :math:`n` qubits each, this plugin requires at
    least :math:`n-1` clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits

        # The synthesis method needs n-1 clean ancilla qubits
        if num_state_qubits - 1 > options.get("num_clean_ancillas", 0):
            return None

        return adder_ripple_v95(num_state_qubits, kind="fixed")


class ModularAdderSynthesisD00(HighLevelSynthesisPlugin):
    r"""A QFT-based adder, modulo :math:`2^n`.

    This plugin name is:``ModularAdder.qft_d00`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, ModularAdderGate):
            return None

        return adder_qft_d00(high_level_object.num_state_qubits, kind="fixed", annotated=True)


class HalfAdderSynthesisDefault(HighLevelSynthesisPlugin):
    r"""The default half-adder (no carry in, but a carry out qubit) synthesis.

    The CDKM ripple-carry adder uses :math:`O(n)` CX gates and one clean auxiliary
    qubit. The ancilla-free R25 ripple-carry adder is used when it has a lower cost
    or no clean auxiliary qubit is available.

    This plugin name is:``HalfAdder.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        # For up to 3 qubits, ripple_r25 is better
        if (
            high_level_object.num_state_qubits <= 3
            and (
                decomposition := HalfAdderSynthesisR25().run(
                    high_level_object, coupling_map, target, qubits, **options
                )
            )
            is not None
        ):
            return decomposition

        # The next best option is to use ripple_c04 (if there are enough ancilla qubits)
        if (
            decomposition := HalfAdderSynthesisC04().run(
                high_level_object, coupling_map, target, qubits, **options
            )
        ) is not None:
            return decomposition

        # The ripple_rv_25 adder does not require ancilla qubits and should always succeed
        return HalfAdderSynthesisR25().run(
            high_level_object, coupling_map, target, qubits, **options
        )


class HalfAdderSynthesisC04(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-out bit.

    This plugin name is:``HalfAdder.ripple_c04`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin requires at least one clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        # unless we implement the full adder, this implementation needs an ancilla qubit
        if options.get("num_clean_ancillas", 0) < 1:
            return None

        return adder_ripple_c04(high_level_object.num_state_qubits, kind="half")


class HalfAdderSynthesisV95(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-out bit.

    This plugin name is:``HalfAdder.ripple_v95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For an adder on 2 registers with :math:`n` qubits each, this plugin requires at
    least :math:`n-1` clean auxiliary qubit.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits

        # The synthesis method needs n-1 clean ancilla qubits
        if num_state_qubits - 1 > options.get("num_clean_ancillas", 0):
            return None

        return adder_ripple_v95(num_state_qubits, kind="half")


class HalfAdderSynthesisR25(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-out bit with no ancillary qubits.

    This plugin name is:``HalfAdder.ripple_r25`` which can be used as the key on an
    :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits
        return adder_ripple_r25(num_state_qubits)


class HalfAdderSynthesisD00(HighLevelSynthesisPlugin):
    """A QFT-based adder with a carry-in and a carry-out bit.

    This plugin name is:``HalfAdder.qft_d00`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, HalfAdderGate):
            return None

        return adder_qft_d00(high_level_object.num_state_qubits, kind="half", annotated=True)


class FullAdderSynthesisDefault(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-in and a carry-out bit.

    This plugin name is:``FullAdder.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The plugin supports the ``optimization_metric`` option; see
    :class:`.OptimizationMetric`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, FullAdderGate):
            return None

        # V95 has one fewer CX for n=1, while C04 has the lower T-count.
        if (
            high_level_object.num_state_qubits == 1
            and options.get("optimization_metric", OptimizationMetric.COUNT_2Q)
            == OptimizationMetric.COUNT_2Q
        ):
            decomposition = FullAdderSynthesisV95().run(
                high_level_object, coupling_map, target, qubits, **options
            )
            if decomposition is not None:
                return decomposition

        return FullAdderSynthesisC04().run(
            high_level_object, coupling_map, target, qubits, **options
        )


class FullAdderSynthesisC04(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-in and a carry-out bit.

    This plugin name is:``FullAdder.ripple_c04`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin requires no auxiliary qubits.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, FullAdderGate):
            return None

        return adder_ripple_c04(high_level_object.num_state_qubits, kind="full")


class FullAdderSynthesisV95(HighLevelSynthesisPlugin):
    """A ripple-carry adder with a carry-in and a carry-out bit.

    This plugin name is:``FullAdder.ripple_v95`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    For an adder on 2 registers with :math:`n` qubits each, this plugin requires at
    least :math:`n-1` clean auxiliary qubits.

    The plugin supports the following plugin-specific options:

    * ``num_clean_ancillas``: The number of clean auxiliary qubits available.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, FullAdderGate):
            return None

        num_state_qubits = high_level_object.num_state_qubits

        # The synthesis method needs n-1 clean ancilla qubits
        if num_state_qubits - 1 > options.get("num_clean_ancillas", 0):
            return None

        return adder_ripple_v95(num_state_qubits, kind="full")


class MultiplierSynthesisH18(HighLevelSynthesisPlugin):
    """A cumulative multiplier based on controlled adders.

    This plugin name is:``Multiplier.cumulative_h18`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MultiplierGate):
            return None

        return multiplier_cumulative_h18(
            high_level_object.num_state_qubits, high_level_object.num_result_qubits
        )


class MultiplierSynthesisR17(HighLevelSynthesisPlugin):
    """A QFT-based multiplier.

    This plugin name is:``Multiplier.qft_r17`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MultiplierGate):
            return None

        return multiplier_qft_r17(
            high_level_object.num_state_qubits, high_level_object.num_result_qubits
        )


class MultiplierSynthesisDefault(HighLevelSynthesisPlugin):
    """THe default multiplier plugin.

    This plugin name is:``Multiplier.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, MultiplierGate):
            return None

        # The H18 algorithm is better both for CX-count and T-count
        return multiplier_cumulative_h18(
            high_level_object.num_state_qubits, high_level_object.num_result_qubits
        )


def _cx_size_for_pauli_evo(circuit: QuantumCircuit) -> int:
    """
    Estimate the number of CX-gates in a circuit produced by one of the PauliEvolutionGate
    synthesis algorithms (``basic``, ``rustiq`` or ``mcts``).

    This function is only called for PauliEvolutionGates without projectors. In this case,
    in addition to CX gates, ``basic`` can produce two-qubit rxx, ryy, rzz, and rzx rotations,
    while ``rustiq`` and ``mcts`` can produce swaps. These two-qubit gates are counted according
    to the number of CX gates required to decompose them.
    """
    ops = circuit.count_ops()
    return (
        ops.get("cx", 0)
        + 2 * (ops.get("rxx", 0) + ops.get("ryy", 0) + ops.get("rzz", 0) + ops.get("rzx", 0))
        + 3 * ops.get("swap", 0)
    )


class PauliEvolutionSynthesisDefault(HighLevelSynthesisPlugin):
    """Synthesize a :class:`.PauliEvolutionGate` using the default synthesis algorithm.

    This plugin name is:``PauliEvolution.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    This plugin runs other implemented synthesis plugins, forwarding all options to
    the selected plugins. The choice of which synthesis plugins to run is heuristic.
    The current implementation generally runs :class:`PauliEvolutionSynthesisBasic`,
    but might run :class:`PauliEvolutionSynthesisMcts` to minimize the CX count
    if suitable. This selection is subject to change and should not be relied upon.

    For greater control, specify :class:`~.HLSConfig` with the relevant synthesis plugins
    directly.

    The following plugin option directly influences this plugin:

    * optimization_level: The optimization level used to select the synthesis
      algorithm. Higher levels generate potentially more optimized circuits,
      at the expense of longer transpilation time.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, PauliEvolutionGate):
            return None

        synth_object = PauliEvolutionSynthesisBasic().run(
            high_level_object, coupling_map, target, qubits, **options
        )

        # Currently we only run the MCTS method for higher optimization levels, for all-to-all
        # connectivity, for Pauli evolution gates without projector terms, and with
        # preserve_order=True.
        if (
            (options.get("optimization_level", 2) >= 2)
            and options.get("preserve_order", True)
            and (not _contains_projectors(high_level_object))
            and ((coupling_map is None) or _is_coupling_map_all_to_all(coupling_map))
        ):
            synth_mcts = PauliEvolutionSynthesisMcts().run(
                high_level_object, coupling_map, target, qubits, **options
            )
            # TODO: In this initial implementation, choose the better circuit based on the number of
            # CX gates, using their CX decompositions. Experimentally, this seems to work well both
            # when the targeted basis set contains CX gates and when it contains RZZ gates.
            # In a follow-up, investigate whether target-aware heuristics could improve the
            # results further.
            if _cx_size_for_pauli_evo(synth_mcts) < _cx_size_for_pauli_evo(synth_object):
                synth_object = synth_mcts

        return synth_object


class PauliEvolutionSynthesisBasic(HighLevelSynthesisPlugin):
    """Synthesize a :class:`.PauliEvolutionGate` using the basic synthesis algorithm.

    This plugin name is:``PauliEvolution.basic`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The following plugin option can be set:

    * preserve_order: If ``False``, allow re-ordering the Pauli terms in the Hamiltonian to
      reduce the circuit depth of the decomposition.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, PauliEvolutionGate):
            # Don't do anything if a gate is called "evolution" but is not an
            # actual PauliEvolutionGate
            return None
        algo = high_level_object.synthesis

        original_preserve_order = algo.preserve_order
        if "preserve_order" in options and isinstance(algo, ProductFormula):
            algo.preserve_order = options["preserve_order"]

        synth_object = algo.synthesize(high_level_object)
        algo.preserve_order = original_preserve_order
        return synth_object


class PauliEvolutionSynthesisRustiq(HighLevelSynthesisPlugin):
    """Synthesize a :class:`.PauliEvolutionGate` using Rustiq.

    This plugin name is :``PauliEvolution.rustiq`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    The Rustiq synthesis algorithm is described in [1], and is implemented in
    Rust-based quantum circuit synthesis library available at
    https://github.com/smartiel/rustiq-core.

    On large circuits the plugin may take a significant runtime.

    The plugin supports the following additional options:

    * optimize_count (bool): if `True` the synthesis algorithm will try to optimize
      the 2-qubit gate count; and if `False` then the 2-qubit depth.
    * preserve_order (bool): whether the order of paulis should be preserved, up to
      commutativity.
    * upto_clifford (bool): if `True`, the final Clifford operator is not synthesized.
    * upto_phase (bool): if `True`, the global phase of the returned circuit may
      differ from the global phase of the given pauli network.
    * resynth_clifford_method (int): describes the strategy to synthesize the final
      Clifford operator. Allowed values are `0` (naive approach), `1` (qiskit
      greedy synthesis), `2` (rustiq isometry synthesis).

    References:
        1. Timothée Goubault de Brugière and Simon Martiel,
           *Faster and shorter synthesis of Hamiltonian simulation circuits*,
           `arXiv:2404.03280 [quant-ph] <https://arxiv.org/abs/2404.03280>`_

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, PauliEvolutionGate):
            # Don't do anything if a gate is called "evolution" but is not an
            # actual PauliEvolutionGate
            return None

        # The synthesis function synth_pauli_network_rustiq does not support SparseObservables,
        # so we need to convert them to SparsePauliOps.
        if isinstance(high_level_object.operator, SparsePauliOp):
            pauli_op = high_level_object.operator

        elif isinstance(high_level_object.operator, SparseObservable):
            pauli_op = SparsePauliOp.from_sparse_observable(high_level_object.operator)

        elif isinstance(high_level_object.operator, list):
            pauli_op = []
            for op in high_level_object.operator:
                if isinstance(op, SparseObservable):
                    pauli_op.append(SparsePauliOp.from_sparse_observable(op))
                else:
                    pauli_op.append(op)

        else:
            raise TranspilerError("Invalid PauliEvolutionGate.")

        evo = PauliEvolutionGate(
            pauli_op,
            time=high_level_object.time,
            label=high_level_object.label,
            synthesis=high_level_object.synthesis,
        )
        algo = evo.synthesis

        if not isinstance(algo, ProductFormula):
            warnings.warn(
                "Cannot apply Rustiq if the evolution synthesis does not implement ``expand``. ",
                stacklevel=2,
                category=RuntimeWarning,
            )
            return None

        original_preserve_order = algo.preserve_order
        if "preserve_order" in options:
            algo.preserve_order = options["preserve_order"]

        num_qubits = evo.num_qubits
        pauli_network = algo.expand(evo)
        optimize_count = options.get("optimize_count", True)
        preserve_order = options.get("preserve_order", True)
        upto_clifford = options.get("upto_clifford", False)
        upto_phase = options.get("upto_phase", False)
        resynth_clifford_method = options.get("resynth_clifford_method", 1)

        synth_object = synth_pauli_network_rustiq(
            num_qubits=num_qubits,
            pauli_network=pauli_network,
            optimize_count=optimize_count,
            preserve_order=preserve_order,
            upto_clifford=upto_clifford,
            upto_phase=upto_phase,
            resynth_clifford_method=resynth_clifford_method,
        )

        algo.preserve_order = original_preserve_order
        return synth_object


class PauliEvolutionSynthesisMcts(HighLevelSynthesisPlugin):
    """Synthesize a :class:`.PauliEvolutionGate` using Monte Carlo Tree Search (MCTS).

    This plugin name is registered under the name ``"PauliEvolution.mcts"``, which can
    be used as the key on an :class:`~.HLSConfig` object to use this method with
    :class:`~.HighLevelSynthesis`.

    The synthesis algorithm is described in [1].

    On large circuits, this synthesis method may require a significant runtime.

    The plugin supports the following additional options:

    * preserve_order (bool): Preserve the order of Pauli rotations, up to
      commutativity.
    * upto_clifford (bool): if `True`, the final Clifford operator is not synthesized.
    * upto_phase (bool): if `True`, the returned circuit may differ from the input by
      a global phase.
    * num_simulations (int): Number of Monte Carlo simulations to perform. This value
      must be at least 1.
    * max_parallel_simulations (int | None): Maximum number of simulations that can be
      performed in parallel. If integer, this value must be at least 1. The value of
      `None` means "unlimited".

    References:
        1. Mulundano Machiya, Matt Menickelly, Paul Hovland, Ji Liu,
           *MonteQ: A Monte Carlo Tree Search Based Quantum Circuit Synthesis Framework*,
           `arXiv:2604.19029 <https://arxiv.org/abs/2604.19029>`_

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, PauliEvolutionGate):
            # Don't do anything if a gate is called "evolution" but is not an
            # actual PauliEvolutionGate
            return None

        # The synthesis function synth_pauli_network_mcts does not support SparseObservables,
        # so we need to convert them to SparsePauliOps.
        if isinstance(high_level_object.operator, SparsePauliOp):
            pauli_op = high_level_object.operator

        elif isinstance(high_level_object.operator, SparseObservable):
            pauli_op = SparsePauliOp.from_sparse_observable(high_level_object.operator)

        elif isinstance(high_level_object.operator, list):
            pauli_op = []
            for op in high_level_object.operator:
                if isinstance(op, SparseObservable):
                    pauli_op.append(SparsePauliOp.from_sparse_observable(op))
                else:
                    pauli_op.append(op)

        else:
            raise TranspilerError("Invalid PauliEvolutionGate.")

        evo = PauliEvolutionGate(
            pauli_op,
            time=high_level_object.time,
            label=high_level_object.label,
            synthesis=high_level_object.synthesis,
        )
        algo = evo.synthesis

        if not isinstance(algo, ProductFormula):
            warnings.warn(
                "Cannot apply MCTS if the evolution synthesis does not implement ``expand``. ",
                stacklevel=2,
                category=RuntimeWarning,
            )
            return None

        original_preserve_order = algo.preserve_order
        if "preserve_order" in options:
            algo.preserve_order = options["preserve_order"]

        num_qubits = evo.num_qubits
        pauli_network = algo.expand(evo)
        preserve_order = options.get("preserve_order", True)
        upto_clifford = options.get("upto_clifford", False)
        upto_phase = options.get("upto_phase", False)
        num_simulations = options.get("num_simulations", 1)
        max_parallel_simulations = options.get("max_parallel_simulations", None)

        synth_object = synth_pauli_network_mcts(
            num_qubits=num_qubits,
            pauli_network=pauli_network,
            preserve_order=preserve_order,
            upto_clifford=upto_clifford,
            upto_phase=upto_phase,
            num_simulations=num_simulations,
            max_parallel_simulations=max_parallel_simulations,
        )
        algo.preserve_order = original_preserve_order
        algo.preserve_order = original_preserve_order
        return synth_object


class AnnotatedSynthesisDefault(HighLevelSynthesisPlugin):
    """Synthesize an :class:`.AnnotatedOperation` using the default synthesis algorithm.

    This plugin name is:``annotated.default`` which can be used as the key on
    an :class:`~.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.
    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        # The plugin is triggered based on the name (i.e. for operations called "annotated").
        # However, we should only do something when the operation is truthfully an AnnotatedOperation.
        if not isinstance(high_level_object, AnnotatedOperation):
            return None

        # Combine the modifiers. If there were no modifiers, or the modifiers magically canceled out,
        # return the quantum circuit containing the base operation.
        high_level_object = self._canonicalize_op(high_level_object)
        if not isinstance(high_level_object, AnnotatedOperation):
            return self._instruction_to_circuit(high_level_object)

        operation = high_level_object
        modifiers = high_level_object.modifiers

        # The plugin needs additional information that is not yet passed via the run's method
        # arguments: namely high-level-synthesis data and options, the global qubits over which
        # the operation is defined, and the initial state of each global qubit.
        tracker = options.get("qubit_tracker", None)
        data = options.get("hls_data", None)
        input_qubits = options.get("input_qubits", None)

        if data is None or input_qubits is None:
            raise TranspilerError(
                "The AnnotatedSynthesisDefault plugin should receive data and input_qubits via options."
            )

        # The synthesis consists of two steps:
        #   - First, we synthesize the base operation.
        #   - Second, we apply modifiers to this circuit.
        #
        # An important optimization (similar to the code in ``add_control.py``) is to synthesize
        # the base operation with respect to a larger set of "basis" gates, to which the control
        # logic can be added more efficiently. In addition, we add annotated operations to be
        # in this larger set, exploiting the fact that adding control to annotated operations
        # returns a new annotated operation with an extended list of modifiers.
        #
        # Note that it is fine for this function to return a circuit with high-level objects
        # (including annotated operations) as the HighLevelSynthesis transpiler pass will
        # recursively re-synthesize this circuit, However, we should always guarantee that some
        # progress is made.
        basis = set(EFFICIENTLY_CONTROLLED_GATES + ["annotated", "mcx", "qft"])

        base_synthesis_data = HighLevelSynthesisData(
            hls_config=data.hls_config,
            hls_plugin_manager=data.hls_plugin_manager,
            coupling_map=None,
            target=None,
            equivalence_library=data.equivalence_library,
            hls_op_names=data.hls_op_names,
            device_insts=basis,
            use_physical_indices=data.use_physical_indices,
            min_qubits=0,
            unroll_definitions=data.unroll_definitions,
            optimize_clifford_t=data.optimize_clifford_t,
            optimization_level=data.optimization_level,
        )

        num_ctrl = sum(mod.num_ctrl_qubits for mod in modifiers if isinstance(mod, ControlModifier))
        power = sum(mod.power for mod in modifiers if isinstance(mod, PowerModifier))
        is_inverted = sum(1 for mod in modifiers if isinstance(mod, InverseModifier)) % 2

        # First, synthesize the base operation of this annotated operation.
        # As this step cannot use any control qubits as auxiliary qubits, we use a dedicated
        # tracker (annotated_tracker).
        # The logic is as follows:
        # - annotated_tracker.disable control qubits
        # - if have power or inverse modifiers, annotated_tracker.set_dirty(base_qubits)
        # - synthesize the base operation using annotated tracker
        # - main_tracker.set_dirty(base_qubits)
        #
        # Note that we need to set the base_qubits to dirty if we have power or inverse
        # modifiers. For power: even if the power is a positive integer (that is, we need
        # to repeat the same circuit multiple times), even if the target is initially at |0>,
        # it will generally not be at |0> after one iteration. For inverse: as we
        # flip the order of operations, we cannot exploit which qubits are at |0> as "viewed from
        # the back of the circuit". If we just have control modifiers, we can use the state
        # of base qubits when synthesizing the controlled operation.
        #
        # In addition, all of the other global qubits that are not a part of the annotated
        # operation can be used as they are in all cases, since we are assuming that all of
        # the synthesis methods preserve the states of ancilla qubits.
        annotated_tracker = tracker.copy()
        control_qubits = input_qubits[:num_ctrl]
        base_qubits = input_qubits[num_ctrl:]
        annotated_tracker.disable(control_qubits)  # do not access control qubits
        if power != 0 or is_inverted:
            annotated_tracker.set_dirty(base_qubits)

        # Note that synthesize_operation also returns the output qubits on which the
        # operation is defined, however currently the plugin mechanism has no way
        # to return these (and instead the upstream code greedily grabs some ancilla
        # qubits from the circuit). We should refactor the plugin "run" interface to
        # return the actual ancilla qubits used.
        synthesized_base_op_result = synthesize_operation(
            operation.base_op, base_qubits, base_synthesis_data, annotated_tracker
        )

        # The base operation does not need to be synthesized.
        # For simplicity, we wrap the instruction into a circuit. Note that
        # this should not deteriorate the quality of the result.
        if synthesized_base_op_result is None:
            synthesized_base_op = self._instruction_to_circuit(operation.base_op)
        else:
            synthesized_base_op = QuantumCircuit._from_circuit_data(synthesized_base_op_result[0])
        tracker.set_dirty(base_qubits)

        # As one simple optimization, we apply conjugate decomposition to the circuit obtained
        # while synthesizing the base operator.
        conjugate_decomp = self._conjugate_decomposition(synthesized_base_op)

        if conjugate_decomp is None:
            # Apply annotations to the whole circuit.
            # This step currently does not introduce ancilla qubits. However it makes
            # a lot of sense to allow this in the future.
            synthesized = self._apply_annotations(synthesized_base_op, operation.modifiers)
        else:
            # Apply annotations only to the middle part of the circuit.
            front, middle, back = conjugate_decomp
            synthesized = QuantumCircuit(operation.num_qubits)
            synthesized.compose(
                front, synthesized.qubits[num_ctrl : operation.num_qubits], inplace=True
            )
            synthesized.compose(
                self._apply_annotations(middle, operation.modifiers),
                synthesized.qubits,
                inplace=True,
            )
            synthesized.compose(
                back, synthesized.qubits[num_ctrl : operation.num_qubits], inplace=True
            )

        return synthesized

    @staticmethod
    def _apply_annotations(circuit: QuantumCircuit, modifiers: list[Modifier]) -> QuantumCircuit:
        """
        Applies modifiers to a quantum circuit.
        """
        for modifier in modifiers:
            if isinstance(modifier, InverseModifier):
                circuit = circuit.inverse()

            elif isinstance(modifier, ControlModifier):
                if circuit.num_clbits > 0:
                    raise TranspilerError(
                        "AnnotatedSynthesisDefault: cannot control a circuit with classical bits."
                    )

                # Apply the control modifier to each gate in the circuit.
                controlled_circuit = QuantumCircuit(modifier.num_ctrl_qubits + circuit.num_qubits)
                if circuit.global_phase != 0:
                    controlled_op = GlobalPhaseGate(circuit.global_phase).control(
                        num_ctrl_qubits=modifier.num_ctrl_qubits,
                        label=None,
                        ctrl_state=modifier.ctrl_state,
                        annotated=False,
                    )
                    controlled_qubits = list(range(modifier.num_ctrl_qubits))
                    controlled_circuit.append(controlled_op, controlled_qubits)
                for inst in circuit:
                    inst_op = inst.operation
                    inst_qubits = inst.qubits
                    controlled_op = inst_op.control(
                        num_ctrl_qubits=modifier.num_ctrl_qubits,
                        label=None,
                        ctrl_state=modifier.ctrl_state,
                        annotated=False,
                    )
                    controlled_qubits = list(range(modifier.num_ctrl_qubits)) + [
                        modifier.num_ctrl_qubits + circuit.find_bit(q).index for q in inst_qubits
                    ]
                    controlled_circuit.append(controlled_op, controlled_qubits)

                circuit = controlled_circuit

                if isinstance(circuit, AnnotatedOperation):
                    raise TranspilerError(
                        "AnnotatedSynthesisDefault: failed to synthesize the control modifier."
                    )

            elif isinstance(modifier, PowerModifier):
                circuit = circuit.power(modifier.power)

            else:
                raise TranspilerError(f"AnnotatedSynthesisDefault: Unknown modifier {modifier}.")

        return circuit

    @staticmethod
    def _instruction_to_circuit(op: Operation) -> QuantumCircuit:
        """Wraps a single operation into a quantum circuit."""
        circuit = QuantumCircuit(op.num_qubits, op.num_clbits)
        circuit.append(op, circuit.qubits, circuit.clbits)
        return circuit

    @staticmethod
    def _instruction_to_circuit(op: Operation) -> QuantumCircuit:
        """Wraps a single operation into a quantum circuit."""
        circuit = QuantumCircuit(op.num_qubits, op.num_clbits)
        circuit.append(op, circuit.qubits, circuit.clbits)
        return circuit

    @staticmethod
    def _canonicalize_op(op: Operation) -> Operation:
        """
        Combines recursive annotated operations and canonicalizes modifiers.
        """
        cur = op
        all_modifiers = []

        while isinstance(cur, AnnotatedOperation):
            all_modifiers.append(cur.modifiers)
            cur = cur.base_op

        new_modifiers = []
        for modifiers in all_modifiers[::-1]:
            new_modifiers.extend(modifiers)

        canonical_modifiers = _canonicalize_modifiers(new_modifiers)

        if not canonical_modifiers:
            return cur

        return AnnotatedOperation(cur, canonical_modifiers)

    @staticmethod
    def _are_inverse_ops(inst1: CircuitInstruction, inst2: CircuitInstruction):
        """A very naive function that checks whether two circuit instructions are inverse of
        each other. The main use-case covered is a ``QFTGate`` and its inverse, represented as
        an ``AnnotatedOperation`` with a single ``InverseModifier``.
        """
        res = False

        if (
            inst1.qubits != inst2.qubits
            or inst1.clbits != inst2.clbits
            or len(inst1.params) != len(inst2.params)
        ):
            return False

        op1 = inst1.operation
        op2 = inst2.operation

        ann1 = isinstance(op1, AnnotatedOperation)
        ann2 = isinstance(op2, AnnotatedOperation)

        if not ann1 and not ann2:
            res = op1 == op2.inverse()
        elif not ann1 and ann2 and op2.modifiers == [InverseModifier()]:
            res = op1 == op2.base_op
        elif not ann2 and ann1 and op1.modifiers == [InverseModifier()]:
            res = op1.base_op == op2

        return res

    @staticmethod
    def _conjugate_decomposition(
        circuit: QuantumCircuit,
    ) -> tuple[QuantumCircuit, QuantumCircuit, QuantumCircuit] | None:
        """
        Decomposes a circuit ``A`` into 3 sub-circuits ``P``, ``Q``, ``R`` such that
        ``A = P -- Q -- R`` and ``R = P^{-1}``.

        This is accomplished by iteratively finding inverse nodes at the front and at the back of the
        circuit.

        The function returns ``None`` when ``P`` and ``R`` are empty.
        """
        num_gates = circuit.size()

        idx = 0
        ridx = num_gates - 1

        while True:
            if idx >= ridx:
                break
            if AnnotatedSynthesisDefault._are_inverse_ops(circuit[idx], circuit[ridx]):
                idx += 1
                ridx -= 1
            else:
                break

        if idx == 0:
            return None

        front_circuit = circuit.copy_empty_like()
        front_circuit.global_phase = 0
        for i in range(idx):
            front_circuit.append(circuit[i])
        middle_circuit = circuit.copy_empty_like()  # inherits the global phase
        for i in range(idx, ridx + 1):
            middle_circuit.append(circuit[i])
        back_circuit = circuit.copy_empty_like()
        back_circuit.global_phase = 0
        for i in range(ridx + 1, num_gates):
            back_circuit.append(circuit[i])
        return (front_circuit, middle_circuit, back_circuit)


class WeightedSumSynthesisDefault(HighLevelSynthesisPlugin):
    """Synthesize a :class:`.WeightedSumGate` using the default synthesis algorithm.

    This plugin name is:``WeightedSum.default`` which can be used as the key on
    an :class:`.HLSConfig` object to use this method with :class:`~.HighLevelSynthesis`.

    .. note::

        This default plugin requires auxiliary qubits. There is currently no implementation
        available without auxiliary qubits.

    """

    def run(self, high_level_object, coupling_map=None, target=None, qubits=None, **options):
        if not isinstance(high_level_object, WeightedSumGate):
            return None

        required_auxiliaries = (
            high_level_object.num_sum_qubits - 1 + int(high_level_object.num_sum_qubits > 2)
        )
        if (num_clean := options.get("num_clean_ancillas", 0)) < required_auxiliaries:
            warnings.warn(
                f"Cannot synthesize a WeightedSumGate on {high_level_object.num_state_qubits} state "
                f"qubits with less than {required_auxiliaries} clean auxiliary qubits. Only "
                f"{num_clean} are available. This will likely lead to a error in HighLevelSynthesis."
            )
            return None

        return synth_weighted_sum_carry(high_level_object)
