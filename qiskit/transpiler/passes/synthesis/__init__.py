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

"""Module containing transpiler synthesis passes."""

from .unitary_synthesis import UnitarySynthesis
from .plugin import high_level_synthesis_plugin_names, unitary_synthesis_plugin_names
from .linear_functions_synthesis import LinearFunctionsToPermutations
from .high_level_synthesis import HighLevelSynthesis, HLSConfig
from .solovay_kitaev_synthesis import SolovayKitaev, SolovayKitaevSynthesis
from .ross_selinger_plugin import RossSelingerSynthesis
from .clifford_unitary_synth_plugin import CliffordUnitarySynthesis
from .aqc_plugin import AQCSynthesisPlugin
from .synthesize_rz_rotations import SynthesizeRZRotations

__all__ = [
    "AQCSynthesisPlugin",
    "CliffordUnitarySynthesis",
    "HLSConfig",
    "HighLevelSynthesis",
    "LinearFunctionsToPermutations",
    "RossSelingerSynthesis",
    "SolovayKitaev",
    "SolovayKitaevSynthesis",
    "SynthesizeRZRotations",
    "UnitarySynthesis",
    "high_level_synthesis_plugin_names",
    "unitary_synthesis_plugin_names",
]
