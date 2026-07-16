# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing multi-controlled circuits synthesis"""

from .gray_code import generate_gray_code, gray_code_chain
from .mcmt_vchain import synth_mcmt_vchain
from .mcx_synthesis import (
    synth_mcx_1_clean_kg24,
    synth_mcx_1_dirty_kg24,
    synth_mcx_2_clean_kg24,
    synth_mcx_2_dirty_kg24,
    synth_mcx_n_dirty_i15,
    synth_mcx_n_clean_m15,
    synth_mcx_1_clean_b95,
    synth_mcx_gray_code,
    synth_mcx_noaux_v24,
    synth_mcx_noaux_hp24,
    synth_c3x,
    synth_c4x,
)
from .mcmt_xgate import synth_mcmt_xgate

__all__ = [
    "generate_gray_code",
    "gray_code_chain",
    "synth_c3x",
    "synth_c4x",
    "synth_mcmt_vchain",
    "synth_mcmt_xgate",
    "synth_mcx_1_clean_b95",
    "synth_mcx_1_clean_kg24",
    "synth_mcx_1_dirty_kg24",
    "synth_mcx_2_clean_kg24",
    "synth_mcx_2_dirty_kg24",
    "synth_mcx_gray_code",
    "synth_mcx_n_clean_m15",
    "synth_mcx_n_dirty_i15",
    "synth_mcx_noaux_hp24",
    "synth_mcx_noaux_v24",
]
