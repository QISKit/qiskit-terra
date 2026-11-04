# This code is part of Qiskit.
#
# (C) Copyright IBM 2017 - 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing cnot circuits"""

from .cnot_synth import synth_cnot_count_full_pmh
from .linear_depth_lnn import synth_cnot_depth_line_kms

from qiskit._accelerate.synthesis.linear import (
    gauss_elimination,
    gauss_elimination_with_perm,
    compute_rank_after_gauss_elim,
    compute_rank,
    calc_inverse_matrix,
    binary_matmul,
    random_invertible_binary_matrix,
    check_invertible_binary_matrix,
    row_op,
    col_op,
)

# This re-import is kept for compatibility with Terra 0.23. Eligible for deprecation in 0.25+.
from qiskit.synthesis.linear_phase import synth_cnot_phase_aam as graysynth

__all__ = [
    "binary_matmul",
    "calc_inverse_matrix",
    "check_invertible_binary_matrix",
    "col_op",
    "compute_rank",
    "compute_rank_after_gauss_elim",
    "gauss_elimination",
    "gauss_elimination_with_perm",
    "graysynth",
    "random_invertible_binary_matrix",
    "row_op",
    "synth_cnot_count_full_pmh",
    "synth_cnot_depth_line_kms",
]
