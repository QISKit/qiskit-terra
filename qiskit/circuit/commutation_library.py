# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Provides a commutation checker that caches the determined commutation results during this session"""

from qiskit.circuit import CommutationChecker

from qiskit.circuit._standard_gates_commutations import standard_gates_commutations
from qiskit._accelerate.commutation_checker import get_standard_commutation_checker

StandardGateCommutations = standard_gates_commutations
SessionCommutationChecker = CommutationChecker()
SessionCommutationChecker.cc = get_standard_commutation_checker()
