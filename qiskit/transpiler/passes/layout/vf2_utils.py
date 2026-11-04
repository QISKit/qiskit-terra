# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""This module contains common utils for vf2 layout passes."""

# This re-export was implicitly public by us suggesting users import it from here during its
# announcement release note, so we should maintain it.
__all__ = ["ErrorMap"]

from qiskit._accelerate.error_map import ErrorMap
