# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Helper functions for the XXDecomposer test suite.
"""

from qiskit.circuit.library import RXXGate, RYYGate, RZZGate


def canonical_matrix(a=0.0, b=0.0, c=0.0):
    """
    Produces the matrix form of a "canonical operator"

        exp(-i (a XX + b YY + c ZZ)) .
    """
    return RXXGate(2 * a).to_matrix() @ RYYGate(2 * b).to_matrix() @ RZZGate(-2 * c).to_matrix()
