# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing cnot-dihedral circuit synthesis."""

from .cnotdihedral_decompose_full import synth_cnotdihedral_full
from .cnotdihedral_decompose_two_qubits import synth_cnotdihedral_two_qubits
from .cnotdihedral_decompose_general import synth_cnotdihedral_general

__all__ = [
    "synth_cnotdihedral_full",
    "synth_cnotdihedral_general",
    "synth_cnotdihedral_two_qubits",
]
