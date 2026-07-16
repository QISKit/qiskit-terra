# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=======================================================
Classical expressions (:mod:`qiskit.circuit.classical`)
=======================================================

This module contains an exploratory representation of real-time operations on classical values
during circuit execution.

Currently, only simple expressions on bits and registers that result in a Boolean value are
supported, and these are only valid for use in the conditions of :meth:`.QuantumCircuit.if_test`
(:class:`.IfElseOp`) and :meth:`.QuantumCircuit.while_loop` (:class:`.WhileLoopOp`), and in the
target of :meth:`.QuantumCircuit.switch` (:class:`.SwitchCaseOp`).

.. note::
    This is an exploratory module, and while we will commit to the standard Qiskit deprecation
    policy within it, please be aware that the module will be deliberately limited in scope at the
    start, and early versions may not evolve cleanly into the final version.  It is possible that
    various components of this module will be replaced (subject to deprecations) instead of improved
    into a new form.

    The type system and expression tree will be expanded over time, and it is possible that the
    allowed types of some operations may need to change between versions of Qiskit as the classical
    processing capabilities develop.

.. automodule:: qiskit.circuit.classical.expr
.. automodule:: qiskit.circuit.classical.types
"""

from . import types, expr

__all__ = [
    "expr",
    "types",
]
