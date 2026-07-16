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

"""
=============================================
Circuit Converters (:mod:`qiskit.converters`)
=============================================

.. currentmodule:: qiskit.converters

QuantumCircuit -> circuit components
====================================

.. autofunction:: circuit_to_instruction
.. autofunction:: circuit_to_gate

QuantumCircuit <-> DagCircuit
=============================

.. autofunction:: circuit_to_dag
.. autofunction:: dag_to_circuit

QuantumCircuit <-> DagDependency
================================

.. autofunction:: dagdependency_to_circuit
.. autofunction:: circuit_to_dagdependency

DagCircuit <-> DagDependency
============================

.. autofunction:: dag_to_dagdependency
.. autofunction:: dagdependency_to_dag
"""

from .circuit_to_dag import circuit_to_dag
from .dag_to_circuit import dag_to_circuit
from .circuit_to_instruction import circuit_to_instruction
from .circuit_to_gate import circuit_to_gate
from .circuit_to_dagdependency import circuit_to_dagdependency
from .dagdependency_to_circuit import dagdependency_to_circuit
from .dag_to_dagdependency import dag_to_dagdependency
from .dagdependency_to_dag import dagdependency_to_dag

__all__ = [
    "circuit_to_dag",
    "circuit_to_dagdependency",
    "circuit_to_gate",
    "circuit_to_instruction",
    "dag_to_circuit",
    "dag_to_dagdependency",
    "dagdependency_to_circuit",
    "dagdependency_to_dag",
]


def isinstanceint(obj):
    """Like isinstance(obj,int), but with casting. Except for strings."""
    if isinstance(obj, str):
        return False
    try:
        int(obj)
        return True
    except TypeError:
        return False


def isinstancelist(obj):
    """Like isinstance(obj, list), but with casting. Except for strings and dicts."""
    if isinstance(obj, (str, dict)):
        return False
    try:
        list(obj)
        return True
    except TypeError:
        return False
