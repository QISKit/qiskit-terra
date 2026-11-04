# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
===============================
Utilities (:mod:`qiskit.utils`)
===============================

.. currentmodule:: qiskit.utils

Deprecations
============

.. autofunction:: add_deprecation_to_docstring
.. autofunction:: deprecate_arg
.. autofunction:: deprecate_func

SI unit conversion
==================

.. autofunction:: apply_prefix
.. autofunction:: detach_prefix

Class tools
===========

.. autofunction:: wrap_method

Multiprocessing
===============

.. autofunction:: default_num_processes
.. autofunction:: is_main_process
.. autofunction:: local_hardware_info
.. autofunction:: should_run_in_parallel

A helper function for calling a custom function with Python
:class:`~concurrent.futures.ProcessPoolExecutor`. Tasks can be executed in parallel using this function.

.. autofunction:: parallel_map

Optional Dependency Checkers
============================

.. automodule:: qiskit.utils.optionals
"""

from .deprecation import (
    add_deprecation_to_docstring,
    deprecate_arg,
    deprecate_func,
)
from .units import apply_prefix, detach_prefix
from .classtools import wrap_method
from .lazy_tester import LazyDependencyManager, LazyImportTester, LazySubprocessTester

from . import optionals

from .parallel import (
    parallel_map,
    should_run_in_parallel,
    local_hardware_info,
    is_main_process,
    default_num_processes,
)

__all__ = [
    "LazyDependencyManager",
    "LazyImportTester",
    "LazySubprocessTester",
    "add_deprecation_to_docstring",
    "apply_prefix",
    "default_num_processes",
    "deprecate_arg",
    "deprecate_func",
    "detach_prefix",
    "is_main_process",
    "local_hardware_info",
    "optionals",
    "parallel_map",
    "should_run_in_parallel",
    "wrap_method",
]
