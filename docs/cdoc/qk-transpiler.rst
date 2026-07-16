.. _capi-transpiler:

============
QkTranspiler
============

The :c:func:`qk_transpile` function exposes Qiskit's tranpsiler (:py:mod:`qiskit.transpiler`) to C.
The basic functionality is using the same underlying code as the Python-space version, but the transpiler as
exposed to C has more limitations than what is exposed to Python. The transpiler
assumes a circuit built constructed using solely the C API and is intended to
work solely in the case of a standalone C API. It will potentially not work
correctly when in a mixed Python/C use case. If you're mixing C and Python you
should call the :py:func:`.generate_preset_pass_manager` or
:py:func:`.transpile` functions for those circuits.

Data Types
==========

.. doxygenstruct:: QkTranspileResult
   :members:

.. doxygenstruct:: QkTranspileOptions
   :members:

.. c:struct:: QkTranspilerStageState

A container collecting individual attributes shared by the transpiler stages.
When transpiling correctly, each individual stage writes specific attributes
to this container that will be needed by other stages in sequence. If the container
is not initialized, each stage will initialize a new object when necessary.

   .. doxygengroup:: QkTranspilerStageState
      :members:
      :content-only:

Functions
=========

.. doxygengroup:: QkTranspiler
    :members:
    :content-only:
