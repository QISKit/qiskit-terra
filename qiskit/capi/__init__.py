# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
================================================
C API interface from Python (:mod:`qiskit.capi`)
================================================

.. currentmodule:: qiskit.capi

This module provides Python-space interactions with Qiskit's public C API.

For documentation on the C API itself, see `Qiskit C API
<https://quantum.cloud.ibm.com/docs/api/qiskit-c/>`__.

Build-system interaction
========================

The Python package :mod:`qiskit` contains all of the Qiskit C API header files, and a compiled
shared-object library that includes all of the C-API functions.  You can access the locations of
these two objects with the functions :func:`get_include` and :func:`get_lib` respectively.

.. warning::
    You typically *should not* link directly against the output of :func:`get_lib`, unless you know
    what you are doing.  In particular, directly linking against this object is not a safe way to
    build a distributable Python extension module that uses Qiskit's C API.

    However, if you understand all the caveats of direct linking, you can use the function to get
    the location of the library.

.. autofunction:: get_include
.. autofunction:: get_lib


Native bindings to the C API
============================

Additionally, this module contains :mod:`ctypes` bindings to all Qiskit C API types and functions.
These are available as module attributes on :mod:`qiskit.capi` with the same name as they have in
the C API.  For example, ``qiskit.capi.qk_circuit_new`` corresponds to :c:func:`qk_circuit_new`.

.. py:attribute:: LIB
    :type: ctypes.PyDLL

    A :mod:`ctypes` wrapper around the library containing the Qiskit C API.

    This is provided for completeness, though you can access all the functions, structs and
    enumerations directly from the :mod:`qiskit.capi` module object.

Structs
-------

Concrete ``struct`` types used in the C API are declared as corresponding :class:`ctypes.Structure`
types with a complete :attr:`~ctypes.Structure._fields_` attribute.  These can be instantiated and
inspected directly.

Opaque pointers are represented by a :class:`ctypes.Structure` type with no
:attr:`~ctypes.Structure._fields_` attribute set, and cannot be instantiated.  They are typically
returned from functions wrapped in a :class:`ctypes.POINTER` type wrapper.

Enums
-----

In places where the C API has an enumeration, this module declares a Python :class:`enum.Enum` whose
values are a corresponding :mod:`ctypes` primitive integer type.  The Python-space
:class:`~enum.Enum` type object is still a wrapping Python object, so :mod:`ctypes` functions that
return an enumeration will return the raw numeric value, not the value in the :class:`~enum.Enum`.
The Python-space :class:`~enum.Enum` objects are declared for convenience in constructing calls.

Functions
---------

All the public library functions in the Qiskit C API are fully typed, and re-exported in the module
root with the same name as they have in C.  You can also access the functions from :attr:`LIB`, if
you prefer.

Note that header-only functions, such as :c:func:`qk_import`, are not exported because they are not
part of the C API library object.
"""


from pathlib import Path

from . import _ctypes
from ._ctypes import *
import qiskit._accelerate

__all__ = ["get_include", "get_lib"]
__all__ += _ctypes.__all__


def get_include() -> str:
    """Get the directory containing the ``qiskit.h`` C header file and the internal
    ``qiskit/*.h`` auxiliary files.

    When using Qiskit as a build dependency, you typically want to include this directory on the
    include search path of your compiler, such as:

    .. code-block:: bash

        qiskit_include=$(python -c 'import qiskit.capi; print(qiskit.capi.get_include())')
        gcc -I "$qiskit_include" my_bin.c -o my_bin

    The location of this directory within the Qiskit package data is not fixed, and may change
    between Qiskit versions.  You should always use this function to retrieve the directory.

    Returns:
        an absolute path to the package include-files directory.
    """
    return str(Path(__file__).parent.absolute() / "include")


def get_lib() -> str:
    """Get the path to a shared-object library that contains all the C-API exported symbols.

    .. warning::
        You typically *should not* link directly against this object.  In particular, directly
        linking against this object is not a safe way to build a Python extension module that uses
        Qiskit's C API.

    You can, if you choose, use :mod:`ctypes` to access the C API symbols contained in this object,
    though beware that the C-API types declared in the header file are not interchangeable with the
    Python objects that correspond to them.

    The location and name of this file within the Qiskit package data is not fixed, and may change
    between Qiskit versions.

    Returns:
        an absolute path to the shared-object library containing the C-API exported symbols.
    """
    return str(Path(qiskit._accelerate.__file__).absolute())
