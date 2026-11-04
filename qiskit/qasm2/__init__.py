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

r'''
================================
OpenQASM 2 (:mod:`qiskit.qasm2`)
================================

Qiskit has support for interoperation with OpenQASM 2.0 programs, both :ref:`parsing into Qiskit
formats <qasm2-parse>` and :ref:`exporting back to OpenQASM 2 <qasm2-export>`.

.. note::

    OpenQASM 2 is a simple language, and not suitable for general serialization of Qiskit objects.
    See :ref:`some discussion of alternatives below <qasm2-alternatives>`, if that is what you are
    looking for.


.. _qasm2-parse:

Parsing API
===========

This module contains two public functions, both of which create a :class:`.QuantumCircuit` from an
OpenQASM 2 program. :func:`load` takes a filename, while :func:`loads` takes the program itself as a
string.  Their internals are very similar, so both offer almost the same API.

.. autofunction:: load

.. autofunction:: loads

Both of these loading functions also take an argument ``include_path``, which is an iterable of
directory names to use when searching for files in ``include`` statements.  The directories are
tried from index 0 onwards, and the first match is used.  The import ``qelib1.inc`` is treated
specially; it is always found before looking in the include path, and contains exactly the content
of the `paper describing the OpenQASM 2 language <https://arxiv.org/abs/1707.03429>`__.  The gates
in this include file are mapped to circuit-library gate objects defined by Qiskit.

.. _qasm2-custom-instructions:

Specifying custom instructions
------------------------------

You can extend the quantum components of the OpenQASM 2 language by passing an iterable of
information on custom instructions as the argument ``custom_instructions``.  In files that have
compatible definitions for these instructions, the given ``constructor`` will be used in place of
whatever other handling :mod:`qiskit.qasm2` would have done.  These instructions may optionally be
marked as ``builtin``, which causes them to not require an ``opaque`` or ``gate`` declaration, but
they will silently ignore a compatible declaration.  Either way, it is an error to provide a custom
instruction that has a different number of parameters or qubits as a defined instruction in a parsed
program.  Each element of the argument iterable should be a particular data class:

.. autoclass:: CustomInstruction

This can be particularly useful when trying to resolve ambiguities in the global-phase conventions
of an OpenQASM 2 program.  See :ref:`qasm2-phase-conventions` for more details.

.. _qasm2-custom-classical:

Specifying custom classical functions
-------------------------------------

Similar to the quantum extensions above, you can also extend the processing done to classical
expressions (arguments to gates) by passing an iterable to the argument ``custom_classical`` to either
loader.  This needs the ``name`` (a valid OpenQASM 2 identifier), the number ``num_params`` of
parameters it takes, and a Python callable that implements the function.  The Python callable must
be able to accept ``num_params`` positional floating-point arguments, and must return a float or
integer (which will be converted to a float).  Builtin functions cannot be overridden.

.. autoclass:: CustomClassical

.. _qasm2-strict-mode:

Strict mode
-----------

Both of the loader functions have an optional "strict" mode.  By default, this parser is a little
bit more relaxed than the official specification: it allows trailing commas in parameter lists;
unnecessary (empty-statement) semicolons; the ``OPENQASM 2.0;`` version statement to be omitted; and
a couple of other quality-of-life improvements without emitting any errors.  You can use the
letter-of-the-spec mode with ``strict=True``.


.. _qasm2-export:

Exporting API
=============

Similar to other serialization modules in Python, this module offers two public functions:
:func:`dump` and :func:`dumps`, which take a :class:`.QuantumCircuit` and write out a representative
OpenQASM 2 program to a file-like object or return a string, respectively.

.. autofunction:: dump

.. autofunction:: dumps


Errors
======

This module defines a generic error type that derives from :exc:`.QiskitError` that can be used as a
catch when you care about failures emitted by the interoperation layer specifically.

.. autoexception:: QASM2Error

In cases where the lexer or parser fails due to an invalid OpenQASM 2 file, the conversion functions
will raise a more specific error with a message explaining what the failure is, and where in the
file it occurred.

.. autoexception:: QASM2ParseError

When the exporters fail to export a circuit, likely because it has structure that cannot be
represented by OpenQASM 2.0, they will also emit a custom error.

.. autoexception:: QASM2ExportError

.. _qasm2-examples:

Examples
========

Exporting examples
------------------

Export a simple :class:`.QuantumCircuit` to an OpenQASM 2 string:

.. plot::
   :include-source:
   :nofigs:
   :context: reset

    import qiskit.qasm2
    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    print(qiskit.qasm2.dumps(qc))

.. code-block:: text

    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0],q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];

Write out the same :class:`.QuantumCircuit` to a given filename:

.. code-block:: python

    qiskit.qasm2.dump(qc, "myfile.qasm")

Similarly, one can use general :class:`os.PathLike` instances as the filename:

.. code-block:: python

    import pathlib

    qiskit.qasm2.dump(qc, pathlib.Path.home() / "myfile.qasm")

One can also dump the text to an already-open stream:

.. code-block:: python

    import io

    with io.StringIO() as stream:
        qiskit.qasm2.dump(qc, stream)

Parsing examples
----------------

Use :func:`loads` to import an OpenQASM 2 program in a string into a :class:`.QuantumCircuit`:

.. plot::
   :include-source:
   :nofigs:

    import qiskit.qasm2
    program = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        h q[0];
        cx q[0], q[1];

        measure q -> c;
    """
    circuit = qiskit.qasm2.loads(program)
    circuit.draw()

.. code-block:: text

         ┌───┐     ┌─┐
    q_0: ┤ H ├──■──┤M├───
         └───┘┌─┴─┐└╥┘┌─┐
    q_1: ─────┤ X ├─╫─┤M├
              └───┘ ║ └╥┘
    c: 2/═══════════╩══╩═
                    0  1

You can achieve the same thing if the program is stored in a file by using :func:`load` instead,
passing the filename as an argument:

.. code-block:: python

    import qiskit.qasm2
    circuit = qiskit.qasm2.load("myfile.qasm")

OpenQASM 2 files can include other OpenQASM 2 files via the ``include`` statement.  You can
influence the search path used for finding these files with the ``include_path`` argument to both
:func:`load` and :func:`loads`.  By default, only the current working directory is searched.

.. code-block:: python

    import qiskit.qasm2
    program = """
        include "other.qasm";
        // ... and so on
    """
    circuit = qiskit.qasm2.loads(program, include_path=("/path/to/a", "/path/to/b", "."))

For :func:`load` only, there is an extra argument ``include_input_directory``, which can be used to
either ``'append'``, ``'prepend'`` or ignore (``None``) the directory of the loaded file in the
include path.  By default, this directory is appended to the search path, so it is tried last, but
you can change this.

.. code-block:: python

    import qiskit.qasm2
    filenames = ["./subdirectory/a.qasm", "/path/to/b.qasm", "~/my.qasm"]
    # Search the directory of each file before other parts of the include path.
    circuits = [
        qiskit.qasm2.load(filename, include_input_directory="prepend") for filename in filenames
    ]
    # Override the include path, and don't search the directory of each file unless it's in the
    # absolute path list.
    circuits = [
        qiskit.qasm2.load(
            filename,
            include_path=("/usr/include/qasm", "~/qasm/include"),
            include_input_directory=None,
        )
        for filename in filenames
    ]

Sometimes you may want to influence the :class:`.Gate` objects that the importer emits for given
named instructions.  Gates that are defined by the statement ``include "qelib1.inc";`` will
automatically be associated with a suitable Qiskit circuit-library gate, but you can extend this:

.. plot::
   :include-source:
   :nofigs:

    from qiskit.circuit import Gate
    from qiskit.qasm2 import loads, CustomInstruction

    class MyGate(Gate):
        def __init__(self, theta):
            super().__init__("my", 2, [theta])

    class Builtin(Gate):
        def __init__(self):
            super().__init__("builtin", 1, [])

    program = """
        opaque my(theta) q1, q2;
        qreg q[2];
        my(0.5) q[0], q[1];
        builtin q[0];
    """
    customs = [
        CustomInstruction(name="my", num_params=1, num_qubits=2, constructor=MyGate),
        # Setting 'builtin=True' means the instruction doesn't require a declaration to be usable.
        CustomInstruction("builtin", 0, 1, Builtin, builtin=True),
    ]
    circuit = loads(program, custom_instructions=customs)


Similarly, you can add new classical functions used during the description of arguments to gates,
both in the main body of the program (which come out constant-folded) and within the bodies of
defined gates (which are computed on demand).  Here we provide a Python version of ``atan2(y, x)``,
which mathematically is :math:`\arctan(y/x)` but correctly handling angle quadrants and infinities,
and a custom ``add_one`` function:

.. plot::
   :include-source:
   :nofigs:

    import math
    from qiskit.qasm2 import loads, CustomClassical

    program = """
        include "qelib1.inc";
        qreg q[2];
        rx(atan2(pi, 3 + add_one(0.2))) q[0];
        cx q[0], q[1];
    """

    def add_one(x):
        return x + 1

    customs = [
        # `atan2` takes two parameters, and `math.atan2` implements it.
        CustomClassical("atan2", 2, math.atan2),
        # Our `add_one` takes only one parameter.
        CustomClassical("add_one", 1, add_one),
    ]
    circuit = loads(program, custom_classical=customs)


.. _qasm2-phase-conventions:

OpenQASM 2 Phase Conventions
============================

As a language, OpenQASM 2 does not have a way to specify the global phase of a complete program, nor
of particular gate definitions.  This means that parsers of the language may interpret particular
gates with a different global phase than what you might expect.  For example, the *de facto*
standard library of OpenQASM 2 ``qelib1.inc`` contains definitions of ``u1`` and ``rz`` as follows:

.. code-block:: text

    gate u1(lambda) q {
        U(0, 0, lambda) q;
    }

    gate rz(phi) a {
        u1(phi) a;
    }

In other words, ``rz`` appears to be a direct alias for ``u1``.  However, the interpretation of
``u1`` is specified in `equation (3) of the paper describing the language
<https://arxiv.org/abs/1707.03429>`__ as

.. math::

    u_1(\lambda) = \operatorname{diag}\bigl(1, e^{i\lambda}\bigr) \sim R_z(\lambda)

where the :math:`\sim` symbol denotes equivalence only up to a global phase.  When parsing OpenQASM
2, we need to choose how to handle a distinction between such gates; ``u1`` is defined in the prose
to be different by a phase to ``rz``, but the language is not designed to represent this.

Qiskit's default position is to interpret a usage of the standard-library ``rz`` using
:class:`.RZGate`, and a usage of ``u1`` as using the phase-distinct :class:`.U1Gate`.  If you wish
to use the phase conventions more implied by a direct interpretation of the ``gate`` statements in
the header file, you can use :class:`CustomInstruction` to override how Qiskit builds the circuit.

For the standard ``qelib1.inc`` include there is only one point of difference, and so the override
needed to switch its phase convention is:

.. plot::
   :include-source:
   :nofigs:
   :context: reset

    from qiskit import qasm2
    from qiskit.circuit.library import PhaseGate
    from qiskit.quantum_info import Operator

    program = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rz(pi / 2) q[0];
    """

    custom = [
        qasm2.CustomInstruction("rz", 1, 1, PhaseGate),
    ]

This will use Qiskit's :class:`.PhaseGate` class to represent the ``rz`` instruction, which is
equal (including the phase) to :class:`.U1Gate`:

.. plot::
   :include-source:
   :nofigs:
   :context:

    Operator(qasm2.loads(program, custom_instructions=custom))

.. code-block:: text

    Operator([[1.000000e+00+0.j, 0.000000e+00+0.j],
              [0.000000e+00+0.j, 6.123234e-17+1.j]],
             input_dims=(2,), output_dims=(2,))


.. _qasm2-legacy-compatibility:

Legacy Compatibility
====================

:meth:`.QuantumCircuit.from_qasm_str` and :meth:`~.QuantumCircuit.from_qasm_file` used to make a few
additions on top of the raw specification. Qiskit originally tried to use OpenQASM 2 as a sort of
serialization format, and expanded its behavior as Qiskit expanded.  The new parser under all its
defaults implements the specification more strictly.

In particular, in the legacy importers:

* the `include_path` is effectively:
    1. ``<qiskit>/qasm/libs``, where ``<qiskit>`` is the root of the installed ``qiskit`` package;
    2. the current working directory.

* there are additional instructions defined in ``qelib1.inc``:
    ``csx a, b``
      Controlled :math:`\sqrt X` gate, corresponding to :class:`.CSXGate`.

    ``cu(theta, phi, lambda, gamma) c, t``
      The four-parameter version of a controlled-:math:`U`, corresponding to :class:`.CUGate`.

    ``rxx(theta) a, b``
      Two-qubit rotation around the :math:`XX` axis, corresponding to :class:`.RXXGate`.

    ``rzz(theta) a, b``
      Two-qubit rotation around the :math:`ZZ` axis, corresponding to :class:`.RZZGate`.

    ``rccx a, b, c``
      The double-controlled :math:`X` gate, but with relative phase differences over the standard
      Toffoli gate.  This *should* correspond to the Qiskit gate :class:`~.RCCXGate`, but the legacy
      converter wouldn't actually output this type.

    ``rc3x a, b, c, d``
      The triple-controlled :math:`X` gate, but with relative phase differences over the standard
      definition.  Corresponds to :class:`.RC3XGate`.

    ``c3x a, b, c, d``
      The triple-controlled :math:`X` gate, corresponding to :class:`.C3XGate`.

    ``c3sqrtx a, b, c, d``
      The triple-controlled :math:`\sqrt X` gate, corresponding to :class:`.C3SXGate`.

    ``c4x a, b, c, d, e``
      The quadruple-controlled :math:`X` gate., corresponding to :class:`.C4XGate`.

* if *any* ``opaque`` or ``gate`` definition was given for the name ``delay``, they attempt to
  output a :class:`~qiskit.circuit.Delay` instruction at each call.  To function, this expects a
  definition compatible with ``opaque delay(t) q;``, where the time ``t`` is given in units of
  ``dt``.  The importer will raise errors on construction if there was not exactly one parameter
  and one qubit, or if the parameter is not integer-valued.

* the additional scientific-calculator functions ``asin``, ``acos`` and ``atan`` are available.

* the parsed grammar is effectively the same as :ref:`the strict mode of the new importers
  <qasm2-strict-mode>`.

You can emulate this behavior in :func:`load` and :func:`loads` by setting `include_path`
appropriately (try inspecting the variable ``qiskit.__file__`` to find the installed location), and
by passing a list of :class:`CustomInstruction` instances for each of the custom gates you care
about.  To make things easier we make three tuples available, which each contain one component of
a configuration that is equivalent to Qiskit's legacy converter behavior.

.. py:data:: LEGACY_CUSTOM_INSTRUCTIONS

   A tuple containing the extra `custom_instructions` that Qiskit's legacy built-in converters used
   if ``qelib1.inc`` is included, and there is any definition of a ``delay`` instruction.  The gates
   in the paper version of ``qelib1.inc`` and ``delay`` all require a compatible declaration
   statement to be present within the OpenQASM 2 program, but Qiskit's legacy additions are all
   marked as builtins since they are not actually present in any include file this parser sees.

.. py:data:: LEGACY_CUSTOM_CLASSICAL

   A tuple containing the extra `custom_classical` functions that Qiskit's legacy built-in
   converters use beyond those specified by the paper.  This is the three basic inverse
   trigonometric functions: :math:`\asin`, :math:`\acos` and :math:`\atan`.

.. py:data:: LEGACY_INCLUDE_PATH

   A tuple containing the exact `include_path` used by the legacy Qiskit converter.

On *all* the gates defined in Qiskit's legacy version of ``qelib1.inc`` and the ``delay``
instruction, it does not matter how the gates are actually defined and used, the legacy importer
will always attempt to output its custom objects for them.  This can result in errors during the
circuit construction, even after a successful parse.  There is no way to emulate this buggy
behavior with :mod:`qiskit.qasm2`; only an ``include "qelib1.inc";`` statement or the
`custom_instructions` argument can cause built-in Qiskit instructions to be used, and the signatures
of these match each other.

.. note::

   Circuits imported with :func:`load` and :func:`loads` with the above legacy-compatibility settings
   should compare equal to those created by Qiskit's legacy importer, provided no non-``qelib1.inc``
   user gates are defined.  User-defined gates are handled slightly differently in the new importer,
   and while they should have equivalent :attr:`~.Instruction.definition` fields on inspection, this
   module uses a custom class to lazily load the definition when it is requested (like most Qiskit
   objects), rather than eagerly creating it during the parse.  Qiskit's comparison rules for gates
   will see these two objects as unequal, although any pass through :func:`.transpile` for a
   particular backend should produce the same output circuits.


.. _qasm2-alternatives:

Alternatives
============

The parser components of this module started off as a separate PyPI package: `qiskit-qasm2
<https://pypi.org/project/qiskit-qasm2/>`__.  This package at version 0.5.3 was vendored into Qiskit
Terra 0.24.  Any subsequent changes between the two packages may not necessarily be kept in sync.

There is a newer version of the OpenQASM specification, version 3.0, which is described at
https://openqasm.com.  This includes far more facilities for high-level classical programming.
Qiskit has some rudimentary support for OpenQASM 3 already; see :mod:`qiskit.qasm3` for that.

OpenQASM 2 is not a suitable serialization language for Qiskit's :class:`.QuantumCircuit`.  This
module is provided for interoperability purposes, not as a general serialization format.  If that is
what you need, consider using :mod:`qiskit.qpy` instead.
'''

__all__ = [
    "LEGACY_CUSTOM_CLASSICAL",
    "LEGACY_CUSTOM_INSTRUCTIONS",
    "LEGACY_INCLUDE_PATH",
    "CustomClassical",
    "CustomInstruction",
    "QASM2Error",
    "QASM2ExportError",
    "QASM2ParseError",
    "dump",
    "dumps",
    "load",
    "loads",
]

import os
from pathlib import Path
from typing import Literal
from collections.abc import Iterable


from qiskit._accelerate import qasm2 as _qasm2
from qiskit.circuit import QuantumCircuit
from . import parse as _parse
from .exceptions import QASM2Error, QASM2ParseError, QASM2ExportError
from .parse import (
    CustomInstruction,
    CustomClassical,
    LEGACY_CUSTOM_INSTRUCTIONS,
    LEGACY_CUSTOM_CLASSICAL,
)
from .export import dump, dumps


LEGACY_INCLUDE_PATH = (
    Path(__file__).parents[1] / "qasm" / "libs",
    # This is deliberately left as a relative current-directory import until call time, so it
    # respects changes the user might make from within the interpreter.
    Path("."),
)


def _normalize_path(path: str | os.PathLike) -> str:
    """Normalize a given path into a path-like object that can be passed to Rust.

    Ideally this would be something that we can convert to Rust's `OSString`, but in practice,
    Python uses `os.fsencode` to produce a `bytes` object, but this doesn't map especially well.
    """
    path = Path(path).expanduser().absolute()
    if not path.exists():
        raise FileNotFoundError(str(path))
    return str(path)


def loads(
    string: str,
    *,
    include_path: Iterable[str | os.PathLike] = (".",),
    custom_instructions: Iterable[CustomInstruction] = (),
    custom_classical: Iterable[CustomClassical] = (),
    strict: bool = False,
) -> QuantumCircuit:
    """Parse an OpenQASM 2 program from a string into a :class:`.QuantumCircuit`.

    Args:
        string: The OpenQASM 2 program in a string.
        include_path: order of directories to search when evaluating ``include`` statements.
        custom_instructions: any custom constructors that should be used for specific gates or
            opaque instructions during circuit construction.  See :ref:`qasm2-custom-instructions`
            for more.
        custom_classical: any custom classical functions that should be used during the parsing of
            classical expressions.  See :ref:`qasm2-custom-classical` for more.
        strict: whether to run in :ref:`strict mode <qasm2-strict-mode>`.

    Returns:
        A circuit object representing the same OpenQASM 2 program.
    """
    custom_instructions = list(custom_instructions)
    return _parse.from_bytecode(
        _qasm2.bytecode_from_string(
            string,
            [_normalize_path(path) for path in include_path],
            [
                _qasm2.CustomInstruction(x.name, x.num_params, x.num_qubits, x.builtin)
                for x in custom_instructions
            ],
            tuple(custom_classical),
            strict,
        ),
        custom_instructions,
    )


def load(
    filename: str | os.PathLike,
    *,
    include_path: Iterable[str | os.PathLike] = (".",),
    include_input_directory: Literal["append", "prepend"] | None = "append",
    custom_instructions: Iterable[CustomInstruction] = (),
    custom_classical: Iterable[CustomClassical] = (),
    strict: bool = False,
) -> QuantumCircuit:
    """Parse an OpenQASM 2 program from a file into a :class:`.QuantumCircuit`.  The given path
    should be ASCII or UTF-8 encoded, and contain the OpenQASM 2 program.

    Args:
        filename: The path to the OpenQASM 2 file.
        include_path: order of directories to search when evaluating ``include`` statements.
        include_input_directory: Whether to add the directory of the input file to the
            ``include_path``, and if so, whether to *append* it to search last, or *prepend* it to
            search first.  Pass ``None`` to suppress adding this directory entirely.
        custom_instructions: any custom constructors that should be used for specific gates or
            opaque instructions during circuit construction.  See :ref:`qasm2-custom-instructions`
            for more.
        custom_classical: any custom classical functions that should be used during the parsing of
            classical expressions.  See :ref:`qasm2-custom-classical` for more.
        strict: whether to run in :ref:`strict mode <qasm2-strict-mode>`.

    Returns:
        A circuit object representing the same OpenQASM 2 program.
    """
    filename = Path(filename)
    include_path = [_normalize_path(path) for path in include_path]
    if include_input_directory == "append":
        include_path.append(str(filename.parent))
    elif include_input_directory == "prepend":
        include_path.insert(0, str(filename.parent))
    elif include_input_directory is not None:
        raise ValueError(
            f"unknown value for include_input_directory: '{include_input_directory}'."
            " Valid values are '\"append\"', '\"prepend\"' and 'None'."
        )
    custom_instructions = tuple(custom_instructions)
    return _parse.from_bytecode(
        _qasm2.bytecode_from_file(
            _normalize_path(filename),
            include_path,
            [
                _qasm2.CustomInstruction(x.name, x.num_params, x.num_qubits, x.builtin)
                for x in custom_instructions
            ],
            tuple(custom_classical),
            strict,
        ),
        custom_instructions,
    )
