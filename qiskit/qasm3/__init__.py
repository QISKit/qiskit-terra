# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
================================
OpenQASM 3 (:mod:`qiskit.qasm3`)
================================

.. currentmodule:: qiskit.qasm3

Qiskit provides some tools for converting between `OpenQASM 3 <https://openqasm.com>`__
representations of quantum programs, and the :class:`.QuantumCircuit` class.  These will continue to
evolve as Qiskit's support for the dynamic-circuit capabilities expressed by OpenQASM 3 increases.


Exporting to OpenQASM 3
=======================

The high-level functions are simply :func:`dump` and :func:`dumps`, which respectively export to a
file (given as a filename) and to a Python string.

.. autofunction:: dump
.. autofunction:: dumps

Both of these exporter functions are single-use wrappers around the main :class:`Exporter` class.
For more complex exporting needs, including dumping multiple circuits in a single session, it may be
more convenient or faster to use the complete interface.

.. autoclass:: Exporter
    :members:

All of these interfaces will raise :exc:`QASM3ExporterError` on failure.

.. autoexception:: QASM3ExporterError

Experimental features
---------------------

The OpenQASM 3 language is still evolving as hardware capabilities improve, so there is no final
syntax that Qiskit can reliably target.  In order to represent the evolving language, we will
sometimes release features before formal standardization, which may need to change as the review
process in the OpenQASM 3 design committees progresses.  By default, the exporters will only support
standardised features of the language.  To enable these early-release features, use the
``experimental`` keyword argument of :func:`dump` and :func:`dumps`.  The available feature flags
are:

.. autoclass:: ExperimentalFeatures
    :members:

If you want to enable multiple experimental features, you should combine the flags using the ``|``
operator, such as ``flag1 | flag2``.

For example, to perform an export using the early semantics of ``switch`` support:

.. plot::
    :include-source:
    :nofigs:

    from qiskit import qasm3, QuantumCircuit, QuantumRegister, ClassicalRegister

    # Build the circuit
    qreg = QuantumRegister(3)
    creg = ClassicalRegister(3)
    qc = QuantumCircuit(qreg, creg)
    with qc.switch(creg) as case:
        with case(0):
            qc.x(0)
        with case(1, 2):
            qc.x(1)
        with case(case.DEFAULT):
            qc.x(2)

    # Export to an OpenQASM 3 string.
    qasm_string = qasm3.dumps(qc, experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1)


.. note::

    All features enabled by the experimental flags are naturally transient.  If it becomes necessary
    to remove flags, they will be subject to `the standard Qiskit deprecation policy
    <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`__.  We will leave these experimental
    flags in place for as long as is reasonable.

    However, we cannot guarantee any support windows for *consumers* of OpenQASM 3 code generated
    using these experimental flags, if the OpenQASM 3 language specification changes the proposal
    that the flag is based on.  It is possible that any tool you are using to consume OpenQASM 3
    code created using these flags may update or remove their support while Qiskit continues to
    offer the flag.  You should not rely on the resultant experimental OpenQASM 3 code for long-term
    storage of programs.


Importing from OpenQASM 3
=========================

Currently only two high-level functions are offered, as Qiskit support for importing from OpenQASM 3
is in its infancy, and the implementation is expected to change significantly.  The two functions
are :func:`load` and :func:`loads`, which are direct counterparts of :func:`dump` and :func:`dumps`,
respectively loading a program indirectly from a named file and directly from a given string.

.. note::

    While we are still in the exploratory release period, to use either function, the package
    ``qiskit_qasm3_import`` must be installed.  This can be done by installing Qiskit with the
    ``qasm3-import`` extra, such as by:

    .. code-block:: text

        pip install qiskit[qasm3-import]

    We expect that this functionality will eventually be merged into Qiskit, and no longer
    require an optional import, but we do not yet have a timeline for this.

.. autofunction:: load
.. autofunction:: loads

Both of these two functions raise :exc:`QASM3ImporterError` on failure.

.. autoexception:: QASM3ImporterError

For example, we can define a quantum program using OpenQASM 3, and use :func:`loads` to directly
convert it into a :class:`.QuantumCircuit`:

.. plot::
    :alt: Circuit diagram output by the previous code.
    :include-source:

    import qiskit.qasm3

    program = \"\"\"
        OPENQASM 3.0;
        include "stdgates.inc";

        input float[64] a;
        qubit[3] q;
        bit[2] mid;
        bit[3] out;

        let aliased = q[0:1];

        gate my_gate(a) c, t {
          gphase(a / 2);
          ry(a) c;
          cx c, t;
        }
        gate my_phase(a) c {
          ctrl @ inv @ gphase(a) c;
        }

        my_gate(a * 2) aliased[0], q[{1, 2}][0];
        measure q[0] -> mid[0];
        measure q[1] -> mid[1];

        while (mid == "00") {
          reset q[0];
          reset q[1];
          my_gate(a) q[0], q[1];
          my_phase(a - pi/2) q[1];
          mid[0] = measure q[0];
          mid[1] = measure q[1];
        }

        if (mid[0]) {
          let inner_alias = q[{0, 1}];
          reset inner_alias;
        }

        out = measure q;
    \"\"\"
    circuit = qiskit.qasm3.loads(program)
    circuit.draw("mpl")


Experimental import interface
-----------------------------

The import functions given above rely on the ANTLR-based reference parser from the OpenQASM project
itself, which is more intended as a language reference than a performant parser.  You need to have
the extension ``qiskit-qasm3-import`` installed to use it.

Qiskit is developing a native parser, written in Rust, which is available as part of the core Qiskit
package.  This parser is still in its early experimental stages, so is missing features and its
interface is changing and expanding, but it is typically orders of magnitude more performant for the
subset of OpenQASM 3 it currently supports, and its internals produce better error diagnostics on
parsing failures.

You can use the experimental interface immediately, with similar functions to the main interface
above:

.. autofunction:: load_experimental
.. autofunction:: loads_experimental

These two functions are both experimental, meaning they issue an :exc:`.ExperimentalWarning` on
usage, and their interfaces may be subject to change within the Qiskit 1.x release series.  In
particular, the native parser may be promoted to be the default version of :func:`load` and
:func:`loads`.  If you are happy to accept the risk of using the experimental interface, you can
disable the warning by doing::

    import warnings
    from qiskit.exceptions import ExperimentalWarning

    warnings.filterwarnings("ignore", category=ExperimentalWarning, module="qiskit.qasm3")

These two functions allow for specifying include paths as an iterable of paths, and for specifying
custom Python constructors to use for particular gates.  These custom constructors are specified by
using the :class:`CustomGate` object:

.. autoclass:: CustomGate
    :members:

If ``custom_gates`` is not given, Qiskit will attempt to use its standard-library gate objects for
the gates defined in OpenQASM 3 standard library file ``stdgates.inc``.  This sequence of gates is
available on this module, if you wish to build on top of it:

.. py:data:: STDGATES_INC_GATES

    A tuple of :class:`CustomGate` objects specifying the Qiskit constructors to use for the
    ``stdgates.inc`` include file.
"""

from __future__ import annotations

import functools
import typing
import warnings

from qiskit._accelerate import qasm3 as _qasm3
from qiskit.circuit import library
from qiskit.exceptions import ExperimentalWarning
from qiskit.utils import optionals as _optionals
from qiskit.circuit import Qubit

from .._accelerate.qasm3 import CustomGate
from .exceptions import QASM3Error, QASM3ExporterError, QASM3ImporterError
from .experimental import ExperimentalFeatures
from .exporter import Exporter, DefcalInstruction

__all__ = [
    "CustomGate",
    "DefcalInstruction",
    "ExperimentalFeatures",
    "QASM3Error",
    "QASM3ExporterError",
    "QASM3ImporterError",
    "dump",
    "dump_experimental",
    "dumps",
    "dumps_experimental",
    "load",
    "load_experimental",
    "loads",
    "loads_experimental",
]

if typing.TYPE_CHECKING:
    from qiskit.circuit import annotation, QuantumCircuit

STDGATES_INC_GATES = (
    CustomGate(library.PhaseGate, "p", 1, 1),
    CustomGate(library.XGate, "x", 0, 1),
    CustomGate(library.YGate, "y", 0, 1),
    CustomGate(library.ZGate, "z", 0, 1),
    CustomGate(library.HGate, "h", 0, 1),
    CustomGate(library.SGate, "s", 0, 1),
    CustomGate(library.SdgGate, "sdg", 0, 1),
    CustomGate(library.TGate, "t", 0, 1),
    CustomGate(library.TdgGate, "tdg", 0, 1),
    CustomGate(library.SXGate, "sx", 0, 1),
    CustomGate(library.RXGate, "rx", 1, 1),
    CustomGate(library.RYGate, "ry", 1, 1),
    CustomGate(library.RZGate, "rz", 1, 1),
    CustomGate(library.CXGate, "cx", 0, 2),
    CustomGate(library.CYGate, "cy", 0, 2),
    CustomGate(library.CZGate, "cz", 0, 2),
    CustomGate(library.CPhaseGate, "cp", 1, 2),
    CustomGate(library.CRXGate, "crx", 1, 2),
    CustomGate(library.CRYGate, "cry", 1, 2),
    CustomGate(library.CRZGate, "crz", 1, 2),
    CustomGate(library.CHGate, "ch", 0, 2),
    CustomGate(library.SwapGate, "swap", 0, 2),
    CustomGate(library.CCXGate, "ccx", 0, 3),
    CustomGate(library.CSwapGate, "cswap", 0, 3),
    CustomGate(library.CUGate, "cu", 4, 2),
    CustomGate(library.CXGate, "CX", 0, 2),
    CustomGate(library.PhaseGate, "phase", 1, 1),
    CustomGate(library.CPhaseGate, "cphase", 1, 2),
    CustomGate(library.IGate, "id", 0, 1),
    CustomGate(library.U1Gate, "u1", 1, 1),
    CustomGate(library.U2Gate, "u2", 2, 1),
    CustomGate(library.U3Gate, "u3", 3, 1),
)


def dumps(circuit, **kwargs) -> str:
    """Serialize a :class:`~qiskit.circuit.QuantumCircuit` object in an OpenQASM 3 string.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.
        **kwargs: Arguments for the :obj:`.Exporter` constructor.

    Returns:
        str: The OpenQASM 3 serialization
    """
    return Exporter(**kwargs).dumps(circuit)


def dump(circuit, stream, **kwargs) -> None:
    """Serialize a :class:`~qiskit.circuit.QuantumCircuit` object as an OpenQASM 3 stream to
    file-like object.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.
        stream (TextIOBase): stream-like object to dump the OpenQASM 3 serialization
        **kwargs: Arguments for the :obj:`.Exporter` constructor.

    """
    Exporter(**kwargs).dump(circuit, stream)


@_optionals.HAS_QASM3_IMPORT.require_in_call("loading from OpenQASM 3")
def load(
    filename: str,
    *,
    num_qubits: int | None = None,
    annotation_handlers: dict[str, annotation.OpenQASM3Serializer] | None = None,
) -> QuantumCircuit:
    """Load an OpenQASM 3 program from the file ``filename``.

    Args:
        filename: the filename to load the program from.
        num_qubits: keyword argument which provides number of physical/virtual qubits.
        annotation_handlers: a mapping whose keys are (parent) namespaces and values are serializers
            that can handle children of those namespaces.  Requires ``qiskit_qasm3_import>=0.6.0``.
    Returns:
        QuantumCircuit: a circuit representation of the OpenQASM 3 program.

    Raises:
        QASM3ImporterError: if the OpenQASM 3 file is invalid, or cannot be represented by a
            :class:`.QuantumCircuit`.

    .. versionadded:: 2.1
        The ``annotation_handlers`` argument.  This requires ``qiskit_qasm3_import>=0.6.0``.
    """

    with open(filename) as fptr:
        program = fptr.read()
    return loads(program, num_qubits=num_qubits, annotation_handlers=annotation_handlers)


@_optionals.HAS_QASM3_IMPORT.require_in_call("loading from OpenQASM 3")
def loads(
    program: str,
    *,
    num_qubits: int | None = None,
    annotation_handlers: dict[str, annotation.OpenQASM3Serializer] | None = None,
) -> QuantumCircuit:
    """Load an OpenQASM 3 program from the given string.

    Examples:

        Load a OpenQASM3 string into a quantum circuit with/without `num_qubits` argument.

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit import qasm3

           # An OpenQASM 3 program that only uses 2 physical qubits.
           prog = '''
               OPENQASM 3.0;
               include "stdgates.inc";
               h $0;
               cx $0, $1;
           '''
           # The importer can be supplied with the number of qubits in the target backend.
           # so the result is full width.
           qc = qasm3.loads(prog, num_qubits=5)
           assert qc.num_qubits == 5

    Args:
        program: the OpenQASM 3 program.
        num_qubits: provides number of physical/virtual qubits.
        annotation_handlers: a mapping whose keys are (parent) namespaces and values are serializers
            that can handle children of those namespaces.  Requires ``qiskit_qasm3_import>=0.6.0``.
    Returns:
        QuantumCircuit: a circuit representation of the OpenQASM 3 program.

    Raises:
        QASM3ImporterError: if the OpenQASM 3 file is invalid, or cannot be represented by a
            :class:`.QuantumCircuit`.
        ValueError: if number of qubits in qasm3_ckt is more than num_qubits.

    .. versionadded:: 2.1
        The ``annotation_handlers`` argument.  This requires ``qiskit_qasm3_import>=0.6.0``.
    """

    import qiskit_qasm3_import

    kwargs = {}
    if annotation_handlers is not None:
        if getattr(qiskit_qasm3_import, "VERSION_PARTS", (0, 0)) < (0, 6):
            raise QASM3ImporterError(
                "need 'qiskit_qasm3_import>=0.6.0' to handle annotations, but you have "
                + qiskit_qasm3_import.__version__
            )
        kwargs["annotation_handlers"] = annotation_handlers

    try:
        qasm3_ckt = qiskit_qasm3_import.parse(program, **kwargs)
    except qiskit_qasm3_import.ConversionError as exc:
        raise QASM3ImporterError(str(exc)) from exc

    if num_qubits is not None:
        if qasm3_ckt.num_qubits > num_qubits:
            raise ValueError(
                "Number of qubits cannot be more than the provided number of physical/virtual qubits."
            )
        if (difference := num_qubits - qasm3_ckt.num_qubits) > 0:
            if qasm3_ckt.layout is not None:
                # stripping circuit of layout
                layout, qasm3_ckt._layout = qasm3_ckt._layout, None

                # updating layout
                last = max(layout.initial_layout.get_physical_bits().keys(), default=-1)
                for k in range(last + 1, num_qubits):
                    layout.initial_layout.add(Qubit(), k)

                # updating input_qubit_mapping
                mapping_dict = {}
                for ii in range(last + 1, num_qubits):
                    mapping_dict[layout.initial_layout[ii]] = ii
                layout.input_qubit_mapping.update(mapping_dict)

                # adding qubits to circuit
                qasm3_ckt.add_bits([Qubit() for _ in range(difference)])

                # restoring layout
                qasm3_ckt._layout = layout

            else:
                # adding qubits to circuit
                qasm3_ckt.add_bits([Qubit() for _ in range(difference)])

    return qasm3_ckt


@functools.wraps(_qasm3.loads)
def loads_experimental(source, /, *, custom_gates=None, include_path=None):
    """<overridden by functools.wraps>"""
    warnings.warn(
        "This is an experimental native version of the OpenQASM 3 importer."
        " Beware that its interface might change, and it might be missing features.",
        category=ExperimentalWarning,
    )
    return _qasm3.loads(source, custom_gates=custom_gates, include_path=include_path)


@functools.wraps(_qasm3.load)
def load_experimental(pathlike_or_filelike, /, *, custom_gates=None, include_path=None):
    """<overridden by functools.wraps>"""
    warnings.warn(
        "This is an experimental native version of the OpenQASM 3 importer."
        " Beware that its interface might change, and it might be missing features.",
        category=ExperimentalWarning,
    )
    return _qasm3.load(pathlike_or_filelike, custom_gates=custom_gates, include_path=include_path)


@functools.wraps(_qasm3.dumps)
def dumps_experimental(
    circuit,
    *,
    includes=None,
    basis_gates=None,
    disable_constants=True,
    alias_classical_registers=False,
    allow_aliasing=False,
    indent="  ",
):
    """<overridden by functools.wraps>"""
    warnings.warn(
        "This is an experimental version of serialization of a :class:`~qiskit.circuit.QuantumCircuit`"
        " object in an OpenQASM 3 string."
        " Beware that its interface might change, and it might be missing features.",
        category=ExperimentalWarning,
    )
    if includes is None:
        includes = ["stdgates.inc"]
    if basis_gates is None:
        basis_gates = ["U"]
    return _qasm3.dumps(
        circuit,
        {
            "includes": includes,
            "basis_gates": basis_gates,
            "disable_constants": disable_constants,
            "alias_classical_registers": alias_classical_registers,
            "allow_aliasing": allow_aliasing,
            "indent": indent,
        },
    )


@functools.wraps(_qasm3.dump)
def dump_experimental(
    circuit,
    stream,
    *,
    includes=None,
    basis_gates=None,
    disable_constants=True,
    alias_classical_registers=False,
    allow_aliasing=False,
    indent="  ",
):
    """<overridden by functools.wraps>"""
    warnings.warn(
        "This is an experimental version of serialization of a :class:`~qiskit.circuit.QuantumCircuit`"
        " object in an OpenQASM 3 string."
        " Beware that its interface might change, and it might be missing features.",
        category=ExperimentalWarning,
    )
    if includes is None:
        includes = ["stdgates.inc"]
    if basis_gates is None:
        basis_gates = ["U"]

    return _qasm3.dump(
        circuit,
        stream,
        {
            "includes": includes,
            "basis_gates": basis_gates,
            "disable_constants": disable_constants,
            "alias_classical_registers": alias_classical_registers,
            "allow_aliasing": allow_aliasing,
            "indent": indent,
        },
    )
