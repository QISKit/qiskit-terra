# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=====================================
QPY serialization (:mod:`qiskit.qpy`)
=====================================

.. currentmodule:: qiskit.qpy

QPY is a binary serialization format for :class:`~.QuantumCircuit`
objects that is designed to be cross-platform, Python version agnostic,
and backwards compatible moving forward. QPY should be used if you need
a mechanism to save or copy between systems a :class:`~.QuantumCircuit`
that preserves the full Qiskit object structure (except for custom attributes
defined outside of Qiskit code). This differs from other serialization formats like
`OpenQASM <https://github.com/openqasm/openqasm>`__ (2.0 or 3.0) which has a
different abstraction model and can result in a loss of information contained
in the original circuit (or is unable to represent some aspects of the
Qiskit objects) or Python's `pickle <https://docs.python.org/3/library/pickle.html>`__
which will preserve the Qiskit object exactly but will only work for a single Qiskit
version (it is also
`potentially insecure <https://docs.python.org/3/library/pickle.html#module-pickle>`__).

Basic Usage
===========

Using QPY is defined to be straightforward and mirror the user API of the
serializers in Python's standard library, ``pickle`` and ``json``. There are
2 user facing functions: :func:`qiskit.qpy.dump` and
:func:`qiskit.qpy.load` which are used to dump QPY data
to a file object and load circuits from QPY data in a file object respectively.
For example:

.. plot::
    :nofigs:
    :context: reset

    # This code is hidden from users
    # It's a hack to avoid writing to file when testing the code examples
    import io
    bytestream = io.BytesIO()
    bytestream.close = lambda: bytestream.seek(0)
    def open(*args):
        return bytestream

.. plot::
    :include-source:
    :nofigs:
    :context:

    from qiskit.circuit import QuantumCircuit
    from qiskit import qpy

    qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    with open('bell.qpy', 'wb') as fd:
        qpy.dump(qc, fd)

    with open('bell.qpy', 'rb') as fd:
        new_qc = qpy.load(fd)[0]

The :func:`qiskit.qpy.dump` function also lets you
include multiple circuits in a single QPY file:

.. plot::
    :include-source:
    :nofigs:
    :context:

    with open('twenty_bells.qpy', 'wb') as fd:
        qpy.dump([qc] * 20, fd)

and then loading that file will return a list with all the circuits

.. plot::
    :include-source:
    :nofigs:
    :context:

    with open('twenty_bells.qpy', 'rb') as fd:
        twenty_new_bells = qpy.load(fd)


API documentation
=================

.. autofunction:: load
.. autofunction:: dump
.. autofunction:: get_qpy_version

These functions will raise a custom subclass of :exc:`.QiskitError` if they encounter problems
during serialization or deserialization.

.. autoexception:: QpyError

When a lower-than-maximum target QPY version is set for serialization, but the object to be
serialized contains features that cannot be represented in that format, a subclass of
:exc:`QpyError` is raised:

.. autoexception:: UnsupportedFeatureForVersion

Attributes:
    QPY_VERSION (int): The current QPY format version as of this release. This
        is the default value of the ``version`` keyword argument on
        :func:`.qpy.dump` and also the upper bound for accepted values for
        the same argument. This is also the upper bond on the versions supported
        by :func:`.qpy.load`.

    QPY_COMPATIBILITY_VERSION (int): The current minimum compatibility QPY
        format version. This is the minimum version that :func:`.qpy.dump`
        will accept for the ``version`` keyword argument. :func:`.qpy.load`
        will be able to load all released format versions of QPY (up until
        ``QPY_VERSION``).

.. _qpy_compatibility:

QPY Compatibility
=================

The QPY format is designed to be backwards compatible moving forward. This means
you should be able to load a QPY with any newer Qiskit version than the one
that generated it. However, loading a QPY file with an older Qiskit version is
not supported and may not work.

For example, if you generated a QPY file using qiskit-terra 0.18.1 you could
load that QPY file with qiskit-terra 0.19.0 and a hypothetical qiskit-terra
0.29.0. However, loading that QPY file with 0.18.0 is not supported and may not
work.

Note that circuit metadata and custom :class:`.Annotation` objects are serialized and deserialized
by user-supplied classes, as the objects themselves are completely user-custom, so the forwards- and
backwards-compatibility of these is limited by what the user provides.

If a feature being loaded is deprecated in the corresponding qiskit release, QPY will
raise a :exc:`~.QPYLoadingDeprecatedFeatureWarning` informing of the deprecation period
and how the feature will be internally handled.

.. autoexception:: QPYLoadingDeprecatedFeatureWarning

.. note::

    With versions of Qiskit before 1.2.4, the ``use_symengine=True`` argument to :func:`.qpy.dump`
    could cause problems with backwards compatibility if there were :class:`.ParameterExpression`
    objects to serialize.  In particular:

    * When the loading version of Qiskit is 1.2.4 or greater, QPY files generated with any version
      of Qiskit >= 0.46.0 can be loaded.  If a version of Qiskit between 0.45.0 and 0.45.3 was used
      to generate the files, and the non-default argument ``use_symengine=True`` was given to
      :func:`.qpy.dump`, the file can only be read if the version of ``symengine`` used in the
      generating environment was in the 0.11 or 0.13 series, but if the environment was created
      during the support window of Qiskit 0.45, it is likely that ``symengine==0.9.2`` was used.

    * When the loading version of Qiskit is between 0.46.0 and 1.2.2 inclusive, the file can only be
      read if the installed version of ``symengine`` in the loading environment matches the version
      used in the generating environment.

    To recover a QPY file that fails with ``symengine`` version-related errors during a call to
    :func:`.qpy.load`, first attempt to use Qiskit >= 1.2.4 to load the file.  If this still fails,
    it is likely because Qiskit 0.45.x was used to generate the file with ``use_symengine=True``.
    In this case, use Qiskit 0.45.3 with ``symengine==0.9.2`` to load the file, and then re-export
    it to QPY setting ``use_symengine=False``.  The resulting file can then be loaded by any later
    version of Qiskit.

.. note::

    Starting with Qiskit version 2.0.0, which removed the Pulse module from the library, QPY provides
    limited support for loading payloads that include pulse data. Loading a ``ScheduleBlock`` payload,
    a :class:`.QpyError` exception will be raised. Loading a payload for a circuit that contained pulse
    gates, the output circuit will contain  custom instructions **without** calibration data attached
    for each pulse gate, leaving them undefined.

QPY format version history
--------------------------

If you're planning to load a QPY file between different Qiskit versions knowing
which versions were available in a given release are useful. As the QPY is
backwards compatible but not forwards compatible you need to ensure a given
QPY format version was released in the release you're calling :func:`.load`
with. The following table lists the QPY versions that were supported in every
Qiskit (and qiskit-terra prior to Qiskit 1.0.0) release going back to the introduction
of QPY in qiskit-terra 0.18.0.

.. list-table:: QPY Format Version History
   :header-rows: 1

   * - Qiskit (qiskit-terra for < 1.0.0) version
     - :func:`.dump` format(s) output versions
     - :func:`.load` maximum supported version (older format versions can always be read)
   * - 2.5.0
     - 13, 14, 15, 16, 17
     - 17
   * - 2.4.1
     - 13, 14, 15, 16, 17
     - 17
   * - 2.4.0
     - 13, 14, 15, 16, 17
     - 17
   * - 2.3.1
     - 13, 14, 15, 16, 17
     - 17
   * - 2.3.0
     - 13, 14, 15, 16, 17
     - 17
   * - 2.2.2
     - 13, 14, 15, 16
     - 16
   * - 2.2.1
     - 13, 14, 15, 16
     - 16
   * - 2.2.0
     - 13, 14, 15, 16
     - 16
   * - 2.1.2
     - 13, 14, 15, 16
     - 16
   * - 2.1.1
     - 13, 14, 15, 16
     - 16
   * - 2.1.0
     - 13, 14, 15
     - 15
   * - 2.0.2
     - 13, 14
     - 14
   * - 2.0.1
     - 13, 14
     - 14
   * - 2.0.0
     - 13, 14
     - 14
   * - 1.4.3
     - 10, 11, 12, 13
     - 13
   * - 1.4.2
     - 10, 11, 12, 13
     - 13
   * - 1.4.1
     - 10, 11, 12, 13
     - 13
   * - 1.4.0
     - 10, 11, 12, 13
     - 13
   * - 1.3.3
     - 10, 11, 12, 13
     - 13
   * - 1.3.2
     - 10, 11, 12, 13
     - 13
   * - 1.3.1
     - 10, 11, 12, 13
     - 13
   * - 1.3.0
     - 10, 11, 12, 13
     - 13
   * - 1.2.4
     - 10, 11, 12
     - 12
   * - 1.2.3 (yanked)
     - 10, 11, 12
     - 12
   * - 1.2.2
     - 10, 11, 12
     - 12
   * - 1.2.1
     - 10, 11, 12
     - 12
   * - 1.2.0
     - 10, 11, 12
     - 12
   * - 1.1.0
     - 10, 11, 12
     - 12
   * - 1.0.2
     - 10, 11
     - 11
   * - 1.0.1
     - 10, 11
     - 11
   * - 1.0.0
     - 10, 11
     - 11
   * - 0.46.1
     - 10
     - 10
   * - 0.45.3
     - 10
     - 10
   * - 0.45.2
     - 10
     - 10
   * - 0.45.1
     - 10
     - 10
   * - 0.45.0
     - 10
     - 10
   * - 0.25.3
     - 9
     - 9
   * - 0.25.2
     - 9
     - 9
   * - 0.25.1
     - 9
     - 9
   * - 0.24.2
     - 8
     - 8
   * - 0.24.1
     - 7
     - 7
   * - 0.24.0
     - 7
     - 7
   * - 0.23.3
     - 6
     - 6
   * - 0.23.2
     - 6
     - 6
   * - 0.23.1
     - 6
     - 6
   * - 0.23.0
     - 6
     - 6
   * - 0.22.4
     - 5
     - 5
   * - 0.22.3
     - 5
     - 5
   * - 0.22.2
     - 5
     - 5
   * - 0.22.1
     - 5
     - 5
   * - 0.22.0
     - 5
     - 5
   * - 0.21.2
     - 5
     - 5
   * - 0.21.1
     - 5
     - 5
   * - 0.21.0
     - 5
     - 5
   * - 0.20.2
     - 4
     - 4
   * - 0.20.1
     - 4
     - 4
   * - 0.20.0
     - 4
     - 4
   * - 0.19.2
     - 4
     - 4
   * - 0.19.1
     - 3
     - 3
   * - 0.19.0
     - 2
     - 2
   * - 0.18.3
     - 1
     - 1
   * - 0.18.2
     - 1
     - 1
   * - 0.18.1
     - 1
     - 1
   * - 0.18.0
     - 1
     - 1

.. _qpy_format:

QPY Format
==========

The QPY serialization format is a portable cross-platform binary
serialization format for :class:`~qiskit.circuit.QuantumCircuit` objects in Qiskit. The basic
file format is as follows:

A QPY file (or memory object) always starts with the following 6
byte UTF8 string: ``QISKIT`` which is immediately followed by the overall
file header. The contents of the file header as defined as a C struct are:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
        uint64_t num_circuits;
    }


From V10 on, a new field is added to the file header struct to represent the
encoding scheme used for symbolic expressions:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
        uint64_t num_circuits;
        char symbolic_encoding;
    }

From V16 on, the file header struct is immediately followed by a circuit start table
containing the byte offsets of each circuit payload in the file. There are ``num_circuits``
entries in the circuit start table, each of which is of type ``uint64_t``. In all previous
versions, the file header is immediately followed by the circuit payloads in sequence
without any padding in-between.

All values use network byte order [#f1]_ (big endian) for cross platform
compatibility. The exception to this is for QPY format versions <= 17 the encoding of
integers and floats as part of ``INSTRUCTION_PARAM`` is little endian.

Each individual circuit is composed of the following parts in order from top to bottom:

.. code-block:: text

    HEADER
    METADATA
    REGISTERS
    ANNOTATION_HEADER
    STANDALONE_VARS
    CUSTOM_DEFINITIONS
    INSTRUCTIONS

.. versionchanged:: QPY 15
    ``ANNOTATION_HEADER`` was added between ``REGISTERS`` and ``STANDALONE_VARS``.

.. versionchanged:: QPY 12
    ``STANDALONE_VARS`` was added between ``REGISTERS`` and ``CUSTOM_DEFINITIONS``.

There is a circuit payload for each circuit (where the total number is dictated
by ``num_circuits`` in the file header). There is no padding between the
circuits in the data.

.. _qpy_version_17:

Version 17
----------

Version 17 adds support for serializing and deserializing :class:`.PauliEvolutionGate` containing
:class:`.SparseObservable` as operator(s).

Changes to PAULI_EVOLUTION
~~~~~~~~~~~~~~~~~~~~~~~~~~

The format of `PAULI_EVOLUTION` itself remains unchanged, but the format of the operators
that immediately follow the packed evolution gate is updated. Instead of `operator_count`
elements defined by `SPARSE_PAULI_OP_LIST_ELEM` format, the payload now specifies the type of
each operator to account for either operators of type :class:`.SparsePauliOp` or of
:class:`.SparseObservable`.

The new payload following `PAULI_EVOLUTION` now contains exactly `operator_count` sequences of
a bool (``"!?"``) followed by the operator. If the bool is ``True``, the operator is a
:class:`.SparseObservable` and interpreted with the new `SPARSE_OBSERVABLE` format (see below).
If it is ``False``, the operator is a :class:`.SparsePauliOp` and interpreted according to the
existing `SPARSE_PAULI_OP_LIST_ELEM` format.

New SPARSE_OBSERVABLE
~~~~~~~~~~~~~~~~~~~~~

The `SPARSE_OBSERVABLE` format represents an instance of a :class:`.SparseObservable`, given by

.. code-block:: c

  struct {
    uint32_t num_qubits;
    uint64_t coeff_data_len;
    uint64_t bitterm_data_len;
    uint64_t inds_data_len;
    uint64_t bounds_data_len;
  }

which is immediately followed by the number of qubits and then the data arrays of the
coefficients, bit terms, indices, and boundaries of the observable. The format specifies the
number of bytes each array occupies. The number of elements can be calculated by dividing
the number of bytes by the size of each element.

 * Each coefficient is stored as two consecutive `"!d"` elements, first the real and then
   the imaginary part.
 * The bit term elements are of type `"!H"` and represents the `u8` value of the
   :class:`.SparseObservable.BitTerm`
 * The indices elements are of type `"!I"`.
 * The boundaries elements are of type `"!Q"`.

.. _qpy_version_16:

Version 16
----------

Version 16 adds a circuit start table to the QPY file format. It serves as an index of the
byte offsets of each circuit payload in the file. The motivation for this change is to
enable a more efficient loading of circuits using multi-threading in a future Rust
implementation of the QPY deserializer.

Changes to DURATION
~~~~~~~~~~~~~~~~~~~

A new variant has been added to the existing DURATION type encoding for picoseconds. This
is encoded as follows, and is in addition to the previously supported variants.

==============================  =========  =========================================================
Qiskit class                    Type code  Payload
==============================  =========  =========================================================
:class:`~.circuit.Duration.ps`    ``p``    One ``double value``.

==============================  =========  =========================================================

.. _qpy_version_15:

Version 15
----------

Version 15 adds the concept of custom annotations to the payload format.  QPY itself does not
specify how annotations are serialized or deserialized, as they are custom user objects.  The format
does co-operate with sub-serializers, however.

Version 15 adds the ``ANNOTATION_HEADER`` field between the ``STANDALONE_VARS`` and
``CUSTOM_DEFINITIONS`` fields in the top level of a single circuit payload.  It modifies the
interpretation of one field of the ``INSTRUCTION`` struct in an ABI-compatible manner, and adds a
``INSTRUCTION_ANNOTATIONS`` trailer to ``INSTRUCTION`` which is present conditional on a set bit in
the ``INSTRUCTION`` payload.

New ANNOTATION_HEADER
~~~~~~~~~~~~~~~~~~~~~

The ``ANNOTATION_HEADER`` field is a variable-size payload in the header.  It begins with an
instance of ``ANNOTATION_HEADER_STATIC``, which is the C struct:

.. code-block:: c

    struct ANNOTATION_HEADER_STATIC {
        uint32_t num_namespaces;
    }

This is immediately followed by ``num_namespaces`` instances of the ``ANNOTATION_STATE`` payload.
The order of these is important and should be retained during the deserialization process, as
subsequent ``INSTRUCTION_ANNOTATION`` payloads will index into it.

The ``ANNOTATION_STATE`` payload begins with the fixed C struct:

.. code-block:: c

    struct ANNOTATION_STATE_HEADER {
        uint32_t namespace_size;
        uint64_t state_size;
    }

This header is immediately followed by ``namespace_size`` bytes of UTF-8 encoded text, which
comprise the namespace.  Those bytes are immediately followed by ``state_size`` bytes of arbitrary
data.  The format of this "state" payload is not defined by QPY.  Instead, it is the responsibility
of an external object associated with the stored namespace.  The format does not dictate how to
produce these objects; as annotations are entirely custom, the user must supply the serialization
and deserialization methods.


.. _qpy_instruction_v15:

Changes to INSTRUCTION
~~~~~~~~~~~~~~~~~~~~~~

The ``INSTRUCTION`` struct is modified in an ABI compatible manner to :ref:`its previous definition
in version 9 <qpy_instruction_v9>`.  The new struct is the C struct (recall that there is no padding
between any fields, nor at the end of the struct):

.. code-block:: c

    struct INSTRUCTION {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_parameters;
        uint32_t num_qargs;
        uint32_t num_cargs;
        uint8_t extras_key;
        uint16_t conditional_reg_name_size;
        int64_t conditional_value;
        uint32_t num_ctrl_qubits;
        uint32_t ctrl_state;
    }

where the field ``uint8_t extras_key`` replaces the previous ``uint8_t conditional_key``.  The
difference is purely in interpretation.  The low two bits of the byte are still interpreted as
defining the condition and its type.  The high bit of the byte is now a flag, indicated whether an
``INSTRUCTION_ANNOTATIONS_HEADER`` field is present (if the bit is set) in the trailing data of the
``INSTRUCTION`` struct.

A complete instruction payload appears in the data stream, including trailing objects and without
any padding bytes inbetween elements, as:

.. code-block:: text

    struct INSTRUCTION;
    uint8_t name[name_size];
    uint8_t label[label_size];
    uint8_t register[conditional_reg_name_size]; (1)
    struct INSTRUCTION_PARAM;                    (2)
    struct INSTRUCTION_ARG[num_qargs];
    struct INSTRUCTION_ARG[num_cargs];
    struct INSTRUCTION_PARAM[num_parameters];
    INSTRUCTION_ANNOTATIONS;                     (3)

The following notes apply:

1. if the two low bits of the ``extras_key`` have the value ``2``, indicating the condition is an
   ``EXPRESSION``, the ``conditional_reg_name_size`` is always zero.
2. this field is present if and only if the two low bits of the ``extras_key`` have the value ``2``,
   indicating the condition is an ``EXPRESSION``.
3. this field is present if and only if the high bit of the ``extras_key`` is set.  This field has
   a variable size; see :ref:`qpy_instruction_annotations_v15`.

.. _qpy_instruction_annotations_v15:

New INSTRUCTION_ANNOTATIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``INSTRUCTION_ANNOTATIONS`` payload begins with the C struct:

.. code-block:: c

    struct INSTRUCTION_ANNOTATIONS_HEADER {
        uint32_t num_annotations;
    }

This payload is immediately followed by ``num_annotations`` instances of the
``INSTRUCTION_ANNOTATION`` payload, which is of a variable size.

The ``INSRTUCTION_ANNOTATION`` payload is defined by the following C struct plus a trailing number
of bytes equal to the ``payload_size``, called ``ANNOTATION_PAYLOAD``.

.. code-block:: c

    struct INSTRUCTION_ANNOTATION {
        uint32_t namespace_index;
        uint32_t payload_size;
    }

The ``namespace_index`` is an integer index into the list of defined ``ANNOTATION_NAMESPACE``
objects in the ``ANNOTATION_HEADER``.  The serialization namespace for an annotation is the UTF-8
encoded string in the relevant payload.

The format of the ``ANNOTATION_PAYLOAD`` object is not specified by QPY.  It is defined by an
external serialization object associated with the namespace referred to by the ``namespace_index``
and its associated serializer state in the ``ANNOTATION_HEADER``.


Changes within PARAM_EXPR_ELEM_V13
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The struct itself is unchanged.  However, for a ``PARAM_EXPR_ELEM_V13`` representing a
:meth:`.ParameterExpression.subs` call (with ``op_code = 15``, and therefore ``lhs_type = 'p'`` and
``rhs_type = 'n'``), the trailing :ref:`qpy_mapping` now maps keys of the raw bytes of the
:class:`.Parameter` UUIDs to the substituted values.  Previously (in QPY versions 13 and 14), this
mapping stored the parameter names as the keys.


.. _qpy_version_14:

Version 14
----------

Version 14 adds a new core DURATION type, support for additional :class:`~.types.Type`
classes :class:`~.types.Float` and :class:`~.types.Duration`, and a new expression
node type :class:`~.expr.Stretch`.

DURATION
~~~~~~~~

A :class:`~.circuit.Duration` is encoded by a single-byte ASCII ``char`` that encodes the kind of
type, followed by a payload that varies depending on the type.  The defined codes are:

==============================  =========  =========================================================
Qiskit class                    Type code  Payload
==============================  =========  =========================================================
:class:`~.circuit.Duration.dt`   ``t``     One ``unsigned long long value``.

:class:`~.circuit.Duration.ns`   ``n``     One ``double value``.

:class:`~.circuit.Duration.us`   ``u``     One ``double value``.

:class:`~.circuit.Duration.ms`   ``m``     One ``double value``.

:class:`~.circuit.Duration.s`    ``s``     One ``double value``.

==============================  =========  =========================================================

Changes to EXPR_VAR_DECLARATION
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``EXPR_VAR_DECLARATION`` type is now used to represent both :class:`~.expr.Var` standalone
variables and :class:`~.expr.Stretch` identifiers. To support this change, the usage type code has
two new possible entries, in addition to the existing ones:

=========  =========================================================================================
Type code  Meaning
=========  =========================================================================================
``A``      A ``capture`` stretch to the circuit.

``O``      A locally declared stretch to the circuit.

=========  =========================================================================================

Changes to EXPRESSION
---------------------

The EXPRESSION type code has a new possible entry, ``s``, corresponding to :class:`.expr.Stretch`
nodes.

=======================  =========  ======================================================  ========
Qiskit class             Type code  Payload                                                 Children
=======================  =========  ======================================================  ========
:class:`~.expr.Stretch`  ``s``      One ``unsigned short var_index``                        0
=======================  =========  ======================================================  ========

Changes to EXPR_TYPE
~~~~~~~~~~~~~~~~~~~~

The following table shows the new type classes added in the version:

=========================  =========  ==============================================================
Qiskit class               Type code  Payload
=========================  =========  ==============================================================
:class:`~.types.Float`     ``f``      None.

:class:`~.types.Duration`  ``d``      None.

=========================  =========  ==============================================================

Changes to EXPR_VALUE
~~~~~~~~~~~~~~~~~~~~~

The classical expression's type system now supports new encoding types for value literals, in
addition to the existing encodings for int and bool. The new value type encodings are below:

===========================  =========  ============================================================
Python type                  Type code  Payload
===========================  =========  ============================================================
``float``                    ``f``      One ``double value``.

:class:`~.circuit.Duration`  ``t``      One ``DURATION``.

===========================  =========  ============================================================

.. _qpy_version_13:

Version 13
----------

Version 13 added a native Qiskit serialization representation for :class:`.ParameterExpression`.
Previous QPY versions relied on either ``sympy`` or ``symengine`` to serialize the underlying symbolic
expression. Starting in Version 13, QPY now represents the sequence of API calls used to create the
:class:`.ParameterExpression`.

The main change in the serialization format is in the :ref:`qpy_param_expr_v3` payload.  The
``expr_size`` bytes following the head now contain an array of ``PARAM_EXPR_ELEM_V13`` structs. The
intent is for this array to be read one struct at a time, where each struct describes one of the
calls to make to reconstruct the :class:`.ParameterExpression`.

PARAM_EXPR_ELEM_V13
~~~~~~~~~~~~~~~~~~~

The struct format is defined as:

.. code-block:: c

    struct {
        unsigned char op_code;
        char lhs_type;
        char lhs[16];
        char rhs_type;
        char rhs[16];
    } PARAM_EXPR_ELEM_V13;

The ``op_code`` field is used to define the operation added to the :class:`.ParameterExpression`.
The value can be:

.. list-table:: PARAM_EXPR_ELEM_V13 op code values
   :header-rows: 1

   * - ``op_code``
     - :class:`.ParameterExpression` method
   * - 0
     - :meth:`~.ParameterExpression.__add__`
   * - 1
     - :meth:`~.ParameterExpression.__sub__`
   * - 2
     - :meth:`~.ParameterExpression.__mul__`
   * - 3
     - :meth:`~.ParameterExpression.__truediv__`
   * - 4
     - :meth:`~.ParameterExpression.__pow__`
   * - 5
     - :meth:`~.ParameterExpression.sin`
   * - 6
     - :meth:`~.ParameterExpression.cos`
   * - 7
     - :meth:`~.ParameterExpression.tan`
   * - 8
     - :meth:`~.ParameterExpression.arcsin`
   * - 9
     - :meth:`~.ParameterExpression.arccos`
   * - 10
     - :meth:`~.ParameterExpression.exp`
   * - 11
     - :meth:`~.ParameterExpression.log`
   * - 12
     - :meth:`~.ParameterExpression.sign`
   * - 13
     - :meth:`~.ParameterExpression.gradient`
   * - 14
     - :meth:`~.ParameterExpression.conjugate`
   * - 15
     - :meth:`~.ParameterExpression.subs`
   * - 16
     - :meth:`~.ParameterExpression.abs`
   * - 17
     - :meth:`~.ParameterExpression.arctan`
   * - 255
     - NULL

The ``NULL`` value of 255 is only used to fill the op code field for
entries that are not actual operations but indicate recursive definitions.
Then the ``lhs_type`` and ``rhs_type`` fields are used to describe
the operand types and can be one of the following UTF-8 encoded
characters:

.. list-table:: PARAM_EXPR_ELEM_V13 operand type values
   :header-rows: 1

   * - Value
     - Type
   * - ``n``
     - ``None``
   * - ``p``
     - :class:`.Parameter`
   * - ``f``
     - ``float``
   * - ``c``
     - ``complex``
   * - ``i``
     - ``int``
   * - ``s``
     - Recursive :class:`.ParameterExpression` definition start
   * - ``e``
     - Recursive :class:`.ParameterExpression` definition stop
   * - ``u``
     - substitution

If the type value is ``f``, ``c``, or ``i``, the corresponding ``lhs`` or ``rhs``
field widths are 128 bits each. In the case of floats, the literal value is encoded as a double
with 0 padding, while complex numbers are encoded as real part followed by imaginary part,
taking up 64 bits each. For ``i``, the value is encoded as a 64 bit signed integer with 0 padding
for the full 128 bit width. ``n`` is used to represent a ``None`` and typically isn't directly used
as it indicates an argument that's not used. For ``p`` the data is the UUID for the
:class:`.Parameter` which can be looked up in the symbol map described in the
``map_elements`` outer :ref:`qpy_param_expr_v3` payload. If the type value is
``s`` this marks the start of a a new recursive section for a nested
:class:`.ParameterExpression`. For example, in the following snippet there is an inner ``expr``
contained in ``final_expr``, constituting a nested expression::

    from qiskit.circuit import Parameter

    x = Parameter("x")
    y = Parameter("y")
    z = Parameter("z")

    expr = (x + y) / 2
    final_expr = z**2 + expr

When ``s`` is encountered, this indicates that until an ``e` struct is reached, the next structs
are used for a recursive definition. For both
``s`` and ``e`` types, the data values are not used, and always set to 0. The type value
of ``u`` is used to represent a substitution call. This is only used for ``lhs_type``
and is always paired with an ``rhs_type`` of ``n``. The data value is the size in bytes of
a :ref:`qpy_mapping` encoded mapping of :class:`.Parameter` names to their value for the
:meth:`~.ParameterExpression.subs` call. The mapping data is immediately following the
struct, and the next struct starts immediately after the mapping data.

.. _qpy_version_12:

Version 12
----------

Version 12 adds support for:

* circuits containing memory-owning :class:`.expr.Var` variables.

Changes to HEADER
~~~~~~~~~~~~~~~~~

The HEADER struct for an individual circuit has added three ``uint32_t`` counts of the input,
captured and locally declared variables in the circuit.  The new form looks like:

.. code-block:: c

    struct {
        uint16_t name_size;
        char global_phase_type;
        uint16_t global_phase_size;
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
        uint32_t num_vars;
    } HEADER_V12;

The ``HEADER_V12`` struct is followed immediately by the same name, global-phase, metadata
and register information as the V2 version of the header.  Immediately following the registers is
``num_vars`` instances of ``EXPR_VAR_STANDALONE`` that define the variables in this circuit.  After
that, the data continues with custom definitions and instructions as in prior versions of QPY.


EXPR_VAR_DECLARATION
~~~~~~~~~~~~~~~~~~~~

An ``EXPR_VAR_DECLARATION`` defines an :class:`.expr.Var` instance that is standalone; that is, it
represents a self-owned memory location rather than wrapping a :class:`.Clbit` or
:class:`.ClassicalRegister`.  The payload is a C struct:

.. code-block:: c

    struct {
        char uuid_bytes[16];
        char usage;
        uint16_t name_size;
    }

which is immediately followed by an ``EXPR_TYPE`` payload and then ``name_size`` bytes of UTF-8
encoding string data containing the name of the variable.

The ``char`` usage type code takes the following values:

=========  =========================================================================================
Type code  Meaning
=========  =========================================================================================
``I``      An ``input`` variable to the circuit.

``C``      A ``capture`` variable to the circuit.

``L``      A locally declared variable to the circuit.
=========  =========================================================================================


Changes to EXPR_VAR
~~~~~~~~~~~~~~~~~~~

The ``EXPR_VAR`` variable has gained a new type code and payload, in addition to the pre-existing ones:

===========================  =========  ============================================================
Python class                 Type code  Payload
===========================  =========  ============================================================
:class:`.UUID`               ``U``      One ``uint32_t`` index of the variable into the series of
                                        ``EXPR_VAR_STANDALONE`` variables that were written
                                        immediately after the circuit header.
===========================  =========  ============================================================

Notably, this new type-code indexes into pre-defined variables from the circuit header, rather than
redefining the variable again in each location it is used.


Changes to EXPRESSION
---------------------

The EXPRESSION type code has a new possible entry, ``i``, corresponding to :class:`.expr.Index`
nodes.

======================  =========  =======================================================  ========
Qiskit class            Type code  Payload                                                  Children
======================  =========  =======================================================  ========
:class:`~.expr.Index`   ``i``      No additional payload. The children are the target       2
                                   and the index, in that order.
======================  =========  =======================================================  ========


.. _qpy_version_11:

Version 11
----------

Version 11 is identical to Version 10 except for the following.
First, the names in the CUSTOM_INSTRUCTION blocks
have a suffix of the form ``"_{uuid_hex}"`` where ``uuid_hex`` is a uuid
hexadecimal string such as returned by :attr:`.UUID.hex`. For example:
``"b3ecab5b4d6a4eb6bc2b2dbf18d83e1e"``.
Second, it adds support for :class:`.AnnotatedOperation`
objects. The base operation of an annotated operation is stored using the INSTRUCTION block,
and an additional ``type`` value ``'a'``is added to indicate that the custom instruction is an
annotated operation. The list of modifiers are stored as instruction parameters using INSTRUCTION_PARAM,
with an additional value ``'m'`` is added to indicate that the parameter is of type
:class:`~qiskit.circuit.annotated_operation.Modifier`. Each modifier is stored using the
MODIFIER struct.

.. _modifier_qpy:

MODIFIER
~~~~~~~~

This represents :class:`~qiskit.circuit.annotated_operation.Modifier`

.. code-block:: c

    struct {
        char type;
        uint32_t num_ctrl_qubits;
        uint32_t ctrl_state;
        double power;
    }

This is sufficient to store different types of modifiers required for serializing objects
of type :class:`.AnnotatedOperation`.
The field ``type`` is either ``'i'``, ``'c'`` or ``'p'``, representing whether the modifier
is respectively an inverse modifier, a control modifier or a power modifier. In the second
case, the fields ``num_ctrl_qubits`` and ``ctrl_state`` specify the control logic of the base
operation, and in the third case the field ``power`` represents the power of the base operation.

.. _qpy_version_10:

Version 10
----------

Version 10 adds support for:

* symengine-native serialization for objects of type :class:`~.ParameterExpression` as well as
  symbolic expressions in Pulse schedule blocks.
* new fields in the :class:`~.TranspileLayout` class added in the Qiskit 0.45.0 release.

The symbolic_encoding field is added to the file header, and a new encoding type char
is introduced, mapped to each symbolic library as follows: ``p`` refers to sympy
encoding and ``e`` refers to symengine encoding.

Changes to FILE_HEADER
~~~~~~~~~~~~~~~~~~~~~~

The contents of FILE_HEADER after V10 are defined as a C struct as:

.. code-block:: c

    struct {
        uint8_t qpy_version;
        uint8_t qiskit_major_version;
        uint8_t qiskit_minor_version;
        uint8_t qiskit_patch_version;
        uint64_t num_circuits;
        char symbolic_encoding;
    } FILE_HEADER_V10;

Changes to LAYOUT
~~~~~~~~~~~~~~~~~

The ``LAYOUT`` struct is updated to have an additional ``input_qubit_count`` field.
With version 10 the ``LAYOUT`` struct is now:

.. code-block:: c

    struct {
        char exists;
        int32_t initial_layout_size;
        int32_t input_mapping_size;
        int32_t final_layout_size;
        uint32_t extra_registers;
        int32_t input_qubit_count;
    }

The rest of the layout data after the ``LAYOUT`` struct is represented as in previous versions. If
``input qubit_count`` is < 0 that indicates that both ``_input_qubit_count``
and ``_output_qubit_list`` in the :class:`~.TranspileLayout` object are ``None``.

.. _qpy_version_9:

Version 9
---------

Version 9 adds support for classical :class:`~.expr.Expr` nodes and their associated
:class:`~.types.Type`\\ s.


EXPRESSION
~~~~~~~~~~

An :class:`~.expr.Expr` node is represented by a stream of variable-width data.  A node itself is
represented by (in order in the byte stream):

#. a one-byte type code discriminator;
#. an EXPR_TYPE object;
#. a type-code-specific additional payload;
#. a type-code-specific number of child EXPRESSION payloads (the number of these is implied by the
   type code and not explicitly stored).

Each of these are described in the following table:

======================  =========  =======================================================  ========
Qiskit class            Type code  Payload                                                  Children
======================  =========  =======================================================  ========
:class:`~.expr.Var`     ``x``      One ``EXPR_VAR``.                                        0

:class:`~.expr.Value`   ``v``      One ``EXPR_VALUE``.                                      0

:class:`~.expr.Cast`    ``c``      One ``_Bool``  that corresponds to the value of          1
                                   ``implicit``.

:class:`~.expr.Unary`   ``u``      One ``uint8_t`` with the same numeric value as the       1
                                   :class:`.Unary.Op`.

:class:`~.expr.Binary`  ``b``      One ``uint8_t`` with the same numeric value as the       2
                                   :class:`.Binary.Op`.
======================  =========  =======================================================  ========


EXPR_TYPE
~~~~~~~~~

A :class:`~.types.Type` is encoded by a single-byte ASCII ``char`` that encodes the kind of type,
followed by a payload that varies depending on the type.  The defined codes are:

======================  =========  =================================================================
Qiskit class            Type code  Payload
======================  =========  =================================================================
:class:`~.types.Bool`   ``b``      None.

:class:`~.types.Uint`   ``u``      One ``uint32_t width``.
======================  =========  =================================================================


EXPR_VAR
~~~~~~~~

This represents a runtime variable of a :class:`~.expr.Var` node.  These are a type code, followed
by a type-code-specific payload:

===========================  =========  ============================================================
Python class                 Type code  Payload
===========================  =========  ============================================================
:class:`.Clbit`              ``C``      One ``uint32_t index`` that is the index of the
                                        :class:`.Clbit` in the containing circuit.

:class:`.ClassicalRegister`  ``R``      One ``uint16_t reg_name_size``, followed by that many bytes
                                        of UTF-8 string data of the register name.
===========================  =========  ============================================================


EXPR_VALUE
~~~~~~~~~~

This represents a literal object in the classical type system, such as an integer.  Currently there
are very few such literals.  These are encoded as a type code, followed by a type-code-specific
payload.

===========  =========  ============================================================================
Python type  Type code  Payload
===========  =========  ============================================================================
``bool``     ``b``      One ``_Bool value``.

``int``      ``i``      One ``uint8_t num_bytes``, followed by the integer encoded into that many
                        many bytes (network order) in a two's complement representation.
===========  =========  ============================================================================


.. _qpy_instruction_v9:

Changes to INSTRUCTION
~~~~~~~~~~~~~~~~~~~~~~

To support the use of :class:`~.expr.Expr` nodes in the fields :attr:`.IfElseOp.condition`,
:attr:`.WhileLoopOp.condition` and :attr:`.SwitchCaseOp.target`, the INSTRUCTION struct is changed
in an ABI compatible-manner to :ref:`its previous definition <qpy_instruction_v5>`.  The new struct
is the C struct:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_parameters;
        uint32_t num_qargs;
        uint32_t num_cargs;
        uint8_t conditional_key;
        uint16_t conditional_reg_name_size;
        int64_t conditional_value;
        uint32_t num_ctrl_qubits;
        uint32_t ctrl_state;
    }

where the only change is that a ``uint8_t conditional_key`` entry has replaced ``_Bool
has_conditional``.  This new ``conditional_key`` takes the following numeric values, with these
effects:

=====  =============================================================================================
Value  Effects
=====  =============================================================================================
0      The instruction has its ``.condition`` field set to ``None``.  The
       ``conditional_reg_name_size`` and ``conditional_value`` fields should be ignored.

1      The instruction has its ``.condition`` field set to a 2-tuple of either a :class:`.Clbit`
       or a :class:`.ClassicalRegister`, and a integer of value ``conditional_value``.  The
       INSTRUCTION payload, including its trailing data is parsed exactly as it would be in QPY
       versions less than 8.

2      The instruction has its ``.condition`` field set to a :class:`~.expr.Expr` node.  The
       ``conditional_reg_name_size`` and ``conditional_value`` fields should be ignored.  The data
       following the struct is followed (as in QPY versions less than 9) by ``name_size`` bytes of
       UTF-8 string data for the class name and ``label_size`` bytes of UTF-8 string data for the
       label (if any). Then, there is one INSTRUCTION_PARAM, which will contain an EXPRESSION. After
       that, parsing continues with the INSTRUCTION_ARG structs, as in previous versions of QPY.
=====  =============================================================================================


Changes to INSTRUCTION_PARAM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A new type code ``x`` is added that defines an EXPRESSION parameter.


.. _qpy_version_8:

Version 8
---------

Version 8 adds support for handling a :class:`~.TranspileLayout` stored in the
:attr:`.QuantumCircuit.layout` attribute. In version 8 immediately following the
calibrations block at the end of the circuit payload there is now the
``LAYOUT`` struct. This struct outlines the size of the three attributes of a
:class:`~.TranspileLayout` class.

LAYOUT
~~~~~~

.. code-block:: c

    struct {
        char exists;
        int32_t initial_layout_size;
        int32_t input_mapping_size;
        int32_t final_layout_size;
        uint32_t extra_registers;
    }

If any of the signed values are ``-1`` this indicates the corresponding
attribute is ``None``.

Immediately following the ``LAYOUT`` struct there is a :ref:`qpy_registers` struct
for ``extra_registers`` (specifically the format introduced in :ref:`qpy_version_4`)
standalone register definitions that aren't present in the circuit. Then there
are ``initial_layout_size`` ``INITIAL_LAYOUT_BIT`` structs to define the
:attr:`.TranspileLayout.initial_layout` attribute.

INITIAL_LAYOUT_BIT
~~~~~~~~~~~~~~~~~~

.. code-block:: c

    struct {
        int32_t index;
        int32_t register_size;
    }

Where a value of ``-1`` indicates ``None`` (as in no register is associated
with the bit). Following each ``INITIAL_LAYOUT_BIT`` struct is ``register_size``
bytes for a ``utf8`` encoded string for the register name.

Following the initial layout there is ``input_mapping_size`` array of
``uint32_t`` integers representing the positions of the physical bit from the
initial layout. This enables constructing a list of virtual bits where the
array index is its input mapping position.

Finally, there is an array of ``final_layout_size`` ``uint32_t`` integers. Each
element is an index in the circuit's ``qubits`` attribute which enables building
a mapping from qubit starting position to the output position at the end of the
circuit.

.. _qpy_version_7:

Version 7
---------

Version 7 adds support for :class:`.~Reference` instruction and serialization of
a ``ScheduleBlock`` program while keeping its reference to subroutines::

    from qiskit import pulse
    from qiskit import qpy

    with pulse.build() as schedule:
        pulse.reference("cr45p", "q0", "q1")
        pulse.reference("x", "q0")
        pulse.reference("cr45p", "q0", "q1")

    with open('template_ecr.qpy', 'wb') as fd:
        qpy.dump(schedule, fd)

The conventional :ref:`qpy_schedule_block` data model is preserved, but in
version 7 it is immediately followed by an extra :ref:`qpy_mapping` utf8 bytes block
representing the data of the referenced subroutines.

New type key character is added to the :ref:`qpy_schedule_instructions` group
for the :class:`.~Reference` instruction.

- ``y``: :class:`~qiskit.pulse.instructions.Reference` instruction

New type key character is added to the :ref:`qpy_schedule_operands` group
for the operands of :class:`.~Reference` instruction,
which is a tuple of strings, e.g. ("cr45p", "q0", "q1").

- ``o``: string (operand string)

Note that this is the same encoding with the built-in Python string, however,
the standard value encoding in QPY uses ``s`` type character for string data,
which conflicts with the :class:`~qiskit.pulse.library.SymbolicPulse` in the scope of
pulse instruction operands. A special type character ``o`` is reserved for
the string data that appears in the pulse instruction operands.

In addition, version 7 adds two new type keys to the INSTRUCTION_PARM struct.  ``"d"`` is followed
by no data and represents the literal value :data:`.CASE_DEFAULT` for switch-statement support.
``"R"`` represents a :class:`.ClassicalRegister` or :class:`.Clbit`, and is followed by the same
format as the description of register or classical bit as used in the first element of :ref:`the
condition of an INSTRUCTION field <qpy_instructions>`.

.. _qpy_version_6:

Version 6
---------

Version 6 adds support for :class:`.~ScalableSymbolicPulse`. These objects are saved and read
like `SymbolicPulse` objects, and the class name is added to the data to correctly handle
the class selection.

`SymbolicPulse` block now starts with SYMBOLIC_PULSE_V2 header:

.. code-block:: c

    struct {
        uint16_t class_name_size;
        uint16_t type_size;
        uint16_t envelope_size;
        uint16_t constraints_size;
        uint16_t valid_amp_conditions_size;
        _bool amp_limited;
    }

The only change compared to :ref:`qpy_version_5` is the addition of `class_name_size`. The header
is then immediately followed by ``class_name_size`` utf8 bytes with the name of the class. Currently,
either `SymbolicPulse` or `ScalableSymbolicPulse` are supported. The rest of the data is then
identical to :ref:`qpy_version_5`.

.. _qpy_version_5:

Version 5
---------

Version 5 changes from :ref:`qpy_version_4` by adding support for ``ScheduleBlock``
and changing two payloads the INSTRUCTION metadata payload and the CUSTOM_INSTRUCTION block.
These now have new fields to better account for :class:`~.ControlledGate` objects in a circuit.
In addition, new payload MAP_ITEM is defined to implement the :ref:`qpy_mapping` block.

.. note::

    Support for representing pulse schedules and custom calibrations was removed in Qiskit v2.0.
    When loading QPY payloads, these data fields are now ignored or raise an error when using
    Qiskit for deserialization.

In QPY version 5 and above,

.. code-block:: c

    struct {
        char type;
    }

immediately follows the file header block to represent the program type stored in the file.

- When ``type==c``, :class:`~.QuantumCircuit` payload follows
- When ``type==s``, ``ScheduleBlock`` payload follows

.. note::

    Different programs cannot be packed together in the same file.
    You must create different files for different program types.
    Multiple objects with the same type can be saved in a single file.

.. _qpy_schedule_block:

SCHEDULE_BLOCK
~~~~~~~~~~~~~~

``ScheduleBlock`` is first supported in QPY Version 5. This allows
users to save pulse programs in the QPY binary format as follows:

.. code-block:: python

    from qiskit import pulse, qpy

    with pulse.build() as schedule:
        pulse.play(pulse.Gaussian(160, 0.1, 40), pulse.DriveChannel(0))

    with open('schedule.qpy', 'wb') as fd:
        qpy.dump(schedule, fd)

    with open('schedule.qpy', 'rb') as fd:
        new_schedule = qpy.load(fd)[0]

Note that circuit and schedule block are serialized and deserialized through
the same QPY interface. Input data type is implicitly analyzed and
no extra option is required to save the schedule block.

.. _qpy_schedule_block_header:

SCHEDULE_BLOCK_HEADER
~~~~~~~~~~~~~~~~~~~~~

``ScheduleBlock`` block starts with the following header:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint64_t metadata_size;
        uint16_t num_element;
    }

which is immediately followed by ``name_size`` utf8 bytes of schedule name and
``metadata_size`` utf8 bytes of the JSON serialized metadata dictionary
attached to the schedule.

.. _qpy_schedule_alignments:

SCHEDULE_BLOCK_ALIGNMENTS
~~~~~~~~~~~~~~~~~~~~~~~~~

Then, alignment context of the schedule block starts with ``char``
representing the supported context type followed by the :ref:`qpy_sequence` block representing
the parameters associated with the alignment context :attr:`AlignmentKind._context_params`.
The context type char is mapped to each alignment subclass as follows:

- ``l``: :class:`~.AlignLeft`
- ``r``: :class:`~.AlignRight`
- ``s``: :class:`~.AlignSequential`
- ``e``: :class:`~.AlignEquispaced`

Note that :class:`~.AlignFunc` context is not supported because of the callback function
stored in the context parameters.

.. _qpy_schedule_instructions:

SCHEDULE_BLOCK_INSTRUCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This alignment block is further followed by ``num_element`` length of block elements which may
consist of nested schedule blocks and schedule instructions.
Each schedule instruction starts with ``char`` representing the instruction type
followed by the :ref:`qpy_sequence` block representing the instruction
:attr:`~qiskit.pulse.instructions.Instruction.operands`.
Note that the data structure of pulse :class:`~qiskit.pulse.instructions.Instruction`
is unified so that instance can be uniquely determined by the class and a tuple of operands.
The mapping of type char to the instruction subclass is defined as follows:

- ``a``: :class:`~qiskit.pulse.instructions.Acquire` instruction
- ``p``: :class:`~qiskit.pulse.instructions.Play` instruction
- ``d``: :class:`~qiskit.pulse.instructions.Delay` instruction
- ``f``: :class:`~qiskit.pulse.instructions.SetFrequency` instruction
- ``g``: :class:`~qiskit.pulse.instructions.ShiftFrequency` instruction
- ``q``: :class:`~qiskit.pulse.instructions.SetPhase` instruction
- ``r``: :class:`~qiskit.pulse.instructions.ShiftPhase` instruction
- ``b``: :class:`~qiskit.pulse.instructions.RelativeBarrier` instruction
- ``t``: :class:`~qiskit.pulse.instructions.TimeBlockade` instruction
- ``y``: :class:`~qiskit.pulse.instructions.Reference` instruction (new in version 0.7)

.. _qpy_schedule_operands:

SCHEDULE_BLOCK_OPERANDS
~~~~~~~~~~~~~~~~~~~~~~~

The operands of these instances can be serialized through the standard QPY value serialization
mechanism, however there are special object types that only appear in the schedule operands.
Since the operands are serialized as :ref:`qpy_sequence`, each element must be packed with the
INSTRUCTION_PARAM pack struct, where each payload starts with a header block consists of
the char ``type`` and uint64_t ``size``.
Special objects start with the following type key:

- ``c``: :class:`~qiskit.pulse.channels.Channel`
- ``w``: :class:`~qiskit.pulse.library.Waveform`
- ``s``: :class:`~qiskit.pulse.library.SymbolicPulse`
- ``o``: string (operand string, new in version 0.7)

.. _qpy_schedule_channel:

CHANNEL
~~~~~~~

Channel block starts with channel subtype ``char`` that maps an object data to
:class:`~qiskit.pulse.channels.Channel` subclass. Mapping is defined as follows:

- ``d``: :class:`~qiskit.pulse.channels.DriveChannel`
- ``c``: :class:`~qiskit.pulse.channels.ControlChannel`
- ``m``: :class:`~qiskit.pulse.channels.MeasureChannel`
- ``a``: :class:`~qiskit.pulse.channels.AcquireChannel`
- ``e``: :class:`~qiskit.pulse.channels.MemorySlot`
- ``r``: :class:`~qiskit.pulse.channels.RegisterSlot`

The key is immediately followed by the channel index serialized as the INSTRUCTION_PARAM.

.. _qpy_schedule_waveform:

Waveform
~~~~~~~~

Waveform block starts with WAVEFORM header:

.. code-block:: c

    struct {
        double epsilon;
        uint32_t data_size;
        _bool amp_limited;
    }

which is followed by ``data_size`` bytes of complex ``ndarray`` binary generated by numpy.save_.
This represents the complex IQ data points played on a quantum device.
:attr:`~qiskit.pulse.library.Waveform.name` is saved after the samples in the
INSTRUCTION_PARAM pack struct, which can be string or ``None``.

.. _numpy.save: https://numpy.org/doc/stable/reference/generated/numpy.save.html

.. _qpy_schedule_symbolic_pulse:

SymbolicPulse
~~~~~~~~~~~~~

SymbolicPulse block starts with SYMBOLIC_PULSE header:

.. code-block:: c

    struct {
        uint16_t type_size;
        uint16_t envelope_size;
        uint16_t constraints_size;
        uint16_t valid_amp_conditions_size;
        _bool amp_limited;
    }

which is followed by ``type_size`` utf8 bytes of :attr:`.SymbolicPulse.pulse_type` string
that represents a class of waveform, such as "Gaussian" or "GaussianSquare".
Then, ``envelope_size``, ``constraints_size``, ``valid_amp_conditions_size`` utf8 bytes of
serialized symbolic expressions are generated for :attr:`.SymbolicPulse.envelope`,
:attr:`.SymbolicPulse.constraints`, and :attr:`.SymbolicPulse.valid_amp_conditions`, respectively.
Since string representation of these expressions are usually lengthy,
the expression binary is generated by the python zlib_ module with data compression.

To uniquely specify a pulse instance, we also need to store the associated parameters,
which consist of ``duration`` and the rest of parameters as a dictionary.
Dictionary parameters are first dumped in the :ref:`qpy_mapping` form, and then ``duration``
is dumped with the INSTRUCTION_PARAM pack struct.
Lastly, :attr:`~qiskit.pulse.library.SymbolicPulse.name` is saved also with the
INSTRUCTION_PARAM pack struct, which can be string or ``None``.

.. _zlib: https://docs.python.org/3/library/zlib.html

.. _qpy_mapping:

MAPPING
~~~~~~~

The MAPPING is a representation for arbitrary mapping object. This is a fixed length
:ref:`qpy_sequence` of key-value pair represented by the MAP_ITEM payload.

A MAP_ITEM starts with a header defined as:

.. code-block:: c

    struct {
        uint16_t key_size;
        char type;
        uint16_t size;
    }

which is immediately followed by the ``key_size`` utf8 bytes representing
the dictionary key in string and ``size`` utf8 bytes of arbitrary object data of
QPY serializable ``type``.

.. _qpy_circuit_calibrations:

CIRCUIT_CALIBRATIONS
~~~~~~~~~~~~~~~~~~~~

The CIRCUIT_CALIBRATIONS block is a dictionary to define pulse calibrations of the custom
instruction set. This block starts with the following CALIBRATION header:

.. code-block:: c

    struct {
        uint16_t num_cals;
    }

which is followed by the ``num_cals`` length of calibration entries, each starts with
the CALIBRATION_DEF header:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t num_qubits;
        uint16_t num_params;
        char type;
    }

The calibration definition header is then followed by ``name_size`` utf8 bytes of
the gate name, ``num_qubits`` length of integers representing a sequence of qubits,
and ``num_params`` length of INSTRUCTION_PARAM payload for parameters
associated to the custom instruction.
The ``type`` indicates the class of pulse program which is either, in principle,
``ScheduleBlock`` or :class:`~.Schedule`. As of QPY Version 5,
only ``ScheduleBlock`` payload is supported.
Finally, :ref:`qpy_schedule_block` payload is packed for each CALIBRATION_DEF entry.

.. _qpy_instruction_v5:

INSTRUCTION
~~~~~~~~~~~

The INSTRUCTION block was modified to add two new fields ``num_ctrl_qubits`` and ``ctrl_state``
which are used to model the :attr:`.ControlledGate.num_ctrl_qubits` and
:attr:`.ControlledGate.ctrl_state` attributes. The new payload packed struct
format is:

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_parameters;
        uint32_t num_qargs;
        uint32_t num_cargs;
        _Bool has_conditional;
        uint16_t conditional_reg_name_size;
        int64_t conditional_value;
        uint32_t num_ctrl_qubits;
        uint32_t ctrl_state;
    }

The rest of the instruction payload is the same. You can refer to
:ref:`qpy_instructions` for the details of the full payload.

CUSTOM_INSTRUCTION
~~~~~~~~~~~~~~~~~~

The CUSTOM_INSTRUCTION block in QPY version 5 adds a new field
``base_gate_size`` which is used to define the size of the
:class:`qiskit.circuit.Instruction` object stored in the
:attr:`.ControlledGate.base_gate` attribute for a custom
:class:`~.ControlledGate` object. With this change the CUSTOM_INSTRUCTION
metadata block becomes:

.. code-block:: c

    struct {
        uint16_t name_size;
        char type;
        uint32_t num_qubits;
        uint32_t num_clbits;
        _Bool custom_definition;
        uint64_t size;
        uint32_t num_ctrl_qubits;
        uint32_t ctrl_state;
        uint64_t base_gate_size
    }

Immediately following the CUSTOM_INSTRUCTION struct is the utf8 encoded name
of size ``name_size``.

If ``custom_definition`` is ``True`` that means that the immediately following
``size`` bytes contains a QPY circuit data which can be used for the custom
definition of that gate. If ``custom_definition`` is ``False`` then the
instruction can be considered opaque (ie no definition). The ``type`` field
determines what type of object will get created with the custom definition.
If it's ``'g'`` it will be a :class:`~qiskit.circuit.Gate` object, ``'i'``
it will be a :class:`~qiskit.circuit.Instruction` object.

Following this the next ``base_gate_size`` bytes contain the ``INSTRUCTION``
payload for the :attr:`.ControlledGate.base_gate`.

Additionally an addition value for ``type`` is added ``'c'`` which is used to
indicate the custom instruction is a custom :class:`~.ControlledGate`.

.. _qpy_version_4:

Version 4
---------

Version 4 is identical to :ref:`qpy_version_3` except that it adds 2 new type strings
to the INSTRUCTION_PARAM struct, ``z`` to represent ``None`` (which is encoded as
no data), ``q`` to represent a :class:`.QuantumCircuit` (which is encoded as
a QPY circuit), ``r`` to represent a ``range`` of integers (which is encoded as
a :ref:`qpy_range_pack`), and ``t`` to represent a ``sequence`` (which is encoded as
defined by :ref:`qpy_sequence`). Additionally, version 4 changes the type of register
index mapping array from ``uint32_t`` to ``int64_t``. If the values of any of the
array elements are negative they represent a register bit that is not present in the
circuit.

The :ref:`qpy_registers` header format has also been updated to

.. code-block:: c

    struct {
        char type;
        _Bool standalone;
        uint32_t size;
        uint16_t name_size;
        _bool in_circuit;
    }

which just adds the ``in_circuit`` field which represents whether the register is
part of the circuit or not.

.. _qpy_range_pack:

RANGE
~~~~~

A RANGE is a representation of a ``range`` object. It is defined as:

.. code-block:: c

    struct {
        int64_t start;
        int64_t stop;
        int64_t step;
    }

.. _qpy_sequence:

SEQUENCE
~~~~~~~~

A SEQUENCE is a representation of an arbitrary sequence object. As sequence are just fixed length
containers of arbitrary python objects their QPY can't fully represent any sequence,
but as long as the contents in a sequence are other QPY serializable types for
the INSTRUCTION_PARAM payload the ``sequence`` object can be serialized.

A sequence instruction parameter starts with a header defined as:

.. code-block:: c

    struct {
        uint64_t size;
    }

followed by ``size`` elements that are INSTRUCTION_PARAM payloads, where each of
these define an element in the sequence. The sequence object will be typecasted
into proper type, e.g. ``tuple``, afterwards.

.. _qpy_version_3:

Version 3
---------

Version 3 of the QPY format is identical to :ref:`qpy_version_2` except that it defines
a struct format to represent a :class:`~qiskit.circuit.library.PauliEvolutionGate`
natively in QPY. To accomplish this the :ref:`qpy_custom_definition` struct now supports
a new type value ``'p'`` to represent a :class:`~qiskit.circuit.library.PauliEvolutionGate`.
Enties in the custom instructions tables have unique name generated that start with the
string ``"###PauliEvolutionGate_"`` followed by a uuid string. This gate name is reserved
in QPY and if you have a custom :class:`~qiskit.circuit.Instruction` object with a definition
set and that name prefix it will error. If it's of type ``'p'`` the data payload is defined
as follows:

.. _pauli_evo_qpy:

PAULI_EVOLUTION
~~~~~~~~~~~~~~~

This represents the high level :class:`~qiskit.circuit.library.PauliEvolutionGate`

.. code-block:: c

    struct {
        uint64_t operator_count;
        _Bool standalone_op;
        char time_type;
        uint64_t time_size;
        uint64_t synthesis_size;
    }

This is immediately followed by ``operator_count`` elements defined by the :ref:`qpy_pauli_sum_op`
payload.  Following that we have ``time_size`` bytes representing the ``time`` attribute. If
``standalone_op`` is ``True`` then there must only be a single operator. The
encoding of these bytes is determined by the value of ``time_type``. Possible values of
``time_type`` are ``'f'``, ``'p'``, and ``'e'``. If ``time_type`` is ``'f'`` it's a double,
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is represented by a
:ref:`qpy_param_struct`, ``e`` defines a :class:`~qiskit.circuit.ParameterExpression` object
(that's not a :class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`qpy_param_expr`.
Following that is ``synthesis_size`` bytes which is a utf8 encoded json payload representing
the :class:`.EvolutionSynthesis` class used by the gate.

.. _qpy_pauli_sum_op:

SPARSE_PAULI_OP_LIST_ELEM
~~~~~~~~~~~~~~~~~~~~~~~~~

This represents an instance of :class:`.SparsePauliOp`.


.. code-block:: c

    struct {
        uint32_t pauli_op_size;
    }

which is immediately followed by ``pauli_op_size`` bytes which are .npy format [#f2]_
data which represents the :class:`~qiskit.quantum_info.SparsePauliOp`.

Version 3 of the QPY format also defines a struct format to represent a
:class:`~qiskit.circuit.ParameterVectorElement` as a distinct subclass from
a :class:`~qiskit.circuit.Parameter`. This adds a new parameter type char ``'v'``
to represent a :class:`~qiskit.circuit.ParameterVectorElement` which is now
supported as a type string value for an INSTRUCTION_PARAM. The payload for these
parameters are defined below as :ref:`qpy_param_vector`.

.. _qpy_param_vector:

PARAMETER_VECTOR_ELEMENT
~~~~~~~~~~~~~~~~~~~~~~~~

A PARAMETER_VECTOR_ELEMENT represents a :class:`~qiskit.circuit.ParameterVectorElement`
object the data for a INSTRUCTION_PARAM. The contents of the PARAMETER_VECTOR_ELEMENT are
defined as:

.. code-block:: c

    struct {
        uint16_t vector_name_size;
        uint64_t vector_size;
        char uuid[16];
        uint64_t index;
    }

which is immediately followed by ``vector_name_size`` utf8 bytes representing
the parameter's vector name.

.. _qpy_param_expr_v3:


PARAMETER_EXPR
~~~~~~~~~~~~~~

Additionally, since QPY format version v3 distinguishes between a
:class:`~qiskit.circuit.Parameter` and :class:`~qiskit.circuit.ParameterVectorElement`
the payload for a :class:`~qiskit.circuit.ParameterExpression` needs to be updated
to distinguish between the types. The following is the modified payload format
which is mostly identical to the format in Version 1 and :ref:`qpy_version_2` but just
modifies the ``map_elements`` struct to include a symbol type field.

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined as:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
    }

Immediately following the header is ``expr_size`` bytes of utf8 data containing
the expression string, which is the sympy srepr of the expression for the
parameter expression. Following that is a symbol map which contains
``map_elements`` elements with the format

.. code-block:: c

    struct {
        char symbol_type;
        char type;
        uint64_t size;
    }

The ``symbol_type`` key determines the payload type of the symbol representation
for the element. If it's ``p`` it represents a :class:`~qiskit.circuit.Parameter`
and if it's ``v`` it represents a :class:`~qiskit.circuit.ParameterVectorElement`.
The map element struct is immediately followed by the symbol map key payload, if
``symbol_type`` is ``p`` then it is followed immediately by a :ref:`qpy_param_struct`
object (both the struct and utf8 name bytes) and if ``symbol_type`` is ``v``
then the struct is immediately followed by :ref:`qpy_param_vector` (both the struct
and utf8 name bytes). That is followed by ``size`` bytes for the
data of the symbol. The data format is dependent on the value of ``type``. If
``type`` is ``p`` then it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. Similarly if the
``type`` is ``v`` then it represents a :class:`~qiskit.circuit.ParameterVectorElement`
and size will be 0 as the value will just be the same as the key. If
``type`` is ``f`` then it represents a double precision float. If ``type`` is
``c`` it represents a double precision complex, which is represented by the
:ref:`qpy_complex`. Finally, if type is ``i`` it represents an integer which is an
``int64_t``.

.. _qpy_version_2:

Version 2
---------

Version 2 of the QPY format is identical to version 1 except for the HEADER
section is slightly different. You can refer to the :ref:`qpy_version_1` section
for the details on the rest of the payload format.

HEADER
~~~~~~

The contents of HEADER are defined as a C struct are:

.. code-block:: c

    struct {
        uint16_t name_size;
        char global_phase_type;
        uint16_t global_phase_size;
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
    }

This is immediately followed by ``name_size`` bytes of utf8 data for the name
of the circuit. Following this is immediately ``global_phase_size`` bytes
representing the global phase. The content of that data is dictated by the
value of ``global_phase_type``. If it's ``'f'`` the data is a float and is the
size of a ``double``. If it's ``'p'`` defines a :class:`~qiskit.circuit.Parameter`
object  which is represented by a PARAM struct (see below), ``e`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that's not a
:class:`~qiskit.circuit.Parameter`) which is represented by a PARAM_EXPR struct
(see below).

.. _qpy_version_1:

Version 1
---------

HEADER
~~~~~~

The contents of HEADER as defined as a C struct are:

.. code-block:: c

    struct {
        uint16_t name_size;
        double global_phase;
        uint32_t num_qubits;
        uint32_t num_clbits;
        uint64_t metadata_size;
        uint32_t num_registers;
        uint64_t num_instructions;
    }

This is immediately followed by ``name_size`` bytes of utf8 data for the name
of the circuit.

METADATA
~~~~~~~~

The METADATA field is a UTF8 encoded JSON string. After reading the HEADER
(which is a fixed size at the start of the QPY file) and the ``name`` string
you then read the ``metadata_size`` number of bytes and parse the JSON to get
the metadata for the circuit.

.. _qpy_registers:

REGISTERS
~~~~~~~~~

The contents of REGISTERS is a number of REGISTER object. If num_registers is
> 0 then after reading METADATA you read that number of REGISTER structs defined
as:

.. code-block:: c

    struct {
        char type;
        _Bool standalone;
        uint32_t size;
        uint16_t name_size;
    }

``type`` can be ``'q'`` or ``'c'``.

Immediately following the REGISTER struct is the utf8 encoded register name of
size ``name_size``. After the ``name`` utf8 bytes there is then an array of
int64_t values of size ``size`` that contains a map of the register's index to
the circuit's qubit index. For example, array element 0's value is the index
of the ``register[0]``'s position in the containing circuit's qubits list.

.. note::

    Prior to QPY :ref:`qpy_version_4` the type of array elements was uint32_t. This was changed
    to enable negative values which represent bits in the array not present in the
    circuit


The standalone boolean determines whether the register is constructed as a
standalone register that was added to the circuit or was created from existing
bits. A register is considered standalone if it has bits constructed solely
as part of it, for example::

    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)

the register ``qr`` would be a standalone register. While something like::

    bits = [Qubit(), Qubit()]
    qr2 = QuantumRegister(bits=bits)
    qc = QuantumCircuit(qr2)

``qr2`` would have ``standalone`` set to ``False``.


.. _qpy_custom_definition:

CUSTOM_DEFINITIONS
~~~~~~~~~~~~~~~~~~

This section specifies custom definitions for any of the instructions in the circuit.

CUSTOM_DEFINITION_HEADER contents are defined as:

.. code-block:: c

    struct {
        uint64_t size;
    }

If size is greater than 0 that means the circuit contains custom instruction(s).
Each custom instruction is defined with a CUSTOM_INSTRUCTION block defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char type;
        uint32_t num_qubits;
        uint32_t num_clbits;
        _Bool custom_definition;
        uint64_t size;
    }

Immediately following the CUSTOM_INSTRUCTION struct is the utf8 encoded name
of size ``name_size``.

If ``custom_definition`` is ``True`` that means that the immediately following
``size`` bytes contains a QPY circuit data which can be used for the custom
definition of that gate. If ``custom_definition`` is ``False`` then the
instruction can be considered opaque (ie no definition). The ``type`` field
determines what type of object will get created with the custom definition.
If it's ``'g'`` it will be a :class:`~qiskit.circuit.Gate` object, ``'i'``
it will be a :class:`~qiskit.circuit.Instruction` object.

.. _qpy_instructions:

INSTRUCTIONS
~~~~~~~~~~~~

The contents of INSTRUCTIONS is a list of INSTRUCTION metadata objects

.. code-block:: c

    struct {
        uint16_t name_size;
        uint16_t label_size;
        uint16_t num_parameters;
        uint32_t num_qargs;
        uint32_t num_cargs;
        _Bool has_conditional;
        uint16_t conditional_reg_name_size;
        int64_t conditional_value;
    }

This metadata object is immediately followed by ``name_size`` bytes of utf8 bytes
for the ``name``. ``name`` here is the Qiskit class name for the Instruction
class if it's defined in Qiskit. Otherwise it falls back to the custom
instruction name. Following the ``name`` bytes there are ``label_size`` bytes of
utf8 data for the label if one was set on the instruction. Following the label
bytes if ``has_conditional`` is ``True`` then there are
``conditional_reg_name_size`` bytes of utf8 data for the name of the conditional
register name. In case of single classical bit conditions the register name
utf8 data will be prefixed with a null character "\\x00" and then a utf8 string
integer representing the classical bit index in the circuit that the condition
is on.

This is immediately followed by the INSTRUCTION_ARG structs for the list of
arguments of that instruction. These are in the order of all quantum arguments
(there are num_qargs of these) followed by all classical arguments (num_cargs
of these).

The contents of each INSTRUCTION_ARG is:

.. code-block:: c

    struct {
        char type;
        uint32_t index;
    }

``type`` can be ``'q'`` or ``'c'``.

After all arguments for an instruction the parameters are specified with
``num_parameters`` INSTRUCTION_PARAM structs.

The contents of each INSTRUCTION_PARAM is:

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

After each INSTRUCTION_PARAM the next ``size`` bytes are the parameter's data.
The ``type`` field can be ``'i'``, ``'f'``, ``'p'``, ``'e'``, ``'s'``, ``'c'``
or ``'n'`` which dictate the format. For ``'i'`` it's an integer, ``'f'`` it's
a double, ``'s'`` if it's a string (encoded as utf8), ``'c'`` is a complex and
the data is represented by the struct format in the :ref:`qpy_param_expr` section.
``'p'`` defines a :class:`~qiskit.circuit.Parameter` object  which is
represented by a :ref:`qpy_param_struct` struct, ``e`` defines a
:class:`~qiskit.circuit.ParameterExpression` object (that's not a
:class:`~qiskit.circuit.Parameter`) which is represented by a :ref:`qpy_param_expr`
struct (on QPY format :ref:`qpy_version_3` the format is tweak slightly see:
:ref:`qpy_param_expr_v3`), ``'n'`` represents an object from numpy (either an
``ndarray`` or a numpy type) which means the data is .npy format [#f2]_ data,
and in QPY :ref:`qpy_version_3` ``'v'`` represents a
:class:`~qiskit.circuit.ParameterVectorElement` which is represented by a
:ref:`qpy_param_vector` struct.

.. _qpy_param_struct:

PARAMETER
~~~~~~~~~

A PARAMETER represents a :class:`~qiskit.circuit.Parameter` object the data for
a INSTRUCTION_PARAM. The contents of the PARAMETER are defined as:

.. code-block:: c

    struct {
        uint16_t name_size;
        char uuid[16];
    }

which is immediately followed by ``name_size`` utf8 bytes representing the
parameter name.

.. _qpy_param_expr:

PARAMETER_EXPR
~~~~~~~~~~~~~~

A PARAMETER_EXPR represents a :class:`~qiskit.circuit.ParameterExpression`
object that the data for an INSTRUCTION_PARAM. The contents of a PARAMETER_EXPR
are defined as:

The PARAMETER_EXPR data starts with a header:

.. code-block:: c

    struct {
        uint64_t map_elements;
        uint64_t expr_size;
    }

Immediately following the header is ``expr_size`` bytes of utf8 data containing
the expression string, which is the sympy srepr of the expression for the
parameter expression. Following that is a symbol map which contains
``map_elements`` elements with the format

.. code-block:: c

    struct {
        char type;
        uint64_t size;
    }

Which is followed immediately by ``PARAMETER`` object (both the struct and utf8
name bytes) for the symbol map key. That is followed by ``size`` bytes for the
data of the symbol. The data format is dependent on the value of ``type``. If
``type`` is ``p`` then it represents a :class:`~qiskit.circuit.Parameter` and
size will be 0, the value will just be the same as the key. If
``type`` is ``f`` then it represents a double precision float. If ``type`` is
``c`` it represents a double precision complex, which is represented by :ref:`qpy_complex`.
Finally, if type is ``i`` it represents an integer which is an ``int64_t``.

.. _qpy_complex:

COMPLEX
~~~~~~~

When representing a double precision complex value in QPY the following
struct is used:


.. code-block:: c

    struct {
        double real;
        double imag;
    }

this matches the internal C representation of Python's complex type. [#f3]_

References
==========

.. [#f1] https://tools.ietf.org/html/rfc1700
.. [#f2] https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
.. [#f3] https://docs.python.org/3/c-api/complex.html#c.Py_complex
"""

from .exceptions import QpyError, UnsupportedFeatureForVersion, QPYLoadingDeprecatedFeatureWarning
from .interface import dump, load, get_qpy_version

# For backward compatibility. Provide, Runtime, Experiment call these private functions.
# ruff: disable[F401]
from .binary_io import (
    _write_instruction,
    _read_instruction,
    _write_parameter_expression,
    _read_parameter_expression,
    _read_parameter_expression_v3,
)

# ruff: enable[F401]

from .common import QPY_VERSION, QPY_COMPATIBILITY_VERSION

__all__ = [
    "QPY_COMPATIBILITY_VERSION",
    "QPY_VERSION",
    "QPYLoadingDeprecatedFeatureWarning",
    "QpyError",
    "UnsupportedFeatureForVersion",
    "dump",
    "get_qpy_version",
    "load",
]
