# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
======================================================
Circuit annotations (:mod:`qiskit.circuit.annotation`)
======================================================

.. currentmodule:: qiskit.circuit.annotation

This module contains the infrastructure for working with custom circuit annotations.

The main user-facing class is the base class :class:`qiskit.circuit.Annotation`, which is also
re-exported from this module.

.. _circuit-annotation-subclassing:

Custom annotation subclasses
============================

The :class:`.Annotation` class is intended to be subclassed.  Subclasses must set their
:attr:`~.Annotation.namespace` field.  This can be specific to an instance, or static for an entire
subclass.  The namespace is used as part of the dispatch mechanism, as described in
:ref:`circuit-annotation-namespacing`.

Circuit equality checks also compare annotations on objects in an order-dependent manner.  You will
likely want to implement the :meth:`~object.__eq__` magic method on any subclasses.

If you intend your annotation to be able to be serialized via :ref:`QPY <qiskit-qpy>` or :ref:`
OpenQASM 3 <qiskit-qasm3>`, you must provide separate implementations of the serialization and
deserialization methods as discussed in :ref:`circuit-annotation-serialization`.

.. _circuit-annotation-namespacing:

Namespacing
-----------

The "namespace" of an annotation is used as a look-up key when any consumer is deciding which
handler to invoke.  This includes in QPY and OpenQASM 3 serialization contexts, but in general,
transpiler passes will also look at annotations' namespaces to determine if they are relevant, and
so on.

This can be a standard Python identifier (e.g. ``my_namespace``), or a dot-separated list of
identifiers (e.g. ``my_namespace.subnamespace``).  The namespace is used by all consumers of
annotations to determine what handler should be invoked.

A stand-alone function allows iterating through namespaces and parent namespaces in priority order
from most specific to least specific.

.. autofunction:: iter_namespaces


.. _circuit-annotation-serialization:

Serialization and deserialization
---------------------------------

Annotations represent completely custom data, that may persist after compilation.  This may include
data that should be serialized for later consumption, such as additional data that is interpreted by
a backend-compiler.  Qiskit's native binary QPY format (see :mod:`qiskit.qpy`) supports the concept
of arbitrary annotations in its payloads from version 15 onwards.  In OpenQASM 3 (see
:mod:`qiskit.qasm3`), annotations are a core language feature, and Qiskit's import/export support
for OpenQASM 3 includes serialization of annotations.

However, since annotations are generally custom subclasses and unknown to Qiskit, we cannot have
built-in support for serialization.  On the deserialization front, Qiskit will not, in general, have
an existing :class:`~.Annotation` object to call deserialization methods from.  It is also expected
that annotations may relate to some unknown-to-Qiskit shared state within a given circuit context.

For all of these reasons, serialization and deserialization of annotations is handled by custom
objects, which must be passed at the interface points of the relevant serialization functions.  For
example in QPY, the ``annotation_factories`` argument in :func:`.qpy.dump` and :func:`.qpy.load` are
used to pass serializers.

.. autoclass:: QPYSerializer
.. autoclass:: QPYFromOpenQASM3Serializer
.. autoclass:: OpenQASM3Serializer


Examples
========

A block-collection transpiler pass
----------------------------------

A principal goal of the annotation framework is to allow custom analyses and commands to be stored
on circuits in an instruction-local manner, either by the user on entry to the compiler, or for one
compiler pass to store information for later consumption.

For example, we can write a simple transpiler pass that collects runs of single-qubit operations,
and puts each run into a :class:`.BoxOp`, then calculates the total unitary action and attaches it as
a custom annotation, so the same analysis does not need to be repeated later, even if the internals
of each block are optimized.

.. code-block:: python

    from qiskit.circuit import annotation, QuantumCircuit, BoxOp
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import TransformationPass

    class PerformsUnitary(annotation.Annotation):
        namespace = "unitary"
        def __init__(self, matrix):
            self.matrix = matrix

    class Collect1qRuns(TransformationPass):
        def run(self, dag):
            for run in dag.collect_1q_runs():
                block = QuantumCircuit(1)
                for node in run:
                    block.append(node.op, [0], [])
                box = BoxOp(block, annotations=[PerformsUnitary(Operator(block).data)])
                dag.replace_block_with_op(run, box, {run[0].qargs[0]: 0})
            return dag

In order to serialize the annotation to OpenQASM 3, we must define custom logic, since the analysis
itself is entirely custom.  The serialization is separate to the annotation; there may be
circumstances in which serialization should be done differently.

.. code-block:: python

    import ast
    import numpy as np

    class Serializer(annotation.OpenQASM3Serializer):
        def dump(self, annotation):
            if annotation.namespace != "unitary":
                return NotImplemented
            line = lambda row: "[" + ", ".join(repr(x) for x in row) + "]"
            return "[" + ", ".join(line(row) for row in annotation.matrix.tolist()) + "]"

        def load(self, namespace, payload):
            if namespace != "unitary":
                return NotImplemented
            return PerformsUnitary(np.array(ast.literal_eval(payload), dtype=complex))

Finally, this can be put together, showing the output OpenQASM 3.

.. code-block:: python

    from qiskit import qasm3

    qc = QuantumCircuit(3)
    qc.s(0)
    qc.t(0)
    qc.y(1)
    qc.x(1)
    qc.h(2)
    qc.s(2)
    collected = Collect1qRuns()(qc)

    handlers = {"unitary": Serializer()}
    dumped = qasm3.dumps(collected, annotation_handlers=handlers)
    print(dumped)

.. code-block:: openqasm3

    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    @unitary [[(1+0j), 0j], [0j, (-0.7071067811865475+0.7071067811865475j)]]
    box {
      s q[0];
      t q[0];
    }
    @unitary [[1j, 0j], [0j, -1j]]
    box {
      y q[1];
      x q[1];
    }
    @unitary [[(0.7071067811865475+0j), (0.7071067811865475+0j)], [0.7071067811865475j, \
-0.7071067811865475j]]
    box {
      h q[2];
      s q[2];
    }

"""

from __future__ import annotations

import abc
from typing import Literal
from collections.abc import Iterator

from qiskit._accelerate.circuit import Annotation


__all__ = [
    "Annotation",  # Also exported in `qiskit.circuit`, but for convenience is here too.
    "OpenQASM3Serializer",
    "QPYFromOpenQASM3Serializer",
    "QPYSerializer",
    "iter_namespaces",
]


def iter_namespaces(namespace: str) -> Iterator[str]:
    """An iterator over all namespaces that can be used to lookup the given namespace.

    This includes the namespace and all parents, including the root empty-string namespace.

    Examples:

    .. code-block:: python

        from qiskit.circuit.annotation import iter_namespaces
        assert list(iter_namespaces("hello.world")) == ["hello.world", "hello", ""]
    """
    while namespace:
        yield namespace
        split = namespace.rsplit(".", 1)
        if len(split) == 1:
            break
        namespace = split[0]
    yield ""


class QPYSerializer(abc.ABC):
    """The interface for serializers and deserializers of :class:`.Annotation` objects to QPY.

    For more information on QPY, see :mod:`qiskit.qpy`.

    This interface-definition class is designed to be subclassed.  The individual methods describe
    their contracts, and how they will be called.

    During QPY serialization and deserialization, the main QPY logic will call a factory function to
    create instances of subclasses of this class.  The return value from a given factory function
    will be used in *either* a serialization or deserialization context, but not both.

    The structure of calls during serialization of a single circuit is:

    1. many calls to :meth:`dump_annotation`, which will all share the same ``namespace`` argument,
       which will always be a (non-strict) prefix of all the :class:`.Annotation` objects given.
    2. one call to :meth:`dump_state`.

    The general structure of calls during deserialization of a single circuit out of a QPY payload
    is:

    1. one call to :meth:`load_state`, passing a ``namespace`` (with the same non-strict prefixing
       behavior as the "serializing" form).
    2. many calls to :meth:`load_annotation`, corresponding to annotations serialized under that
       namespace-prefix lookup.

    When subclassing this, recall that QPY is intended to have strict backwards-compatibility
    guarantees, and it is strongly recommended that annotation-serialisation subclasses maintain
    this.  In particular, it is suggested that any non-trivial serializer includes "version"
    information for the serializer in its total "state" (see :meth:`dump_state`), and the
    deserialization should make every effort to support backwards compatibility with previous
    versions of the same serializer.
    """

    @abc.abstractmethod
    def dump_annotation(
        self, namespace: str, annotation: Annotation
    ) -> bytes | Literal[NotImplemented]:
        """Serialize an annotation to a bytestream.

        This method may mutate the serializer's internal state (the object that will be serialized
        by :meth:`dump_state`).

        The ``namespace`` argument is the resolved key used to lookup this serializer.  It may not
        be identical to the :attr:`.Annotation.namespace` field of the ``annotation`` argument; it
        might be an ancestor, up to and including the empty string (the root namespace).  All calls
        to an instance of this class, as retrieved by a factory function in :func:`.qpy.dump` will
        be made using the same ``namespace``.

        The method can return :data:`NotImplemented` if the serializer cannot serialize a particular
        annotation.  In this case, the QPY logic will attempt to use a serializer registered for
        the parent namespaces.

        This method is the mirror of :meth:`load_annotation`.

        Args:
            namespace: the namespace that this serializer was accessed under.  This may be an
                ancestor of the annotation.
            annotation: the object to serialize to a bytestream.

        Returns:
            Either the serialized form of the annotation (optionally after mutating the
            serialization state of this class), or :data:`NotImplemented` if the annotation cannot
            be handled.
        """

    @abc.abstractmethod
    def load_annotation(self, payload: bytes) -> Annotation:
        """Load an annotation from a view of memory.

        A subclass can assume that :meth:`load_state` will have been called exactly once before
        this method is called, and all subsequent calls to this method will be for payloads
        corresponding to annotations serialized under that parent namespace.

        If a user configures QPY correctly, instances of this class will only be asked to
        deserialize payloads that the corresponding :meth:`dump_annotation` can successfully handle
        (i.e. return a payload, not :data:`NotImplemented`).  Subclasses may raise an arbitrary
        exception if this is not the case and this will abort the QPY load operation.  Such a
        situation would require that the user supplied a different serializer configuration on the
        two sides of the QPY load and dump.

        This method is the mirror of :meth:`dump_annotation`.

        Args:
            payload: the bytes to deserialized into an annotation.

        Returns:
            The deserialized annotation.
        """

    def dump_state(self) -> bytes:
        """Serialize a state object for the given serializer.

        When in a QPY dumping context, this method will be called exactly once, after all calls to
        :meth:`dump_annotation`.

        The default state is the empty bytestring; if your serializer is stateless, you do not need
        to override this method.

        This method is the mirror of :meth:`load_state`.
        """
        return b""

    def load_state(self, namespace: str, payload: bytes):
        """Initialize the state of the deserializer for a given ``namespace`` key.

        When in a QPY loading context, this method will be called exactly once, before all calls to
        :meth:`load_annotation`.  The ``namespace`` will be the same namespace that was passed to
        all calls to :meth:`dump_annotation` in the dumping context; that is, a (non-strict) prefix
        of the namespaces of all the :class:`.Annotation` objects its counterpart was asked to
        serialize.  For example, if the QPY dump was configured with::

            from qiskit import qpy
            from qiskit.circuit import annotation, Annotation

            class MyA(Annotation):
                namespace = "my.a"
            class MyB(Annotation):
                namespace = "my.b"

            class MyQPYSerializer(annotation.QPYSerializer):
                ...

            qpy.dump(..., annotation_factories={"my": MyQPYSerializer})

        then during the corresponding call to :func:`.qpy.load`, this method in ``MyQPYSerializer``
        will be called with ``"my"``, even though the annotations serialized had namespaces ``my.a``
        and ``my.b``. It is up to individual dumpers to do any sub-namespace handling they choose.

        The default implementation is a no-op; if you have not overridden :meth:`dump_state`, you do
        not need to override this method.

        This method is the mirror of :meth:`dump_state`.

        Args:
            namespace: the namespace key that the corresponding dump was resolved under.
            payload: the state payload that was dumped by the corresponding call to
                :meth:`dump_state`.
        """


class OpenQASM3Serializer(abc.ABC):
    """The interface for serializers and deserializers of :class:`.Annotation` objects to
    OpenQASM 3.

    For more information on OpenQASM 3 support in Qiskit, see :mod:`qiskit.qasm3`.

    This interface-definition class is designed to be subclassed.  OpenQASM 3 annotations are
    stateless within a program, therefore a subclass must not track state.
    """

    @abc.abstractmethod
    def dump(self, annotation: Annotation) -> str | Literal[NotImplemented]:
        """Serialize the payload of an annotation to a single line of UTF-8 text.

        The output of this method should not include the annotation's
        :attr:`~.Annotation.namespace` attribute; this is handled automatically by the OpenQASM 3
        exporter.

        The serialized form must not contain newline characters; it must be valid as the "arbitrary"
        component of the annotation as defined by OpenQASM 3.  If there is no data required, the
        method should return the empty string.  If this serializer cannot handle the particular
        annotation, it should return :data:`NotImplemented`.

        Args:
            annotation: the annotation object to serialize.

        Returns:
            the serialized annotation (without the namespace component), or the sentinel
            :data:`NotImplemented` if it cannot be handled by this object.
        """

    @abc.abstractmethod
    def load(self, namespace: str, payload: str) -> Annotation | Literal[NotImplemented]:
        """Load an annotation, if possible, from an OpenQASM 3 program.

        The two arguments will be the two components of an annotation, as defined by the OpenQASM 3
        specification.  The method should return :data:`NotImplemented` if it cannot handle the
        annotation.

        Args:
            namespace: the OpenQASM 3 "namespace" of the annotation.
            payload: the rest of the payload for the annotation.  This is arbitrary and free-form,
                and in general should have been serialized by a call to :meth:`dump`.

        Returns:
            the created :class:`.Annotation` object, whose :attr:`.Annotation.namespace` attribute
            should be identical to the incoming ``namespace`` argument.  If this class cannot handle
            the annotation, it can also return :data:`NotImplemented`.
        """

    def as_qpy(self) -> QPYFromOpenQASM3Serializer:
        """Derive a serializer/deserializer for QPY from this OpenQASM 3 variant.

        OpenQASM 3 serialization and deserialization is intended to be stateless and return single
        lines of UTF-8 encoded text.  This is a subset of the allowable serializations for QPY."""
        return QPYFromOpenQASM3Serializer(self)


class QPYFromOpenQASM3Serializer(QPYSerializer):
    """An adaptor that converts a :class:`OpenQASM3Serializer` into a :class:`QPYSerializer`.

    This works because OpenQASM 3 annotation serializers are required to be stateless and return
    UTF-8-encoded single lines of text, which is a subset of what QPY permits.

    Typically you create one of these using the :meth:`~OpenQASM3Serializer.as_qpy` method
    of an OpenQASM 3 annotation serializer.

    Examples:

    Instances of this class can be called like a zero-argument function and return themselves.  This
    lets you use them directly as a factory function to the QPY entry points, such as:

    .. code-block:: python

        import io
        from qiskit.circuit import OpenQASM3Serializer, Annotation
        from qiskit import qpy

        class MyAnnotation(Annotation):
            namespace = "my_namespace"

        class MySerializer(OpenQASM3Serializer):
            def dump(self, annotation):
                if not isinstance(annotation, MyAnnotation):
                    return NotImplemented
                return ""

            def load(self, namespace, payload):
                assert namespace == "my_namespace"
                assert payload == ""
                return MyAnnotation()

        qc = QuantumCircuit(2)
        with qc.box(annotations=[MyAnnotation()]):
            qc.cx(0, 1)

        with io.BytesIO() as fptr:
            qpy.dump(fptr, qc, annotation_serializers = {"my_namespace": MySerializer().as_qpy()})

    This is safe, without returning separate instances, because the base OpenQASM 3 serializers are
    necessarily stateless.
    """

    def __init__(self, inner: OpenQASM3Serializer):
        """
        Args:
            inner: the OpenQASM 3 serializer that this is derived from.
        """
        self.inner = inner

    def dump_annotation(self, namespace, annotation):
        qasm3 = self.inner.dump(annotation)
        if qasm3 is NotImplemented:
            return NotImplemented
        return (
            # For OpenQASM 3 serialisation, we need the exact namespace the annotation claims.
            annotation.namespace.encode("utf-8")
            + b"\x00"
            + qasm3.encode("utf-8")
        )

    def load_annotation(self, payload):
        namespace, payload = payload.split(b"\x00", maxsplit=1)
        out = self.inner.load(namespace.decode("utf-8"), payload.decode("utf-8"))
        if out is NotImplemented:
            raise ValueError(
                "asked to deserialize an object the provided OpenQASM deserializer cannot handle"
            )
        return out

    def __call__(self) -> QPYFromOpenQASM3Serializer:
        # Our internal object is stateless because it's an OpenQASM 3 exporter (which is a stateless
        # format).  Defining this method allows an instance of ourself to be used as a factory
        # function, simplifying the interface for creating a QPY serializer from an OpenQASM 3 one,
        # and attaching this to the class means that pickling would "just work".
        return self
