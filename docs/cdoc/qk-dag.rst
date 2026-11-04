.. _capi-dag:

=====
QkDag
=====

.. code-block:: c

   typedef struct QkDag QkDag

The ``QkDag`` struct  exposes a low level interface to the Qiskit transpiler's
directed acyclic graph (DAG) representation of a quantum circuit for use in
transpiler passes. It exposes only what is defined in the inner data model of
Qiskit. Therefore it is missing some functionality that is available in the
higher level Python :class:`.DAGCircuit` class.

The C API currently only supports building DAGs that contain
operations defined in Qiskit's internal Rust data model. Generally this
includes only gates in the standard gate library, standard non-unitary
operations (currently :class:`.Barrier`, :class:`.Measure`, :class:`.Reset`, and
:class:`.Delay`) and :class:`.UnitaryGate`. This functionality will be
expanded over time as the Rust data model is expanded to natively support
more functionality.

Data Types
==========

.. doxygenenum:: QkDagNodeType

.. doxygenenum:: QkOperationKind

.. doxygenstruct:: QkDagNeighbors
   :members:

Functions
=========

.. doxygengroup:: QkDag
   :members:
   :content-only:
