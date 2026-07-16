.. _capi-classical-expressions:

=====================
Classical Expressions
=====================

The Qiskit C API for classical expressions provides facilities for inspecting classical expressions used in Qiskit. This API enables users to
traverse and examine expression trees through a set of enums, structs, and accessor functions. You can inspect the structure of expressions,
identify node types, query type information and access detailed properties of each node in the expression tree. Currently, classical expressions can
only be accessed through the control-flow API, for example when querying conditions that are comprised of classical expressions.

Note that the functions in this API return borrowed pointers that remain valid as long as the parent expression tree exists,
so these pointers do not need to be freed.

Data Types
==========

.. doxygenenum:: QkExprNodeKind

.. doxygenenum:: QkExprType

.. doxygenstruct:: QkExprTypeInfo
    :members:

.. doxygenenum:: QkUnaryOpType

.. doxygenstruct:: QkUnaryExprInfo
    :members:

.. doxygenenum:: QkBinaryOpType

.. doxygenstruct:: QkBinaryExprInfo
    :members:

.. doxygenstruct:: QkCastExprInfo
    :members:

.. doxygenstruct:: QkIndexExprInfo
    :members:

.. doxygenenum:: QkDurationType

.. doxygenunion:: QkDurationValue

.. doxygenstruct:: QkDurationInfo
    :members:


Functions
=========

.. doxygengroup:: QkClassicalExpressions
    :members:
    :content-only:



