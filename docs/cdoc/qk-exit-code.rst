.. _capi-exit-code:

==========
QkExitCode
==========

.. code-block:: c

    enum QkExitCode

Function exit codes.

Values
------

* ``QkExitCode_Success``
    Success.

    Value: 0

* ``QkExitCode_CInputError``
    Error related to data input.

    Value: 100

* ``QkExitCode_NullPointerError``
    Unexpected null pointer.

    Value: 101

* ``QkExitCode_AlignmentError``
    Pointer is not aligned to expected data.

    Value: 102

* ``QkExitCode_IndexError``
    Index out of bounds.

    Value: 103

* ``QkExitCode_ArithmeticError``
    Error related to arithmetic operations or similar.

    Value: 200

* ``QkExitCode_MismatchedQubits``
    Mismatching number of qubits.

    Value: 201

Functions
---------
