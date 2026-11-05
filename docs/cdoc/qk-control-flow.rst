.. _capi-control-flow:

==============
Control Flow
==============

The Qiskit C API for control flow provides facilities for inspecting control flow instructions used in quantum circuits. 
You can query the type of control flow instruction, access nested circuit blocks, inspect conditions and loop parameters, 
and retrieve qubit and classical bit mappings between nested blocks and the top-level circuit.

.. code-block:: c

   typedef struct QkControlFlowInstruction QkControlFlowInstruction

The ``QkControlFlowInstruction`` opaque struct is the primary handle used by the C API to represent and reason about control-flow 
instructions within a circuit. It is obtained by a call to :c:func:`qk_circuit_get_control_flow_instruction` and must be freed with a call 
to :c:func:`qk_control_flow_instruction_free`. Most C API functions that operate on control-flow constructs expect a pointer to 
this opaque struct, making it the central vehicle for inspecting control-flow instructions from C. Note that most of the functions 
in this API which use ``QkControlFlowInstruction`` as input, return borrowed pointers that remain valid as long as the control 
flow instruction exists, so these pointers do not need to be freed separately. 

Data Types
==========

.. doxygenenum:: QkControlFlowKind

.. doxygenenum:: QkConditionType

.. doxygenstruct:: QkConditionBitInfo
    :members:

.. doxygenenum:: QkBoxDurationKind

.. doxygenstruct:: QkSwitchCaseLabels
    :members:

.. doxygenenum:: QkLoopParamKind

.. doxygenenum:: QkSymbolType

.. doxygenstruct:: QkSymbolInfo
    :members:

.. doxygenenum:: QkLoopCollectionType

.. doxygenstruct:: QkLoopElements
    :members:

Functions
=========

.. doxygengroup:: QkControlFlow
    :members:
    :content-only:
