.. _capi-transpiler-passes:

=================
Transpiler Passes
=================

Transpilation is the process of rewriting a given input circuit to match the topology of a specific
quantum device, and/or to optimize the circuit for execution on a quantum system.

Most circuits must undergo a series of transformations that make them compatible with a given target
device, and optimize them to reduce the effects of noise on the resulting outcomes. Rewriting quantum
circuits to match hardware constraints and optimizing for performance can be far from trivial. The flow
of logic in the rewriting tool chain need not be linear, and can often have iterative sub-loops,
conditional branches, and other complex behaviors.

In Qiskit, the transpiler is built up by executing as a series of passes that each perform a singular task
to analyze or transform a quantum circuit. The Python :mod:`~qiskit.transpiler` documentation contains a
more detailed explanation of the transpilation process.

The Qiskit C API provides transpiler pass functions in two forms: ones that operate on a :c:struct:`QkDag` 
and another set that operate on a :c:struct:`QkCircuit`. The DAG‑based functions, which follow the naming 
convention ``qk_transpiler_pass_*``, accept a :c:struct:`QkDag` along with a :c:struct:`QkTarget` and any 
pass‑specific configuration parameters. These functions are the recommended choice when chaining multiple 
passes, e.g. when creating a custom transpilation pipeline, because they operate directly on the DAG object 
and allow it to be passed efficiently from one pass to the next within a transpilation session. By contrast, 
the circuit‑based functions, following the ``qk_transpiler_pass_standalone_*`` naming convention, operate on a 
:c:struct:`QkCircuit` and are intended for executing individual passes in isolation. While they can also be 
used to build custom workflows, each call incurs additional overhead because the input circuit must be converted 
to a DAG internally and, if a transformed circuit is returned, the resulting DAG must then be converted back to 
a circuit.

DAG-based Functions
===================

.. doxygengroup:: QkTranspilerPasses
    :members:
    :content-only:

Circuit-based Functions
=======================

.. doxygengroup:: QkTranspilerPassesStandalone
    :members:
    :content-only:
