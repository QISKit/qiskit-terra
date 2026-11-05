.. _capi-transpile-layout:

=================
QkTranspileLayout
=================

.. code-block:: c

   typedef struct QkTranspileLayout QkTranspileLayout

The ``QkTranspileLayout`` type is used to model the permutations introduced by
the transpiler. In general Qiskit's transpiler is unitary-preserving up to the
initial layout and output permutations. The initial layout is the mapping
from virtual circuit qubits to physical qubits on the target and the output
permutation is caused by swap gate insertion or permutation elision prior to
the initial layout being set in the transpiler pipeline. This type tracks these
details and provide an interface to reason about these permutations.

For example if you had a circuit constructed like:

.. code-block:: c

    #include <qiskit.h>

    QkCircuit *qc = qk_circuit_new(3, 0)
    uint32_t h_qargs[1] = {0};
    qk_circuit_gate(qc, QkGate_H, h_qargs, NULL);
    uint32_t cx_0_qargs[2] = {0, 1};
    qk_circuit_gate(qc, QkGate_CX, cx_0_qargs, NULL);
    uint32_t cx_1_qargs[2] = {0, 2};
    qk_circuit_gate(qc, QkGate_CX, cx_1_qargs, NULL);

and during the layout stage the transpiler maps the virtual qubits in that
circuit to the physical circuits as:

.. code-block:: text

    0 -> 2, 1 -> 1, 2 -> 0

so the circuit would look like:

.. code-block:: c

    #include <qiskit.h>

    QkCircuit *qc = qk_circuit_new(3, 0)
    uint32_t h_qargs[1] = {2};
    qk_circuit_gate(qc, QkGate_H, h_qargs, NULL);
    uint32_t cx_0_qargs[2] = {2, 1};
    qk_circuit_gate(qc, QkGate_CX, cx_0_qargs, NULL);
    uint32_t cx_1_qargs[2] = {2, 0};
    qk_circuit_gate(qc, QkGate_CX, cx_1_qargs, NULL);

then the result of ``qk_transpile_layout_initial_layout`` will be an array:
``[2, 1, 0]``

If routing was required to insert a swap gate to the circuit after layout was applied
this will result in a output permutation being set. For example, if a swap was inserted
like:

.. code-block:: c

    #include <qiskit.h>

    QkCircuit *qc = qk_circuit_new(3, 0)
    uint32_t h_qargs[1] = {2};
    qk_circuit_gate(qc, QkGate_H, h_qargs, NULL);
    uint32_t cx_0_qargs[2] = {2, 1};
    qk_circuit_gate(qc, QkGate_CX, cx_0_qargs, NULL);
    uint32_t swap_qargs[2] = {1, 0};
    qk_circuit_gate(qc, QkGate_Swap, swap_qargs, NULL);
    uint32_t cx_1_qargs[2] = {2, 1};
    qk_circuit_gate(qc, QkGate_CX, cx_1_qargs, NULL);

this results in the output state of qubit 0 moving to qubit 1 and qubit 1's to qubit 0.
This results in ``qk_transpile_layout_output_permutation`` returning the array:
``[1, 0, 2]`` to indicate this is the final position of each qubit's state after the
initial layout mapping. If no swaps or permutation elisions were made during the
transpilation this will be a trivial array of the form ``[0, 1, 2]`` as the output state
of the qubit is not moved by the transpiler.

Then combining these two is the final layout which is for tracking the final
position of a virtual qubit in the input circuit. So from the above example (with the swap),
the ``qk_transpile_layout_final_layout`` function would return ``[2, 0, 1]`` because
following the initial layout and then any routing permutation the final position of
qubit 0 in the input circuit is now physical qubit 2, virtual qubit 1 in the input circuit is
physical qubit 0, and virtual qubit 2 in the input circuit is physical qubit 1.

The transpiler will also allocate ancilla qubits to the circuit if the target
has more qubits available than the original input circuit. This is what
results in two functions ``qk_transpile_layout_num_input_qubits`` and
``qk_transpile_layout_num_output_qubits`` being necessary which tracks the
number of qubits in the input circuit and output circuit respectively. Additionally, the
``qk_transpile_layout_initial_layout`` and ``qk_transpile_layout_final_layout``
functions take an argument to filter ancillas from the output array. If set to true
the output array will be filtered to just the virtual qubits in the original input circuit
to the transpiler call that generated the ``QkTranspileLayout``.

Functions
=========

.. doxygengroup:: QkTranspileLayout
    :members:
    :content-only:
