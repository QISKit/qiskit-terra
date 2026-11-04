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

"""FFI to call qiskit C API."""

import ctypes

import qiskit
from qiskit import capi
import numpy


def build_homogenous_target(
    cmap: qiskit.transpiler.CouplingMap,
    basis_gates: list[str],
    seed: int,
    ideal_gates: bool = False,
) -> ctypes.POINTER(capi.QkTarget):
    """Build a C API QkTarget with a homogenous gate set.

    This function will build a target taking a coupling map and basis gate
    list. It assumes that all 1q gates are on all qubits and all 2q gates are on
    all coupling map edges. Larger gates (and 0 qubit gates) are not supported
    and will error. All gates will have a random error rate and duration assigned on
    each qubit. Measurements will be added as a supported instruction on
    every qubit unconditionally.

    Args:
        cmap: The coupling map representing the connectivity of the target to generate
        basis_gates: The list of standard gate name strings to use in the target
        seed: An rng seed
        ideal_gates: If set to False no error rate or duration will be assigned to
            any instruction on the target.

    Returns:
        A ctypes pointer to the C API QkTarget object
    """

    name_mapping = qiskit.circuit.library.standard_gates.get_standard_gate_name_mapping()
    c_target = capi.qk_target_new(cmap.size())
    rng = numpy.random.default_rng(seed)
    for gate in basis_gates:
        gate_obj = name_mapping[gate]
        entry = capi.qk_target_entry_new(int(gate_obj._standard_gate))
        if gate_obj.num_qubits == 2:
            for edge in cmap.get_edges():
                qubits = ctypes.cast((ctypes.c_uint32 * 2)(*edge), ctypes.POINTER(ctypes.c_uint32))
                if not ideal_gates:
                    error = rng.uniform(0.0, 0.1)
                    duration = rng.uniform(0.0, 1.0)
                else:
                    error = float("nan")
                    duration = float("nan")
                capi.qk_target_entry_add_property(entry, qubits, 2, duration, error)
        elif gate_obj.num_qubits == 1:
            for qubit in range(cmap.size()):
                qubits = ctypes.byref(ctypes.c_uint32(qubit))
                if not ideal_gates:
                    error = rng.uniform(0.0, 0.01)
                    duration = rng.uniform(0.0, 1.0)
                else:
                    error = float("nan")
                    duration = float("nan")
                capi.qk_target_entry_add_property(entry, qubits, 1, duration, error)
        else:
            raise qiskit.transpiler.exceptions.TranspilerError(f"Invalid gate: {gate}")
        capi.qk_target_add_instruction(c_target, entry)
    measure_entry = capi.qk_target_entry_new_measure()
    for qubit in range(cmap.size()):
        qubits = ctypes.byref(ctypes.c_uint32(qubit))
        if not ideal_gates:
            error = rng.uniform(0.0, 0.1)
            duration = rng.uniform(0.0, 1.0)
        else:
            error = float("nan")
            duration = float("nan")

        capi.qk_target_entry_add_property(measure_entry, qubits, 1, duration, error)
    capi.qk_target_add_instruction(c_target, measure_entry)
    return c_target


def transpile_from_c(
    circuit: qiskit.circuit.QuantumCircuit,
    target: ctypes.POINTER(capi.QkTarget),
    optimization_level,
    approximation_degree,
    seed,
) -> qiskit.QuantumCircuit:
    """Transpile a circuit and return it as a python circuit."""
    args = [
        2 if optimization_level is None else optimization_level,
        -1 if seed is None else seed,
        1.0 if approximation_degree is None else approximation_degree,
    ]
    options = capi.QkTranspileOptions(*args)
    result = capi.QkTranspileResult(None, None)
    error = ctypes.pointer(ctypes.c_char())
    res = capi.qk_transpile(
        capi.qk_circuit_borrow_from_python(circuit._data),
        target,
        ctypes.byref(options),
        ctypes.byref(result),
        ctypes.byref(error),
    )
    if res != 0:
        raise qiskit.transpiler.exceptions.TranspilerError(
            f"Transpilation failed: {error.contents.value.decode('utf8')}"
        )
    layout = capi.qk_transpile_layout_to_python(result.layout, result.circuit)
    capi.qk_transpile_layout_free(result.layout)
    out = capi.qk_circuit_to_python_full(result.circuit)
    out._layout = layout
    return out
