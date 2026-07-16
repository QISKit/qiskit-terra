#!/usr/bin/env python3

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

"""Test cases to verify qpy backwards compatibility."""

import argparse
import itertools
import random
import sys

import numpy as np
from packaging.version import Version

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Clbit
from qiskit.circuit import Qubit
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info import Operator
from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, QFT, DCXGate, PauliGate
from qiskit.circuit.gate import Gate

try:
    from qiskit.qpy import dump, load
except ModuleNotFoundError:
    from qiskit.circuit.qpy_serialization import dump, load


def generate_full_circuit():
    """Generate a multiregister circuit with name, metadata, phase."""
    qr_a = QuantumRegister(4, "a")
    qr_b = QuantumRegister(4, "b")
    cr_c = ClassicalRegister(4, "c")
    cr_d = ClassicalRegister(4, "d")
    full_circuit = QuantumCircuit(
        qr_a,
        qr_b,
        cr_c,
        cr_d,
        name="MyCircuit",
        metadata={"test": 1, "a": 2},
        global_phase=3.14159,
    )
    full_circuit.h(qr_a)
    full_circuit.cx(qr_a, qr_b)
    full_circuit.barrier(qr_a)
    full_circuit.barrier(qr_b)
    full_circuit.measure(qr_a, cr_c)
    full_circuit.measure(qr_b, cr_d)
    return full_circuit


def generate_unitary_gate_circuit():
    """Generate a circuit with a unitary gate."""
    unitary_circuit = QuantumCircuit(5, name="unitary_circuit")
    unitary_circuit.unitary(random_unitary(32, seed=100), [0, 1, 2, 3, 4])
    unitary_circuit.measure_all()
    return unitary_circuit


def generate_random_circuits(version, forward_tests):
    """Generate multiple random circuits."""
    random_circuits = []
    for i in range(1, 15):
        qc = QuantumCircuit(i, name=f"random_circuit-{i}")
        qc.h(0)
        if i > 1:
            for j in range(i - 1):
                qc.cx(0, j + 1)
        qc.measure_all()
        for j in range(i):
            qc.reset(j)
            if (version.release >= (1, 4, 0) and forward_tests) or (
                version.release >= (2, 0, 0) and not forward_tests
            ):
                condition = (qc.cregs[0], i)
                body = QuantumCircuit([qc.qubits[0]])
                body.x(0)
                qc.if_else(condition, body, None, [qc.qubits[0]], [])
            else:
                qc.x(0).c_if(qc.cregs[0], i)
        for j in range(i):
            qc.measure(j, j)
        random_circuits.append(qc)
    return random_circuits


def generate_string_parameters():
    """Generate a circuit for the XYZ pauli string."""
    op_circuit = QuantumCircuit(3, name="X^Y^Z")
    op_circuit.append(PauliGate("XYZ"), op_circuit.qubits, [])
    return op_circuit


def generate_register_edge_cases():
    """Generate register edge case circuits."""
    register_edge_cases = []
    # Circuit with shared bits in a register
    qubits = [Qubit() for _ in range(5)]
    shared_qc = QuantumCircuit(name="shared_bits")
    shared_qc.add_bits(qubits)
    shared_qr = QuantumRegister(bits=qubits)
    shared_qc.add_register(shared_qr)
    shared_qc.h(shared_qr)
    shared_qc.cx(0, 1)
    shared_qc.cx(0, 2)
    shared_qc.cx(0, 3)
    shared_qc.cx(0, 4)
    shared_qc.measure_all()
    register_edge_cases.append(shared_qc)
    # Circuit with registers that have a mix of standalone and shared register
    # bits
    qr = QuantumRegister(5, "foo")
    qr = QuantumRegister(name="bar", bits=qr[:3] + [Qubit(), Qubit()])
    cr = ClassicalRegister(5, "foo")
    cr = ClassicalRegister(name="classical_bar", bits=cr[:3] + [Clbit(), Clbit()])
    hybrid_qc = QuantumCircuit(qr, cr, name="mix_standalone_bits_registers")
    hybrid_qc.h(0)
    hybrid_qc.cx(0, 1)
    hybrid_qc.cx(0, 2)
    hybrid_qc.cx(0, 3)
    hybrid_qc.cx(0, 4)
    hybrid_qc.measure(qr, cr)
    register_edge_cases.append(hybrid_qc)
    # Circuit with mixed standalone and shared registers
    qubits = [Qubit() for _ in range(5)]
    clbits = [Clbit() for _ in range(5)]
    mixed_qc = QuantumCircuit(name="mix_standalone_bits_with_registers")
    mixed_qc.add_bits(qubits)
    mixed_qc.add_bits(clbits)
    qr = QuantumRegister(bits=qubits)
    cr = ClassicalRegister(bits=clbits)
    mixed_qc.add_register(qr)
    mixed_qc.add_register(cr)
    qr_standalone = QuantumRegister(2, "standalone")
    mixed_qc.add_register(qr_standalone)
    cr_standalone = ClassicalRegister(2, "classical_standalone")
    mixed_qc.add_register(cr_standalone)
    mixed_qc.unitary(random_unitary(32, seed=42), qr)
    mixed_qc.unitary(random_unitary(4, seed=100), qr_standalone)
    mixed_qc.measure(qr, cr)
    mixed_qc.measure(qr_standalone, cr_standalone)
    register_edge_cases.append(mixed_qc)
    # Circuit with out of order register bits
    qr_standalone = QuantumRegister(2, "standalone")
    qubits = [Qubit() for _ in range(5)]
    clbits = [Clbit() for _ in range(5)]
    ooo_qc = QuantumCircuit(name="out_of_order_bits")
    ooo_qc.add_bits(qubits)
    ooo_qc.add_bits(clbits)
    random.seed(42)
    random.shuffle(qubits)
    random.shuffle(clbits)
    qr = QuantumRegister(bits=qubits)
    cr = ClassicalRegister(bits=clbits)
    ooo_qc.add_register(qr)
    ooo_qc.add_register(cr)
    qr_standalone = QuantumRegister(2, "standalone")
    cr_standalone = ClassicalRegister(2, "classical_standalone")
    ooo_qc.add_bits([qr_standalone[1], qr_standalone[0]])
    ooo_qc.add_bits([cr_standalone[1], cr_standalone[0]])
    ooo_qc.add_register(qr_standalone)
    ooo_qc.add_register(cr_standalone)
    ooo_qc.unitary(random_unitary(32, seed=42), qr)
    ooo_qc.unitary(random_unitary(4, seed=100), qr_standalone)
    ooo_qc.measure(qr, cr)
    ooo_qc.measure(qr_standalone, cr_standalone)
    register_edge_cases.append(ooo_qc)
    return register_edge_cases


def generate_parameterized_circuit():
    """Generate a circuit with parameters and parameter expressions."""
    param_circuit = QuantumCircuit(1, name="parameterized")
    theta = Parameter("theta")
    lam = Parameter("λ")
    theta_pi = 3.14159 * theta
    pe = theta_pi / lam
    param_circuit.append(U3Gate(theta, theta_pi, lam), [0])
    param_circuit.append(U1Gate(pe), [0])
    param_circuit.append(U2Gate(theta_pi, lam), [0])
    return param_circuit


def generate_qft_circuit():
    """Generate a QFT circuit with initialization."""
    k = 5
    state = (1 / np.sqrt(8)) * np.array(
        [
            np.exp(-1j * 2 * np.pi * k * (0) / 8),
            np.exp(-1j * 2 * np.pi * k * (1) / 8),
            np.exp(-1j * 2 * np.pi * k * (2) / 8),
            np.exp(-1j * 2 * np.pi * k * 3 / 8),
            np.exp(-1j * 2 * np.pi * k * 4 / 8),
            np.exp(-1j * 2 * np.pi * k * 5 / 8),
            np.exp(-1j * 2 * np.pi * k * 6 / 8),
            np.exp(-1j * 2 * np.pi * k * 7 / 8),
        ]
    )

    qubits = 3
    qft_circ = QuantumCircuit(qubits, qubits, name="QFT")
    qft_circ.initialize(state)
    qft_circ.append(QFT(qubits), range(qubits))
    qft_circ.measure(range(qubits), range(qubits))
    return qft_circ


def generate_param_phase():
    """Generate circuits with parameterize global phase."""
    output_circuits = []
    # Generate circuit with ParameterExpression global phase
    theta = Parameter("theta")
    phi = Parameter("phi")
    sum_param = theta + phi
    qc = QuantumCircuit(5, 1, global_phase=sum_param, name="parameter_phase")
    qc.h(0)
    for i in range(4):
        qc.cx(i, i + 1)
    qc.barrier()
    qc.rz(sum_param, range(3))
    qc.rz(phi, 3)
    qc.rz(theta, 4)
    qc.barrier()
    for i in reversed(range(4)):
        qc.cx(i, i + 1)
    qc.h(0)
    qc.measure(0, 0)
    output_circuits.append(qc)
    # Generate circuit with Parameter global phase
    theta = Parameter("theta")
    bell_qc = QuantumCircuit(2, global_phase=theta, name="bell_param_global_phase")
    bell_qc.h(0)
    bell_qc.cx(0, 1)
    bell_qc.measure_all()
    output_circuits.append(bell_qc)
    return output_circuits


def generate_single_clbit_condition_teleportation(version, forward_tests):
    """Generate single clbit condition teleportation circuit."""
    qr = QuantumRegister(1)
    cr = ClassicalRegister(2, name="name")
    teleport_qc = QuantumCircuit(qr, cr, name="Reset Test")
    teleport_qc.x(0)
    teleport_qc.measure(0, cr[0])
    if (version.release >= (1, 4, 0) and forward_tests) or (
        version.release >= (2, 0, 0) and not forward_tests
    ):
        condition = (cr[0], 1)
        body = QuantumCircuit([teleport_qc.qubits[0]])
        body.x(0)
        teleport_qc.if_else(condition, body, None, [teleport_qc.qubits[0]], [])
    else:
        teleport_qc.x(0).c_if(cr[0], 1)
    teleport_qc.measure(0, cr[1])
    return teleport_qc


def generate_parameter_vector():
    """Generate tests for parameter vector element ordering."""
    qc = QuantumCircuit(11, name="parameter_vector")
    input_params = ParameterVector("x_par", 11)
    user_params = ParameterVector("θ_par", 11)
    for i, param in enumerate(user_params):
        qc.ry(param, i)
    for i, param in enumerate(input_params):
        qc.rz(param, i)
    return qc


def generate_parameter_vector_expression():
    """Generate tests for parameter vector element ordering."""
    qc = QuantumCircuit(7, name="vector_expansion")
    entanglement = [[i, i + 1] for i in range(7 - 1)]
    input_params = ParameterVector("x_par", 14)
    user_params = ParameterVector("\u03b8_par", 1)

    for i in range(qc.num_qubits):
        qc.ry(user_params[0], qc.qubits[i])

    for source, target in entanglement:
        qc.cz(qc.qubits[source], qc.qubits[target])

    for i in range(qc.num_qubits):
        qc.rz(-2 * input_params[2 * i + 1], qc.qubits[i])
        qc.rx(-2 * input_params[2 * i], qc.qubits[i])

    return qc


def generate_evolution_gate():
    """Generate a circuit with a pauli evolution gate."""
    # Runtime import since this only exists in terra 0.19.0
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.synthesis import SuzukiTrotter
    from qiskit.quantum_info import SparsePauliOp

    synthesis = SuzukiTrotter()
    op = SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)])
    evo = PauliEvolutionGate([op] * 5, time=2.0, synthesis=synthesis)
    qc = QuantumCircuit(2, name="pauli_evolution_circuit")
    qc.append(evo, range(2))
    return qc


def generate_control_flow_circuits():
    """Test qpy serialization with control flow instructions."""
    from qiskit.circuit.controlflow import WhileLoopOp, IfElseOp, ForLoopOp

    # If instruction
    circuits = []
    qc = QuantumCircuit(2, 2, name="control_flow")
    qc.h(0)
    qc.measure(0, 0)
    true_body = QuantumCircuit(1)
    true_body.x(0)
    if_op = IfElseOp((qc.clbits[0], True), true_body=true_body)
    qc.append(if_op, [1])
    qc.measure(1, 1)
    circuits.append(qc)
    # If else instruction
    qc = QuantumCircuit(2, 2, name="if_else")
    qc.h(0)
    qc.measure(0, 0)
    false_body = QuantumCircuit(1)
    false_body.y(0)
    if_else_op = IfElseOp((qc.clbits[0], True), true_body, false_body)
    qc.append(if_else_op, [1])
    qc.measure(1, 1)
    circuits.append(qc)
    # While loop
    qc = QuantumCircuit(2, 1, name="while_loop")
    block = QuantumCircuit(2, 1)
    block.h(0)
    block.cx(0, 1)
    block.measure(0, 0)
    while_loop = WhileLoopOp((qc.clbits[0], 0), block)
    qc.append(while_loop, [0, 1], [0])
    circuits.append(qc)
    # for loop range
    qc = QuantumCircuit(2, 1, name="for_loop")
    body = QuantumCircuit(2, 1)
    body.h(0)
    body.cx(0, 1)
    body.measure(0, 0)
    if_body = QuantumCircuit(qc.qubits, qc.clbits)
    if_body.break_loop()
    body.if_else((0, True), if_body, None, qc.qubits, qc.clbits)
    for_loop_op = ForLoopOp(range(5), None, body=body)
    qc.append(for_loop_op, [0, 1], [0])
    circuits.append(qc)
    # For loop iterator
    qc = QuantumCircuit(2, 1, name="for_loop_iterator")
    for_loop_op = ForLoopOp(iter(range(5)), None, body=body)
    qc.append(for_loop_op, [0, 1], [0])
    circuits.append(qc)
    return circuits


def generate_for_loop_negative_circuits():
    """Test qpy serialization with for loop negative list."""
    # for loop negative list
    qc = QuantumCircuit(1, name="for_loop_negative_list")
    body = QuantumCircuit(1)
    body.x(0)
    qc.for_loop([-1, 0, 1], None, body, [0], [])
    return [qc]


def generate_control_flow_switch_circuits():
    """Generate circuits with switch-statement instructions."""
    from qiskit.circuit.controlflow import CASE_DEFAULT

    circuits = []

    qc = QuantumCircuit(2, 1, name="switch_clbit")
    case_t = qc.copy_empty_like()
    case_t.x(0)
    case_f = qc.copy_empty_like()
    case_f.z(1)
    qc.switch(qc.clbits[0], [(True, case_t), (False, case_f)], qc.qubits, qc.clbits)
    circuits.append(qc)

    qreg = QuantumRegister(2, "q")
    creg = ClassicalRegister(3, "c")
    qc = QuantumCircuit(qreg, creg, name="switch_creg")

    case_0 = QuantumCircuit(qreg, creg)
    case_0.x(0)
    case_1 = QuantumCircuit(qreg, creg)
    case_1.z(1)
    case_2 = QuantumCircuit(qreg, creg)
    case_2.x(1)
    qc.switch(
        creg, [(0, case_0), ((1, 2), case_1), ((3, 4, CASE_DEFAULT), case_2)], qc.qubits, qc.clbits
    )
    circuits.append(qc)

    return circuits


def generate_schedule_blocks(current_version):
    """Standard QPY testcase for schedule blocks."""

    from qiskit.pulse import builder, channels, library

    # Parameterized schedule test is avoided.
    # Generated reference and loaded QPY object may induce parameter uuid mismatch.
    # As workaround, we need test with bounded parameters, however, schedule.parameters
    # are returned as Set and thus its order is random.
    # Since schedule parameters are validated, we cannot assign random numbers.
    # We need to upgrade testing framework.
    schedule_blocks = []

    # Instructions without parameters
    with builder.build() as block:
        with builder.align_sequential():
            builder.set_frequency(5e9, channels.DriveChannel(0))
            builder.shift_frequency(10e6, channels.DriveChannel(1))
            builder.set_phase(1.57, channels.DriveChannel(0))
            builder.shift_phase(0.1, channels.DriveChannel(1))
            builder.barrier(channels.DriveChannel(0), channels.DriveChannel(1))
            gaussian_amp = 0.1
            gaussian_angle = 0.7
            if current_version.release < (1, 0, 0):
                builder.play(
                    library.Gaussian(160, gaussian_amp * np.exp(1j * gaussian_angle), 40),
                    channels.DriveChannel(0),
                )
            else:
                builder.play(
                    library.Gaussian(160, gaussian_amp, 40, gaussian_angle),
                    channels.DriveChannel(0),
                )
            builder.play(library.GaussianSquare(800, 0.1, 64, 544), channels.ControlChannel(0))
            builder.play(library.Drag(160, 0.1, 40, 1.5), channels.DriveChannel(1))
            builder.play(library.Constant(800, 0.1), channels.MeasureChannel(0))
            builder.acquire(1000, channels.AcquireChannel(0), channels.MemorySlot(0))
    schedule_blocks.append(block)
    # Raw symbolic pulse
    import symengine as sym

    duration, amp, t = sym.symbols("duration amp t")
    expr = amp * sym.sin(2 * sym.pi * t / duration)
    my_pulse = library.SymbolicPulse(
        pulse_type="Sinusoidal",
        duration=100,
        parameters={"amp": 0.1},
        envelope=expr,
        valid_amp_conditions=sym.Abs(amp) <= 1.0,
    )
    with builder.build() as block:
        builder.play(my_pulse, channels.DriveChannel(0))
    schedule_blocks.append(block)
    # Raw waveform
    my_waveform = 0.1 * np.sin(2 * np.pi * np.linspace(0, 1, 100))
    with builder.build() as block:
        builder.play(my_waveform, channels.DriveChannel(0))
    schedule_blocks.append(block)

    return schedule_blocks


def generate_referenced_schedule():
    """Test for QPY serialization of unassigned reference schedules."""

    from qiskit.pulse import builder, channels, library

    schedule_blocks = []

    # Completely unassigned schedule
    with builder.build() as block:
        builder.reference("cr45p", "q0", "q1")
        builder.reference("x", "q0")
        builder.reference("cr45m", "q0", "q1")
    schedule_blocks.append(block)

    # Partly assigned schedule
    with builder.build() as x_q0:
        builder.play(library.Constant(100, 0.1), channels.DriveChannel(0))
    with builder.build() as block:
        builder.reference("cr45p", "q0", "q1")
        builder.call(x_q0)
        builder.reference("cr45m", "q0", "q1")
    schedule_blocks.append(block)

    return schedule_blocks


def generate_calibrated_circuits():
    """Test for QPY serialization with calibrations."""

    from qiskit.pulse import builder, Constant, DriveChannel

    circuits = []

    # custom gate
    mygate = Gate("mygate", 1, [])
    qc = QuantumCircuit(1, name="calibrated_circuit_1")
    qc.append(mygate, [0])
    with builder.build() as caldef:
        builder.play(Constant(100, 0.1), DriveChannel(0))
    qc.add_calibration(mygate, (0,), caldef)
    circuits.append(qc)
    # override instruction
    qc = QuantumCircuit(1, name="calibrated_circuit_2")
    qc.x(0)
    with builder.build() as caldef:
        builder.play(Constant(100, 0.1), DriveChannel(0))
    qc.add_calibration("x", (0,), caldef)
    circuits.append(qc)

    return circuits


def generate_controlled_gates(version):
    """Test QPY serialization with custom ControlledGates."""
    circuits = []
    qc = QuantumCircuit(3, name="custom_controlled_gates")
    if version.release >= (2, 3, 0):
        controlled_gate = DCXGate().control(1, annotated=False)
    else:
        controlled_gate = DCXGate().control(1)
    qc.append(controlled_gate, [0, 1, 2])
    circuits.append(qc)
    custom_gate = Gate("black_box", 1, [])
    custom_definition = QuantumCircuit(1)
    custom_definition.h(0)
    custom_definition.sdg(0)
    custom_gate.definition = custom_definition
    nested_qc = QuantumCircuit(3, name="nested_qc")
    qc.append(custom_gate, [0])
    if version.release >= (2, 3, 0):
        controlled_gate = custom_gate.control(2, annotated=False)
    else:
        controlled_gate = custom_gate.control(2)
    nested_qc.append(controlled_gate, [0, 1, 2])
    circuits.append(nested_qc)
    qc_open = QuantumCircuit(2, name="open_cx")
    qc_open.cx(0, 1, ctrl_state=0)
    circuits.append(qc_open)
    return circuits


def generate_open_controlled_gates(version):
    """Test QPY serialization with custom ControlledGates with open controls."""
    circuits = []
    qc = QuantumCircuit(3, name="open_controls_simple")
    if version.release >= (2, 3, 0):
        controlled_gate = DCXGate().control(1, ctrl_state=0, annotated=False)
    else:
        controlled_gate = DCXGate().control(1, ctrl_state=0)
    qc.append(controlled_gate, [0, 1, 2])
    circuits.append(qc)

    custom_gate = Gate("black_box", 1, [])
    custom_definition = QuantumCircuit(1)
    custom_definition.h(0)
    custom_definition.sdg(0)
    custom_gate.definition = custom_definition
    nested_qc = QuantumCircuit(3, name="open_controls_nested")
    nested_qc.append(custom_gate, [0])
    if version.release >= (2, 3, 0):
        controlled_gate = custom_gate.control(2, ctrl_state=1, annotated=False)
    else:
        controlled_gate = custom_gate.control(2, ctrl_state=1)
    nested_qc.append(controlled_gate, [0, 1, 2])
    circuits.append(nested_qc)

    return circuits


def generate_acquire_instruction_with_kernel_and_discriminator():
    """Test QPY serialization with Acquire instruction with kernel and discriminator."""

    from qiskit.pulse import builder, AcquireChannel, MemorySlot, Discriminator, Kernel

    schedule_blocks = []

    with builder.build() as block:
        builder.acquire(
            100,
            AcquireChannel(0),
            MemorySlot(0),
            kernel=Kernel(
                name="my_kernel", my_params_1={"param1": 0.1, "param2": 0.2}, my_params_2=[0, 1]
            ),
        )
    schedule_blocks.append(block)

    with builder.build() as block:
        builder.acquire(
            100,
            AcquireChannel(0),
            MemorySlot(0),
            discriminator=Discriminator(
                name="my_disc", my_params_1={"param1": 0.1, "param2": 0.2}, my_params_2=[0, 1]
            ),
        )
    schedule_blocks.append(block)

    return schedule_blocks


def generate_layout_circuits():
    """Test qpy circuits with layout set."""

    from qiskit.transpiler.layout import TranspileLayout, Layout

    qr = QuantumRegister(3, "foo")
    qc = QuantumCircuit(qr, name="GHZ with layout")
    qc.h(0)
    qc.cx(0, 1)
    qc.swap(0, 1)
    qc.cx(0, 2)
    input_layout = {qr[index]: index for index in range(len(qc.qubits))}
    qc._layout = TranspileLayout(
        Layout(input_layout),
        input_qubit_mapping=input_layout,
        final_layout=Layout.from_qubit_list([qc.qubits[1], qc.qubits[0], qc.qubits[2]]),
    )
    return [qc]


def generate_clifford_circuits():
    """Test qpy circuits with Clifford operations."""
    from qiskit.quantum_info import Clifford

    cliff = Clifford.from_dict(
        {
            "stabilizer": ["-IZX", "+ZYZ", "+ZII"],
            "destabilizer": ["+ZIZ", "+ZXZ", "-XIX"],
        }
    )
    qc = QuantumCircuit(3, name="Clifford Circuits")
    qc.append(cliff, [0, 1, 2])
    return [qc]


def generate_annotated_circuits():
    """Test qpy circuits with annotated operations."""
    from qiskit.circuit import AnnotatedOperation, ControlModifier, InverseModifier, PowerModifier
    from qiskit.circuit.library import XGate, CXGate

    op1 = AnnotatedOperation(
        CXGate(), [InverseModifier(), ControlModifier(1), PowerModifier(1.4), InverseModifier()]
    )
    op2 = AnnotatedOperation(XGate(), InverseModifier())
    qc = QuantumCircuit(6, 1, name="Annotated circuits")
    qc.cx(0, 1)
    qc.append(op1, [0, 1, 2])
    qc.h(4)
    qc.append(op2, [1])
    return [qc]


def generate_control_flow_expr():
    """`IfElseOp`, `WhileLoopOp` and `SwitchCaseOp` with `Expr` nodes in their discriminators."""
    from qiskit.circuit.classical import expr, types

    body1 = QuantumCircuit(1)
    body1.x(0)
    qr1 = QuantumRegister(2, "q1")
    cr1 = ClassicalRegister(2, "c1")
    qc1 = QuantumCircuit(qr1, cr1, name="cf-expr1")
    qc1.if_test(expr.equal(cr1, 3), body1.copy(), [0], [])
    qc1.while_loop(expr.logic_not(cr1[1]), body1.copy(), [0], [])

    inner2 = QuantumCircuit(1)
    inner2.x(0)
    outer2 = QuantumCircuit(1, 1)
    outer2.if_test(expr.logic_not(outer2.clbits[0]), inner2, [0], [])
    qr2 = QuantumRegister(2, "q2")
    cr1_2 = ClassicalRegister(3, "c1")
    cr2_2 = ClassicalRegister(3, "c2")
    qc2 = QuantumCircuit(qr2, cr1_2, cr2_2, name="cf-expr2")
    qc2.if_test(expr.logic_or(expr.less(cr1_2, cr2_2), cr1_2[1]), outer2, [1], [1])

    inner3 = QuantumCircuit(1)
    inner3.x(0)
    outer3 = QuantumCircuit(1, 1)
    outer3.switch(expr.logic_not(outer2.clbits[0]), [(False, inner2)], [0], [])
    qr3 = QuantumRegister(2, "q2")
    cr1_3 = ClassicalRegister(3, "c1")
    cr2_3 = ClassicalRegister(3, "c2")
    qc3 = QuantumCircuit(qr3, cr1_3, cr2_3, name="cf-expr3")
    qc3.switch(expr.bit_xor(cr1_3, cr2_3), [(0, outer2)], [1], [1])

    cr1_4 = ClassicalRegister(256, "c1")
    cr2_4 = ClassicalRegister(4, "c2")
    cr3_4 = ClassicalRegister(4, "c3")
    inner4 = QuantumCircuit(1)
    inner4.x(0)
    outer_loose = Clbit()
    outer4 = QuantumCircuit(QuantumRegister(2, "q_outer"), cr2_4, [outer_loose], cr1_4)
    outer4.if_test(
        expr.logic_and(
            expr.logic_or(
                expr.greater(expr.bit_or(cr2_4, 7), 10),
                expr.equal(expr.bit_and(cr1_4, cr1_4), expr.bit_not(cr1_4)),
            ),
            expr.logic_or(
                outer_loose,
                expr.cast(cr1_4, types.Bool()),
            ),
        ),
        inner4,
        [0],
        [],
    )
    qc4_loose = Clbit()
    qc4 = QuantumCircuit(
        QuantumRegister(2, "qr4"), cr1_4, cr2_4, cr3_4, [qc4_loose], name="cf-expr4"
    )
    qc4.rz(np.pi, 0)
    qc4.switch(
        expr.logic_and(
            expr.logic_or(
                expr.logic_or(
                    expr.less(cr2_4, cr3_4),
                    expr.logic_not(expr.greater_equal(cr3_4, cr2_4)),
                ),
                expr.logic_or(
                    expr.logic_not(expr.less_equal(cr3_4, cr2_4)),
                    expr.greater(cr2_4, cr3_4),
                ),
            ),
            expr.logic_and(
                expr.equal(cr3_4, 2),
                expr.not_equal(expr.bit_xor(cr1_4, 0x0F), 0x0F),
            ),
        ),
        [(False, outer4)],
        [1, 0],
        list(cr2_4) + [qc4_loose] + list(cr1_4),
    )
    qc4.rz(np.pi, 0)

    return [qc1, qc2, qc3, qc4]


def generate_standalone_var():
    """Circuits that use standalone variables."""
    import uuid
    from qiskit.circuit.classical import expr, types

    # This is the low-level, non-preferred way to construct variables, but we need the UUIDs to be
    # deterministic between separate invocations of the script.
    uuids = [
        uuid.UUID(bytes=b"hello, qpy world", version=4),
        uuid.UUID(bytes=b"not a good uuid4", version=4),
        uuid.UUID(bytes=b"but it's ok here", version=4),
        uuid.UUID(bytes=b"any old 16 bytes", version=4),
        uuid.UUID(bytes=b"and another load", version=4),
    ]
    a = expr.Var(uuids[0], types.Bool(), name="a")
    b = expr.Var(uuids[1], types.Bool(), name="θψφ")
    b_other = expr.Var(uuids[2], types.Bool(), name=b.name)
    c = expr.Var(uuids[3], types.Uint(8), name="🐍🐍🐍")
    d = expr.Var(uuids[4], types.Uint(8), name="d")

    qc = QuantumCircuit(1, 1, inputs=[a], name="standalone_var")
    qc.add_var(b, expr.logic_not(a))

    qc.add_var(c, expr.lift(0, c.type))
    with qc.if_test(b) as else_:
        qc.store(c, expr.lift(3, c.type))
        with qc.while_loop(b):
            qc.add_var(c, expr.lift(7, c.type))
    with else_:
        qc.add_var(d, expr.lift(7, d.type))

    qc.measure(0, 0)
    with qc.switch(c) as case:
        with case(0):
            qc.store(b, True)
        with case(1):
            qc.store(qc.clbits[0], False)
        with case(2):
            # Explicit shadowing.
            qc.add_var(b_other, True)
        with case(3):
            qc.store(a, False)
        with case(case.DEFAULT):
            pass

    return [qc]


def generate_v12_expr():
    """Circuits that contain the `Index` and bitshift operators new in QPY v12."""
    import uuid
    from qiskit.circuit.classical import expr, types

    a = expr.Var(uuid.UUID(bytes=b"hello, qpy world", version=4), types.Uint(8), name="a")
    cr = ClassicalRegister(4, "cr")

    index = QuantumCircuit(cr, inputs=[a], name="index_expr")
    index.store(expr.index(cr, 0), expr.index(a, a))

    shift = QuantumCircuit(cr, inputs=[a], name="shift_expr")
    with shift.if_test(expr.equal(expr.shift_right(expr.shift_left(a, 1), 1), a)):
        pass

    return [index, shift]


def generate_replay_with_expressions():
    """Circuits with parameters that have expressions in the replay"""
    out = []

    a = Parameter("a")
    b = Parameter("b")
    c = Parameter("c")
    d = Parameter("d")

    a1 = a * 2
    a2 = a1.subs({a: 3 * b})
    qc = QuantumCircuit(1)
    qc.rz(a2, 0)
    out.append(qc)

    qc2 = QuantumCircuit(1)
    theta = Parameter("θ")
    rz = Parameter("rz")
    exp = theta + np.pi
    exp = exp.subs({theta: rz})
    exp = exp.subs({rz: theta})
    qc2.rz(exp, 0)
    out.append(qc2)

    pv = ParameterVector("rz", 2)
    qc3 = QuantumCircuit(1, name="subs-vector")
    qc3.rz((pv[0] + 0.5).subs({pv[0]: pv[1]}), 0)

    everything = QuantumCircuit(1, name="everything")
    expression = (a + b.sin() * 0.25) * c**2
    final_expr = (
        (expression.cos() + d.arccos() - d.arcsin() + d.arctan() + d.tan()) / d.exp()
        + expression.gradient(a)
        + expression.log().sign()
        - a.sin()
        - b.conjugate()
    )
    final_expr = final_expr.abs()
    final_expr = final_expr.subs({c: a})
    everything.rz(final_expr, 0)
    out.append(everything)

    return out


def generate_replay_with_cancellations():
    # Qiskit versions between 2.2.0rc1 and 2.4.0rc2 inclusive generate invalid QPY files for
    # expressions involving cancellations.
    a = Parameter("a")
    b = Parameter("b")
    pv = ParameterVector("rz", 2)

    out = []
    cases = (
        0 * a,
        0 * a + 2,
        0 * a + b,
        a - a,
        0 * pv[0],
        0 * (pv[0] + pv[1] + a),
    )
    for i, case in enumerate(cases):
        cancelled = QuantumCircuit(1, name=f"cancellations-{i}")
        cancelled.rz(case, 0)
        out.append(cancelled)
    return out


def generate_v14_expr():
    """Circuits that contain expressions and types new in QPY v14."""
    import uuid
    from qiskit.circuit.classical import expr
    from qiskit.circuit import Duration

    float_expr = QuantumCircuit(name="float_expr")
    with float_expr.if_test(expr.less(1.0, 2.0)):
        pass

    duration_expr = QuantumCircuit(name="duration_expr")
    with duration_expr.if_test(
        expr.logic_and(
            expr.logic_and(
                expr.equal(Duration.dt(1), Duration.ns(2)),
                expr.equal(Duration.us(3), Duration.ms(4)),
            ),
            expr.equal(Duration.s(5), Duration.dt(6)),
        )
    ):
        pass

    math_expr = QuantumCircuit(name="math_expr")
    with math_expr.if_test(
        expr.logic_and(
            expr.logic_and(
                expr.equal(expr.mul(Duration.dt(1), 2.0), expr.div(Duration.ns(2), 2.0)),
                expr.equal(
                    expr.add(Duration.us(3), Duration.us(4)),
                    expr.sub(Duration.ms(5), Duration.ms(6)),
                ),
            ),
            expr.logic_and(
                expr.equal(expr.mul(1.0, 2.0), expr.div(4.0, 2.0)),
                expr.equal(expr.add(3.0, 4.0), expr.sub(10.5, 4.3)),
            ),
        )
    ):
        pass

    stretch_expr = QuantumCircuit(name="stretch_expr")
    s = expr.Stretch(uuid.UUID(bytes=b"hello, qpy world", version=4), "a")
    stretch = stretch_expr.add_stretch(s)
    with stretch_expr.if_test(expr.equal(stretch, Duration.dt(100))):
        pass

    return [
        float_expr,
        duration_expr,
        math_expr,
        stretch_expr,
    ]


def generate_box():
    """Circuits that contain `Box`.  Only added in Qiskit 2.0."""
    bare = QuantumCircuit(2, name="box-bare")
    with bare.box():
        bare.h(0)
        bare.cx(0, 1)

    nested = QuantumCircuit(2, name="box-nested")
    with nested.box():
        with nested.box(duration=2, unit="dt"):
            nested.h(0)
            nested.cx(0, 1)
        with nested.box(duration=200.0, unit="ns"):
            nested.x(0)
            nested.noop(1)

    labeled = QuantumCircuit(2, name="labeled boxes")
    with labeled.box():
        labeled.cx(0, 1)
    with labeled.box(duration=1, unit="dt", label="hello"):
        with labeled.box(duration=2.5, unit="s", label="world"):
            labeled.cx(0, 1)

    return [bare, nested, labeled]


def generate_delay():
    """Circuits that contain a delay"""
    qc = QuantumCircuit(1, name="delay dt")
    qc.delay(1, 0)
    qc.delay(9223372036854775806, 0)
    return qc


def generate_delay_stretch():
    """Circuits that contain a stretch delay. Added in QPY 14 in Qiskit 2.0."""
    from qiskit.circuit.classical import expr
    from qiskit.circuit import Duration
    import uuid

    stretch_expr = QuantumCircuit(1, name="stretch_expr_delay_circuit")
    s = expr.Stretch(uuid.UUID(bytes=b"hallo, QPY_world", version=4), "a")
    stretch = stretch_expr.add_stretch(s)
    stretch_expr.delay(stretch, 0)
    stretch_expr.delay(expr.add(Duration.dt(200), stretch), 0)
    stretch_expr.delay(expr.sub(Duration.ns(3.14159), stretch), 0)
    return [stretch_expr]


def generate_pauli_product_measurement():
    """Circuits that contain a Pauli Product Measurement gate"""
    from qiskit.circuit.library import PauliProductMeasurement
    from qiskit.quantum_info import Pauli

    ppm = PauliProductMeasurement(Pauli("ZXY"))
    qc = QuantumCircuit(ppm.num_qubits, ppm.num_clbits)
    qc.append(ppm, range(ppm.num_qubits), range(ppm.num_clbits))
    return [qc]


def generate_pauli_product_rotation():
    """Circuits that contain a Pauli Product Rotation gate"""
    from qiskit.circuit.library import PauliProductRotationGate
    from qiskit.quantum_info import Pauli

    ppr = PauliProductRotationGate(Pauli("ZXY"), angle=1.2)
    qc = QuantumCircuit(ppr.num_qubits)
    qc.append(ppr, range(ppr.num_qubits))
    return [qc]


def generate_circuits(
    generating_version, current_version, load_context=False, qpy_version=None, forward_tests=False
):
    """Generate reference circuits.

    If load_context is True, avoid generating Pulse-based reference
    circuits. For those circuits, load_qpy only checks that the cached
    circuits can be loaded without erroring."""

    output_circuits = {
        "full.qpy": [generate_full_circuit()],
        "unitary.qpy": [generate_unitary_gate_circuit()],
        "multiple.qpy": generate_random_circuits(current_version, forward_tests),
        "string_parameters.qpy": [generate_string_parameters()],
        "register_edge_cases.qpy": generate_register_edge_cases(),
        "parameterized.qpy": [generate_parameterized_circuit()],
        "delay.qpy": [generate_delay()],
    }

    if generating_version.release >= (0, 18, 1):
        output_circuits["qft_circuit.qpy"] = [generate_qft_circuit()]
        output_circuits["teleport.qpy"] = [
            generate_single_clbit_condition_teleportation(current_version, forward_tests)
        ]

    if generating_version.release >= (0, 19, 0):
        output_circuits["param_phase.qpy"] = generate_param_phase()

    if generating_version.release >= (0, 19, 1):
        output_circuits["parameter_vector.qpy"] = [generate_parameter_vector()]
        output_circuits["pauli_evo.qpy"] = [generate_evolution_gate()]
        output_circuits["parameter_vector_expression.qpy"] = [
            generate_parameter_vector_expression()
        ]
    if generating_version.release >= (0, 19, 2):
        output_circuits["control_flow.qpy"] = generate_control_flow_circuits()
    if (0, 21, 0) <= generating_version.release < (2, 0):
        output_circuits["schedule_blocks.qpy"] = (
            None if load_context else generate_schedule_blocks(current_version)
        )
        output_circuits["pulse_gates.qpy"] = (
            None if load_context else generate_calibrated_circuits()
        )
    if (0, 24, 0) <= generating_version.release < (2, 0):
        output_circuits["referenced_schedule_blocks.qpy"] = (
            None if load_context else generate_referenced_schedule()
        )
    if generating_version.release >= (0, 24, 0):
        output_circuits["control_flow_switch.qpy"] = generate_control_flow_switch_circuits()
    if generating_version.release >= (0, 24, 1):
        output_circuits["open_controlled_gates.qpy"] = generate_open_controlled_gates(
            current_version
        )
        output_circuits["controlled_gates.qpy"] = generate_controlled_gates(current_version)
    if generating_version.release >= (0, 24, 2):
        output_circuits["layout.qpy"] = generate_layout_circuits()
    if (0, 25, 0) <= generating_version.release < (2, 0):
        output_circuits["acquire_inst_with_kernel_and_disc.qpy"] = (
            None if load_context else generate_acquire_instruction_with_kernel_and_discriminator()
        )
        output_circuits["control_flow_expr.qpy"] = generate_control_flow_expr()
    if generating_version.release >= (0, 45, 2):
        output_circuits["clifford.qpy"] = generate_clifford_circuits()
    if generating_version.release >= (1, 0, 0):
        output_circuits["annotated.qpy"] = generate_annotated_circuits()
    if generating_version.release >= (1, 1, 0):
        output_circuits["standalone_vars.qpy"] = generate_standalone_var()
        output_circuits["v12_expr.qpy"] = generate_v12_expr()
    if generating_version.release >= (1, 4, 1):
        output_circuits["replay_with_expressions.qpy"] = generate_replay_with_expressions()
        # Qiskit versions between 2.2.0rc1 and 2.4.0rc2 inclusive generate invalid QPY files for
        # expressions involving cancellations.
        if not Version("2.2.0rc1") <= generating_version <= Version("2.4.0rc2"):
            output_circuits["replay_with_cancellations.qpy"] = generate_replay_with_cancellations()

    if (
        generating_version.release >= (2, 0, 0)
        and (qpy_version is None or qpy_version > 13)
        and current_version.release >= (2, 0, 0)
    ):
        output_circuits["v14_expr.qpy"] = generate_v14_expr()
        output_circuits["box.qpy"] = generate_box()
        output_circuits["delay_stretch.qpy"] = generate_delay_stretch()

    # The Pauli Product Measurement gate was added in 2.3.0, but a bug in 2.4 prevents it from being read correctly
    if (
        generating_version.release >= (2, 3, 0)
        and current_version.release >= (2, 3, 0)
        and (not forward_tests or current_version.release[1] != 4)
    ):
        output_circuits["ppm.qpy"] = generate_pauli_product_measurement()

    # The Pauli Product Rotation gate was added in 2.4.0, but a bug in 2.4 prevents it from being read correctly
    if (
        generating_version.release >= (2, 4, 0)
        and current_version.release >= (2, 4, 0)
        and (not forward_tests or current_version.release[1] != 4)
        and (
            qpy_version is None or qpy_version >= 17
        )  # PPR gates are not supported in Python QPY, which currently handles qpy versions up to 16
    ):
        output_circuits["ppr.qpy"] = generate_pauli_product_rotation()

    # The excluded versions have a bug in the Rust code and couldn't use negative indices.
    if (
        Version("0.19.2") < generating_version
        and (not Version("2.0.0a1") <= generating_version < Version("2.5.0"))
        and (not Version("2.0.0a1") <= current_version < Version("2.5.0"))
    ):
        output_circuits["for_loop_negative.qpy"] = generate_for_loop_negative_circuits()

    return output_circuits


def assert_equal(
    reference,
    qpy,
    count,
    generating_version,
    bind=None,
    equivalent=False,
    context="Unknown context",
):
    """Compare two circuits."""
    reference_parameter_names = [x.name for x in reference.parameters]
    qpy_parameter_names = [x.name for x in qpy.parameters]
    if bind is not None:
        reference = reference.assign_parameters(bind[: reference.num_parameters])
        qpy = qpy.assign_parameters(bind[: qpy.num_parameters])

    if equivalent:
        if not Operator.from_circuit(reference).equiv(Operator.from_circuit(qpy)):
            msg = (
                f"QPY Error: \nFor {context}:\n"
                f"Reference Circuit {count}:\n{reference}\nis not equivalent to "
                f"qpy loaded circuit {count}:\n{qpy}\n"
            )
            sys.stderr.write(msg)
            sys.exit(1)
    elif reference != qpy:
        msg = (
            f"QPY Error: \nFor {context}:\n"
            f"Reference Circuit {count}:\n{reference}\nis not equal to "
            f"qpy loaded circuit {count}:\n{qpy}\n"
        )
        sys.stderr.write(msg)
        sys.exit(1)

    if reference_parameter_names != qpy_parameter_names:
        msg = (
            f"QPY Error: "
            f"Circuit {count} parameter mismatch:"
            f" {reference_parameter_names} != {qpy_parameter_names}\n"
        )
        sys.stderr.write(msg)
        sys.exit(4)

    # Check deprecated bit properties, if set.  The QPY dumping code before Terra 0.23.2 didn't
    # include enough information for us to fully reconstruct this, so we only test if newer.
    if generating_version.release >= (0, 23, 2) and isinstance(reference, QuantumCircuit):
        for ref_bit, qpy_bit in itertools.chain(
            zip(reference.qubits, qpy.qubits), zip(reference.clbits, qpy.clbits)
        ):
            if ref_bit._register is not None and ref_bit != qpy_bit:
                msg = (
                    f"QPY Error: "
                    f"For {context}:\n"
                    f"Reference Circuit {count}:\n"
                    "deprecated bit-level register information mismatch\n"
                    f"reference bit: {ref_bit}\n"
                    f"loaded bit: {qpy_bit}\n"
                )
                sys.stderr.write(msg)
                sys.exit(1)

    if (
        generating_version.release >= (0, 24, 2)
        and isinstance(reference, QuantumCircuit)
        and reference.layout != qpy.layout
    ):
        msg = (
            f"QPY Error: "
            f"For {context}:\nCircuit {count} layout mismatch {reference.layout} != {qpy.layout}\n"
        )
        sys.stderr.write(msg)
        sys.exit(4)

    # Don't compare name on bound circuits
    if bind is None and reference.name != qpy.name:
        msg = f"QPY Error: \nFor {context}:\nCircuit {count} name mismatch {reference.name} != {qpy.name}\n{reference}\n{qpy}"
        sys.stderr.write(msg)
        sys.exit(2)
    if reference.metadata != qpy.metadata:
        msg = f"QPY Error: \nFor {context}:\nCircuit {count} metadata mismatch: {reference.metadata} != {qpy.metadata}"
        sys.stderr.write(msg)
        sys.exit(3)


def generate_qpy(qpy_files, qpy_version=None):
    """Generate qpy files from reference circuits."""
    for path, circuits in qpy_files.items():
        with open(path, "wb") as fd:
            if qpy_version is None:  # avoid the version param because it's missing in old qiskit
                dump(circuits, fd)
            else:
                dump(circuits, fd, version=qpy_version)


def load_qpy(qpy_files, generating_version):
    """Load qpy circuits from files and compare to reference circuits."""
    pulse_files = {
        "schedule_blocks.qpy": (0, 21, 0),
        "pulse_gates.qpy": (0, 21, 0),
        "referenced_schedule_blocks.qpy": (0, 24, 0),
        "acquire_inst_with_kernel_and_disc.qpy": (0, 25, 0),
    }
    for path, circuits in qpy_files.items():
        if path in pulse_files.keys():
            # Qiskit Pulse was removed in version 2.0. Loading ScheduleBlock payloads
            # raises an exception and loading pulse gates results with undefined instructions
            # so not loading and comparing these payloads.
            # See https://github.com/Qiskit/qiskit/pull/13814
            continue
        print(f"Loading qpy file: {path}")  # noqa: T201
        try:
            with open(path, "rb") as fd:
                qpy_circuits = load(fd)
        except Exception as ex:
            msg = f"QPY Error: Failed to load {path} with the exception: {ex}\n"
            sys.stderr.write(msg)
            sys.exit(1)
        equivalent = path in {"open_controlled_gates.qpy", "controlled_gates.qpy"}
        for i, circuit in enumerate(circuits):
            bind = None
            if path == "parameterized.qpy":
                bind = [1, 2]
            elif path == "param_phase.qpy":
                if i == 0:
                    bind = [1, 2]
                else:
                    bind = [1]
            elif path == "parameter_vector.qpy":
                bind = np.linspace(1.0, 2.0, 22)
            elif path == "parameter_vector_expression.qpy":
                bind = np.linspace(1.0, 2.0, 15)
            elif path in ("replay_with_expressions.qpy", "replay_with_cancellations.qpy"):
                bind = [2.0, 1.5, 0.125, -0.25, 0.75]

            context = f"Version {generating_version}, QPY file {path}, Circuit number {i}"
            assert_equal(
                circuit,
                qpy_circuits[i],
                i,
                generating_version,
                bind=bind,
                equivalent=equivalent,
                context=context,
            )

    from qiskit.qpy.exceptions import QpyError

    for path, min_version in pulse_files.items():

        # version_parts is the version of Qiskit used to generate the payloads being loaded in this test.
        # min_version is the minimal version of Qiskit this pulse payload was generated with.
        if generating_version.release < min_version or generating_version.release >= (2, 0):
            continue

        if path == "pulse_gates.qpy":
            try:
                with open(path, "rb") as fd:
                    load(fd)
            except Exception:
                msg = "QPY Error:\nLoading circuit with pulse gates should not raise\n"
                sys.stderr.write(msg)
                sys.exit(1)
        else:
            try:
                # A ScheduleBlock payload, should raise QpyError.
                with open(path, "rb") as fd:
                    load(fd)
            except QpyError:
                continue

            msg = f"QPY Error:\nLoading payload {path} didn't raise QpyError\n"
            sys.stderr.write(msg)
            sys.exit(1)


def _main():
    parser = argparse.ArgumentParser(description="Test QPY backwards compatibility")
    parser.add_argument("command", choices=["generate", "load", "load_forward"])
    parser.add_argument(
        "--version",
        "-v",
        help=(
            "Optionally specify the version being tested. "
            "This will enable additional circuit features "
            "to test generating and loading QPY."
        ),
    )
    parser.add_argument(
        "--qpy-version",
        type=int,
        default=None,
        help=(
            "Optionally specify the QPY format version to use for serialization. "
            "If not specified, the default (latest) QPY version is used."
        ),
    )
    args = parser.parse_args()

    current_version = Version(qiskit.__version__)

    # Terra 0.18.0 was the first release with QPY, so that's the default.
    generating_version = Version(args.version or "0.18.0")
    if args.command == "generate":
        qpy_files = generate_circuits(
            generating_version, current_version, qpy_version=args.qpy_version
        )
        generate_qpy(qpy_files, qpy_version=args.qpy_version)
    else:
        forward_tests = args.command == "load_forward"
        print(  # noqa: T201
            f"QPY {'forwards' if forward_tests else 'backwards'} compatibility loader tests running\n"
            f"generating_version={generating_version}\n"
            f"current_version={current_version}\n"
            f"QPY version={args.qpy_version if args.qpy_version else 'default'}"
        )
        qpy_files = generate_circuits(
            generating_version,
            current_version,
            load_context=True,
            qpy_version=args.qpy_version,
            forward_tests=forward_tests,
        )
        load_qpy(qpy_files, generating_version)


if __name__ == "__main__":
    _main()
