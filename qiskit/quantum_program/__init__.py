# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
=================================================
Quantum programs (:mod:`qiskit.quantum_program`)
=================================================

.. currentmodule:: qiskit.quantum_program

A quantum program describes a quantum-centric computation. The heart of such a computation is a
shot loop, which can optionally be prefixed or suffixed with classical operations that prepare
input parameter values for the circuits in the shot loop, or post-process samples collected
from running the shot loop. A quantum program is itself an inert structure: it describes a
computation without performing one.

All data processing inside of a quantum program is tensor based. A quantum program is a data flow
of tensors. Quantum work is therefore structured as an operation that maps tensors of parameter
value data to bit-valued arrays of output data via quantum circuits. Examples of pre-processing
include tasks like randomizing circuit paramaters. Examples of post-processing include tasks like
computing expectation values or performing post-selection. Quantum programs do not contain
control flow; they are not intended to be used to orchestrate closed loop experiments like VQE.
Instead, they aim to capture the semantics of out-of-coherence operations that have the potential
to reduce the amount of data traveling to and from the Quantum Computer by formatting it in the
precise way that an experiment requires.

A program's inputs and outputs are expressed as a data tree with tensor-valued leaves. The
tree structure is user specified.

Writing a program
=================

A :func:`shot_loop` creates an instruction that is the core quantum execution unit. It holds a
list of quantum circuits and a shot count, and represents a directive to sample classical
register values from each circuit for that number of shots::

    from qiskit.circuit import QuantumCircuit
    from qiskit.quantum_program import build, shot_loop

    circuit = QuantumCircuit(5)
    circuit.h(0)
    circuit.measure_all()

    outcomes = shot_loop([circuit], shots=1024)
    outcomes[0]["meas"].type      # TensorType(Bit[1024, 5])

A shot loop produces one value per classical register of each circuit, addressed by circuit and
register name, whose trailing axis is that register's own bits in order, so element ``i`` of it is
bit ``i`` of the register. Each value is a :class:`Tracer`, which stands for a value the program
will produce rather than one in hand. The arithmetic operators, the bitwise operators and the
reductions all build another tracer, so post-processing is written as ordinary arithmetic, and each
operation reports the type it produces as it is written, so a dtype or shape mistake is raised at
the line that made it::

    excited = outcomes[0]["meas"].mean(axis=0)
    excited.type                  # TensorType(F64[5])

A tracer records an expression and nothing else. :func:`build` names the outputs, with whatever
nesting they should come back in, and gives the program::

    program = build({"excited": excited, "samples": outcomes})
    program.output_types()
    # DataTree([excited: TensorType(F64[5]), samples: [[meas: TensorType(Bit[1024, 5])]]])

Only the expressions the declared outputs reach are built, so a value used twice becomes one
instruction and a value nothing uses costs nothing.

Circuit parameters
==================

When a circuit is parametric, it requires a tensor input whose last axis matches the number of
parameters, in the order :attr:`.QuantumCircuit.parameters` gives them, and whose leading axes
result in a sweep over parameter value sets. In this case, the output tensor for some classical
register of the circuit has shape ``(*leading_axes, shots, creg_size)``, i.e. the same number of
shots are collected for every parametric configuration of the circuit::

    from qiskit.circuit import Parameter, QuantumCircuit
    from qiskit.quantum_program import f64, qp_input, shot_loop

    circuit1 = QuantumCircuit(2, 2)
    circuit1.measure([0, 1], [0, 1])

    circuit2 = QuantumCircuit(2, 2)
    circuit2.rx(Parameter("phi"), 0)
    circuit2.rx(Parameter("theta"), 1)
    circuit2.measure([0, 1], [0, 1])

    angles = qp_input("angles", f64[16, 2])
    outcomes = shot_loop([circuit1, circuit2], shots=1024, parameter_values=[None, angles])
    outcomes[1]["c"].type         # TensorType(Bit[16, 1024, 2])

In the above, supplying ``None`` as the parameter value array is equivalent to supplying an empty
constant array.

:func:`qp_input` declares a typed input, supplied when the program is called, so ``angles`` is
whatever the caller passes. The program input ``angles`` could also be replaced, for example, by a
constant array ``np.linspace(0, 1, 32).reshape(16, 2)``.

Running a program
=================

Circuits are sampled by a backend, so a program holding a shot loop cannot run in process,
without a QPU or simulator present. However, a program of only classical operations can be run
directly without any external tools, which may be helpful for setting up and debugging portions
of a program. To do this, provide a keyword argument per declared input::

    x = qp_input("x", f64[3])
    program = build({"z": (x * x).mean(axis=0)})
    program(x=[1.0, 2.0, 3.0])    # DataTree([z: array(4.66666667)])

Looking at a program
====================

A program, or partially constructed program, can be visualized either as a code listing::

    print(program.listing())

which gives::

    @0: // entry point
      %0: F64[3] = qiskit.parameter x
      %1: F64[3] = qiskit.multiply(%0, %0)
      %2: F64[] = qiskit.mean[axis=0](%1)
      results:
        z = %2

Or as a data-flow directed acyclic graph, where every edge represents a tensor, and every
instruction a tensor operation, ``program.draw()``, which needs the optional Graphviz and Pillow
packages. The
:func:`listing` and :func:`draw` functions do the same for one :class:`Tracer`.

Reading a program
=================

A built program is readable instruction by instruction. It holds one or more
:class:`ProgramFunction`, ending at the entry point, and each function is a sequence of
:class:`Instruction`::

    for instruction in program.entry:
        print(instruction.role, instruction.full_name, instruction.output_types())

An instruction's operands are :class:`Value` objects, which refer positionally to
a previous instruction, and to one of that instruction's results::

    for value in instruction.operands:
        producer = function[value.instruction]
        value_type = producer.output_types()[value.slot]

Each operation is a member of :mod:`qiskit.quantum_program.ops`::

    match instruction.op:
        case ops.ShotLoop() as op:
            sample(op.circuits(), op.shots)
        case ops.Mean() as op:
            average(op.axis)

Partitioning a program
======================

A backend with more than one execution resource, such as a QPU beside a classical processor, can have
each resource's work put into a function of its own. Declaring the ops each resource handles gives
back an equivalent program, plus the resource each of its functions belongs to::

    parts, resources = program.partition([["qiskit.multiply"]])
    parts.num_functions           # 2
    resources                     # [0]

An op no resource declares stays in the entry point, where Qiskit evaluates it in process, so here
the mean is left where it was and only the multiplication is handed anywhere. Each function the
partition produces is called once from the entry point and can be run as a whole. The table holds
one entry per function other than the entry point, which is the last, so ``zip(resources, parts)``
pairs each function with the resource that handles it.

Stepping a program
==================

Once each resource's work is in a function of its own, a :class:`ProgramStepper` runs the program
against a caller who performs that work. Each step runs everything Qiskit can and hands over a
:class:`Request` per function whose operands have arrived, and :attr:`~ProgramStepper.outputs` is
not ``None`` once the program has finished::

    stepper = ProgramStepper(parts, range(len(resources)), {"x": numpy.array([1.0, 2.0, 3.0])})
    while (outputs := stepper.outputs) is None:
        stepper.step()
        for request in stepper.outstanding():
            stepper.fulfil(request.id, sample(request.function, request.inputs))

One step surfaces every ready request at once, so work that does not depend on other work is handed
over together. A stepper holds no callbacks and never gives up control, so a caller may fulfil a
request however it likes, including by submitting it and coming back to the stepper later.
:class:`~qiskit.providers.basic_provider.BasicProgramSimulator` drives one this way, answering each
request with a state-vector simulation.

Operations
==========

Each operation an instruction can perform has a class in :mod:`qiskit.quantum_program.ops`, which
both builds the operation and reads back the payload it was built with. The functions here are what
writes a program, and they apply those operations for you, so a class is what you reach for to read
an operation back::

    from qiskit.quantum_program import ops

    ops.Mean(0).full_name    # 'qiskit.mean'

Dtypes
======

A dtype is a member of :class:`DType`. Each has a lower-case alias, and indexing one
with a shape gives the :class:`TensorType` of a value::

    f64[3]                        # a vector of three 64-bit floats
    bit[4000, 2]                  # a 4000x2 array of bits
    f64[()]                       # a scalar
    bit[1024, bounded(64)]        # an axis whose size is known only at run time

The current aliases are ``bit``, ``u8``, ``u16``, ``u32``, ``u64``, ``i8``, ``i16``, ``i32``,
``i64``, ``f32``, ``f64``, ``c64`` and ``c128``.

Functions
=========

.. autosummary::
   :toctree: ../stubs/

   shot_loop
   bind_parameters
   qp_input
   constant
   build
   add
   subtract
   multiply
   divide
   remainder
   power
   bitwise_and
   bitwise_or
   bitwise_xor
   bitwise_not
   mean
   var
   std
   parity
   cast
   broadcast_to
   draw
   listing

Classes
=======

.. autosummary::
   :toctree: ../stubs/

   DataTree
   DType
   Instruction
   InstructionRole
   ProgramFunction
   ProgramStepper
   QuantumProgram
   Request
   TensorType
   Tracer
   Value
   bounded

Submodules
==========

.. autosummary::
   :toctree: ../stubs/

   ops
"""

from qiskit._accelerate.quantum_program import (
    DataTree,
    DType,
    Instruction,
    InstructionRole,
    ProgramFunction,
    ProgramStepper,
    QuantumProgram,
    Request,
    TensorType,
    Value,
    bounded,
)

from . import ops
from ._render import draw, listing
from ._tracer import (
    Tracer,
    add,
    bind_parameters,
    bitwise_and,
    bitwise_not,
    bitwise_or,
    bitwise_xor,
    broadcast_to,
    build,
    cast,
    constant,
    divide,
    mean,
    multiply,
    parity,
    power,
    qp_input,
    remainder,
    shot_loop,
    std,
    subtract,
    var,
)

bit = DType.Bit
u8 = DType.U8
u16 = DType.U16
u32 = DType.U32
u64 = DType.U64
i8 = DType.I8
i16 = DType.I16
i32 = DType.I32
i64 = DType.I64
f32 = DType.F32
f64 = DType.F64
c64 = DType.C64
c128 = DType.C128

__all__ = [
    "DType",
    "DataTree",
    "Instruction",
    "InstructionRole",
    "ProgramFunction",
    "ProgramStepper",
    "QuantumProgram",
    "Request",
    "TensorType",
    "Tracer",
    "Value",
    "add",
    "bind_parameters",
    "bit",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "bounded",
    "broadcast_to",
    "build",
    "c64",
    "c128",
    "cast",
    "constant",
    "divide",
    "draw",
    "f32",
    "f64",
    "i8",
    "i16",
    "i32",
    "i64",
    "listing",
    "mean",
    "multiply",
    "ops",
    "parity",
    "power",
    "qp_input",
    "remainder",
    "shot_loop",
    "std",
    "subtract",
    "u8",
    "u16",
    "u32",
    "u64",
    "var",
]
