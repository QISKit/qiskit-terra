# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Quantum circuit object."""

from __future__ import annotations

import warnings
import collections.abc
import copy as _copy

import itertools
import multiprocessing
import typing
from collections import OrderedDict
from typing import (
    TypeVar,
    Any,
    Literal,
    overload,
)
from collections.abc import Sequence, Callable, Mapping, Iterable
import os
from math import pi
import numpy as np
from qiskit._accelerate.circuit import CircuitData
from qiskit._accelerate.circuit import StandardGate
from qiskit._accelerate.circuit import BitLocations
from qiskit._accelerate.circuit_duration import compute_estimated_duration
from qiskit._accelerate.target import estimate_fidelity
from qiskit.exceptions import QiskitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils import deprecate_func, deprecate_arg
from . import (
    Bit,
    QuantumRegister,
    Qubit,
    AncillaRegister,
    AncillaQubit,
    Clbit,
    ClassicalRegister,
    Register,
)
from . import _classical_resource_map
from .controlflow import ControlFlowOp, _builder_utils
from .controlflow.builder import CircuitScopeInterface, ControlFlowBuilderBlock
from .controlflow.break_loop import BreakLoopOp, BreakLoopPlaceholder
from .controlflow.box import BoxOp, BoxContext
from .controlflow.continue_loop import ContinueLoopOp, ContinueLoopPlaceholder
from .controlflow.for_loop import ForLoopOp, ForLoopContext
from .controlflow.if_else import IfElseOp, IfContext
from .controlflow.switch_case import SwitchCaseOp, SwitchContext
from .controlflow.while_loop import WhileLoopOp, WhileLoopContext
from .classical import expr, types
from .parameterexpression import ParameterExpression, ParameterValueType
from .parametertable import ParameterView
from .parametervector import ParameterVector
from .instructionset import InstructionSet
from .operation import Operation
from .quantumcircuitdata import QuantumCircuitData, CircuitInstruction
from .delay import Delay
from .store import Store


if typing.TYPE_CHECKING:
    import types as builtin_types
    import qiskit
    from qiskit.circuit import Annotation
    from qiskit.transpiler.layout import TranspileLayout
    from qiskit.quantum_info.operators.base_operator import BaseOperator
    from qiskit.quantum_info.states.statevector import Statevector

# The following types are not marked private to avoid leaking this "private/public" abstraction out
# into the documentation.  They are not imported by circuit.__init__, nor are they meant to be.

# Arbitrary type variables for marking up generics.
S = TypeVar("S")
T = TypeVar("T")

# Types that can be coerced to a valid Qubit specifier in a circuit.
QubitSpecifier = Qubit | QuantumRegister | int | slice | Sequence[Qubit | int]

# Types that can be coerced to a valid Clbit specifier in a circuit.
ClbitSpecifier = Clbit | ClassicalRegister | int | slice | Sequence[Clbit | int]

# Generic type which is either :obj:`~Qubit` or :obj:`~Clbit`, used to specify types of functions
# which operate on either type of bit, but not both at the same time.
BitType = TypeVar("BitType", Qubit, Clbit)


# NOTE:
#
# If you're adding methods or attributes to `QuantumCircuit`, be sure to update the class docstring
# to document them in a suitable place.  The class is huge, so we do its documentation manually so
# it has at least some amount of organizational structure.


class QuantumCircuit:
    """Core Qiskit representation of a quantum circuit.

    .. note::
        For more details setting the :class:`QuantumCircuit` in context of all of the data
        structures that go with it, how it fits into the rest of the :mod:`qiskit` package, and the
        different regimes of quantum-circuit descriptions in Qiskit, see the module-level
        documentation of :mod:`qiskit.circuit`.

    Example:

    .. plot::
       :include-source:
       :nofigs:

       from qiskit import QuantumCircuit

       # Create a new circuit with two qubits
       qc = QuantumCircuit(2)

       # Add a Hadamard gate to qubit 0
       qc.h(0)

       # Perform a controlled-X gate on qubit 1, controlled by qubit 0
       qc.cx(0, 1)

       # Return a text drawing of the circuit.
       qc.draw()

    .. code-block:: text

            ┌───┐
       q_0: ┤ H ├──■──
            └───┘┌─┴─┐
       q_1: ─────┤ X ├
                 └───┘

    Circuit attributes
    ==================

    :class:`QuantumCircuit` has a small number of public attributes, which are mostly older
    functionality.  Most of its functionality is accessed through methods.

    A small handful of the attributes are intentionally mutable, the rest are data attributes that
    should be considered immutable.

    ============================== =================================================================
    Mutable attribute              Summary
    ============================== =================================================================
    :attr:`global_phase`           The global phase of the circuit, measured in radians.
    :attr:`metadata`               Arbitrary user mapping, which Qiskit will preserve through the
                                   transpiler, but otherwise completely ignore.
    :attr:`name`                   An optional string name for the circuit.
    ============================== =================================================================

    ============================== =================================================================
    Immutable data attribute       Summary
    ============================== =================================================================
    :attr:`ancillas`               List of :class:`AncillaQubit`\\ s tracked by the circuit.
    :attr:`cregs`                  List of :class:`ClassicalRegister`\\ s tracked by the circuit.

    :attr:`clbits`                 List of :class:`Clbit`\\ s tracked by the circuit.
    :attr:`data`                   List of individual :class:`CircuitInstruction`\\ s that make up
                                   the circuit.
    :attr:`_data`                  Python-space handle to the C API :c:struct:`QkCircuit` object.
    :attr:`duration`               Total duration of the circuit, added by scheduling transpiler
                                   passes.
                                   This attribute is deprecated and :meth:`.estimate_duration`
                                   should be used instead.

    :attr:`layout`                 Hardware layout and routing information added by the transpiler.
    :attr:`num_ancillas`           The number of ancilla qubits in the circuit.
    :attr:`num_clbits`             The number of clbits in the circuit.
    :attr:`num_captured_vars`      Number of captured real-time classical variables.
    :attr:`num_captured_stretches` Number of captured stretches.
    :attr:`num_declared_vars`      Number of locally declared real-time classical variables in the
                                   outer circuit scope.
    :attr:`num_declared_stretches` Number of locally declared stretches in the outer circuit scope.
    :attr:`num_input_vars`         Number of input real-time classical variables.
    :attr:`num_parameters`         Number of compile-time :class:`Parameter`\\ s in the circuit.
    :attr:`num_qubits`             Number of qubits in the circuit.

    :attr:`num_vars`               Total number of real-time classical variables in the outer
                                   circuit scope.
    :attr:`num_stretches`          Total number of stretches in the outer circuit scope.
    :attr:`num_identifiers`        Total number of both variables and stretches in the outer
                                   circuit.
    :attr:`op_start_times`         Start times of scheduled operations, added by scheduling
                                   transpiler passes.
    :attr:`parameters`             Ordered set-like view of the compile-time :class:`Parameter`\\ s
                                   tracked by the circuit.
    :attr:`qregs`                  List of :class:`QuantumRegister`\\ s tracked by the circuit.

    :attr:`qubits`                 List of :class:`Qubit`\\ s tracked by the circuit.
    :attr:`unit`                   The unit of the :attr:`duration` field.
    ============================== =================================================================

    The core attribute is :attr:`data`.  This is a sequence-like object that exposes the
    :class:`CircuitInstruction`\\ s contained in an ordered form.  You generally should not mutate
    this object directly; :class:`QuantumCircuit` is only designed for append-only operations (which
    should use :meth:`append`).  Most operations that mutate circuits in place should be written as
    transpiler passes (:mod:`qiskit.transpiler`).  The C API interacts with an internal object,
    called :attr:`_data`, which is not part of the public Python API, other than as a handle to pass
    to C-API calls.

    .. autoattribute:: data

    .. py::attribute:: _data
        An opaque handle to the C API object ``QkCircuit``.

        .. warning::
            No part of this object other than its existence is part of the public API.

        The only valid use of this object from within the public Python API is as part of the
        extraction of a :c:struct:`QkCircuit` using :c:func:`qk_circuit_borrow_from_python` or
        similar methods.  The Python-space type of the object is not specified in the public API,
        and none of its methods, regardless of name, should be considered public.

    Alongside the :attr:`data`, the :attr:`global_phase` of a circuit can have some impact on its
    output, if the circuit is used to describe a :class:`.Gate` that may be controlled.  This is
    measured in radians and is directly settable.

    .. autoattribute:: global_phase

    The :attr:`name` of a circuit becomes the name of the :class:`~.circuit.Instruction` or
    :class:`.Gate` resulting from :meth:`to_instruction` and :meth:`to_gate` calls, which can be
    handy for visualizations.

    .. autoattribute:: name

    You can attach arbitrary :attr:`metadata` to a circuit.  No part of core Qiskit will inspect
    this or change its behavior based on metadata, but it will be faithfully passed through the
    transpiler, so you can tag your circuits yourself.  When serializing a circuit with QPY (see
    :mod:`qiskit.qpy`), the metadata will be JSON-serialized and you may need to pass a custom
    serializer to handle non-JSON-compatible objects within it (see :func:`.qpy.dump` for more
    detail).  This field is ignored during export to OpenQASM 2 or 3.

    .. autoattribute:: metadata

    :class:`QuantumCircuit` exposes data attributes tracking its internal quantum and classical bits
    and registers.  These appear as Python :class:`list`\\ s, but you should treat them as
    immutable; changing them will *at best* have no effect, and more likely will simply corrupt
    the internal data of the :class:`QuantumCircuit`.

    .. autoattribute:: qregs
    .. autoattribute:: cregs
    .. autoattribute:: qubits
    .. autoattribute:: ancillas
    .. autoattribute:: clbits

    The :ref:`compile-time parameters <circuit-compile-time-parameters>` present in instructions on
    the circuit are available in :attr:`parameters`.  This has a canonical order (mostly lexical,
    except in the case of :class:`.ParameterVector`), which matches the order that parameters will
    be assigned when using the list forms of :meth:`assign_parameters`, but also supports
    :class:`set`-like constant-time membership testing.

    .. autoattribute:: parameters

    If you have transpiled your circuit, so you have a physical circuit, you can inspect the
    :attr:`layout` attribute for information stored by the transpiler about how the virtual qubits
    of the source circuit map to the hardware qubits of your physical circuit, both at the start and
    end of the circuit.

    .. autoattribute:: layout

    If your circuit was also *scheduled* as part of a transpilation, it will expose the individual
    timings of each instruction, along with the total :attr:`duration` of the circuit.

    .. autoattribute:: duration
    .. autoattribute:: unit
    .. autoattribute:: op_start_times

    Finally, :class:`QuantumCircuit` exposes several simple properties as dynamic read-only numeric
    attributes.

    .. autoattribute:: num_ancillas
    .. autoattribute:: num_clbits
    .. autoattribute:: num_captured_vars
    .. autoattribute:: num_captured_stretches
    .. autoattribute:: num_declared_vars
    .. autoattribute:: num_declared_stretches
    .. autoattribute:: num_input_vars
    .. autoattribute:: num_identifiers
    .. autoattribute:: num_parameters
    .. autoattribute:: num_qubits
    .. autoattribute:: num_stretches
    .. autoattribute:: num_vars

    Creating new circuits
    =====================

    =========================  =====================================================================
    Method                     Summary
    =========================  =====================================================================
    :meth:`__init__`           Default constructor of no-instruction circuits.
    :meth:`copy`               Make a complete copy of an existing circuit.
    :meth:`copy_empty_like`    Copy data objects from one circuit into a new one without any
                               instructions.
    :meth:`from_instructions`  Infer data objects needed from a list of instructions.
    :meth:`from_qasm_file`     Legacy interface to :func:`.qasm2.load`.
    :meth:`from_qasm_str`      Legacy interface to :func:`.qasm2.loads`.
    =========================  =====================================================================

    The default constructor (``QuantumCircuit(...)``) produces a circuit with no initial
    instructions. The arguments to the default constructor can be used to seed the circuit with
    quantum and classical data storage, and to provide a name, global phase and arbitrary metadata.
    All of these fields can be expanded later.

    .. automethod:: __init__

    If you have an existing circuit, you can produce a copy of it using :meth:`copy`, including all
    its instructions.  This is useful if you want to keep partial circuits while extending another,
    or to have a version you can mutate in-place while leaving the prior one intact.

    .. automethod:: copy

    Similarly, if you want a circuit that contains all the same data objects (bits, registers,
    variables, etc) but with none of the instructions, you can use :meth:`copy_empty_like`.  This is
    quite common when you want to build up a new layer of a circuit to then apply onto the back
    with :meth:`compose`, or to do a full rewrite of a circuit's instructions.

    .. automethod:: copy_empty_like

    In some cases, it is most convenient to generate a list of :class:`.CircuitInstruction`\\ s
    separately to an entire circuit context, and then to build a circuit from this.  The
    :meth:`from_instructions` constructor will automatically capture all :class:`.Qubit` and
    :class:`.Clbit` instances used in the instructions, and create a new :class:`QuantumCircuit`
    object that has the correct resources and all the instructions.

    .. automethod:: from_instructions

    :class:`QuantumCircuit` also still has two constructor methods that are legacy wrappers around
    the importers in :mod:`qiskit.qasm2`.  These automatically apply :ref:`the legacy compatibility
    settings <qasm2-legacy-compatibility>` of :func:`~.qasm2.load` and :func:`~.qasm2.loads`.

    .. automethod:: from_qasm_file
    .. automethod:: from_qasm_str

    Data objects on circuits
    ========================

    .. _circuit-adding-data-objects:

    Adding data objects
    -------------------

    =============================  =================================================================
    Method                         Adds this kind of data
    =============================  =================================================================
    :meth:`add_bits`               :class:`.Qubit`\\ s and :class:`.Clbit`\\ s.
    :meth:`add_register`           :class:`.QuantumRegister` and :class:`.ClassicalRegister`.
    :meth:`add_var`                :class:`~.expr.Var` nodes with local scope and initializers.
    :meth:`add_stretch`            :class:`~.expr.Stretch` nodes with local scope.
    :meth:`add_input`              :class:`~.expr.Var` nodes that are treated as circuit inputs.
    :meth:`add_capture`            :class:`~.expr.Var` or :class:`~.expr.Stretch` nodes captured
                                   from containing scopes.
    :meth:`add_uninitialized_var`  :class:`~.expr.Var` nodes with local scope and undefined state.
    =============================  =================================================================

    Typically you add most of the data objects (:class:`.Qubit`, :class:`.Clbit`,
    :class:`.ClassicalRegister`, etc) to the circuit as part of using the :meth:`__init__` default
    constructor, or :meth:`copy_empty_like`.  However, it is also possible to add these afterwards.
    Typed classical data, such as standalone :class:`~.expr.Var` nodes (see
    :ref:`circuit-repr-real-time-classical`), can be both constructed and added with separate
    methods.

    New registerless :class:`.Qubit` and :class:`.Clbit` objects are added using :meth:`add_bits`.
    These objects must not already be present in the circuit.  You can check if a bit exists in the
    circuit already using :meth:`find_bit`.

    .. automethod:: add_bits

    Registers are added to the circuit with :meth:`add_register`.  In this method, it is not an
    error if some of the bits are already present in the circuit.  In this case, the register will
    be an "alias" over the bits.  This is not generally well-supported by hardware backends; it is
    probably best to stay away from relying on it.  The registers a given bit is in are part of the
    return of :meth:`find_bit`.

    .. automethod:: add_register

    :ref:`Real-time, typed classical data <circuit-repr-real-time-classical>` is represented on the
    circuit by :class:`~.expr.Var` nodes with a well-defined :class:`~.types.Type`.  It is possible
    to instantiate these separately to a circuit (see :meth:`.Var.new`), but it is often more
    convenient to use circuit methods that will automatically manage the types and expression
    initialization for you.  The two most common methods are :meth:`add_var` (locally scoped
    variables) and :meth:`add_input` (inputs to the circuit). Additionally, the method
    :meth:`add_stretch` can be used to add stretches to the circuit.

    .. automethod:: add_var
    .. automethod:: add_input
    .. automethod:: add_stretch

    In addition, there are two lower-level methods that can be useful for programmatic generation of
    circuits.  When working interactively, you will most likely not need these; most uses of
    :meth:`add_uninitialized_var` are part of :meth:`copy_empty_like`, and most uses of
    :meth:`add_capture` would be better off using :ref:`the control-flow builder interface
    <circuit-control-flow-methods>`.

    .. automethod:: add_uninitialized_var
    .. automethod:: add_capture

    Working with bits and registers
    -------------------------------

    A :class:`.Bit` instance is, on its own, just a unique handle for circuits to use in their own
    contexts.  If you have got a :class:`.Bit` instance and a circuit, you can find the contexts
    that the bit exists in using :meth:`find_bit`, such as its integer index in the circuit and any
    registers it is contained in.

    .. automethod:: find_bit

    Similarly, you can query a circuit to see if a register has already been added to it by using
    :meth:`has_register`.

    .. automethod:: has_register

    Working with compile-time parameters
    ------------------------------------

    .. seealso::
        :ref:`circuit-compile-time-parameters`
            A more complete discussion of what compile-time parametrization is, and how it fits into
            Qiskit's data model.

    Unlike bits, registers, and real-time typed classical data, compile-time symbolic parameters are
    not manually added to a circuit.  Their presence is inferred by being contained in operations
    added to circuits and the global phase.  An ordered list of all parameters currently in a
    circuit is at :attr:`QuantumCircuit.parameters`.

    The most common operation on :class:`.Parameter` instances is to replace them in symbolic
    operations with some numeric value, or another symbolic expression.  This is done with
    :meth:`assign_parameters`.

    .. automethod:: assign_parameters

    The circuit tracks parameters by :class:`.Parameter` instances themselves, and forbids having
    multiple parameters of the same name to avoid some problems when interoperating with OpenQASM or
    other external formats.  You can use :meth:`has_parameter` and :meth:`get_parameter` to query
    the circuit for a parameter with the given string name.

    .. automethod:: has_parameter
    .. automethod:: get_parameter

    .. _circuit-real-time-methods:

    Working with real-time typed classical data
    -------------------------------------------

    .. seealso::
        :mod:`qiskit.circuit.classical`
            Module-level documentation for how the variable-, expression- and type-systems work, the
            objects used to represent them, and the classical operations available.

        :ref:`circuit-repr-real-time-classical`
            A discussion of how real-time data fits into the entire :mod:`qiskit.circuit` data model
            as a whole.

        :ref:`circuit-adding-data-objects`
            The methods for adding new :class:`~.expr.Var` or :class:`~.expr.Stretch` identifiers
            to a circuit after initialization.

    You can retrieve identifiers attached to a circuit (e.g. a :class:`~.expr.Var` or
    :class:`~.expr.Stretch`) by name with methods :meth:`get_var`, :meth:`get_stretch`, or
    :meth:`get_identifier`. You can also check if a circuit
    contains a given identifier with :meth:`has_var`, :meth:`has_stretch`, or
    :meth:`has_identifier`.

    .. automethod:: get_var
    .. automethod:: get_stretch
    .. automethod:: get_identifier
    .. automethod:: has_var
    .. automethod:: has_stretch
    .. automethod:: has_identifier


    There are also several iterator methods that you can use to get the full set of identifiers
    tracked by a circuit.  At least one of :meth:`iter_input_vars` and :meth:`iter_captured_vars`
    will be empty, as inputs and captures are mutually exclusive.  All of the iterators have
    corresponding dynamic properties on :class:`QuantumCircuit` that contain their length:
    :attr:`num_vars`, :attr:`num_stretches`, :attr:`num_input_vars`, :attr:`num_captured_vars`,
    :attr:`num_captured_stretches`, :attr:`num_declared_vars`, or :attr:`num_declared_stretches`.

    .. automethod:: iter_vars
    .. automethod:: iter_stretches
    .. automethod:: iter_input_vars
    .. automethod:: iter_captured_vars
    .. automethod:: iter_captured_stretches
    .. automethod:: iter_declared_vars
    .. automethod:: iter_declared_stretches


    .. _circuit-adding-operations:

    Adding operations to circuits
    =============================

    You can add anything that implements the :class:`.Operation` interface to a circuit as a single
    instruction, though most things you will want to add will be :class:`~.circuit.Instruction` or
    :class:`~.circuit.Gate` instances.

    .. seealso::
        :ref:`circuit-operations-instructions`
            The :mod:`qiskit.circuit`-level documentation on the different interfaces that Qiskit
            uses to define circuit-level instructions.

    .. _circuit-append-compose:

    Methods to add general operations
    ---------------------------------

    These are the base methods that handle adding any object, including user-defined ones, onto
    circuits.

    ===============  ===============================================================================
    Method           When to use it
    ===============  ===============================================================================
    :meth:`append`   Add an instruction as a single object onto a circuit.
    :meth:`_append`  Same as :meth:`append`, but a low-level interface that elides almost all error
                     checking.
    :meth:`compose`  Inline the instructions from one circuit onto another.
    :meth:`tensor`   Like :meth:`compose`, but strictly for joining circuits that act on disjoint
                     qubits.
    ===============  ===============================================================================

    :class:`QuantumCircuit` has two main ways that you will add more operations onto a circuit.
    Which to use depends on whether you want to add your object as a single "instruction"
    (:meth:`append`), or whether you want to join the instructions from two circuits together
    (:meth:`compose`).

    A single instruction or operation appears as a single entry in the :attr:`data` of the circuit,
    and as a single box when drawn in the circuit visualizers (see :meth:`draw`).  A single
    instruction is the "unit" that a hardware backend might be defined in terms of (see
    :class:`.Target`).  An :class:`~.circuit.Instruction` can come with a
    :attr:`~.circuit.Instruction.definition`, which is one rule the transpiler (see
    :mod:`qiskit.transpiler`) will be able to fall back on to decompose it for hardware, if needed.
    An :class:`.Operation` that is not also an :class:`~.circuit.Instruction` can
    only be decomposed if it has some associated high-level synthesis method registered for it (see
    :mod:`qiskit.transpiler.passes.synthesis.plugin`).

    A :class:`QuantumCircuit` alone is not a single :class:`~.circuit.Instruction`; it is rather
    more complicated, since it can, in general, represent a complete program with typed classical
    memory inputs and outputs, and control flow.  Qiskit's (and most hardware's) data model does not
    yet have the concept of re-usable callable subroutines with virtual quantum operands.  You can
    convert simple circuits that act only on qubits with unitary operations into a :class:`.Gate`
    using :meth:`to_gate`, and simple circuits acting only on qubits and clbits into a
    :class:`~.circuit.Instruction` with :meth:`to_instruction`.

    When you have an :class:`.Operation`, :class:`~.circuit.Instruction`, or :class:`.Gate`, add it
    to the circuit, specifying the qubit and clbit arguments with :meth:`append`.

    .. automethod:: append

    :meth:`append` does quite substantial error checking to ensure that you cannot accidentally
    break the data model of :class:`QuantumCircuit`.  If you are programmatically generating a
    circuit from known-good data, you can elide much of this error checking by using the fast-path
    appender :meth:`_append`, but at the risk that the caller is responsible for ensuring they are
    passing only valid data.

    .. automethod:: _append

    In other cases, you may want to join two circuits together, applying the instructions from one
    circuit onto specified qubits and clbits on another circuit.  This "inlining" operation is
    called :meth:`compose` in Qiskit.  :meth:`compose` is, in general, more powerful than
    a :meth:`to_instruction`-plus-:meth:`append` combination for joining two circuits, because it
    can also link typed classical data together, and allows for circuit control-flow operations to
    be joined onto another circuit.

    The downsides to :meth:`compose` are that it is a more complex operation that can involve more
    rewriting of the operand, and that it necessarily must move data from one circuit object to
    another.  If you are building up a circuit for yourself and raw performance is a core goal,
    consider passing around your base circuit and having different parts of your algorithm write
    directly to the base circuit, rather than building a temporary layer circuit.

    .. automethod:: compose

    If you are trying to join two circuits that will apply to completely disjoint qubits and clbits,
    :meth:`tensor` is a convenient wrapper around manually adding bit objects and calling
    :meth:`compose`.

    .. automethod:: tensor

    As some rules of thumb:

    * If you have a single :class:`.Operation`, :class:`~.circuit.Instruction` or :class:`.Gate`,
      you should definitely use :meth:`append` or :meth:`_append`.
    * If you have a :class:`QuantumCircuit` that represents a single atomic instruction for a larger
      circuit that you want to re-use, you probably want to call :meth:`to_instruction` or
      :meth:`to_gate`, and then apply the result of that to the circuit using :meth:`append`.
    * If you have a :class:`QuantumCircuit` that represents a larger "layer" of another circuit, or
      contains typed classical variables or control flow, you should use :meth:`compose` to merge it
      onto another circuit.
    * :meth:`tensor` is wanted far more rarely than either :meth:`append` or :meth:`compose`.
      Internally, it is mostly a wrapper around :meth:`add_bits` and :meth:`compose`.

    Some potential pitfalls to beware of:

    * Even if you re-use a custom :class:`~.circuit.Instruction` during circuit construction, the
      transpiler will generally have to "unroll" each invocation of it to its inner decomposition
      before beginning work on it.  This should not prevent you from using the
      :meth:`to_instruction`-plus-:meth:`append` pattern, as the transpiler will improve in this
      regard over time.
    * :meth:`compose` will, by default, produce a new circuit for backwards compatibility.  This is
      more expensive, and not usually what you want, so you should set ``inplace=True``.
    * Both :meth:`append` and :meth:`compose` (but not :meth:`_append`) have a ``copy`` keyword
      argument that defaults to ``True``.  In these cases, the incoming :class:`.Operation`
      instances will be copied if Qiskit detects that the objects have mutability about them (such
      as taking gate parameters).  If you are sure that you will not re-use the objects again in
      other places, you should set ``copy=False`` to prevent this copying, which can be a
      substantial speed-up for large objects.

    Methods to add standard instructions
    ------------------------------------

    The :class:`QuantumCircuit` class has helper methods to add many of the Qiskit standard-library
    instructions and gates onto a circuit.  These are generally equivalent to manually constructing
    an instance of the relevant :mod:`qiskit.circuit.library` object, then passing that to
    :meth:`append` with the remaining arguments placed into the ``qargs`` and ``cargs`` fields as
    appropriate.

    The following methods apply special non-unitary :class:`~.circuit.Instruction` operations to the
    circuit:

    ===============================   ====================================================
    :class:`QuantumCircuit` method    :mod:`qiskit.circuit` :class:`~.circuit.Instruction`
    ===============================   ====================================================
    :meth:`barrier`                   :class:`Barrier`
    :meth:`delay`                     :class:`Delay`
    :meth:`initialize`                :class:`~library.Initialize`
    :meth:`measure`                   :class:`Measure`
    :meth:`reset`                     :class:`Reset`
    :meth:`store`                     :class:`Store`
    ===============================   ====================================================

    These methods apply uncontrolled unitary :class:`.Gate` instances to the circuit:

    ===============================   ============================================
    :class:`QuantumCircuit` method    :mod:`qiskit.circuit.library` :class:`.Gate`
    ===============================   ============================================
    :meth:`dcx`                       :class:`~library.DCXGate`
    :meth:`ecr`                       :class:`~library.ECRGate`
    :meth:`h`                         :class:`~library.HGate`
    :meth:`id`                        :class:`~library.IGate`
    :meth:`iswap`                     :class:`~library.iSwapGate`
    :meth:`ms`                        :class:`~library.MSGate`
    :meth:`p`                         :class:`~library.PhaseGate`
    :meth:`pauli`                     :class:`~library.PauliGate`
    :meth:`prepare_state`             :class:`~library.StatePreparation`
    :meth:`r`                         :class:`~library.RGate`
    :meth:`rcccx`                     :class:`~library.RC3XGate`
    :meth:`rccx`                      :class:`~library.RCCXGate`
    :meth:`rv`                        :class:`~library.RVGate`
    :meth:`rx`                        :class:`~library.RXGate`
    :meth:`rxx`                       :class:`~library.RXXGate`
    :meth:`ry`                        :class:`~library.RYGate`
    :meth:`ryy`                       :class:`~library.RYYGate`
    :meth:`rz`                        :class:`~library.RZGate`
    :meth:`rzx`                       :class:`~library.RZXGate`
    :meth:`rzz`                       :class:`~library.RZZGate`
    :meth:`s`                         :class:`~library.SGate`
    :meth:`sdg`                       :class:`~library.SdgGate`
    :meth:`swap`                      :class:`~library.SwapGate`
    :meth:`sx`                        :class:`~library.SXGate`
    :meth:`sxdg`                      :class:`~library.SXdgGate`
    :meth:`t`                         :class:`~library.TGate`
    :meth:`tdg`                       :class:`~library.TdgGate`
    :meth:`u`                         :class:`~library.UGate`
    :meth:`unitary`                   :class:`~library.UnitaryGate`
    :meth:`x`                         :class:`~library.XGate`
    :meth:`y`                         :class:`~library.YGate`
    :meth:`z`                         :class:`~library.ZGate`
    ===============================   ============================================

    The following methods apply :class:`Gate` instances that are also controlled gates, so are
    direct subclasses of :class:`ControlledGate`:

    ===============================   ======================================================
    :class:`QuantumCircuit` method    :mod:`qiskit.circuit.library` :class:`.ControlledGate`
    ===============================   ======================================================
    :meth:`ccx`                       :class:`~library.CCXGate`
    :meth:`ccz`                       :class:`~library.CCZGate`
    :meth:`ch`                        :class:`~library.CHGate`
    :meth:`cp`                        :class:`~library.CPhaseGate`
    :meth:`crx`                       :class:`~library.CRXGate`
    :meth:`cry`                       :class:`~library.CRYGate`
    :meth:`crz`                       :class:`~library.CRZGate`
    :meth:`cs`                        :class:`~library.CSGate`
    :meth:`csdg`                      :class:`~library.CSdgGate`
    :meth:`cswap`                     :class:`~library.CSwapGate`
    :meth:`csx`                       :class:`~library.CSXGate`
    :meth:`cu`                        :class:`~library.CUGate`
    :meth:`cx`                        :class:`~library.CXGate`
    :meth:`cy`                        :class:`~library.CYGate`
    :meth:`cz`                        :class:`~library.CZGate`
    ===============================   ======================================================

    Finally, these methods apply particular generalized multiply controlled gates to the circuit,
    often with eager syntheses.  They are listed in terms of the *base* gate they are controlling,
    since their exact output is often a synthesized version of a gate.

    ===============================   =================================================
    :class:`QuantumCircuit` method    Base :mod:`qiskit.circuit.library` :class:`.Gate`
    ===============================   =================================================
    :meth:`mcp`                       :class:`~library.PhaseGate`
    :meth:`mcrx`                      :class:`~library.RXGate`
    :meth:`mcry`                      :class:`~library.RYGate`
    :meth:`mcrz`                      :class:`~library.RZGate`
    :meth:`mcx`                       :class:`~library.XGate`
    ===============================   =================================================

    The rest of this section is the API listing of all the individual methods; the tables above are
    summaries whose links will jump you to the correct place.

    .. automethod:: barrier
    .. automethod:: ccx
    .. automethod:: ccz
    .. automethod:: ch
    .. automethod:: cp
    .. automethod:: crx
    .. automethod:: cry
    .. automethod:: crz
    .. automethod:: cs
    .. automethod:: csdg
    .. automethod:: cswap
    .. automethod:: csx
    .. automethod:: cu
    .. automethod:: cx
    .. automethod:: cy
    .. automethod:: cz
    .. automethod:: dcx
    .. automethod:: delay
    .. automethod:: ecr
    .. automethod:: h
    .. automethod:: id
    .. automethod:: initialize
    .. automethod:: iswap
    .. automethod:: mcp
    .. automethod:: mcrx
    .. automethod:: mcry
    .. automethod:: mcrz
    .. automethod:: mcx
    .. automethod:: measure
    .. automethod:: ms
    .. automethod:: p
    .. automethod:: pauli
    .. automethod:: prepare_state
    .. automethod:: r
    .. automethod:: rcccx
    .. automethod:: rccx
    .. automethod:: reset
    .. automethod:: rv
    .. automethod:: rx
    .. automethod:: rxx
    .. automethod:: ry
    .. automethod:: ryy
    .. automethod:: rz
    .. automethod:: rzx
    .. automethod:: rzz
    .. automethod:: s
    .. automethod:: sdg
    .. automethod:: store
    .. automethod:: swap
    .. automethod:: sx
    .. automethod:: sxdg
    .. automethod:: t
    .. automethod:: tdg
    .. automethod:: u
    .. automethod:: unitary
    .. automethod:: x
    .. automethod:: y
    .. automethod:: z


    .. _circuit-control-flow-methods:

    Adding control flow to circuits
    -------------------------------

    .. seealso::
        :ref:`circuit-control-flow-repr`

        Discussion of how control-flow operations are represented in the whole :mod:`qiskit.circuit`
        context.

    ==============================  ================================================================
    :class:`QuantumCircuit` method  Control-flow instruction
    ==============================  ================================================================
    :meth:`if_test`                 :class:`.IfElseOp` with only a ``True`` body
    :meth:`if_else`                 :class:`.IfElseOp` with both ``True`` and ``False`` bodies
    :meth:`while_loop`              :class:`.WhileLoopOp`
    :meth:`switch`                  :class:`.SwitchCaseOp`
    :meth:`for_loop`                :class:`.ForLoopOp`
    :meth:`box`                     :class:`.BoxOp`
    :meth:`break_loop`              :class:`.BreakLoopOp`
    :meth:`continue_loop`           :class:`.ContinueLoopOp`
    ==============================  ================================================================

    :class:`QuantumCircuit` has corresponding methods for all of the control-flow operations that
    are supported by Qiskit.  These have two forms for calling them.  The first is a very
    straightforward convenience wrapper that takes in the block bodies of the instructions as
    :class:`QuantumCircuit` arguments, and simply constructs and appends the corresponding
    :class:`.ControlFlowOp`.

    The second form, which we strongly recommend you use for constructing control flow, is called
    *the builder interface*.  Here, the methods take only the real-time discriminant of the
    operation, and return `context managers
    <https://docs.python.org/3/library/stdtypes.html#typecontextmanager>`__ that you enter using
    ``with``.  You can then use regular :class:`QuantumCircuit` methods within those blocks to build
    up the control-flow bodies, and Qiskit will automatically track which of the data resources are
    needed for the inner blocks, building the complete :class:`.ControlFlowOp` as you leave the
    ``with`` statement.  It is far simpler and less error-prone to build control flow
    programmatically this way.

    When using the control-flow builder interface, you may sometimes want a qubit to be included in
    a block, even though it has no operations defined.  In this case, you can use the :meth:`noop`
    method.

    To check whether a circuit contains a :class:`.ControlFlowOp` you can use the helper method
    :meth:`.QuantumCircuit.has_control_flow_op`.

    ..
        TODO: expand the examples of the builder interface.

    .. automethod:: box
    .. automethod:: break_loop
    .. automethod:: continue_loop
    .. automethod:: for_loop
    .. automethod:: if_else
    .. automethod:: if_test
    .. automethod:: switch
    .. automethod:: while_loop
    .. automethod:: noop
    .. automethod:: has_control_flow_op


    Converting circuits to other objects
    -------------------------------------

    As discussed in :ref:`circuit-append-compose`, you can convert a circuit to either an
    :class:`~.circuit.Instruction` or a :class:`.Gate` using two helper methods.

    .. automethod:: to_instruction
    .. automethod:: to_gate

    In addition, you can convert the entire circuit into the :class:`.DAGCircuit` representation:

    .. automethod:: to_dag


    Helper mutation methods
    -----------------------

    There are two higher-level methods on :class:`QuantumCircuit` for appending measurements to the
    end of a circuit.  Note that by default, these also add an extra register.

    .. automethod:: measure_active
    .. automethod:: measure_all

    There are two "subtractive" methods on :class:`QuantumCircuit` as well.  This is not a use-case
    that :class:`QuantumCircuit` is designed for; typically you should just look to use
    :meth:`copy_empty_like` in place of :meth:`clear`, and run :meth:`remove_final_measurements` as
    its transpiler-pass form :class:`.RemoveFinalMeasurements`.

    .. automethod:: clear
    .. automethod:: remove_final_measurements


    Circuit properties
    ==================

    Simple circuit metrics
    ----------------------

    When constructing quantum circuits, there are several properties that help quantify
    the "size" of the circuits, and their ability to be run on a noisy quantum device.
    Some of these, like number of qubits, are straightforward to understand, while others
    like depth and number of tensor components require a bit more explanation.  Here we will
    explain all of these properties, and, in preparation for understanding how circuits change
    when run on actual devices, highlight the conditions under which they change.

    Consider the following circuit:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit import QuantumCircuit
       qc = QuantumCircuit(12)
       for idx in range(5):
          qc.h(idx)
          qc.cx(idx, idx+5)

       qc.cx(1, 7)
       qc.x(8)
       qc.cx(1, 9)
       qc.x(7)
       qc.cx(1, 11)
       qc.swap(6, 11)
       qc.swap(6, 9)
       qc.swap(6, 10)
       qc.x(6)
       qc.draw('mpl')

    From the plot, it is easy to see that this circuit has 12 qubits, and a collection of
    Hadamard, CNOT, X, and SWAP gates.  But how to quantify this programmatically? Because we
    can do single-qubit gates on all the qubits simultaneously, the number of qubits in this
    circuit is equal to the :meth:`width` of the circuit::

       assert qc.width() == 12

    We can also just get the number of qubits directly using :attr:`num_qubits`::

       assert qc.num_qubits == 12

    .. important::

       For a quantum circuit composed from just qubits, the circuit width is equal
       to the number of qubits. This is the definition used in quantum computing. However,
       for more complicated circuits with classical registers, and classically controlled gates,
       this equivalence breaks down. As such, from now on we will not refer to the number of
       qubits in a quantum circuit as the width.

    It is also straightforward to get the number and type of the gates in a circuit using
    :meth:`count_ops`::

       qc.count_ops()

    .. code-block:: text

       OrderedDict([('cx', 8), ('h', 5), ('x', 3), ('swap', 3)])

    We can also get just the raw count of operations by computing the circuits
    :meth:`size`::

       assert qc.size() == 19

    .. automethod:: count_ops
    .. automethod:: depth
    .. automethod:: get_instructions
    .. automethod:: num_connected_components
    .. automethod:: num_nonlocal_gates
    .. automethod:: num_tensor_factors
    .. automethod:: num_unitary_factors
    .. automethod:: size
    .. automethod:: width

    Accessing scheduling information
    --------------------------------

    If a :class:`QuantumCircuit` has been scheduled as part of a transpilation pipeline, the timing
    information for individual qubits can be accessed.  The whole-circuit timing information is
    available through the :meth:`estimate_duration` method and :attr:`op_start_times` attribute.

    .. automethod:: estimate_duration
    .. automethod:: qubit_duration
    .. automethod:: qubit_start_time
    .. automethod:: qubit_stop_time


    .. _circuit-abstract-physical:

    Abstract and physical circuits
    ==============================

    Circuits are a fairly low-level abstraction of quantum algorithms.  However, even within this,
    there are distinctions. Quantum programmers often want to use a wide array of gates and
    instructions, and work in a regime where all qubits and interact with all others.  Quantum
    hardware, however, typically has a restrictive set of native gates, and only certain pairs of
    hardware qubits can interact.  We term these two regimes "abstract circuits" and "physical
    circuits", respectively.

    Qiskit has two ways of distinguishing a circuit that is intended to be physical.  This is a
    fuzzy check, for historical reasons; originally, Qiskit never made the distinction at all (which
    is why :func:`.transpile` is called that, and not called ``compile``!).  The most explicit way
    is through the :attr:`layout` attribute of circuits; if this is set, the circuit is certainly
    intended to be physical.  The older, more implicit, way is the metadata of the :class:`.Qubit`
    objects and :class:`.QuantumRegister` instances in the circuit.  A circuit can only be
    considered (as judged by several transpiler passes) as physical if it contains exactly one
    quantum register, which is called ``q`` and owns all the circuit qubits in index order.  Again
    for historical reasons, this is the default for the ``QuantumCircuit(int [, int])`` form of the
    default constructor.

    Normally, you create a :class:`QuantumCircuit` and build it in the abstract sense (regardless of
    the qubit metadata).  You then call :func:`.transpile` to compile the circuit into a
    hardware-supported circuit.  However, in cases where you want to write a hardware efficient
    circuit from the beginning, you can short-circuit the full compilation infrastructure using the
    :meth:`ensure_physical` method.  This will ensure that, no matter how you defined the initial
    qubit metadata, all requirements for the circuit to be considered physical will be satisfied,
    with the qubit indices mapped to the hardware qubits.

    For more complete control over choosing a virtual-to-physical mapping and routing, see :ref:`the
    layout <transpiler-preset-stage-layout>` and `routing <transpiler-preset-stage-routing>` stages
    of the preset compilation pipelines.

    .. automethod:: ensure_physical
    .. automethod:: estimate_fidelity


    Instruction-like methods
    ========================

    ..
        These methods really shouldn't be on `QuantumCircuit` at all.  They're generally more
        appropriate as `Instruction` or `Gate` methods.  `reverse_ops` shouldn't be a method _full
        stop_---it was copying a `DAGCircuit` method from an implementation detail of the original
        `SabreLayout` pass in Qiskit.

    :class:`QuantumCircuit` also contains a small number of methods that are very
    :class:`~.circuit.Instruction`-like in detail.  You may well find better integration and more
    API support if you first convert your circuit to an :class:`~.circuit.Instruction`
    (:meth:`to_instruction`) or :class:`.Gate` (:meth:`to_gate`) as appropriate, then call the
    corresponding method.

    .. automethod:: control
    .. automethod:: inverse
    .. automethod:: power
    .. automethod:: repeat
    .. automethod:: reverse_ops

    Visualization
    =============

    Qiskit includes some drawing tools to give you a quick feel for what your circuit looks like.
    This tooling is primarily targeted at producing either a `Matplotlib
    <https://matplotlib.org/>`__- or text-based drawing.  There is also a lesser-featured LaTeX
    backend for drawing, but this is only for simple circuits, and is not as actively maintained.

    .. seealso::
        :mod:`qiskit.visualization`
            The primary documentation for all of Qiskit's visualization tooling.

    .. automethod:: draw

    In addition to the core :meth:`draw` driver, there are two visualization-related helper methods,
    which are mostly useful for quickly unwrapping some inner instructions or reversing the
    :ref:`qubit-labelling conventions <circuit-conventions>` in the drawing.  For more general
    mutation, including basis-gate rewriting, you should use the transpiler
    (:mod:`qiskit.transpiler`).

    .. automethod:: decompose
    .. automethod:: reverse_bits
    """

    instances = 0
    prefix = "circuit"
    name: str
    """A human-readable name for the circuit.

    Example:

        .. plot::
            :include-source:
            :nofigs:
            :context: reset

            from qiskit import QuantumCircuit

            qc = QuantumCircuit(2, 2, name="my_circuit")
            print(qc.name)

        .. code-block:: text

            my_circuit
    """

    def __init__(
        self,
        *regs: Register | int | Sequence[Bit],
        name: str | None = None,
        global_phase: ParameterValueType = 0,
        metadata: dict | None = None,
        inputs: Iterable[expr.Var] = (),
        captures: Iterable[expr.Var | expr.Stretch] = (),
        declarations: Mapping[expr.Var, expr.Expr] | Iterable[tuple[expr.Var, expr.Expr]] = (),
    ):
        """
        Default constructor of :class:`QuantumCircuit`.

        ..
            `QuantumCircuit` documents its `__init__` method explicitly, unlike most classes where
            it's implicitly appended to the class-level documentation, just because the class is so
            huge and has a lot of introductory material to its class docstring.

        Args:
            regs: The registers to be included in the circuit.

                * If a list of :class:`~.Register` objects, represents the :class:`.QuantumRegister`
                  and/or :class:`.ClassicalRegister` objects to include in the circuit.

                  For example:

                    * ``QuantumCircuit(QuantumRegister(4))``
                    * ``QuantumCircuit(QuantumRegister(4), ClassicalRegister(3))``
                    * ``QuantumCircuit(QuantumRegister(4, 'qr0'), QuantumRegister(2, 'qr1'))``

                * If a list of ``int``, the amount of qubits and/or classical bits to include in
                  the circuit. It can either be a single int for just the number of quantum bits,
                  or 2 ints for the number of quantum bits and classical bits, respectively.

                  For example:

                    * ``QuantumCircuit(4) # A QuantumCircuit with 4 qubits``
                    * ``QuantumCircuit(4, 3) # A QuantumCircuit with 4 qubits and 3 classical bits``

                * If a list of python lists containing :class:`.Bit` objects, a collection of
                  :class:`.Bit` s to be added to the circuit.

            name: the name of the quantum circuit. If not set, an automatically generated string
                will be assigned.
            global_phase: The global phase of the circuit in radians.
            metadata: Arbitrary key value metadata to associate with the circuit. This gets
                stored as free-form data in a dict in the
                :attr:`~qiskit.circuit.QuantumCircuit.metadata` attribute. It will not be directly
                used in the circuit.
            inputs: any variables to declare as ``input`` runtime variables for this circuit.  These
                should already be existing :class:`.expr.Var` nodes that you build from somewhere
                else; if you need to create the inputs as well, use
                :meth:`QuantumCircuit.add_input`.  The variables given in this argument will be
                passed directly to :meth:`add_input`.  A circuit cannot have both ``inputs`` and
                ``captures``.
            captures: any variables that this circuit scope should capture from a containing
                scope.  The variables given here will be passed directly to :meth:`add_capture`.  A
                circuit cannot have both ``inputs`` and ``captures``.
            declarations: any variables that this circuit should declare and initialize immediately.
                You can order this input so that later declarations depend on earlier ones
                (including inputs or captures). If you need to depend on values that will be
                computed later at runtime, use :meth:`add_var` at an appropriate point in the
                circuit execution.

                This argument is intended for convenient circuit initialization when you already
                have a set of created variables.  The variables used here will be directly passed to
                :meth:`add_var`, which you can use directly if this is the first time you are
                creating the variable.

        Raises:
            CircuitError: if the circuit name, if given, is not valid.
            CircuitError: if both ``inputs`` and ``captures`` are given.
        """
        if any(not isinstance(reg, (list, QuantumRegister, ClassicalRegister)) for reg in regs):
            # check if inputs are integers, but also allow e.g. 2.0

            try:
                valid_reg_size = all(reg == int(reg) for reg in regs)
            except (ValueError, TypeError):
                valid_reg_size = False

            if not valid_reg_size:
                raise CircuitError(
                    "Circuit args must be Registers or integers. ("
                    f"{[type(reg).__name__ for reg in regs]} '{regs}' was "
                    "provided)"
                )

            regs = tuple(int(reg) for reg in regs)  # cast to int
        self._base_name = None
        if name is None:
            self._base_name = self._cls_prefix()
            self._name_update()
        elif not isinstance(name, str):
            raise CircuitError(
                "The circuit name should be a string (or None to auto-generate a name)."
            )
        else:
            self._base_name = name
            self.name = name
        self._increment_instances()

        # An explicit implementation of the circuit scope builder interface used to dispatch appends
        # and the like to the relevant control-flow scope.
        self._builder_api = _OuterCircuitScopeInterface(self)

        self._op_start_times = None

        # A stack to hold the instruction sets that are being built up during for-, if- and
        # while-block construction.  These are stored as a stripped down sequence of instructions,
        # and sets of qubits and clbits, rather than a full QuantumCircuit instance because the
        # builder interfaces need to wait until they are completed before they can fill in things
        # like `break` and `continue`.  This is because these instructions need to "operate" on the
        # full width of bits, but the builder interface won't know what bits are used until the end.
        self._control_flow_scopes: list[
            qiskit.circuit.controlflow.builder.ControlFlowBuilderBlock
        ] = []

        # Data contains a list of instructions and their contexts,
        # in the order they were applied.
        self._data: CircuitData = CircuitData()

        self._ancillas: list[AncillaQubit] = []
        self.add_register(*regs)

        self._layout = None
        self.global_phase = global_phase

        # Add classical variables.  Resolve inputs and captures first because they can't depend on
        # anything, but declarations might depend on them.
        for input_ in inputs:
            self.add_input(input_)
        for capture in captures:
            self.add_capture(capture)
        if isinstance(declarations, Mapping):
            declarations = declarations.items()
        for var, initial in declarations:
            self.add_var(var, initial)

        self._duration = None
        self._unit = "dt"
        self.metadata = {} if metadata is None else metadata
        """Arbitrary user-defined dictionary of metadata for the circuit.

        Qiskit will not examine the content of this mapping, but it will pass it through the
        transpiler and reattach it to the output, so you can track your own metadata.

        Example:

            .. plot::
                :include-source:
                :nofigs:

                from qiskit import QuantumCircuit

                qc = QuantumCircuit(2, 2, metadata={'experiment_type': 'Bell state experiment'})

                print(qc.metadata)

            .. code-block:: text

                {'experiment_type': 'Bell state experiment'}
        """

    @property
    @deprecate_func(since="1.3.0", removal_timeline="in Qiskit 3.0.0", is_property=True)
    def duration(self):
        """The total duration of the circuit, set by a scheduling transpiler pass.  Its unit is
        specified by :attr:`unit`."""
        return self._duration

    @duration.setter
    def duration(self, value: int | float | None):
        self._duration = value

    @property
    @deprecate_func(since="1.3.0", removal_timeline="in Qiskit 3.0.0", is_property=True)
    def unit(self):
        """The unit that :attr:`duration` is specified in."""
        return self._unit

    @unit.setter
    def unit(self, value):
        self._unit = value

    @classmethod
    def _from_circuit_data(
        cls, data: CircuitData, legacy_qubits: bool = False, name: str | None = None
    ) -> typing.Self:
        """A private constructor from rust space circuit data."""
        out = QuantumCircuit(name=name)
        out._data = data
        if legacy_qubits:
            out.ensure_physical(apply_layout=False)
        return out

    @staticmethod
    def from_instructions(
        instructions: Iterable[
            CircuitInstruction
            | tuple[qiskit.circuit.Instruction]
            | tuple[qiskit.circuit.Instruction, Iterable[Qubit]]
            | tuple[qiskit.circuit.Instruction, Iterable[Qubit], Iterable[Clbit]]
        ],
        *,
        qubits: Iterable[Qubit] = (),
        clbits: Iterable[Clbit] = (),
        name: str | None = None,
        global_phase: ParameterValueType = 0,
        metadata: dict | None = None,
    ) -> QuantumCircuit:
        """Construct a circuit from an iterable of :class:`.CircuitInstruction`\\ s.

        Args:
            instructions: The instructions to add to the circuit.
            qubits: Any qubits to add to the circuit. This argument can be used,
                for example, to enforce a particular ordering of qubits.
            clbits: Any classical bits to add to the circuit. This argument can be used,
                for example, to enforce a particular ordering of classical bits.
            name: The name of the circuit.
            global_phase: The global phase of the circuit in radians.
            metadata: Arbitrary key value metadata to associate with the circuit.

        Returns:
            The quantum circuit.
        """
        circuit = QuantumCircuit(name=name, global_phase=global_phase, metadata=metadata)
        added_qubits = set()
        added_clbits = set()
        if qubits:
            qubits = list(qubits)
            circuit.add_bits(qubits)
            added_qubits.update(qubits)
        if clbits:
            clbits = list(clbits)
            circuit.add_bits(clbits)
            added_clbits.update(clbits)
        for instruction in instructions:
            if not isinstance(instruction, CircuitInstruction):
                instruction = CircuitInstruction(*instruction)
            qubits = [qubit for qubit in instruction.qubits if qubit not in added_qubits]
            clbits = [clbit for clbit in instruction.clbits if clbit not in added_clbits]
            circuit.add_bits(qubits)
            circuit.add_bits(clbits)
            added_qubits.update(qubits)
            added_clbits.update(clbits)
            circuit._append(instruction)
        return circuit

    @property
    def layout(self) -> TranspileLayout | None:
        """Return any associated layout information about the circuit.

        This attribute contains an optional :class:`~.TranspileLayout`
        object. This is typically set on the output from :func:`~.transpile`
        or :meth:`.PassManager.run` to retain information about the
        permutations caused on the input circuit by transpilation.

        There are two types of permutations caused by the :func:`~.transpile`
        function: an initial layout that permutes the qubits based on the
        selected physical qubits on the :class:`~.Target`, and a final layout,
        which is an output permutation caused by :class:`~.SwapGate`\\ s
        inserted during routing.

        Example:

            .. plot::
                :include-source:
                :nofigs:

                from qiskit import QuantumCircuit
                from qiskit.providers.fake_provider import GenericBackendV2
                from qiskit.transpiler import generate_preset_pass_manager

                # Create circuit to test transpiler on
                qc = QuantumCircuit(3, 3)
                qc.h(0)
                qc.cx(0, 1)
                qc.swap(1, 2)
                qc.cx(0, 1)

                # Add measurements to the circuit
                qc.measure([0, 1, 2], [0, 1, 2])

                # Specify the QPU to target
                backend = GenericBackendV2(3)

                # Transpile the circuit
                pass_manager = generate_preset_pass_manager(
                optimization_level=1, backend=backend
                )
                transpiled = pass_manager.run(qc)

                # Print the layout after transpilation
                print(transpiled.layout.routing_permutation())

            .. code-block:: text

                [0, 1, 2]

        """
        return self._layout

    @property
    def data(self) -> QuantumCircuitData:
        """The circuit data (instructions and context).

        Example:

            .. plot::
                :include-source:
                :nofigs:

                from qiskit import QuantumCircuit

                qc = QuantumCircuit(2, 2)
                qc.measure([0], [1])
                print(qc.data)

            .. code-block:: text

                [CircuitInstruction(operation=Instruction(name='measure', num_qubits=1,
                num_clbits=1, params=[]), qubits=(Qubit(QuantumRegister(2, 'q'), 0),),
                clbits=(Clbit(ClassicalRegister(2, 'c'), 1),))]

        Returns:
            A list-like object containing the :class:`.CircuitInstruction` instances in the circuit.
        """
        return QuantumCircuitData(self)

    @data.setter
    def data(self, data_input: Iterable):
        """Sets the circuit data from a list of instructions and context.

        Args:
            data_input (Iterable): A sequence of instructions with their execution contexts.  The
                elements must either be instances of :class:`.CircuitInstruction` (preferred), or a
                3-tuple of ``(instruction, qargs, cargs)`` (legacy).  In the legacy format,
                ``instruction`` must be an :class:`~.circuit.Instruction`, while ``qargs`` and
                ``cargs`` must be iterables of :class:`~.circuit.Qubit` or :class:`.Clbit`
                specifiers (similar to the allowed forms in calls to :meth:`append`).
        """
        # If data_input is QuantumCircuitData(self), clearing self._data
        # below will also empty data_input, so make a shallow copy first.
        if isinstance(data_input, CircuitData):
            data_input = data_input.copy()
        else:
            data_input = list(data_input)
        self._data.clear()
        # Repopulate the parameter table with any global-phase entries.
        self.global_phase = self.global_phase
        if not data_input:
            return
        if isinstance(data_input[0], CircuitInstruction):
            for instruction in data_input:
                self.append(instruction, copy=False)
        else:
            for instruction, qargs, cargs in data_input:
                self.append(instruction, qargs, cargs, copy=False)

    @property
    def op_start_times(self) -> list[int]:
        """Return a list of operation start times.

        .. note::
           This attribute computes the estimate starting time of the operations in the scheduled circuit
           and only works for simple circuits that have no control flow or other classical feed-forward
           operations.

        This attribute is enabled once one of scheduling analysis passes
        runs on the quantum circuit.

        Example:

            .. plot::
                :include-source:
                :nofigs:

                from qiskit import QuantumCircuit
                from qiskit.providers.fake_provider import GenericBackendV2
                from qiskit.transpiler import generate_preset_pass_manager

                qc = QuantumCircuit(2)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure_all()

                # Print the original circuit
                print("Original circuit:")
                print(qc)

                # Transpile the circuit with a specific basis gates list and print the resulting circuit
                backend = GenericBackendV2(2, basis_gates=['u1', 'u2', 'u3', 'cx'])
                pm = generate_preset_pass_manager(
                    optimization_level=1, backend=backend, scheduling_method="alap"
                )
                transpiled_qc = pm.run(qc)
                print("Transpiled circuit with basis gates ['u1', 'u2', 'u3', 'cx']:")
                print(transpiled_qc)

                # Print the start times of each instruction in the transpiled circuit
                print("Start times of instructions in the transpiled circuit:")
                for instruction, start_time in zip(transpiled_qc.data, transpiled_qc.op_start_times):
                    print(f"{instruction.operation.name}: {start_time}")

            .. code-block:: text


                Original circuit:
                        ┌───┐      ░ ┌─┐
                q_0: ┤ H ├──■───░─┤M├───
                        └───┘┌─┴─┐ ░ └╥┘┌─┐
                q_1: ─────┤ X ├─░──╫─┤M├
                            └───┘ ░  ║ └╥┘
                meas: 2/══════════════╩══╩═
                                    0  1

                Transpiled circuit with basis gates ['u1', 'u2', 'u3', 'cx']:
                            ┌─────────┐          ░ ┌─────────────────┐┌─┐
                q_0 -> 0 ───┤ U2(0,π) ├──────■───░─┤ Delay(1255[dt]) ├┤M├
                        ┌──┴─────────┴───┐┌─┴─┐ ░ └───────┬─┬───────┘└╥┘
                q_1 -> 1 ┤ Delay(196[dt]) ├┤ X ├─░─────────┤M├─────────╫─
                        └────────────────┘└───┘ ░         └╥┘         ║
                meas: 2/═══════════════════════════════════╩══════════╩═
                                                            1          0

                Start times of instructions in the transpiled circuit:
                u2: 0
                delay: 0
                cx: 196
                barrier: 2098
                delay: 2098
                measure: 3353
                measure: 2098

        Returns:
            List of integers representing instruction estimated start times.
            The index corresponds to the index of instruction in :attr:`QuantumCircuit.data`.

        Raises:
            AttributeError: When circuit is not scheduled.
        """
        if self._op_start_times is None:
            raise AttributeError(
                "This circuit is not scheduled. "
                "To schedule it run the circuit through one of the transpiler scheduling passes."
            )
        return self._op_start_times

    @property
    def metadata(self) -> dict:
        """The user-provided metadata associated with the circuit.

        The metadata for the circuit is a user-provided ``dict`` of metadata
        for the circuit. It will not be used to influence the execution or
        operation of the circuit, but it is expected to be passed between
        all transforms of the circuit (i.e., transpilation) and that providers will
        associate any circuit metadata with the results it returns from
        execution of that circuit.

        Example:

            .. plot::
                :include-source:
                :nofigs:

                from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

                q = QuantumRegister(2)
                c = ClassicalRegister(2)
                qc = QuantumCircuit(q, c)

                qc.metadata = {'experiment_type': 'Bell state experiment'}

                print(qc.metadata)

            .. code-block:: text

               {'experiment_type': 'Bell state experiment'}
        """
        return self._metadata

    @metadata.setter
    def metadata(self, metadata: dict):
        """Update the circuit metadata"""
        if not isinstance(metadata, dict):
            raise TypeError("Only a dictionary is accepted for circuit metadata")
        self._metadata = metadata

    def __str__(self) -> str:
        return str(self.draw(output="text"))

    def __eq__(self, other) -> bool:
        if not isinstance(other, QuantumCircuit):
            return False

        # TODO: remove the DAG from this function
        from qiskit.converters import circuit_to_dag

        return circuit_to_dag(self, copy_operations=False) == circuit_to_dag(
            other, copy_operations=False
        )

    def __deepcopy__(self, memo=None):
        # This is overridden to minimize memory pressure when we don't
        # actually need to pickle (i.e. the typical deepcopy case).
        # Note:
        #   This is done here instead of in CircuitData since PyO3
        #   doesn't include a native way to recursively call
        #   copy.deepcopy(memo).
        cls = self.__class__
        result = cls.__new__(cls)
        for k in self.__dict__.keys() - {"_data", "_builder_api"}:
            setattr(result, k, _copy.deepcopy(self.__dict__[k], memo))

        result._builder_api = _OuterCircuitScopeInterface(result)

        # Avoids pulling self._data into a Python list
        # like we would when pickling.
        result._data = self._data.copy(deepcopy=True)
        result._data.replace_bits(
            qubits=_copy.deepcopy(self._data.qubits, memo),
            clbits=_copy.deepcopy(self._data.clbits, memo),
            qregs=_copy.deepcopy(self._data.qregs, memo),
            cregs=_copy.deepcopy(self._data.cregs, memo),
        )
        return result

    @classmethod
    def _increment_instances(cls):
        cls.instances += 1

    @classmethod
    def _cls_instances(cls) -> int:
        """Return the current number of instances of this class,
        useful for auto naming."""
        return cls.instances

    @classmethod
    def _cls_prefix(cls) -> str:
        """Return the prefix to use for auto naming."""
        return cls.prefix

    def _name_update(self) -> None:
        """update name of instance using instance number"""
        if multiprocessing.parent_process() is None:
            pid_name = ""
        else:
            pid_name = f"-{multiprocessing.current_process().pid}"
        self.name = f"{self._base_name}-{self._cls_instances()}{pid_name}"

    def has_register(self, register: Register) -> bool:
        """
        Test if this circuit has the register r.

        Args:
            register (Register): a quantum or classical register.

        Returns:
            bool: True if the register is contained in this circuit.
        """
        has_reg = False
        if isinstance(register, QuantumRegister) and register in self.qregs:
            has_reg = True
        elif isinstance(register, ClassicalRegister) and register in self.cregs:
            has_reg = True
        return has_reg

    def ensure_physical(self, num_qubits: int | None = None, *, apply_layout: bool = True) -> bool:
        """Put this circuit into canonical physical form, with the given number of qubits, if it is
        not already.

        Several Qiskit transpiler passes only make sense when applied to circuits defined in terms
        of physical qubits.  If you have manually constructed a circuit where the qubit indices
        correspond to physical qubits, use this function to ensure that the metadata of the circuit
        matches the canonical physical form.  This means replacing the qubit data with a single
        owning register called ``"q"``, and (optionally) setting the :attr:`layout` field of the
        circuit to link these physical qubits with the original virtual ones.

        If the circuit does not already have a layout, this method (with ``apply_layout=True``) is
        equivalent to applying the full :ref:`trivial layout method
        <transpiler-preset-stage-layout-trivial>` of the preset compilation pipeline.

        If the circuit is already canonically physical, nothing happens to it.  This method cannot
        change the number of qubits in the circuit if it already has a :attr:`layout` set.

        Args:
            num_qubits: if given, expand the circuit with ancillas up to this size.  The ancillas
                will always be the highest qubit indices of the circuit.  If not given (the
                default), the circuit stays the same width.  This option cannot be set if the
                circuit already as a :attr:`layout`.
            apply_layout: if true (the default), set the :attr:`layout` attribute of the circuit
                appropriately so that the circuit appears to have been laid out with the "trivial"
                layout, including ancilla expansion, for a backend of width ``num_qubits``.  This
                has no effect if the circuit already had a :attr:`layout`.

        Returns:
            whether the circuit was modified in order to make it physical.

        Raises:
            ValueError: if ``num_qubits`` is too small for the circuit.
            CircuitError: if ``num_qubits`` is set to attempt to expand the circuit, but the circuit
                already has a layout set.
        """
        original_num_qubits = self.num_qubits
        if num_qubits is not None and num_qubits < original_num_qubits:
            raise ValueError(
                f"cannot have fewer physical qubits ({num_qubits})"
                f" than virtual ({original_num_qubits})"
            )
        if self._layout is not None:
            if num_qubits is not None and num_qubits != original_num_qubits:
                raise CircuitError("cannot expand a circuit that already has a set layout")
            expected = QuantumRegister(original_num_qubits, "q")
            if self.qubits != list(expected) or self.qregs != [expected]:  # pragma: no cover
                # This is mostly defensive; we _shouldn't_ get a circuit in this form, but it's
                # possible for a badly formed transpiler pass to do it by accident.
                raise CircuitError(
                    "This circuit has a set layout, but its qubits are not in the canonical"
                    " physical form.  This might indicate a faulty layout transpilation stage has"
                    " been used."
                )
            return False

        if apply_layout:
            from qiskit.transpiler import Layout

            virtuals = self.qubits.copy()
            if num_qubits is not None and num_qubits > original_num_qubits:

                ancilla_register_name = "ancilla"
                ancilla_suffix = 0
                reg_names = {reg.name for reg in self.qregs}
                while ancilla_register_name in reg_names:
                    ancilla_register_name = f"ancilla{ancilla_suffix}"
                    ancilla_suffix += 1
                virtuals.extend(
                    QuantumRegister(num_qubits - original_num_qubits, ancilla_register_name)
                )
            initial_layout = Layout(dict(enumerate(virtuals)))
        else:
            initial_layout = None
        self._data.make_physical(num_qubits)
        if initial_layout is not None:
            from qiskit.transpiler import TranspileLayout

            self._layout = TranspileLayout(
                initial_layout=initial_layout,
                input_qubit_mapping=initial_layout.get_virtual_bits().copy(),
                final_layout=None,
                _input_qubit_count=original_num_qubits,
                _output_qubit_list=self.qubits.copy(),
            )
        return True

    def reverse_ops(self) -> QuantumCircuit:
        """Reverse the circuit by reversing the order of instructions.

        This is done by recursively reversing all instructions.
        It does not invert (adjoint) any gate.

        Returns:
            QuantumCircuit: the reversed circuit.

        Examples:

            input:

            .. code-block:: text

                     ┌───┐
                q_0: ┤ H ├─────■──────
                     └───┘┌────┴─────┐
                q_1: ─────┤ RX(1.57) ├
                          └──────────┘

            output:

            .. code-block:: text

                                 ┌───┐
                q_0: ─────■──────┤ H ├
                     ┌────┴─────┐└───┘
                q_1: ┤ RX(1.57) ├─────
                     └──────────┘
        """
        reverse_circ = self.copy_empty_like(self.name + "_reverse")

        for instruction in reversed(self.data):
            reverse_circ._append(instruction.replace(operation=instruction.operation.reverse_ops()))

        reverse_circ._duration = self._duration
        reverse_circ._unit = self._unit
        return reverse_circ

    def reverse_bits(self) -> QuantumCircuit:
        """Return a circuit with the opposite order of wires.

        The circuit is "vertically" flipped. If a circuit is
        defined over multiple registers, the resulting circuit will have
        the same registers but with their order flipped.

        This method is useful for converting a circuit written in little-endian
        convention to the big-endian equivalent, and vice versa.

        Returns:
            QuantumCircuit: the circuit with reversed bit order.

        Examples:

            input:

            .. code-block:: text

                     ┌───┐
                a_0: ┤ H ├──■─────────────────
                     └───┘┌─┴─┐
                a_1: ─────┤ X ├──■────────────
                          └───┘┌─┴─┐
                a_2: ──────────┤ X ├──■───────
                               └───┘┌─┴─┐
                b_0: ───────────────┤ X ├──■──
                                    └───┘┌─┴─┐
                b_1: ────────────────────┤ X ├
                                         └───┘

            output:

            .. code-block:: text

                                         ┌───┐
                b_0: ────────────────────┤ X ├
                                    ┌───┐└─┬─┘
                b_1: ───────────────┤ X ├──■──
                               ┌───┐└─┬─┘
                a_0: ──────────┤ X ├──■───────
                          ┌───┐└─┬─┘
                a_1: ─────┤ X ├──■────────────
                     ┌───┐└─┬─┘
                a_2: ┤ H ├──■─────────────────
                     └───┘
        """
        circ = QuantumCircuit(
            list(reversed(self.qubits)),
            list(reversed(self.clbits)),
            name=self.name,
            global_phase=self.global_phase,
        )
        new_qubit_map = circ.qubits[::-1]
        new_clbit_map = circ.clbits[::-1]
        for reg in reversed(self.qregs):
            bits = [new_qubit_map[self.find_bit(qubit).index] for qubit in reversed(reg)]
            circ.add_register(QuantumRegister(bits=bits, name=reg.name))
        for reg in reversed(self.cregs):
            bits = [new_clbit_map[self.find_bit(clbit).index] for clbit in reversed(reg)]
            circ.add_register(ClassicalRegister(bits=bits, name=reg.name))

        for instruction in self.data:
            qubits = [new_qubit_map[self.find_bit(qubit).index] for qubit in instruction.qubits]
            clbits = [new_clbit_map[self.find_bit(clbit).index] for clbit in instruction.clbits]
            circ._append(instruction.replace(qubits=qubits, clbits=clbits))
        return circ

    def inverse(self, annotated: bool = False) -> QuantumCircuit:
        """Invert (take adjoint of) this circuit.

        This is done by recursively inverting all gates.

        Args:
            annotated: indicates whether the inverse gate can be implemented
                as an annotated gate.

        Returns:
            QuantumCircuit: the inverted circuit

        Raises:
            CircuitError: if the circuit cannot be inverted.

        Examples:

            input:

            .. code-block:: text

                     ┌───┐
                q_0: ┤ H ├─────■──────
                     └───┘┌────┴─────┐
                q_1: ─────┤ RX(1.57) ├
                          └──────────┘

            output:

            .. code-block:: text

                                  ┌───┐
                q_0: ──────■──────┤ H ├
                     ┌─────┴─────┐└───┘
                q_1: ┤ RX(-1.57) ├─────
                     └───────────┘
        """
        inverse_circ = QuantumCircuit(
            self.qubits,
            self.clbits,
            *self.qregs,
            *self.cregs,
            name=self.name + "_dg",
            global_phase=-self.global_phase,
        )

        for instruction in reversed(self._data):
            inverse_circ._append(
                instruction.replace(operation=instruction.operation.inverse(annotated=annotated))
            )
        return inverse_circ

    def repeat(self, reps: int, *, insert_barriers: bool = False) -> QuantumCircuit:
        """Repeat this circuit ``reps`` times.

        Args:
            reps (int): How often this circuit should be repeated.
            insert_barriers (bool): Whether to include barriers between circuit repetitions.

        Returns:
            QuantumCircuit: A circuit containing ``reps`` repetitions of this circuit.
        """
        repeated_circ = QuantumCircuit(
            self.qubits, self.clbits, *self.qregs, *self.cregs, name=self.name + f"**{reps}"
        )

        # benefit of appending instructions: decomposing shows the subparts, i.e. the power
        # is actually `reps` times this circuit, and it is currently much faster than `compose`.
        if reps > 0:
            try:  # try to append as gate if possible to not disallow to_gate
                inst: Instruction = self.to_gate()
            except QiskitError:
                inst = self.to_instruction()
            for i in range(reps):
                repeated_circ.append(inst, self.qubits, self.clbits)
                if insert_barriers and i != reps - 1:
                    repeated_circ.barrier()

        return repeated_circ

    def power(
        self, power: float, matrix_power: bool = False, annotated: bool = False
    ) -> QuantumCircuit:
        """Raise this circuit to the power of ``power``.

        If ``power`` is a positive integer and both ``matrix_power`` and ``annotated``
        are ``False``, this implementation defaults to calling ``repeat``. Otherwise,
        the circuit is converted into a gate, and a new circuit, containing this gate
        raised to the given power, is returned. The gate raised to the given power is
        implemented either as a unitary gate if ``annotated`` is ``False`` or as an
        annotated operation if ``annotated`` is ``True``.

        Args:
            power (float): The power to raise this circuit to.
            matrix_power (bool): indicates whether the inner power gate can be implemented
                as a unitary gate.
            annotated (bool): indicates whether the inner power gate can be implemented
                as an annotated operation.

        Raises:
            CircuitError: If the circuit needs to be converted to a unitary gate, but is
                not unitary.

        Returns:
            QuantumCircuit: A circuit implementing this circuit raised to the power of ``power``.
        """
        if (
            power >= 0
            and isinstance(power, (int, np.integer))
            and not matrix_power
            and not annotated
        ):
            return self.repeat(power)

        # attempt conversion to gate
        if self.num_parameters > 0:
            raise CircuitError(
                "Cannot raise a parameterized circuit to a non-positive power "
                "or matrix-power, please bind the free parameters: "
                f"{self.parameters}"
            )

        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError(
                "The circuit contains non-unitary operations and cannot be "
                "raised to a power. Note that no qiskit.circuit.Instruction "
                "objects may be in the circuit for this operation."
            ) from ex

        power_circuit = QuantumCircuit(self.qubits, self.clbits, *self.qregs, *self.cregs)
        power_circuit.append(gate.power(power, annotated=annotated), list(range(gate.num_qubits)))
        return power_circuit

    @deprecate_arg(
        name="annotated",
        since="2.3",
        additional_msg=(
            "The method QuantumCircuit.control() no longer accepts `annotated=None`. "
            "The new default is `annotated=True`, which represents the controlled gate "
            "in the new quantum circuit as an `AnnotatedOperation` "
            "(unless a dedicated controlled-gate class already exists). You can explicitly set "
            "`annotated=False` to preserve the previous behavior. However, using `annotated=True` "
            "is recommended, as it defers construction of the controlled circuit to transpiler, "
            "and furthermore enables additional controlled-gate optimizations (typically leading "
            "to higher-quality circuits)."
        ),
        predicate=lambda my_arg: my_arg is None,
        removal_timeline="in Qiskit 3.0",
    )
    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ) -> QuantumCircuit:
        """Return the controlled version of this circuit.

        The original circuit is converted into a gate, and the resulting circuit contains
        the controlled version of this gate.
        This controlled gate is implemented as :class:`.ControlledGate` when ``annotated``
        is ``False``, and as :class:`.AnnotatedOperation` when ``annotated`` is ``True``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: An optional label to give the controlled gate for visualization.
                Defaults to ``None``. Ignored if the controlled gate is implemented as an annotated
                operation.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Indicates whether the controlled gate should be implemented as a controlled gate
                or as an annotated operation.

        Returns:
            QuantumCircuit: The controlled version of this circuit.

        Raises:
            CircuitError: If the circuit contains a non-unitary operation and cannot be controlled.
        """
        try:
            gate = self.to_gate()
        except QiskitError as ex:
            raise CircuitError(
                "The circuit contains non-unitary operations and cannot be "
                "controlled. Note that no qiskit.circuit.Instruction objects may "
                "be in the circuit for this operation."
            ) from ex

        controlled_gate = gate.control(num_ctrl_qubits, label, ctrl_state, annotated)
        control_qreg = QuantumRegister(num_ctrl_qubits)
        controlled_circ = QuantumCircuit(
            control_qreg, self.qubits, *self.qregs, name=f"c_{self.name}"
        )
        controlled_circ.append(controlled_gate, controlled_circ.qubits)

        return controlled_circ

    @overload
    def compose(
        self,
        other: QuantumCircuit | Instruction,
        qubits: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        clbits: ClbitSpecifier | Sequence[ClbitSpecifier] | None = None,
        front: bool = False,
        inplace: Literal[True] = True,
        wrap: bool = False,
        *,
        copy: bool = True,
        var_remap: (
            Mapping[str | expr.Var | expr.Stretch, str | expr.Var | expr.Stretch] | None
        ) = None,
        inline_captures: bool = False,
    ) -> None: ...

    @overload
    def compose(
        self,
        other: QuantumCircuit | Instruction,
        qubits: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        clbits: ClbitSpecifier | Sequence[ClbitSpecifier] | None = None,
        front: bool = False,
        inplace: Literal[False] = False,
        wrap: bool = False,
        *,
        copy: bool = True,
        var_remap: (
            Mapping[str | expr.Var | expr.Stretch, str | expr.Var | expr.Stretch] | None
        ) = None,
        inline_captures: bool = False,
    ) -> QuantumCircuit: ...

    def compose(
        self,
        other: QuantumCircuit | Instruction,
        qubits: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        clbits: ClbitSpecifier | Sequence[ClbitSpecifier] | None = None,
        front: bool = False,
        inplace: bool = False,
        wrap: bool = False,
        *,
        copy: bool = True,
        var_remap: (
            Mapping[str | expr.Var | expr.Stretch, str | expr.Var | expr.Stretch] | None
        ) = None,
        inline_captures: bool = False,
    ) -> QuantumCircuit | None:
        """Apply the instructions from one circuit onto specified qubits and/or clbits on another.

        .. note::

            By default, this creates a new circuit object, leaving ``self`` untouched.  For most
            uses of this function, it is far more efficient to set ``inplace=True`` and modify the
            base circuit in-place.

        When dealing with realtime variables (:class:`.expr.Var` and :class:`.expr.Stretch` instances),
        there are two principal strategies for using :meth:`compose`:

        1. The ``other`` circuit is treated as entirely additive, including its variables.  The
           variables in ``other`` must be entirely distinct from those in ``self`` (use
           ``var_remap`` to help with this), and all variables in ``other`` will be declared anew in
           the output with matching input/capture/local scoping to how they are in ``other``.  This
           is generally what you want if you're joining two unrelated circuits.

        2. The ``other`` circuit was created as an exact extension to ``self`` to be inlined onto
           it, including acting on the existing variables in their states at the end of ``self``.
           In this case, ``other`` should be created with all these variables to be inlined declared
           as "captures", and then you can use ``inline_captures=True`` in this method to link them.
           This is generally what you want if you're building up a circuit by defining layers
           on-the-fly, or rebuilding a circuit using layers taken from itself.  You might find the
           ``vars_mode="captures"`` argument to :meth:`copy_empty_like` useful to create each
           layer's base, in this case.

        Args:
            other (qiskit.circuit.Instruction or QuantumCircuit):
                (sub)circuit or instruction to compose onto self.  If not a :obj:`.QuantumCircuit`,
                this can be anything that :obj:`.append` will accept.
            qubits (list[Qubit|int]): qubits of self to compose onto.
            clbits (list[Clbit|int]): clbits of self to compose onto.
            front (bool): If ``True``, front composition will be performed.  This is not possible within
                control-flow builder context managers.
            inplace (bool): If ``True``, modify the object. Otherwise, return composed circuit.
            copy (bool): If ``True`` (the default), then the input is treated as shared, and any
                contained instructions will be copied, if they might need to be mutated in the
                future.  You can set this to ``False`` if the input should be considered owned by
                the base circuit, in order to avoid unnecessary copies; in this case, it is not
                valid to use ``other`` afterward, and some instructions may have been mutated in
                place.
            var_remap (Mapping): mapping to use to rewrite :class:`.expr.Var` and
                :class:`.expr.Stretch` nodes in ``other`` as they are inlined into ``self``.
                This can be used to avoid naming conflicts.

                Both keys and values can be given as strings or direct identifier instances.
                If a key is a string, it matches any :class:`~.expr.Var` or :class:`~.expr.Stretch`
                with the same name.  If a value is a string, whenever a new key matches it, a new
                :class:`~.expr.Var` or :class:`~.expr.Stretch` is created with the correct type.
                If a value is a :class:`~.expr.Var`, its :class:`~.expr.Expr.type` must exactly
                match that of the variable it is replacing.
            inline_captures (bool): if ``True``, then all "captured" identifier nodes in
                the ``other`` :class:`.QuantumCircuit` are assumed to refer to identifiers already
                declared in ``self`` (as any input/capture/local type), and the uses in ``other``
                will apply to the existing identifiers.  If you want to build up a layer for an
                existing circuit to use with :meth:`compose`, you might find the
                ``vars_mode="captures"`` argument to :meth:`copy_empty_like` useful.  Any remapping
                in ``vars_remap`` occurs before evaluating this variable inlining.

                If this is ``False`` (the default), then all identifiers in ``other`` will be required
                to be distinct from those in ``self``, and new declarations will be made for them.
            wrap (bool): If True, wraps the other circuit into a gate (or instruction, depending on
                whether it contains only unitary instructions) before composing it onto self.
                Rather than using this option, it is almost always better to manually control this
                yourself by using :meth:`to_instruction` or :meth:`to_gate`, and then call
                :meth:`append`.

        Returns:
            QuantumCircuit: the composed circuit (returns None if inplace==True).

        Raises:
            CircuitError: if no correct wire mapping can be made between the two circuits, such as
                if ``other`` is wider than ``self``.
            CircuitError: if trying to emit a new circuit while ``self`` has a partially built
                control-flow context active, such as the context-manager forms of :meth:`if_test`,
                :meth:`for_loop` and :meth:`while_loop`.
            CircuitError: if trying to compose to the front of a circuit when a control-flow builder
                block is active; there is no clear meaning to this action.

        Examples:
            .. code-block:: python

                >>> lhs.compose(rhs, qubits=[3, 2], inplace=True)

            .. code-block:: text

                            ┌───┐                   ┌─────┐                ┌───┐
                lqr_1_0: ───┤ H ├───    rqr_0: ──■──┤ Tdg ├    lqr_1_0: ───┤ H ├───────────────
                            ├───┤              ┌─┴─┐└─────┘                ├───┤
                lqr_1_1: ───┤ X ├───    rqr_1: ┤ X ├───────    lqr_1_1: ───┤ X ├───────────────
                         ┌──┴───┴──┐           └───┘                    ┌──┴───┴──┐┌───┐
                lqr_1_2: ┤ U1(0.1) ├  +                     =  lqr_1_2: ┤ U1(0.1) ├┤ X ├───────
                         └─────────┘                                    └─────────┘└─┬─┘┌─────┐
                lqr_2_0: ─────■─────                           lqr_2_0: ─────■───────■──┤ Tdg ├
                            ┌─┴─┐                                          ┌─┴─┐        └─────┘
                lqr_2_1: ───┤ X ├───                           lqr_2_1: ───┤ X ├───────────────
                            └───┘                                          └───┘
                lcr_0: 0 ═══════════                           lcr_0: 0 ═══════════════════════

                lcr_1: 0 ═══════════                           lcr_1: 0 ═══════════════════════

        """

        if inplace and front and self._control_flow_scopes:
            # If we're composing onto ourselves while in a stateful control-flow builder context,
            # there's no clear meaning to composition to the "front" of the circuit.
            raise CircuitError(
                "Cannot compose to the front of a circuit while a control-flow context is active."
            )
        if not inplace and self._control_flow_scopes:
            # If we're inside a stateful control-flow builder scope, even if we successfully cloned
            # the partial builder scope (not simple), the scope wouldn't be controlled by an active
            # `with` statement, so the output circuit would be permanently broken.
            raise CircuitError(
                "Cannot emit a new composed circuit while a control-flow context is active."
            )

        # Avoid mutating `dest` until as much of the error checking as possible is complete, to
        # avoid an in-place composition getting `self` in a partially mutated state for a simple
        # error that the user might want to correct in an interactive session.
        dest = self if inplace else self.copy()

        var_remap = {} if var_remap is None else var_remap

        # This doesn't use `functools.cache` so we can access it during the variable remapping of
        # instructions.  We cache all replacement lookups for a) speed and b) to ensure that
        # the same variable _always_ maps to the same replacement even if it's used in different
        # places in the recursion tree (such as being a captured variable).
        def replace_var(
            var: expr.Var | expr.Stretch, cache: Mapping[expr.Var, expr.Var]
        ) -> expr.Var | expr.Stretch:
            # This is closing over an argument to `compose`.
            nonlocal var_remap

            if out := cache.get(var):
                return out
            if (replacement := var_remap.get(var)) or (replacement := var_remap.get(var.name)):
                if isinstance(var, expr.Var):
                    if isinstance(replacement, expr.Stretch):
                        raise CircuitError(
                            "mismatched identifier kinds in replacement:"
                            f" '{var}' cannot become '{replacement}'"
                        )
                    if isinstance(replacement, str):
                        replacement = expr.Var.new(replacement, var.type)
                    if replacement.type != var.type:
                        raise CircuitError(
                            f"mismatched types in replacement for '{var.name}':"
                            f" '{var.type}' cannot become '{replacement.type}'"
                        )
                else:
                    if isinstance(replacement, expr.Var):
                        raise CircuitError(
                            "mismatched identifier kind in replacement:"
                            f" '{var}' cannot become '{replacement}'"
                        )
                    if isinstance(replacement, str):
                        replacement = expr.Stretch.new(replacement)
            else:
                replacement = var
            cache[var] = replacement
            return replacement

        # As a special case, allow composing some clbits onto no clbits - normally the destination
        # has to be strictly larger. This allows composing final measurements onto unitary circuits.
        if isinstance(other, QuantumCircuit):
            if not self.clbits and other.clbits:
                if dest._control_flow_scopes:
                    raise CircuitError(
                        "cannot implicitly add clbits while within a control-flow scope"
                    )
                dest.add_bits(other.clbits)
                for reg in other.cregs:
                    dest.add_register(reg)

        if wrap and isinstance(other, QuantumCircuit):
            other = (
                other.to_gate()
                if all(isinstance(ins.operation, Gate) for ins in other.data)
                else other.to_instruction()
            )

        if not isinstance(other, QuantumCircuit):
            if qubits is None:
                qubits = self.qubits[: other.num_qubits]
            if clbits is None:
                clbits = self.clbits[: other.num_clbits]
            if front:
                # Need to keep a reference to the data for use after we've emptied it.
                old_data = dest._data.copy(copy_instructions=copy)
                dest.clear()
                dest.append(other, qubits, clbits, copy=copy)
                for instruction in old_data:
                    dest._append(instruction)
            else:
                dest.append(other, qargs=qubits, cargs=clbits, copy=copy)
            return None if inplace else dest

        if other.num_qubits > dest.num_qubits:
            raise CircuitError(
                "Cannot compose onto a circuit with fewer qubits "
                f"({other.num_qubits} > {dest.num_qubits})."
            )
        if other.num_clbits > dest.num_clbits:
            raise CircuitError(
                "Cannot compose onto a circuit with fewer classical bits "
                f"({other.num_clbits} > {dest.num_clbits})."
            )

        # Maps bits in 'other' to bits in 'dest'.
        mapped_qubits: list[Qubit] | None
        mapped_clbits: list[Clbit] | None
        edge_map: dict[Qubit | Clbit, Qubit | Clbit] = {}
        if qubits is None:
            mapped_qubits = None
            edge_map.update(zip(other.qubits, dest.qubits))
        else:
            mapped_qubits = dest._qbit_argument_conversion(qubits)
            if len(mapped_qubits) != other.num_qubits:
                raise CircuitError(
                    f"Number of items in qubits parameter ({len(mapped_qubits)}) does not"
                    f" match number of qubits in the circuit ({other.num_qubits})."
                )
            if len(set(mapped_qubits)) != len(mapped_qubits):
                raise CircuitError(
                    f"Duplicate qubits referenced in 'qubits' parameter: '{mapped_qubits}'"
                )
            edge_map.update(zip(other.qubits, mapped_qubits))

        if clbits is None:
            mapped_clbits = None
            edge_map.update(zip(other.clbits, dest.clbits))
        else:
            mapped_clbits = dest._cbit_argument_conversion(clbits)
            if len(mapped_clbits) != other.num_clbits:
                raise CircuitError(
                    f"Number of items in clbits parameter ({len(mapped_clbits)}) does not"
                    f" match number of clbits in the circuit ({other.num_clbits})."
                )
            if len(set(mapped_clbits)) != len(mapped_clbits):
                raise CircuitError(
                    f"Duplicate clbits referenced in 'clbits' parameter: '{mapped_clbits}'"
                )
            edge_map.update(zip(other.clbits, mapped_clbits))

        dest.duration = None
        dest.unit = "dt"
        dest.global_phase += other.global_phase

        # This is required to trigger data builds if the `other` is an unbuilt `BlueprintCircuit`,
        # so we can the access the complete `CircuitData` object at `_data`.
        _ = other.data

        def copy_with_remapping(
            source, dest, bit_map, var_map, inline_captures, new_qubits=None, new_clbits=None
        ):
            # Copy the instructions from `source` into `dest`, remapping variables in instructions
            # according to `var_map`.  If `new_qubits` or `new_clbits` are given, the qubits and
            # clbits of the source instruction are remapped to those as well.
            for var in source.iter_input_vars():
                dest.add_input(replace_var(var, var_map))
            if inline_captures:
                for var in source.iter_captures():
                    replacement = replace_var(var, var_map)
                    if not dest.has_identifier(replace_var(var, var_map)):
                        if var is replacement:
                            raise CircuitError(
                                f"Variable '{var}' to be inlined is not in the base circuit."
                                " If you wanted it to be automatically added, use"
                                " `inline_captures=False`."
                            )
                        raise CircuitError(
                            f"Replacement '{replacement}' for variable '{var}' is not in the"
                            " base circuit.  Is the replacement correct?"
                        )
            else:
                for var in source.iter_captures():
                    dest.add_capture(replace_var(var, var_map))
            for var in source.iter_declared_vars():
                dest.add_uninitialized_var(replace_var(var, var_map))
            for stretch in source.iter_declared_stretches():
                dest.add_stretch(replace_var(stretch, var_map))

            def recurse_block(block):
                # Recurse the remapping into a control-flow block.  Note that this doesn't remap the
                # clbits within; the story around nested classical-register-based control-flow
                # doesn't really work in the current data model, and we hope to replace it with
                # `Expr`-based control-flow everywhere.
                new_block = block.copy_empty_like(vars_mode="drop")
                # For the recursion, we never want to inline captured variables because we're not
                # copying onto a base that has variables.
                copy_with_remapping(block, new_block, bit_map, var_map, inline_captures=False)
                return new_block

            variable_mapper = _classical_resource_map.VariableMapper(
                dest.cregs, bit_map, var_map, add_register=dest.add_register
            )

            def map_vars(op):
                n_op = op
                if isinstance(n_op, ControlFlowOp):
                    n_op = n_op.replace_blocks(recurse_block(block) for block in n_op.blocks)
                    if isinstance(n_op, (IfElseOp, WhileLoopOp)):
                        n_op.condition = variable_mapper.map_condition(n_op._condition)
                    elif isinstance(n_op, SwitchCaseOp):
                        n_op.target = variable_mapper.map_target(n_op.target)
                elif isinstance(n_op, Store):
                    n_op = Store(
                        variable_mapper.map_expr(n_op.lvalue), variable_mapper.map_expr(n_op.rvalue)
                    )
                elif isinstance(n_op, Delay) and n_op.unit == "expr":
                    n_op = n_op.copy()
                    n_op.duration = variable_mapper.map_expr(n_op.duration)
                return n_op.copy() if n_op is op and copy else n_op

            instructions = source._data.copy(copy_instructions=copy)
            instructions.map_nonstandard_ops(map_vars)
            dest._current_scope().extend(instructions, qubits=new_qubits, clbits=new_clbits)

        append_existing = None
        if front:
            append_existing = dest._data.copy(copy_instructions=copy)
            dest.clear()
        copy_with_remapping(
            other,
            dest,
            bit_map=edge_map,
            # The actual `Var: Var` map gets built up from the more freeform user input as we
            # encounter the variables, since the user might be using string keys to refer to more
            # than one variable in separated scopes of control-flow operations.
            var_map={},
            inline_captures=inline_captures,
            new_qubits=mapped_qubits,
            new_clbits=mapped_clbits,
        )
        if append_existing:
            dest._current_scope().extend(append_existing)

        return None if inplace else dest

    def tensor(self, other: QuantumCircuit, inplace: bool = False) -> QuantumCircuit | None:
        """Tensor ``self`` with ``other``.

        Remember that in the little-endian convention the leftmost operation will be at the bottom
        of the circuit. See also
        `the docs <https://quantum.cloud.ibm.com/docs/guides/construct-circuits>`__
        for more information.

        .. code-block:: text

                 ┌────────┐        ┌─────┐          ┌─────┐
            q_0: ┤ bottom ├ ⊗ q_0: ┤ top ├  = q_0: ─┤ top ├──
                 └────────┘        └─────┘         ┌┴─────┴─┐
                                              q_1: ┤ bottom ├
                                                   └────────┘

        Args:
            other (QuantumCircuit): The other circuit to tensor this circuit with.
            inplace (bool): If ``True``, modify the object. Otherwise return composed circuit.

        Examples:

            .. plot::
               :alt: Circuit diagram output by the previous code.
               :include-source:

               from qiskit import QuantumCircuit
               top = QuantumCircuit(1)
               top.x(0);
               bottom = QuantumCircuit(2)
               bottom.cry(0.2, 0, 1);
               tensored = bottom.tensor(top)
               tensored.draw('mpl')

        Returns:
            QuantumCircuit: The tensored circuit (returns ``None`` if ``inplace=True``).
        """
        num_qubits = self.num_qubits + other.num_qubits
        num_clbits = self.num_clbits + other.num_clbits

        # If a user defined both circuits with via register sizes and not with named registers
        # (e.g. QuantumCircuit(2, 2)) then we have a naming collision, as the registers are by
        # default called "q" resp. "c". To still allow tensoring we define new registers of the
        # correct sizes.
        if (
            len(self.qregs) == len(other.qregs) == 1
            and self.qregs[0].name == other.qregs[0].name == "q"
        ):
            # check if classical registers are in the circuit
            if num_clbits > 0:
                dest = QuantumCircuit(num_qubits, num_clbits)
            else:
                dest = QuantumCircuit(num_qubits)

        # handle case if ``measure_all`` was called on both circuits, in which case the
        # registers are both named "meas"
        elif (
            len(self.cregs) == len(other.cregs) == 1
            and self.cregs[0].name == other.cregs[0].name == "meas"
        ):
            cr = ClassicalRegister(self.num_clbits + other.num_clbits, "meas")
            dest = QuantumCircuit(*other.qregs, *self.qregs, cr)

        # Now we don't have to handle any more cases arising from special implicit naming
        else:
            dest = QuantumCircuit(
                other.qubits,
                self.qubits,
                other.clbits,
                self.clbits,
                *other.qregs,
                *self.qregs,
                *other.cregs,
                *self.cregs,
            )

        # compose self onto the output, and then other
        dest.compose(other, range(other.num_qubits), range(other.num_clbits), inplace=True)
        dest.compose(
            self,
            range(other.num_qubits, num_qubits),
            range(other.num_clbits, num_clbits),
            inplace=True,
        )

        # Replace information from tensored circuit into self when inplace = True
        if inplace:
            self.__dict__.update(dest.__dict__)
            return None
        return dest

    @property
    def _clbit_indices(self) -> dict[Clbit, BitLocations]:
        """Dict mapping Clbit instances to an object comprised of `.index` the
        corresponding index in circuit. and `.registers` a list of
        Register-int pairs for each Register containing the Bit and its index
        within that register."""
        return self._data._clbit_indices

    @property
    def _qubit_indices(self) -> dict[Qubit, BitLocations]:
        """Dict mapping Qubit instances to an object comprised of `.index` the
        corresponding index in circuit. and `.registers` a list of
        Register-int pairs for each Register containing the Bit and its index
        within that register."""
        return self._data._qubit_indices

    @property
    def qubits(self) -> list[Qubit]:
        """A list of :class:`Qubit`\\ s in the order that they were added.  You should not mutate
        this."""
        return self._data.qubits

    @property
    def clbits(self) -> list[Clbit]:
        """A list of :class:`Clbit`\\ s in the order that they were added.  You should not mutate
        this.

        Example:

            .. plot::
                :include-source:
                :nofigs:
                :context: reset

                from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

                qr1 = QuantumRegister(2)
                qr2 = QuantumRegister(1)
                cr1 = ClassicalRegister(2)
                cr2 = ClassicalRegister(1)
                qc = QuantumCircuit(qr1, qr2, cr1, cr2)

                print("List the qubits in this circuit:", qc.qubits)
                print("List the classical bits in this circuit:", qc.clbits)

            .. code-block:: text

                List the qubits in this circuit: [Qubit(QuantumRegister(2, 'q0'), 0),
                Qubit(QuantumRegister(2, 'q0'), 1), Qubit(QuantumRegister(1, 'q1'), 0)]
                List the classical bits in this circuit: [Clbit(ClassicalRegister(2, 'c0'), 0),
                Clbit(ClassicalRegister(2, 'c0'), 1), Clbit(ClassicalRegister(1, 'c1'), 0)]
        """
        return self._data.clbits

    @property
    def qregs(self) -> list[QuantumRegister]:
        """A list of :class:`Qubit`\\ s in the order that they were added.  You should not mutate
        this."""
        return self._data.qregs

    @qregs.setter
    def qregs(self, other: list[QuantumRegister]):
        self._data.qregs = other

    @property
    def cregs(self) -> list[ClassicalRegister]:
        """A list of :class:`Clbit`\\ s in the order that they were added.  You should not mutate
        this."""
        return self._data.cregs

    @cregs.setter
    def cregs(self, other: list[ClassicalRegister]):
        self._data.cregs = other

    @property
    def ancillas(self) -> list[AncillaQubit]:
        """A list of :class:`AncillaQubit`\\ s in the order that they were added.  You should not
        mutate this."""
        return self._ancillas

    @property
    def num_vars(self) -> int:
        """The number of real-time classical variables in the circuit.

        This is the length of the :meth:`iter_vars` iterable."""
        return self.num_input_vars + self.num_captured_vars + self.num_declared_vars

    @property
    def num_stretches(self) -> int:
        """The number of stretches in the circuit.

        This is the length of the :meth:`iter_stretches` iterable."""
        return self.num_captured_stretches + self.num_declared_stretches

    @property
    def num_identifiers(self) -> int:
        """The number of real-time classical variables and stretches in
        the circuit.

        This is equal to :meth:`num_vars` + :meth:`num_stretches`.
        """
        return self.num_vars + self.num_stretches

    @property
    def num_input_vars(self) -> int:
        """The number of real-time classical variables in the circuit marked as circuit inputs.

        This is the length of the :meth:`iter_input_vars` iterable.  If this is non-zero,
        :attr:`num_captured_vars` must be zero."""
        return self._data.num_input_vars

    @property
    def num_captured_vars(self) -> int:
        """The number of real-time classical variables in the circuit marked as captured from an
        enclosing scope.

        This is the length of the :meth:`iter_captured_vars` iterable.  If this is non-zero,
        :attr:`num_input_vars` must be zero."""
        return self._data.num_captured_vars

    @property
    def num_captured_stretches(self) -> int:
        """The number of stretches in the circuit marked as captured from an
        enclosing scope.

        This is the length of the :meth:`iter_captured_stretches` iterable.  If this is non-zero,
        :attr:`num_input_vars` must be zero."""
        return self._data.num_captured_stretches

    @property
    def num_declared_vars(self) -> int:
        """The number of real-time classical variables in the circuit that are declared by this
        circuit scope, excluding inputs or captures.

        This is the length of the :meth:`iter_declared_vars` iterable."""
        return self._data.num_declared_vars

    @property
    def num_declared_stretches(self) -> int:
        """The number of stretches in the circuit that are declared by this
        circuit scope, excluding captures.

        This is the length of the :meth:`iter_declared_stretches` iterable."""
        return self._data.num_declared_stretches

    def iter_vars(self) -> typing.Iterable[expr.Var]:
        """Get an iterable over all real-time classical variables in scope within this circuit.

        This method will iterate over all variables in scope.  For more fine-grained iterators, see
        :meth:`iter_declared_vars`, :meth:`iter_input_vars` and :meth:`iter_captured_vars`."""
        if self._control_flow_scopes:
            builder = self._control_flow_scopes[-1]
            return itertools.chain(builder.iter_captured_vars(), builder.iter_local_vars())
        return itertools.chain(
            self._data.get_input_vars(),
            self._data.get_captured_vars(),
            self._data.get_declared_vars(),
        )

    def iter_stretches(self) -> typing.Iterable[expr.Stretch]:
        """Get an iterable over all stretches in scope within this circuit.

        This method will iterate over all stretches in scope.  For more fine-grained iterators, see
        :meth:`iter_declared_stretches` and :meth:`iter_captured_stretches`."""
        if self._control_flow_scopes:
            builder = self._control_flow_scopes[-1]
            return itertools.chain(
                builder.iter_captured_stretches(), builder.iter_local_stretches()
            )
        return itertools.chain(
            self._data.get_captured_stretches(), self._data.get_declared_stretches()
        )

    def iter_declared_vars(self) -> typing.Iterable[expr.Var]:
        """Get an iterable over all real-time classical variables that are declared with automatic
        storage duration in this scope.  This excludes input variables (see :meth:`iter_input_vars`)
        and captured variables (see :meth:`iter_captured_vars`)."""
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].iter_local_vars()
        return self._data.get_declared_vars()

    def iter_declared_stretches(self) -> typing.Iterable[expr.Stretch]:
        """Get an iterable over all stretches that are declared in this scope.
        This excludes captured stretches (see :meth:`iter_captured_stretches`)."""
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].iter_local_stretches()
        return self._data.get_declared_stretches()

    def iter_input_vars(self) -> typing.Iterable[expr.Var]:
        """Get an iterable over all real-time classical variables that are declared as inputs to
        this circuit scope.  This excludes locally declared variables (see
        :meth:`iter_declared_vars`) and captured variables (see :meth:`iter_captured_vars`)."""
        if self._control_flow_scopes:
            return ()
        return self._data.get_input_vars()

    def iter_captures(self) -> typing.Iterable[expr.Var | expr.Stretch]:
        """Get an iterable over all identifiers are captured by this circuit scope from a
        containing scope.  This excludes input variables (see :meth:`iter_input_vars`)
        and locally declared variables and stretches (see :meth:`iter_declared_vars`
        and :meth:`iter_declared_stretches`)."""
        if self._control_flow_scopes:
            return itertools.chain(
                self._control_flow_scopes[-1].iter_captured_vars(),
                self._control_flow_scopes[-1].iter_captured_stretches(),
            )
        return itertools.chain(self._data.get_captured_vars(), self._data.get_captured_stretches())

    def iter_captured_vars(self) -> typing.Iterable[expr.Var]:
        """Get an iterable over all real-time classical variables that are captured by this circuit
        scope from a containing scope.  This excludes input variables (see :meth:`iter_input_vars`)
        and locally declared variables (see :meth:`iter_declared_vars`)."""
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].iter_captured_vars()
        return self._data.get_captured_vars()

    def iter_captured_stretches(self) -> typing.Iterable[expr.Stretch]:
        """Get an iterable over stretches that are captured by this circuit
        scope from a containing scope.  This excludes locally declared stretches
        (see :meth:`iter_declared_stretches`)."""
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].iter_captured_stretches()
        return self._data.get_captured_stretches()

    def __and__(self, rhs: QuantumCircuit) -> QuantumCircuit:
        """Overload & to implement self.compose."""
        return self.compose(rhs)

    def __iand__(self, rhs: QuantumCircuit) -> QuantumCircuit:
        """Overload &= to implement self.compose in place."""
        self.compose(rhs, inplace=True)
        return self

    def __xor__(self, top: QuantumCircuit) -> QuantumCircuit:
        """Overload ^ to implement self.tensor."""
        return self.tensor(top)

    def __ixor__(self, top: QuantumCircuit) -> QuantumCircuit:
        """Overload ^= to implement self.tensor in place."""
        self.tensor(top, inplace=True)
        return self

    def __len__(self) -> int:
        """Return number of operations in circuit."""
        return len(self._data)

    @typing.overload
    def __getitem__(self, item: int) -> CircuitInstruction: ...

    @typing.overload
    def __getitem__(self, item: slice) -> list[CircuitInstruction]: ...

    def __getitem__(self, item):
        """Return indexed operation."""
        return self._data[item]

    @staticmethod
    def _cast(value: S, type_: Callable[..., T]) -> S | T:
        """Best effort to cast value to type. Otherwise, returns the value."""
        try:
            return type_(value)
        except (ValueError, TypeError):
            return value

    def _qbit_argument_conversion(self, qubit_representation: QubitSpecifier) -> list[Qubit]:
        """
        Converts several qubit representations (such as indexes, range, etc.)
        into a list of qubits.

        Args:
            qubit_representation: Representation to expand.

        Returns:
            The resolved instances of the qubits.
        """
        return self._data._qbit_argument_conversion(qubit_representation)

    def _cbit_argument_conversion(self, clbit_representation: ClbitSpecifier) -> list[Clbit]:
        """
        Converts several classical bit representations (such as indexes, range, etc.)
        into a list of classical bits.

        Args:
            clbit_representation: Representation to expand.

        Returns:
            A list of tuples where each tuple is a classical bit.
        """
        return self._data._cbit_argument_conversion(clbit_representation)

    def _append_standard_gate(
        self,
        op: StandardGate,
        qargs: Sequence[QubitSpecifier] = (),
        params: Sequence[ParameterValueType] = (),
        label: str | None = None,
    ) -> InstructionSet:
        """An internal method to bypass some checking when directly appending a standard gate."""
        circuit_scope = self._current_scope()

        if params is None:
            params = []

        expanded_qargs = [self._qbit_argument_conversion(qarg) for qarg in qargs or []]
        for param in params:
            Gate.validate_parameter(op, param)

        instructions = InstructionSet(resource_requester=circuit_scope.resolve_classical_resource)
        for qarg, _ in Gate.broadcast_arguments(op, expanded_qargs, []):
            self._check_dups(qarg)
            instruction = CircuitInstruction.from_standard(op, qarg, params, label=label)
            circuit_scope.append(instruction, _standard_gate=True)
            instructions._add_ref(circuit_scope.instructions, len(circuit_scope.instructions) - 1)
        return instructions

    def append(
        self,
        instruction: Operation | CircuitInstruction,
        qargs: Sequence[QubitSpecifier] | None = None,
        cargs: Sequence[ClbitSpecifier] | None = None,
        *,
        copy: bool = True,
    ) -> InstructionSet:
        """Append one or more instructions to the end of the circuit, modifying the circuit in
        place.

        The ``qargs`` and ``cargs`` will be expanded and broadcast according to the rules of the
        given :class:`~.circuit.Instruction`, and any non-:class:`.Bit` specifiers (such as
        integer indices) will be resolved into the relevant instances.

        If a :class:`.CircuitInstruction` is given, it will be unwrapped, verified in the context of
        this circuit, and a new object will be appended to the circuit.  In this case, you may not
        pass ``qargs`` or ``cargs`` separately.

        Args:
            instruction: :class:`~.circuit.Instruction` instance to append, or a
                :class:`.CircuitInstruction` with all its context. Objects implementing
                ``to_instruction`` are also supported, but passing an
                :class:`~.circuit.Instruction` directly is generally preferred, since that
                avoids the repeated conversion cost.
            qargs: specifiers of the :class:`~.circuit.Qubit`\\ s to attach instruction to.
            cargs: specifiers of the :class:`.Clbit`\\ s to attach instruction to.
            copy: if ``True`` (the default), then the incoming ``instruction`` is copied before
                adding it to the circuit if it contains symbolic parameters, so it can be safely
                mutated without affecting other circuits the same instruction might be in.  If you
                are sure this instruction will not be in other circuits, you can set this ``False``
                for a small speedup.

        Returns:
            qiskit.circuit.InstructionSet: a handle to the :class:`.CircuitInstruction`\\ s that
            were actually added to the circuit.

        Raises:
            CircuitError: if the operation passed is not an instance of :class:`~.circuit.Instruction`,
              or cannot be converted to one by calling ``to_instruction`` on it.
        """
        if isinstance(instruction, CircuitInstruction):
            operation = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
        else:
            operation = instruction

        # Convert input to instruction
        if not isinstance(operation, Operation):
            if hasattr(operation, "to_instruction"):
                operation = operation.to_instruction()
                if not isinstance(operation, Operation):
                    raise CircuitError("operation.to_instruction() is not an Operation.")
            else:
                if issubclass(operation, Operation):
                    raise CircuitError(
                        "Object is a subclass of Operation, please add () to "
                        "pass an instance of this object."
                    )

                raise CircuitError(
                    "Object to append must be an Operation or have a to_instruction() method."
                )

        circuit_scope = self._current_scope()

        # Make copy of parameterized gate instances
        if params := getattr(operation, "params", ()):
            is_parameter = False
            for param in params:
                is_parameter = is_parameter or isinstance(param, ParameterExpression)
                # Control flow ops have a different validation for classical expressions
                if isinstance(param, expr.Expr) and not isinstance(operation, ControlFlowOp):
                    param = _validate_expr(circuit_scope, param)
            if copy and is_parameter:
                operation = _copy.deepcopy(operation)
        if isinstance(operation, ControlFlowOp):
            if operation.name == "box" and operation.unit == "expr":
                _validate_expr(circuit_scope, operation.duration)
            # Verify that any variable bindings are valid.  Control-flow ops are already enforced
            # by the class not to contain 'input' variables.
            if bad_captures := {
                var
                for var in itertools.chain.from_iterable(
                    block.iter_captures() for block in operation.blocks
                )
                if not self.has_identifier(var)
            }:
                raise CircuitError(
                    f"Control-flow op attempts to capture '{bad_captures}'"
                    " which are not in this circuit"
                )

        expanded_qargs = [self._qbit_argument_conversion(qarg) for qarg in qargs or []]
        expanded_cargs = [self._cbit_argument_conversion(carg) for carg in cargs or []]

        instructions = InstructionSet(resource_requester=circuit_scope.resolve_classical_resource)
        # For Operations that are non-Instructions, we use the Instruction's default method
        broadcast_iter = (
            operation.broadcast_arguments(expanded_qargs, expanded_cargs)
            if isinstance(operation, Instruction)
            else Instruction.broadcast_arguments(operation, expanded_qargs, expanded_cargs)
        )
        base_instruction = CircuitInstruction(operation, (), ())
        for qarg, carg in broadcast_iter:
            self._check_dups(qarg)
            self._check_dups(carg)
            instruction = base_instruction.replace(qubits=qarg, clbits=carg)
            circuit_scope.append(instruction)
            instructions._add_ref(circuit_scope.instructions, len(circuit_scope.instructions) - 1)
        return instructions

    # Preferred new style.
    @typing.overload
    def _append(
        self, instruction: CircuitInstruction, *, _standard_gate: bool
    ) -> CircuitInstruction: ...

    # To-be-deprecated old style.
    @typing.overload
    def _append(
        self,
        instruction: Operation,
        qargs: Sequence[Qubit],
        cargs: Sequence[Clbit],
    ) -> Operation: ...

    def _append(self, instruction, qargs=(), cargs=(), *, _standard_gate: bool = False):
        """Append an instruction to the end of the circuit, modifying the circuit in place.

        .. warning::

            This is an internal fast-path function, and it is the responsibility of the caller to
            ensure that all the arguments are valid; there is no error checking here.  In
            particular:

            * all the qubits and clbits must already exist in the circuit and there can be no
              duplicates in the list.
            * any control-flow operations instructions must act only on variables present in the
              circuit.
            * the circuit must not be within a control-flow builder context.

        .. note::

            This function may be used by callers other than :obj:`.QuantumCircuit` when the caller
            is sure that all error-checking, broadcasting and scoping has already been performed,
            and the only reference to the circuit the instructions are being appended to is within
            that same function.  In particular, it is not safe to call
            :meth:`QuantumCircuit._append` on a circuit that is received by a function argument.
            This is because :meth:`.QuantumCircuit._append` will not recognize the scoping
            constructs of the control-flow builder interface.

        Args:
            instruction: A complete well-formed :class:`.CircuitInstruction` of the operation and
                its context to be added.

                In the legacy compatibility form, this can be a bare :class:`.Operation`, in which
                case ``qargs`` and ``cargs`` must be explicitly given.
            qargs: Legacy argument for qubits to attach the bare :class:`.Operation` to.  Ignored if
                the first argument is in the preferential :class:`.CircuitInstruction` form.
            cargs: Legacy argument for clbits to attach the bare :class:`.Operation` to.  Ignored if
                the first argument is in the preferential :class:`.CircuitInstruction` form.

        Returns:
            CircuitInstruction: a handle to the instruction that was just added.

        :meta public:
        """
        if _standard_gate:
            self._data.append(instruction)
            self._duration = None
            self._unit = "dt"
            return instruction

        old_style = not isinstance(instruction, CircuitInstruction)
        if old_style:
            instruction = CircuitInstruction(instruction, qargs, cargs)
        # If there is a reference to the outer circuit in an
        # instruction param the inner rust append method will raise a runtime error.
        # When this happens we need to handle the parameters separately.
        # This shouldn't happen in practice but 2 tests were doing this and it's not
        # explicitly prohibited by the API.
        try:
            self._data.append(instruction)
        except RuntimeError:
            params = [
                (idx, param.parameters)
                for idx, param in enumerate(instruction.operation.params)
                if isinstance(param, (ParameterExpression, QuantumCircuit))
            ]
            self._data.append_manual_params(instruction, params)

        # Invalidate whole circuit duration if an instruction is added
        self._duration = None
        self._unit = "dt"
        return instruction.operation if old_style else instruction

    @typing.overload
    def get_parameter(self, name: str, default: T) -> Parameter | T: ...
    @typing.overload
    def get_parameter(self, name: str, default: builtin_types.EllipsisType = ...) -> Parameter: ...

    # We use a _literal_ `Ellipsis` as the marker value to leave `None` available as a default.
    def get_parameter(self, name: str, default: typing.Any = ...) -> Parameter:
        """Retrieve a compile-time parameter that is accessible in this circuit scope by name.

        Args:
            name: the name of the parameter to retrieve.
            default: if given, this value will be returned if the parameter is not present.  If it
                is not given, a :exc:`KeyError` is raised instead.

        Returns:
            The corresponding parameter.

        Raises:
            KeyError: if no default is given, but the parameter does not exist in the circuit.

        Examples:
            Retrieve a parameter by name from a circuit::

                from qiskit.circuit import QuantumCircuit, Parameter

                my_param = Parameter("my_param")

                # Create a parametrized circuit.
                qc = QuantumCircuit(1)
                qc.rx(my_param, 0)

                # We can use 'my_param' as a parameter, but let's say we've lost the Python object
                # and need to retrieve it.
                my_param_again = qc.get_parameter("my_param")

                assert my_param == my_param_again

            Get a variable from a circuit by name, returning some default if it is not present::

                assert qc.get_parameter("my_param", None) == my_param
                assert qc.get_parameter("unknown_param", None) is None

        See also:
            :meth:`get_var`
                A similar method, but for :class:`.expr.Var` run-time variables instead of
                :class:`.Parameter` compile-time parameters.
        """
        if (parameter := self._data.get_parameter_by_name(name)) is None:
            if default is Ellipsis:
                raise KeyError(f"no parameter named '{name}' is present")
            return default
        return parameter

    def has_parameter(self, name_or_param: str | Parameter, /) -> bool:
        """Check whether a parameter object exists in this circuit.

        Args:
            name_or_param: the parameter, or name of a parameter to check.  If this is a
                :class:`.Parameter` node, the parameter must be exactly the given one for this
                function to return ``True``.

        Returns:
            whether a matching parameter is assignable in this circuit.

        See also:
            :meth:`QuantumCircuit.get_parameter`
                Retrieve the :class:`.Parameter` instance from this circuit by name.
            :meth:`QuantumCircuit.has_var`
                A similar method to this, but for run-time :class:`.expr.Var` variables instead of
                compile-time :class:`.Parameter`\\ s.
        """
        if isinstance(name_or_param, str):
            return self.get_parameter(name_or_param, None) is not None
        return (
            isinstance(name_or_param, Parameter)
            and self.get_parameter(name_or_param.name, None) == name_or_param
        )

    @typing.overload
    def get_var(self, name: str, default: T) -> expr.Var | T: ...
    @typing.overload
    def get_var(self, name: str, default: builtin_types.EllipsisType = ...) -> expr.Var: ...

    # We use a _literal_ `Ellipsis` as the marker value to leave `None` available as a default.
    def get_var(self, name: str, default: typing.Any = ...):
        """Retrieve a variable that is accessible in this circuit scope by name.

        Args:
            name: the name of the variable to retrieve.
            default: if given, this value will be returned if the variable is not present.  If it
                is not given, a :exc:`KeyError` is raised instead.

        Returns:
            The corresponding variable.

        Raises:
            KeyError: if no default is given, but the variable does not exist.

        Examples:
            Retrieve a variable by name from a circuit::

                from qiskit.circuit import QuantumCircuit

                # Create a circuit and create a variable in it.
                qc = QuantumCircuit()
                my_var = qc.add_var("my_var", False)

                # We can use 'my_var' as a variable, but let's say we've lost the Python object and
                # need to retrieve it.
                my_var_again = qc.get_var("my_var")

                assert my_var == my_var_again

            Get a variable from a circuit by name, returning some default if it is not present::

                assert qc.get_var("my_var", None) == my_var
                assert qc.get_var("unknown_variable", None) is None

        See also:
            :meth:`get_parameter`
                A similar method, but for :class:`.Parameter` compile-time parameters instead of
                :class:`.expr.Var` run-time variables.
        """
        if (out := self._current_scope().get_var(name)) is not None:
            return out
        if default is Ellipsis:
            raise KeyError(f"no variable named '{name}' is present")
        return default

    def has_var(self, name_or_var: str | expr.Var, /) -> bool:
        """Check whether a variable is accessible in this scope.

        Args:
            name_or_var: the variable, or name of a variable to check.  If this is a
                :class:`.expr.Var` node, the variable must be exactly the given one for this
                function to return ``True``.

        Returns:
            whether a matching variable is accessible.

        See also:
            :meth:`QuantumCircuit.get_var`
                Retrieve the :class:`.expr.Var` instance from this circuit by name.
            :meth:`QuantumCircuit.has_parameter`
                A similar method to this, but for compile-time :class:`.Parameter`\\ s instead of
                run-time :class:`.expr.Var` variables.
        """
        if isinstance(name_or_var, str):
            return self.get_var(name_or_var, None) is not None
        return self.get_var(name_or_var.name, None) == name_or_var

    @typing.overload
    def get_stretch(self, name: str, default: T) -> expr.Stretch | T: ...
    @typing.overload
    def get_stretch(self, name: str, default: builtin_types.EllipsisType = ...) -> expr.Stretch: ...

    def get_stretch(self, name: str, default: typing.Any = ...):
        """Retrieve a stretch that is accessible in this circuit scope by name.

        Args:
            name: the name of the stretch to retrieve.
            default: if given, this value will be returned if the variable is not present.  If it
                is not given, a :exc:`KeyError` is raised instead.

        Returns:
            The corresponding stretch.

        Raises:
            KeyError: if no default is given, but the variable does not exist.

        Examples:
            Retrieve a stretch by name from a circuit::

                from qiskit.circuit import QuantumCircuit

                # Create a circuit and create a variable in it.
                qc = QuantumCircuit()
                my_stretch = qc.add_stretch("my_stretch")

                # We can use 'my_stretch' as a variable, but let's say we've lost the Python object and
                # need to retrieve it.
                my_stretch_again = qc.get_stretch("my_stretch")

                assert my_stretch == my_stretch_again

            Get a variable from a circuit by name, returning some default if it is not present::

                assert qc.get_stretch("my_stretch", None) == my_stretch
                assert qc.get_stretch("unknown_stretch", None) is None
        """
        if (out := self._current_scope().get_stretch(name)) is not None:
            return out
        if default is Ellipsis:
            raise KeyError(f"no stretch named '{name}' is present")
        return default

    def has_stretch(self, name_or_stretch: str | expr.Stretch, /) -> bool:
        """Check whether a stretch is accessible in this scope.

        Args:
            name_or_stretch: the stretch, or name of a stretch to check.  If this is a
                :class:`.expr.Stretch` node, the stretch must be exactly the given one for this
                function to return ``True``.

        Returns:
            whether a matching stretch is accessible.

        See also:
            :meth:`QuantumCircuit.get_stretch`
                Retrieve the :class:`.expr.Stretch` instance from this circuit by name.
        """
        if isinstance(name_or_stretch, str):
            return self.get_stretch(name_or_stretch, None) is not None
        return self.get_stretch(name_or_stretch.name, None) == name_or_stretch

    @typing.overload
    def get_identifier(self, name: str, default: T) -> expr.Var | expr.Stretch | T: ...
    @typing.overload
    def get_identifier(
        self, name: str, default: builtin_types.EllipsisType = ...
    ) -> expr.Var | expr.Stretch: ...

    # We use a _literal_ `Ellipsis` as the marker value to leave `None` available as a default.
    def get_identifier(self, name: str, default: typing.Any = ...):
        """Retrieve an identifier that is accessible in this circuit scope by name.

        This currently includes both real-time classical variables and stretches.

        Args:
            name: the name of the identifier to retrieve.
            default: if given, this value will be returned if the variable is not present.  If it
                is not given, a :exc:`KeyError` is raised instead.

        Returns:
            The corresponding variable.

        Raises:
            KeyError: if no default is given, but the identifier does not exist.

        See also:
            :meth:`get_var`
                Gets an identifier known to be a :class:`.expr.Var` instance.
            :meth:`get_stretch`
                Gets an identifier known to be a :class:`.expr.Stretch` instance.
            :meth:`get_parameter`
                A similar method, but for :class:`.Parameter` compile-time parameters instead of
                :class:`.expr.Var` run-time variables.
        """
        if (out := self._current_scope().get_var(name)) is not None:
            return out
        if (out := self._current_scope().get_stretch(name)) is not None:
            return out
        if default is Ellipsis:
            raise KeyError(f"no identifier named '{name}' is present")
        return default

    def has_identifier(self, name_or_ident: str | expr.Var | expr.Stretch, /) -> bool:
        """Check whether an identifier is accessible in this scope.

        Args:
            name_or_ident: the instance, or name of the identifier to check.  If this is a
                :class:`.expr.Var` or :class:`.expr.Stretch` node, the matched instance must
                be exactly the given one for this function to return ``True``.

        Returns:
            whether a matching identifier is accessible.

        See also:
            :meth:`QuantumCircuit.get_identifier`
                Retrieve the :class:`.expr.Var` or :class:`.expr.Stretch` instance from this
                circuit by name.
            :meth:`QuantumCircuit.has_var`
                The same as this method, but ignoring anything that isn't a
                run-time :class:`expr.Var` variable.
            :meth:`QuantumCircuit.has_stretch`
                The same as this method, but ignoring anything that isn't a
                run-time :class:`expr.Stretch` variable.
            :meth:`QuantumCircuit.has_parameter`
                A similar method to this, but for compile-time :class:`.Parameter`\\ s instead of
                run-time :class:`.expr.Var` variables.
        """
        if isinstance(name_or_ident, str):
            return self.get_identifier(name_or_ident, None) is not None
        return self.get_identifier(name_or_ident.name, None) == name_or_ident

    def _prepare_new_var(
        self, name_or_var: str | expr.Var, type_: types.Type | None, /
    ) -> expr.Var:
        """The common logic for preparing and validating a new :class:`~.expr.Var` for the circuit.

        The given ``type_`` can be ``None`` if the variable specifier is already a :class:`.Var`,
        and must be a :class:`~.types.Type` if it is a string.  The argument is ignored if the given
        first argument is a :class:`.Var` already.

        Returns the validated variable, which is guaranteed to be safe to add to the circuit."""
        if isinstance(name_or_var, str):
            if type_ is None:
                raise CircuitError("the type must be known when creating a 'Var' from a string")
            var = expr.Var.new(name_or_var, type_)
        else:
            var = name_or_var
            if not var.standalone:
                raise CircuitError(
                    "cannot add variables that wrap `Clbit` or `ClassicalRegister` instances."
                    " Use `add_bits` or `add_register` as appropriate."
                )

        # The `var` is guaranteed to have a name because we already excluded the cases where it's
        # wrapping a bit/register.
        if (previous := self.get_identifier(var.name, default=None)) is not None:
            if previous == var:
                raise CircuitError(f"'{var}' is already present in the circuit")
            raise CircuitError(f"cannot add '{var}' as its name shadows the existing '{previous}'")
        return var

    def _prepare_new_stretch(self, name_or_stretch: str | expr.Stretch, /) -> expr.Stretch:
        """The common logic for preparing and validating a new :class:`~.expr.Stretch` for the circuit.

        Returns the validated stretch, which is guaranteed to be safe to add to the circuit."""
        if isinstance(name_or_stretch, str):
            stretch = expr.Stretch.new(name_or_stretch)
        else:
            stretch = name_or_stretch

        if (previous := self.get_identifier(stretch.name, default=None)) is not None:
            if previous == stretch:
                raise CircuitError(f"'{stretch}' is already present in the circuit")
            raise CircuitError(
                f"cannot add '{stretch}' as its name shadows the existing '{previous}'"
            )
        return stretch

    def add_stretch(self, name_or_stretch: str | expr.Stretch) -> expr.Stretch:
        """Declares a new stretch scoped to this circuit.

        Args:
            name_or_stretch: either a string of the stretch name, or an existing instance of
                :class:`~.expr.Stretch` to re-use.  Stretches cannot shadow names that are already in
                use within the circuit.

        Returns:
            The created stretch.  If a :class:`~.expr.Stretch` instance was given, the exact same
            object will be returned.

        Raises:
            CircuitError: if the stretch cannot be created due to shadowing an existing
                identifier.

        Examples:
            Define and use a new stretch given just a name::

                from qiskit.circuit import QuantumCircuit, Duration
                from qiskit.circuit.classical import expr

                qc = QuantumCircuit(2)
                my_stretch = qc.add_stretch("my_stretch")

                qc.delay(expr.add(Duration.dt(200), my_stretch), 1)
        """
        if isinstance(name_or_stretch, str):
            stretch = expr.Stretch.new(name_or_stretch)
        else:
            stretch = name_or_stretch
        self._current_scope().add_stretch(stretch)
        return stretch

    def add_var(self, name_or_var: str | expr.Var, /, initial: typing.Any) -> expr.Var:
        """Add a classical variable with automatic storage and scope to this circuit.

        The variable is considered to have been "declared" at the beginning of the circuit, but it
        only becomes initialized at the point of the circuit that you call this method, so it can
        depend on variables defined before it.

        Args:
            name_or_var: either a string of the variable name, or an existing instance of
                :class:`~.expr.Var` to re-use.  Variables cannot shadow names that are already in
                use within the circuit.
            initial: the value to initialize this variable with.  If the first argument was given
                as a string name, the type of the resulting variable is inferred from the initial
                expression; to control this more manually, either use :meth:`.Var.new` to manually
                construct a new variable with the desired type, or use :func:`.expr.cast` to cast
                the initializer to the desired type.

                This must be either a :class:`~.expr.Expr` node, or a value that can be lifted to
                one using :class:`.expr.lift`.

        Returns:
            The created variable.  If a :class:`~.expr.Var` instance was given, the exact same
            object will be returned.

        Raises:
            CircuitError: if the variable cannot be created due to shadowing an existing identifier.

        Examples:
            Define a new variable given just a name and an initializer expression::

                from qiskit.circuit import QuantumCircuit

                qc = QuantumCircuit(2)
                my_var = qc.add_var("my_var", False)

            Reuse a variable that may have been taken from a related circuit, or otherwise
            constructed manually, and initialize it to some more complicated expression::

                from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
                from qiskit.circuit.classical import expr, types

                my_var = expr.Var.new("my_var", types.Uint(8))

                cr1 = ClassicalRegister(8, "cr1")
                cr2 = ClassicalRegister(8, "cr2")
                qc = QuantumCircuit(QuantumRegister(8), cr1, cr2)

                # Get some measurement results into each register.
                qc.h(0)
                for i in range(1, 8):
                    qc.cx(0, i)
                qc.measure(range(8), cr1)

                qc.reset(range(8))
                qc.h(0)
                for i in range(1, 8):
                    qc.cx(0, i)
                qc.measure(range(8), cr2)

                # Now when we add the variable, it is initialized using the real-time state of the
                # two classical registers we measured into above.
                qc.add_var(my_var, expr.bit_and(cr1, cr2))
        """
        # Validate the initializer first to catch cases where the variable to be declared is being
        # used in the initializer.
        circuit_scope = self._current_scope()
        # Convenience method to widen Python integer literals to the right width during the initial
        # lift, if the type is already known via the variable.
        if (
            isinstance(name_or_var, expr.Var)
            and name_or_var.type.kind is types.Uint
            and isinstance(initial, int)
            and not isinstance(initial, bool)
        ):
            coerce_type = name_or_var.type
        else:
            coerce_type = None
        initial = _validate_expr(circuit_scope, expr.lift(initial, coerce_type))
        if isinstance(name_or_var, str):
            var = expr.Var.new(name_or_var, initial.type)
        elif not name_or_var.standalone:
            raise CircuitError(
                "cannot add variables that wrap `Clbit` or `ClassicalRegister` instances."
            )
        else:
            var = name_or_var

        # Store is responsible for ensuring the type safety of the initialization.
        # This will raise a CircuitError if type safety cannot be ensured.
        store = Store(var, initial)
        circuit_scope.add_uninitialized_var(var)
        circuit_scope.append(CircuitInstruction(store, (), ()))
        return var

    def add_uninitialized_var(self, var: expr.Var, /):
        """Add a variable with no initializer.

        In most cases, you should use :meth:`add_var` to initialize the variable.  To use this
        function, you must already hold a :class:`~.expr.Var` instance, as the use of the function
        typically only makes sense in copying contexts.

        .. warning::

            Qiskit makes no assertions about what an uninitialized variable will evaluate to at
            runtime, and some hardware may reject this as an error.

            You should treat this function with caution, and as a low-level primitive that is useful
            only in special cases of programmatically rebuilding two like circuits.

        Args:
            var: the variable to add.
        """
        # This function is deliberately meant to be a bit harder to find, to have a long descriptive
        # name, and to be a bit less ergonomic than `add_var` (i.e. not allowing the (name, type)
        # overload) to discourage people from using it when they should use `add_var`.
        #
        # This function exists so that there is a method to emulate `copy_empty_like`'s behavior of
        # adding uninitialised variables, which there's no obvious way around.  We need to be sure
        # that _some_ sort of handling of uninitialised variables is taken into account in our
        # structures, so that doesn't become a huge edge case, even though we make no assertions
        # about the _meaning_ if such an expression was run on hardware.
        if self._control_flow_scopes:
            raise CircuitError("cannot add an uninitialized variable in a control-flow scope")
        if not var.standalone:
            raise CircuitError("cannot add a variable wrapping a bit or register to a circuit")
        self._builder_api.add_uninitialized_var(var)

    @typing.overload
    def add_capture(self, var: expr.Var): ...

    @typing.overload
    def add_capture(self, stretch: expr.Stretch): ...

    def add_capture(self, var):
        """Add an identifier to the circuit that it should capture from a scope it will
        be contained within.

        This method requires a :class:`~.expr.Var` or :class:`~.expr.Stretch` node to enforce that
        you've got a handle to an identifier, because you will need to declare the same identifier
        using the same object in the outer circuit.

        This is a low-level method, which is only really useful if you are manually constructing
        control-flow operations. You typically will not need to call this method, assuming you
        are using the builder interface for control-flow scopes (``with`` context-manager statements
        for :meth:`if_test` and the other scoping constructs).  The builder interface will
        automatically make the inner scopes closures on your behalf by capturing any identifiers
        that are used within them.

        Args:
            var (Union[expr.Var, expr.Stretch]): the variable or stretch to capture from an
                enclosing scope.

        Raises:
            CircuitError: if the identifier cannot be created due to shadowing an existing
                identifier.
        """
        if self._control_flow_scopes:
            # Allow manual capturing.  Not sure why it'd be useful, but there's a clear expected
            # behavior here.
            if isinstance(var, expr.Stretch):
                self._control_flow_scopes[-1].use_stretch(var)
            else:
                self._control_flow_scopes[-1].use_var(var)
            return
        if isinstance(var, expr.Stretch):
            self._data.add_captured_stretch(self._prepare_new_stretch(var))
        else:
            self._data.add_captured_var(self._prepare_new_var(var, None))

    @typing.overload
    def add_input(self, name_or_var: str, type_: types.Type, /) -> expr.Var: ...

    @typing.overload
    def add_input(self, name_or_var: expr.Var, type_: None = None, /) -> expr.Var: ...

    def add_input(
        self, name_or_var: str | expr.Var, type_: types.Type | None = None, /
    ) -> expr.Var:
        """Register a variable as an input to the circuit.

        Args:
            name_or_var: either a string name, or an existing :class:`~.expr.Var` node to use as the
                input variable.
            type_: if the name is given as a string, then this must be a :class:`~.types.Type` to
                use for the variable.  If the variable is given as an existing :class:`~.expr.Var`,
                then this must not be given, and will instead be read from the object itself.

        Returns:
            the variable created, or the same variable as was passed in.

        Raises:
            CircuitError: if the variable cannot be created due to shadowing an existing variable.
        """
        if self._control_flow_scopes:
            # ForLoopOp uses an input variable for its loop variable, but that should be exactly one
            # input variable and handled separately.
            raise CircuitError("cannot add an input variable in a control-flow scope")

        if isinstance(name_or_var, expr.Var) and type_ is not None:
            raise ValueError("cannot give an explicit type with an existing Var")
        var = self._prepare_new_var(name_or_var, type_)
        self._data.add_input_var(var)
        return var

    def add_register(self, *regs: Register | int | Sequence[Bit]) -> None:
        """Add registers.

        .. warning::

            If the quantum circuit has an existing :attr:`layout` attribute, adding a
            :class:`.QuantumRegister` will only increase the number of qubits. It will
            not update the layout.
        """
        if not regs:
            return

        if any(isinstance(reg, int) for reg in regs):
            # QuantumCircuit defined without registers
            if len(regs) == 1 and isinstance(regs[0], int):
                # QuantumCircuit with anonymous quantum wires e.g. QuantumCircuit(2)
                if regs[0] == 0:
                    regs = ()
                else:
                    regs = (QuantumRegister(regs[0], "q"),)
            elif len(regs) == 2 and all(isinstance(reg, int) for reg in regs):
                # QuantumCircuit with anonymous wires e.g. QuantumCircuit(2, 3)
                if regs[0] == 0:
                    qregs: tuple[QuantumRegister, ...] = ()
                else:
                    qregs = (QuantumRegister(regs[0], "q"),)
                if regs[1] == 0:
                    cregs: tuple[ClassicalRegister, ...] = ()
                else:
                    cregs = (ClassicalRegister(regs[1], "c"),)
                regs = qregs + cregs
            else:
                raise CircuitError(
                    "QuantumCircuit parameters can be Registers or Integers."
                    " If Integers, up to 2 arguments. QuantumCircuit was called"
                    f" with {(regs,)}."
                )

        for register in regs:
            if isinstance(register, Register) and any(
                register.name == reg.name for reg in self.qregs + self.cregs
            ):
                raise CircuitError(f'register name "{register.name}" already exists')

            if isinstance(register, AncillaRegister):
                for bit in register:
                    if bit not in self._qubit_indices:
                        self._ancillas.append(bit)

            if isinstance(register, QuantumRegister):
                if getattr(self, "layout", None) is not None:
                    warnings.warn(
                        "Trying to add QuantumRegister to a QuantumCircuit having a layout",
                        stacklevel=2,
                    )
                self._data.add_qreg(register)

            elif isinstance(register, ClassicalRegister):
                self._data.add_creg(register)

            elif isinstance(register, list):
                self.add_bits(register)
            else:
                raise CircuitError("expected a register")

    def add_bits(self, bits: Iterable[Bit]) -> None:
        """Add Bits to the circuit.

        .. warning::

            If the quantum circuit has an existing :attr:`layout` attribute,
            adding a :class:`.Qubit` will only increase the number of qubits.
            It will not update the layout.
        """
        duplicate_bits = {
            bit for bit in bits if bit in self._qubit_indices or bit in self._clbit_indices
        }
        if duplicate_bits:
            raise CircuitError(f"Attempted to add bits found already in circuit: {duplicate_bits}")

        for bit in bits:
            if isinstance(bit, AncillaQubit):
                self._ancillas.append(bit)
            if isinstance(bit, Qubit):
                if getattr(self, "layout", None) is not None:
                    warnings.warn(
                        "Trying to add bits to a QuantumCircuit having a layout",
                        stacklevel=2,
                    )
                self._data.add_qubit(bit)
            elif isinstance(bit, Clbit):
                self._data.add_clbit(bit)
            else:
                raise CircuitError(
                    "Expected an instance of Qubit, Clbit, or "
                    f"AncillaQubit, but was passed {bit}"
                )

    def find_bit(self, bit: Bit) -> BitLocations:
        """Find locations in the circuit which can be used to reference a given :obj:`~Bit`.

        In particular, this function can find the integer index of a qubit, which corresponds to its
        hardware index for a transpiled circuit.

        .. note::
            The circuit index of a :class:`.AncillaQubit` will be its index in :attr:`qubits`, not
            :attr:`ancillas`.

        Args:
            bit (Bit): The bit to locate.

        Returns:
            namedtuple(int, List[Tuple(Register, int)]): A 2-tuple. The first element (``index``)
            contains the index at which the ``Bit`` can be found (in either
            :obj:`~QuantumCircuit.qubits`, :obj:`~QuantumCircuit.clbits`, depending on its
            type). The second element (``registers``) is a list of ``(register, index)``
            pairs with an entry for each :obj:`~Register` in the circuit which contains the
            :obj:`~Bit` (and the index in the :obj:`~Register` at which it can be found).

        Raises:
            CircuitError: If the supplied :obj:`~Bit` was of an unknown type.
            CircuitError: If the supplied :obj:`~Bit` could not be found on the circuit.

        Examples:
            Loop through a circuit, getting the qubit and clbit indices of each operation::

                from qiskit.circuit import QuantumCircuit, Qubit

                qc = QuantumCircuit(3, 3)
                qc.h(0)
                qc.cx(0, 1)
                qc.cx(1, 2)
                qc.measure([0, 1, 2], [0, 1, 2])

                # The `.qubits` and `.clbits` fields are not integers.
                assert isinstance(qc.data[0].qubits[0], Qubit)
                # ... but we can use `find_bit` to retrieve them.
                assert qc.find_bit(qc.data[0].qubits[0]).index == 0

                simple = [
                    (
                        instruction.operation.name,
                        [qc.find_bit(bit).index for bit in instruction.qubits],
                        [qc.find_bit(bit).index for bit in instruction.clbits],
                    )
                    for instruction in qc.data
                ]
        """

        try:
            if isinstance(bit, Qubit):
                return self._data._qubit_indices[bit]
            elif isinstance(bit, Clbit):
                return self._data._clbit_indices[bit]
            else:
                raise CircuitError(f"Could not locate bit of unknown type: {type(bit)}")
        except KeyError as err:
            raise CircuitError(
                f"Could not locate provided bit: {bit}. Has it been added to the QuantumCircuit?"
            ) from err

    def _check_dups(self, bits: Sequence[Bit]) -> None:
        """Raise exception if list of bits contains duplicates."""
        match bits:
            case () | (_,):
                pass
            case (a, b):
                if a == b:
                    raise CircuitError("duplicate bit arguments")
            case bits:
                if len(bits) != len(set(bits)):
                    raise CircuitError("duplicate bit arguments")

    def to_instruction(
        self,
        parameter_map: dict[Parameter, ParameterValueType] | None = None,
        label: str | None = None,
    ) -> Instruction:
        """Create an :class:`~.circuit.Instruction` out of this circuit.

        .. seealso::
            :func:`circuit_to_instruction`
                The underlying driver of this method.

        Args:
            parameter_map: For parameterized circuits, a mapping from
               parameters in the circuit to parameters to be used in the
               instruction. If None, existing circuit parameters will also
               parameterize the instruction.
            label: Optional gate label.

        Returns:
            qiskit.circuit.Instruction: a composite instruction encapsulating this circuit (can be
                decomposed back).
        """

        from qiskit.converters.circuit_to_instruction import circuit_to_instruction

        return circuit_to_instruction(self, parameter_map, label=label)

    def to_gate(
        self,
        parameter_map: dict[Parameter, ParameterValueType] | None = None,
        label: str | None = None,
    ) -> Gate:
        """Create a :class:`.Gate` out of this circuit.  The circuit must act only on qubits and
        contain only unitary operations.

        .. seealso::
            :func:`circuit_to_gate`
                The underlying driver of this method.

        Args:
            parameter_map: For parameterized circuits, a mapping from parameters in the circuit to
                parameters to be used in the gate. If ``None``, existing circuit parameters will
                also parameterize the gate.
            label : Optional gate label.

        Returns:
            Gate: a composite gate encapsulating this circuit (can be decomposed back).
        """

        from qiskit.converters.circuit_to_gate import circuit_to_gate

        return circuit_to_gate(self, parameter_map, label=label)

    def decompose(
        self,
        gates_to_decompose: (
            str | type[Instruction] | Sequence[str | type[Instruction]] | None
        ) = None,
        reps: int = 1,
    ) -> typing.Self:
        """Call a decomposition pass on this circuit, to decompose one level (shallow decompose).

        Args:
            gates_to_decompose: Optional subset of gates to decompose. Can be a gate type, such as
                ``HGate``, or a gate name, such as "h", or a gate label, such as "My H Gate", or a
                list of any combination of these. If a gate name is entered, it will decompose all
                gates with that name, whether the gates have labels or not. Defaults to all gates in
                the circuit.
            reps: Optional number of times the circuit should be decomposed.
                For instance, ``reps=2`` equals calling ``circuit.decompose().decompose()``.

        Returns:
            QuantumCircuit: a circuit one level decomposed
        """

        from qiskit.transpiler.passes.basis.decompose import Decompose
        from qiskit.converters.circuit_to_dag import circuit_to_dag
        from qiskit.converters.dag_to_circuit import dag_to_circuit

        dag = circuit_to_dag(self, copy_operations=True)

        pass_ = Decompose(gates_to_decompose, apply_synthesis=True)
        for _ in range(reps):
            dag = pass_.run(dag)

        # do not copy operations, this is done in the conversion with circuit_to_dag
        return dag_to_circuit(dag, copy_operations=False)

    def to_dag(self, *, copy_operations: bool = True) -> qiskit.dagcircuit.DAGCircuit:
        """Convert this circuit to a :class:`.DAGCircuit`.

        This is a simple wrapper around :func:`.circuit_to_dag`.

        Args:
            copy_operations: whether to deep copy the individual instructions.  If set to ``False``,
                the operation is cheaper but mutations to the instructions in the DAG will affect
                the original circuit.

        Returns:
            a DAG representing this same circuit.
        """

        from qiskit.converters import circuit_to_dag

        return circuit_to_dag(self, copy_operations=copy_operations)

    def draw(
        self,
        output: str | None = None,
        scale: float | None = None,
        filename: str | None = None,
        style: dict | str | None = None,
        interactive: bool = False,
        plot_barriers: bool = True,
        reverse_bits: bool | None = None,
        justify: str | None = None,
        vertical_compression: str | None = "medium",
        idle_wires: bool | str | None = None,
        with_layout: bool = True,
        fold: int | None = None,
        # The type of ax is matplotlib.axes.Axes, but this is not a fixed dependency, so cannot be
        # safely forward-referenced.
        ax: Any | None = None,
        initial_state: bool = False,
        cregbundle: bool | None = None,
        wire_order: list[int] | None = None,
        expr_len: int = 30,
        measure_arrows: bool | None = None,
        barrier_label_len: int = 16,
    ):
        r"""Draw the quantum circuit. Use the output parameter to choose the drawing format:

        ``text``
            ASCII art TextDrawing that can be printed in the console.

        ``mpl``
            Images with color rendered purely in Python using matplotlib.

        ``latex``
            High-quality images compiled via LaTeX.

            .. warning::
                This will call an installed system version of ``pdflatex`` on arbitrary user input
                by design (such as to render custom code in :attr:`.Instruction.label`), so should
                only be used on trusted input.

        ``latex_source``
            Raw uncompiled LaTeX output.  This is the source of what would be rendered by the
            ``latex`` drawer.

        .. warning::

            Support for :class:`~.expr.Expr` nodes in conditions and :attr:`.SwitchCaseOp.target`
            fields is preliminary and incomplete.  The ``text`` and ``mpl`` drawers will make a
            best-effort attempt to show data dependencies, but the LaTeX-based drawers will skip
            these completely.

        Args:
            output: Select the output method to use for drawing the circuit.
                Valid choices are ``text``, ``mpl``, ``latex``, ``latex_source``.
                By default, the ``text`` drawer is used unless the user config file
                (usually ``~/.qiskit/settings.conf``) has an alternative backend set
                as the default. For example, ``circuit_drawer = latex``. If the output
                kwarg is set, that backend will always be used over the default in
                the user config file.
            scale: Scale of image to draw (shrink if ``< 1.0``). Only used by
                the ``mpl``, ``latex`` and ``latex_source`` outputs. Defaults to ``1.0``.
            filename: File path to save image to. Defaults to ``None`` (result not saved in a file).
            style: Style name, file name of style JSON file, or a dictionary specifying the style.

                * The supported style names are ``"iqp"`` (default), ``"iqp-dark"``, ``"clifford"``,
                  ``"textbook"`` and ``"bw"``.
                * If given a JSON file, e.g. ``my_style.json`` or ``my_style`` (the ``.json``
                  extension may be omitted), this function attempts to load the style dictionary
                  from that location. Note, that the JSON file must completely specify the
                  visualization specifications. The file is searched for in
                  ``qiskit/visualization/circuit/styles``, the current working directory, and
                  the location specified in ``~/.qiskit/settings.conf``.
                * If a dictionary, every entry overrides the default configuration. If the
                  ``"name"`` key is given, the default configuration is given by that style.
                  For example, ``{"name": "textbook", "subfontsize": 5}`` loads the ``"textbook"``
                  style and sets the subfontsize (e.g. the gate angles) to ``5``.
                * If ``None`` the default style ``"iqp"`` is used or, if given, the default style
                  specified in ``~/.qiskit/settings.conf``.

            interactive: When set to ``True``, show the circuit in a new window
                (for ``mpl`` this depends on the matplotlib backend being used
                supporting this). Note when used with either the `text` or the
                ``latex_source`` output type this has no effect and will be silently
                ignored. Defaults to ``False``.
            reverse_bits: When set to ``True``, reverse the bit order inside
                registers for the output visualization. Defaults to ``False`` unless the
                user config file (usually ``~/.qiskit/settings.conf``) has an
                alternative value set. For example, ``circuit_reverse_bits = True``.
            plot_barriers: Enable/disable drawing barriers in the output
                circuit. Defaults to ``True``.
            justify: Options are ``"left"``, ``"right"`` or ``"none"`` (str).
                If anything else is supplied, left justified will be used instead.
                It refers to where gates should be placed in the output circuit if
                there is an option. ``none`` results in each gate being placed in
                its own column. Defaults to ``left``.
            vertical_compression: ``high``, ``medium`` or ``low``. It
                merges the lines generated by the `text` output so the drawing
                will take less vertical room.  Default is ``medium``. Only used by
                the ``text`` output, will be silently ignored otherwise.
            idle_wires: Include (or not) idle wires (wires with no circuit elements)
                in output visualization. The string ``"auto"`` is also possible, in which
                case idle wires are shown except when the circuit has a layout attached.
                Default is ``"auto"`` unless the
                user config file (usually ``~/.qiskit/settings.conf``) has an
                alternative value set. For example, ``circuit_idle_wires = False``.
            with_layout: Include layout information, with labels on the
                physical layout. Default is ``True``.
            fold: Sets pagination. It can be disabled using -1. In ``text``,
                sets the length of the lines. This is useful when the drawing does
                not fit in the console. If None (default), it will try to guess the
                console width using ``shutil.get_terminal_size()``. However, if
                running in jupyter, the default line length is set to 80 characters.
                In ``mpl``, it is the number of (visual) layers before folding.
                Default is 25.
            ax: Only used by the `mpl` backend. An optional ``matplotlib.axes.Axes``
                object to be used for the visualization output. If none is
                specified, a new matplotlib Figure will be created and used.
                Additionally, if specified there will be no returned Figure since
                it is redundant.
            initial_state: Adds :math:`|0\rangle` in the beginning of the qubit wires and
                :math:`0` to classical wires. Default is ``False``.
            cregbundle: If set to ``True``, bundle classical registers.
                Default is ``True``, except for when ``output`` is set to  ``"text"``.
            wire_order: A list of integers used to reorder the display
                of the bits. The list must have an entry for every bit with the bits
                in the range 0 to (``num_qubits`` + ``num_clbits``).
            expr_len: The number of characters to display if an :class:`~.expr.Expr`
                is used for the condition in a :class:`.ControlFlowOp`. If this number is exceeded,
                the string will be truncated at that number and '...' added to the end.
            measure_arrows: If True, draw an arrow from each measure box down to the classical bit
                or register where the measure value is placed. If False, do not draw arrow, but
                instead place the name of the bit or register in the measure box.
                Default is ``True`` unless the user config file (usually ``~/.qiskit/settings.conf``)
                has an alternative value set. For example, ``circuit_measure_arrows = False``.
            barrier_label_len: The number of characters to display for
                :class:`.Barrier` labels in the output circuit. If this number is exceeded,
                the string will be truncated at that number and '...' added to the end.

        Returns:
            :class:`.TextDrawing` or :class:`matplotlib.figure` or :class:`PIL.Image` or
            :class:`str`:

            * ``TextDrawing`` (if ``output='text'``)
                A drawing that can be printed as ascii art.
            * ``matplotlib.figure.Figure`` (if ``output='mpl'``)
                A matplotlib figure object for the circuit diagram.
            * ``PIL.Image`` (if ``output='latex``')
                An in-memory representation of the image of the circuit diagram.
            * ``str`` (if ``output='latex_source'``)
                The LaTeX source code for visualizing the circuit diagram.

        Raises:
            VisualizationError: when an invalid output method is selected
            ImportError: when the output methods requires non-installed libraries.

        Example:
            .. plot::
               :alt: Circuit diagram output by the previous code.
               :include-source:

               from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
               qc = QuantumCircuit(1, 1)
               qc.h(0)
               qc.measure(0, 0)
               qc.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
        """

        from qiskit.visualization import circuit_drawer

        return circuit_drawer(
            self,
            scale=scale,
            filename=filename,
            style=style,
            output=output,
            interactive=interactive,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            vertical_compression=vertical_compression,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            ax=ax,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=wire_order,
            expr_len=expr_len,
            barrier_label_len=barrier_label_len,
            measure_arrows=measure_arrows,
        )

    def size(
        self,
        filter_function: Callable[..., int] = lambda x: not getattr(
            x.operation, "_directive", False
        ),
    ) -> int:
        """Returns total number of instructions in circuit.

        Args:
            filter_function (callable): a function to filter out some instructions.
                Should take as input a tuple of (Instruction, list(Qubit), list(Clbit)).
                By default, filters out "directives", such as barrier or snapshot.

        Returns:
            int: Total number of gate operations.
        """
        return sum(map(filter_function, self._data))

    def depth(
        self,
        filter_function: Callable[[CircuitInstruction], bool] = lambda x: not getattr(
            x.operation, "_directive", False
        ),
    ) -> int:
        """Return circuit depth (i.e., length of critical path).

        The depth of a quantum circuit is a measure of how many
        "layers" of quantum gates, executed in parallel, it takes to
        complete the computation defined by the circuit.  Because
        quantum gates take time to implement, the depth of a circuit
        roughly corresponds to the amount of time it takes the quantum
        computer to execute the circuit.


        .. warning::
            This operation is not well defined if the circuit contains control-flow operations.

        Args:
            filter_function: A function to decide which instructions count to increase depth.
                Should take as a single positional input a :class:`CircuitInstruction`.
                Instructions for which the function returns ``False`` are ignored in the
                computation of the circuit depth.  By default, filters out "directives", such as
                :class:`.Barrier`.

        Returns:
            int: Depth of circuit.

        Examples:
            Simple calculation of total circuit depth::

                from qiskit.circuit import QuantumCircuit
                qc = QuantumCircuit(4)
                qc.h(0)
                qc.cx(0, 1)
                qc.h(2)
                qc.cx(2, 3)
                assert qc.depth() == 2

            Modifying the previous example to only calculate the depth of multi-qubit gates::

                assert qc.depth(lambda instr: len(instr.qubits) > 1) == 1
        """
        obj_depths = {
            obj: 0 for objects in (self.qubits, self.clbits, self.iter_vars()) for obj in objects
        }

        def update_from_expr(objects, node):
            for var in expr.iter_vars(node):
                if var.standalone:
                    objects.add(var)
                else:
                    objects.update(_builder_utils.node_resources(var).clbits)

        for instruction in self._data:
            objects = set(itertools.chain(instruction.qubits, instruction.clbits))
            if (condition := getattr(instruction.operation, "_condition", None)) is not None:
                objects.update(_builder_utils.condition_resources(condition).clbits)
                if isinstance(condition, expr.Expr):
                    update_from_expr(objects, condition)
                else:
                    objects.update(_builder_utils.condition_resources(condition).clbits)
            elif isinstance(instruction.operation, SwitchCaseOp):
                update_from_expr(objects, expr.lift(instruction.operation.target))
            elif isinstance(instruction.operation, Store):
                update_from_expr(objects, instruction.operation.lvalue)
                update_from_expr(objects, instruction.operation.rvalue)

            # If we're counting this as adding to depth, do so.  If not, it still functions as a
            # data synchronisation point between the objects (think "barrier"), so the depths still
            # get updated to match the current max over the affected objects.
            new_depth = max((obj_depths[obj] for obj in objects), default=0)
            if filter_function(instruction):
                new_depth += 1
            for obj in objects:
                obj_depths[obj] = new_depth
        return max(obj_depths.values(), default=0)

    def width(self) -> int:
        """Return number of qubits plus clbits in circuit.

        Returns:
            int: Width of circuit.

        """
        return self._data.width()

    @property
    def num_qubits(self) -> int:
        """Return number of qubits."""
        return self._data.num_qubits

    @property
    def num_ancillas(self) -> int:
        """Return the number of ancilla qubits.

        Example:

            .. plot::
                :include-source:
                :nofigs:

                from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister

                # Create a 2-qubit quantum circuit
                reg = QuantumRegister(2)
                qc = QuantumCircuit(reg)

                # Create an ancilla register with 1 qubit
                anc = AncillaRegister(1)
                qc.add_register(anc)  # Add the ancilla register to the circuit

                print("Number of ancilla qubits:", qc.num_ancillas)

            .. code-block:: text

                Number of ancilla qubits: 1
        """
        return len(self.ancillas)

    @property
    def num_clbits(self) -> int:
        """Return number of classical bits.

        Example:

            .. plot::
                :include-source:
                :nofigs:

                from qiskit import QuantumCircuit

                # Create a new circuit with two qubits and one classical bit
                qc = QuantumCircuit(2, 1)
                print("Number of classical bits:", qc.num_clbits)

            .. code-block:: text

                Number of classical bits: 1


        """
        return self._data.num_clbits

    def count_ops(self) -> OrderedDict[str, int]:
        """Count each operation kind in the circuit.

        Returns:
            A breakdown of how many operations of each kind, sorted by amount.
        """
        ops_dict = self._data.count_ops()
        return OrderedDict(ops_dict)

    def num_nonlocal_gates(self) -> int:
        """Return number of non-local gates (i.e. involving 2+ qubits).

        Conditional nonlocal gates are also included.
        """
        return self._data.num_nonlocal_gates()

    def get_instructions(self, name: str) -> list[CircuitInstruction]:
        """Get instructions matching name.

        Args:
            name (str): The name of instruction to get.

        Returns:
            list(tuple): list of (instruction, qargs, cargs).
        """
        return [match for match in self._data if match.name == name]

    def num_connected_components(self, unitary_only: bool = False) -> int:
        """How many non-entangled subcircuits can the circuit be factored to.

        Args:
            unitary_only (bool): Compute only unitary part of graph.

        Returns:
            int: Number of connected components in circuit.
        """
        # Convert registers to ints (as done in depth).
        bits = self.qubits if unitary_only else (self.qubits + self.clbits)
        bit_indices: dict[Qubit | Clbit, int] = {bit: idx for idx, bit in enumerate(bits)}

        # Start with each qubit or clbit being its own subgraph.
        sub_graphs = [[bit] for bit in range(len(bit_indices))]

        num_sub_graphs = len(sub_graphs)

        # Here we are traversing the gates and looking to see
        # which of the sub_graphs the gate joins together.
        for instruction in self._data:
            if unitary_only:
                args = instruction.qubits
                num_qargs = len(args)
            else:
                args = instruction.qubits + instruction.clbits
                num_qargs = len(args)

            if num_qargs >= 2 and not getattr(instruction.operation, "_directive", False):
                graphs_touched = []
                num_touched = 0
                # Controls necessarily join all the cbits in the
                # register that they use.
                for item in args:
                    reg_int = bit_indices[item]
                    for k in range(num_sub_graphs):
                        if reg_int in sub_graphs[k]:
                            if k not in graphs_touched:
                                graphs_touched.append(k)
                                break

                graphs_touched = list(set(graphs_touched))
                num_touched = len(graphs_touched)

                # If the gate touches more than one subgraph
                # join those graphs together and return
                # reduced number of subgraphs
                if num_touched > 1:
                    connections = []
                    for idx in graphs_touched:
                        connections.extend(sub_graphs[idx])
                    _sub_graphs = []
                    for idx in range(num_sub_graphs):
                        if idx not in graphs_touched:
                            _sub_graphs.append(sub_graphs[idx])
                    _sub_graphs.append(connections)
                    sub_graphs = _sub_graphs
                    num_sub_graphs -= num_touched - 1
            # Cannot go lower than one so break
            if num_sub_graphs == 1:
                break
        return num_sub_graphs

    def num_unitary_factors(self) -> int:
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.
        """
        return self.num_connected_components(unitary_only=True)

    def num_tensor_factors(self) -> int:
        """Computes the number of tensor factors in the unitary
        (quantum) part of the circuit only.

        Notes:
            This is here for backwards compatibility, and will be
            removed in a future release of Qiskit. You should call
            `num_unitary_factors` instead.
        """
        return self.num_unitary_factors()

    def copy(self, name: str | None = None) -> typing.Self:
        """Copy the circuit.

        Args:
          name (str): name to be given to the copied circuit. If None, then the name stays the same.

        Returns:
          QuantumCircuit: a deepcopy of the current circuit, with the specified name
        """

        # vars_mode is "drop" since ``cpy._data`` is overridden anyway with the call to copy,
        # so no need to copy variables in this call.
        cpy = self.copy_empty_like(name, vars_mode="drop")
        cpy._data = self._data.copy()
        return cpy

    def copy_empty_like(
        self,
        name: str | None = None,
        *,
        vars_mode: Literal["alike", "captures", "drop"] = "alike",
    ) -> typing.Self:
        """Return a copy of self with the same structure but empty.

        That structure includes:

        * name and other metadata
        * global phase
        * all the qubits and clbits, including the registers
        * the realtime variables defined in the circuit, handled according to the ``vars`` keyword
          argument.

        .. warning::

            If the circuit contains any local variable declarations (those added by the
            ``declarations`` argument to the circuit constructor, or using :meth:`add_var`), they
            may be **uninitialized** in the output circuit.  You will need to manually add store
            instructions for them (see :class:`.Store` and :meth:`.QuantumCircuit.store`) to
            initialize them.

        Args:
            name: Name for the copied circuit. If None, then the name stays the same.
            vars_mode: The mode to handle realtime variables in.

                alike
                    The variables in the output circuit will have the same declaration semantics as
                    in the original circuit.  For example, ``input`` variables in the source will be
                    ``input`` variables in the output circuit.
                    Note that this causes the local variables to be uninitialised, because the stores are
                    not copied.  This can leave the circuit in a potentially dangerous state for users if
                    they don't re-add initializer stores.

                captures
                    All variables will be converted to captured variables.  This is useful when you
                    are building a new layer for an existing circuit that you will want to
                    :meth:`compose` onto the base, since :meth:`compose` can inline captures onto
                    the base circuit (but not other variables).

                drop
                    The output circuit will have no variables defined.

        Returns:
            QuantumCircuit: An empty copy of self.
        """
        if not (name is None or isinstance(name, str)):
            raise TypeError(
                f"invalid name for a circuit: '{name}'. The name must be a string or 'None'."
            )
        cpy = _copy.copy(self)

        _copy_metadata(self, cpy)

        cpy._data = self._data.copy_empty_like(vars_mode=vars_mode)

        if name:
            cpy.name = name
        return cpy

    def clear(self) -> None:
        """Clear all instructions in self.

        Clearing the circuits will keep the metadata.

        .. seealso::
            :meth:`copy_empty_like`
                A method to produce a new circuit with no instructions and all the same tracking of
                quantum and classical typed data, but without mutating the original circuit.
        """
        self._data.clear()
        # Repopulate the parameter table with any phase symbols.
        self.global_phase = self.global_phase

    def _create_creg(self, length: int, name: str) -> ClassicalRegister:
        """Creates a creg, checking if ClassicalRegister with same name exists"""
        if name in [creg.name for creg in self.cregs]:
            new_creg = ClassicalRegister._new_with_prefix(length, name)
        else:
            new_creg = ClassicalRegister(length, name)
        return new_creg

    def _create_qreg(self, length: int, name: str) -> QuantumRegister:
        """Creates a qreg, checking if QuantumRegister with same name exists"""
        if name in [qreg.name for qreg in self.qregs]:
            new_qreg = QuantumRegister._new_with_prefix(length, name)
        else:
            new_qreg = QuantumRegister(length, name)
        return new_qreg

    def reset(self, qubit: QubitSpecifier) -> InstructionSet:
        """Reset the quantum bit(s) to their default state.

        Args:
            qubit: qubit(s) to reset.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instruction.
        """
        from .reset import Reset

        return self.append(Reset(), [qubit], [], copy=False)

    def store(self, lvalue: typing.Any, rvalue: typing.Any, /) -> InstructionSet:
        """Store the result of the given real-time classical expression ``rvalue`` in the memory
        location defined by ``lvalue``.

        Typically ``lvalue`` will be a :class:`~.expr.Var` node and ``rvalue`` will be some
        :class:`~.expr.Expr` to write into it, but anything that :func:`.expr.lift` can raise to an
        :class:`~.expr.Expr` is permissible in both places, and it will be called on them.

        Args:
            lvalue: a valid specifier for a memory location in the circuit.  This will typically be
                a :class:`~.expr.Var` node, but you can also write to :class:`.Clbit` or
                :class:`.ClassicalRegister` memory locations if your hardware supports it.  The
                memory location must already be present in the circuit.
            rvalue: a real-time classical expression whose result should be written into the given
                memory location.

        .. seealso::
            :class:`~.circuit.Store`
                The backing :class:`~.circuit.Instruction` class that represents this operation.

            :meth:`add_var`
                Create a new variable in the circuit that can be written to with this method.
        """
        # As a convenience, lift integer-literal rvalues to the matching width.
        lvalue = expr.lift(lvalue)
        rvalue_type = (
            lvalue.type if isinstance(rvalue, int) and not isinstance(rvalue, bool) else None
        )
        rvalue = expr.lift(rvalue, rvalue_type)
        return self.append(Store(lvalue, rvalue), (), (), copy=False)

    def measure(self, qubit: QubitSpecifier, cbit: ClbitSpecifier) -> InstructionSet:
        r"""Measure a quantum bit (``qubit``) in the Z basis into a classical bit (``cbit``).

        When a quantum state is measured, a qubit is projected in the computational (Pauli Z) basis
        to either :math:`\lvert 0 \rangle` or :math:`\lvert 1 \rangle`. The classical bit ``cbit``
        indicates the result
        of that projection as a ``0`` or a ``1`` respectively. This operation is non-reversible.

        Args:
            qubit: qubit(s) to measure.
            cbit: classical bit(s) to place the measurement result(s) in,
                or a :class:`.expr.Var` of :class:`.types.Uint` type indexing the target clbit.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instructions.

        Raises:
            CircuitError: if arguments have bad format.

        Examples:
            In this example, a qubit is measured and the result of that measurement is stored in the
            classical bit (usually expressed in diagrams as a double line):

            .. plot::
               :include-source:
               :nofigs:
               :context: reset

               from qiskit import QuantumCircuit
               circuit = QuantumCircuit(1, 1)
               circuit.h(0)
               circuit.measure(0, 0)
               circuit.draw()


            .. code-block:: text

                      ┌───┐┌─┐
                   q: ┤ H ├┤M├
                      └───┘└╥┘
                 c: 1/══════╩═
                            0

            It is possible to call ``measure`` with lists of ``qubits`` and ``cbits`` as a shortcut
            for one-to-one measurement. These two forms produce identical results:

            .. plot::
               :include-source:
               :nofigs:
               :context:

               circuit = QuantumCircuit(2, 2)
               circuit.measure([0,1], [0,1])

            .. plot::
               :include-source:
               :nofigs:
               :context:

               circuit = QuantumCircuit(2, 2)
               circuit.measure(0, 0)
               circuit.measure(1, 1)

            Instead of lists, you can use :class:`~qiskit.circuit.QuantumRegister` and
            :class:`~qiskit.circuit.ClassicalRegister` under the same logic.

            .. plot::
               :include-source:
               :nofigs:
               :context: reset

                from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
                qreg = QuantumRegister(2, "qreg")
                creg = ClassicalRegister(2, "creg")
                circuit = QuantumCircuit(qreg, creg)
                circuit.measure(qreg, creg)

            This is equivalent to:

            .. plot::
               :include-source:
               :nofigs:
               :context:

                circuit = QuantumCircuit(qreg, creg)
                circuit.measure(qreg[0], creg[0])
                circuit.measure(qreg[1], creg[1])

        """
        from .measure import Measure

        return self.append(Measure(), [qubit], [cbit], copy=False)

    def measure_active(self, inplace: bool = True) -> QuantumCircuit | None:
        """Adds measurement to all non-idle qubits. Creates a new ClassicalRegister with
        a size equal to the number of non-idle qubits being measured.

        Returns a new circuit with measurements if `inplace=False`.

        Args:
            inplace (bool): All measurements inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns circuit with measurements when ``inplace = False``.
        """

        from qiskit.converters.circuit_to_dag import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()
        dag = circuit_to_dag(circ)
        qubits_to_measure = [qubit for qubit in circ.qubits if qubit not in dag.idle_wires()]
        new_creg = circ._create_creg(len(qubits_to_measure), "meas")
        circ.add_register(new_creg)
        circ.barrier()
        circ.measure(qubits_to_measure, new_creg)

        if not inplace:
            return circ
        else:
            return None

    def measure_all(self, inplace: bool = True, add_bits: bool = True) -> QuantumCircuit | None:
        """Adds measurement to all qubits.

        By default, adds new classical bits in a :obj:`.ClassicalRegister` to store these
        measurements.  If ``add_bits=False``, the results of the measurements will instead be stored
        in the already existing classical bits, with qubit ``n`` being measured into classical bit
        ``n``.

        Returns a new circuit with measurements if ``inplace=False``.

        Args:
            inplace (bool): All measurements inplace or return new circuit.
            add_bits (bool): Whether to add new bits to store the results.

        Returns:
            QuantumCircuit: Returns circuit with measurements when ``inplace=False``.

        Raises:
            CircuitError: if ``add_bits=False`` but there are not enough classical bits.
        """
        if inplace:
            circ = self
        else:
            circ = self.copy()
        if add_bits:
            new_creg = circ._create_creg(circ.num_qubits, "meas")
            circ.add_register(new_creg)
            circ.barrier()
            circ.measure(circ.qubits, new_creg)
        else:
            if circ.num_clbits < circ.num_qubits:
                raise CircuitError(
                    "The number of classical bits must be equal or greater than "
                    "the number of qubits."
                )
            circ.barrier()
            circ.measure(circ.qubits, circ.clbits[0 : circ.num_qubits])

        if not inplace:
            return circ
        else:
            return None

    def remove_final_measurements(self, inplace: bool = True) -> QuantumCircuit | None:
        """Removes final measurements and barriers on all qubits if they are present.
        Deletes the classical registers that were used to store the values from these measurements
        that become idle as a result of this operation, and deletes classical bits that are
        referenced only by removed registers, or that aren't referenced at all but have
        become idle as a result of this operation.

        Measurements and barriers are considered final if they are
        followed by no other operations (aside from other measurements or barriers.)

        .. note::
            This method has rather complex behavior, particularly around the removal of newly idle
            classical bits and registers.  It is much more efficient to avoid adding unnecessary
            classical data in the first place, rather than trying to remove it later.

        .. seealso::
            :class:`.RemoveFinalMeasurements`
                A transpiler pass that removes final measurements and barriers.  This does not
                remove the classical data.  If this is your goal, you can call that with::

                    from qiskit.circuit import QuantumCircuit
                    from qiskit.transpiler.passes import RemoveFinalMeasurements

                    qc = QuantumCircuit(2, 2)
                    qc.h(0)
                    qc.cx(0, 1)
                    qc.barrier()
                    qc.measure([0, 1], [0, 1])

                    pass_ = RemoveFinalMeasurements()
                    just_bell = pass_(qc)

        Args:
            inplace (bool): All measurements removed inplace or return new circuit.

        Returns:
            QuantumCircuit: Returns the resulting circuit when ``inplace=False``, else None.
        """

        from qiskit.transpiler.passes import RemoveFinalMeasurements
        from qiskit.converters import circuit_to_dag

        if inplace:
            circ = self
        else:
            circ = self.copy()

        dag = circuit_to_dag(circ)
        remove_final_meas = RemoveFinalMeasurements()
        new_dag = remove_final_meas.run(dag)
        kept_cregs = set(new_dag.cregs.values())
        kept_clbits = set(new_dag.clbits)

        # Filter only cregs/clbits still in new DAG, preserving original circuit order
        cregs_to_add = [creg for creg in circ.cregs if creg in kept_cregs]
        clbits_to_add = [clbit for clbit in circ._data.clbits if clbit in kept_clbits]

        # Save the old qregs
        old_qregs = circ.qregs

        # Clear instruction info
        circ._data = CircuitData(
            qubits=circ._data.qubits, reserve=len(circ._data), global_phase=circ.global_phase
        )

        # Re-add old registers.  We avoid `add_register` since we already know the registers are
        # valid, the bits exist, and we don't want to trigger the "modified the quantum registers
        # with a set layout" warning.
        for qreg in old_qregs:
            circ._data.add_qreg(qreg)

        # We must add the clbits first to preserve the original circuit
        # order. This way, add_register never adds clbits and just
        # creates registers that point to them.
        circ.add_bits(clbits_to_add)
        for creg in cregs_to_add:
            circ._data.add_creg(creg)

        # Set circ instructions to match the new DAG
        for node in new_dag.topological_op_nodes():
            # Get arguments for classical condition (if any)
            inst = node.op.copy()
            circ.append(inst, node.qargs, node.cargs)

        if not inplace:
            return circ
        else:
            return None

    @staticmethod
    def from_qasm_file(path: str | os.PathLike) -> QuantumCircuit:
        """Read an OpenQASM 2.0 program from a file and convert to an instance of
        :class:`.QuantumCircuit`.

        Args:
          path: Path to the file for an OpenQASM 2 program

        Return:
          QuantumCircuit: The QuantumCircuit object for the input OpenQASM 2.

        See also:
            :func:`.qasm2.load`: the complete interface to the OpenQASM 2 importer.
        """

        from qiskit import qasm2

        return qasm2.load(
            path,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )

    @staticmethod
    def from_qasm_str(qasm_str: str) -> QuantumCircuit:
        """Convert a string containing an OpenQASM 2.0 program to a :class:`.QuantumCircuit`.

        Args:
          qasm_str (str): A string containing an OpenQASM 2.0 program.
        Return:
          QuantumCircuit: The QuantumCircuit object for the input OpenQASM 2

        See also:
            :func:`.qasm2.loads`: the complete interface to the OpenQASM 2 importer.
        """

        from qiskit import qasm2

        return qasm2.loads(
            qasm_str,
            include_path=qasm2.LEGACY_INCLUDE_PATH,
            custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
            custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
            strict=False,
        )

    @property
    def global_phase(self) -> ParameterValueType:
        """The global phase of the current circuit scope in radians.

        Example:

            .. plot::
                :include-source:
                :nofigs:
                :context: reset

                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(2)
                circuit.h(0)
                circuit.cx(0, 1)
                print(circuit.global_phase)

            .. code-block:: text

                0.0

            .. plot::
                :include-source:
                :nofigs:
                :context:

                from numpy import pi

                circuit.global_phase = pi/4
                print(circuit.global_phase)

            .. code-block:: text

                0.7853981633974483
        """

        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].global_phase
        return self._data.global_phase

    @global_phase.setter
    def global_phase(self, angle: ParameterValueType):
        """Set the phase of the current circuit scope.

        Args:
            angle (float, ParameterExpression): radians
        """
        if self._control_flow_scopes:
            self._control_flow_scopes[-1].global_phase = angle
        else:
            self._data.global_phase = angle

    @property
    def parameters(self) -> ParameterView:
        """The parameters defined in the circuit.

        This attribute returns the :class:`.Parameter` objects in the circuit sorted
        alphabetically. Note that parameters instantiated with a :class:`.ParameterVector`
        are still sorted numerically.

        Examples:

            The snippet below shows that insertion order of parameters does not matter.

            .. plot::
               :include-source:
               :nofigs:

                >>> from qiskit.circuit import QuantumCircuit, Parameter
                >>> a, b, elephant = Parameter("a"), Parameter("b"), Parameter("elephant")
                >>> circuit = QuantumCircuit(1)
                >>> circuit.rx(b, 0)
                >>> circuit.rz(elephant, 0)
                >>> circuit.ry(a, 0)
                >>> circuit.parameters  # sorted alphabetically!
                ParameterView([Parameter(a), Parameter(b), Parameter(elephant)])

            Bear in mind that alphabetical sorting might be unintuitive when it comes to numbers.
            The literal "10" comes before "2" in strict alphabetical sorting.

            .. plot::
               :include-source:
               :nofigs:

                >>> from qiskit.circuit import QuantumCircuit, Parameter
                >>> angles = [Parameter("angle_1"), Parameter("angle_2"), Parameter("angle_10")]
                >>> circuit = QuantumCircuit(1)
                >>> circuit.u(*angles, 0)
                >>> circuit.draw()
                   ┌─────────────────────────────┐
                q: ┤ U(angle_1,angle_2,angle_10) ├
                   └─────────────────────────────┘
                >>> circuit.parameters
                ParameterView([Parameter(angle_1), Parameter(angle_10), Parameter(angle_2)])

            To respect numerical sorting, a :class:`.ParameterVector` can be used.

            .. plot::
               :include-source:
               :nofigs:

                >>> from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
                >>> x = ParameterVector("x", 12)
                >>> circuit = QuantumCircuit(1)
                >>> for x_i in x:
                ...     circuit.rx(x_i, 0)
                >>> circuit.parameters
                ParameterView([
                    ParameterVectorElement(x[0]), ParameterVectorElement(x[1]),
                    ParameterVectorElement(x[2]), ParameterVectorElement(x[3]),
                    ..., ParameterVectorElement(x[11])
                ])


        Returns:
            The sorted :class:`.Parameter` objects in the circuit.
        """
        # return as parameter view, which implements the set and list interface
        return ParameterView(self._data.parameters)

    @property
    def num_parameters(self) -> int:
        """The number of parameter objects in the circuit."""
        return self._data.num_parameters()

    def _unsorted_parameters(self) -> set[Parameter]:
        """Efficiently get all parameters in the circuit, without any sorting overhead.

        .. warning::

            The returned object may directly view onto the ``ParameterTable`` internals, and so
            should not be mutated.  This is an internal performance detail.  Code outside of this
            package should not use this method.
        """
        # This should be free, by accessing the actual backing data structure of the table, but that
        # means that we need to copy it if adding keys from the global phase.
        return self._data.unsorted_parameters()

    @overload
    def assign_parameters(
        self,
        parameters: Mapping[Parameter, ParameterValueType] | Iterable[ParameterValueType],
        inplace: Literal[False] = ...,
        *,
        flat_input: bool = ...,
        strict: bool = ...,
    ) -> QuantumCircuit: ...

    @overload
    def assign_parameters(
        self,
        parameters: Mapping[Parameter, ParameterValueType] | Iterable[ParameterValueType],
        inplace: Literal[True] = ...,
        *,
        flat_input: bool = ...,
        strict: bool = ...,
    ) -> None: ...

    def assign_parameters(
        self,
        parameters: Mapping[Parameter, ParameterValueType] | Iterable[ParameterValueType],
        inplace: bool = False,
        *,
        flat_input: bool = False,
        strict: bool = True,
    ) -> QuantumCircuit | None:
        """Assign parameters to new parameters or values.

        If ``parameters`` is passed as a dictionary, the keys should be :class:`.Parameter`
        instances in the current circuit. The values of the dictionary can either be numeric values
        or new parameter objects.

        If ``parameters`` is passed as a list or array, the elements are assigned to the
        current parameters in the order of :attr:`parameters` which is sorted
        alphabetically (while respecting the ordering in :class:`.ParameterVector` objects).

        The values can be assigned to the current circuit object or to a copy of it.

        .. note::
            When ``parameters`` is given as a mapping, it is permissible to have keys that are
            strings of the parameter names; these will be looked up using :meth:`get_parameter`.
            You can also have keys that are :class:`.ParameterVector` instances, and in this case,
            the dictionary value should be a sequence of values of the same length as the vector.

            If you use either of these cases, you must leave the setting ``flat_input=False``;
            changing this to ``True`` enables the fast path, where all keys must be
            :class:`.Parameter` instances.

        Args:
            parameters: Either a dictionary or iterable specifying the new parameter values.
            inplace: If False, a copy of the circuit with the bound parameters is returned.
                If True the circuit instance itself is modified.
            flat_input: If ``True`` and ``parameters`` is a mapping type, it is assumed to be
                exactly a mapping of ``{parameter: value}``.  By default (``False``), the mapping
                may also contain :class:`.ParameterVector` keys that point to a corresponding
                sequence of values, and these will be unrolled during the mapping, or string keys,
                which will be converted to :class:`.Parameter` instances using
                :meth:`get_parameter`.
            strict: If ``False``, any parameters given in the mapping that are not used in the
                circuit will be ignored.  If ``True`` (the default), an error will be raised
                indicating a logic error.

        Raises:
            CircuitError: If parameters is a dict and contains parameters not present in the
                circuit.
            ValueError: If parameters is a list/array and the length mismatches the number of free
                parameters in the circuit.

        Returns:
            A copy of the circuit with bound parameters if ``inplace`` is False, otherwise None.

        Examples:

            Create a parameterized circuit and assign the parameters in-place.

            .. plot::
               :alt: Circuit diagram output by the previous code.
               :include-source:

               from qiskit.circuit import QuantumCircuit, Parameter

               circuit = QuantumCircuit(2)
               params = [Parameter('A'), Parameter('B'), Parameter('C')]
               circuit.ry(params[0], 0)
               circuit.crx(params[1], 0, 1)
               circuit.draw('mpl')
               circuit.assign_parameters({params[0]: params[2]}, inplace=True)
               circuit.draw('mpl')

            Bind the values out-of-place by list and get a copy of the original circuit.

            .. plot::
               :alt: Circuit diagram output by the previous code.
               :include-source:

               from qiskit.circuit import QuantumCircuit, ParameterVector

               circuit = QuantumCircuit(2)
               params = ParameterVector('P', 2)
               circuit.ry(params[0], 0)
               circuit.crx(params[1], 0, 1)

               bound_circuit = circuit.assign_parameters([1, 2])
               bound_circuit.draw('mpl')

               circuit.draw('mpl')

        """
        if inplace:
            target = self
        else:
            if not isinstance(parameters, dict):
                # We're going to need to access the sorted order within the inner Rust method on
                # `target`, so warm up our own cache first so that subsequent calls to
                # `assign_parameters` on `self` benefit as well.
                _ = self._data.parameters
            target = self.copy()
            target._increment_instances()
            target._name_update()

        if isinstance(parameters, collections.abc.Mapping):
            raw_mapping = parameters if flat_input else self._unroll_param_dict(parameters, strict)
            if strict and (
                extras := [
                    parameter for parameter in raw_mapping if not self.has_parameter(parameter)
                ]
            ):
                raise CircuitError(
                    f"Cannot bind parameters ({', '.join(str(x) for x in extras)}) not present in"
                    " the circuit."
                )

            target._data.assign_parameters_mapping(raw_mapping)
        else:
            target._data.assign_parameters_iterable(parameters)

        return None if inplace else target

    def has_control_flow_op(self) -> bool:
        """Checks whether the circuit has an instance of :class:`.ControlFlowOp`
        present amongst its operations."""
        return self._data.has_control_flow_op()

    def _unroll_param_dict(
        self, parameter_binds: Mapping[Parameter, ParameterValueType], strict: bool = True
    ) -> Mapping[Parameter, ParameterValueType]:
        out = {}
        for parameter, value in parameter_binds.items():
            if isinstance(parameter, ParameterVector):
                if len(parameter) != len(value):
                    raise CircuitError(
                        f"Parameter vector '{parameter.name}' has length {len(parameter)},"
                        f" but was assigned to {len(value)} values."
                    )
                out.update(zip(parameter, value))
            elif isinstance(parameter, str):
                if strict:
                    out[self.get_parameter(parameter)] = value
                if not strict and self.has_parameter(parameter):
                    out[self.get_parameter(parameter)] = value
            else:
                out[parameter] = value
        return out

    def barrier(self, *qargs: QubitSpecifier, label=None) -> InstructionSet:
        """Apply :class:`~.library.Barrier`. If ``qargs`` is empty, applies to all qubits
        in the circuit.

        Args:
            qargs (QubitSpecifier): Specification for one or more qubit arguments.
            label (str): The string label of the barrier.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instructions.
        """
        from .barrier import Barrier

        if qargs:
            # This uses a `dict` not a `set` to guarantee a deterministic order to the arguments.
            qubits = tuple(
                {q: None for qarg in qargs for q in self._qbit_argument_conversion(qarg)}
            )
            return self.append(
                CircuitInstruction(Barrier(len(qubits), label=label), qubits, ()), copy=False
            )
        else:
            qubits = self.qubits.copy()
            return self._current_scope().append(
                CircuitInstruction(Barrier(len(qubits), label=label), qubits, ())
            )

    def delay(
        self,
        duration: ParameterValueType | expr.Expr,
        qarg: QubitSpecifier | None = None,
        unit: str | None = None,
    ) -> InstructionSet:
        """Apply :class:`~.circuit.Delay`. If qarg is ``None``, applies to all qubits.
        When applying to multiple qubits, delays with the same duration will be created.

        Args:
            duration (Object):
                duration of the delay. If this is an :class:`~.expr.Expr`, it must be
                a constant expression of type :class:`~.types.Duration`.
            qarg (Object): qubit argument to apply this delay.
            unit (str | None): unit of the duration, unless ``duration`` is an :class:`~.expr.Expr`
                in which case it must not be specified. Supported units: ``'s'``, ``'ms'``, ``'us'``,
                ``'ns'``, ``'ps'``, and ``'dt'``. Default is ``'dt'``, i.e. integer time unit
                depending on the target backend.

        Returns:
            qiskit.circuit.InstructionSet: handle to the added instructions.

        Raises:
            CircuitError: if arguments have bad format.
        """
        if qarg is None:
            qarg = self.qubits
        return self.append(Delay(duration, unit=unit), [qarg], [], copy=False)

    def h(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.HGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.H, [qubit], ())

    def ch(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CHGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CH, [control_qubit, target_qubit], (), label=label
            )

        from .library.standard_gates.h import CHGate

        return self.append(
            CHGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def id(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.IGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.I, [qubit], ())

    def ms(self, theta: ParameterValueType, qubits: Sequence[QubitSpecifier]) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.MSGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            qubits: The qubits to apply the gate to.

        Returns:
            A handle to the instructions created.
        """

        from .library.generalized_gates.gms import MSGate

        return self.append(MSGate(len(qubits), theta), qubits, copy=False)

    def p(self, theta: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.PhaseGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.Phase, [qubit], (theta,))

    def cp(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CPhaseGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CPhase, [control_qubit, target_qubit], (theta,), label=label
            )

        from .library.standard_gates.p import CPhaseGate

        return self.append(
            CPhaseGate(theta, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def mcp(
        self,
        lam: ParameterValueType,
        control_qubits: Sequence[QubitSpecifier],
        target_qubit: QubitSpecifier,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.MCPhaseGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            lam: The angle of the rotation.
            control_qubits: The qubits used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        from .library.standard_gates.p import MCPhaseGate

        num_ctrl_qubits = len(control_qubits)
        return self.append(
            MCPhaseGate(lam, num_ctrl_qubits, ctrl_state=ctrl_state),
            control_qubits[:] + [target_qubit],
            [],
            copy=False,
        )

    def mcrx(
        self,
        theta: ParameterValueType,
        q_controls: Sequence[QubitSpecifier],
        q_target: QubitSpecifier,
        use_basis_gates: bool = False,
    ):
        """
        Apply Multiple-Controlled X rotation gate

        Args:
            theta: The angle of the rotation.
            q_controls: The qubits used as the controls.
            q_target: The qubit targeted by the gate.
            use_basis_gates: use p, u, cx basis gates.
        """

        from .library.standard_gates.rx import RXGate
        from qiskit.synthesis.multi_controlled.multi_control_rotation_gates import (
            _apply_cu,
            _apply_mcu_graycode,
            _mcsu2_real_diagonal,
        )

        control_qubits = self._qbit_argument_conversion(q_controls)
        target_qubit = self._qbit_argument_conversion(q_target)
        if len(target_qubit) != 1:
            raise QiskitError("The mcrx gate needs a single qubit as target.")
        all_qubits = control_qubits + target_qubit
        target_qubit = target_qubit[0]
        self._check_dups(all_qubits)

        n_c = len(control_qubits)
        if n_c == 0:
            self.rx(theta, target_qubit)
        elif n_c == 1:  # cu
            _apply_cu(
                self,
                theta,
                -pi / 2,
                pi / 2,
                control_qubits[0],
                target_qubit,
                use_basis_gates=use_basis_gates,
            )
        elif n_c < 4:
            theta_step = theta * (1 / (2 ** (n_c - 1)))
            _apply_mcu_graycode(
                self,
                theta_step,
                -pi / 2,
                pi / 2,
                control_qubits,
                target_qubit,
                use_basis_gates=use_basis_gates,
            )
        else:
            cgate = _mcsu2_real_diagonal(
                RXGate(theta),
                num_controls=len(control_qubits),
                use_basis_gates=use_basis_gates,
            )
            self.compose(cgate, control_qubits + [target_qubit], inplace=True)

    def mcry(
        self,
        theta: ParameterValueType,
        q_controls: Sequence[QubitSpecifier],
        q_target: QubitSpecifier,
        q_ancillae: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        mode: str | None = None,
        use_basis_gates: bool = False,
    ):
        """
        Apply Multiple-Controlled Y rotation gate

        Args:
            theta: The angle of the rotation.
            q_controls: The qubits used as the controls.
            q_target: The qubit targeted by the gate.
            q_ancillae: The list of ancillary qubits.
            mode: The implementation mode to use.
            use_basis_gates: use p, u, cx basis gates
        """

        from .library.standard_gates.ry import RYGate
        from qiskit.synthesis.multi_controlled.multi_control_rotation_gates import (
            _apply_cu,
            _apply_mcu_graycode,
            _mcsu2_real_diagonal,
        )

        control_qubits = self._qbit_argument_conversion(q_controls)
        target_qubit = self._qbit_argument_conversion(q_target)
        if len(target_qubit) != 1:
            raise QiskitError("The mcry gate needs a single qubit as target.")
        ancillary_qubits = [] if q_ancillae is None else self._qbit_argument_conversion(q_ancillae)
        all_qubits = control_qubits + target_qubit + ancillary_qubits
        target_qubit = target_qubit[0]
        self._check_dups(all_qubits)

        if len(control_qubits) == 0:
            self.ry(theta, target_qubit)
            return

        # auto-select the best mode
        if mode is None:
            # if enough ancillary qubits are provided, use the 'v-chain' method
            additional_vchain = max(0, len(control_qubits) - 2)
            if len(ancillary_qubits) >= additional_vchain:
                mode = "basic"
            else:
                mode = "noancilla"

        if mode == "basic":
            from qiskit.synthesis.multi_controlled import synth_mcx_n_clean_m15

            self.ry(theta / 2, q_target)
            if len(control_qubits) == 1:
                self.cx(control_qubits[0], q_target)
            elif len(control_qubits) == 2:
                self.ccx(control_qubits[0], control_qubits[1], q_target)
            else:
                qubits = control_qubits + [target_qubit] + ancillary_qubits
                mcx = synth_mcx_n_clean_m15(len(control_qubits))
                self.compose(mcx, qubits, inplace=True)
            self.ry(-theta / 2, q_target)
            if len(control_qubits) == 1:
                self.cx(control_qubits[0], q_target)
            elif len(control_qubits) == 2:
                self.ccx(control_qubits[0], control_qubits[1], q_target)
            else:
                qubits = control_qubits + [target_qubit] + ancillary_qubits
                mcx = synth_mcx_n_clean_m15(len(control_qubits))
                self.compose(mcx, qubits, inplace=True)
        elif mode == "noancilla":
            n_c = len(control_qubits)
            if n_c == 1:  # cu
                _apply_cu(
                    self,
                    theta,
                    0,
                    0,
                    control_qubits[0],
                    target_qubit,
                    use_basis_gates=use_basis_gates,
                )
            elif n_c < 4:
                theta_step = theta * (1 / (2 ** (n_c - 1)))
                _apply_mcu_graycode(
                    self,
                    theta_step,
                    0,
                    0,
                    control_qubits,
                    target_qubit,
                    use_basis_gates=use_basis_gates,
                )
            else:
                cgate = _mcsu2_real_diagonal(
                    RYGate(theta),
                    num_controls=len(control_qubits),
                    use_basis_gates=use_basis_gates,
                )
                self.compose(cgate, control_qubits + [target_qubit], inplace=True)
        else:
            raise QiskitError(f"Unrecognized mode for building MCRY circuit: {mode}.")

    def mcrz(
        self,
        lam: ParameterValueType,
        q_controls: Sequence[QubitSpecifier],
        q_target: QubitSpecifier,
        use_basis_gates: bool = False,
    ):
        """
        Apply Multiple-Controlled Z rotation gate

        Args:
            lam: The angle of the rotation.
            q_controls: The qubits used as the controls.
            q_target: The qubit targeted by the gate.
            use_basis_gates: use p, u, cx basis gates.
        """

        from .library.standard_gates.rz import CRZGate, RZGate
        from qiskit.synthesis.multi_controlled.multi_control_rotation_gates import (
            _mcsu2_real_diagonal,
        )

        control_qubits = self._qbit_argument_conversion(q_controls)
        target_qubit = self._qbit_argument_conversion(q_target)
        if len(target_qubit) != 1:
            raise QiskitError("The mcrz gate needs a single qubit as target.")
        all_qubits = control_qubits + target_qubit
        target_qubit = target_qubit[0]
        self._check_dups(all_qubits)

        n_c = len(control_qubits)

        if n_c == 0:
            self.rz(lam, target_qubit)

        elif n_c == 1:
            if use_basis_gates:
                self.u(0, 0, lam / 2, target_qubit)
                self.cx(control_qubits[0], target_qubit)
                self.u(0, 0, -lam / 2, target_qubit)
                self.cx(control_qubits[0], target_qubit)
            else:
                self.append(CRZGate(lam), control_qubits + [target_qubit])
        else:
            cgate = _mcsu2_real_diagonal(
                RZGate(lam),
                num_controls=len(control_qubits),
                use_basis_gates=use_basis_gates,
            )
            self.compose(cgate, control_qubits + [target_qubit], inplace=True)

    def r(
        self, theta: ParameterValueType, phi: ParameterValueType, qubit: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            phi: The angle of the axis of rotation in the x-y plane.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.R, [qubit], [theta, phi])

    def rv(
        self,
        vx: ParameterValueType,
        vy: ParameterValueType,
        vz: ParameterValueType,
        qubit: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RVGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Rotation around an arbitrary rotation axis :math:`v`, where :math:`|v|` is the angle of
        rotation in radians.

        Args:
            vx: x-component of the rotation axis.
            vy: y-component of the rotation axis.
            vz: z-component of the rotation axis.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        from .library.generalized_gates.rv import RVGate

        return self.append(RVGate(vx, vy, vz), [qubit], [], copy=False)

    def rccx(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RCCXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(
            StandardGate.RCCX, [control_qubit1, control_qubit2, target_qubit], ()
        )

    def rcccx(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        control_qubit3: QubitSpecifier,
        target_qubit: QubitSpecifier,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RC3XGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            control_qubit3: The qubit(s) used as the third control.
            target_qubit: The qubit(s) targeted by the gate.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(
            StandardGate.RC3X,
            [control_qubit1, control_qubit2, control_qubit3, target_qubit],
            (),
        )

    def rx(
        self, theta: ParameterValueType, qubit: QubitSpecifier, label: str | None = None
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.
            label: The string label of the gate in the circuit.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.RX, [qubit], [theta], label=label)

    def crx(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CRXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CRX, [control_qubit, target_qubit], [theta], label=label
            )

        from .library.standard_gates.rx import CRXGate

        return self.append(
            CRXGate(theta, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def rxx(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RXXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.RXX, [qubit1, qubit2], [theta])

    def ry(
        self, theta: ParameterValueType, qubit: QubitSpecifier, label: str | None = None
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.
            label: The string label of the gate in the circuit.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.RY, [qubit], [theta], label=label)

    def cry(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CRYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CRY, [control_qubit, target_qubit], [theta], label=label
            )

        from .library.standard_gates.ry import CRYGate

        return self.append(
            CRYGate(theta, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def ryy(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RYYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.RYY, [qubit1, qubit2], [theta])

    def rz(self, phi: ParameterValueType, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            phi: The rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.RZ, [qubit], [phi])

    def crz(
        self,
        theta: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CRZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The angle of the rotation.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CRZ, [control_qubit, target_qubit], [theta], label=label
            )

        from .library.standard_gates.rz import CRZGate

        return self.append(
            CRZGate(theta, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def rzx(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RZXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.RZX, [qubit1, qubit2], [theta])

    def rzz(
        self, theta: ParameterValueType, qubit1: QubitSpecifier, qubit2: QubitSpecifier
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.RZZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The rotation angle of the gate.
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.RZZ, [qubit1, qubit2], [theta])

    def ecr(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.ECRGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1: The first qubit to apply the gate to.
            qubit2: The second qubit to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.ECR, [qubit1, qubit2], ())

    def s(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.S, [qubit], ())

    def sdg(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.Sdg, [qubit], ())

    def cs(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CS, [control_qubit, target_qubit], (), label=label
            )

        from .library.standard_gates.s import CSGate

        return self.append(
            CSGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def csdg(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CSdg, [control_qubit, target_qubit], (), label=label
            )

        from .library.standard_gates.s import CSdgGate

        return self.append(
            CSdgGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def swap(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SwapGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1: The first qubit to apply the gate to.
            qubit2: The second qubit to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(
            StandardGate.Swap,
            [qubit1, qubit2],
            (),
        )

    def iswap(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.iSwapGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1: The first qubit to apply the gate to.
            qubit2: The second qubit to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.ISwap, [qubit1, qubit2], ())

    def cswap(
        self,
        control_qubit: QubitSpecifier,
        target_qubit1: QubitSpecifier,
        target_qubit2: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSwapGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit1: The qubit(s) targeted by the gate.
            target_qubit2: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. ``'1'``).  Defaults to controlling
                on the ``'1'`` state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CSwap,
                [control_qubit, target_qubit1, target_qubit2],
                (),
                label=label,
            )

        from .library.standard_gates.swap import CSwapGate

        return self.append(
            CSwapGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit1, target_qubit2],
            [],
            copy=False,
        )

    def sx(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.SX, [qubit], ())

    def sxdg(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.SXdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.SXdg, [qubit], ())

    def csx(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.CSXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CSX, [control_qubit, target_qubit], (), label=label
            )

        from .library.standard_gates.sx import CSXGate

        return self.append(
            CSXGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def t(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.TGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.T, [qubit], ())

    def tdg(self, qubit: QubitSpecifier) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.TdgGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.Tdg, [qubit], ())

    def u(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        qubit: QubitSpecifier,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.UGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The :math:`\theta` rotation angle of the gate.
            phi: The :math:`\phi` rotation angle of the gate.
            lam: The :math:`\lambda` rotation angle of the gate.
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.U, [qubit], [theta, phi, lam])

    def cu(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        gamma: ParameterValueType,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CUGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            theta: The :math:`\theta` rotation angle of the gate.
            phi: The :math:`\phi` rotation angle of the gate.
            lam: The :math:`\lambda` rotation angle of the gate.
            gamma: The global phase applied of the U gate, if applied.
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CU,
                [control_qubit, target_qubit],
                [theta, phi, lam, gamma],
                label=label,
            )

        from .library.standard_gates.u import CUGate

        return self.append(
            CUGate(theta, phi, lam, gamma, label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def x(self, qubit: QubitSpecifier, label: str | None = None) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.XGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.
            label: The string label of the gate in the circuit.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.X, [qubit], (), label=label)

    def cx(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CX,
                [control_qubit, target_qubit],
                (),
                label=label,
            )

        from .library.standard_gates.x import CXGate

        return self.append(
            CXGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def dcx(self, qubit1: QubitSpecifier, qubit2: QubitSpecifier) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.DCXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit1: The qubit(s) to apply the gate to.
            qubit2: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.DCX, [qubit1, qubit2], ())

    def ccx(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CCXGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            target_qubit: The qubit(s) targeted by the gate.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |11> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["11", 3]:
            return self._append_standard_gate(
                StandardGate.CCX,
                [control_qubit1, control_qubit2, target_qubit],
                (),
            )

        from .library.standard_gates.x import CCXGate

        return self.append(
            CCXGate(ctrl_state=ctrl_state),
            [control_qubit1, control_qubit2, target_qubit],
            [],
            copy=False,
        )

    @deprecate_arg(
        name="mode",
        since="2.1",
        additional_msg=(
            "Instead, add a generic MCXGate to the circuit and specify the synthesis method "
            "via the ``hls_config`` in the transpilation. Alternatively, specific decompositions "
            "are available at https://qisk.it/mcx."
        ),
    )
    def mcx(
        self,
        control_qubits: Sequence[QubitSpecifier],
        target_qubit: QubitSpecifier,
        ancilla_qubits: QubitSpecifier | Sequence[QubitSpecifier] | None = None,
        mode: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.MCXGate`.

        The multi-cX gate can be implemented using different techniques, which use different numbers
        of ancilla qubits and have varying circuit depth. These modes are:

        - ``'noancilla'``: Requires 0 ancilla qubits.
        - ``'recursion'``: Requires 1 ancilla qubit if more than 4 controls are used, otherwise 0.
        - ``'v-chain'``: Requires 2 less ancillas than the number of control qubits.
        - ``'v-chain-dirty'``: Same as for the clean ancillas (but the circuit will be longer).

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubits: The qubits used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            ancilla_qubits: The qubits used as the ancillae, if the mode requires them.
            mode: The choice of mode, explained further above.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.

        Raises:
            ValueError: if the given mode is not known, or if too few ancilla qubits are passed.
            AttributeError: if no ancilla qubits are passed, but some are needed.
        """
        from .library.standard_gates.x import MCXGate, MCXRecursive, MCXVChain

        num_ctrl_qubits = len(control_qubits)

        available_implementations = {
            "noancilla",
            "recursion",
            "v-chain",
            "v-chain-dirty",
            "advanced",
            "basic",
            "basic-dirty-ancilla",
        }

        # check ancilla input
        if ancilla_qubits:
            _ = self._qbit_argument_conversion(ancilla_qubits)

        if mode is None:
            gate = MCXGate(num_ctrl_qubits, ctrl_state=ctrl_state)
        elif mode in available_implementations:
            if mode == "noancilla":
                gate = MCXGate(num_ctrl_qubits, ctrl_state=ctrl_state)
            elif mode in ["recursion", "advanced"]:
                gate = MCXRecursive(num_ctrl_qubits, ctrl_state=ctrl_state)
            elif mode in ["v-chain", "basic"]:
                gate = MCXVChain(num_ctrl_qubits, False, ctrl_state=ctrl_state)
            elif mode in ["v-chain-dirty", "basic-dirty-ancilla"]:
                gate = MCXVChain(num_ctrl_qubits, dirty_ancillas=True, ctrl_state=ctrl_state)
            else:
                raise ValueError("unreachable.")
            # else is unreachable, we exhausted all options
        else:
            raise ValueError(
                f"Unsupported mode ({mode}) selected, choose one of {available_implementations}"
            )

        if mode is None or not hasattr(gate, "num_ancilla_qubits"):
            ancilla_qubits = []
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                required = gate.num_ancilla_qubits

            if required > 0:
                if ancilla_qubits is None:
                    raise AttributeError(f"No ancillas provided, but {required} are needed!")

                # convert ancilla qubits to a list if they were passed as int or qubit
                if not hasattr(ancilla_qubits, "__len__"):
                    ancilla_qubits = [ancilla_qubits]

                if len(ancilla_qubits) < required:
                    actually = len(ancilla_qubits)
                    raise ValueError(
                        f"At least {required} ancillas required, but {actually} given."
                    )
                # size down if too many ancillas were provided
                ancilla_qubits = ancilla_qubits[:required]
            else:
                ancilla_qubits = []

        return self.append(gate, control_qubits[:] + [target_qubit] + ancilla_qubits[:], [])

    def y(self, qubit: QubitSpecifier) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.YGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.Y, [qubit], ())

    def cy(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CYGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CY,
                [control_qubit, target_qubit],
                (),
                label=label,
            )

        from .library.standard_gates.y import CYGate

        return self.append(
            CYGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def z(self, qubit: QubitSpecifier) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.ZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            qubit: The qubit(s) to apply the gate to.

        Returns:
            A handle to the instructions created.
        """
        return self._append_standard_gate(StandardGate.Z, [qubit], ())

    def cz(
        self,
        control_qubit: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit: The qubit(s) used as the controls.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '1').  Defaults to controlling
                on the '1' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |1> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["1", 1]:
            return self._append_standard_gate(
                StandardGate.CZ, [control_qubit, target_qubit], (), label=label
            )

        from .library.standard_gates.z import CZGate

        return self.append(
            CZGate(label=label, ctrl_state=ctrl_state),
            [control_qubit, target_qubit],
            [],
            copy=False,
        )

    def ccz(
        self,
        control_qubit1: QubitSpecifier,
        control_qubit2: QubitSpecifier,
        target_qubit: QubitSpecifier,
        label: str | None = None,
        ctrl_state: str | int | None = None,
    ) -> InstructionSet:
        r"""Apply :class:`~qiskit.circuit.library.CCZGate`.

        For the full matrix form of this gate, see the underlying gate documentation.

        Args:
            control_qubit1: The qubit(s) used as the first control.
            control_qubit2: The qubit(s) used as the second control.
            target_qubit: The qubit(s) targeted by the gate.
            label: The string label of the gate in the circuit.
            ctrl_state:
                The control state in decimal, or as a bitstring (e.g. '10').  Defaults to controlling
                on the '11' state.

        Returns:
            A handle to the instructions created.
        """
        # if the control state is |11> use the fast Rust version of the gate
        if ctrl_state is None or ctrl_state in ["11", 3]:
            return self._append_standard_gate(
                StandardGate.CCZ,
                [control_qubit1, control_qubit2, target_qubit],
                (),
                label=label,
            )

        from .library.standard_gates.z import CCZGate

        return self.append(
            CCZGate(label=label, ctrl_state=ctrl_state),
            [control_qubit1, control_qubit2, target_qubit],
            [],
            copy=False,
        )

    def pauli(
        self,
        pauli_string: str,
        qubits: Sequence[QubitSpecifier],
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.library.PauliGate`.

        Args:
            pauli_string: A string representing the Pauli operator to apply, e.g. 'XX'.
            qubits: The qubits to apply this gate to.

        Returns:
            A handle to the instructions created.
        """
        from qiskit.circuit.library.generalized_gates.pauli import PauliGate

        return self.append(PauliGate(pauli_string), qubits, [], copy=False)

    def prepare_state(
        self,
        state: Statevector | Sequence[complex] | str | int,
        qubits: Sequence[QubitSpecifier] | None = None,
        label: str | None = None,
        normalize: bool = False,
    ) -> InstructionSet:
        r"""Prepare qubits in a specific state.

        This class implements a state preparing unitary. Unlike
        :meth:`.initialize` it does not reset the qubits first.

        Args:
            state: The state to initialize to, can be either of the following.

                * Statevector or vector of complex amplitudes to initialize to.
                * Labels of basis states of the Pauli eigenstates Z, X, Y. See
                  :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with
                  respect to the qubit index to be applied to. Example label '01' initializes the
                  qubit zero to :math:`|1\rangle` and the qubit one to :math:`|0\rangle`.
                * An integer that is used as a bitmap indicating which qubits to initialize to
                  :math:`|1\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit
                  2 to :math:`|1\rangle` and qubit 1 to :math:`|0\rangle`.

            qubits: Qubits to initialize. If ``None`` the initialization is applied to all qubits in
                the circuit.
            label: An optional label for the gate
            normalize: Whether to normalize an input array to a unit vector.

        Returns:
            A handle to the instruction that was just initialized

        Examples:
            Prepare a qubit in the state :math:`(|0\rangle - |1\rangle) / \sqrt{2}`.

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(1)
                circuit.prepare_state([1/np.sqrt(2), -1/np.sqrt(2)], 0)
                circuit.draw()

            output:

            .. code-block:: text

                     ┌─────────────────────────────────────┐
                q_0: ┤ State Preparation(0.70711,-0.70711) ├
                     └─────────────────────────────────────┘


            Prepare from a string two qubits in the state :math:`|10\rangle`.
            The order of the labels is reversed with respect to qubit index.
            More information about labels for basis states are in
            :meth:`.Statevector.from_label`.

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(2)
                circuit.prepare_state('01', circuit.qubits)
                circuit.draw()

            output:

            .. code-block:: text

                     ┌─────────────────────────┐
                q_0: ┤0                        ├
                     │  State Preparation(0,1) │
                q_1: ┤1                        ├
                     └─────────────────────────┘


            Initialize two qubits from an array of complex amplitudes

            .. plot::
                :include-source:
                :nofigs:

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(2)
                circuit.prepare_state([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)
                circuit.draw()

            output:

            .. code-block:: text

                     ┌───────────────────────────────────────────┐
                q_0: ┤0                                          ├
                     │  State Preparation(0,0.70711,-0.70711j,0) │
                q_1: ┤1                                          ├
                     └───────────────────────────────────────────┘
        """

        from qiskit.circuit.library.data_preparation import StatePreparation

        if qubits is None:
            qubits = self.qubits
        elif isinstance(qubits, (int, np.integer, slice, Qubit)):
            qubits = [qubits]

        num_qubits = len(qubits) if isinstance(state, int) else None

        return self.append(
            StatePreparation(state, num_qubits, label=label, normalize=normalize),
            qubits,
            copy=False,
        )

    def initialize(
        self,
        params: Statevector | Sequence[complex] | str | int,
        qubits: Sequence[QubitSpecifier] | None = None,
        normalize: bool = False,
    ):
        r"""Initialize qubits in a specific state.

        Qubit initialization is done by first resetting the qubits to :math:`|0\rangle`
        followed by calling :class:`~qiskit.circuit.library.StatePreparation`
        class to prepare the qubits in a specified state.
        Both these steps are included in the
        :class:`~qiskit.circuit.library.Initialize` instruction.

        Args:
            params: The state to initialize to, can be either of the following.

                * Statevector or vector of complex amplitudes to initialize to.
                * Labels of basis states of the Pauli eigenstates Z, X, Y. See
                  :meth:`.Statevector.from_label`. Notice the order of the labels is reversed with
                  respect to the qubit index to be applied to. Example label ``'01'`` initializes the
                  qubit zero to :math:`|1\rangle` and the qubit one to :math:`|0\rangle`.
                * An integer that is used as a bitmap indicating which qubits to initialize to
                  :math:`|1\rangle`. Example: setting params to 5 would initialize qubit 0 and qubit
                  2 to :math:`|1\rangle` and qubit 1 to :math:`|0\rangle`.

            qubits: Qubits to initialize. If ``None`` the initialization is applied to all qubits in
                the circuit.
            normalize: Whether to normalize an input array to a unit vector.

        Returns:
            A handle to the instructions created.

        Examples:
            Prepare a qubit in the state :math:`(|0\rangle - |1\rangle) / \sqrt{2}`.

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(1)
                circuit.initialize([1/np.sqrt(2), -1/np.sqrt(2)], 0)
                circuit.draw()

            output:

            .. code-block:: text

                     ┌──────────────────────────────┐
                q_0: ┤ Initialize(0.70711,-0.70711) ├
                     └──────────────────────────────┘


            Initialize from a string two qubits in the state :math:`|10\rangle`.
            The order of the labels is reversed with respect to qubit index.
            More information about labels for basis states are in
            :meth:`.Statevector.from_label`.

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(2)
                circuit.initialize('01', circuit.qubits)
                circuit.draw()

            output:

            .. code-block:: text

                     ┌──────────────────┐
                q_0: ┤0                 ├
                     │  Initialize(0,1) │
                q_1: ┤1                 ├
                     └──────────────────┘

            Initialize two qubits from an array of complex amplitudes.

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit import QuantumCircuit

                circuit = QuantumCircuit(2)
                circuit.initialize([0, 1/np.sqrt(2), -1.j/np.sqrt(2), 0], circuit.qubits)
                circuit.draw()

            output:

            .. code-block:: text

                     ┌────────────────────────────────────┐
                q_0: ┤0                                   ├
                     │  Initialize(0,0.70711,-0.70711j,0) │
                q_1: ┤1                                   ├
                     └────────────────────────────────────┘
        """

        from .library.data_preparation.initializer import Initialize

        if qubits is None:
            qubits = self.qubits
        elif isinstance(qubits, (int, np.integer, slice, Qubit)):
            qubits = [qubits]

        num_qubits = len(qubits) if isinstance(params, int) else None

        return self.append(Initialize(params, num_qubits, normalize), qubits, copy=False)

    def unitary(
        self,
        obj: np.ndarray | Gate | BaseOperator,
        qubits: Sequence[QubitSpecifier],
        label: str | None = None,
    ):
        """Apply unitary gate specified by ``obj`` to ``qubits``.

        Args:
            obj: Unitary operator.
            qubits: The circuit qubits to apply the transformation to.
            label: Unitary name for backend [Default: None].

        Returns:
            QuantumCircuit: The quantum circuit.

        Example:

            Apply a gate specified by a unitary matrix to a quantum circuit

            .. plot::
               :include-source:
               :nofigs:

                from qiskit import QuantumCircuit
                matrix = [[0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]]
                circuit = QuantumCircuit(2)
                circuit.unitary(matrix, [0, 1])
        """

        from .library.generalized_gates.unitary import UnitaryGate

        gate = UnitaryGate(obj, label=label)

        # correctly treat as single-qubit gate if it only acts as 1 qubit, i.e.
        # allow a single qubit specifier and enable broadcasting
        if gate.num_qubits == 1:
            if isinstance(qubits, (int, Qubit)) or len(qubits) > 1:
                qubits = [qubits]

        return self.append(gate, qubits, [], copy=False)

    def noop(self, *qargs: QubitSpecifier):
        """Mark the given qubit(s) as used within the current scope, without adding an operation.

        This has no effect (other than raising an exception on invalid input) when called in the
        top scope of a :class:`QuantumCircuit`.  Within a control-flow builder, this causes the
        qubit to be "used" by the control-flow block, if it wouldn't already be used, without adding
        any additional operations on it.

        For example::

            from qiskit.circuit import QuantumCircuit

            qc = QuantumCircuit(3)
            with qc.box():
                # This control-flow block will only use qubits 0 and 1.
                qc.cx(0, 1)
            with qc.box():
                # This control-flow block will contain only the same operation as the previous
                # block, but it will also mark qubit 2 as "used" by the box.
                qc.cx(0, 1)
                qc.noop(2)

        Args:
            *qargs: variadic list of valid qubit specifiers.  Anything that can be passed as a qubit
                or collection of qubits is valid for each argument here.

        Raises:
            CircuitError: if any requested qubit is not valid for the circuit.
        """
        scope = self._current_scope()
        for qarg in qargs:
            for qubit in self._qbit_argument_conversion(qarg):
                # It doesn't matter if we pass duplicates along here, and the inner scope is going
                # to have to hash them to check anyway, so no point de-duplicating.
                scope.use_qubit(qubit)

    def _current_scope(self) -> CircuitScopeInterface:
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1]
        return self._builder_api

    def _push_scope(
        self,
        qubits: Iterable[Qubit] = (),
        clbits: Iterable[Clbit] = (),
        registers: Iterable[Register] = (),
        allow_jumps: bool = True,
        forbidden_message: str | None = None,
        loop_var: expr.Var | None = None,
    ) -> int:
        """Add a scope for collecting instructions into this circuit.

        This should only be done by the control-flow context managers, which will handle cleaning up
        after themselves at the end as well.

        Args:
            qubits: Any qubits that this scope should automatically use.
            clbits: Any clbits that this scope should automatically use.
            registers: Any registers that this scope should automatically use.
            allow_jumps: Whether this scope allows jumps to be used within it.
            forbidden_message: If given, all attempts to add instructions to this scope will raise a
                :exc:`.CircuitError` with this message.
            loop_var: If given, the runtime loop variable of a for-loop block.  This should not be
                set for the legacy :class:`.Parameter` form.

        Returns:
            the depth of control-flow scopes (after the push)
        """
        self._control_flow_scopes.append(
            ControlFlowBuilderBlock(
                qubits,
                clbits,
                parent=self._current_scope(),
                registers=registers,
                allow_jumps=allow_jumps,
                forbidden_message=forbidden_message,
                loop_var=loop_var,
            )
        )
        return len(self._control_flow_scopes)

    def _pop_scope(self) -> ControlFlowBuilderBlock:
        """Finish a scope used in the control-flow builder interface, and return it to the caller.

        This should only be done by the control-flow context managers, since they naturally
        synchronize the creation and deletion of stack elements."""
        return self._control_flow_scopes.pop()

    def _peek_previous_instruction_in_scope(self) -> CircuitInstruction:
        """Return the instruction 3-tuple of the most recent instruction in the current scope, even
        if that scope is currently under construction.

        This function is only intended for use by the control-flow ``if``-statement builders, which
        may need to modify a previous instruction."""
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].peek()
        if not self._data:
            raise CircuitError("This circuit contains no instructions.")
        return self._data[-1]

    def _pop_previous_instruction_in_scope(self) -> CircuitInstruction:
        """Return the instruction 3-tuple of the most recent instruction in the current scope, even
        if that scope is currently under construction, and remove it from that scope.

        This function is only intended for use by the control-flow ``if``-statement builders, which
        may need to replace a previous instruction with another.
        """
        if self._control_flow_scopes:
            return self._control_flow_scopes[-1].pop()
        if not self._data:
            raise CircuitError("This circuit contains no instructions.")
        instruction = self._data.pop()
        return instruction

    def box(
        self,
        body_or_annotations: QuantumCircuit | typing.Iterable[Annotation] = ...,
        /,
        qubits: Sequence[QubitSpecifier] | None = None,
        clbits: Sequence[ClbitSpecifier] | None = None,
        *,
        label: str | None = None,
        duration: None = None,
        unit: Literal["dt", "s", "ms", "us", "ns", "ps", "expr"] | None = None,
        annotations: typing.Iterable[Annotation] = ...,
    ):
        """Create a ``box`` of operations on this circuit that are treated atomically in the greater
        context.

        A "box" is a control-flow construct that is entered unconditionally.  The contents of the
        box behave somewhat as if the start and end of the box were barriers (see :meth:`barrier`),
        except it is permissible to commute operations "all the way" through the box.  The box is
        also an explicit scope for the purposes of variables, stretches and compiler passes.

        There are two forms for calling this function:

        * Pass a :class:`QuantumCircuit` positionally, and the ``qubits`` and ``clbits`` it acts
          on.  In this form, a :class:`.BoxOp` is immediately created and appended using the circuit
          as the body.

        * Use in a ``with`` statement with no ``body``, ``qubits`` or ``clbits``.  This is the
          "builder-interface form", where you then use other :class:`QuantumCircuit` methods within
          the Python ``with`` scope to add instructions to the ``box``.  This is the preferred form,
          and much less error prone.

        Examples:

            Using the builder interface to add two boxes in sequence.  The two boxes in this circuit
            can execute concurrently, and the second explicitly inserts a data-flow dependency on
            qubit 8 for the duration of the box, even though the qubit is idle.

            .. code-block:: python

                from qiskit.circuit import QuantumCircuit, Annotation

                class MyAnnotation(Annotation):
                    namespace = "my.namespace"

                qc = QuantumCircuit(9)
                with qc.box():
                    qc.cz(0, 1)
                    qc.cz(2, 3)
                with qc.box([MyAnnotation()]):
                    qc.cz(4, 5)
                    qc.cz(6, 7)
                    qc.noop(8)

            Using the explicit construction of box.  This creates the same circuit as above, and
            should give an indication why the previous form is preferred for interactive use.

            .. code-block:: python

                from qiskit.circuit import QuantumCircuit, BoxOp

                body_0 = QuantumCircuit(4)
                body_0.cz(0, 1)
                body_0.cz(2, 3)

                # Note that the qubit indices inside a body related only to the body.  The
                # association with qubits in the containing circuit is made by the ``qubits``
                # argument to `QuantumCircuit.box`.
                body_1 = QuantumCircuit(5)
                body_1.cz(0, 1)
                body_1.cz(2, 3)

                qc = QuantumCircuit(9)
                qc.box(body_0, [0, 1, 2, 3], [])
                qc.box(body_1, [4, 5, 6, 7, 8], [])

        Args:
            body_or_annotations: the first positional argument is unnamed.  If a
                :class:`QuantumCircuit` is passed positionally, it is immediately used as the body
                of the box, and ``qubits`` and ``clbits`` must also be specified.  If not given, or
                if given an iterable of :class:`.Annotation` objects, the context-manager form of
                this method is triggered.
            qubits: the qubits to apply the :class:`.BoxOp` to, in the explicit form.
            clbits: the clbits to apply the :class:`.BoxOp` to, in the explicit form.
            label: an optional string label for the instruction.
            duration: an optional explicit duration for the :class:`.BoxOp`.  Scheduling passes are
                constrained to schedule the contained scope to match a given duration, including
                delay insertion if required.
            unit: the unit of the ``duration``.
            annotations: any :class:`.Annotation` objects the box should have.  When this method is
                used in context-manager form, this argument can instead be passed as the only
                positional argument.
        """
        if isinstance(body_or_annotations, QuantumCircuit):
            # Explicit-body form.
            body = body_or_annotations
            if qubits is None or clbits is None:
                raise CircuitError("When using 'box' with a body, you must pass qubits and clbits.")
            if annotations is Ellipsis:
                annotations = []
            return self.append(
                BoxOp(body, duration=duration, unit=unit, label=label, annotations=annotations),
                qubits,
                clbits,
                copy=False,
            )
        if body_or_annotations is ...:
            annotations = () if annotations is ... else annotations
        elif annotations is not ...:
            raise TypeError("QuantumCircuit.box() got multiple values for argument 'annotations'")
        else:
            annotations = body_or_annotations
        # Context-manager form.
        if qubits is not None or clbits is not None:
            raise CircuitError(
                "When using 'box' as a context manager, you cannot pass qubits or clbits."
            )
        return BoxContext(self, duration=duration, unit=unit, label=label, annotations=annotations)

    @typing.overload
    def while_loop(
        self,
        condition: tuple[ClassicalRegister | Clbit, int] | expr.Expr,
        body: None,
        qubits: None,
        clbits: None,
        *,
        label: str | None,
    ) -> WhileLoopContext: ...

    @typing.overload
    def while_loop(
        self,
        condition: tuple[ClassicalRegister | Clbit, int] | expr.Expr,
        body: QuantumCircuit,
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: str | None,
    ) -> InstructionSet: ...

    def while_loop(self, condition, body=None, qubits=None, clbits=None, *, label=None):
        """Create a ``while`` loop on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a
        :obj:`~qiskit.circuit.controlflow.WhileLoopOp` with the given ``body``.  If ``body`` (and
        ``qubits`` and ``clbits``) are *not* passed, then this acts as a context manager, which
        will automatically build a :obj:`~qiskit.circuit.controlflow.WhileLoopOp` when the scope
        finishes.  In this form, you do not need to keep track of the qubits or clbits you are
        using, because the scope will handle it for you.

        Example usage::

            from qiskit.circuit import QuantumCircuit, Clbit, Qubit
            bits = [Qubit(), Qubit(), Clbit()]
            qc = QuantumCircuit(bits)

            with qc.while_loop((bits[2], 0)):
                qc.h(0)
                qc.cx(0, 1)
                qc.measure(0, 0)

        Args:
            condition (Tuple[Union[ClassicalRegister, Clbit], int]): An equality condition to be
                checked prior to executing ``body``. The left-hand side of the condition must be a
                :obj:`~ClassicalRegister` or a :obj:`~Clbit`, and the right-hand side must be an
                integer or boolean.
            body (Optional[QuantumCircuit]): The loop body to be repeatedly executed.  Omit this to
                use the context-manager mode.
            qubits (Optional[Sequence[Qubit]]): The circuit qubits over which the loop body should
                be run.  Omit this to use the context-manager mode.
            clbits (Optional[Sequence[Clbit]]): The circuit clbits over which the loop body should
                be run.  Omit this to use the context-manager mode.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or WhileLoopContext: If used in context-manager mode, then this should be
            used as a ``with`` resource, which will infer the block content and operands on exit.
            If the full form is used, then this returns a handle to the instructions created.

        Raises:
            CircuitError: if an incorrect calling convention is used.
        """
        circuit_scope = self._current_scope()
        if isinstance(condition, expr.Expr):
            condition = _validate_expr(circuit_scope, condition)
        else:
            condition = (circuit_scope.resolve_classical_resource(condition[0]), condition[1])

        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'while_loop' as a context manager,"
                    " you cannot pass qubits or clbits."
                )
            return WhileLoopContext(self, condition, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError(
                "When using 'while_loop' with a body, you must pass qubits and clbits."
            )

        return self.append(WhileLoopOp(condition, body, label), qubits, clbits, copy=False)

    @typing.overload
    def for_loop(
        self,
        indexset: Iterable[int],
        loop_parameter: Parameter | expr.Var | None,
        body: None,
        qubits: None,
        clbits: None,
        *,
        label: str | None,
    ) -> ForLoopContext: ...

    @typing.overload
    def for_loop(
        self,
        indexset: Iterable[int],
        loop_parameter: Parameter | expr.Var | None,
        body: QuantumCircuit,
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: str | None,
    ) -> InstructionSet: ...

    def for_loop(
        self,
        indexset,
        loop_parameter=None,
        body=None,
        qubits=None,
        clbits=None,
        *,
        label=None,
    ):
        """Create a ``for`` loop on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a
        :class:`~qiskit.circuit.ForLoopOp` with the given ``body``.  If ``body`` (and
        ``qubits`` and ``clbits``) are *not* passed, then this acts as a context manager, which,
        when entered, provides a loop variable (unless one is given, in which case it will be
        reused) and will automatically build a :class:`~qiskit.circuit.ForLoopOp` when the
        scope finishes.  In this form, you do not need to keep track of the qubits or clbits you are
        using, because the scope will handle it for you.

        For example::

            from qiskit import QuantumCircuit
            qc = QuantumCircuit(2, 1)

            with qc.for_loop(range(5)) as i:
                qc.h(0)
                qc.cx(0, 1)
                qc.measure(0, 0)
                with qc.if_test((0, True)):
                    qc.break_loop()

        Args:
            indexset (Iterable[int]): A collection of integers to loop over.  Always necessary.
            loop_parameter (Optional[Parameter|expr.Var]): The parameter used within ``body`` to which
                the values from ``indexset`` will be assigned.  In the context-manager form, if this
                argument is not supplied, then a loop parameter will be allocated for you and
                returned as the value of the ``with`` statement.  This will only be bound into the
                circuit if it is used within the body.

                If this argument is ``None`` in the manual form of this method, ``body`` will be
                repeated once for each of the items in ``indexset`` but their values will be
                ignored.
            body (Optional[QuantumCircuit]): The loop body to be repeatedly executed.  Omit this to
                use the context-manager mode.
            qubits (Optional[Sequence[QubitSpecifier]]): The circuit qubits over which the loop body
                should be run.  Omit this to use the context-manager mode.
            clbits (Optional[Sequence[ClbitSpecifier]]): The circuit clbits over which the loop body
                should be run.  Omit this to use the context-manager mode.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or ForLoopContext: depending on the call signature, either a context
            manager for creating the for loop (it will automatically be added to the circuit at the
            end of the block), or an :obj:`~InstructionSet` handle to the appended loop operation.

        Raises:
            CircuitError: if an incorrect calling convention is used.
        """
        if body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'for_loop' as a context manager, you cannot pass qubits or clbits."
                )
            return ForLoopContext(self, indexset, loop_parameter, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError(
                "When using 'for_loop' with a body, you must pass qubits and clbits."
            )

        return self.append(
            ForLoopOp(indexset, loop_parameter, body, label), qubits, clbits, copy=False
        )

    @typing.overload
    def if_test(self, condition: tuple[ClassicalRegister | Clbit, int]) -> IfContext: ...

    @typing.overload
    def if_test(
        self,
        condition: tuple[ClassicalRegister | Clbit, int],
        true_body: QuantumCircuit,
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: str | None = None,
    ) -> InstructionSet: ...

    def if_test(
        self,
        condition,
        true_body=None,
        qubits=None,
        clbits=None,
        *,
        label=None,
    ):
        """Create an ``if`` statement on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a
        :obj:`~qiskit.circuit.IfElseOp` with the given ``true_body``, and there will be
        no branch for the ``false`` condition (see also the :meth:`.if_else` method).  However, if
        ``true_body`` (and ``qubits`` and ``clbits``) are *not* passed, then this acts as a context
        manager, which can be used to build ``if`` statements.  The return value of the ``with``
        statement is a chainable context manager, which can be used to create subsequent ``else``
        blocks.  In this form, you do not need to keep track of the qubits or clbits you are using,
        because the scope will handle it for you.

        For example::

            from qiskit.circuit import QuantumCircuit, Qubit, Clbit
            bits = [Qubit(), Qubit(), Qubit(), Clbit(), Clbit()]
            qc = QuantumCircuit(bits)

            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 1)

            with qc.if_test((bits[3], 0)) as else_:
                qc.x(2)
            with else_:
                qc.h(2)
                qc.z(2)

        Args:
            condition (Tuple[Union[ClassicalRegister, Clbit], int]): A condition to be evaluated in
                real time during circuit execution, which, if true, will trigger the evaluation of
                ``true_body``. Can be specified as either a tuple of a ``ClassicalRegister`` to be
                tested for equality with a given ``int``, or as a tuple of a ``Clbit`` to be
                compared to either a ``bool`` or an ``int``.
            true_body (Optional[QuantumCircuit]): The circuit body to be run if ``condition`` is
                true.
            qubits (Optional[Sequence[QubitSpecifier]]): The circuit qubits over which the if/else
                should be run.
            clbits (Optional[Sequence[ClbitSpecifier]]): The circuit clbits over which the if/else
                should be run.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or IfContext: depending on the call signature, either a context
            manager for creating the ``if`` block (it will automatically be added to the circuit at
            the end of the block), or an :obj:`~InstructionSet` handle to the appended conditional
            operation.

        Raises:
            CircuitError: If the provided condition references Clbits outside the
                enclosing circuit.
            CircuitError: if an incorrect calling convention is used.

        Returns:
            A handle to the instruction created.
        """
        circuit_scope = self._current_scope()
        if isinstance(condition, expr.Expr):
            condition = _validate_expr(circuit_scope, condition)
        else:
            condition = (circuit_scope.resolve_classical_resource(condition[0]), condition[1])

        if true_body is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'if_test' as a context manager, you cannot pass qubits or clbits."
                )
            # We can only allow jumps if we're in a loop block, but the default path (no scopes)
            # also allows adding jumps to support the more verbose internal mode.
            in_loop = bool(self._control_flow_scopes and self._control_flow_scopes[-1].allow_jumps)
            return IfContext(self, condition, in_loop=in_loop, label=label)
        elif qubits is None or clbits is None:
            raise CircuitError("When using 'if_test' with a body, you must pass qubits and clbits.")

        return self.append(IfElseOp(condition, true_body, None, label), qubits, clbits, copy=False)

    def if_else(
        self,
        condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | tuple[Clbit, bool],
        true_body: QuantumCircuit,
        false_body: QuantumCircuit,
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        label: str | None = None,
    ) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.IfElseOp`.

        .. note::

            This method does not have an associated context-manager form, because it is already
            handled by the :meth:`.if_test` method.  You can use the ``else`` part of that with
            something such as::

                from qiskit.circuit import QuantumCircuit, Qubit, Clbit
                bits = [Qubit(), Qubit(), Clbit()]
                qc = QuantumCircuit(bits)
                qc.h(0)
                qc.cx(0, 1)
                qc.measure(0, 0)
                with qc.if_test((bits[2], 0)) as else_:
                    qc.h(0)
                with else_:
                    qc.x(0)

        Args:
            condition: A condition to be evaluated in real time at circuit execution, which,
                if true, will trigger the evaluation of ``true_body``. Can be
                specified as either a tuple of a ``ClassicalRegister`` to be
                tested for equality with a given ``int``, or as a tuple of a
                ``Clbit`` to be compared to either a ``bool`` or an ``int``.
            true_body: The circuit body to be run if ``condition`` is true.
            false_body: The circuit to be run if ``condition`` is false.
            qubits: The circuit qubits over which the if/else should be run.
            clbits: The circuit clbits over which the if/else should be run.
            label: The string label of the instruction in the circuit.

        Raises:
            CircuitError: If the provided condition references Clbits outside the
                enclosing circuit.

        Returns:
            A handle to the instruction created.
        """
        circuit_scope = self._current_scope()
        if isinstance(condition, expr.Expr):
            condition = _validate_expr(circuit_scope, condition)
        else:
            condition = (circuit_scope.resolve_classical_resource(condition[0]), condition[1])

        return self.append(
            IfElseOp(condition, true_body, false_body, label), qubits, clbits, copy=False
        )

    @typing.overload
    def switch(
        self,
        target: ClbitSpecifier | ClassicalRegister,
        cases: None,
        qubits: None,
        clbits: None,
        *,
        label: str | None,
    ) -> SwitchContext: ...

    @typing.overload
    def switch(
        self,
        target: ClbitSpecifier | ClassicalRegister,
        cases: Iterable[tuple[typing.Any, QuantumCircuit]],
        qubits: Sequence[QubitSpecifier],
        clbits: Sequence[ClbitSpecifier],
        *,
        label: str | None,
    ) -> InstructionSet: ...

    def switch(self, target, cases=None, qubits=None, clbits=None, *, label=None):
        """Create a ``switch``/``case`` structure on this circuit.

        There are two forms for calling this function.  If called with all its arguments (with the
        possible exception of ``label``), it will create a :class:`.SwitchCaseOp` with the given
        case structure.  If ``cases`` (and ``qubits`` and ``clbits``) are *not* passed, then this
        acts as a context manager, which will automatically build a :class:`.SwitchCaseOp` when the
        scope finishes.  In this form, you do not need to keep track of the qubits or clbits you are
        using, because the scope will handle it for you.

        Example usage::

            from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
            qreg = QuantumRegister(3)
            creg = ClassicalRegister(3)
            qc = QuantumCircuit(qreg, creg)
            qc.h([0, 1, 2])
            qc.measure([0, 1, 2], [0, 1, 2])

            with qc.switch(creg) as case:
                with case(0):
                    qc.x(0)
                with case(1, 2):
                    qc.z(1)
                with case(case.DEFAULT):
                    qc.cx(0, 1)

        Args:
            target (Union[ClassicalRegister, Clbit]): The classical value to switch one.  This must
                be integer-like.
            cases (Iterable[Tuple[typing.Any, QuantumCircuit]]): A sequence of case specifiers.
                Each tuple defines one case body (the second item).  The first item of the tuple can
                be either a single integer value, the special value :data:`.CASE_DEFAULT`, or a
                tuple of several integer values.  Each of the integer values will be tried in turn;
                control will then pass to the body corresponding to the first match.
                :data:`.CASE_DEFAULT` matches all possible values.  Omit in context-manager form.
            qubits (Sequence[Qubit]): The circuit qubits over which all case bodies execute. Omit in
                context-manager form.
            clbits (Sequence[Clbit]): The circuit clbits over which all case bodies execute. Omit in
                context-manager form.
            label (Optional[str]): The string label of the instruction in the circuit.

        Returns:
            InstructionSet or SwitchCaseContext: If used in context-manager mode, then this should
            be used as a ``with`` resource, which will return an object that can be repeatedly
            entered to produce cases for the switch statement.  If the full form is used, then this
            returns a handle to the instructions created.

        Raises:
            CircuitError: if an incorrect calling convention is used.
        """

        circuit_scope = self._current_scope()
        if isinstance(target, expr.Expr):
            target = _validate_expr(circuit_scope, target)
        else:
            target = circuit_scope.resolve_classical_resource(target)
        if cases is None:
            if qubits is not None or clbits is not None:
                raise CircuitError(
                    "When using 'switch' as a context manager, you cannot pass qubits or clbits."
                )
            in_loop = bool(self._control_flow_scopes and self._control_flow_scopes[-1].allow_jumps)
            return SwitchContext(self, target, in_loop=in_loop, label=label)

        if qubits is None or clbits is None:
            raise CircuitError("When using 'switch' with cases, you must pass qubits and clbits.")
        return self.append(SwitchCaseOp(target, cases, label=label), qubits, clbits, copy=False)

    def break_loop(self) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.BreakLoopOp`.

        .. warning::

            If you are using the context-manager "builder" forms of :meth:`.if_test`,
            :meth:`.for_loop` or :meth:`.while_loop`, you can only call this method if you are
            within a loop context, because otherwise the "resource width" of the operation cannot be
            determined.  This would quickly lead to invalid circuits, and so if you are trying to
            construct a reusable loop body (without the context managers), you must also use the
            non-context-manager form of :meth:`.if_test` and :meth:`.if_else`.  Take care that the
            :obj:`.BreakLoopOp` instruction must span all the resources of its containing loop, not
            just the immediate scope.

        Returns:
            A handle to the instruction created.

        Raises:
            CircuitError: if this method was called within a builder context, but not contained
                within a loop.
        """
        if self._control_flow_scopes:
            operation = BreakLoopPlaceholder()
            resources = operation.placeholder_resources()
            return self.append(operation, resources.qubits, resources.clbits, copy=False)
        return self.append(
            BreakLoopOp(self.num_qubits, self.num_clbits), self.qubits, self.clbits, copy=False
        )

    def continue_loop(self) -> InstructionSet:
        """Apply :class:`~qiskit.circuit.ContinueLoopOp`.

        .. warning::

            If you are using the context-manager "builder" forms of :meth:`.if_test`,
            :meth:`.for_loop` or :meth:`.while_loop`, you can only call this method if you are
            within a loop context, because otherwise the "resource width" of the operation cannot be
            determined.  This would quickly lead to invalid circuits, and so if you are trying to
            construct a reusable loop body (without the context managers), you must also use the
            non-context-manager form of :meth:`.if_test` and :meth:`.if_else`.  Take care that the
            :class:`~qiskit.circuit.ContinueLoopOp` instruction must span all the resources of its
            containing loop, not just the immediate scope.

        Returns:
            A handle to the instruction created.

        Raises:
            CircuitError: if this method was called within a builder context, but not contained
                within a loop.
        """
        if self._control_flow_scopes:
            operation = ContinueLoopPlaceholder()
            resources = operation.placeholder_resources()
            return self.append(operation, resources.qubits, resources.clbits, copy=False)
        return self.append(
            ContinueLoopOp(self.num_qubits, self.num_clbits), self.qubits, self.clbits, copy=False
        )

    # Functions only for scheduled circuits
    def qubit_duration(self, *qubits: Qubit | int) -> float:
        """Return the duration between the start and stop time of the first and last instructions,
        excluding delays, over the supplied qubits. Its time unit is ``self.unit``.

        Args:
            *qubits: Qubits within ``self`` to include.

        Returns:
            Return the duration between the first start and last stop time of non-delay instructions
        """
        return self.qubit_stop_time(*qubits) - self.qubit_start_time(*qubits)

    def qubit_start_time(self, *qubits: Qubit | int) -> float:
        """Return the start time of the first instruction, excluding delays,
        over the supplied qubits. Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the start time of the first instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self._duration is None:
            # circuit has only delays, this is kind of scheduled
            for instruction in self._data:
                if not isinstance(instruction.operation, Delay):
                    raise CircuitError(
                        "qubit_start_time undefined. Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        starts = dict.fromkeys(qubits, 0)
        dones = dict.fromkeys(qubits, False)
        for instruction in self._data:
            for q in qubits:
                if q in instruction.qubits:
                    if isinstance(instruction.operation, Delay):
                        if not dones[q]:
                            starts[q] += instruction.operation.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return min(start for start in starts.values())

        return 0  # If there are no instructions over bits

    def qubit_stop_time(self, *qubits: Qubit | int) -> float:
        """Return the stop time of the last instruction, excluding delays, over the supplied qubits.
        Its time unit is ``self.unit``.

        Return 0 if there are no instructions over qubits

        Args:
            *qubits: Qubits within ``self`` to include. Integers are allowed for qubits, indicating
            indices of ``self.qubits``.

        Returns:
            Return the stop time of the last instruction, excluding delays, over the qubits

        Raises:
            CircuitError: if ``self`` is a not-yet scheduled circuit.
        """
        if self._duration is None:
            # circuit has only delays, this is kind of scheduled
            for instruction in self._data:
                if not isinstance(instruction.operation, Delay):
                    raise CircuitError(
                        "qubit_stop_time undefined. Circuit must be scheduled first."
                    )
            return 0

        qubits = [self.qubits[q] if isinstance(q, int) else q for q in qubits]

        stops = dict.fromkeys(qubits, self._duration)
        dones = dict.fromkeys(qubits, False)
        for instruction in reversed(self._data):
            for q in qubits:
                if q in instruction.qubits:
                    if isinstance(instruction.operation, Delay):
                        if not dones[q]:
                            stops[q] -= instruction.operation.duration
                    else:
                        dones[q] = True
            if len(qubits) == len([done for done in dones.values() if done]):  # all done
                return max(stop for stop in stops.values())

        if len(stops) > 0:  # not all but some qubits has instructions
            return max(stops.values())
        else:
            return 0  # If there are no instructions over bits

    def estimate_duration(self, target, unit: str = "s") -> int | float:
        """Estimate the duration of a scheduled circuit

        This method computes the estimate of the circuit duration by finding
        the longest duration path in the circuit based on the durations
        provided by a given target. This method only works for simple circuits
        that have no control flow or other classical feed-forward operations.

        Args:
            target (Target): The :class:`.Target` instance that contains durations for
                the instructions if the target is missing duration data for any of the
                instructions in the circuit an :class:`.QiskitError` will be raised. This
                should be the same target object used as the target for transpilation.
            unit: The unit to return the duration in. This defaults to "s" for seconds
                but this can be a supported SI prefix for seconds returns. For example
                setting this to "n" will return in unit of nanoseconds. Supported values
                of this type are "f", "p", "n", "u", "µ", "m", "k", "M", "G", "T", and
                "P". Additionally, a value of "dt" is also accepted to output an integer
                in units of "dt". For this to function "dt" must be specified in the
                ``target``.

        Returns:
            The estimated duration for the execution of a single shot of the circuit in
            the specified unit.

        Raises:
            QiskitError: If the circuit is not scheduled or contains other
                details that prevent computing an estimated duration from
                (such as parameterized delay).
        """
        from qiskit.converters import circuit_to_dag

        dur = compute_estimated_duration(circuit_to_dag(self), target)
        if unit == "s":
            return dur
        if unit == "dt":
            from qiskit.circuit.duration import duration_in_dt

            return duration_in_dt(dur, target.dt)

        prefix_dict = {
            "f": 1e-15,
            "p": 1e-12,
            "n": 1e-9,
            "u": 1e-6,
            "µ": 1e-6,
            "m": 1e-3,
            "k": 1e3,
            "M": 1e6,
            "G": 1e9,
            "T": 1e12,
            "P": 1e15,
        }
        if unit not in prefix_dict:
            raise QiskitError(
                f"Specified unit: {unit} is not a valid/supported SI prefix, 's', or 'dt'"
            )
        return dur / prefix_dict[unit]

    def estimate_fidelity(self, target) -> float | None:
        """Estimate the fidelity of a physical circuit.

        This function will compute the product of the error rates for each
        gate in the circuit to estimate the fidelity of the circuit:

        ..math::
          :label: estimated circuit fidelity

          \\prod_{g \\in \\text{gates}} \\bigl(1 - \\operatorname{error}(g)\\bigr)

        where :math:`\\operatorname{error}(g)` is the error rate in the target for the instruction :math:`g` from the circuit
        in the target. If the circuit is not physical, meaning any instruction in the circuit (as
        in operation and qargs) is not found in the target, this will return ``None``. This method is not
        intended to compute a realistic simulation of the fidelity of execution on real hardware. It is
        designed to provide an estimate of how the transpiler would work with the fidelity for various
        heuristics in its operation. It is typically only useful for comparing different compilation
        outputs against each other to estimate which one would produce a better quality execution on
        hardware.

        Args:
            target (Target): The :class:`.Target` instance that the circuit will be executed on and used to
                get the error rates for the instructions in the circuit.

        Returns:
            The estimated fidelity of executing the circuit on the given target. If any
            instruction in the circuit is not present in target

        Raises:
            QiskitError: If the circuit contains any control flow operations.
        """
        if self.has_control_flow_op():
            raise CircuitError(
                "The circuit contains control flow, this method cannot estimate the fidelity with control flow"
            )
        return estimate_fidelity(self._data, target)


class _OuterCircuitScopeInterface(CircuitScopeInterface):
    # This is an explicit interface-fulfilling object friend of QuantumCircuit that acts as its
    # implementation of the control-flow builder scope methods.

    __slots__ = ("circuit",)

    def __init__(self, circuit: QuantumCircuit):
        self.circuit = circuit

    @property
    def instructions(self):
        return self.circuit._data

    def append(self, instruction, *, _standard_gate: bool = False):
        # QuantumCircuit._append is semi-public, so we just call back to it.
        return self.circuit._append(instruction, _standard_gate=_standard_gate)

    def extend(
        self,
        data: CircuitData,
        qubits: list[Qubit] | None = None,
        clbits: list[Clbit] | None = None,
    ):
        qubits = None if qubits is None else [self.circuit.find_bit(q).index for q in qubits]
        clbits = None if clbits is None else [self.circuit.find_bit(c).index for c in clbits]
        self.circuit._data.native_extend(data, qubits=qubits, clbits=clbits)
        self.circuit.duration = None
        self.circuit.unit = "dt"

    def resolve_classical_resource(self, specifier):
        # This is slightly different to cbit_argument_conversion, because it should not
        # unwrap :obj:`.ClassicalRegister` instances into lists, and in general it should not allow
        # iterables or broadcasting.
        if isinstance(specifier, Clbit):
            if specifier not in self.circuit._clbit_indices:
                raise CircuitError(f"Clbit {specifier} is not present in this circuit.")
            return specifier
        if isinstance(specifier, ClassicalRegister):
            # This is linear complexity for something that should be constant, but QuantumCircuit
            # does not currently keep a hashmap of registers, and requires non-trivial changes to
            # how it exposes its registers publicly before such a map can be safely stored so it
            # doesn't miss updates. (Jake, 2021-11-10).
            if specifier not in self.circuit.cregs:
                raise CircuitError(f"Register {specifier} is not present in this circuit.")
            return specifier
        if isinstance(specifier, int):
            try:
                return self.circuit._data.clbits[specifier]
            except IndexError:
                raise CircuitError(f"Classical bit index {specifier} is out-of-range.") from None
        raise CircuitError(f"Unknown classical resource specifier: '{specifier}'.")

    def add_uninitialized_var(self, var):
        var = self.circuit._prepare_new_var(var, None)
        self.circuit._data.add_declared_var(var)

    def add_stretch(self, stretch):
        stretch = self.circuit._prepare_new_stretch(stretch)
        self.circuit._data.add_declared_stretch(stretch)

    def get_var(self, name):
        return self.circuit._data.get_var(name)

    def get_stretch(self, name):
        return self.circuit._data.get_stretch(name)

    def use_var(self, var):
        if self.get_var(var.name) != var:
            raise CircuitError(f"'{var}' is not present in this circuit")

    def use_stretch(self, stretch):
        if self.get_stretch(stretch.name) != stretch:
            raise CircuitError(f"'{stretch}' is not present in this circuit")

    def use_qubit(self, qubit):
        # Since the qubit is guaranteed valid, there's nothing for us to do.
        pass


def _validate_expr(circuit_scope: CircuitScopeInterface, node: expr.Expr) -> expr.Expr:
    # This takes the `circuit_scope` object as an argument rather than being a circuit method and
    # inferring it because we may want to call this several times, and we almost invariably already
    # need the interface implementation for something else anyway.
    # If we're not in a capturing scope (i.e. we're in the root scope), then the
    # `use_{var,stretch}` calls are no-ops.
    for ident in set(expr.iter_identifiers(node)):
        if isinstance(ident, expr.Stretch):
            circuit_scope.use_stretch(ident)
        elif ident.standalone:
            circuit_scope.use_var(ident)
        else:
            circuit_scope.resolve_classical_resource(ident.var)
    return node


def _copy_metadata(original, cpy):
    # copy registers correctly, in copy.copy they are only copied via reference
    cpy._builder_api = _OuterCircuitScopeInterface(cpy)
    cpy._ancillas = original._ancillas.copy()
    cpy._metadata = _copy.deepcopy(original._metadata)
