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

r"""
=============================================
Quantum circuit model (:mod:`qiskit.circuit`)
=============================================

.. currentmodule:: qiskit.circuit

The fundamental element of quantum computing is the *quantum circuit*.  This is a computational
routine that can be run, one shot at a time, on a quantum processing unit (QPU).  A circuit will act
on a predefined amount of quantum data (in Qiskit, we only directly support qubits) with unitary
operations (gates), measurements and resets.  In addition, a quantum circuit can contain operations
on classical data, including real-time computations and control-flow constructs, which are executed
by the controllers of the QPU.

.. note::

    You may wish to skip the introductory material and jump directly to:

    * :ref:`the API overview of the whole circuit module <circuit-module-api>`
    * :ref:`the detailed discussion about how circuits are represented <circuit-repr>`
    * the core :class:`QuantumCircuit` class for how to build and query circuits
    * :ref:`information on construction custom instructions <circuit-custom-gates>`
    * :ref:`ways to work with circuit-level objects <circuit-working-with>`
    * :ref:`discussion of Qiskit conventions for circuits, matrices and state labelling
      <circuit-conventions>`

Circuits are at a low level of abstraction when building up quantum programs.  They are the
construct that is used to build up to higher levels of abstraction, such as the :ref:`primitives of
quantum computation <qiskit-primitives>`, which accumulate data from many shots of quantum-circuit
execution, along with advanced error-mitigation techniques and measurement optimizations, into
well-typed classical data and error statistics.

In Qiskit, circuits can be :ref:`defined in one of two regimes <circuit-abstract-physical>`:

* an *abstract* circuit, which is defined in terms of *virtual qubits* and arbitrary high-level
  operations, like encapsulated algorithms and user-defined gates.

* a *physical* circuit, which is defined in terms of the *hardware qubits* of one particular
  backend, and contains only operations that this backend natively supports.  You might also see
  this concept referred to as an *ISA circuit*.

You convert from an abstract circuit to a physical circuit by using :ref:`Qiskit's transpilation
package <qiskit-transpiler>`, of which the top-level access point is :func:`.transpile`.  If you
define a circuit, where you intend the qubit indices to refer to physical qubits, you can use
:meth:`.QuantumCircuit.ensure_physical` to rewrite the circuit's metadata to ensure that Qiskit
recognizes the circuit as a physical circuit, though unlike transpilation, this does not enforce
that the basis-gates and hardware-coupling constraints will be respected.

In Qiskit, a quantum circuit is represented by the :class:`QuantumCircuit` class.  Below is an
example of a quantum circuit that makes a three-qubit Greenberger–Horne–Zeilinger (GHZ) state
defined as:

.. math::

   |\psi\rangle = \left( |000\rangle + |111\rangle \right) / \sqrt{2}


.. plot::
   :alt: Circuit diagram output by the previous code.
   :include-source:

   from qiskit import QuantumCircuit

   # Create a circuit with a register of three qubits
   circ = QuantumCircuit(3)
   # H gate on qubit 0, putting this qubit in a superposition of |0> + |1>.
   circ.h(0)
   # A CX (CNOT) gate on control qubit 0 and target qubit 1 generating a Bell state.
   circ.cx(0, 1)
   # CX (CNOT) gate on control qubit 0 and target qubit 2 resulting in a GHZ state.
   circ.cx(0, 2)
   # Draw the circuit
   circ.draw('mpl')


.. _circuit-definitions:

Circuit concepts and definitions
================================

There is a lot of specialized terminology around quantum circuits.  Much of this is common in
quantum-computing literature, while some is more specific to quantum software packages, and a small
amount specific to Qiskit.  This is an alphabetical listing of some of the important concepts as a
quick reference, but does not go into detail of the foundational concepts.  Consider using the `IBM
Quantum Learning platform <https://learning.quantum.ibm.com/>`_ if you want to start from the
beginning.

abstract circuit
    A *circuit* defined in terms of abstract mathematical operations and *virtual qubits*.  This is
    typically how you think about quantum algorithms; an abstract circuit can be made up of
    completely arbitrary unitary operations, measurements, and potentially *real-time classical
    computation*, with no restrictions about which qubits can interact with each other.

    You turn an abstract circuit into a *physical circuit* by using :ref:`Qiskit's transpilation
    package <qiskit-transpiler>`.

ancilla qubit
    An extra qubit that is used to help implement operations on other qubits, but whose final state
    is not important for the program.

circuit
    A computational routine the defines a single execution to be taken on a QPU.  This can either be
    an *abstract circuit* or a *physical circuit*.

clbit
    A Qiskit-specific abbreviation meaning a single classical bit of data.

gate
    A *unitary operation* on one or more qubits.

hardware qubit
    The representation of a single qubit on a particular *QPU*.  A hardware qubit has some physical
    quantum-mechanical system backing it, such as superconducting circuits; unlike a *virtual
    qubit*, it has particular coupling constraints and only certain gates can be applied to certain
    groups of hardware qubits.

    Qiskit does not distinguish *logical qubits* from any individual *physical qubits* when talking
    about hardware qubits.  A QPU may implement its hardware qubits as logical qubits, where each
    hardware qubit comprises many physical qubits that are controlled and error-corrected opaquely
    to Qiskit by the control electronics.  More likely, for near-term applications, a QPU will be
    directly exposing its physical qubits as the hardware qubits for Qiskit to reason about.

    Both physical and logical qubits will have coupling constraints between them, only permit
    certain quantum operations on them, and have scheduling concerns between them.  Qiskit abstracts
    these concerns together in the concept of hardware qubits. In the early days of quantum error
    correction, particular backends may let you access their qubit resources either as high-level
    logical qubits or as low-level physical qubits through Qiskit.

instruction set architecture (ISA)
    The abstract model of which operations are available on which sets of *hardware qubits* on one
    particular *QPU*.  For example, one QPU may allow :math:`\sqrt X` and :math:`R_Z` operations on
    all single hardware qubits, and :math:`CX` operations on certain pairs of hardware qubits.

logical qubit
    A collection of several *physical qubits* that are controlled together by a QPU (from the user's
    perspective) to apply real-time quantum error correction.  A logical qubit is a type of
    *hardware qubit* for Qiskit.

measurement
    The act of extracting one classical bit of a data from a single qubit state.  This is an
    irreversible operation, and usually destroys entanglement and phase coherence between the target
    qubit and the rest of the system.

physical circuit
    A *circuit* defined in terms of *hardware qubits* and only the quantum operations available in a
    particular *QPU's* *ISA*.  Physical circuits are tied to one particular QPU architecture, and
    will not run on other incompatible architectures.  You may also hear this referred to as an *ISA
    circuit*.

    You typically get a physical circuit by using :ref:`Qiskit's transpilation routines
    <qiskit-transpiler>` on an *abstract circuit* that you constructed.

physical qubit
    A controllable two-level quantum system.  This is literally one "physics" qubit, such as a
    transmon or the electronic state of a trapped ion.  A QPU may expose this directly as its
    *hardware qubit*, or combine several physical qubits into a *logical qubit*.

quantum processing unit (QPU)
    Analogous to a CPU in classical computing or a GPU in graphics processing, a QPU is the hardware
    that runs quantum operations on quantum data.  You can always expect a QPU that uses the
    *circuit* model of computation to be able to perform some set of *gates*, and *measurement*
    operations.  Depending on the particular technology, they also may be able to run some real-time
    classical computations as well, such as classical control flow and bitwise calculations on
    classical data.

qubit
    The basic unit of quantum information.

real-time classical computation
    Any classical computation that can happen within the execution of a single shot of a *circuit*,
    where the results of the classical computation can affect later execution of the circuit.  The
    amount of real-time classical computation available with particular *QPU*\ s will vary
    significantly dependent on many factors, such as the controlling electronics and the qubit
    technology in use. You should consult your hardware vendor's documentation for more information
    on this.

unitary operation
    A reversible operation on a quantum state. All quantum *gates* are unitary operations (by
    definition).

virtual qubit
    An abstract, mathematical *qubit* used to build an *abstract circuit*.  Virtual qubits are how
    one typically thinks about quantum algorithms at a high level; we assume that all quantum gates
    are valid on all virtual qubits, and all virtual qubits are always connected to every other
    virtual qubit.

    When mapping to hardware, virtual qubits must be assigned to *hardware qubits*.  This mapping
    need not be one-to-one.  Typically, one virtual qubit will need to be swapped from one hardware
    qubit to another over the course of a circuit execution in order to satisfy coupling constraints
    of the underlying QPU.  It is not strictly necessary for all virtual qubits used in a circuit to
    be mapped to a physical qubit at any given point in a *physical circuit*; it could be that a
    virtual qubit is measured (collapsing its state) and then never used again, so a new virtual
    qubit could take its place.  Evaluating these conditions to map a virtual circuit to a physical
    circuit is the job of :ref:`Qiskit's transpilation package <qiskit-transpiler>`.


.. _circuit-module-api:

API overview of qiskit.circuit
==============================

All objects here are described in more detail, and in their greater context in the following
sections.  This section provides an overview of the API elements documented here.

..
    TODO: actually write the "in-depth section on building circuits and cross-ref to it.

The principal class is :class:`.QuantumCircuit`, which has its own documentation page, including
an in-depth section on building circuits. Quantum data and the simplest classical data are
represented by "bits" and "registers":

* :class:`Bit`, an atom of data
    * :class:`Qubit`
    * :class:`Clbit`
    * :class:`AncillaQubit`

* :class:`Register`, a collection of bits
    * :class:`QuantumRegister`
    * :class:`ClassicalRegister`
    * :class:`AncillaRegister`

Within a circuit, each complete :class:`CircuitInstruction` is made up of an :class:`Operation`
(which might be an :class:`Instruction`, a :class:`Gate`, or some other subclass) and the qubit
and clbit operands.  The core base classes here are:

* :class:`CircuitInstruction`, an operation and its operands
* :class:`InstructionSet`, a temporary handle to a slice of circuit data
* :class:`Operation`, any abstract mathematical object or hardware instruction
    * :class:`AnnotatedOperation`, a subclass with applied abstract modifiers
        * :class:`InverseModifier`
        * :class:`ControlModifier`
        * :class:`PowerModifier`

The most common concrete subclass of the minimal, abstract :class:`Operation` interface is the
:class:`Instruction`.  While :class:`Operation` can include abstract mathematical objects,
an :class:`Instruction` is something that could conceivably run directly on hardware. This is in
turn subclassed by :class:`Gate` and :class:`ControlledGate` that further add unitarity and
controlled semantics on top:

* :class:`Instruction`, representing a hardware-based instruction
* :class:`Gate`, representing a hardware instruction that is unitary
* :class:`ControlledGate`, representing a gate with control structure.

Qiskit includes a large library of standard gates and circuits, which is documented in
:mod:`qiskit.circuit.library`.  Many of these are declared as Python-object singletons.  The
machinery for this is described in detail in :mod:`qiskit.circuit.singleton`, of which the main
classes are each a singleton form of the standard instruction–gate hierarchy:

* :class:`~singleton.SingletonInstruction`
* :class:`~singleton.SingletonGate`
* :class:`~singleton.SingletonControlledGate`

Some instructions are particularly special in that they affect the control flow or data flow of the
circuit.  The top-level ones are:

* :class:`Barrier`, to mark parts of the circuit that should be optimized independently
* :class:`Delay`, to insert a real-time wait period
* :class:`Measure`, to measure a :class:`Qubit` into a :class:`Clbit`
* :class:`Reset`, to irreversibly reset a qubit to the :math:`\lvert0\rangle` state
* :class:`Store`, to write a real-time classical expression to a storage location
* :class:`ControlFlowOp`, which has specific subclasses:
    * :class:`BreakLoopOp`, to break out of the nearest containing loop
    * :class:`ContinueLoopOp`, to move immediately to the next iteration of the containing loop
    * :class:`BoxOp`, a simple grouping of instructions
    * :class:`ForLoopOp`, to loop over a fixed range of values
    * :class:`IfElseOp`, to conditionally enter one of two subcircuits
    * :class:`SwitchCaseOp`, to conditionally enter one of many subcircuits
    * :class:`WhileLoopOp`, to repeat a subcircuit until a condition is falsified.

Certain instructions can be "annotated" with metadata, which is typically intended to be consumed by
a compiler pass either locally, or in later backend processing.  Currently this is limited to
:class:`BoxOp`.  These annotations are represented by custom subclasses of :class:`Annotation`, and
there is further discussion of the support infrastructure in :mod:`qiskit.circuit.annotation`.

:ref:`Circuits can include classical expressions that are evaluated in real time
<circuit-repr-real-time-classical>`, while the QPU is executing a single shot of the circuit.  These
are primarily documented in the module documentation of :mod:`qiskit.circuit.classical`.  You might
be particularly interested in the base classes (which are not exposed from the :mod:`qiskit.circuit`
root):

* :class:`~classical.expr.Var`, a typed classical storage location in a circuit
* :class:`~classical.expr.Expr`, a real-time-evaluated expression
* :class:`~classical.types.Type`, the classical type of an expression.

In addition to this real-time expression evaluation, which is limited by classical hardware
representations of data, Qiskit has the concept of "compile-time" parametrization, which is done in
abstract symbolic algebra.  These are typically used to represent gate angles in high-level
algorithms that might want to perform numerical derivatives, but they are an older part of Qiskit
than the real-time evaluation, so are still used in some places to do general parametrization.  The
main related classes are:

* :class:`Parameter`, the atom of compile-time expressions
* :class:`ParameterExpression`, a symbolic calculation on parameters
* :class:`ParameterVector`, a convenience collection of many :class:`Parameter`\ s
* :class:`ParameterVectorElement`, a subclass of :class:`Parameter` used by :class:`ParameterVector`

The :mod:`qiskit.circuit` module also exposes some calculation classes that work with circuits to
assist compilation workflows.  These include:

* :class:`EquivalenceLibrary`, a database of decomposition relations between gates and circuits
* :data:`SessionEquivalenceLibrary`, a mutable instance of :class:`EquivalenceLibrary` which is used
  by default by the compiler's :class:`.BasisTranslator`.

There are also utilities for generating random circuits:

* :func:`random.random_circuit`
* :func:`random.random_circuit_from_graph`
* :func:`random.random_clifford_circuit`

Finally, the circuit module has its own exception class, to indicate when things went wrong in
circuit-specific manners:

* :exc:`CircuitError`


.. _circuit-repr:

Representation of circuits in Qiskit
====================================

The main user-facing class for representing circuits is :class:`QuantumCircuit`.  This can be either
an abstract circuit or a physical circuit.  There is much more information about the
:class:`QuantumCircuit` class itself and the multitude of available methods on it in its class
documentation.

Internally, a :class:`QuantumCircuit` contains the qubits, classical bits, compile-time parameters,
real-time variables, and other tracking information about the data it acts on and how it is
parametrized.  It then contains a sequence of :class:`CircuitInstruction`\ s, which contain
the particular operation (gate, measurement, etc) and its operands (the qubits and classical bits).


Bits and registers
------------------

Qubits and classical bits are represented by a shared base :class:`Bit` type, which is just intended
to be a "type tag"; the classes have no behavior other than being immutable objects:

.. autoclass:: Bit
.. autoclass:: Qubit
    :show-inheritance:
    :class-doc-from: class
.. autoclass:: Clbit
    :show-inheritance:
    :class-doc-from: class

Qubits and clbits are instantiated by users with no arguments, such as by ``Qubit()``.  Bits compare
equal if they are the same Python object, or if they were both created by a register of the same
name and size, and they refer to the same index within that register.  There is also a special type
tag for "ancilla" qubits, but this is little used in the current state
of Qiskit:

.. autoclass:: AncillaQubit
    :show-inheritance:
    :class-doc-from: class

A collection bits of the same type can be encapsulated in a register of the matching type.  The base
functionality is in a base class that is not directly instantiated:

.. autoclass:: Register
    :members:

Each of the defined bit subtypes has an associated register, which have the same constructor
signatures, methods and properties as the base class:

.. autoclass:: QuantumRegister
    :show-inheritance:
    :class-doc-from: class
.. autoclass:: ClassicalRegister
    :show-inheritance:
    :class-doc-from: class
.. autoclass:: AncillaRegister
    :show-inheritance:
    :class-doc-from: class

A common way to instantiate several bits at once is to create a register, such as by
``QuantumRegister("my_qreg", 5)``.  This has the advantage that you can give that collection of bits
a name, which will appear during circuit visualizations (:meth:`QuantumCircuit.draw`) and exports to
interchange languages (see :mod:`.qasm2` and :mod:`.qasm3`).  You can also pass a name and a list of
pre-constructed bits, but this creates an "aliasing register", which are very poorly supported on
hardware.

Circuits track registers, but registers themselves impart almost no behavioral differences on
circuits.  The only exception is that :class:`ClassicalRegister`\ s can be implicitly cast to
unsigned integers for use in conditional comparisons of :ref:`control flow operations
<circuit-control-flow-repr>`.

Classical registers and bits were the original way of representing classical data in Qiskit, and
remain the most supported currently.  Longer term, the data model is moving towards a more complete
and strongly typed representation of a range of classical data (see
:ref:`circuit-repr-real-time-classical`), but you will still very commonly use classical bits in
current Qiskit.


Instruction contexts
--------------------

The scalar type of the :attr:`QuantumCircuit.data` sequence is the "instruction context" object,
:class:`CircuitInstruction`.  This is essentially just a data class that contains a representation
of what is to be done (its :attr:`~CircuitInstruction.operation`), and the data it acts on (the
:attr:`~CircuitInstruction.qubits` and :attr:`~CircuitInstruction.clbits`).

.. autosummary::
    :toctree: ../stubs/

    CircuitInstruction

Programmatically, this class is actually implemented in Rust and is a constructed handle to internal
data within Rust space.  Mutations to instances of this class will not be reflected in the circuit.
In general, you cannot mutate instruction contexts that are already in the circuit directly; the
:class:`QuantumCircuit` interface is designed for storing and building circuits, while the
:ref:`transpiler and its passes <qiskit-transpiler>`, and its intermediate :class:`.DAGCircuit`
representation, are where you should look for an interface to mutate circuits.

The :class:`QuantumCircuit` methods that add instructions to circuits (such as
:meth:`~QuantumCircuit.append`, and all the helper standard-gate methods) return an
:class:`InstructionSet`, which is a handle to several :class:`CircuitInstruction`\ s simultaneously.

.. autosummary::
    :toctree: ../stubs/

    InstructionSet

This :class:`InstructionSet` is now little used in Qiskit.  It provides a very minimal set of
methods to perform post-append mutations on instructions (which *will* be propagated to the
circuit), but these are now discouraged and you should use the alternatives noted in those methods.


.. _circuit-operations-instructions:

Operations, instructions and gates
----------------------------------

Within a :class:`CircuitInstruction`, the minimal interface that any operation must fulfill is
:class:`Operation`.  This is a *very* high level view, and only usable for abstract circuits.  The
main purpose of treating operations as :class:`Operation` is to allow arbitrary mathematical
objects (such as :class:`.quantum_info.Operator`) to be added to abstract circuits directly.

.. autosummary::
    :toctree: ../stubs/

    Operation

Most operations, including all operations on physical circuits, are instances of the more concretely
defined :class:`Instruction`.  This represents any instruction that some QPU might be able to carry
out natively, such as :class:`Measure`.  :class:`Instruction` need not be unitary (much as
:class:`Measure` isn't); an instruction is specifically unitary if it is a :class:`Gate`.

.. autosummary::
    :toctree: ../stubs/

    Instruction

:class:`Instruction`\ s can be near arbitrary, provided they only act on :class:`Qubit`\ s and
:class:`Clbit`\ s, and are parametrized by their :attr:`~Instruction.params`; they should not
attempt to "close over" outer circuit registers, or use hidden parameters inside themselves.
:class:`Instruction`\ s can be related to other circuits to provide a decompositions by using
their :attr:`Instruction.definition` attribute, which provides a local, one-off decomposition.  This
can be in whatever basis set of operations is most convenient to you, as long as the definitions of
all contained gates have some topological order; that is, you cannot use a gate in a definition if
its own definition depends on the parent.  If the :class:`Instruction` should be considered entirely
opaque to optimizers, its :attr:`~Instruction.definition` can be ``None``.  See
:ref:`circuit-custom-gates` for more detail.

The :attr:`~Instruction.params` of an instruction can technically be arbitrary, but in general you
should attempt to stick to parametrizations in terms of real numbers, wherever possible.  Qiskit
itself breaks this rule in many places, and you will find all sorts of unusual types in
:attr:`Instruction.params` fields, but these are an annoying source of bugs because they often imply
the need for type-aware special casing.  If your instruction is parametrized in terms of angles, you
will be able to reliably use :ref:`compile-time parametrization in it
<circuit-compile-time-parameters>`, and it will integrate well with
:meth:`QuantumCircuit.assign_parameters`.

While :class:`Instruction` is not necessarily unitary, its subclass :class:`Gate` implies unitarity,
and adds :meth:`~Gate.to_matrix` and :meth:`~Gate.control` methods to all the methods inherited from
:class:`Instruction`.

.. autosummary::
    :toctree: ../stubs/

    Gate

:class:`Gate` inherits all the methods for :class:`Instruction` and all the same considerations
about its :attr:`~Instruction.params` and :attr:`~Instruction.definition` field, except of course
that :class:`Gate`\ s cannot act on any classical resources.

:class:`Gate` instances can (and should) have a base :attr:`~Instruction.definition`, but you can
also specify several different decompositions in different bases by using an
:class:`EquivalenceLibrary`.

Subclassing :class:`Gate`, Qiskit has a special :class:`ControlledGate` class as well.  This class
is the base of many standard-library gates that are controlled (such as :class:`CXGate`), which is
where you are most likely to encounter it:

.. autosummary::
    :toctree: ../stubs/

    ControlledGate

Each of :class:`Instruction`, :class:`Gate` and :class:`ControlledGate` has a corresponding
singleton type, built using the machinery described in :mod:`qiskit.circuit.singleton`.  The
module-level documentation contains full details, along with descriptions of
:class:`.SingletonInstruction`, :class:`.SingletonGate` and :class:`.SingletonControlledGate`.  From
a user's perspective, little changes based on whether the base class is a singleton or not; the
intention always remains that you should call :meth:`~Instruction.to_mutable` first if you need to
get a safe-to-mutate owned copy of an instruction (you cannot assume that an arbitrary instruction
is mutable), and while direct :class:`type` inspection is discouraged, if you do need it, the
reliable way to find the "base" type of a potentially singleton instruction is to use
:attr:`~Instruction.base_class`.

:class:`ControlledGate` uses the same mechanisms as :ref:`subclassing gates <circuit-custom-gates>`
to define a fixed, lazy synthesis for itself.  This is naturally not hardware-aware, and harder to
hook into the synthesis routines of the compiler, but works better as a concrete
:class:`Instruction` that could potentially be run natively on hardware.  For cases where synthesis
and abstract optimization is more important, Qiskit offers a composable class called
:class:`AnnotatedOperation`, which tracks "gate modifiers" (of which :class:`ControlModifier` is
one) to apply to the inner :attr:`~AnnotatedOperation.base_op`.

.. autosummary::
    :toctree: ../stubs/

    AnnotatedOperation

The available modifiers for :class:`AnnotatedOperation` are:

.. autoclass:: InverseModifier
.. autoclass:: ControlModifier
.. autoclass:: PowerModifier

For information on how to create custom gates and instructions, including how to build one-off
objects, and re-usable parametric gates via subclassing, see :ref:`circuit-custom-gates` below.
The Qiskit circuit library in :mod:`qiskit.circuit.library` contains many predefined gates and
circuits for you to use.


Built-in special instructions
-----------------------------

Qiskit contains a few :class:`Instruction` classes that are in some ways "special".  These typically
have special handling in circuit code, in the transpiler, or the models of hardware.  These are all
generally instructions you might already be familiar with.

Measurements in Qiskit are of a single :class:`Qubit` into a single :class:`Clbit`.  These are the
two that the instruction is applied to.  Measurements are in the computational basis.

.. autoclass:: Measure(label=None)
    :show-inheritance:

Related to measurements, there is a :class:`Reset` operation, which produces no classical data but
instructs hardware to return the qubit to the :math:`\lvert0\rangle` state.  This is assumed to
happen incoherently and to collapse any entanglement.

.. autoclass:: Reset(label=None)
    :show-inheritance:

Hardware can be instructed to apply a real-time idle period on a given qubit.  A scheduled circuit
(see :mod:`qiskit.transpiler`) will include all the idle times on qubits explicitly in terms of this
:class:`Delay`.  :class:`.BoxOp` can also have an explicit duration attached, in its
:attr:`.BoxOp.duration` field.

.. autoclass:: Delay
    :show-inheritance:

Delay durations can be specified either with concrete, constant times, or with delayed-resolution
"duration expressions" built out of :class:`.expr.Stretch` objects.  See :ref:`circuit-stretches`
for more on this.

The :class:`Barrier` instruction can span an arbitrary number of qubits and clbits, and is a no-op
in hardware.  During transpilation and optimization, however, it blocks any optimizations from
"crossing" the barrier; that is, in:

.. plot::
    :include-source:
    :nofigs:

    from qiskit.circuit import QuantumCircuit

    qc = QuantumCircuit(1)
    qc.x(0)
    qc.barrier()
    qc.x(0)

it is forbidden for the optimizer to cancel out the two :math:`X` instructions.

.. autoclass:: Barrier
    :show-inheritance:

The :class:`Store` instruction is particularly special, in that it allows writing the result of a
:ref:`real-time classical computation expression <circuit-repr-real-time-classical>` (an
:class:`.expr.Expr`) in a local classical variable (a :class:`.expr.Var`).  It takes *neither*
:class:`Qubit` nor :class:`Clbit` operands, but has an explicit :attr:`~Store.lvalue` and
:attr:`~Store.rvalue`.

For example, to determine the parity of a bitstring ``cr`` and store it in another register ``creg``,
the :class:`Store` instruction can be used in the following way::

    parity = expr.lift(cr[0])
    for i in range(1,n):
        parity = expr.bit_xor(cr[i], parity)
    qc.store(creg[0], parity)



.. autoclass:: Store
    :show-inheritance:
    :members:
    :no-inherited-members:


.. _circuit-repr-real-time-classical:

Real-time classical computation
-------------------------------

.. seealso::
    :mod:`qiskit.circuit.classical`
        Module-level documentation for how the variable-, expression- and type-systems work, the
        objects used to represent them, and the classical operations available.

    :ref:`circuit-real-time-methods`
        The :class:`QuantumCircuit` methods for working with these variables in the context of a
        single circuit.

Qiskit has rudimentary low-level support for representing real-time classical computations, which
happen during the QPU execution and affect the results.  We are still relatively early into hardware
support for these concepts as well, so beware that you will need to work closely with your hardware
provider's documentation to get the best use out of any real-time classical computation.

These real-time calculations are represented by the expression and type system in
:mod:`qiskit.circuit.classical`.  At a high level, all real-time expressions are represented by an
:class:`.Expr` node, which is part of an expression "tree" representation, which has a well-defined
:class:`~.classical.Type` associated with it at every level.  See the module-level documentation for
much more detail on the internal representations of these classes.

The result of a real-time :class:`.Expr` can be used directly in certain places.  Currently this is
limited to conditions of :class:`.IfElseOp` and :class:`.WhileLoopOp`, and the target of
:class:`.SwitchCaseOp`.  The result can also be stored in a typed classical storage location, using
the :class:`.Store` instruction (or its :meth:`QuantumCircuit.store` constructor), backed by a
:class:`.expr.Var` node.

A circuit can contain manual classical storage locations, represented internally by the
:class:`~.expr.Var` node of the :class:`.Expr` tree.  These have an attached classical type (like
any other expression).  These can either be declared and initialized within each execution of the
circuit (:meth:`~QuantumCircuit.add_var`), or be inputs to the circuit
(:meth:`~QuantumCircuit.add_input`).

.. _circuit-compile-time-parameters:

Compile-time parametrization
----------------------------

Various parametric :class:`Instruction` instances in Qiskit can be parametrized in ways that are
designed to be resolved at compile time.  These are characterized by the use of the
:class:`Parameter` and :class:`ParameterExpression` classes.

.. autosummary::
    :toctree: ../stubs/

    Parameter
    ParameterExpression

The main way that this differs from the :class:`expr.Var` variables used in real-time classical
computation is that :class:`ParameterExpression` is a symbolic representation of a mathematical
expression.  The semantics of the expression are those of regular mathematics over the continuous
real numbers (and, in limited cases, over the complex numbers). In contrast, :class:`.Var` is a
handle to a variable stored on a classical computer, such as a floating-point value or an
fixed-width integer, which are always discrete.

In other words, you can expect :class:`ParameterExpression` to do symbolic simplifications that are
valid in mathematics, such as simplifying :math:`(x + y - x) / y \to 1`.  Such a simplification is
not valid in floating-point arithmetic, and :class:`.expr.Expr` will not do this.

The "compile-time" part of these parameters means that you typically will want to "assign" values to
the parameters before sending the circuit for execution.  These parameters can typically be used
anywhere that expects a mathematical angle (like a rotation gate's parameters), with the caveat that
hardware will usually require them to be assigned to a proper classically typed value before
execution.  You can do this assignment using :meth:`QuantumCircuit.assign_parameters`.

You may want to use many parameters that are related to each other.  To make this easier (and to
avoid you needing to come up with many names), you can use the convenience constructor
:class:`ParameterVector`.  The elements of the vector are all valid :class:`Parameter` instances, of
a special subclass :class:`ParameterVectorElement`.

.. autosummary::
    :toctree: ../stubs/

    ParameterVector
    ParameterVectorElement

.. _circuit-control-flow-repr:

Control flow in circuits
------------------------

Within :class:`QuantumCircuit`, classical control flow is represented by specific
:class:`Instruction`\ s, which are subclasses of :class:`ControlFlowOp`.

.. autosummary::
    :toctree: ../stubs/

    ControlFlowOp

For convenience, there is a :class:`frozenset` instance containing the :attr:`.Instruction.name`
attributes of each of the control-flow operations.

.. data:: CONTROL_FLOW_OP_NAMES

    Set of the instruction names of Qiskit's known control-flow operations.

The :func:`.get_control_flow_name_mapping` function allows to access the control-flow operation
classes associated to each name.

.. autofunction:: get_control_flow_name_mapping

These control-flow operations (:class:`IfElseOp`, :class:`WhileLoopOp`,
:class:`SwitchCaseOp`, :class:`ForLoopOp`, and :class:`BoxOp`) all have specific state that defines
the branching conditions and strategies, but contain all the different subcircuit blocks that might
be entered in their :attr:`~ControlFlowOp.blocks` property.

.. autosummary::
    :toctree: ../stubs/

    IfElseOp
    WhileLoopOp
    SwitchCaseOp
    ForLoopOp
    BoxOp

The :class:`.SwitchCaseOp` also understands a special value:

.. autodata:: CASE_DEFAULT

In addition to the block-structure control-flow operations, there are also two special instructions
that affect the flow of control when within loops.  These correspond to typical uses of the
``break`` and ``continue`` statements in classical programming languages.

.. autosummary::
    :toctree: ../stubs/

    BreakLoopOp
    ContinueLoopOp

.. note::
    The classes representations are documented here, but please note that manually constructing
    these classes is a low-level operation that we do not expect users to need to do frequently.

    Users should read :ref:`circuit-control-flow-methods` for the recommended workflows for building
    control-flow-enabled circuits.

Since :class:`ControlFlowOp` subclasses are also :class:`Instruction` subclasses, this means that
the way they are stored in :class:`CircuitInstruction` instances has them "applied" to a sequence of
qubits and clbits in its :attr:`~CircuitInstruction.qubits` and :attr:`~CircuitInstruction.clbits`
attributes.  This can lead to subtle data-coherence problems: the :class:`Qubit` and :class:`Clbit`
objects used inside the subcircuit blocks of the control-flow ops will not necessarily be identical
to the corresponding objects in the :class:`CircuitInstruction`.  Any code that consumes
control-flow operations in Qiskit needs to be aware of this; within a subcircuit, you should treat
``subcircuit.qubits[i]`` as if it were really ``outer_instruction.qubits[i]``, and so on.  You can
generate an easy lookup table for this by doing::

    cf_instruction: CircuitInstruction = ...
    cf_operation: ControlFlowOp = cf_instruction.operation
    for block in blocks:
        # Mappings of "inner" qubits/clbits to the outer ones.
        qubit_map = dict(zip(block.qubits, cf_instruction.qubits))
        clbit_map = dict(zip(block.clbits, cf_instruction.clbits))

        # ... do something with `block` ...

Remember that you will need to propagate this information if you recurse into subblocks of
control-flow operations.

..
    TODO: insert cross-ref to control-flow builder guide into below paragraph once written.

All the subcircuit blocks in a :class:`ControlFlowOp` are required to contain the same numbers of
:class:`Qubit`\ s and :class:`Clbit`\ s, referring to the same outer bits in the same order, such
that the :class:`zip` loop given in the code block above works.  The inner-circuit :class:`Bit`
objects do not need to be literally the same objects.  When using the control-flow builder interface
(which, it cannot be stressed enough, is *highly* recommended for users), the builders will arrange
that the inner bit objects *are* identical to the outer bit objects; the ``qubit_map`` in the code
block above will always be a mapping ``{x: x}``, but if you are consuming the blocks, you should be
prepared for the case that the mapping is required.

Any :class:`ClassicalRegister`\ s used in a control-flow subcircuit must also be present in all
containing blocks (*i.e.* any containing control-flow operations, and the outermost circuit), and
all blocks in the same :class:`ControlFlowOp` need to contain the same registers.  Again, the
builder interface will arrange for this to be the case (or produce an eager error if they cannot).

When the low-level construction is being used the inner :class:`QuantumCircuit` blocks must
manually close over any outer-scope :ref:`real-time classical computation variables
<circuit-repr-real-time-classical>` that they use.  This is marked by these being in the
:meth:`~QuantumCircuit.iter_captured_vars` iterator for that block.  Libraries constructing these
blocks manually will need to track these captures when building control-flow circuit blocks and add
them to the block using :meth:`~QuantumCircuit.add_capture` (or the ``captures`` constructor
argument), but user code will typically use the control-flow builder interface, which handles this
automatically.

Consult :ref:`the control-flow construction documentation <circuit-control-flow-methods>` for more
information on how to build circuits with control flow.


Instruction-local annotations
-----------------------------

.. seealso::

    :mod:`qiskit.circuit.annotation`
        The module-level discussion of the annotation framework, including how to defined custom
        annotations, and how the system interacts with the compiler and with serialization to other
        formats.


Certain circuit instructions can be "annotated" with instruction-local annotations.  As of Qiskit
2.1.0, this is limited to :class:`.BoxOp`.  All annotations are subclasses of one base
interface-defining object, but typically represent entirely custom analyses and commands.

.. autosummary::
    :toctree: ../stubs

    Annotation


Investigating commutation relations
-----------------------------------

If two operations in a circuit commute, we can swap the order in which they are applied.
This can allow for optimizations and simplifications, for example, if it allows to merge
or cancel gates:

.. code-block:: text

         ┌─────────┐     ┌─────────┐               ┌─────────┐
    q_0: ┤ Rz(0.5) ├──■──┤ Rz(1.2) ├──■──     q_0: ┤ Rz(1.7) ├
         └─────────┘┌─┴─┐└──┬───┬──┘┌─┴─┐  =       └──┬───┬──┘
    q_1: ───────────┤ X ├───┤ X ├───┤ X ├     q_1: ───┤ X ├───
                    └───┘   └───┘   └───┘             └───┘

Performing these optimizations are part of the transpiler, but the tools to investigate commutations
are available in the :class:`CommutationChecker`.

.. autosummary::
   :toctree: ../stubs/

   CommutationChecker


.. _circuit-stretches:

Delayed-resolution scheduling
-----------------------------

Typically, the output of Qiskit's compiler cannot be directly executed on a QPU.  First, it is
likely to pass through some vendor-specific pulse-level compiler, which converts the gates and
measurements into signals to the controlling electronics of the QPU.  While Qiskit's
:class:`.Target` can represent *some* of the timing constraints that these pulses will have, it is
generally not entirely complete.  This is especially true when dynamic circuits (feed-forward
operations) are involved; the delays induced by the classical components are often dependent on
low-level post-optimization details of backend compilers, and cannot be known by Qiskit.

In these situations, a user can still exert control over the relative scheduling of pulses, such as
for dynamical decoupling, by using "stretch" durations.  These are constructed by
:meth:`.QuantumCircuit.add_stretch`, and interact with the classical-expression system described in
:mod:`qiskit.circuit.classical`, although they are not real-time mutable.

For example, we can add stretches and boxes to set up a system where two separate dynamic-decoupling
sequences are applied to the same qubit, while a pair of other qubits undergoes a delay of unknown
duration.  The two sequences are constrained to have the same length, even though internally the
concrete DD pulses have differing lengths.

.. code-block:: python

    from qiskit import QuantumCircuit
    from qiskit.circuit.classical import expr

    qc = QuantumCircuit(3, 3)
    # This sets up three duration "degrees of freedom" that will
    # be resolved later by a backend compiler.
    a = qc.add_stretch("a")
    b = qc.add_stretch("b")
    c = qc.add_stretch("c")

    # This set of operations involves feed-forward operations that
    # Qiskit cannot know the length of.
    with qc.box():
        qc.h(1)
        qc.cx(1, 2)
        qc.measure([1, 2], [1, 2])
        with qc.if_test(expr.equal(qc.clbits[1], qc.clbits[2])):
            qc.h(1)

    # While that stuff is happening to qubits (1, 2), we want
    # qubit 0 to do two different DD sequences.  The two DD
    # sequences are fixed to be the same length as each other,
    # even though they're both internally stretchy.
    with qc.box(duration=a):
        # Textbook NMRish XX DD.
        qc.delay(b, 0)
        qc.x(0)
        qc.delay(expr.mul(2, b), 0)
        qc.x(0)
        qc.delay(b, 0)
    with qc.box(duration=a):
        # XY4-like DD.
        for _ in range(2):
            qc.delay(c, 0)
            qc.y(0)
            qc.delay(expr.mul(2, c), 0)
            qc.x(0)
            qc.delay(c, 0)

.. _circuit-custom-gates:

Creating custom instructions
============================

If you wish to create simple one-off instructions or gates that will be added to a circuit, and the
blocks are just being used for visualization or grouping purposes, the easiest way to create a
custom instruction or gate is simply to build its definition as a :class:`QuantumCircuit`, and then
use its :meth:`~QuantumCircuit.to_instruction` or :meth:`~QuantumCircuit.to_gate` method as
appropriate.  The results can be given directly to :meth:`QuantumCircuit.append` on the larger
circuit.  These methods will create base :class:`Instruction` or :class:`Gate` instances whose
:attr:`~Instruction.definition` attribute is the circuit as supplied, meaning it will automatically
be accessible to the transpiler, and to other Qiskit functions that attempt to decompose circuits.

Note that standalone instructions and gates should act only on qubits and clbits; instructions that
need to use complex control-flow will need to be inlined onto the :class:`QuantumCircuit` using
:meth:`~QuantumCircuit.compose`.


Creating instruction subclasses
-------------------------------

The base classes :class:`Instruction`, :class:`Gate` and :class:`ControlledGate` are all designed to
be safe to subclass, and have hook points for subclasses to implement.  If your custom gate is
parameterless and stateless, you may also want to derive from the corresponding singleton class in
:mod:`qiskit.circuit.singleton`, such as :class:`SingletonGate`.  You should consult the
documentation in :mod:`qiskit.circuit.singleton` for additional methods and hook points for the
singleton machinery.

Subclasses should typically define a default constructor that calls the :class`super` constructor
with the correct arguments for your instruction.  It is permissible to have extra state in the
class, but your subclasses will most reliably integrate with the rest of the Qiskit machinery if you
depend only on your :attr:`Instruction.params`, and these parameters are purely gate angles.

Subclasses of :class:`Instruction` (or one of its subclasses) should implement the private
:meth:`Instruction._define` method, which lazily populates the hidden ``_definition`` cache that
backs the public :attr:`~Instruction.definition` method.

.. automethod:: Instruction._define

In subclasses of :class:`ControlledGate`, the :meth:`~Instruction._define` method should implement
the decomposition only for the all-ones control state.  The :attr:`ControlledGate.definition
<Instruction.definition>` machinery will modify this to handle the actual control state.

If the subclass is using the singleton machinery, beware that :meth:`~Instruction._define` will be
called eagerly immediately after the class-body statement has been executed, in order to produce the
definition object for the canonical singleton object. This means that your definition must only use
gates that are already defined; if you are writing a library with many singleton gates, you will
have to order your files and imports to ensure that this is possible.

Subclasses of :class:`Gate` will also likely wish to override `the Numpy array-protocol instance
method <https://numpy.org/devdocs/user/basics.interoperability.html#the-array-method>`__,
``__array__``.  This is used by :meth:`Gate.to_matrix`, and has the signature:

.. currentmodule:: None
.. py:method:: object.__array__(dtype=None, copy=None)

    Return a Numpy array representing the gate. This can use the gate's
    :attr:`~qiskit.circuit.Instruction.params` field, and may assume that these are numeric
    values (assuming the subclass expects that) and not
    :ref:`compile-time parameters <circuit-compile-time-parameters>`.

    For greatest efficiency, the returned array should default to a dtype of :class:`complex`.
.. currentmodule:: qiskit.circuit

If your custom subclass has natural representations of its controlled or inverse forms, you may also
wish to override the :meth:`~Instruction.inverse` and :meth:`~Gate.control` methods.


As an example of defining a custom :math:`R_{xz}` gate; that is, a single-angle rotation about the
:math:`XZ` axis.  This is essentially :class:`RZXGate`, if the qubits were the other way around, so
we will write our definition in terms of that.  We are parametric, so cannot be a singleton, but we
are unitary, so should be a :class:`Gate`::

    import math
    import numpy as np
    from qiskit.circuit import Gate, QuantumCircuit

    class RXZGate(Gate):
        def __init__(self, theta):
            # Initialize with our name, number of qubits and parameters.
            super().__init__("rxz", 2, [theta])

        def _define(self):
            # Our base definition is an RZXGate, applied "backwards".
            defn = QuantumCircuit(2)
            defn.rzx(1, 0)
            self._definition = defn

        def inverse(self, annotated = False):
            # We have an efficient representation of our inverse,
            # so we'll override this method.
            return RXZGate(-self.params[0])

        def power(self, exponent: float):
            # Also we have an efficient representation of power.
            return RXZGate(exponent * self.params[0])

        def __array__(self, dtype=None, copy=None):
            if copy is False:
                raise ValueError("unable to avoid copy while creating an array as requested")
            cos = math.cos(0.5 * self.params[0])
            isin = 1j * math.sin(0.5 * self.params[0])
            return np.array([
                [cos, -isin, 0, 0],
                [-isin, cos, 0, 0],
                [0, 0, cos, isin],
                [0, 0, isin, cos],
            ], dtype=dtype)


In this example, we defined a base definition in terms of :class:`RZXGate`, but to enable faster
decompositions to a range of bases, we might want to add some more equivalences to
:data:`SessionEquivalenceLibrary`.  Note that the :class:`.BasisTranslator` translation search will
search through all possible equivalences at all possible depths, so providing an equivalence in
terms of (say) :class:`.XGate` will automatically make decompositions in terms of :class:`.RXGate`
available as well.

Let us add an equivalence in terms of :math:`H`, :math:`CX` and :math:`R_z` for an arbitrary symbolic
parameter::

    from qiskit.circuit import SessionEquivalenceLibrary, Parameter

    theta = Parameter("theta")

    equiv = QuantumCircuit(2)
    equiv.h(0)
    equiv.cx(1, 0)
    equiv.rz(theta, 0)
    equiv.cx(1, 0)
    equiv.h(0)

    SessionEquivalenceLibrary.add_equivalence(RZXGate(theta), equiv)

After this, for the duration of the Python interpreter session, translators like
:class:`.BasisTranslator` will find our new definition in their search.


.. _circuit-working-with:

Working with circuit-level objects
==================================

.. _circuit-abstract-to-physical:

Converting abstract circuits to physical circuits
-------------------------------------------------

..
    Note that this is just a "jumping-off" section - this should just provide an overview of links
    to where the real information is.

An abstract :class:`QuantumCircuit` cannot reliably be run on hardware.  You might be able to use
some of the high-level simulators linked to in :ref:`circuit-simulation` to produce quick results
for small scale circuits, but to run utility-scale circuits, you will need to use real hardware,
which involves compiling to a physical circuit.

The high-level function to do this is :func:`.transpile`; it takes in an abstract circuit and a
hardware ``backend`` or ``target``, and returns a physical circuit.  To get more access and control
over the stages of the passes that will be run, use :func:`.generate_preset_pass_manager` to build a
:class:`~.transpiler.StagedPassManager` first, which you can then modify.

The full transpilation and compilation machinery is described in detail in the
:mod:`qiskit.transpiler` module documentation, and detail on all the passes built into Qiskit is
available in :mod:`qiskit.transpiler.passes`.


.. _circuit-simulation:

Simulating circuits
-------------------

..
    Note that this is just a "jumping-off" section - this should just provide an overview of links
    to where the real information is.


While not part of the :mod:`qiskit.circuit` interface, one of the most common needs is to get quick
simulation results for :class:`QuantumCircuit` objects.  This section provides a quick jumping-off
point to other places in the documentation to find the relevant information.

For unitary circuits, you can simulate the effects on the :math:`\lvert0\dotsm0\rangle` state by
passing the :class:`QuantumCircuit` directly to the :class:`~.quantum_info.Statevector` default
constructor.  You can similar get a unitary matrix representing the circuit as an operator by
passing it to the :class:`~.quantum_info.Operator` default constructor.  If you have a physical
circuit, you may want to instead pass it to :meth:`.Operator.from_circuit` method to apply
transformations from the :attr:`QuantumCircuit.layout` to map it back to the "abstract" qubit space.

For a more backend-like simulation experience, there are simulator-backed implementations of all the
Qiskit hardware interfaces.  In particular, you might be interested in:

* :class:`.BasicProvider` and the raw backends it can return to you.
* :class:`.StatevectorSimulator` for a backend-like wrapper around :class:`.Statevector`
* The :mod:`qiskit_aer` for full, high-performance simulation capabilities.
* :class:`.StatevectorSampler` and :class:`.StatevectorEstimator` for simulator-backed reference
  implementations of the :ref:`Qiskit Primitives <qiskit-primitives>`.


Defining equivalence relationships
----------------------------------

A common task in mapping abstract circuits to physical hardware and optimizing the result is to find
equivalence relations that map a gate to a different basis set.  Qiskit stores this information in a
database class called :class:`EquivalenceLibrary`.

.. autosummary::
    :toctree: ../stubs/

    EquivalenceLibrary

Qiskit ships with a large set of predefined equivalence relationships for all of its standard gates.
This base library is called :data:`StandardEquivalenceLibrary`, and should be treated as immutable.

.. py:data:: StandardEquivalenceLibrary

    A :class:`EquivalenceLibrary` that stores of all Qiskit's built-in standard gate relationships.
    You should not mutate this, but instead either create your own :class:`EquivalenceLibrary` using
    this one as its ``base``, or modify the global-state :data:`SessionEquivalenceLibrary`.

Qiskit also defines a shared global-state object, :data:`SessionEquivalenceLibrary`, which is the
default equivalences used by various places in Qiskit, most notably the :class:`.BasisTranslator`
transpiler pass.  You should feel free to add your own equivalences to this using its
:meth:`~EquivalenceLibrary.add_equivalence` method, and they will be automatically picked up by
default instances of the :class:`.BasisTranslator`.

.. py:data:: SessionEquivalenceLibrary

    The default instance of :class:`EquivalenceLibrary`, which will be used by most Qiskit objects
    if no library is manually specified.  You can feel free to add equivalences to this using
    :meth:`~EquivalenceLibrary.add_equivalence`.  It inherits all the built-in rules of
    :data:`StandardEquivalenceLibrary`.


Apply Pauli twirling to a circuit
---------------------------------

There are two primary types of noise when executing quantum circuits. The first is stochastic,
or incoherent, noise that is mainly due to the unwanted interaction between the quantum processor
and the external environment in which it resides. The second is known as coherent error, and these
errors arise due to imperfect control of a quantum system. This can be unwanted terms in a system
Hamiltonian, i.e. incorrect unitary evolution, or errors from incorrect temporal control of the
quantum system, which includes things like incorrect pulse-shapes for gates.

Pauli twirling is a quantum error suppression technique that uses randomization to shape coherent
error into stochastic errors by combining the results from many random, but logically equivalent
circuits, together. Qiskit provides a function to apply Pauli twirling to a given circuit for
standard two qubit gates. For more details you can refer to the documentation of the function
below:

.. autofunction:: qiskit.circuit.pauli_twirl_2q_gates


Exceptions
==========

Almost all circuit functions and methods will raise a :exc:`CircuitError` when encountering an error
that is particular to usage of Qiskit (as opposed to regular typing or indexing problems, which will
typically raise the corresponding standard Python error).

.. autoexception:: CircuitError


.. _circuit-conventions:

Circuit conventions
===================

When constructing circuits out of abstract objects and more concrete matrices, there are several
possible conventions around bit-labelling, bit-ordering, and how the abstract tensor product is
realized in concrete matrix algebra.

Qiskit's conventions are:

* in bitstring representations, bits are labelled with the right-most bit in the string called
  :math:`0` and the left-most bit in the string of :math:`n` bits called :math:`n - 1`.

* when using integers as bit-specifier indices in circuit-construction functions, the integer is
  treated as an index into :attr:`QuantumCircuit.qubits` (or :attr:`~QuantumCircuit.clbits`).

* when drawing circuits, we put the lowest-index bits on top.

* in statevector representations, we realize the abstract tensor product as the Kronecker product,
  and order the arguments to this such that the amplitude of computational-basis state
  :math:`\lvert x\rangle`, where :math:`x` is the bitstring interpreted as an integer, is at
  location ``statevector[x]``.

* when controlling a gate, the control qubit(s) is placed first in the argument list, *e.g.* in the
  call ``qc.cx(0, 1)``, qubit 0 will be the control and qubit 1 will be the target.  Similarly, in
  the manual call ``qc.append(CXGate(), [0, 1])``, qubit 0 will be the control and qubit 1 will be
  the target.

Let us illustrate these conventions with some examples.

Bit labelling
-------------

Take the circuit:

.. plot::
    :include-source:
    :nofigs:
    :context:
    :show-source-link: False

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5, 5)
    qc.x(0)
    qc.x(1)
    qc.x(4)
    qc.measure(range(5), range(5))

This flips the states of qubits 0, 1 and 4 from :math:`\lvert0\rangle` to :math:`\lvert1\rangle`,
then measures all qubits :math:`n` into the corresponding clbit :math:`n` using the computational
(:math:`Z`) basis.  If simulated noiselessly, the bitstring output from this circuit will be
:math:`10011` every time; qubits 0, 1, and 4 are flipped, and the "one" values in the bitstring are
in the zeroth, first and fourth digits *from the right*.

In Qiskit, we would write the qubit state immediately before the measurement in ket-notation
shorthand as :math:`\lvert10011\rangle`.  Note that the ket label matches the classical bitstring,
and has the numeric binary value of 19.

If we draw this circuit, we will see that Qiskit places the zeroth qubit on the top of the circuit
drawing:

.. plot::
    :alt: Circuit diagram output by the previous code.
    :include-source:
    :context:
    :show-source-link: False

    qc.draw("mpl")


Matrix representations
----------------------

Statevectors are defined in the convention that for a two-level system, the relationship between
abstract representation and matrix representation is such that

.. math::

 \alpha\lvert0\rangle + \beta\lvert1\rangle
    \leftrightarrow \begin{pmatrix} \alpha \\ \beta \end{pmatrix}

where :math:`\alpha` and :math:`\beta` are complex numbers.  We store the statevector as a 1D Numpy
:class:`~numpy.ndarray` with data ``sv = [alpha, beta]``, *i.e.* ``sv[0] == alpha`` and ``sv[1] ==
beta``; note that the indices into the statevector match the ket labels.

We construct `the tensor product of two qubit states
<https://en.wikipedia.org/wiki/Tensor_product>`_ in matrix algebra using `the Kronecker product
<https://en.wikipedia.org/wiki/Kronecker_product>`_, with qubit 0 on the right and qubit 1 on the
left, such that the :math:`Z` basis state :math:`\lvert x\rangle` (where :math:`x` is the integer
interpretation of the bitstring) has its non-zero term in the statevector ``sv`` at ``sv[x]``::

    import numpy
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    state_0 = [1, 0]  # defined representation of |0>
    state_1 = [0, 1]  # defined representation of |1>

    # Circuit that creates basis state |10011>, where
    # binary 10011 has the decimal value 19.
    qc = QuantumCircuit(5)
    qc.x(0)
    qc.x(1)
    qc.x(4)
    qiskit_sv = Statevector(qc)

    # List index 'n' corresponds to qubit 'n'.
    individual_states = [
        state_1,
        state_1,
        state_0,
        state_0,
        state_1,
    ]
    # Start from a scalar.
    manual_sv = [1]
    for qubit_state in individual_states:
        # Each new qubit goes "on the left".
        manual_sv = numpy.kron(qubit_state, manual_sv)

    # Now `qiskit_sv` and `manual_sv` are the same, and:
    assert manual_sv[19] == 1
    assert qiskit_sv[19] == 1

This feeds through to the matrix representation of operators, and joins with the conventions on bit
orders for controlled operators.  For example, the matrix form of :class:`.CXGate` is::

    import numpy
    from qiskit.circuit.library import CXGate

    numpy.array(CXGate())

.. math::

    \operatorname{array}(CX) =
        \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0 \\
            0 & 1 & 0 & 0
        \end{pmatrix}

This might be different to other matrix representations you have seen for :math:`CX`, but recall
that the choice of matrix representation is conventional, and this form matches Qiskit's conventions
of *control qubits come first* and *the tensor product is represented such that there is a
correspondence between the index of the "one amplitude" and the bitstring value of a state*.

In the case of multiple controls for a gate, such as for :class:`.CCXGate`, the ``ctrl_state``
argument is interpreted as the bitstring value of the control qubits, using the same zero-based
labelling conventions.  For example, given that the default ``ctrl_state`` is the all-ones
bitstring, we can see that the matrix form of :class:`.CCXGate` with ``ctrl_state = 1`` is the same
as if we took the all-ones control-state :class:`.CCXGate`, but flipped the value of the higher
indexed control qubit on entry and exist to the gate::

    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator

    # Build the natural representation of `CCX` with the
    # control qubits being `[0, 1]`, relative to the
    # bitstring state "01", such that qubit 0 must be in |1>
    # and qubit 1 must be in |0>.  The target qubit is 2.
    ccx_natural = QuantumCircuit(3)
    ccx_natural.ccx(0, 1, 2, ctrl_state=1)

    # Build the same circuit in terms of the all-ones CCX.
    # Note that we flip _qubit 1_, because that's the one
    # that differs from the all-ones state.
    ccx_relative = QuantumCircuit(3)
    ccx_relative.x(1)
    ccx_relative.ccx(0, 1, 2)
    ccx_relative.x(1)

    assert Operator(ccx_relative) == Operator(ccx_natural)

In both these cases, the matrix form of :class:`.CCXGate` in ``ctrl_state = 1`` is:

.. math::

    \operatorname{array}\bigl(CCX(\text{ctrl\_state}=1)\bigr) =
        \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
        \end{pmatrix}
"""

from qiskit._accelerate.circuit import (
    Bit,
    Qubit,
    AncillaQubit,
    Clbit,
    QuantumRegister,
    AncillaRegister,
    ClassicalRegister,
    Register,
    Duration,
)

from .exceptions import CircuitError
from .quantumcircuit import QuantumCircuit
from .gate import Gate


from .annotation import Annotation
from .controlledgate import ControlledGate
from .instruction import Instruction
from .instructionset import InstructionSet
from .operation import Operation
from .barrier import Barrier
from .delay import Delay
from .measure import Measure
from .reset import Reset
from .store import Store
from .parameter import Parameter
from .parametervector import ParameterVector, ParameterVectorElement
from .parameterexpression import ParameterExpression
from .quantumcircuitdata import CircuitInstruction
from .equivalence import EquivalenceLibrary
from . import library
from .equivalence_library import StandardEquivalenceLibrary, SessionEquivalenceLibrary
from .commutation_checker import CommutationChecker

from .controlflow import *

from .annotated_operation import AnnotatedOperation, InverseModifier, ControlModifier, PowerModifier
from .twirling import pauli_twirl_2q_gates

from . import annotation, singleton, classical, random, controlflow

__all__ = [
    "AncillaQubit",
    "AncillaRegister",
    "AnnotatedOperation",
    "Annotation",
    "Barrier",
    "Bit",
    "CircuitError",
    "CircuitInstruction",
    "ClassicalRegister",
    "Clbit",
    "CommutationChecker",
    "ControlModifier",
    "ControlledGate",
    "Delay",
    "Duration",
    "EquivalenceLibrary",
    "Gate",
    "Instruction",
    "InstructionSet",
    "InverseModifier",
    "Measure",
    "Operation",
    "Parameter",
    "ParameterExpression",
    "ParameterVector",
    "ParameterVectorElement",
    "PowerModifier",
    "QuantumCircuit",
    "QuantumRegister",
    "Qubit",
    "Register",
    "Reset",
    "SessionEquivalenceLibrary",
    "StandardEquivalenceLibrary",
    "Store",
    "annotation",
    "classical",
    "library",
    "pauli_twirl_2q_gates",
    "random",
    "singleton",
]
__all__ += controlflow.__all__
