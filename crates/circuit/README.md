# `qiskit-circuit`

The Rust-based data structures for circuits.
This currently defines the core data collections for `QuantumCircuit`, but may expand in the future to back `DAGCircuit` as well.

This crate is a very low part of the Rust stack, if not the very lowest.

The data model exposed by this crate is as follows.

## CircuitData

The core representation of a quantum circuit in Rust is the `CircuitData` struct. This contains the list
of instructions that are comprising the circuit. Each element in this list is modeled by a
`CircuitInstruction` struct. The `CircuitInstruction` contains the operation object and its operands.
This includes the parameters and bits. It also contains the potential mutable state of the Operation representation from the legacy Python data model; namely `duration`, `unit`, `condition`, and `label`.
In the future we'll be able to remove all of that except for label.

At rest a `CircuitInstruction` is compacted into a `PackedInstruction` which caches reused qargs
in the instructions to reduce the memory overhead of `CircuitData`. The `PackedInstruction` objects
get unpacked back to `CircuitInstruction` when accessed for a more convenient working form.

Additionally the `CircuitData` contains a `param_table` field which is used to track parameterized
instructions that are using python defined `ParameterExpression` objects for any parameters and also
a global phase field which is used to track the global phase of the circuit.

## Operation Model

In the circuit crate all the operations used in a `CircuitInstruction` are part of the `OperationType`
enum. The `OperationType` enum has four variants which are used to define the different types of
operation objects that can be on a circuit:

 - `StandardGate`: a rust native representation of a member of the Qiskit standard gate library. This is
    an `enum` that enumerates all the gates in the library and statically defines all the gate properties
    except for gates that take parameters,
 - `PyGate`: A struct that wraps a gate outside the standard library defined in Python. This struct wraps
    a `Gate` instance (or subclass) as a `PyObject`. The static properties of this object (such as name,
    number of qubits, etc) are stored in Rust for performance but the dynamic properties such as
    the matrix or definition are accessed by calling back into Python to get them from the stored
    `PyObject`
 - `PyInstruction`: A struct that wraps an instruction defined in Python. This struct wraps an
    `Instruction` instance (or subclass) as a `PyObject`. The static properties of this object (such as
    name, number of qubits, etc) are stored in Rust for performance but the dynamic properties such as
    the definition are accessed by calling back into Python to get them from the stored `PyObject`. As
    the primary difference between `Gate` and `Instruction` in the python data model are that `Gate` is a
    specialized `Instruction` subclass that represents unitary operations the primary difference between
    this and `PyGate` are that `PyInstruction` will always return `None` when it's matrix is accessed.
 - `PyOperation`: A struct that wraps an operation defined in Python. This struct wraps an `Operation`
    instance (or subclass) as a `PyObject`. The static properties of this object (such as name, number
    of qubits, etc) are stored in Rust for performance. As `Operation` is the base abstract interface
    definition of what can be put on a circuit this is mostly just a container for custom Python objects.
    Anything that's operating on a bare operation will likely need to access it via the `PyObject`
    manually because the interface doesn't define many standard properties outside of what's cached in
    the struct.

There is also an `Operation` trait defined which defines the common access pattern interface to these
4 types along with the `OperationType` parent. This trait defines methods to access the standard data
model attributes of operations in Qiskit. This includes things like the name, number of qubits, the matrix, the definition, etc.

## ParamTable

The `ParamTable` struct is used to track which circuit instructions are using `ParameterExpression`
objects for any of their parameters. The Python space `ParameterExpression` is comprised of a symengine
symbolic expression that defines operations using `Parameter` objects. Each `Parameter` is modeled by
a uuid and a name to uniquely identify it. The parameter table maps the `Parameter` objects to the
`CircuitInstruction` in the `CircuitData` that are using them. The `ParamTable` comprises 3 `HashMaps` internally that map the uuid (as `u128`, which is accessible in Python by using `uuid.int`) to the `ParamEntry`, the `name` to the uuid, and the uuid to the PyObject for the actual `Parameter`.

The `ParamEntry` is just a `HashSet` of 2-tuples with usize elements. The two usizes represent the instruction index in the `CircuitData` and the index of the `CircuitInstruction.params` field of
a give instruction where the given `Parameter` is used in the circuit. If the instruction index is
`GLOBAL_PHASE_MAX`, that points to the global phase property of the circuit instead of a `CircuitInstruction`.
