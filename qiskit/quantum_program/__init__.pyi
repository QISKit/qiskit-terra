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

# The classes here are declared by hand because they come from the compiled extension module,
# which has no signatures of its own. Each member's summary line is repeated so that a hover
# keeps showing one; the full documentation lives on the Rust type. Nothing may be declared here
# that the class does not have, which test_quantum_program.py checks. Everything written in Python
# is imported from the module that defines it, so its types are read from there.

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

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

__all__ = [
    "DType",
    "DataTree",
    "ProgramStepper",
    "QuantumProgram",
    "Request",
    "TensorType",
    "Tracer",
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

class DataTree:
    """A leaf holding one value, or a branch of ordered children."""

    # Equality reads the leaves, which need not be hashable. The ignore is what typeshed uses to
    # narrow an inherited method to None.
    __hash__: ClassVar[None]  # type: ignore[assignment]

    def __init__(self, object: Any, /) -> None: ...
    @staticmethod
    def leaf_of(value: Any, /) -> DataTree:
        """Return a leaf holding `value`, which is not parsed."""

    @property
    def is_leaf(self) -> bool:
        """Whether this is a leaf, as opposed to a branch of children."""

    @property
    def leaf(self) -> Any:
        """The value this leaf holds. Raises `TypeError` on a branch."""

    @property
    def is_mapping(self) -> bool:
        """Whether this branch names its children. A branch that names them has a mapping form."""

    def keys(self) -> list[str]:
        """Return the name of every child, in order."""

    def values(self) -> list[Any]:
        """Return every child, in order."""

    def items(self) -> list[tuple[str, Any]]:
        """Return every child with its name, in order."""

    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...
    def __getitem__(self, key: str | int) -> Any: ...
    def __contains__(self, key: object) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...

class DType:
    """The element type of a tensor."""

    Bit: ClassVar[DType]
    U8: ClassVar[DType]
    U16: ClassVar[DType]
    U32: ClassVar[DType]
    U64: ClassVar[DType]
    I8: ClassVar[DType]
    I16: ClassVar[DType]
    I32: ClassVar[DType]
    I64: ClassVar[DType]
    F32: ClassVar[DType]
    F64: ClassVar[DType]
    C64: ClassVar[DType]
    C128: ClassVar[DType]

    def __getitem__(self, shape: int | bounded | tuple[int | bounded, ...]) -> TensorType:
        """Return a ``TensorType`` of this dtype whose shape is ``shape``."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...

class bounded:
    """An axis whose size is not known until run time, but is provably at most ``max``."""

    def __init__(self, max: int, /) -> None: ...
    @property
    def max(self) -> int:
        """The largest size this axis can have."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class TensorType:
    """A dtype paired with a shape, describing a tensor without holding one."""

    def __init__(self, dtype: DType, shape: Sequence[int | bounded], /) -> None: ...
    @property
    def dtype(self) -> DType:
        """The element type."""

    @property
    def shape(self) -> tuple[int | bounded, ...]:
        """The size of each axis, an integer or a `bounded()`."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class QuantumProgram:
    """A hybrid quantum-classical computation, described rather than performed."""

    def input_types(self) -> DataTree:
        """Return the declared type of every input, arranged as the program's input structure."""

    def output_types(self) -> DataTree:
        """Return the type of every output, arranged as the program's output structure."""

    def __call__(self, **inputs: Any) -> DataTree:
        """Evaluate the program on one keyword argument per declared input."""

    def partition(self, resources: Sequence[Sequence[str]], /) -> tuple[QuantumProgram, list[int]]:
        """Rewrite this program to put each execution resource's work in a function of its own."""

    def listing(self) -> str:
        """Return this program as a listing."""

    def draw(self) -> Image:
        """Draw this program's dataflow as a graph, one box per instruction."""

    @property
    def num_functions(self) -> int:
        """How many functions the program holds."""

    @property
    def entry(self) -> ProgramFunction:
        """The entry point, which is the last function the program defines."""

    def function(self, index: int, /) -> ProgramFunction:
        """Return the function at ``index`` in definition order."""

    def __iter__(self) -> Iterator[ProgramFunction]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ProgramFunction:
    """One function of a program: what it consumes, what it produces, and the instructions between."""

    @property
    def index(self) -> int:
        """Where this function sits in the program's definition order."""

    @property
    def is_entry(self) -> bool:
        """Whether this is the program's entry point."""

    def input_types(self) -> list[TensorType]:
        """Return the type of each parameter, in the order the function takes them."""

    def output_types(self) -> list[TensorType]:
        """Return the type of each result, in the order the function declares them."""

    @property
    def parameters(self) -> list[Instruction]:
        """The instructions declaring this function's parameters, in order."""

    @property
    def results(self) -> list[Instruction]:
        """The instructions declaring this function's results, in order."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Instruction: ...
    def __iter__(self) -> Iterator[Instruction]: ...
    def __repr__(self) -> str: ...

class Instruction:
    """One instruction of a program function: what it does, what it reads, and what it produces."""

    @property
    def id(self) -> int:
        """This instruction's id, which is its position in the function holding it."""

    @property
    def role(self) -> InstructionRole:
        """What part this instruction plays in its function."""

    @property
    def full_name(self) -> str:
        """The type name of what this instruction does, qualified by its namespace."""

    @property
    def describe(self) -> str | None:
        """A summary of the op's payload, such as `axis=0`, and `None` for anything else."""

    @property
    def operands(self) -> list[Value]:
        """The values this instruction consumes, in operand order."""

    @property
    def outputs(self) -> list[Value]:
        """The values this instruction produces, in the order it produces them."""

    def operand_types(self) -> list[TensorType]:
        """Return the type of each operand, read from the instructions producing them."""

    def output_types(self) -> list[TensorType]:
        """Return the type of each value this instruction produces."""

    @property
    def op(self) -> ops.ProgramOp | None:
        """The operation this instruction applies, and `None` for a parameter, a call or a result."""

    @property
    def callee(self) -> ProgramFunction | None:
        """The function this instruction calls, and `None` unless it is a call."""

    @property
    def callee_id(self) -> int | None:
        """The function this instruction calls, by its position in definition order."""

    def __repr__(self) -> str: ...

class InstructionRole:
    """What part an instruction plays in its function."""

    Parameter: ClassVar[InstructionRole]
    Op: ClassVar[InstructionRole]
    Call: ClassVar[InstructionRole]
    Result: ClassVar[InstructionRole]

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...

class Value:
    """One tensor value: an output slot of the instruction that produces it."""

    @property
    def instruction(self) -> int:
        """The instruction producing this value, by its position in the function holding it."""

    @property
    def slot(self) -> int:
        """Which of that instruction's results this value is."""

    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

class ProgramStepper:
    """Evaluate a :class:`~.QuantumProgram`, requesting an execution for each external function.

    Args:
        program: The program to step through.
        external: The program functions, by index, that require external evaluation.
        inputs: Arguments to the quantum program, anything coercible into an array-valued
            :class:`~.DataTree`.
    """

    def __init__(
        self,
        program: QuantumProgram,
        external: Sequence[int],
        inputs: Any | None = None,
        /,
    ) -> None: ...
    def step(self) -> None:
        """Run everything the answers in hand allow, stopping when only external work is left."""

    def outstanding(self) -> list[Request]:
        """Return every request the caller has still to answer."""

    def fulfil(self, request_id: int, outputs: Sequence[ArrayLike], /) -> None:
        """Answer the request ``request_id`` with one array per result the call declares."""

    @property
    def outputs(self) -> DataTree | None:
        """Return the program's outputs, or ``None`` if the evaluation is unfinished."""

    def __repr__(self) -> str: ...

class Request:
    """One piece of work a :class:`~.ProgramStepper` has handed over."""

    @property
    def id(self) -> int:
        """This request's identity within the stepper that raised it."""

    @property
    def function(self) -> ProgramFunction:
        """The function to perform."""

    @property
    def function_id(self) -> int:
        """The function to perform, by its position in the program's definition order."""

    @property
    def inputs(self) -> list[np.ndarray]:
        """One array per parameter of that function, in declaration order."""

    @property
    def output_types(self) -> list[TensorType]:
        """The type each array of the answer must satisfy, one per result of that function."""

    def __repr__(self) -> str: ...

bit: DType
u8: DType
u16: DType
u32: DType
u64: DType
i8: DType
i16: DType
i32: DType
i64: DType
f32: DType
f64: DType
c64: DType
c128: DType

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import ArrayLike
    from PIL.Image import Image
