# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Python interface to the Rust lowering pass manager objects.

Much of the actual driver logic here is defined in the Rust ``qiskit_pyext::passmanager`` module,
but that's the very low-level private interface.  This module provides the Python-friendly
higher-order wrapper logic around that low-level interface.
"""

from __future__ import annotations

import abc
import copy as _copy
from itertools import chain
import reprlib
import typing
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping, MutableSequence
from types import MappingProxyType
from typing import Generic, ClassVar, TypeAlias
from typing_extensions import TypeVar


from qiskit._accelerate import passmanager
from .exceptions import PassManagerError

__all__ = [
    "IR",
    "LoweringPassManager",
    "LoweringPassManagerError",
    "LoweringTask",
    "Pass",
    "PassContextHandle",
    "PassError",
    "StagesLike",
    "TaskSequence",
    "TaskStages",
]


class IR:
    """An interface for objects that can be used as IRs by a :class:`Pass` in a
    :class:`LoweringPassManager`.

    Note that all methods named ``_qiskit_*_`` are interface methods that are only for Qiskit to
    call.  These are documented for implementers, but are generally not stable for users to call;
    Qiskit may allow implementations to fulfill one of several contracts for the method, and
    additional alternatives may be added in new versions.  Qiskit will maintain stability within its
    own use of these protocols, but they are not stable for public consumption.

    Implementers of subclasses should override the attributes and methods, and are free to change
    the methods and signatures in line with this documentation about per-Qiskit-version calling
    signatures.

    Qiskit reserves the attribute and method namespace ``_qiskit_ir_*_`` for future expansion of
    this interface.  Qiskit will not define any attributes or methods in the interface outside this
    namespace.

    .. note::
        The native :class:`IR` system does not have a concept of Python's subclasses; an object is
        identified as being of at most one IR type, which is the first type in its resolution order
        that sets :attr:`_qiskit_ir_base_`, or the sole direct subclass of :class:`IR` if there is
        only one.
    """

    _qiskit_ir_name_: ClassVar[str] = ""
    """A human-readable name for the type of IR.

    Defaults to the name of base class implementing :class:`IR` if unset (or empty)."""

    _qiskit_ir_base_: ClassVar[type[IR] | None] = None
    """Mark a subclass of an existing :class:`IR` as a *new* intermediate representation.

    You only very rarely need to set this.

    There are three cases to think about:

    * You define a new class that subclasses :class:`IR` directly, and none of its supertypes are
      instances of :class:`IR` (the most common case).

      You do not need to set this attribute, but may explicitly set it to ``None`` (these have the
      same meaning).

    * You define a new class that subclasses an existing implementation of :class:`IR` for reasons
      unrelated to the :class:`IR` system.  You want instances of your subclass to be treated by the
      pass-manager system as if they're compatible with the base :class:`IR`.

      You must not set this attribute (nor override any other attributes or methods).

    * You define a new class that subclasses an existing implementation of :class:`IR`, and you want
      it to be considered as a *new* IR.

      You must set this attribute to ``None``.

    After class creation, the value of this class variable will be set to the resolved base
    :class:`IR` implementer.
    """

    def __init_subclass__(cls) -> None:
        cls._qiskit_ir_name_ = cls._qiskit_ir_name_ or cls.__name__
        cls._qiskit_ir_base_ = cls._qiskit_ir_base_ or cls


class _BuiltinIR(typing.Protocol):
    _qiskit_ir_builtin_: ClassVar[None]


AnyIR: TypeAlias = IR | _BuiltinIR
# Using `typing_extensions.TypeVar` because `default` is only Python 3.13+.
IRIn = TypeVar("IRIn", bound=AnyIR)
IROut = TypeVar("IROut", bound=AnyIR, default=IRIn)


@typing.final
class PassError(PassManagerError):
    """Raised by implementers of :meth:`Pass._qiskit_pass_run_` to provide user-facing errors."""


@typing.final
class LoweringPassManagerError(PassManagerError):
    """Raised by :meth:`LoweringPassManager.run` when compilation errors have occurred."""

    # This will gain some structured methods / whatever at some point in the future, but the initial
    # state we'll just use the regular display.


class LoweringTask(abc.ABC):
    """A single unit of execution flow in a :class:`LoweringPassManager`.

    A task is either a single :class:`~.Pass`, or a container of other tasks.
    Containers are freely mutable, and are permitted to be ill-typed while they are being
    assembled: a :class:`TaskSequence` may hold two consecutive tasks whose IR types disagree, for
    example. IR type checking happens when a task is given to a :class:`LoweringPassManager`.
    """


StagesLike: TypeAlias = Mapping[str, LoweringTask] | Iterable[tuple[str, LoweringTask]]
"""Types that :class:`TaskStages` accepts as a collection of stages."""


class TaskSequence(LoweringTask, MutableSequence):
    """A mutable sequence of tasks to run in order.

    Task sequences are permitted to be ill-typed while they are being assembled: they may hold two
    consecutive tasks whose IR types disagree.  IR type checking happens when this task sequence is
    given to a :class:`LoweringPassManager`.
    """

    def __init__(self, tasks: Iterable[LoweringTask] = ()):
        """
        Args:
            tasks: the initial tasks.
        """
        self.tasks: list[LoweringTask] = list(tasks)
        """The tasks, in execution order.
        
        This list can be safely mutated in-place.
        """

    def __len__(self) -> int:
        return len(self.tasks)

    @typing.overload
    def __getitem__(self, index: int) -> LoweringTask: ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.Self: ...

    @typing.overload
    def __getitem__(self, index: list[int]) -> typing.Self: ...

    def __getitem__(self, index):
        """Get the task at a position, or a new :class:`~.TaskSequence` of the tasks in a slice."""
        if isinstance(index, slice):
            return type(self)(self.tasks[index])
        if isinstance(index, list):
            return type(self)(self.tasks[idx] for idx in index)
        return self.tasks[index]

    def __setitem__(self, index, task):
        self.tasks[index] = task

    def __delitem__(self, index):
        del self.tasks[index]

    def insert(self, index: int, value: LoweringTask):
        """Insert ``task`` before position ``index``."""
        self.tasks.insert(index, value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskSequence):
            return NotImplemented
        return self.tasks == other.tasks

    def __add__(self, other: Iterable[LoweringTask]) -> typing.Self:
        return type(self)(chain(self.tasks, other))

    def __radd__(self, other: Iterable[LoweringTask]) -> typing.Self:
        return type(self)(chain(other, self.tasks))

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.tasks!r})"


class TaskStages(LoweringTask, MutableMapping):
    """A mutable sequence of named tasks to run in order.

    Task stages have the same execution behavior as a :class:`~.TaskSequence`. They simply
    additionally assign a name to each task they run.

    Task stages are permitted to be ill-typed while they are being assembled: they may hold two
    consecutive tasks whose IR types disagree.  IR type checking happens when these task stages are
    given to a :class:`LoweringPassManager`.
    """

    def __init__(self, stages: StagesLike = (), /, **kw_stages: LoweringTask):
        """
        Args:
            stages: the initial stages, as a mapping from name to task or an iterable of
                ``(name, task)`` pairs.
            kw_stages: initial stages whose names are given by keyword arguments, appended to any
                entries of ``stages``.

        Raises:
            TypeError: if ``stages`` is not a mapping or an iterable of ``(name, task)`` pairs.
            ValueError: if a stage name is repeated, including between ``stages`` and
                ``kw_stages``.
        """
        self.stages: dict[str, LoweringTask] = _as_stages(stages, kw_stages)
        """The stages, in execution order.
        
        This dictionary can be safely mutated in-place.
        """

    def _index_of(self, pos: str | int) -> int:
        """Resolve a stage name, or a position that may be negative, to an index.

        Raises:
            KeyError: if ``pos`` is a name that is not present.
            IndexError: if ``pos`` is a position that is out of range.
        """
        if isinstance(pos, str):
            try:
                return list(self.stages).index(pos)
            except ValueError:
                raise KeyError(f"no stage named {pos!r}") from None
        if not -len(self.stages) <= pos < len(self.stages):
            raise IndexError(f"stage position out of range: {pos}")
        return pos % len(self.stages)

    def _splice(self, at: int, name: str, task: LoweringTask) -> typing.Self:
        """Return new task stages with ``name`` inserted at index ``at``."""
        items = [pair for pair in self.stages.items() if pair[0] != name]
        return type(self)(chain(items[:at], [(name, task)], items[at:]))

    def insert_before(self, pos: str | int, stage: tuple[str, LoweringTask], /) -> typing.Self:
        """Return new task stages with ``stage`` inserted directly before ``pos``.

        Args:
            pos: the stage to insert before, by name or by position.
            stage: the ``(name, task)`` pair to insert.  A name that is already present moves to
                the new position.

        Raises:
            KeyError: if ``pos`` is a name that is not present.
            IndexError: if ``pos`` is a position that is out of range.
            TypeError: if ``stage`` is not a ``(name, task)`` pair.
        """
        return self._splice(self._index_of(pos), *_as_stage(stage))

    def insert_after(self, pos: str | int, stage: tuple[str, LoweringTask], /) -> typing.Self:
        """Return new task stages with ``stage`` inserted directly after ``pos``.

        Args:
            pos: the stage to insert after, by name or by position.
            stage: the ``(name, task)`` pair to insert.  A name that is already present moves to
                the new position.

        Raises:
            KeyError: if ``pos`` is a name that is not present.
            IndexError: if ``pos`` is a position that is out of range.
            TypeError: if ``stage`` is not a ``(name, task)`` pair.
        """
        return self._splice(self._index_of(pos) + 1, *_as_stage(stage))

    def __len__(self) -> int:
        return len(self.stages)

    @typing.overload
    def __getitem__(self, index: str | int) -> LoweringTask: ...

    @typing.overload
    def __getitem__(self, index: slice) -> typing.Self: ...

    @typing.overload
    def __getitem__(self, index: list[str | int]) -> typing.Self: ...

    def __getitem__(self, index):
        """Get the task of a stage by name or position, or new task stages from a slice."""
        if isinstance(index, str):
            return self.stages[index]
        items = list(self.stages.items())
        if isinstance(index, slice):
            return type(self)(items[index])
        if isinstance(index, list):
            return type(self)(items[self._index_of(pos)] for pos in index)
        return items[index][1]

    def __setitem__(self, name: str, task: LoweringTask):
        self.stages[name] = task

    def __delitem__(self, name: str):
        del self.stages[name]

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate over the stage names, in execution order."""
        return iter(self.stages)

    def __eq__(self, other: object) -> bool:
        """Compare equal to other task stages with the same stages in the same order."""
        # `Mapping` provides an `__eq__` that ignores order and accepts any mapping.  Stage order
        # is execution order, so it is part of the value, and a plain dict is not a task.
        if not isinstance(other, TaskStages):
            return NotImplemented
        return list(self.stages.items()) == list(other.stages.items())

    def __add__(self, other: StagesLike) -> typing.Self:
        """Return new task stages of this object's stages updated with ``other``'s."""
        out = type(self)(self.stages)
        out.update(_as_stages(other))
        return out

    def __radd__(self, other: StagesLike) -> typing.Self:
        """Return new task stages of ``other``'s stages updated with this object's."""
        out = type(self)(other)
        out.update(self.stages)
        return out

    def __iadd__(self, other: StagesLike) -> typing.Self:
        """Update these task stages in place with ``other``'s stages."""
        self.update(_as_stages(other))
        return self

    @reprlib.recursive_repr()
    def __repr__(self) -> str:
        return f"{type(self).__name__}({list(self.stages.items())!r})"


def _as_stage(stage: tuple[str, LoweringTask]) -> tuple[str, LoweringTask]:
    """Validate a single ``(name, task)`` pair."""
    match stage:
        case (str(), _):
            return stage
    raise TypeError(f"expected a (name, task) pair, got {stage!r}")


def _as_stages(
    stages: StagesLike, extra: Mapping[str, LoweringTask] = MappingProxyType({})
) -> dict[str, LoweringTask]:
    """Normalize stages-like input to a single dictionary.

    Raises:
        TypeError: if ``stages`` is not a mapping or an iterable of ``(name, task)`` pairs.
        ValueError: if a stage name is repeated, including between ``stages`` and ``extra``.
    """
    if isinstance(stages, Mapping):
        pairs = list(stages.items())
    else:
        try:
            pairs = [_as_stage(stage) for stage in stages]
        except TypeError as exc:
            raise TypeError(
                f"expected a mapping or an iterable of (name, task) pairs, got {stages!r}"
            ) from exc
    pairs += extra.items()
    out = dict(pairs)
    if len(out) != len(pairs):
        counts = Counter(name for name, _ in pairs)
        repeated = sorted(name for name, count in counts.items() if count > 1)
        raise ValueError(f"stage names must be unique, but these repeat: {repeated}")
    return out


@typing.final
class PassContextHandle:
    """The execution context for a given pass in a lowering pipeline.

    This object is created within the Rust components of the pass manager; you do not need to
    instantiate one yourself.

    This object is given to the :meth:`Pass._qiskit_pass_run_` method, and passes may use its
    methods to interact with the execution context.  The object is specific to a single call of
    :meth:`._qiskit_pass_run_`; its data is invalidated once that method has returned, and attempts
    to use it will just raise exceptions.

    You should not store or leak instances of this object anywhere.
    """

    _native: passmanager.PassContextHandle

    def __init__(self, native: passmanager.PassContextHandle):
        self._native = native

    def get(self, key: str, default=None) -> object:
        """Get the ``key`` from the pipeline's inter-pass context, or return a default.

        Raises:
            TypeError: if the key is present, but could not be exposed to Python.
        """
        return self._native.get_context(key, default)

    def __getitem__(self, key: str) -> object:
        """Get the ``key`` from the pipeline's inter-pass context.

        Raises:
            KeyError: if the key is not present.  Use :meth:`get` to avoid this.
            TypeError: if the key is present, but could not be exposed to Python.
        """
        sentinel = object()
        if (out := self._native.get_context(key, sentinel)) is sentinel:
            raise KeyError(f"key not found: {key}")
        return out

    def __setitem__(self, key: str, value: object) -> None:
        """Set the ``key`` to a given ``value`` in the pipeline's inter-pass context."""
        self._native.set_context(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete the ``key`` from the pipeline's inter-pass context.

        Raises:
            KeyError: if the key is not present.
        """
        self._native.del_context(key)

    @property
    def ir_modified(self) -> bool:
        """Whether the pass has modified the IR or not.

        This begins by assuming that the pass *did* modify the IR, but the pass author can assign
        ``False`` to assert that nothing was modified.
        """
        return self._native.get_ir_modified()

    @ir_modified.setter
    def ir_modified(self, val: bool):
        self._native.set_ir_modified(val)


class Pass(Generic[IRIn, IROut], LoweringTask):
    """An interface for defining passes over IRs.

    This is primarily an *implementation* interface.  The typical way to safely consume a
    :class:`Pass` is to put it into a :class:`LoweringPassManager`, and call the pass manager's
    methods.

    Note that all methods named ``_qiskit_*_`` are interface methods that are only for Qiskit to
    call.  These are documented for implementers, but are generally not stable for users to call;
    Qiskit may allow implementations to fulfill one of several contracts for the method, and
    additional alternatives may be added in new versions.  Qiskit will maintain stability within its
    own use of these protocols, but they are not stable for public consumption.

    Implementers of subclasses should override the methods, and are free to change the methods and
    signatures in line with this documentation about per-Qiskit-version calling signatures.

    Qiskit reserves the attribute and method namespace ``_qiskit_pass_*_`` for future expansion of
    this interface.  Qiskit will not define any attributes or methods in the interface outside this
    namespace.
    """

    _qiskit_pass_name_: str = ""
    """A human-readable name for the pass.  **Optional**.

    This is provided primarily for debugging purposes in pipelines."""

    # TODO: these could possibly be automatically set by `__init_subclass__` by inspection of the
    # generics, but we'd have to clearly define the semantics of `ForwardRef`, and other funky
    # type-system stuff.  For a first implementation, just require manual specification.
    _qiskit_pass_ir_in_: type[AnyIR]
    """The type of the IR that the pass expects."""
    _qiskit_pass_ir_out_: type[AnyIR]
    """The type of the IR that the pass outputs."""

    def __repr__(self) -> str:
        return f"{self._qiskit_pass_name_ or type(self).__name__}()"

    @abc.abstractmethod
    def _qiskit_pass_run_(self, ir: IRIn, context: PassContextHandle) -> IROut:
        """Run the pass on the given IR, returning the next IR object.

        The pass "owns" the ``ir`` object that comes in, and can do anything it likes with it;
        passes can assume they hold the only reference to the object.  However, the execution
        environment of passes also assumes that it has sole ownership of the output in a similar
        manner.  This means that you cannot "leak" references to the value returned outside the
        function; if you store references anywhere, they may see nonsense but valid data after this
        method has completed, or cause the execution pipeline to raise an access error.  If you want
        to leak out the data, you will need to ensure you produce copies.

        Args:
            ir: the input IR to run the pass on.  This can be assumed to be of the correct type as
                specified by the left generic in the subclass.
            context: a context object for interaction with the executing pipeline.  See the type
                documentation of :class:`PassContextHandle` for more information.

        Returns:
            The next IR object to use.  This must match the specified right generic from the
            subclass of :class:`Pass` (or the only generic, if used in single-argument
            form).

        Raises:
            PassError: when the pass encounters a compilation-specific error and wants to
                provide structured error messages back to the user.
        """


def _native_pass_from_lowering_pass(pass_: Pass) -> passmanager.PyPass:
    """Interpret the given implementation of :class:`Pass`, resolving any version /
    calling-signature conventions, into a Rust-native ``PyPass``."""
    # Right now this is a fairly simple passthrough.
    name = pass_._qiskit_pass_name_ or type(pass_).__name__
    return passmanager.PyPass(pass_, name, pass_._qiskit_pass_ir_in_, pass_._qiskit_pass_ir_out_)


@typing.final
class LoweringPassManager:
    """A multi-step pass manager that may lower through several IRs.

    Internally, this holds a handle to a low-level Rust-native pass manager.  This class then
    provides a Pythonic interface for interacting with the base object.  The corresponding object of
    the Rust-native component in the C API is :c:type:`QkPassManager`.

    .. warning::
        Due to internal limitations, users must ensure that this object does not participate in any
        Python reference cycles; the Python garbage collector cannot fully traverse this object.  We
        intend to lift this restriction in the future.
    """

    def __init__(self, tasks: Iterable[LoweringTask] = ()) -> None:
        """
        Args:
            tasks: any tasks with which to initialize the pass manager.  Passing this is equivalent
                to calling :meth:`append` in a loop with the arguments.

        Raises:
            TypeError: if any element is not an executable :class:`LoweringTask`, or if the
                resulting pipeline is ill-typed.
        """
        self._native = passmanager.PassManager()
        for task in tasks:
            self.append(task)

    def append(self, task: LoweringTask, /):
        """Push a task to the end of the current pipeline.

        Args:
            task: the task to push.  This may be a single :class:`Pass`, or a container such as a
                :class:`TaskSequence` or :class:`TaskStages`.

        Raises:
            TypeError: if ``task`` is not a :class:`Pass`, :class:`TaskSequence` or
                :class:`TaskStages`, if it is ill-typed internally, or if its input IR type
                does not match the output type of the pipeline so far.  The pass manager is left
                unmodified.
        """
        self._native.push_task(task)

    def __repr__(self) -> str:
        return repr(self._native)

    def run(self, ir: IRIn, *, copy: bool = True) -> object:
        """Run the pipeline on the given IR.

        Args:
            ir: the initial IR.
            copy: by default, attempt to :func:`copy.copy` the input on entry.  If set to ``False``,
                then the pass manager assumes complete ownership of ``ir``, and may arbitrarily
                mutate the object.  This will typically result in the object becoming empty.

        Returns:
            The compiled and lowered IR.

        Raises:
            LoweringPassManagerError: if the pipeline raised an error (or errors) related to the
                actual compilation process.
            TypeError: if ``ir`` does not match the type expected by the first pass.
        """
        if copy:
            ir = _copy.copy(ir)
        return self._native.run_simple(ir)

    def to_task_sequence(self) -> TaskSequence:
        """Return a new task sequence of the tasks that this pass manager runs.

        Mutating the returned task sequence does not affect this pass manager. If your goal is to
        modify the structure of this pass manager, modify the task sequence as needed and then
        instantiate a new pass manager. Note, on the other hand, that :meth:`~.append` mutates this
        pass manager in-place. The :class:`~.Pass` objects within are shared references
        rather than copies.

        .. code-block:: python

            pass_manager = LoweringPassManager([MyLoweringPass(), MyOptimizationPass()])

            # Insert a pass in the middle
            tasks = pass_manager.to_task_sequence()
            tasks.insert(1, MyAnalysisPass())
            edited = LoweringPassManager(tasks)

        Returns:
             A new :class:`TaskSequence`.
        """
        return TaskSequence(self._native.to_py_tasks())
