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
import typing
from collections.abc import Iterable
from typing import Generic, ClassVar, TypeAlias
from typing_extensions import TypeVar

from qiskit._accelerate import passmanager
from .exceptions import PassManagerError

__all__ = [
    "IR",
    "LoweringPassManager",
    "LoweringPassManagerError",
    "Pass",
    "PassContextHandle",
    "PassError",
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


class Pass(Generic[IRIn, IROut], abc.ABC):
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

    def __init__(self, tasks: Iterable[Pass] = ()) -> None:
        """
        Args:
            tasks: any tasks with which to initialize the pass manager.  Passing this is equivalent
                to calling :meth:`append` in a loop with the arguments.
        """
        self._native = passmanager.PassManager()
        for task in tasks:
            self.append(task)

    def append(self, pass_: Pass, /):
        """Push a Python-native pass to the end of the current pipeline.

        Args:
            pass_: the pass to push.
        """
        # We can extend this method with dispatched type-checking once we expose other tasks, etc.
        self._native.push_pass(_native_pass_from_lowering_pass(pass_))

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
