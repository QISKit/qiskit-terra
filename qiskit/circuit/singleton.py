# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
========================================================
Singleton instructions (:mod:`qiskit.circuit.singleton`)
========================================================

.. currentmodule:: qiskit.circuit.singleton

The machinery in this module is for defining subclasses of :class:`~.circuit.Instruction` and
:class:`.Gate` that preferentially return a shared immutable singleton instance when instantiated.
Taking the example of :class:`.XGate`, the final user-facing result is that:

* There is a regular class called :class:`XGate`, which derives from :class:`.Gate`.

* Doing something like ``XGate(label="my_gate")`` produces an object whose type is exactly
  ``XGate``, and all the mutability works completely as expected; all the methods resolve to exactly
  those defined by :class:`.XGate`, :class:`.Gate`, or parents.

* Doing ``XGate()`` produces a singleton object whose type is a synthetic ``_SingletonXGate`` class,
  which derives :class:`.XGate` but overrides :meth:`~object.__setattr__` to make itself immutable.
  The object itself has precisely the same instance attributes as ``XGate()`` would have if there
  was no singleton handling.  This object will return itself under :func:`~copy.copy`,
  :func:`~copy.deepcopy` and roundtrip through :mod:`pickle`.

The same can be true for, for example, :class:`.Measure`, except that it's a subclass of
:class:`~.circuit.Instruction` only, and not :class:`.Gate`.

.. note::

    The classes in this module are for advanced use, because they are closely entwined with the
    heart of Qiskit's data model for circuits.

From a library-author perspective, the minimum that is needed to enhance a :class:`.Gate` or
:class:`~.circuit.Instruction` with this behavior is to inherit from :class:`SingletonGate`
(:class:`SingletonInstruction`) instead of :class:`.Gate` (:class:`~.circuit.Instruction`), and for
the ``__init__`` method to have defaults for all of its arguments (these will be the state of the
singleton instance).  For example::

    class XGate(SingletonGate):
        def __init__(self, label=None):
            super().__init__("x", 1, [], label=label)

    assert XGate() is XGate()


Interface
=========

The public classes correspond to the standard classes :class:`~.circuit.Instruction` and
:class:`.Gate`, respectively, and are subclasses of these.

.. autoclass:: SingletonInstruction
   :class-doc-from: class
.. autoclass:: SingletonGate
   :class-doc-from: class
.. autoclass:: SingletonControlledGate
   :class-doc-from: class

When inheriting from one of these classes, the produced class will have an eagerly created singleton
instance that will be returned whenever the class is constructed with arguments that have been
defined to be singletons.  Typically this will be the defaults.  These instances are immutable;
attempts to modify their properties will raise :exc:`TypeError`.

*All* subclasses of :class:`~.circuit.Instruction` have a :attr:`~.Instruction.mutable` property.
For most instructions this is ``True``, while for the singleton instances it is ``False``.  One can
use the :meth:`~.Instruction.to_mutable` method to get a version of the instruction that is owned
and safe to mutate.

The singleton instances are not exact instances of their base class; they are special subclasses
that cannot construct new objects.  This means that::

    type(XGate()) is not XGate

You should not rely on :class:`type` having an exact value; use :func:`isinstance` instead for type
checking.  If you need to reliably retrieve the base class from an :class:`~.circuit.Instruction`,
see the :attr:`.Instruction.base_class` attribute; singleton instances set this correctly.  For most
cases in using Qiskit, :attr:`.Instruction.name` is a more suitable determiner of what an
instruction "means" in a circuit.

Deriving new singletons
-----------------------

The simplest example of deriving a new singleton instruction is simply to inherit from the correct
base and supply an :meth:`~object.__init__` method that has immutable defaults for any arguments.
For example::

    from qiskit.circuit.singleton import SingletonInstruction

    class MyInstruction(SingletonInstruction):
        def __init__(self, label=None):
            super().__init__("my_instruction", 1, 0, label=label)

    assert MyInstruction() is MyInstruction()
    assert MyInstruction(label="some label") is not MyInstruction()
    assert MyInstruction(label="some label").mutable

The singleton instance will use all the constructor's defaults.

You can also derive from an instruction that is itself a singleton.  The singleton nature of the
class will be inherited, though the singleton instances of the two classes will be different::

    class MyOtherInstruction(MyInstruction):
        pass

    assert MyOtherInstruction() is MyOtherInstruction()
    assert MyOtherInstruction() is not MyInstruction()

If for some reason you want to derive from :class:`SingletonInstruction`, or one of the related or
subclasses but *do not* want the default singleton instance to be created, such as if you are
defining a new abstract base class, you can set the keyword argument
``create_default_singleton=False`` in the class definition::

    class NotASingleton(SingletonInstruction, create_default_singleton=False):
        def __init__(self):
            return super().__init__("my_mutable", 1, 0, [])

    assert NotASingleton() is not NotASingleton()

If your constructor does not have defaults for all its arguments, you must set
``create_default_singleton=False``.

Subclasses of :class:`SingletonInstruction` and the other associated classes can control how their
constructor's arguments are interpreted, in order to help the singleton machinery return the
singleton even in the case that an optional argument is explicitly set to its default.

.. automethod:: SingletonInstruction._singleton_lookup_key

This is set by all Qiskit standard-library gates such that the :attr:`~Instruction.label` and
similar keyword arguments are ignored in the key calculation if they are their defaults, or a
mutable instance is returned if they are not.

You can also specify other combinations of constructor arguments to produce singleton instances
for, using the ``additional_singletons`` argument in the class definition.  This takes an iterable
of ``(args, kwargs)`` tuples, and will build singletons equivalent to ``cls(*args, **kwargs)``.  You
do not need to handle the case of the default arguments with this.  For example, given a class
definition::

    class MySingleton(SingletonGate, additional_singletons=[((2,), {"label": "two"})]):
        def __init__(self, n=1, label=None):
            super().__init__("my", n, [], label=label)

        @staticmethod
        def _singleton_lookup_key(n=1, label=None):
            return (n, label)

there will be two singleton instances instantiated.  One corresponds to ``n=1`` and ``label=None``,
and the other to ``n=2`` and ``label="two"``.  Whenever ``MySingleton`` is constructed with
arguments consistent with one of those two cases, the relevant singleton will be returned.  For
example::

    assert MySingleton() is MySingleton(1, label=None)
    assert MySingleton(2, "two") is MySingleton(n=2, label="two")

The case of the class being instantiated with zero arguments is handled specially to allow an
absolute fast-path for inner-loop performance (although the general machinery is not desperately
slow anyway).


Implementation
==============

.. note::

    This section is primarily developer documentation for the code; none of the machinery described
    here is public, and it is not safe to inherit from any of it directly.

There are several moving parts to tackle here.  The behavior of having ``XGate()`` return some
singleton object that is an (inexact) instance of :class:`.XGate` but *without* calling ``__init__``
requires us to override :class:`type.__call__ <type>`.  This means that :class:`.XGate` must have a
metaclass that defines ``__call__`` to return the singleton instance.

Next, we need to ensure that there *is* a singleton instance for ``XGate()`` to return.  This can be
done dynamically on each call (i.e. check if the instance exists and create it if not), but since
we also want that instance to be very special, it's easier to hook in and create it during the
definition of the ``XGate`` type object.  This also has the advantage that we do not need to make
the singleton object pickleable; we only need to specify where to retrieve it from during the
unpickle, because the creation of the base type object will recreate the singleton.

We want the singleton instance to:

* be immutable; it should reject all attempts to mutate itself.
* have exactly the same state as an ``XGate()`` would have had if there was no singleton handling.

We do this in a three-step procedure:

1. Before creating any singletons, we separately define the overrides needed to make an
   :class:`~.circuit.Instruction` and a :class:`.Gate` immutable.  This is
   ``_SingletonInstructionOverrides`` and the other ``_*Overrides`` classes.

2. While we are creating the ``XGate`` type object, we dynamically *also* create a subclass of it
   that has the immutable overrides in its method-resolution order in the correct place. These
   override the standard methods / properties that are defined on the mutable gate (we do not
   attempt to override any cases where the type object we are creating has extra inplace methods).

3. We can't instantiate this new subclass, because when it calls ``XGate.__init__``, it will attempt
   to set some attributes, and these will be rejected by immutability.  Instead, we first create a
   completely regular ``XGate`` instance, and then we dynamically change its type to the singleton
   class, freezing it.

We could do this entirely within the metaclass machinery, but that would require ``XGate`` to be
defined as something like::

  class XGate(Gate, metaclass=_SingletonMeta, overrides=_SingletonGateOverrides): ...

which is super inconvenient (or we'd have to have ``_SingletonMeta`` do a bunch of fragile
introspection).  Instead, we use the :class:`abc.ABC`/:class:`abc.ABCMeta` pattern of defining a
concrete middle class (:class:`SingletonGate` in the :class:`.XGate` case) that sets the metaclass,
selects the overrides to be applied, and has an :meth:`~object.__init_subclass__` that applies the
singleton-subclass-creation steps above.  The overrides are in separate classes so that *mutable*
:class:`.XGate` instances do not have them in their own method-resolution orders; doing this is
easier to implement, but requires all the setters and checkers to dance around at runtime trying to
validate whether mutating the instance is allowed.

Finally, to actually build all this machinery up, the base is ``_SingletonMeta``, which is a
metaclass compatible with any metaclass of :class:`~.circuit.Instruction`.  This defines the
:meth:`~object.__call__` machinery that overrides :class:`type.__call__ <type>` to return the
singleton instances.  The other component of it is its :meth:`~object.__new__`, which is called
(non-trivially) during the creation of :class:`SingletonGate` and :class:`SingletonInstruction` with
its ``overrides`` keyword argument set to define the ``__init_subclass__`` of those classes with the
above properties.  We use the metaclass to add this method dynamically, because the
:meth:`~object.__init_subclass__` machinery wants to be abstract, closing over the ``overrides`` and
the base class, but still able to call :class:`super`.  It's more convenient to do this dynamically,
closing over the desired class variable and using the two-argument form of :class:`super`, since the
zero-argument form does magic introspection based on where its containing function was defined.

Handling multiple singletons requires storing the initialization arguments in some form, to allow
the :meth:`~.Instruction.to_mutable` method and pickling to be defined.  We do this as a lookup
dictionary on the singleton *type object*.  This is logically an instance attribute, but because we
need to dynamically switch in the dynamic `_Singleton` type onto an instance of the base type, that
gets rather complex; either we have to require that the base already has an instance dictionary, or we
risk breaking the ``__slots__`` layout during the switch.  Since the singletons have lifetimes that
last until garbage collection of their base class's type object, we can fake out this instance
dictionary using a type-object dictionary that maps instance pointers to the data we want to store.
An alternative would be to build a new type object for each individual singleton that closes over
(or stores) the initializer arguments, but type objects are quite heavy and the principle is largely
same anyway.
"""

from __future__ import annotations

import functools

from .instruction import Instruction
from .gate import Gate
from .controlledgate import ControlledGate, _ctrl_state_to_int


def _impl_init_subclass(
    base: type[_SingletonBase], overrides: type[_SingletonInstructionOverrides]
):
    # __init_subclass__ for the classes that make their children singletons (e.g. `SingletonGate`)

    def __init_subclass__(
        instruction_class, *, create_default_singleton=True, additional_singletons=(), **kwargs
    ):
        super(base, instruction_class).__init_subclass__(**kwargs)
        if not create_default_singleton and not additional_singletons:
            return

        # If we're creating singleton instances, then the _type object_ needs a lookup mapping the
        # "keys" to the pre-created singleton instances.  It can't share this with subclasses.
        instruction_class._singleton_static_lookup = {}

        class _Singleton(overrides, instruction_class, create_default_singleton=False):
            __module__ = None
            # We want this to match the slots layout (if any) of `cls` so it's safe to dynamically
            # switch the type of an instance of `cls` to this.
            __slots__ = ()

            # Class variables mapping singleton instances (as pointers) to the arguments used to
            # create them, for use by `to_mutable` and `__reduce__`.  We're safe to use the `id` of
            # (value of the pointer to) each object because they a) are singletons and b) have
            # lifetimes tied to the type object in their `base_class`, so will not be garbage
            # collected until the class no longer exists.  This is effectively faking out an entry
            # in an instance dictionary, but this works without affecting the slots layout, and
            # doesn't require that the object has an instance dictionary.
            _singleton_init_arguments = {}

            # Docstrings are all inherited, and we use more descriptive class methods to better
            # distinguish the `_Singleton` class (`singleton_class`) from the instruction class
            # (`instruction_class`) that it's wrapping.

            def __new__(singleton_class, *_args, **_kwargs):
                raise TypeError(f"cannot create '{singleton_class.__name__}' instances")

            @property
            def base_class(self):
                return instruction_class

            @property
            def mutable(self):
                return False

            def to_mutable(self):
                args, kwargs = type(self)._singleton_init_arguments[id(self)]
                return self.base_class(*args, **kwargs, _force_mutable=True)

            def __setattr__(self, key, value):
                raise TypeError(
                    f"This '{self.base_class.__name__}' object is immutable."
                    " You can get a mutable version by calling 'to_mutable()'."
                )

            def __copy__(self):
                return self

            def __deepcopy__(self, memo=None):
                return self

            def __reduce__(self):
                # The principle is that the unpickle operation will first create the `base_class`
                # type object just by re-importing its module so all the singletons are guaranteed
                # to exist before we get to doing anything with these arguments.  All we then need
                # to do is pass the init arguments to the base type object and its logic will return
                # the singleton object.
                args, kwargs = type(self)._singleton_init_arguments[id(self)]
                return (functools.partial(instruction_class, **kwargs), args)

        # This is just to let the type name offer slightly more hint to what's going on if it ever
        # appears in an error message, so it says (e.g.) `_SingletonXGate`, not just `_Singleton`.
        _Singleton.__name__ = _Singleton.__qualname__ = f"_Singleton{instruction_class.__name__}"

        def _create_singleton_instance(args, kwargs):
            # Make a mutable instance, fully instantiate all lazy properties, then freeze it.
            out = instruction_class(*args, **kwargs, _force_mutable=True)
            out = overrides._prepare_singleton_instance(out)
            out.__class__ = _Singleton

            _Singleton._singleton_init_arguments[id(out)] = (args, kwargs)
            key = instruction_class._singleton_lookup_key(*args, **kwargs)
            if key is not None:
                instruction_class._singleton_static_lookup[key] = out
            return out

        # This static lookup is only for singletons generated at class-description time.  A separate
        # lookup that manages an LRU or similar cache should be used for singletons created on
        # demand.  This static dictionary is separate to ensure that the class-requested singletons
        # have lifetimes tied to the class object, while dynamic ones can be freed again.
        if create_default_singleton:
            instruction_class._singleton_default_instance = _create_singleton_instance((), {})
        for class_args, class_kwargs in additional_singletons:
            _create_singleton_instance(class_args, class_kwargs)

    return classmethod(__init_subclass__)


class _SingletonMeta(type(Instruction)):
    # The inheritance above is to ensure metaclass compatibility with `Instruction`, though pylint
    # doesn't understand that this is a subclass of `type`, so uses metaclass `self` conventions.

    # Beware the difference between `type.__new__` and `type.__call__`.  The former is called during
    # creation of the _type object_ (e.g. during a `class A: ...` statement), and the latter is
    # called by type instantiation (e.g. `A()`).

    __slots__ = ()

    def __new__(mcs, name, bases, namespace, *, overrides=None, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if overrides is not None:
            # The `__init_subclass__` logic is shared between `SingletonInstruction` and
            # `SingletonGate`, but we can't do `__init_subclass__ = _impl_init_subclass(...)`
            # inside the class definition bodies because it couldn't make `super()` calls.
            cls.__init_subclass__ = _impl_init_subclass(cls, overrides)
        return cls

    def __call__(cls, *args, _force_mutable=False, **kwargs):
        if _force_mutable:
            return super().__call__(*args, **kwargs)
        if not args and not kwargs:
            # This is a fast-path to handle constructions of the form `XGate()`, which is the
            # idiomatic way of building gates during high-performance circuit construction.  If
            # there are any arguments or kwargs, we delegate to the overridable method to
            # determine the cache key to use for lookup.
            return cls._singleton_default_instance
        if (key := cls._singleton_lookup_key(*args, **kwargs)) is not None:
            try:
                singleton = cls._singleton_static_lookup.get(key)
            except TypeError:
                # Catch the case of the returned key being unhashable; a subclass could not easily
                # determine this because it's working with arbitrary user inputs.
                singleton = None
            if singleton is not None:
                return singleton
            # The logic can be extended to have an LRU cache for key requests that are absent,
            # to allow things like parametric gates to have reusable singletons as well.
        return super().__call__(*args, **kwargs)


class _SingletonBase(metaclass=_SingletonMeta):
    """Base class of all the user-facing (library-author-facing) singleton classes such as
    :class:`SingletonGate`.

    This defines the shared interface for those singletons."""

    __slots__ = ()

    @staticmethod
    def _singleton_lookup_key(*_args, **_kwargs):
        """Given the arguments to the constructor, return a key tuple that identifies the singleton
        instance to retrieve, or ``None`` if the arguments imply that a mutable object must be
        created.

        For performance, as a special case, this method will not be called if the class constructor
        was given zero arguments (e.g. the construction ``XGate()`` will not call this method, but
        ``XGate(label=None)`` will), and the default singleton will immediately be returned.

        This static method can (and probably should) be overridden by subclasses.  The derived
        signature should match the class's ``__init__``; this method should then examine the
        arguments to determine whether it requires mutability, or what the cache key (if any) should
        be.

        The function should return either ``None`` or valid ``dict`` key (i.e. hashable and
        implements equality).  Returning ``None`` means that the created instance must be mutable.
        No further singleton-based processing will be done, and the class creation will proceed as
        if there was no singleton handling.  Otherwise, the returned key can be anything hashable
        and no special meaning is ascribed to it.  Whenever this method returns the same key, the
        same singleton instance will be returned.  We suggest that you use a tuple of the values of
        all arguments that can be set while maintaining the singleton nature.

        Only keys that match the default arguments or arguments given to ``additional_singletons``
        at class-creation time will actually return singletons; other values will return a standard
        mutable instance.

        .. note::

            The singleton machinery will handle an unhashable return from this function gracefully
            by returning a mutable instance.  Subclasses should ensure that their key is hashable in
            the happy path, but they do not need to manually verify that the user-supplied arguments
            are hashable.  For example, it's safe to implement this as::

                @staticmethod
                def _singleton_lookup_key(*args, **kwargs):
                    return None if kwargs else args

            even though a user might give some unhashable type as one of the ``args``.
        """
        return None


class _frozenlist(list):
    __slots__ = ()

    def _reject_mutation(self, *args, **kwargs):
        raise TypeError("'params' of singletons cannot be mutated")

    append = clear = extend = insert = pop = remove = reverse = sort = _reject_mutation
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _reject_mutation


class _SingletonInstructionOverrides(Instruction):
    """Overrides for the mutable methods and properties of `Instruction` to make it immutable."""

    __slots__ = ()

    # The split between what's defined here and what's defined in the dynamic `_Singleton` class is
    # slightly arbitrary, but generally these overrides are for things that are about the nature of
    # the `Instruction` class itself, while `_Singleton` handles the Python data model and things
    # that can't be written in terms of the `Instruction` interface (like the overrides of
    # `base_class` and `to_mutable`).

    @staticmethod
    def _prepare_singleton_instance(instruction: Instruction):
        """Class-creation hook point.  Given an instance of the type that these overrides correspond
        to, this method should ensure that all lazy properties and caches that require mutation to
        write to are eagerly defined.

        Subclass "overrides" classes can override this method if the user/library-author-facing
        class they are providing overrides for has more lazy attributes or user-exposed state
        with interior mutability."""
        instruction._define()
        # We use this `list` subclass that rejects all mutation rather than a simple `tuple` because
        # the `params` typing is specified as `list`. Various places in the library and beyond do
        # `x.params.copy()` when they want to produce a version they own, which is good behavior,
        # and would fail if we switched to a `tuple`, which has no `copy` method.
        instruction._params = _frozenlist(instruction._params)
        return instruction

    def copy(self, name=None):
        if name is None:
            return self
        out = self.to_mutable()
        out.name = name
        return out


class SingletonInstruction(Instruction, _SingletonBase, overrides=_SingletonInstructionOverrides):
    """A base class to use for :class:`~.circuit.Instruction` objects that by default are singleton
    instances.

    This class should be used for instruction classes that have fixed definitions and do not contain
    any unique state. The canonical example of something like this is :class:`.Measure` which has an
    immutable definition and any instance of :class:`.Measure` is the same. Using singleton
    instructions as a base class for these types of gate classes provides a large advantage in the
    memory footprint of multiple instructions.

    The exception to be aware of with this class though are the :class:`~.circuit.Instruction`
    attribute :attr:`~.Instruction.label` which can be set differently for
    specific instances of gates.  For :class:`SingletonInstruction` usage to be sound setting these
    attributes is not available and they can only be set at creation time, or on an object that has
    been specifically made mutable using :meth:`~.Instruction.to_mutable`. If any of these
    attributes are used during creation, then instead of using a single shared global instance of
    the same gate a new separate instance will be created."""

    __slots__ = ()


class _SingletonGateOverrides(_SingletonInstructionOverrides, Gate):
    """Overrides for all the mutable methods and properties of `Gate` to make it immutable.

    This class just exists for the principle; there's no additional overrides required compared
    to :class:`~.circuit.Instruction`."""

    __slots__ = ()


class SingletonGate(Gate, _SingletonBase, overrides=_SingletonGateOverrides):
    """A base class to use for :class:`.Gate` objects that by default are singleton instances.

    This class is very similar to :class:`SingletonInstruction`, except implies unitary
    :class:`.Gate` semantics as well.  The same caveats around setting attributes in that class
    apply here as well."""

    __slots__ = ()


class _SingletonControlledGateOverrides(_SingletonInstructionOverrides, ControlledGate):
    """Overrides for all the mutable methods and properties of `ControlledGate` to make it immutable.

    This class just exists for the principle; there's no additional overrides required compared
    to :class:`~.circuit.Instruction`.
    """

    __slots__ = ()


class SingletonControlledGate(
    ControlledGate,
    _SingletonBase,
    overrides=_SingletonControlledGateOverrides,
):
    """A base class to use for :class:`.ControlledGate` objects that by default are singleton instances

    This class is very similar to :class:`SingletonInstruction`, except implies unitary
    :class:`.ControlledGate` semantics as well.  The same caveats around setting attributes in
    that class apply here as well.
    """

    __slots__ = ()


def stdlib_singleton_key(*, num_ctrl_qubits: int = 0):
    """Create an implementation of the abstract method
    :meth:`SingletonInstruction._singleton_lookup_key`, for standard-library instructions whose
    ``__init__`` signatures match the one given here.

    .. warning::

        This method is not safe for use in classes defined outside of Qiskit; it is not included in
        the backwards compatibility guarantees.  This is because we guarantee that the call
        signatures of the base classes are backwards compatible in the sense that we will only
        replace them (without warning) contravariantly, but if you use this method, you effectively
        use the signature *invariantly*, and we cannot guarantee that.

    Args:
        num_ctrl_qubits: if given, this implies that the gate is a :class:`.ControlledGate`, and
            will have a fixed number of qubits that are used as the control.  This is necessary to
            allow ``ctrl_state`` to be given as either ``None`` or as an all-ones integer/string.
    """

    if num_ctrl_qubits:

        def key(label=None, ctrl_state=None, *, _base_label=None):
            if label is None and _base_label is None:
                # Normalisation; we want all types for the control state to key the same.
                ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
                return (ctrl_state,)
            return None

    else:

        def key(label=None):
            if label is None:
                return ()
            return None

    return staticmethod(key)
