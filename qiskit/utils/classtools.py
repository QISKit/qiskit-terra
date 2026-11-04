# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tools useful for creating decorators, and other high-level callables."""

import functools
import inspect
import types
from collections.abc import Callable


# On user-defined classes, `__new__` is magically inferred to be a staticmethod, `__init_subclass__`
# is magically inferred to be a class method and `__prepare__` must be defined as a classmethod, but
# the CPython types implemented in C (such as `object` and `type`) are `types.BuiltinMethodType`,
# which we can't distinguish properly, so we need a little magic.
_MAGIC_STATICMETHODS = {"__new__"}
_MAGIC_CLASSMETHODS = {"__init_subclass__", "__prepare__"}


class _lift_to_method:
    """A decorator that ensures that an input callable object implements ``__get__``.  It is
    returned unchanged if so, otherwise it is turned into the default implementation for functions,
    which makes them bindable to instances.

    Python-space functions and lambdas already have this behavior, but builtins like ``print``
    don't; using this class allows us to do::

        wrap_method(MyClass, "maybe_mutates_arguments", before=print, after=print)

    to simply print all the arguments on entry and exit of the function, which otherwise wouldn't be
    valid, since ``print`` isn't a descriptor.
    """

    __slots__ = ("_method",)

    def __new__(cls, method):
        if hasattr(method, "__get__"):
            return method
        return super().__new__(cls)

    def __init__(self, method):
        if method is self:
            # Prevent double-initialization if we are passed an instance of this object to lift.
            return
        self._method = method

    def __get__(self, obj, objtype):
        # This is effectively the same implementation as `types.FunctionType.__get__`, but we can't
        # bind that directly because it also includes a requirement that its `self` reference is of
        # the correct type, and this isn't.
        if obj is None:
            return self._method
        return types.MethodType(self._method, obj)


class _WrappedMethod:
    """Descriptor which calls its two arguments in succession, correctly handling instance- and
    class-method calls.

    It is intended that this class will replace the attribute that ``inner`` previously was on a
    class or instance.  When accessed as that attribute, this descriptor will behave it is the same
    function call, but with the ``function`` called before or after.
    """

    __slots__ = ("_after", "_before", "_method", "_method_decorator", "_method_has_get")

    def __init__(self, method, before=None, after=None):
        if isinstance(method, (classmethod, staticmethod)):
            self._method_decorator = type(method)
        elif isinstance(method, type(self)):
            self._method_decorator = method._method_decorator
        elif getattr(method, "__name__", None) in _MAGIC_STATICMETHODS:
            self._method_decorator = staticmethod
        elif getattr(method, "__name__", None) in _MAGIC_CLASSMETHODS:
            self._method_decorator = classmethod
        else:
            self._method_decorator = _lift_to_method
        before = (self._method_decorator(before),) if before is not None else ()
        after = (self._method_decorator(after),) if after is not None else ()
        if isinstance(method, type(self)):
            self._method = method._method
            self._before = before + method._before
            self._after = method._after + after
        else:
            self._before = before
            self._after = after
            self._method = method
        # If the inner method doesn't have `__get__` (like some builtin methods), it's faster to
        # test a Boolean each time than the repeatedly raise and catch an exception, which is what
        # `hasattr` does.
        self._method_has_get = hasattr(self._method, "__get__")

    def __get__(self, obj, objtype=None):
        # `self._method` doesn't invoke the `_method` descriptor (if it is one) because that only
        # happens for class variables. Here it's an instance variable, so we can pass through `obj`
        # and `objtype` correctly like this.
        method = self._method.__get__(obj, objtype) if self._method_has_get else self._method

        @functools.wraps(method)
        def out(*args, **kwargs):
            for callback in self._before:
                callback.__get__(obj, objtype)(*args, **kwargs)
            retval = method(*args, **kwargs)
            for callback in self._after:
                callback.__get__(obj, objtype)(*args, **kwargs)
            return retval

        return out


def wrap_method(
    cls: type, name: str, *, before: Callable | None = None, after: Callable | None = None
):
    """Wrap the functionality the instance- or class method ``cls.name`` with additional behavior
    ``before`` and ``after``.

    This mutates ``cls``, replacing the attribute ``name`` with the new functionality.  This is
    useful when creating class decorators.  The method is allowed to be defined on any parent class
    instead.

    If either ``before`` or ``after`` are given, they should be callables with a compatible
    signature to the method referred to.  They will be called immediately before or after the method
    as appropriate, and any return value will be ignored.

    Args:
        cls: the class to modify.
        name: the name of the method on the class to wrap.
        before: a callable that should be called before the method that is being wrapped.
        after: a callable that should be called after the method that is being wrapped.

    Raises:
        ValueError: if the named method is not defined on the class or any parent class.
    """
    # The best time to apply decorators to methods is before they are bound (e.g. by using function
    # decorators during the class definition), but if we're making a class decorator, we can't do
    # that.  We need the actual definition of the method, so we have to dodge the normal output of
    # `type.__getattribute__`, which evaluates descriptors if it finds them.
    method = inspect.getattr_static(cls, name)
    setattr(cls, name, _WrappedMethod(method, before, after))
