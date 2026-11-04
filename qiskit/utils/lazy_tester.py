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

"""Lazy testers for optional features."""

import abc
import collections
import contextlib
import functools
import importlib
import subprocess
import typing
import warnings

from collections.abc import Iterable, Callable

from qiskit.exceptions import MissingOptionalLibraryError, OptionalDependencyImportWarning
from .classtools import wrap_method


class _RequireNow:
    """Helper callable that accepts all function signatures and simply calls
    :meth:`.LazyDependencyManager.require_now`.  This is helpful when used with :func:`.wrap_method`,
    as the callable needs to be compatible with all signatures and be picklable."""

    __slots__ = ("_feature", "_tester")

    def __init__(self, tester, feature):
        self._tester = tester
        self._feature = feature

    def __call__(self, *_args, **_kwargs):
        self._tester.require_now(self._feature)


class LazyDependencyManager(abc.ABC):
    """A manager for some optional features that are expensive to import, or to verify the
    existence of.

    These objects can be used as Booleans, such as ``if x``, and will evaluate ``True`` if the
    dependency they test for is available, and ``False`` if not.  The presence of the dependency
    will only be tested when the Boolean is evaluated, so it can be used as a runtime test in
    functions and methods without requiring an import-time test.

    These objects also encapsulate the error handling if their dependency is not present, so you can
    do things such as::

        from qiskit.utils import LazyImportManager
        HAS_MATPLOTLIB = LazyImportManager("matplotlib")

        @HAS_MATPLOTLIB.require_in_call
        def my_visualisation():
            ...

        def my_other_visualisation():
            # ... some setup ...
            HAS_MATPLOTLIB.require_now("my_other_visualisation")
            ...

        def my_third_visualisation():
            if HAS_MATPLOTLIB:
                from matplotlib import pyplot
            else:
                ...

    In all of these cases, ``matplotlib`` is not imported until the functions are entered.  In the
    case of the decorator, ``matplotlib`` is tested for import when the function is called for
    the first time.  In the second and third cases, the loader attempts to import ``matplotlib``
    when the :meth:`require_now` method is called, or when the Boolean context is evaluated.  For
    the ``require`` methods, an error is raised if the library is not available.

    This is the base class, which provides the Boolean context checking and error management.  The
    concrete classes :class:`LazyImportTester` and :class:`LazySubprocessTester` provide convenient
    entry points for testing that certain symbols are importable from modules, or certain
    command-line tools are available, respectively.
    """

    __slots__ = ("_bool", "_callback", "_install", "_msg", "_name")

    def __init__(self, *, name=None, callback=None, install=None, msg=None):
        """
        Args:
            name: the name of this optional dependency.
            callback: a callback that is called immediately after the availability of the library is
                tested with the result.  This will only be called once.
            install: how to install this optional dependency.  Passed to
                :class:`.MissingOptionalLibraryError` as the ``pip_install`` parameter.
            msg: an extra message to include in the error raised if this is required.
        """
        self._bool = None
        self._callback = callback
        self._name = name
        self._install = install
        self._msg = msg

    @abc.abstractmethod
    def _is_available(self) -> bool:
        """Subclasses of :class:`LazyDependencyManager` should override this method to implement the
        actual test of availability.  This method should return a Boolean, where ``True`` indicates
        that the dependency was available.  This method will only ever be called once.

        :meta public:
        """
        return False

    def __bool__(self):
        if self._bool is None:
            self._bool = self._is_available()
            if self._callback is not None:
                self._callback(self._bool)
        return self._bool

    @typing.overload
    def require_in_call(self, feature_or_callable: Callable) -> Callable: ...

    @typing.overload
    def require_in_call(self, feature_or_callable: str) -> Callable[[Callable], Callable]: ...

    def require_in_call(self, feature_or_callable):
        """Create a decorator for callables that requires that the dependency is available when the
        decorated function or method is called.

        Args:
            feature_or_callable (str or Callable): the name of the feature that requires these
                dependencies.  If this function is called directly as a decorator (for example
                ``@HAS_X.require_in_call`` as opposed to
                ``@HAS_X.require_in_call("my feature")``), then the feature name will be taken to be
                the function name, or class and method name as appropriate.

        Returns:
            Callable: a decorator that will make its argument require this dependency before it is
            called.
        """
        if isinstance(feature_or_callable, str):
            feature = feature_or_callable

            def decorator(function):
                @functools.wraps(function)
                def out(*args, **kwargs):
                    self.require_now(feature)
                    return function(*args, **kwargs)

                return out

            return decorator

        function = feature_or_callable
        feature = (
            getattr(function, "__qualname__", None)
            or getattr(function, "__name__", None)
            or str(function)
        )

        @functools.wraps(function)
        def out(*args, **kwargs):
            self.require_now(feature)
            return function(*args, **kwargs)

        return out

    @typing.overload
    def require_in_instance(self, feature_or_class: type) -> type: ...

    @typing.overload
    def require_in_instance(self, feature_or_class: str) -> Callable[[type], type]: ...

    def require_in_instance(self, feature_or_class):
        """A class decorator that requires the dependency is available when the class is
        initialized.  This decorator can be used even if the class does not define an ``__init__``
        method.

        Args:
            feature_or_class (str or Type): the name of the feature that requires these
                dependencies.  If this function is called directly as a decorator (for example
                ``@HAS_X.require_in_instance`` as opposed to
                ``@HAS_X.require_in_instance("my feature")``), then the feature name will be taken
                as the name of the class.

        Returns:
            Callable: a class decorator that ensures that the wrapped feature is present if the
            class is initialized.
        """
        if isinstance(feature_or_class, str):
            feature = feature_or_class

            def decorator(class_):
                wrap_method(class_, "__init__", before=_RequireNow(self, feature))
                return class_

            return decorator

        class_ = feature_or_class
        feature = (
            getattr(class_, "__qualname__", None)
            or getattr(class_, "__name__", None)
            or str(class_)
        )
        wrap_method(class_, "__init__", before=_RequireNow(self, feature))
        return class_

    def require_now(self, feature: str):
        """Eagerly attempt to import the dependencies in this object, and raise an exception if they
        cannot be imported.

        Args:
            feature: the name of the feature that is requiring these dependencies.

        Raises:
            MissingOptionalLibraryError: if the dependencies cannot be imported.
        """
        if self:
            return
        raise MissingOptionalLibraryError(
            libname=self._name, name=feature, pip_install=self._install, msg=self._msg
        )

    @contextlib.contextmanager
    def disable_locally(self):
        """
        Create a context, during which the value of the dependency manager will be ``False``.  This
        means that within the context, any calls to this object will behave as if the dependency is
        not available, including raising errors.  It is valid to call this method whether or not the
        dependency has already been evaluated.  This is most useful in tests.
        """
        previous = self._bool
        self._bool = False
        try:
            yield
        finally:
            self._bool = previous


class LazyImportTester(LazyDependencyManager):
    """A lazy dependency tester for importable Python modules.  Any required objects will only be
    imported at the point that this object is tested for its Boolean value."""

    __slots__ = ("_modules",)

    def __init__(
        self,
        name_map_or_modules: str | dict[str, Iterable[str]] | Iterable[str],
        *,
        name: str | None = None,
        callback: Callable[[bool], None] | None = None,
        install: str | None = None,
        msg: str | None = None,
    ):
        """
        Args:
            name_map_or_modules: if a name map, then a dictionary where the keys are modules or
                packages, and the values are iterables of names to try and import from that
                module.  It should be valid to write ``from <module> import <name1>, <name2>, ...``.
                If simply a string or iterable of strings, then it should be valid to write
                ``import <module>`` for each of them.
            name: the name of this optional dependency.
            callback: a callback that is called immediately after the availability of the library is
                tested with the result.  This will only be called once.
            install: how to install this optional dependency.  Passed to
                :class:`.MissingOptionalLibraryError` as the ``pip_install`` parameter.
            msg: an extra message to include in the error raised if this is required.

        Raises:
            ValueError: if no modules are given.
        """
        if isinstance(name_map_or_modules, dict):
            self._modules = {module: tuple(names) for module, names in name_map_or_modules.items()}
        elif isinstance(name_map_or_modules, str):
            self._modules = {name_map_or_modules: ()}
        else:
            self._modules = dict.fromkeys(name_map_or_modules, ())
        if not self._modules:
            raise ValueError("no modules supplied")
        if name is not None:
            pass
        elif len(self._modules) == 1:
            (name,) = self._modules.keys()
        else:
            all_names = tuple(self._modules.keys())
            name = f"{', '.join(all_names[:-1])} and {all_names[-1]}"
        super().__init__(name=name, callback=callback, install=install, msg=msg)

    def _is_available(self):
        failed_modules = {}
        failed_names = collections.defaultdict(list)
        for module, names in self._modules.items():
            try:
                imported = importlib.import_module(module)
            except ModuleNotFoundError as exc:
                failed_parts = exc.name.split(".")
                target_parts = module.split(".")
                if failed_parts == target_parts[: len(failed_parts)]:
                    # If the module that wasn't found is the one we were explicitly searching for
                    # (or one of its parents), then it's just not installed.
                    return False
                # Otherwise, we _did_ find the module, it just didn't import, which is a problem.
                failed_modules[module] = exc
                continue
            except ImportError as exc:
                failed_modules[module] = exc
                continue
            for name in names:
                try:
                    _ = getattr(imported, name)
                except AttributeError:
                    failed_names[module].append(name)
        if failed_modules or failed_names:
            package_description = f"'{self._name}'" if self._name else "optional packages"
            message = (
                f"While trying to import {package_description},"
                " some components were located but raised other errors during import."
                " You might have an incompatible version installed."
                " Qiskit will continue as if the optional is not available."
            )
            for module, exc in failed_modules.items():
                message += "".join(f"\n - module '{module}' failed to import with: {exc!r}")
            for module, names in failed_names.items():
                attributes = f"attribute '{names[0]}'" if len(names) == 1 else f"attributes {names}"
                message += "".join(f"\n - '{module}' imported, but {attributes} couldn't be found")
            warnings.warn(message, category=OptionalDependencyImportWarning)
            return False
        return True


class LazySubprocessTester(LazyDependencyManager):
    """A lazy checker that a command-line tool is available.  The command will only be run once, at
    the point that this object is checked for its Boolean value.
    """

    __slots__ = ("_command",)

    def __init__(
        self,
        command: str | Iterable[str],
        *,
        name: str | None = None,
        callback: Callable[[bool], None] | None = None,
        install: str | None = None,
        msg: str | None = None,
    ):
        """
        Args:
            command: the strings that make up the command to be run.  For example,
                ``["pdflatex", "-version"]``.
            name: the name of this optional dependency.
            callback: a callback that is called immediately after the availability of the library is
                tested with the result.  This will only be called once.
            install: how to install this optional dependency.  Passed to
                :class:`.MissingOptionalLibraryError` as the ``pip_install`` parameter.
            msg: an extra message to include in the error raised if this is required.

        Raises:
            ValueError: if an empty command is given.
        """
        self._command = (command,) if isinstance(command, str) else tuple(command)
        if not self._command:
            raise ValueError("no command supplied")
        super().__init__(name=name or self._command[0], callback=callback, install=install, msg=msg)

    def _is_available(self):
        try:
            subprocess.run(  # noqa: S603
                self._command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError):
            return False
        else:
            return True
