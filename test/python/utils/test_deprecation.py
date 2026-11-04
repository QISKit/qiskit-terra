# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the functions in ``utils.deprecation``."""

from __future__ import annotations

import sys
from textwrap import dedent

from qiskit.utils.deprecation import (
    add_deprecation_to_docstring,
    deprecate_arg,
    deprecate_func,
)
from test import QiskitTestCase


@deprecate_func(
    since="9.99",
    additional_msg="Instead, use new_func().",
    removal_timeline="in 2 releases",
)
def _deprecated_func():
    pass


class _Foo:
    @deprecate_func(since="9.99", pending=True)
    def __init__(self):
        super().__init__()

    @property
    @deprecate_func(since="9.99", is_property=True)
    def my_property(self):
        """Property."""
        return 0

    @my_property.setter
    @deprecate_func(since="9.99")
    def my_property(self, value):
        pass

    @deprecate_func(since="9.99", additional_msg="Stop using this!")
    def my_method(self):
        """Method."""

    def normal_method(self):
        """Method."""


class TestDeprecationDecorators(QiskitTestCase):
    """Test that the decorators in ``utils.deprecation`` correctly log warnings and get added to
    docstring."""

    def test_deprecate_func_docstring(self) -> None:
        """Test that `@deprecate_func` adds the correct message to the docstring."""

        self.assertEqual(
            _deprecated_func.__doc__,
            dedent(
                f"""\

                .. deprecated:: 9.99
                  The function ``{__name__}._deprecated_func()`` is deprecated as of Qiskit \
9.99. It will be removed in 2 releases. Instead, use new_func().
                """
            ),
        )
        self.assertEqual(
            _Foo.__init__.__doc__,
            dedent(
                f"""\

                .. deprecated:: 9.99_pending
                  The class ``{__name__}._Foo`` is pending deprecation as of Qiskit 9.99. It \
will be marked deprecated in a future release, and then removed no earlier than 3 months after \
the release date.
                """
            ),
        )
        self.assertEqual(
            _Foo.my_method.__doc__,
            dedent(
                f"""\
                Method.

                .. deprecated:: 9.99
                  The method ``{__name__}._Foo.my_method()`` is deprecated as of Qiskit \
9.99. It will be removed no earlier than 3 months after the release date. Stop using this!
                """
            ),
        )
        self.assertEqual(
            _Foo.my_property.__doc__,
            dedent(
                f"""\
                Property.

                .. deprecated:: 9.99
                  The property ``{__name__}._Foo.my_property`` is deprecated as of Qiskit \
9.99. It will be removed no earlier than 3 months after the release date.
                """
            ),
        )

    def test_deprecate_func_package_name(self) -> None:
        """Test setting the package name works."""

        @deprecate_func(since="9.99", package_name="riskit")
        def my_func() -> None:
            pass

        with self.assertWarnsRegex(DeprecationWarning, "is deprecated as of riskit 9.99"):
            my_func()

    def test_deprecate_arg_docstring(self) -> None:
        """Test that `@deprecate_arg` adds the correct message to the docstring."""

        @deprecate_arg("arg1", since="9.99", removal_timeline="in 2 releases")
        @deprecate_arg("arg2", pending=True, since="9.99")
        @deprecate_arg(
            "arg3",
            since="9.99",
            deprecation_description="Using the argument arg3",
            new_alias="new_arg3",
        )
        @deprecate_arg(
            "arg4",
            since="9.99",
            additional_msg="Instead, use foo.",
            # This predicate always fails, but it should not impact storing the deprecation
            # metadata. That ensures the deprecation still shows up in our docs.
            predicate=lambda arg4: False,
        )
        def my_func() -> None:
            pass

        self.assertEqual(
            my_func.__doc__,
            dedent(
                f"""\

                .. deprecated:: 9.99
                  ``{__name__}.{my_func.__qualname__}()``'s argument ``arg4`` is deprecated as of \
Qiskit 9.99. It will be removed no earlier than 3 months after the release date. Instead, \
use foo.

                .. deprecated:: 9.99
                  Using the argument arg3 is deprecated as of Qiskit 9.99. It will be \
removed no earlier than 3 months after the release date. Instead, use the argument ``new_arg3``, \
which behaves identically.

                .. deprecated:: 9.99_pending
                  ``{__name__}.{my_func.__qualname__}()``'s argument ``arg2`` is pending \
deprecation as of Qiskit 9.99. It will be marked deprecated in a future release, and then \
removed no earlier than 3 months after the release date.

                .. deprecated:: 9.99
                  ``{__name__}.{my_func.__qualname__}()``'s argument ``arg1`` is deprecated as of \
Qiskit 9.99. It will be removed in 2 releases.
                """
            ),
        )

    def test_deprecate_func_runtime_warning(self) -> None:
        """Test that `@deprecate_func` warns whenever the function is used."""

        with self.assertWarns(DeprecationWarning):
            _deprecated_func()
        with self.assertWarns(PendingDeprecationWarning):
            instance = _Foo()
        with self.assertWarns(DeprecationWarning):
            instance.my_method()
        with self.assertWarns(DeprecationWarning):
            _ = instance.my_property
        with self.assertWarns(DeprecationWarning):
            instance.my_property = 1
        instance.normal_method()

    def test_deprecate_arg_runtime_warning(self) -> None:
        """Test that `@deprecate_arg` warns whenever the arguments are used.

        Also check these edge cases:
        * If `new_alias` is set, pass the old argument as the new alias.
        * If `predicate` is set, only warn if the predicate is True.
        """

        @deprecate_arg("arg1", new_alias="new_arg1", since="9.99")
        @deprecate_arg("arg2", since="9.99")
        @deprecate_arg("arg3", predicate=lambda arg3: arg3 == "deprecated value", since="9.99")
        def my_func(
            arg1: str = "a",
            arg2: str = "a",
            arg3: str = "a",
            new_arg1: str | None = None,
        ) -> None:
            del arg2
            del arg3
            # If the old arg was set, we should set its `new_alias` to that value.
            if arg1 != "a":
                self.assertEqual(new_arg1, "z")
            if new_arg1 is not None:
                self.assertEqual(new_arg1, "z")

        # No warnings if no deprecated args used.
        my_func()
        my_func(new_arg1="z")

        # Warn if argument is specified, regardless of positional vs kwarg.
        with self.assertWarnsRegex(DeprecationWarning, "arg1"):
            my_func("z")
        with self.assertWarnsRegex(DeprecationWarning, "arg1"):
            my_func(arg1="z")
        with self.assertWarnsRegex(DeprecationWarning, "arg2"):
            my_func("z", "z")
        with self.assertWarnsRegex(DeprecationWarning, "arg2"):
            my_func(arg2="z")

        # Error if new_alias specified at the same time as old argument name.
        with self.assertRaises(TypeError):
            my_func("a", new_arg1="z")
        with self.assertRaises(TypeError):
            my_func(arg1="a", new_arg1="z")
        with self.assertRaises(TypeError):
            my_func("a", "a", "a", "z")

        # Test the `predicate` functionality.
        my_func(arg3="okay value")
        with self.assertWarnsRegex(DeprecationWarning, "arg3"):
            my_func(arg3="deprecated value")
        with self.assertWarnsRegex(DeprecationWarning, "arg3"):
            my_func("z", "z", "deprecated value")

    def test_deprecate_arg_runtime_warning_variadic_args(self) -> None:
        """Test that `@deprecate_arg` errors when defined on a function using `*args` but can
        handle `**kwargs`.

        We know how to support *args robustly via `inspect.signature().bind()`, but it has worse
        performance (see the commit history of PR #9884). Given how infrequent we expect
        deprecations to involve *args, we simply error.
        """

        with self.assertRaises(ValueError):

            @deprecate_arg("my_kwarg", since="9.99")
            def my_func1(*args: int, my_kwarg: int | None = None) -> None:
                del args
                del my_kwarg

        @deprecate_arg("my_kwarg", since="9.99")
        def my_func2(my_kwarg: int | None = None, **kwargs) -> None:
            # We assign an arbitrary variable `c` because it will be included in
            # `my_func.__code__.co_varnames` and we want to make sure that does not break the
            # implementation.
            c = 0  # noqa: F841
            del kwargs
            del my_kwarg

        # No warnings if no deprecated args used.
        my_func2()
        my_func2(another_arg=0, yet_another=0)

        # Warn if argument is specified.
        with self.assertWarnsRegex(DeprecationWarning, "my_kwarg"):
            my_func2(my_kwarg=5)
        with self.assertWarnsRegex(DeprecationWarning, "my_kwarg"):
            my_func2(my_kwarg=5, another_arg=0, yet_another=0)


class AddDeprecationDocstringTest(QiskitTestCase):
    """Test that we correctly insert the deprecation directive at the right location.

    When determining the ``expected`` output, manually modify the docstring of a function
    (in any Qiskit repo) to have the same structure. Then, build the docs to make sure that it
    renders correctly.
    """

    def test_add_deprecation_docstring_no_meta_lines(self) -> None:
        """When no metadata lines like Args, the directive should be added to the end."""

        def func1():
            pass

        add_deprecation_to_docstring(func1, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func1.__doc__,
            dedent(
                """\

                .. deprecated:: 9.99
                  Deprecated!
                """
            ),
        )

        def func2():
            """Docstring."""

        add_deprecation_to_docstring(func2, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func2.__doc__,
            dedent(
                """\
                Docstring.

                .. deprecated:: 9.99
                  Deprecated!
                """
            ),
        )

        indent = "            "

        def func3():
            """Docstring extending
            to a new line."""

        add_deprecation_to_docstring(func3, msg="Deprecated!", since="9.99", pending=False)
        expected_doc = f"""Docstring extending
            to a new line.
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""
        if sys.version_info >= (3, 13, 0):
            expected_doc = """Docstring extending
to a new line.

.. deprecated:: 9.99
  Deprecated!
"""

        self.assertEqual(func3.__doc__, expected_doc)

        def func4():
            """
            Docstring starting on a new line.
            """

        add_deprecation_to_docstring(func4, msg="Deprecated!", since="9.99", pending=False)
        expected_doc = f"""\

            Docstring starting on a new line.
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""
        if sys.version_info >= (3, 13, 0):
            expected_doc = """\

Docstring starting on a new line.

.. deprecated:: 9.99
  Deprecated!
"""

        self.assertEqual(
            func4.__doc__,
            expected_doc,
        )

        def func5():
            """
            Paragraph 1, line 1.
            Line 2.

            Paragraph 2.

            """

        expected_doc = f"""\

            Paragraph 1, line 1.
            Line 2.

            Paragraph 2.

{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""

        if sys.version_info >= (3, 13, 0):
            expected_doc = """\

Paragraph 1, line 1.
Line 2.

Paragraph 2.


.. deprecated:: 9.99
  Deprecated!
"""

        add_deprecation_to_docstring(func5, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func5.__doc__,
            expected_doc,
        )

        def func6():
            """Blah.

            A list.
              * element 1
              * element 2
                continued
            """

        expected_doc = f"""Blah.

            A list.
              * element 1
              * element 2
                continued
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}"""

        if sys.version_info >= (3, 13, 0):
            expected_doc = """Blah.

A list.
  * element 1
  * element 2
    continued

.. deprecated:: 9.99
  Deprecated!
"""

        add_deprecation_to_docstring(func6, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func6.__doc__,
            expected_doc,
        )

    def test_add_deprecation_docstring_meta_lines(self) -> None:
        """When there are metadata lines like Args, the directive should be inserted in-between the
        summary and those lines."""
        indent = "            "

        def func1():
            """
            Returns:
                Content.

            Raises:
                SomeError
            """

        add_deprecation_to_docstring(func1, msg="Deprecated!", since="9.99", pending=False)
        expected_doc = f"""\
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}

            Returns:
                Content.

            Raises:
                SomeError
{indent}"""
        if sys.version_info >= (3, 13, 0):
            expected_doc = """\

.. deprecated:: 9.99
  Deprecated!


Returns:
    Content.

Raises:
    SomeError"""

        self.assertEqual(
            func1.__doc__,
            expected_doc,
        )

        def func2():
            """Docstring.

            Returns:
                Content.
            """

        add_deprecation_to_docstring(func2, msg="Deprecated!", since="9.99", pending=False)
        expected_doc = f"""Docstring.
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}

            Returns:
                Content.
{indent}"""
        if sys.version_info >= (3, 13, 0):
            expected_doc = """Docstring.

.. deprecated:: 9.99
  Deprecated!


Returns:
    Content."""

        self.assertEqual(func2.__doc__, expected_doc)

        def func3():
            """
            Docstring starting on a new line.

            Paragraph 2.

            Examples:
                Content.
            """

        expected_doc = f"""\

            Docstring starting on a new line.

            Paragraph 2.
{indent}
            .. deprecated:: 9.99
              Deprecated!
{indent}

            Examples:
                Content.
{indent}"""
        if sys.version_info >= (3, 13, 0):
            expected_doc = """\

Docstring starting on a new line.

Paragraph 2.

.. deprecated:: 9.99
  Deprecated!


Examples:
    Content."""

        add_deprecation_to_docstring(func3, msg="Deprecated!", since="9.99", pending=False)
        self.assertEqual(
            func3.__doc__,
            expected_doc,
        )

    def test_add_deprecation_docstring_multiple_entries(self) -> None:
        """Multiple entries are appended correctly."""

        def func1():
            pass

        add_deprecation_to_docstring(func1, msg="Deprecated #1!", since="9.99", pending=False)
        add_deprecation_to_docstring(func1, msg="Deprecated #2!", since="9.99", pending=False)
        self.assertEqual(
            func1.__doc__,
            dedent(
                """\

                .. deprecated:: 9.99
                  Deprecated #1!

                .. deprecated:: 9.99
                  Deprecated #2!
                """
            ),
        )

        indent = "            "

        def func2():
            """
            Docstring starting on a new line.
            """

        add_deprecation_to_docstring(func2, msg="Deprecated #1!", since="9.99", pending=False)
        add_deprecation_to_docstring(func2, msg="Deprecated #2!", since="9.99", pending=False)
        expected_doc = f"""\

            Docstring starting on a new line.
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated #1!
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated #2!
{indent}"""
        if sys.version_info >= (3, 13, 0):
            expected_doc = """\

Docstring starting on a new line.

.. deprecated:: 9.99
  Deprecated #1!

.. deprecated:: 9.99
  Deprecated #2!
"""

        self.assertEqual(
            func2.__doc__,
            expected_doc,
        )

        def func3():
            """Docstring.

            Yields:
                Content.
            """

        expected_doc = f"""Docstring.
{indent}
            .. deprecated:: 9.99
              Deprecated #1!
{indent}
{indent}
            .. deprecated:: 9.99
              Deprecated #2!
{indent}

            Yields:
                Content.
{indent}"""

        if sys.version_info >= (3, 13, 0):
            expected_doc = """Docstring.

.. deprecated:: 9.99
  Deprecated #1!


.. deprecated:: 9.99
  Deprecated #2!


Yields:
    Content."""

        add_deprecation_to_docstring(func3, msg="Deprecated #1!", since="9.99", pending=False)
        add_deprecation_to_docstring(func3, msg="Deprecated #2!", since="9.99", pending=False)
        self.assertEqual(
            func3.__doc__,
            expected_doc,
        )

    def test_add_deprecation_docstring_pending(self) -> None:
        """The version string should end in `_pending` when pending."""

        def func():
            pass

        add_deprecation_to_docstring(func, msg="Deprecated!", since="9.99", pending=True)
        self.assertEqual(
            func.__doc__,
            dedent(
                """\

                .. deprecated:: 9.99_pending
                  Deprecated!
                """
            ),
        )

    def test_add_deprecation_docstring_since_not_set(self) -> None:
        """The version string should be `unknown` when ``None``."""

        def func():
            pass

        add_deprecation_to_docstring(func, msg="Deprecated!", since=None, pending=False)
        self.assertEqual(
            func.__doc__,
            dedent(
                """\

                .. deprecated:: unknown
                  Deprecated!
                """
            ),
        )

    def test_add_deprecation_docstring_newline_msg_banned(self) -> None:
        """Test that `\n` is banned in the deprecation message, as it breaks Sphinx rendering."""

        def func():
            pass

        with self.assertRaises(ValueError):
            add_deprecation_to_docstring(func, msg="line1\nline2", since="9.99", pending=False)

    def test_add_deprecation_docstring_initial_metadata_line_banned(self) -> None:
        """Test that the docstring cannot start with e.g. `Args:`."""

        def func():
            """Args:
            Foo.
            """

        with self.assertRaises(ValueError):
            add_deprecation_to_docstring(func, msg="Deprecated!", since="9.99", pending=False)
