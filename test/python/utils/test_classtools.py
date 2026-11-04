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

"""Tests for the methods in ``utils.classtools``."""

# If `utils.wrap_method` is horrendously broken, then the test suite probably won't even have been
# able to collect, because it's used in `QiskitTestCase`.
#
# Throughout this file, we use ``this`` as an argument name to refer to a general reference, which
# may be an instance or the type of the instance.  This is just for clarity when we are using the
# same function to wrap both instance and class methods.
#
# We use a lot of dummy classes in test cases, for which there is absolutely no point defining
# docstrings.


import unittest
import sys

from qiskit.utils import wrap_method
from test import QiskitTestCase


def call_first_argument_with(*args, **kwargs):
    """Create a function that calls its first and only argument with the signature given to this
    function."""

    def out(mock):
        return mock(*args, **kwargs)

    return out


def call_second_argument_with(*args, **kwargs):
    """Create a function that calls its second argument with the first, and the signature given to
    this function."""

    def out(this, mock):
        mock(this, *args, **kwargs)

    return out


class TestWrapMethod(QiskitTestCase):
    """Tests of ``utils.classtools.wrap_method``.  These can be rather tricky, because there's a lot
    of messing around with descriptors and some special cases necessary to support builtin Python
    functions, so we sometimes have rather funny access patterns."""

    def test_called_with(self):
        """Test the basic call patterns are correct.  We use regular Python functions rather than
        mocks to make the instances and callbacks in this simplest case, because the low-level
        descriptor use means that there might be side-effects to binding mocks directly."""

        class Dummy:
            def instance(self, mock):
                mock(self, "method")

            @classmethod
            def class_(cls, mock):
                mock(cls, "method")

            @staticmethod
            def static(mock):
                mock("method")

        wrap_method(
            Dummy,
            "instance",
            before=call_second_argument_with("before"),
            after=call_second_argument_with("after"),
        )
        wrap_method(
            Dummy,
            "class_",
            before=call_second_argument_with("before"),
            after=call_second_argument_with("after"),
        )
        wrap_method(
            Dummy,
            "static",
            before=call_first_argument_with("before"),
            after=call_first_argument_with("after"),
        )

        with self.subTest("from instance"):
            source = Dummy()
            with self.subTest("instance"):
                mock = unittest.mock.Mock()

                # Test that the before/after are not called when the method is accessed.
                caller = source.instance
                mock.assert_not_called()
                caller(mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call(source, "before"),
                        unittest.mock.call(source, "method"),
                        unittest.mock.call(source, "after"),
                    ],
                    any_order=False,
                )
            with self.subTest("class"):
                mock = unittest.mock.Mock()
                caller = source.class_
                mock.assert_not_called()
                caller(mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call(Dummy, "before"),
                        unittest.mock.call(Dummy, "method"),
                        unittest.mock.call(Dummy, "after"),
                    ],
                    any_order=False,
                )
            with self.subTest("static"):
                mock = unittest.mock.Mock()
                caller = source.static
                mock.assert_not_called()
                caller(mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call("before"),
                        unittest.mock.call("method"),
                        unittest.mock.call("after"),
                    ],
                    any_order=False,
                )

        with self.subTest("from type"):
            with self.subTest("instance"):
                mock = unittest.mock.Mock()

                # Test that the before/after are not called when the method is accessed.
                caller = Dummy.instance
                mock.assert_not_called()
                caller("this", mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call("this", "before"),
                        unittest.mock.call("this", "method"),
                        unittest.mock.call("this", "after"),
                    ],
                    any_order=False,
                )
            with self.subTest("class"):
                mock = unittest.mock.Mock()
                caller = Dummy.class_
                mock.assert_not_called()
                caller(mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call(Dummy, "before"),
                        unittest.mock.call(Dummy, "method"),
                        unittest.mock.call(Dummy, "after"),
                    ],
                    any_order=False,
                )
            with self.subTest("static"):
                mock = unittest.mock.Mock()
                caller = Dummy.static
                mock.assert_not_called()
                caller(mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call("before"),
                        unittest.mock.call("method"),
                        unittest.mock.call("after"),
                    ],
                    any_order=False,
                )

    def test_can_wrap_with_lambda(self):
        """Test that lambda functions can be used as the callbacks."""

        class Dummy:
            def instance(self, mock):
                mock(self, "method")

        wrap_method(Dummy, "instance", after=lambda self, mock: mock(self, "after"))
        with self.subTest("from instance"):
            mock = unittest.mock.Mock()
            source = Dummy()
            caller = source.instance
            mock.assert_not_called()
            caller(mock)
            mock.assert_has_calls(
                [unittest.mock.call(source, "method"), unittest.mock.call(source, "after")],
                any_order=False,
            )

        with self.subTest("from type"):
            mock = unittest.mock.Mock()
            caller = Dummy.instance
            mock.assert_not_called()
            caller("this", mock)
            mock.assert_has_calls(
                [unittest.mock.call("this", "method"), unittest.mock.call("this", "after")],
                any_order=False,
            )

    def test_can_wrap_with_callable_class(self):
        """Test that a class with a ``__call__`` but no descriptor protocol can be used as the
        callbacks."""

        class Dummy:
            def instance(self, mock):
                mock(self, "method")

        class Callback:
            # Note that this class does not implement the descriptor protocol, unlike normal Python
            # functions.  ``__call__`` itself implements ``__get__``, but that just bounds the
            # ``Callback`` instance to ``self``.
            def __call__(self, this, mock):
                mock(this, "after")

        wrap_method(Dummy, "instance", after=Callback())
        with self.subTest("from instance"):
            mock = unittest.mock.Mock()
            source = Dummy()
            caller = source.instance
            mock.assert_not_called()
            caller(mock)
            mock.assert_has_calls(
                [unittest.mock.call(source, "method"), unittest.mock.call(source, "after")],
                any_order=False,
            )

        with self.subTest("from type"):
            mock = unittest.mock.Mock()
            caller = Dummy.instance
            mock.assert_not_called()
            caller("this", mock)
            mock.assert_has_calls(
                [unittest.mock.call("this", "method"), unittest.mock.call("this", "after")],
                any_order=False,
            )

    def test_can_wrap_with_builtin(self):
        """Test that builtin functions can be used a callback.  Many CPython builtins don't
        implement the descriptor protocol that all functions defined with ``def`` or ``lambda`` do,
        which means we need to take special care that they work.  This is most relevant for
        C-extension functions created via pybind11, Cython or similar, rather than actual Python
        builtins."""

        class Dummy:
            def instance(self, mock):
                mock(self, "method")

        # We don't want to add additional compilation requirements for one simple test in the suite,
        # so instead we need a CPython builtin function (of type ``types.BuiltinFunctionType``) that
        # has side effects when being called with an arbitrary object as the first positional
        # argument.  ``breakpoint`` is a good choice, because it's designed to support dependency
        # injection with arbitrary arguments; it calls out to the overridable ``sys.breakpointhook``
        # with all its arguments.  We can't use ``eval`` to test the bound-method calls because the
        # first argument would be the instance, not the string with a side-effect-y programme.
        with unittest.mock.patch.object(sys, "breakpointhook") as mock:
            wrap_method(Dummy, "instance", before=breakpoint)

            with self.subTest("from instance"):
                mock.reset_mock()
                source = Dummy()
                source.instance(mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call(source, mock),
                        unittest.mock.call(source, "method"),
                    ]
                )

            with self.subTest("from type"):
                mock.reset_mock()
                Dummy.instance("this", mock)
                mock.assert_has_calls(
                    [
                        unittest.mock.call("this", mock),
                        unittest.mock.call("this", "method"),
                    ]
                )

    def test_can_wrap_with_mock(self):
        """This is kind of a meta test, to check that we can use a ``unittest.mock.Mock`` instance
        as the callback."""

        class Dummy:
            def instance(self, x):
                pass

        mock = unittest.mock.Mock()
        wrap_method(Dummy, "instance", before=mock)

        with self.subTest("from instance"):
            mock.reset_mock()
            source = Dummy()
            source.instance("hello, world")
            mock.assert_called_once_with(source, "hello, world")
        with self.subTest("from type"):
            mock.reset_mock()
            Dummy.instance("this", "hello, world")
            mock.assert_called_once_with("this", "hello, world")

    def test_wrapping_inherited_method(self):
        """Test that ``wrap_method`` will correctly find a method defined only on a parent class."""

        class Parent:
            def instance(self, mock):
                mock(self, "method")

        class Child(Parent):
            pass

        wrap_method(Child, "instance", before=call_second_argument_with("before"))

        with self.subTest("from instance"):
            mock = unittest.mock.Mock()
            source = Child()
            caller = source.instance
            mock.assert_not_called()
            caller(mock)
            mock.assert_has_calls(
                [unittest.mock.call(source, "before"), unittest.mock.call(source, "method")],
                any_order=False,
            )

        with self.subTest("from type"):
            mock = unittest.mock.Mock()
            caller = Child.instance
            mock.assert_not_called()
            caller("this", mock)
            mock.assert_has_calls(
                [unittest.mock.call("this", "before"), unittest.mock.call("this", "method")],
                any_order=False,
            )

    def test_wrapping___init__(self):
        """Test that wrapping the magic __init__ method works."""

        class Dummy:
            def __init__(self, mock):
                mock("__init__")
                self.mock = mock

        def add_extra_property(self, _):
            mock("extra")
            self.extra = "hello, world"

        wrap_method(Dummy, "__init__", after=add_extra_property)

        mock = unittest.mock.Mock()
        dummy = Dummy(mock)
        self.assertIs(dummy.mock, mock)
        self.assertEqual(dummy.extra, "hello, world")
        mock.assert_has_calls(
            [unittest.mock.call("__init__"), unittest.mock.call("extra")],
            any_order=False,
        )

    def test_wrapping___add__(self):
        """Test that wrapping an arithmetic operator works.  There is nothing particularly special
        about ``__add__`` that (say) ``__init__`` doesn't also do, but this is just a further check
        that the magic methods can work.  Note that ``__add__`` must be defined on the type; all
        magic methods ignore re-definitions in instance dictionaries."""

        mock = unittest.mock.Mock()

        class Dummy:
            def __init__(self, n):
                self.n = n

            def __add__(self, other):
                return type(self)(self.n + other.n)

        wrap_method(Dummy, "__add__", before=mock)

        left = Dummy(1)
        right = Dummy(2)
        out = left + right
        self.assertIsInstance(out, Dummy)
        self.assertEqual(out.n, 3)
        mock.assert_has_calls([unittest.mock.call(left, right)])

    def test_wrapping___new__(self):
        """Test that wrapping the magic __new__ method works.  This method is implicitly made into a
        static method with no decorator, but still gets called with the class in the first
        position.  Note that ``type`` implements ``__new__``, so the getter needs to ensure that it
        doesn't accidentally"""

        class Dummy:
            def __new__(cls, mock):
                mock(cls, "__new__")
                return super().__new__(cls)

        wrap_method(Dummy, "__new__", before=call_second_argument_with("extra"))

        mock = unittest.mock.Mock()
        Dummy(mock)
        mock.assert_has_calls(
            [unittest.mock.call(Dummy, "extra"), unittest.mock.call(Dummy, "__new__")],
            any_order=False,
        )

    def test_wrapping_object___new__(self):
        """Test that wrapping the magic __new__ method works when it is inherited.  This is a very
        special case, because by inheritance ``A.__new__`` is ``type.__new__``, but that's not we
        want to wrap; we need to have used ``type.__getattribute__`` to make sure that we're getting
        the default implementation ``object.__new__``, and not the ``__new__`` method that literally
        constructs new types.
        """

        class Dummy:
            pass

        mock = unittest.mock.Mock()
        wrap_method(Dummy, "__new__", before=mock)
        Dummy()
        mock.assert_called_once_with(Dummy)

    def test_wrapping_object___eq__(self):
        """Test that wrapping equality works.  ``type`` also implements ``__eq__`` in a way that
        returns ``NotImplemented`` if one of the operands is not a ``type``, so this tests that we
        are successfully finding ``object.__eq__`` in the resolution of the wrapped method."""

        class Dummy:
            def __init__(self, n):
                self.n = n

            def __eq__(self, other):
                return self.n == other.n

        mock = unittest.mock.Mock()
        wrap_method(Dummy, "__eq__", before=mock)

        left = Dummy(1)
        right = Dummy(2)
        self.assertNotEqual(left, right)
        mock.assert_has_calls([unittest.mock.call(left, right)])

    def test_wrapping_object___init_subclass__(self):
        """Test that wrapping the magic ``__init_subclass__`` method works.  This method is
        implicitly made into a class method without needing a decorator."""

        class Dummy:
            pass

        mock = unittest.mock.Mock()
        wrap_method(Dummy, "__init_subclass__", before=mock)

        class Child(Dummy):
            pass

        mock.assert_called_once_with(Child)

    def test_wrapping_already_wrapped_method(self):
        """Test that a chain of wrapped methods evaluate correctly, in the right order.  Methods
        that are explicitly overridden in child class definitions do not create chains of wrapped
        methods in the same way, even if they call ``super().method`` because the actual object in
        the child definition would be a regular function.  This tests the case that method we're
        wrapping is the exact output of a previous wrapping."""

        class Grandparent:
            def method(self, mock):
                mock(self, "grandparent")

        wrap_method(
            Grandparent,
            "method",
            before=call_second_argument_with("before 1"),
            after=call_second_argument_with("after 1"),
        )

        class Parent(Grandparent):
            pass

        wrap_method(
            Parent,
            "method",
            before=call_second_argument_with("before 2"),
            after=call_second_argument_with("after 2"),
        )

        class Child(Parent):
            pass

        wrap_method(
            Child,
            "method",
            before=call_second_argument_with("before 3"),
            after=call_second_argument_with("after 3"),
        )

        mock = unittest.mock.Mock()
        child = Child()
        child.method(mock)
        mock.assert_has_calls(
            [
                unittest.mock.call(child, "before 3"),
                unittest.mock.call(child, "before 2"),
                unittest.mock.call(child, "before 1"),
                unittest.mock.call(child, "grandparent"),
                unittest.mock.call(child, "after 1"),
                unittest.mock.call(child, "after 2"),
                unittest.mock.call(child, "after 3"),
            ],
            any_order=False,
        )

    def test_docstring_inherited(self):
        """Test that the docstring of a method is correctly passed through, to avoid clobbering
        documentation."""

        class Dummy:
            def method(self):
                """This is documentation."""

        wrap_method(Dummy, "method", before=lambda self: None)
        self.assertEqual(Dummy.method.__doc__, "This is documentation.")

    def test_raises_on_invalid_name(self):
        """Test that a suitable error is raised if the method doesn't exist."""

        class Dummy:
            pass

        with self.assertRaisesRegex(AttributeError, "bad"):
            wrap_method(Dummy, "bad", before=lambda self: None)
