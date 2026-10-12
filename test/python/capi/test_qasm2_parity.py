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

"""Replay the ``test/python/qasm2`` suite against the C API's native OpenQASM 2 importer.

Each ``TestCase`` in the modules named below is dynamically subclassed with
``qiskit.qasm2.loads`` monkeypatched to go through the C API; known gaps become skips rather
than failures.
"""

import functools
import importlib
import inspect
import unittest
from unittest import mock

import qiskit.qasm2
import qiskit.qpy.exceptions

from . import ffi

# Exception type + message substring for each known, pre-existing gap between the two importers.
# A pattern that stops matching anything just means those tests now pass for real; remove it then.
_KNOWN_GAPS = (
    (NotImplementedError, "Custom operations from Rust cannot be exposed to Python"),
    (NotImplementedError, "Custom operations from Rust cannot be checked by instance"),
    (qiskit.qpy.exceptions.QpyError, "compiled custom operations are not yet supported in QPY"),
)
_KNOWN_GAP_EXCEPTION_TYPES = tuple({exc_type for exc_type, _ in _KNOWN_GAPS})

# Doesn't raise, so can't be caught like _KNOWN_GAPS exceptions.
_KNOWN_DIVERGENT_TESTS = {
    "test.python.qasm2.test_structure.TestInclude.test_qelib1_include": (
        "the `id` gate maps to UGate(0, 0, 0) in Python but IGate via the C API"
    ),
}

# test_arxiv_examples uses @ddt.data binding at import time (can't monkeypatch after).
# test_circuit_methods never calls loads/load.
_MODULE_NAMES = (
    "test.python.qasm2.test_structure",
    "test.python.qasm2.test_parse_errors",
    "test.python.qasm2.test_expression",
    "test.python.qasm2.test_lexer",
    "test.python.qasm2.test_export",
)


def _loads_via_c(
    string,
    *,
    include_path=(".",),
    custom_instructions=(),
    custom_classical=(),
    strict=False,
):
    """Drop-in replacement for ``qiskit.qasm2.loads`` that goes through the C API importer."""
    if custom_instructions or custom_classical:
        raise unittest.SkipTest(
            "C API importer does not support custom instructions or custom classical functions"
        )
    return ffi.load_qasm2_from_c(string, include_path=include_path, strict=strict)


def _skip_known_gaps(test_method, qualified_name):
    """Run `test_method` against the C API importer, turning a known gap into a skip.

    Conversion from the Rust circuit to Python happens lazily, per instruction, so a known
    exception can surface anywhere in the test body, not just inside ``loads`` -- hence wrapping
    the whole method rather than just the load call.
    """

    @functools.wraps(test_method)
    def wrapper(self, *args, **kwargs):
        if qualified_name in _KNOWN_DIVERGENT_TESTS:
            raise unittest.SkipTest(
                f"known behavioural divergence: {_KNOWN_DIVERGENT_TESTS[qualified_name]}"
            )
        with mock.patch("qiskit.qasm2.loads", _loads_via_c):
            try:
                return test_method(self, *args, **kwargs)
            except _KNOWN_GAP_EXCEPTION_TYPES as exc:
                if any(
                    isinstance(exc, exc_type) and message in str(exc)
                    for exc_type, message in _KNOWN_GAPS
                ):
                    raise unittest.SkipTest(f"known C-API gap: {exc}") from None
                raise

    return wrapper


def _parity_subclass(name, source_class):
    """Build a `source_class` subclass whose test methods run against the C API importer."""
    namespace = {}
    for attr_name, member in vars(source_class).items():
        if attr_name.startswith("test_") and callable(member):
            qualified_name = f"{source_class.__module__}.{source_class.__qualname__}.{attr_name}"
            namespace[attr_name] = _skip_known_gaps(member, qualified_name)
    return type(name, (source_class,), namespace)


for _mod_name in _MODULE_NAMES:
    _mod = importlib.import_module(_mod_name)
    _mod_tag = _mod_name.rsplit(".", maxsplit=1)[-1]
    for _name, _obj in inspect.getmembers(_mod, inspect.isclass):
        if issubclass(_obj, unittest.TestCase) and _obj.__module__ == _mod.__name__:
            _new_name = f"TestCApiParity_{_mod_tag}_{_obj.__name__}"
            globals()[_new_name] = _parity_subclass(_new_name, _obj)

del _mod_name, _mod, _mod_tag, _name, _obj, _new_name
