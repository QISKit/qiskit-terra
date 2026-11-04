# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import copy
import pickle

from qiskit.providers import Options
from qiskit.result.models import MeasLevel
from test import QiskitTestCase


class TestOptions(QiskitTestCase):
    def test_no_validators(self):
        options = Options(shots=1024, method="auto", meas_level=MeasLevel.KERNELED)
        self.assertEqual(options.shots, 1024)
        options.update_options(method="statevector")
        self.assertEqual(options.method, "statevector")

    def test_no_validators_str(self):
        options = Options(shots=1024, method="auto", meas_level=MeasLevel.KERNELED)
        self.assertEqual(
            str(options), "Options(shots=1024, method='auto', meas_level=<MeasLevel.KERNELED: 1>)"
        )

    def test_range_bound_validator(self):
        options = Options(shots=1024)
        options.set_validator("shots", (1, 4096))
        with self.assertRaises(ValueError):
            options.update_options(shots=8192)

    def test_range_bound_string(self):
        options = Options(shots=1024)
        options.set_validator("shots", (1, 1024))
        expected = """Options(shots=1024)
Where:
\tshots is >= 1 and <= 1024\n"""
        self.assertEqual(str(options), expected)

    def test_list_choice(self):
        options = Options(method="auto")
        options.set_validator("method", ["auto", "statevector", "mps"])
        with self.assertRaises(ValueError):
            options.update_options(method="stabilizer")
        options.update_options(method="mps")
        self.assertEqual(options.method, "mps")

    def test_list_choice_string(self):
        options = Options(method="auto")
        options.set_validator("method", ["auto", "statevector", "mps"])
        expected = """Options(method='auto')
Where:
\tmethod is one of ['auto', 'statevector', 'mps']\n"""
        self.assertEqual(str(options), expected)

    def test_type_validator(self):
        options = Options(meas_level=MeasLevel.KERNELED)
        options.set_validator("meas_level", MeasLevel)
        with self.assertRaises(TypeError):
            options.update_options(meas_level=2)
        options.update_options(meas_level=MeasLevel.CLASSIFIED)
        self.assertEqual(2, options.meas_level.value)

    def test_type_validator_str(self):
        options = Options(meas_level=MeasLevel.KERNELED)
        options.set_validator("meas_level", MeasLevel)
        expected = """Options(meas_level=<MeasLevel.KERNELED: 1>)
Where:
\tmeas_level is of type <enum 'MeasLevel'>\n"""
        self.assertEqual(str(options), expected)

    def test_range_bound_validator_multiple_fields(self):
        options = Options(shots=1024, method="auto", meas_level=MeasLevel.KERNELED)
        options.set_validator("shots", (1, 1024))
        options.set_validator("method", ["auto", "statevector", "mps"])
        options.set_validator("meas_level", MeasLevel)
        with self.assertRaises(ValueError):
            options.update_options(shots=2048, method="statevector")
        options.update_options(shots=512, method="statevector")
        self.assertEqual(options.shots, 512)
        self.assertEqual(options.method, "statevector")

    def test_range_bound_validator_multiple_fields_string(self):
        options = Options(shots=1024, method="auto", meas_level=MeasLevel.KERNELED)
        options.set_validator("shots", (1, 1024))
        options.set_validator("method", ["auto", "statevector", "mps"])
        options.set_validator("meas_level", MeasLevel)
        expected = """Options(shots=1024, method='auto', meas_level=<MeasLevel.KERNELED: 1>)
Where:
\tshots is >= 1 and <= 1024
\tmethod is one of ['auto', 'statevector', 'mps']
\tmeas_level is of type <enum 'MeasLevel'>\n"""
        self.assertEqual(str(options), expected)

    def test_hasattr(self):
        options = Options(shots=1024)
        self.assertTrue(hasattr(options, "shots"))
        self.assertFalse(hasattr(options, "method"))

    def test_copy(self):
        options = Options(opt1=1, opt2=2)
        cpy = copy.copy(options)
        cpy.update_options(opt1=10, opt3=20)
        self.assertEqual(options.opt1, 1)
        self.assertEqual(options.opt2, 2)
        self.assertNotIn("opt3", options)
        self.assertEqual(cpy.opt1, 10)
        self.assertEqual(cpy.opt2, 2)
        self.assertEqual(cpy.opt3, 20)

    def test_iterate(self):
        options = Options(opt1=1, opt2=2, opt3="abc")
        options_dict = dict(options)

        self.assertEqual(options_dict, {"opt1": 1, "opt2": 2, "opt3": "abc"})

    def test_iterate_items(self):
        options = Options(opt1=1, opt2=2, opt3="abc")
        items = list(options.items())

        self.assertEqual(items, [("opt1", 1), ("opt2", 2), ("opt3", "abc")])

    def test_mutate_mapping(self):
        options = Options(opt1=1, opt2=2, opt3="abc")

        options["opt4"] = "def"
        self.assertEqual(options.opt4, "def")

        options_dict = dict(options)
        self.assertEqual(options_dict, {"opt1": 1, "opt2": 2, "opt3": "abc", "opt4": "def"})

    def test_mutate_mapping_validator(self):
        options = Options(shots=1024)
        options.set_validator("shots", (1, 2048))

        options["shots"] = 512
        self.assertEqual(options.shots, 512)

        with self.assertRaises(ValueError):
            options["shots"] = 3096

        self.assertEqual(options.shots, 512)


class TestOptionsSimpleNamespaceBackwardCompatibility(QiskitTestCase):
    """Tests that SimpleNamespace-like functionality that qiskit-experiments relies on for Options
    still works."""

    def test_unpacking_dict(self):
        kwargs = {"hello": "world", "a": "b"}
        options = Options(**kwargs)
        self.assertEqual(options.__dict__, kwargs)
        self.assertEqual({**options.__dict__}, kwargs)

    def test_setting_attributes(self):
        options = Options()
        options.hello = "world"
        options.a = "b"
        self.assertEqual(options.get("hello"), "world")
        self.assertEqual(options.get("a"), "b")
        self.assertEqual(options.__dict__, {"hello": "world", "a": "b"})

    def test_overriding_instance_attributes(self):
        """Test that setting instance attributes and methods does not interfere with previously
        defined attributes and methods.  This produces an inconsistency where
            >>> options = Options()
            >>> options.validators = "hello"
            >>> options.validators
            {}
            >>> options.get("validators")
            "hello"
        """
        options = Options(get="a string")
        options.validator = "another string"
        setattr(options, "update_options", "not a method")
        options.update_options(_fields="not a dict")
        options.__dict__ = "also not a dict"

        self.assertEqual(
            options.__dict__,
            {
                "get": "a string",
                "validator": "another string",
                "update_options": "not a method",
                "_fields": "not a dict",
                "__dict__": "also not a dict",
            },
        )
        self.assertEqual(
            options._fields,
            {
                "get": "a string",
                "validator": "another string",
                "update_options": "not a method",
                "_fields": "not a dict",
                "__dict__": "also not a dict",
            },
        )
        self.assertEqual(options.validator, {})
        self.assertEqual(options.get("_fields"), "not a dict")

    def test_copy(self):
        options = Options(shots=1024, method="auto", meas_level=MeasLevel.KERNELED)
        options.set_validator("shots", (1, 1024))
        options.set_validator("method", ["auto", "statevector", "mps"])
        options.set_validator("meas_level", MeasLevel)
        expected = """Options(shots=1024, method='auto', meas_level=<MeasLevel.KERNELED: 1>)
Where:
\tshots is >= 1 and <= 1024
\tmethod is one of ['auto', 'statevector', 'mps']
\tmeas_level is of type <enum 'MeasLevel'>\n"""
        self.assertEqual(str(options), expected)
        self.assertEqual(str(copy.copy(options)), expected)

    def test_pickle(self):
        options = Options(shots=1024, method="auto", meas_level=MeasLevel.KERNELED)
        options.set_validator("shots", (1, 1024))
        options.set_validator("method", ["auto", "statevector", "mps"])
        options.set_validator("meas_level", MeasLevel)
        expected = """Options(shots=1024, method='auto', meas_level=<MeasLevel.KERNELED: 1>)
Where:
\tshots is >= 1 and <= 1024
\tmethod is one of ['auto', 'statevector', 'mps']
\tmeas_level is of type <enum 'MeasLevel'>\n"""
        self.assertEqual(str(options), expected)
        self.assertEqual(str(pickle.loads(pickle.dumps(options))), expected)
