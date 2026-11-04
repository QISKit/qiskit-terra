# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test BindingsArray"""

import ddt
import numpy as np

from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit
from qiskit.primitives.containers.bindings_array import BindingsArray
from test import QiskitTestCase


@ddt.ddt
class BindingsArrayTestCase(QiskitTestCase):
    """Test the BindingsArray class"""

    def setUp(self):
        self.circuit = QuantumCircuit(5)
        self.params = ParameterVector("a", 50)
        param_iter = iter(self.params)
        for _ in range(10):
            for qubit in range(5):
                self.circuit.sx(qubit)
                self.circuit.rz(next(param_iter), qubit)
            self.circuit.cx(0, 1)
            self.circuit.cx(2, 3)
        return super().setUp()

    def test_construction_failures(self):
        """Test all the possible construction failures"""
        with self.assertRaisesRegex(ValueError, "inconsistent with last dimension of"):
            BindingsArray(data={Parameter("a"): [0, 1]}, shape=())

        with self.assertRaisesRegex(ValueError, r"Array with shape \(\) inconsistent with \(1,\)"):
            BindingsArray(data={Parameter("a"): 0}, shape=(1,))

        with self.assertRaisesRegex(ValueError, r"\(3, 5\) inconsistent with \(2,\)"):
            BindingsArray({"a": np.empty((3, 5))}, shape=2)

        with self.assertRaisesRegex(ValueError, "Could not find any consistent shape"):
            BindingsArray({"a": np.empty((5, 8, 3)), "b": np.empty((4, 7, 2))})

        with self.assertRaisesRegex(ValueError, "inconsistent with last dimension of"):
            BindingsArray(
                data={(Parameter("a"), Parameter("b")): np.empty((5, 10, 3))},
            )

        with self.assertRaisesRegex(TypeError, "complex"):
            BindingsArray(
                data={"a": 1j},
            )

    def test_repr(self):
        """Test that the repr doesn't fail"""
        # we are primarily interested in making sure some future change doesn't cause the repr to
        # raise an error. it is more sensible for humans to detect a deficiency in the formatting
        # itself, should one be uncovered
        self.assertTrue(repr(BindingsArray()).startswith("BindingsArray"))
        self.assertTrue(
            repr(BindingsArray({"p": 2, "q": 5, ("a", "b", "c", "d"): [1, 22, 4, 5]})).startswith(
                "BindingsArray"
            )
        )

    def test_bind(self):
        """Test binding at a specified index"""
        vals = np.linspace(0, 1, 1000).reshape((5, 4, 50))
        expected_circuit = self.circuit.assign_parameters(vals[2, 3])
        parameters = tuple(self.circuit.parameters)

        ba = BindingsArray({parameters: vals})
        self.assertEqual(ba.bind(self.circuit, (2, 3)), expected_circuit)

        ba = BindingsArray(
            {
                parameters[:20]: vals[:, :, :20],
                parameters[20:27]: vals[:, :, 20:27],
                parameters[27:]: vals[:, :, 27:],
            }
        )
        self.assertEqual(ba.bind(self.circuit, (2, 3)), expected_circuit)

        order = np.arange(30, 50, dtype=int)
        np.random.default_rng().shuffle(order)
        ba = BindingsArray(
            {
                parameters[0:25]: vals[:, :, :25],
                parameters[25:30]: vals[:, :, 25:30],
                tuple(self.params[i] for i in order): vals[:, :, order],
            },
        )
        self.assertEqual(ba.bind(self.circuit, (2, 3)), expected_circuit)

    def test_bind_all(self):
        """Test binding all possible values"""
        # this test assumes bind_all() is implemented via bind_at_idx(), which we have already
        # tested. so here, we just test that it gets the order right
        vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
        bound_circuits = BindingsArray({tuple(self.circuit.parameters): vals}).bind_all(
            self.circuit
        )
        self.assertIsInstance(bound_circuits, np.ndarray)
        self.assertEqual(bound_circuits.shape, (2, 3))
        for idx in np.ndindex((2, 3)):
            self.assertEqual(bound_circuits[idx], self.circuit.assign_parameters(vals[idx]))

    def test_ravel(self):
        """Test ravel"""
        vals = np.linspace(0, 1, 300).reshape((2, 3, 50))

        ba = BindingsArray({tuple(self.circuit.parameters): vals})
        flat = ba.ravel()
        self.assertEqual(flat.num_parameters, 50)
        self.assertEqual(flat.ndim, 1)
        self.assertEqual(flat.shape, (6,))
        self.assertEqual(flat.size, 6)
        flat_vals = vals.reshape(-1, 50)

        bound_circuits = list(flat.bind_all(self.circuit).reshape(6))
        self.assertEqual(len(bound_circuits), 6)
        for i in range(6):
            self.assertEqual(bound_circuits[i], self.circuit.assign_parameters(flat_vals[i]))

    def test_reshape(self):
        """Test reshape"""
        with self.subTest("reshape"):
            vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
            ba = BindingsArray({tuple(self.circuit.parameters): vals})
            reshape_ba = ba.reshape((3, 2))
            self.assertEqual(reshape_ba.num_parameters, 50)
            self.assertEqual(reshape_ba.ndim, 2)
            self.assertEqual(reshape_ba.shape, (3, 2))
            self.assertEqual(reshape_ba.size, 6)
            np.testing.assert_allclose(
                next(iter(reshape_ba.data.values())), vals.reshape((3, 2, 50))
            )

            bound_circuits = list(reshape_ba.bind_all(self.circuit).ravel())
            reshape_vals = vals.reshape((3, 2, -1))
            self.assertEqual(len(bound_circuits), 6)
            self.assertEqual(bound_circuits[0], self.circuit.assign_parameters(reshape_vals[0, 0]))
            self.assertEqual(bound_circuits[1], self.circuit.assign_parameters(reshape_vals[0, 1]))
            self.assertEqual(bound_circuits[2], self.circuit.assign_parameters(reshape_vals[1, 0]))
            self.assertEqual(bound_circuits[3], self.circuit.assign_parameters(reshape_vals[1, 1]))
            self.assertEqual(bound_circuits[4], self.circuit.assign_parameters(reshape_vals[2, 0]))
            self.assertEqual(bound_circuits[5], self.circuit.assign_parameters(reshape_vals[2, 1]))

        with self.subTest("flatten"):
            vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
            ba = BindingsArray({tuple(self.circuit.parameters): vals})
            reshape_ba = ba.reshape(6)
            self.assertEqual(reshape_ba.num_parameters, 50)
            self.assertEqual(reshape_ba.ndim, 1)
            self.assertEqual(reshape_ba.shape, (6,))
            self.assertEqual(reshape_ba.size, 6)
            np.testing.assert_allclose(next(iter(reshape_ba.data.values())), vals.reshape(6, 50))

            reshape_vals = vals.reshape(-1, 50)
            bound_circuits = list(reshape_ba.bind_all(self.circuit).ravel())
            self.assertEqual(len(bound_circuits), 6)
            for i in range(6):
                self.assertEqual(bound_circuits[i], self.circuit.assign_parameters(reshape_vals[i]))

        with self.subTest("various_formats"):
            ba = BindingsArray(
                {Parameter("a"): np.empty(16), (Parameter("b"), Parameter("c")): np.empty((16, 2))},
            )

            def various_formats(shape):
                # call reshape with a single argument
                yield [shape]
                yield [(-1,) + shape[1:]]
                yield [np.array(shape)]
                yield [list(shape)]
                yield [list(map(np.int64, shape))]
                yield [tuple(map(np.int64, shape))]

                # call reshape with multiple arguments
                yield shape
                yield np.array(shape)
                yield list(shape)
                yield list(map(np.int64, shape))
                yield tuple(map(np.int64, shape))

            for shape in [(16,), (4, 4), (2, 4, 2), (2, 2, 2, 2), (1, 8, 1, 2)]:
                for input_shape in various_formats(shape):
                    reshaped_ba = ba.reshape(input_shape)
                    self.assertEqual(reshaped_ba.shape, shape)

    def test_data(self):
        """Test constructor with data"""
        with self.subTest("binding a single value"):
            vals = np.linspace(0, 1, 50)
            data = {self.params: vals}
            ba = BindingsArray(data=data)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 0)
            self.assertEqual(ba.shape, ())
            self.assertEqual(ba.size, 1)
            self.assertEqual(ba.data, {tuple(param.name for param in self.params): vals})

            bound_circuit = ba.bind(self.circuit, ())
            self.assertEqual(bound_circuit, self.circuit.assign_parameters(vals))

        with self.subTest("binding an array"):
            vals = np.linspace(0, 1, 300).reshape((2, 3, 50))
            data = {self.params: vals}
            ba = BindingsArray(data=data)
            self.assertEqual(ba.num_parameters, 50)
            self.assertEqual(ba.ndim, 2)
            self.assertEqual(ba.shape, (2, 3))
            self.assertEqual(ba.size, 6)
            self.assertEqual(ba.data, {tuple(param.name for param in self.params): vals})

            bound_circuits = ba.bind_all(self.circuit)
            self.assertEqual(bound_circuits.shape, (2, 3))
            self.assertEqual(bound_circuits[0, 0], self.circuit.assign_parameters(vals[0, 0]))
            self.assertEqual(bound_circuits[0, 1], self.circuit.assign_parameters(vals[0, 1]))
            self.assertEqual(bound_circuits[0, 2], self.circuit.assign_parameters(vals[0, 2]))
            self.assertEqual(bound_circuits[1, 0], self.circuit.assign_parameters(vals[1, 0]))
            self.assertEqual(bound_circuits[1, 1], self.circuit.assign_parameters(vals[1, 1]))
            self.assertEqual(bound_circuits[1, 2], self.circuit.assign_parameters(vals[1, 2]))

        with self.subTest("binding a single param"):
            vals = np.linspace(0, 1, 50)
            data = {self.params[0]: vals}
            ba = BindingsArray(data=data)
            self.assertEqual(ba.num_parameters, 1)
            self.assertEqual(ba.ndim, 1)
            self.assertEqual(ba.shape, (50,))
            self.assertEqual(ba.size, 50)
            self.assertEqual(list(ba.data.keys()), [(self.params[0].name,)])
            np.testing.assert_allclose(list(ba.data.values()), [vals[..., np.newaxis]])

    def test_simple_data(self):
        """Test simple constructions of BindingsArrays using data."""
        with self.subTest("Single number kwval 1"):
            ba = BindingsArray({Parameter("a"): 1.0})
            self.assertEqual(ba.shape, ())

        with self.subTest("Single number kwval 1 with shape"):
            ba = BindingsArray(data={Parameter("a"): 1.0}, shape=())
            self.assertEqual(ba.shape, ())

        with self.subTest("Single number kwval 1 ndarray"):
            ba = BindingsArray(data={Parameter("a"): np.array(1.0)})
            self.assertEqual(ba.shape, ())

        with self.subTest("Single number kwval 2"):
            ba = BindingsArray(data={Parameter("a"): 1.0, Parameter("b"): 0.0})
            self.assertEqual(ba.shape, ())

        with self.subTest("Empty kwval"):
            ba = BindingsArray(data={Parameter("a"): []})
            self.assertEqual(ba.shape, (0,))

        with self.subTest("Single kwval"):
            ba = BindingsArray(data={Parameter("a"): [0.0]})
            self.assertEqual(ba.shape, ())

        with self.subTest("Single kwval ndarray"):
            ba = BindingsArray(data={Parameter("a"): np.array([0.0])})
            self.assertEqual(ba.shape, ())

        with self.subTest("Multi kwval"):
            ba = BindingsArray(data={Parameter("a"): [0.0, 1.0]})
            self.assertEqual(ba.shape, (2,))

        with self.subTest("Multiple data empty"):
            ba = BindingsArray(data={Parameter("a"): [], Parameter("b"): []})
            self.assertEqual(ba.shape, (0,))

        with self.subTest("Multiple data single"):
            ba = BindingsArray(data={Parameter("a"): [0.0], Parameter("b"): [1.0]})
            self.assertEqual(ba.shape, ())

        with self.subTest("Multiple data multi"):
            ba = BindingsArray(data={Parameter("a"): [0.0, 1.0], Parameter("b"): [1.0, 0.0]})
            self.assertEqual(ba.shape, (2,))

    def test_empty(self):
        """Test simple constructions of BindingsArrays with empty cases"""
        with self.subTest("Empty 1"):
            ba = BindingsArray()
            self.assertEqual(ba.shape, ())

        with self.subTest("Empty 2"):
            ba = BindingsArray({}, shape=())
            self.assertEqual(ba.shape, ())

        with self.subTest("Empty 3"):
            ba = BindingsArray(shape=())
            self.assertEqual(ba.shape, ())

        with self.subTest("Empty 5"):
            ba = BindingsArray(data={}, shape=())
            self.assertEqual(ba.shape, ())

    def test_coerce(self):
        """Test the coerce() method."""
        # BindingsArray passthrough
        arg = BindingsArray({"a": np.linspace(0, 1, 5)})
        ba = BindingsArray.coerce(arg)
        self.assertEqual(ba, arg)

        # dict-valued input
        arg = {"a": np.linspace(0, 1, 5)}
        ba = BindingsArray.coerce(arg)
        self.assertEqual(ba.num_parameters, 1)

        # None-valued input
        arg = None
        ba = BindingsArray.coerce(None)
        self.assertEqual(ba.num_parameters, 0)

    @ddt.data(
        ((0,), 0, True),
        ((), 0, True),
        ((0,), 1, True),  # this shouldn't work because we don't know if shape is (0,) or (0, 1)
        ((0,), 2, True),
        ((1,), 0, True),
        ((0,), 0, False),
        ((2, 3), 0, True),
        ((), 0, False),
        ((0,), 1, False),
        ((0,), 2, False),
        ((1,), 0, False),
        ((2, 3), 0, False),
    )
    @ddt.unpack
    def test_shape_with_0(self, shape, num_params, do_inference):
        """Tests various combinations of inputs that include 0-d axes."""
        ba = BindingsArray(
            {tuple(f"a{idx}" for idx in range(num_params)): np.empty(shape + (num_params,))},
            shape=(None if do_inference else shape),
        )
        self.assertEqual(ba.shape, shape)
        self.assertEqual(ba.num_parameters, num_params)

        if num_params == 1:
            # if there is 1 parameter, we should be allowed to leave it off as an axis
            ba = BindingsArray(
                {tuple(f"a{idx}" for idx in range(num_params)): np.empty(shape)},
                shape=(None if do_inference else shape),
            )
            self.assertEqual(ba.shape, shape)
            self.assertEqual(ba.num_parameters, num_params)

    @ddt.idata([(True, True), (True, False), (False, True), (False, False)])
    @ddt.unpack
    def test_as_array_bad_param_raises(self, kwvals_str, args_str):
        """Test as_array() raises when a parameter key is missing."""
        kwval_param = lambda param: Parameter(param) if kwvals_str else param
        args_param = lambda param: Parameter(param) if args_str else param

        ba = BindingsArray({(kwval_param("a"), kwval_param("b")): np.empty((5, 2))})
        with self.assertRaisesRegex(ValueError, "Expected 2 parameters but 1 received"):
            ba.as_array([args_param("b")])

        ba = BindingsArray({(kwval_param("a"), kwval_param("b")): np.empty((5, 2))})
        with self.assertRaisesRegex(ValueError, "Expected 2 parameters but 3 received"):
            ba.as_array([args_param("b"), args_param("a"), args_param("b")])

        with self.assertRaisesRegex(ValueError, "Could not find placement for parameter 'a'"):
            ba.as_array([args_param("b"), args_param("c")])

    @ddt.idata([(True, True), (True, False), (False, True), (False, False)])
    @ddt.unpack
    def test_as_array(self, kwvals_str, args_str):
        """Test as_array() works for various combinations of string/Parameter inputs."""
        kwval_param = lambda param: Parameter(param) if kwvals_str else param
        args_param = lambda param: Parameter(param) if args_str else param

        arr_a = np.linspace(0, 20, 250).reshape((25, 5, 2))
        arr_b = np.linspace(0, 5, 375).reshape((25, 5, 3))
        ba = BindingsArray(
            {
                (kwval_param("a"), kwval_param("b")): arr_a,
                (kwval_param("c"), kwval_param("d"), kwval_param("e")): arr_b,
            }
        )
        np.testing.assert_allclose(ba.as_array(), np.concatenate([arr_a, arr_b], axis=2))

        params = map(args_param, "dabec")
        expected = [arr_b[..., 1], arr_a[..., 0], arr_a[..., 1], arr_b[..., 2], arr_b[..., 0]]
        expected = np.concatenate([arr[..., None] for arr in expected], axis=2)
        np.testing.assert_allclose(ba.as_array(params), expected)

    def test_as_array_cases(self):
        """Test as_array() works in various edge cases."""
        ba = BindingsArray({(): np.ones((1, 2, 0))})
        arr = ba.as_array()
        self.assertEqual(arr.shape, (1, 2, 0))

        ba = BindingsArray({(): np.ones((0,))})
        arr = ba.as_array()
        self.assertEqual(arr.shape, (0,))

        ba = BindingsArray({("a", "b", "c"): np.ones((0, 3))})
        arr = ba.as_array()
        self.assertEqual(arr.shape, (0, 3))

        ba = BindingsArray({"a": np.ones((0, 1))}, shape=(0,))
        arr = ba.as_array()
        self.assertEqual(arr.shape, (0, 1))

    def test_get_item(self):
        """Test the __getitem__() method."""
        ba = BindingsArray()
        self.assertEqual(ba[:].shape, ())
        self.assertEqual(ba[:].num_parameters, 0)

        data = np.linspace(0, 1, 300).reshape((5, 6, 10))
        params = tuple(f"a{idx:03d}" for idx in range(10))
        ba = BindingsArray({params: data})
        np.testing.assert_allclose(ba[...].as_array(params), data)
        np.testing.assert_allclose(ba[0].as_array(params), data[0])
        np.testing.assert_allclose(ba[6:2:-1, -1].as_array(params), data[6:2:-1, -1])

        data = np.linspace(0, 1, 300).reshape((5, 6, 10))
        params = tuple(f"a{idx:03d}" for idx in range(10))
        ba = BindingsArray({params[:3]: data[..., :3], params[3:]: data[..., 3:]})
        np.testing.assert_allclose(ba[...].as_array(params), data)
        np.testing.assert_allclose(ba[0].as_array(params), data[0])
        np.testing.assert_allclose(ba[6:2:-1, -1].as_array(params), data[6:2:-1, -1])
