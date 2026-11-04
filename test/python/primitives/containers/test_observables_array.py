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

"""Test ObservablesArray"""

import itertools as it
import ddt
import numpy as np

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives.containers.observables_array import (
    ObservablesArray,
    object_array,
)
from test import QiskitTestCase


@ddt.ddt
class ObservablesArrayTestCase(QiskitTestCase):
    """Test the ObservablesArray class"""

    @ddt.data(0, 1, 2)
    def test_coerce_observable_str(self, num_qubits):
        """Test coerce_observable for allowed basis str input"""
        for chars in it.permutations("IXYZ01+-lr", num_qubits):
            label = "".join(chars)
            obs = ObservablesArray.coerce_observable(label)
            self.assertEqual(obs, qi.SparseObservable.from_label(label))

    @ddt.data("iXX", "012", "+/-")
    def test_coerce_observable_invalid_str(self, basis):
        """Test coerce_observable for Pauli input"""
        with self.assertRaises(ValueError):
            ObservablesArray.coerce_observable(basis)

    @ddt.data(1, 2, 3)
    def test_coerce_observable_pauli(self, num_qubits):
        """Test coerce_observable for Pauli input"""
        for p in qi.pauli_basis(num_qubits):
            obs = ObservablesArray.coerce_observable(p)
            self.assertEqual(obs, qi.SparseObservable.from_pauli(p))

    @ddt.data(0, 1, 2, 3)
    def test_coerce_observable_phased_pauli(self, phase):
        """Test coerce_observable for phased Pauli input"""
        pauli = qi.Pauli("IXYZ")
        pauli.phase = phase
        coeff = (-1j) ** phase
        if phase % 2:
            with self.assertRaises(ValueError):
                ObservablesArray.coerce_observable(pauli)
        else:
            obs = ObservablesArray.coerce_observable(pauli)
            self.assertIsInstance(obs, qi.SparseObservable)
            obs_pauli, obs_qubits, obs_coeff = obs.to_sparse_list()[0]

            # ZYX and not XYZ because the qubits in `obs_qubits` are ascending
            self.assertEqual(obs_pauli, "ZYX")
            self.assertEqual(obs_qubits, [0, 1, 2])

            np.testing.assert_almost_equal(
                obs_coeff, coeff, err_msg=f"Wrong value for Pauli {pauli}"
            )

    @ddt.data("+IXYZ", "-IXYZ", "iIXYZ", "+iIXYZ", "-IXYZ")
    def test_coerce_observable_phased_pauli_str(self, pauli):
        """Test coerce_observable for phased Pauli input"""
        pauli = qi.Pauli(pauli)
        coeff = (-1j) ** pauli.phase
        if pauli.phase % 2:
            with self.assertRaises(ValueError):
                ObservablesArray.coerce_observable(pauli)
        else:
            obs = ObservablesArray.coerce_observable(pauli)
            self.assertIsInstance(obs, qi.SparseObservable)
            obs_pauli, obs_qubits, obs_coeff = obs.to_sparse_list()[0]

            # ZYX and not XYZ because the qubits in `obs_qubits` are ascending
            self.assertEqual(obs_pauli, "ZYX")
            self.assertEqual(obs_qubits, [0, 1, 2])

            np.testing.assert_almost_equal(
                obs_coeff, coeff, err_msg=f"Wrong value for Pauli {pauli}"
            )

    def test_coerce_observable_signed_sparse_pauli_op(self):
        """Test coerce_observable for SparsePauliOp input with phase paulis"""
        op = qi.SparsePauliOp(["+I", "-X", "Y", "-Z"], [1, 2, 3, 4])
        obs = ObservablesArray.coerce_observable(op)
        self.assertIsInstance(obs, qi.SparseObservable)
        sparse_list = sorted(obs.to_sparse_list())
        self.assertEqual(len(sparse_list), 4)
        obs_paulis = [term[0] for term in sparse_list]
        obs_coeffs = [term[2] for term in sparse_list]
        self.assertEqual(obs_paulis, ["", "X", "Y", "Z"])
        np.testing.assert_allclose(obs_coeffs, [1, -2, 3, -4])

    def test_coerce_observable_zero_sparse_pauli_op(self):
        """Test coerce_observable for SparsePauliOp input with zero val coeffs"""
        op = qi.SparsePauliOp(["I", "X", "Y", "Z"], [0, 0, 0, 1])
        obs = ObservablesArray.coerce_observable(op)
        self.assertIsInstance(obs, qi.SparseObservable)
        sparse_list = obs.to_sparse_list()
        self.assertEqual(len(sparse_list), 1)
        obs_pauli, _, obs_coeff = sparse_list[0]
        self.assertEqual(obs_pauli, "Z")
        self.assertEqual(obs_coeff, 1)

    def test_coerce_observable_duplicate_sparse_pauli_op(self):
        """Test coerce_observable for SparsePauliOp with duplicate paulis"""
        op = qi.SparsePauliOp(["XX", "-XX", "XX", "-XX"], [2, 1, 3, 2])
        obs = ObservablesArray.coerce_observable(op)
        self.assertIsInstance(obs, qi.SparseObservable)
        sparse_list = obs.to_sparse_list()
        self.assertEqual(len(sparse_list), 1)
        obs_pauli, _, obs_coeff = sparse_list[0]
        self.assertEqual(obs_pauli, "XX")
        self.assertEqual(obs_coeff, 2)

    def test_coerce_observable_pauli_mapping(self):
        """Test coerce_observable for pauli-keyed Mapping input"""
        mapping = dict(zip(qi.pauli_basis(1), range(1, 5)))
        obs = ObservablesArray.coerce_observable(mapping)
        target = qi.SparseObservable.from_list(
            [(key.to_label(), val) for key, val in mapping.items()]
        ).simplify()
        self.assertEqual(obs, target)

    def test_coerce_0d(self):
        """Test the coerce() method with 0-d input."""
        obs = ObservablesArray.coerce("X")
        self.assertEqual(obs.shape, ())
        self.assertDictAlmostEqual(obs[()], {"X": 1})

        obs = ObservablesArray.coerce({"I": 2})
        self.assertEqual(obs.shape, ())
        self.assertDictAlmostEqual(obs[()], {"I": 2})

        obs = ObservablesArray.coerce(qi.SparsePauliOp(["X", "Y"], [1, 3]))
        self.assertEqual(obs.shape, ())
        self.assertDictAlmostEqual(obs[()], {"X": 1, "Y": 3})

    def test_format_invalid_mapping_qubits(self):
        """Test an error is raised when different qubits in mapping keys"""
        mapping = {"IX": 1, "XXX": 2}
        with self.assertRaises(ValueError):
            ObservablesArray.coerce_observable(mapping)

    def test_format_invalid_mapping_basis(self):
        """Test an error is raised when keys contain invalid characters"""
        mapping = {"XX": 1, "0Z": 2, "02": 3}
        with self.assertRaises(ValueError):
            ObservablesArray.coerce_observable(mapping)

    def test_init_nested_list_str(self):
        """Test init with nested lists of str"""
        obj = [["X", "Y", "Z"], ["0", "1", "+"]]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 6)
        self.assertEqual(obs.shape, (2, 3))

    def test_init_nested_list_sparse_pauli_op(self):
        """Test init with nested lists of SparsePauliOp"""
        obj = [
            [qi.SparsePauliOp(qi.random_pauli_list(2, 3, phase=False)) for _ in range(3)]
            for _ in range(5)
        ]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 15)
        self.assertEqual(obs.shape, (5, 3))

    def test_init_single_sparse_pauli_op(self):
        """Test init with single SparsePauliOps"""
        obj = qi.SparsePauliOp(qi.random_pauli_list(2, 3, phase=False))
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 1)
        self.assertEqual(obs.shape, ())

    def test_init_pauli_list(self):
        """Test init with PauliList"""
        obs = ObservablesArray(qi.pauli_basis(2))
        self.assertEqual(obs.size, 16)
        self.assertEqual(obs.shape, (16,))

    def test_init_nested_pauli_list(self):
        """Test init with nested PauliList"""
        obj = [qi.random_pauli_list(2, 3, phase=False) for _ in range(5)]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.size, 15)
        self.assertEqual(obs.shape, (5, 3))

    def test_init_ragged_array(self):
        """Test init with ragged input"""
        obj = [["X", "Y"], ["X", "Y", "Z"]]
        with self.assertRaises(ValueError):
            ObservablesArray(obj)

    def test_init_validate_false(self):
        """Test init validate kwarg"""
        obj = [["X", "Y", "Z"], ["I", "0", "1"]]
        obs = ObservablesArray(obj, validate=False, num_qubits=1)
        self.assertEqual(obs.shape, (2, 3))
        self.assertEqual(obs.size, 6)
        for i in range(2):
            for j in range(3):
                self.assertEqual(obs._array[i, j], obj[i][j])

    def test_init_validate_true(self):
        """Test init validate kwarg"""
        obj = [["A", "B", "C"], ["D", "E", "F"]]
        with self.assertRaises(ValueError):
            ObservablesArray(obj, validate=True)

    @ddt.data(0, 1, 2, 3)
    def test_size_and_shape_single(self, ndim):
        """Test size and shape method for size=1 array"""
        obs = {"XX": 1}
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs)
        self.assertEqual(arr.size, 1, msg="Incorrect ObservablesArray.size")
        self.assertEqual(arr.shape, (1,) * ndim, msg="Incorrect ObservablesArray.shape")

    @ddt.data(0, 1, 2, 3)
    def test_tolist_single(self, ndim):
        """Test tolist method for size=1 array"""
        obs = {"XX": 1}
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs)
        ls = arr.tolist()
        self.assertEqual(ls, obs)

    @ddt.data(0, 1, 2, 3)
    def test_array_single(self, ndim):
        """Test __array__ method for size=1 array"""
        obs = {"XX": 1}
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs)
        nparr = np.array(arr)
        self.assertEqual(nparr.dtype, object)
        self.assertEqual(nparr.shape, arr.shape)
        self.assertEqual(nparr.size, arr.size)
        self.assertTrue(np.all(nparr == np.array(obs)))

    @ddt.data(0, 1, 2, 3)
    def test_getitem_single(self, ndim):
        """Test __getitem__ method for size=1 array"""
        base_obs = {"XX": 1}
        obs = base_obs
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs)
        idx = ndim * (0,)
        item = arr[idx]
        self.assertEqual(item, base_obs)

    @ddt.data(0, 1, 2, 3)
    def test_slice_single(self, ndim):
        """Test slice method for size=1 array"""
        base_obs = qi.SparseObservable.from_label("XX")
        obs = base_obs
        for _ in range(ndim):
            obs = [obs]
        arr = ObservablesArray(obs)
        idx = ndim * (0,)
        item = arr.slice(idx)
        self.assertEqual(item, base_obs)

    def test_tolist_1d(self):
        """Test tolist method"""
        obj = [{"I": 1}, {"X": 2}, {"Y": 3}, {"Z": 4}]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.tolist(), obj)

    def test_tolist_2d(self):
        """Test tolist method"""
        obj = [[{"II": 1.0}, {"XI": 2.0}, {"IY": 3.0}], [{"XX": 1.0}, {"XY": 2.0}, {"YY": 3.0}]]
        obs = ObservablesArray(obj)
        self.assertEqual(obs.tolist(), obj)

    def test_array_1d(self):
        """Test __array__ dunder method"""
        obj = np.array([{"I": 1}, {"X": 2}, {"Y": 3}, {"Z": 4}], dtype=object)
        obs = ObservablesArray(obj)
        self.assertTrue(np.all(np.array(obs) == obj))

    def test_array_2d(self):
        """Test __array__ dunder method"""
        obj = np.array(
            [[{"II": 1}, {"XI": 2}, {"IY": 3}], [{"XX": 1}, {"XY": 2}, {"YY": 3}]], dtype=object
        )
        obs = ObservablesArray(obj)
        self.assertTrue(np.all(np.array(obs) == obj))

    def test_getitem_1d(self):
        """Test __getitem__ for 1D array"""
        obj = np.array([{"I": 1}, {"X": 2}, {"Y": 3}, {"Z": 4}], dtype=object)
        obs = ObservablesArray(obj)

        for i in range(obj.size):
            self.assertEqual(obs[i], obj[i])

        sub_obs = obs[1:3]
        self.assertTrue(isinstance(sub_obs, ObservablesArray))
        self.assertTrue(np.all(np.array(sub_obs) == obj[1:3]))

    def test_getitem_2d(self):
        """Test __getitem__ for 2D array"""
        obj = np.array(
            [[{"II": 1}, {"XI": 2}, {"IY": 3}], [{"XX": 1}, {"XY": 2}, {"YY": 3}]], dtype=object
        )
        obs = ObservablesArray(obj)

        for i in range(2):
            row = obs[i]
            self.assertEqual(row.shape, (3,))
            self.assertTrue(np.all(np.array(row) == obj[i]))

        sub_obs_slice = obs[1:2]
        row = sub_obs_slice[0]
        self.assertTrue(isinstance(row, ObservablesArray))
        self.assertEqual(row.shape, (3,))
        self.assertTrue(np.all(np.array(row) == obj[1]))

        sub_obs_two_slices = obs[1:2, 1:3]
        row = sub_obs_two_slices[0]
        self.assertTrue(isinstance(row, ObservablesArray))
        self.assertEqual(row.shape, (2,))
        self.assertTrue(np.all(np.array(row) == obj[1, 1:3]))

        sub_obs_int_and_slice = obs[1, 1:3]
        self.assertTrue(isinstance(sub_obs_int_and_slice, ObservablesArray))
        self.assertEqual(sub_obs_int_and_slice.shape, (2,))
        self.assertTrue(np.all(np.array(sub_obs_int_and_slice) == obj[1, 1:3]))

        sub_obs_slice_and_int = obs[1:2, 1]
        self.assertTrue(isinstance(sub_obs_slice_and_int, ObservablesArray))
        elem = sub_obs_slice_and_int[0]
        self.assertTrue(isinstance(elem, dict))
        self.assertEqual(elem, obj[1][1])

        two_ints = obs[1, 1]
        self.assertTrue(isinstance(two_ints, dict))
        self.assertEqual(two_ints, obj[1][1])

        sub_obs_ellipsis = obs[..., 1]
        self.assertTrue(isinstance(sub_obs_ellipsis, ObservablesArray))
        self.assertEqual(sub_obs_ellipsis.shape, (2,))
        self.assertTrue(np.all(np.array(sub_obs_ellipsis) == obj[..., 1]))

        obs_none = obs[None]
        self.assertTrue(isinstance(obs_none, ObservablesArray))
        self.assertTrue(obs_none[0].equivalent(obs))

    def test_slice_1d(self):
        """Test slice for 1D array"""
        obj = [
            qi.SparseObservable.from_list([obs]) for obs in [("I", 1), ("X", 2), ("Y", 3), ("Z", 4)]
        ]
        obs = ObservablesArray(obj)

        for i in range(4):
            self.assertEqual(obs.slice(i), obj[i])

        sub_obs = obs.slice(slice(1, 3))
        self.assertTrue(isinstance(sub_obs, ObservablesArray))
        for i in range(2):
            self.assertEqual(sub_obs.slice(i), obj[i + 1])

    def test_slice_2d(self):
        """Test slice for 2D array"""
        obj = object_array(
            [
                [qi.SparseObservable.from_list([obs]) for obs in [("II", 1), ("XI", 2), ("IY", 3)]],
                [qi.SparseObservable.from_list([obs]) for obs in [("XX", 1), ("XY", 2), ("YY", 3)]],
            ]
        )
        obs = ObservablesArray(obj)

        for i in range(2):
            row = obs.slice(i)
            self.assertEqual(row.shape, (3,))
            for j in range(3):
                self.assertEqual(row.slice(j), obj[i][j])

        sub_obs_slice = obs.slice(slice(1, 2))
        row = sub_obs_slice.slice(0)
        self.assertTrue(isinstance(row, ObservablesArray))
        self.assertEqual(row.shape, (3,))
        for j in range(3):
            self.assertEqual(row.slice(j), obj[1][j])

        sub_obs_two_slices = obs.slice((slice(1, 2), slice(1, 3)))
        row = sub_obs_two_slices.slice(0)
        self.assertTrue(isinstance(row, ObservablesArray))
        self.assertEqual(row.shape, (2,))
        for j in range(2):
            self.assertEqual(row.slice(j), obj[i][j + 1])

        sub_obs_int_and_slice = obs.slice((1, slice(1, 3)))
        self.assertTrue(isinstance(sub_obs_int_and_slice, ObservablesArray))
        self.assertEqual(sub_obs_int_and_slice.shape, (2,))
        for j in range(2):
            self.assertEqual(sub_obs_int_and_slice.slice(j), obj[i][j + 1])

        sub_obs_slice_and_int = obs.slice((slice(1, 2), 1))
        self.assertTrue(isinstance(sub_obs_slice_and_int, ObservablesArray))
        elem = sub_obs_slice_and_int.slice(0)
        self.assertTrue(isinstance(elem, qi.SparseObservable))
        self.assertEqual(elem, obj[1][1])

        two_ints = obs.slice((1, 1))
        self.assertTrue(isinstance(two_ints, qi.SparseObservable))
        self.assertEqual(two_ints, obj[1][1])

        sub_obs_ellipsis = obs.slice((..., 1))
        self.assertTrue(isinstance(sub_obs_ellipsis, ObservablesArray))
        self.assertEqual(sub_obs_ellipsis.shape, (2,))
        for i in range(2):
            self.assertEqual(sub_obs_ellipsis.slice(i), obj[i][1])

        obs_none = obs.slice(None)
        self.assertTrue(isinstance(obs_none, ObservablesArray))
        self.assertTrue(obs_none.slice(0).equivalent(obs))

    def test_get_dim_zero(self):
        """Test __getitem__ and slice for arrays of dimension 0"""
        obs = qi.SparseObservable.from_label("Z")
        arr = ObservablesArray(obs)

        self.assertEqual(arr[()], {"Z": 1})
        self.assertEqual(arr.slice(()), obs)

    def test_ravel(self):
        """Test ravel method"""
        bases_flat = qi.pauli_basis(2).to_labels()
        bases = [bases_flat[4 * i : 4 * (i + 1)] for i in range(4)]
        obs = ObservablesArray(bases)
        flat = obs.ravel()
        self.assertEqual(flat.ndim, 1)
        self.assertEqual(flat.shape, (16,))
        self.assertEqual(flat.size, 16)
        for (
            i,
            label,
        ) in enumerate(bases_flat):
            self.assertEqual(flat[i], {label: 1})

    def test_reshape(self):
        """Test reshape method"""
        bases = qi.pauli_basis(2)
        labels = np.array(bases.to_labels(), dtype=object)
        obs = ObservablesArray(qi.pauli_basis(2))

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
            with self.subTest(shape):
                for input_shape in various_formats(shape):
                    obs_rs = obs.reshape(*input_shape)
                    self.assertEqual(obs_rs.shape, shape)
                    labels_rs = labels.reshape(shape)
                    for idx in np.ndindex(shape):
                        self.assertEqual(
                            obs_rs[idx],
                            {labels_rs[idx]: 1},
                            msg=f"failed for shape {shape} with input format {input_shape}",
                        )

    def test_num_qubits(self):
        """Test num_qubits method"""
        obs = ObservablesArray([{"XXY": 1, "YZI": 2}, {"IYX": 3}])
        self.assertEqual(obs.num_qubits, 3)

        with self.assertRaisesRegex(ValueError, "number of qubits"):
            obs = ObservablesArray([{"XXY": 1, "YZI": 2}, {"IYX": 3}], num_qubits=20)

        with self.assertRaisesRegex(ValueError, "number of qubits"):
            obs = ObservablesArray([{"XXY": 1, "YZI": 2}, {"YX": 3}], num_qubits=20)

        obs = ObservablesArray([{"XX": 1}] * 15).reshape((3, 5))
        self.assertEqual(obs.num_qubits, 2)

        obs = ObservablesArray([{"XX": 1}] * 15)[4:6]
        self.assertEqual(obs.num_qubits, 2)

        obs = ObservablesArray(
            [ObservablesArray.coerce({"XX": 1}), ObservablesArray.coerce({"XYZ": 1})],
            validate=False,
        )
        self.assertEqual(obs.num_qubits, 2)

    def test_estimator_workflow(self):
        """Test that everything plays together when observables are specified with
        SparseObservable."""
        backend = BasicSimulator()
        estimator = BackendEstimatorV2(backend=backend)

        circ = QuantumCircuit(1)
        circ.x(0)

        obs = qi.SparseObservable.from_label("Z")

        res = estimator.run([(circ, [obs])]).result()
        self.assertEqual(res[0].data.evs, -1)

        obs_array = ObservablesArray([obs] * 15).reshape(3, 5)
        res = estimator.run([(circ, obs_array)]).result()
        self.assertTrue(np.all(res[0].data.evs == -np.ones((3, 5))))

    def test_equivalent(self):
        """Test equivalent method"""

        arr1 = ObservablesArray([[{"XY": 1}, {"YZ": 2, "ZI": 3}], [{"IX": 4, "XY": 5}, {"YZ": 6}]])
        arr2 = ObservablesArray([[{"XY": 1}, {"YZ": 2, "ZI": 3}], [{"IX": 4, "XY": 5}, {"YZ": 6}]])
        self.assertTrue(arr1.equivalent(arr2))

        arr2 = ObservablesArray([[{"XY": 1}, {"YZ": 2, "ZI": 3}], [{"IX": 4}, {"YZ": 6}]])
        self.assertFalse(arr1.equivalent(arr2))

        arr2 = ObservablesArray(
            [[{"IXY": 1}, {"IYZ": 2, "IZI": 3}], [{"IIX": 4, "IXY": 5}, {"IYZ": 6}]]
        )
        self.assertFalse(arr1.equivalent(arr2))

        arr2 = ObservablesArray([{"XY": 1}, {"YZ": 2, "ZI": 3}])
        self.assertFalse(arr1.equivalent(arr2))

        arr2 = ObservablesArray({"YZ": 2, "ZI": 3})
        self.assertFalse(arr1.equivalent(arr2))

        arr1 = ObservablesArray({"YZ": 2, "ZI": 3})
        self.assertTrue(arr1.equivalent(arr2))

    def test_apply_layout(self):
        """Test apply_layout method"""

        arr = ObservablesArray([[{"XY": 1}, {"YZ": 2, "ZI": 3}], [{"IX": 4, "XY": 5}, {"YZ": 6}]])
        new_arr = arr.apply_layout([2, 0], 3)
        self.assertTrue(
            new_arr.equivalent(
                ObservablesArray(
                    [[{"YIX": 1}, {"ZIY": 2, "IIZ": 3}], [{"XII": 4, "YIX": 5}, {"ZIY": 6}]]
                )
            )
        )

        new_arr = arr.apply_layout(None, 3)
        self.assertTrue(
            new_arr.equivalent(
                ObservablesArray(
                    [[{"IXY": 1}, {"IYZ": 2, "IZI": 3}], [{"IIX": 4, "IXY": 5}, {"IYZ": 6}]]
                )
            )
        )

        new_arr = arr.apply_layout([1, 0])
        self.assertTrue(
            new_arr.equivalent(
                ObservablesArray([[{"YX": 1}, {"ZY": 2, "IZ": 3}], [{"XI": 4, "YX": 5}, {"ZY": 6}]])
            )
        )

        new_arr = arr.apply_layout(None)
        self.assertTrue(
            new_arr.equivalent(
                ObservablesArray([[{"XY": 1}, {"YZ": 2, "ZI": 3}], [{"IX": 4, "XY": 5}, {"YZ": 6}]])
            )
        )

        arr = ObservablesArray({"YZ": 2, "ZI": 3})
        new_arr = arr.apply_layout([2, 0], 3)
        self.assertTrue(new_arr.equivalent(ObservablesArray({"ZIY": 2, "IIZ": 3})))

    def test_validate(self):
        """Test the validate method"""
        ObservablesArray({"XX": 1}).validate()
        ObservablesArray([{"XX": 1}] * 5).validate()
        ObservablesArray([{"XX": 1}] * 15).reshape((3, 5)).validate()

        obs = ObservablesArray(
            [ObservablesArray.coerce({"XX": 1}), ObservablesArray.coerce({"XYZ": 1})],
            validate=False,
        )
        with self.assertRaisesRegex(ValueError, "number of qubits"):
            obs.validate()

    def test_sparse_observables_array(self):
        """Test sparse_observables_array method"""
        obsarray = ObservablesArray([{"Z": 1}])
        arr = obsarray._array

        arr1 = obsarray.sparse_observables_array(copy=True)
        self.assertEqual(arr1, arr)
        self.assertNotEqual(id(arr1), id(arr))

        arr2 = obsarray.sparse_observables_array(copy=False)
        self.assertEqual(arr2, arr)
        self.assertEqual(id(arr2), id(arr))

    def test_empty_observable(self):
        """Verify that empty observables are not allowed"""
        with self.assertRaisesRegex(ValueError, "Empty observable"):
            ObservablesArray([{"Z": 0}])

    def test_hermitian_after_simplification(self):
        """Verify that no error is raised if observables contain complex coefficients
        that get cancelled during simplification"""
        obs = qi.SparseObservable.from_list([("Z", 1j), ("Z", -1j), ("X", 1)])
        ObservablesArray(obs)

    def test_invalid_basis_type_raises_type_error(self):
        """Test that invalid basis type raises TypeError"""
        invalid_basis = {1: "value", 2: "another_value"}  # Invalid keys (integers)
        with self.assertRaises(TypeError):
            ObservablesArray.coerce_observable(invalid_basis)
