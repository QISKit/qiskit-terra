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

"""Tests for Backend Estimator V2."""

from __future__ import annotations

import unittest
from test import QiskitTestCase, combine
from unittest.mock import patch

import numpy as np
from ddt import ddt

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.utils import optionals
from ..legacy_cmaps import LAGOS_CMAP


def _basic_simulator():
    return BasicSimulator()


def _generic_backend_v2():
    return GenericBackendV2(
        num_qubits=7,
        basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
        coupling_map=LAGOS_CMAP,
        seed=42,
    )


class _BackendFactory:
    def __init__(self, name, factory):
        self.__name__ = name
        self._factory = factory

    def __call__(self):
        return self._factory()

    def __repr__(self):
        return self.__name__


BACKENDS = [
    _BackendFactory("basic_simulator", _basic_simulator),
    _BackendFactory("generic_backend_v2", _generic_backend_v2),
]


@ddt
class TestBackendEstimatorV2(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
        self._precision = 5e-3
        self._rtol = 3e-1
        self._seed = 12
        self._rng = np.random.default_rng(self._seed)
        self._options = {"default_precision": self._precision, "seed_simulator": self._seed}
        self.ansatz = QuantumCircuit(2)
        self.ansatz.append(real_amplitudes(num_qubits=2, reps=2), [0, 1])
        self.observable = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )
        self.expvals = -1.0284380963435145, -1.284366511861733

        ra_2_reps = QuantumCircuit(2)
        ra_2_reps.append(real_amplitudes(num_qubits=2, reps=2), [0, 1])
        ra_3_reps = QuantumCircuit(2)
        ra_3_reps.append(real_amplitudes(num_qubits=2, reps=3), [0, 1])
        self.psi = (ra_2_reps, ra_3_reps)
        self.params = tuple(psi.parameters for psi in self.psi)
        self.hamiltonian = (
            SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)]),
            SparsePauliOp.from_list([("IZ", 1)]),
            SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)]),
        )
        self.theta = (
            [0, 1, 1, 2, 3, 5],
            [0, 1, 1, 2, 3, 5, 8, 13],
            [1, 2, 3, 4, 5, 6],
        )

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_estimator_run(self, backend, abelian_grouping):
        """Test Estimator.run()"""
        backend = backend()
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        psi1, psi2 = pm.run([psi1, psi2])
        estimator = BackendEstimatorV2(backend=backend, options=self._options)
        estimator.options.abelian_grouping = abelian_grouping
        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        ham1 = hamiltonian1.apply_layout(psi1.layout)
        job = estimator.run([(psi1, ham1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.5555572817900956], rtol=self._rtol)

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H1|psi2(theta2)> ]
        ham1 = hamiltonian1.apply_layout(psi2.layout)
        result2 = estimator.run([(psi2, ham1, theta2)]).result()
        np.testing.assert_allclose(result2[0].data.evs, [2.97797666], rtol=self._rtol)

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        ham2 = hamiltonian2.apply_layout(psi1.layout)
        ham3 = hamiltonian3.apply_layout(psi1.layout)
        result3 = estimator.run([(psi1, [ham2, ham3], theta1)]).result()
        np.testing.assert_allclose(result3[0].data.evs, [-0.551653, 0.07535239], rtol=self._rtol)

        # calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
        #              <psi1(theta3)|H3|psi1(theta3)>],
        #             [<psi2(theta2)|H2|psi2(theta2)>] ]
        ham1 = hamiltonian1.apply_layout(psi1.layout)
        ham3 = hamiltonian3.apply_layout(psi1.layout)
        ham2 = hamiltonian2.apply_layout(psi2.layout)
        result4 = estimator.run(
            [
                (psi1, [ham1, ham3], [theta1, theta3]),
                (psi2, ham2, theta2),
            ]
        ).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318], rtol=self._rtol)
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238], rtol=self._rtol)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_estimator_with_pub(self, backend, abelian_grouping):
        """Test estimator with explicit EstimatorPubs."""
        backend = backend()
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        psi1, psi2 = pm.run([psi1, psi2])

        ham1 = hamiltonian1.apply_layout(psi1.layout)
        ham3 = hamiltonian3.apply_layout(psi1.layout)
        obs1 = ObservablesArray.coerce([ham1, ham3])
        bind1 = BindingsArray.coerce({tuple(psi1.parameters): [theta1, theta3]})
        pub1 = EstimatorPub(psi1, obs1, bind1)

        ham2 = hamiltonian2.apply_layout(psi2.layout)
        obs2 = ObservablesArray.coerce(ham2)
        bind2 = BindingsArray.coerce({tuple(psi2.parameters): theta2})
        pub2 = EstimatorPub(psi2, obs2, bind2)

        estimator = BackendEstimatorV2(backend=backend, options=self._options)
        estimator.options.abelian_grouping = abelian_grouping
        result4 = estimator.run([pub1, pub2]).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318], rtol=self._rtol)
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238], rtol=self._rtol)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_estimator_run_no_params(self, backend, abelian_grouping):
        """test for estimator without parameters"""
        backend = backend()
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        circuit = pm.run(circuit)
        est = BackendEstimatorV2(backend=backend, options=self._options)
        est.options.abelian_grouping = abelian_grouping
        observable = self.observable.apply_layout(circuit.layout)
        result = est.run([(circuit, observable)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.284366511861733], rtol=self._rtol)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_run_single_circuit_observable(self, backend, abelian_grouping):
        """Test for single circuit and single observable case."""
        backend = backend()
        est = BackendEstimatorV2(backend=backend, options=self._options)
        est.options.abelian_grouping = abelian_grouping
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)

        with self.subTest("No parameter"):
            qc = QuantumCircuit(1)
            qc.x(0)
            qc = pm.run(qc)
            op = SparsePauliOp("Z")
            op = op.apply_layout(qc.layout)
            param_vals = [None, [], [[]], np.array([]), np.array([[]]), [np.array([])]]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target, rtol=self._rtol)
                self.assertEqual(result[0].metadata["target_precision"], self._precision)

        with self.subTest("One parameter"):
            param = Parameter("x")
            qc = QuantumCircuit(1)
            qc.ry(param, 0)
            qc = pm.run(qc)
            op = SparsePauliOp("Z")
            op = op.apply_layout(qc.layout)
            param_vals = [
                [np.pi],
                np.array([np.pi]),
            ]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target, rtol=self._rtol)
                self.assertEqual(result[0].metadata["target_precision"], self._precision)

        with self.subTest("More than one parameter"):
            qc = self.psi[0]
            qc = pm.run(qc)
            op = self.hamiltonian[0]
            op = op.apply_layout(qc.layout)
            param_vals = [
                self.theta[0],
                [self.theta[0]],
                np.array(self.theta[0]),
                np.array([self.theta[0]]),
                [np.array(self.theta[0])],
            ]
            target = [1.5555572817900956]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target, rtol=self._rtol)
                self.assertEqual(result[0].metadata["target_precision"], self._precision)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_run_1qubit(self, backend, abelian_grouping):
        """Test for 1-qubit cases"""
        backend = backend()
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        qc, qc2 = pm.run([qc, qc2])

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        est = BackendEstimatorV2(backend=backend, options=self._options)
        est.options.abelian_grouping = abelian_grouping
        op_1 = op.apply_layout(qc.layout)
        result = est.run([(qc, op_1)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_2 = op2.apply_layout(qc.layout)
        result = est.run([(qc, op_2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_3 = op.apply_layout(qc2.layout)
        result = est.run([(qc2, op_3)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_4 = op2.apply_layout(qc2.layout)
        result = est.run([(qc2, op_4)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1], rtol=self._rtol)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_run_2qubits(self, backend, abelian_grouping):
        """Test for 2-qubit cases (to check endian)"""
        backend = backend()
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        qc, qc2 = pm.run([qc, qc2])

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        est = BackendEstimatorV2(backend=backend, options=self._options)
        est.options.abelian_grouping = abelian_grouping
        op_1 = op.apply_layout(qc.layout)
        result = est.run([(qc, op_1)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_2 = op.apply_layout(qc2.layout)
        result = est.run([(qc2, op_2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_3 = op2.apply_layout(qc.layout)
        result = est.run([(qc, op_3)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_4 = op2.apply_layout(qc2.layout)
        result = est.run([(qc2, op_4)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_5 = op3.apply_layout(qc.layout)
        result = est.run([(qc, op_5)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1], rtol=self._rtol)

        op_6 = op3.apply_layout(qc2.layout)
        result = est.run([(qc2, op_6)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1], rtol=self._rtol)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_run_errors(self, backend, abelian_grouping):
        """Test for errors"""
        backend = backend()
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = BackendEstimatorV2(backend=backend, options=self._options)
        est.options.abelian_grouping = abelian_grouping
        with self.assertRaises(ValueError):
            est.run([(qc, op2)]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, op, [[1e4]])]).result()
        with self.assertRaises(ValueError):
            est.run([(qc2, op2, [[1, 2]])]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, [op, op2], [[1]])]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, op)], precision=-1).result()
        with self.assertRaises(ValueError):
            est.run([(qc, 1j * op)], precision=0.1).result()
        # precision == 0
        with self.assertRaises(ValueError):
            est.run([(qc, op, None, 0)]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, op)], precision=0).result()
        # precision < 0
        with self.assertRaises(ValueError):
            est.run([(qc, op, None, -1)]).result()
        with self.assertRaises(ValueError):
            est.run([(qc, op)], precision=-1).result()
        with self.subTest("missing []"):
            with self.assertRaisesRegex(ValueError, "An invalid Estimator pub-like was given"):
                _ = est.run((qc, op)).result()

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_run_numpy_params(self, backend, abelian_grouping):
        """Test for numpy array as parameter values"""
        backend = backend()
        qc = QuantumCircuit(2)
        qc.append(real_amplitudes(num_qubits=2, reps=2), [0, 1])
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        qc = pm.run(qc)
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        op = op.apply_layout(qc.layout)
        k = 5
        params_array = self._rng.random((k, qc.num_parameters))
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        statevector_estimator = StatevectorEstimator(seed=123)
        target = statevector_estimator.run([(qc, op, params_list)]).result()

        backend_estimator = BackendEstimatorV2(backend=backend, options=self._options)
        backend_estimator.options.abelian_grouping = abelian_grouping

        with self.subTest("ndarrary"):
            result = backend_estimator.run([(qc, op, params_array)]).result()
            self.assertEqual(result[0].data.evs.shape, (k,))
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs, rtol=self._rtol)

        with self.subTest("list of ndarray"):
            result = backend_estimator.run([(qc, op, params_list_array)]).result()
            self.assertEqual(result[0].data.evs.shape, (k,))
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs, rtol=self._rtol)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_precision(self, backend, abelian_grouping):
        """Test for precision"""
        backend = backend()
        estimator = BackendEstimatorV2(backend=backend, options=self._options)
        estimator.options.abelian_grouping = abelian_grouping
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        psi1 = pm.run(self.psi[0])
        hamiltonian1 = self.hamiltonian[0].apply_layout(psi1.layout)
        theta1 = self.theta[0]
        job = estimator.run([(psi1, hamiltonian1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881], rtol=self._rtol)
        # The result of the second run is the same
        job = estimator.run([(psi1, hamiltonian1, [theta1]), (psi1, hamiltonian1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881], rtol=self._rtol)
        np.testing.assert_allclose(result[1].data.evs, [1.901141473854881], rtol=self._rtol)
        # apply smaller precision value
        job = estimator.run([(psi1, hamiltonian1, [theta1])], precision=self._precision * 0.5)
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.5555572817900956], rtol=self._rtol)

    @combine(backend=BACKENDS, abelian_grouping=[True, False])
    def test_diff_precision(self, backend, abelian_grouping):
        """Test for running different precisions at once"""
        backend = backend()
        estimator = BackendEstimatorV2(backend=backend, options=self._options)
        estimator.options.abelian_grouping = abelian_grouping
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        psi1 = pm.run(self.psi[0])
        hamiltonian1 = self.hamiltonian[0].apply_layout(psi1.layout)
        theta1 = self.theta[0]
        job = estimator.run(
            [(psi1, hamiltonian1, [theta1]), (psi1, hamiltonian1, [theta1], self._precision * 0.8)]
        )
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881], rtol=self._rtol)
        np.testing.assert_allclose(result[1].data.evs, [1.901141473854881], rtol=self._rtol)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    @combine(abelian_grouping=[True, False])
    def test_aer(self, abelian_grouping):
        """Test for Aer simulator"""
        from qiskit_aer import AerSimulator

        backend = AerSimulator()
        seed = 123
        qc = QuantumCircuit(2)
        qc.append(real_amplitudes(num_qubits=2, reps=1), [0, 1])
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        qc = pm.run(qc)
        op = [SparsePauliOp("IX"), SparsePauliOp("YI")]
        shape = (3, 2)
        params_array = self._rng.random(shape + (qc.num_parameters,))
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        statevector_estimator = StatevectorEstimator(seed=seed)
        target = statevector_estimator.run([(qc, op, params_list)]).result()

        backend_estimator = BackendEstimatorV2(backend=backend, options=self._options)
        backend_estimator.options.abelian_grouping = abelian_grouping

        with self.subTest("ndarrary"):
            result = backend_estimator.run([(qc, op, params_array)]).result()
            self.assertEqual(result[0].data.evs.shape, shape)
            np.testing.assert_allclose(
                result[0].data.evs, target[0].data.evs, rtol=self._rtol, atol=1e-1
            )

        with self.subTest("list of ndarray"):
            result = backend_estimator.run([(qc, op, params_list_array)]).result()
            self.assertEqual(result[0].data.evs.shape, shape)
            np.testing.assert_allclose(
                result[0].data.evs, target[0].data.evs, rtol=self._rtol, atol=1e-1
            )

    def test_job_size_limit_backend_v2(self):
        """Test BackendEstimatorV2 respects job size limit"""

        class FakeBackendLimitedCircuits(GenericBackendV2):
            """Generic backend V2 with job size limit."""

            @property
            def max_circuits(self):
                return 1

        backend = FakeBackendLimitedCircuits(num_qubits=5)
        qc = QuantumCircuit(2)
        qc.append(real_amplitudes(num_qubits=2, reps=2), [0, 1])
        # Note: two qubit-wise commuting groups
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        param_list = self._rng.random(qc.num_parameters).tolist()
        estimator = BackendEstimatorV2(backend=backend)
        with patch.object(backend, "run") as run_mock:
            estimator.run([(qc, op, param_list)] * k).result()
        self.assertEqual(run_mock.call_count, 10)

    def test_iter_pub(self):
        """test for an iterable of pubs"""
        backend = BasicSimulator()
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        circuit = pm.run(circuit)
        estimator = BackendEstimatorV2(backend=backend, options=self._options)
        observable = self.observable.apply_layout(circuit.layout)
        result = estimator.run(iter([(circuit, observable), (circuit, observable)])).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.284366511861733], rtol=self._rtol)
        np.testing.assert_allclose(result[1].data.evs, [-1.284366511861733], rtol=self._rtol)

    def test_metadata(self):
        """Test for metadata"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.metadata = {"a": 1}
        backend = BasicSimulator()
        estimator = BackendEstimatorV2(backend=backend)
        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        qc, qc2 = pm.run([qc, qc2])
        op = SparsePauliOp("ZZ").apply_layout(qc.layout)
        op2 = SparsePauliOp("ZZ").apply_layout(qc2.layout)
        result = estimator.run([(qc, op), (qc2, op2)], precision=0.1).result()

        self.assertEqual(len(result), 2)
        self.assertEqual(result.metadata, {"version": 2})
        self.assertEqual(
            result[0].metadata,
            {"target_precision": 0.1, "shots": 100, "circuit_metadata": qc.metadata},
        )
        self.assertEqual(
            result[1].metadata,
            {"target_precision": 0.1, "shots": 100, "circuit_metadata": qc2.metadata},
        )


if __name__ == "__main__":
    unittest.main()
