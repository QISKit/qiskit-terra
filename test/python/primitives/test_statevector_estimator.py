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

"""Tests for Estimator."""

import unittest
from test import QiskitTestCase

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.primitives.containers.bindings_array import BindingsArray
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.quantum_info import SparsePauliOp


class TestStatevectorEstimator(QiskitTestCase):
    """Test Estimator"""

    def setUp(self):
        super().setUp()
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

    def test_estimator_run(self):
        """Test Estimator.run()"""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta
        estimator = StatevectorEstimator()

        # Specify the circuit and observable by indices.
        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        job = estimator.run([(psi1, hamiltonian1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.5555572817900956])

        # Objects can be passed instead of indices.
        # Note that passing objects has an overhead
        # since the corresponding indices need to be searched.
        # User can append a circuit and observable.
        # calculate [ <psi2(theta2)|H1|psi2(theta2)> ]
        result2 = estimator.run([(psi2, hamiltonian1, theta2)]).result()
        np.testing.assert_allclose(result2[0].data.evs, [2.97797666])

        # calculate [ <psi1(theta1)|H2|psi1(theta1)>, <psi1(theta1)|H3|psi1(theta1)> ]
        result3 = estimator.run([(psi1, [hamiltonian2, hamiltonian3], theta1)]).result()
        np.testing.assert_allclose(result3[0].data.evs, [-0.551653, 0.07535239])

        # calculate [ [<psi1(theta1)|H1|psi1(theta1)>,
        #              <psi1(theta3)|H3|psi1(theta3)>],
        #             [<psi2(theta2)|H2|psi2(theta2)>] ]
        result4 = estimator.run(
            [(psi1, [hamiltonian1, hamiltonian3], [theta1, theta3]), (psi2, hamiltonian2, theta2)]
        ).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318])
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238])

    def test_estimator_with_pub(self):
        """Test estimator with explicit EstimatorPubs."""
        psi1, psi2 = self.psi
        hamiltonian1, hamiltonian2, hamiltonian3 = self.hamiltonian
        theta1, theta2, theta3 = self.theta

        obs1 = ObservablesArray.coerce([hamiltonian1, hamiltonian3])
        bind1 = BindingsArray.coerce({tuple(psi1.parameters): [theta1, theta3]})
        pub1 = EstimatorPub(psi1, obs1, bind1)
        obs2 = ObservablesArray.coerce(hamiltonian2)
        bind2 = BindingsArray.coerce({tuple(psi2.parameters): theta2})
        pub2 = EstimatorPub(psi2, obs2, bind2)

        estimator = StatevectorEstimator()
        result4 = estimator.run([pub1, pub2]).result()
        np.testing.assert_allclose(result4[0].data.evs, [1.55555728, -1.08766318])
        np.testing.assert_allclose(result4[1].data.evs, [0.17849238])

    def test_estimator_run_no_params(self):
        """test for estimator without parameters"""
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        est = StatevectorEstimator()
        result = est.run([(circuit, self.observable)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.284366511861733])

    def test_run_single_circuit_observable(self):
        """Test for single circuit and single observable case."""
        est = StatevectorEstimator()

        with self.subTest("No parameter"):
            qc = QuantumCircuit(1)
            qc.x(0)
            op = SparsePauliOp("Z")
            param_vals = [None, [], [[]], np.array([]), np.array([[]]), [np.array([])]]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target)
                self.assertEqual(result[0].metadata["target_precision"], 0)

        with self.subTest("One parameter"):
            param = Parameter("x")
            qc = QuantumCircuit(1)
            qc.ry(param, 0)
            op = SparsePauliOp("Z")
            param_vals = [
                [np.pi],
                np.array([np.pi]),
            ]
            target = [-1]
            for val in param_vals:
                self.subTest(f"{val}")
                result = est.run([(qc, op, val)]).result()
                np.testing.assert_allclose(result[0].data.evs, target)
                self.assertEqual(result[0].metadata["target_precision"], 0)

        with self.subTest("More than one parameter"):
            qc = self.psi[0]
            op = self.hamiltonian[0]
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
                np.testing.assert_allclose(result[0].data.evs, target)
                self.assertEqual(result[0].metadata["target_precision"], 0)

    def test_run_1qubit(self):
        """Test for 1-qubit cases"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc2.x(0)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("Z", 1)])

        est = StatevectorEstimator()
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc, op2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc2, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc2, op2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1])

    def test_run_2qubits(self):
        """Test for 2-qubit cases (to check endian)"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.x(0)

        op = SparsePauliOp.from_list([("II", 1)])
        op2 = SparsePauliOp.from_list([("ZI", 1)])
        op3 = SparsePauliOp.from_list([("IZ", 1)])

        est = StatevectorEstimator()
        result = est.run([(qc, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc2, op)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc, op2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc2, op2)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc, op3)]).result()
        np.testing.assert_allclose(result[0].data.evs, [1])

        result = est.run([(qc2, op3)]).result()
        np.testing.assert_allclose(result[0].data.evs, [-1])

    def test_run_errors(self):
        """Test for errors"""
        qc = QuantumCircuit(1)
        qc2 = QuantumCircuit(2)

        op = SparsePauliOp.from_list([("I", 1)])
        op2 = SparsePauliOp.from_list([("II", 1)])

        est = StatevectorEstimator()
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
        with self.subTest("missing []"):
            with self.assertRaisesRegex(ValueError, "An invalid Estimator pub-like was given"):
                _ = est.run((qc, op)).result()

    def test_run_numpy_params(self):
        """Test for numpy array as parameter values"""
        qc = QuantumCircuit(2)
        qc.append(real_amplitudes(num_qubits=2, reps=2), [0, 1])
        op = SparsePauliOp.from_list([("IZ", 1), ("XI", 2), ("ZY", -1)])
        k = 5
        rng = np.random.default_rng(12)
        params_array = rng.random((k, qc.num_parameters))
        params_list = params_array.tolist()
        params_list_array = list(params_array)
        estimator = StatevectorEstimator()
        target = estimator.run([(qc, op, params_list)]).result()

        with self.subTest("ndarrary"):
            result = estimator.run([(qc, op, params_array)]).result()
            self.assertEqual(len(result[0].data.evs), k)
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs)

        with self.subTest("list of ndarray"):
            result = estimator.run([(qc, op, params_list_array)]).result()
            self.assertEqual(len(result[0].data.evs), k)
            np.testing.assert_allclose(result[0].data.evs, target[0].data.evs)

    def test_precision_seed(self):
        """Test for precision and seed"""
        estimator = StatevectorEstimator(default_precision=1.0, seed=1)
        psi1 = self.psi[0]
        hamiltonian1 = self.hamiltonian[0]
        theta1 = self.theta[0]
        job = estimator.run([(psi1, hamiltonian1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881])
        # The result of the second run is the same
        job = estimator.run([(psi1, hamiltonian1, [theta1]), (psi1, hamiltonian1, [theta1])])
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.901141473854881])
        np.testing.assert_allclose(result[1].data.evs, [1.901141473854881])
        # precision=0 implies the exact expectation value
        job = estimator.run([(psi1, hamiltonian1, [theta1])], precision=0)
        result = job.result()
        np.testing.assert_allclose(result[0].data.evs, [1.5555572817900956])

    def test_iter_pub(self):
        """test for an iterable of pubs"""
        estimator = StatevectorEstimator()
        circuit = self.ansatz.assign_parameters([0, 1, 1, 2, 3, 5])
        observable = self.observable.apply_layout(circuit.layout)
        result = estimator.run(iter([(circuit, observable), (circuit, observable)])).result()
        np.testing.assert_allclose(result[0].data.evs, [-1.284366511861733])
        np.testing.assert_allclose(result[1].data.evs, [-1.284366511861733])

    def test_metadata(self):
        """Test for metadata"""
        qc = QuantumCircuit(2)
        qc2 = QuantumCircuit(2)
        qc2.metadata = {"a": 1}
        estimator = StatevectorEstimator()
        result = estimator.run([(qc, "ZZ"), (qc2, "ZZ")], precision=0.1).result()

        self.assertEqual(len(result), 2)
        self.assertEqual(result.metadata, {"version": 2})
        self.assertEqual(
            result[0].metadata, {"target_precision": 0.1, "circuit_metadata": qc.metadata}
        )
        self.assertEqual(
            result[1].metadata, {"target_precision": 0.1, "circuit_metadata": qc2.metadata}
        )

    def test_reset(self):
        """Test for circuits with reset."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.reset(0)
        op = SparsePauliOp("ZI")

        seed = 12
        n = 1000
        estimator = StatevectorEstimator(seed=seed)
        with self.subTest("precision=0"):
            result = estimator.run([(qc, [op] * n)]).result()
            # expectation values should be stochastic due to reset for subsystems
            np.testing.assert_allclose(result[0].data.evs.mean(), 0, atol=1e-1)

            result2 = estimator.run([(qc, [op] * n)]).result()
            # expectation values should be reproducible due to seed
            np.testing.assert_allclose(result[0].data.evs, result2[0].data.evs)

        with self.subTest("precision=0.01"):
            precision = 0.01
            result = estimator.run([(qc, [op] * n)], precision=precision).result()
            # expectation values should be stochastic due to reset for subsystems
            np.testing.assert_allclose(result[0].data.evs.mean(), 0, atol=1e-1)

            result2 = estimator.run([(qc, [op] * n)], precision=precision).result()
            # expectation values should be reproducible due to seed
            np.testing.assert_allclose(result[0].data.evs, result2[0].data.evs)


if __name__ == "__main__":
    unittest.main()
