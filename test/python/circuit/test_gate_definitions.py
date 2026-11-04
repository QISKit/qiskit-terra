# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test hardcoded decomposition rules and matrix definitions for standard gates."""

import inspect

import numpy as np
from ddt import ddt, data, idata, unpack

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Operator
from qiskit.circuit import ParameterVector, Gate, ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate
from qiskit.circuit.library import standard_gates
from qiskit.circuit.library import (
    HGate,
    CHGate,
    IGate,
    RGate,
    RXGate,
    CRXGate,
    RYGate,
    CRYGate,
    RZGate,
    CRZGate,
    SGate,
    SdgGate,
    CSwapGate,
    TGate,
    TdgGate,
    U1Gate,
    CU1Gate,
    U2Gate,
    U3Gate,
    CU3Gate,
    XGate,
    CXGate,
    ECRGate,
    CCXGate,
    YGate,
    CYGate,
    ZGate,
    CZGate,
    RYYGate,
    PhaseGate,
    PauliGate,
    UCPauliRotGate,
    CPhaseGate,
    UGate,
    CUGate,
    SXGate,
    SXdgGate,
    CSXGate,
    RVGate,
    XXMinusYYGate,
    FullAdderGate,
    HalfAdderGate,
    ModularAdderGate,
    LinearFunction,
    MultiplierGate,
)
from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib,
)
from test import QiskitTestCase


from .gate_utils import _get_free_params


class TestGateDefinitions(QiskitTestCase):
    """Test the decomposition of a gate in terms of other gates
    yields the equivalent matrix as the hardcoded matrix definition
    up to a global phase."""

    def test_ch_definition(self):  # TODO: expand this to all gates
        """Test ch gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.ch(0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_ccx_definition(self):
        """Test ccx gate matrix and definition."""
        circ = QuantumCircuit(3)
        circ.ccx(0, 1, 2)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_crz_definition(self):
        """Test crz gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.crz(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cry_definition(self):
        """Test cry gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.cry(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_crx_definition(self):
        """Test crx gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.crx(1, 0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cswap_definition(self):
        """Test cswap gate matrix and definition."""
        circ = QuantumCircuit(3)
        circ.cswap(0, 1, 2)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cu1_definition(self):
        """Test cu1 gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.append(CU1Gate(1), [0, 1])
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cu3_definition(self):
        """Test cu3 gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.append(CU3Gate(1, 1, 1), [0, 1])
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_cx_definition(self):
        """Test cx gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_ecr_definition(self):
        """Test ecr gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.ecr(0, 1)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_rv_definition(self):
        """Test R(v) gate to_matrix and definition."""
        qreg = QuantumRegister(1)
        circ = QuantumCircuit(qreg)
        vec = np.array([0.1, 0.2, 0.3], dtype=float)
        circ.rv(*vec, 0)
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_rv_r_equiv(self):
        """Test R(v) gate is equivalent to R gate."""
        theta = np.pi / 5
        phi = np.pi / 3
        rgate = RGate(theta, phi)
        axis = np.array([np.cos(phi), np.sin(phi), 0])  # RGate axis
        rotvec = theta * axis
        rv = RVGate(*rotvec)
        rg_matrix = rgate.to_matrix()
        rv_matrix = rv.to_matrix()
        np.testing.assert_array_max_ulp(rg_matrix.real, rv_matrix.real, 4)
        np.testing.assert_array_max_ulp(rg_matrix.imag, rv_matrix.imag, 4)

    def test_rv_zero(self):
        """Test R(v) gate with zero vector returns identity"""
        rv = RVGate(0, 0, 0)
        self.assertTrue(np.array_equal(rv.to_matrix(), np.array([[1, 0], [0, 1]])))

    def test_xx_minus_yy_definition(self):
        """Test XX-YY gate decomposition."""
        theta, beta = np.random.uniform(-10, 10, size=2)
        gate = XXMinusYYGate(theta, beta)
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1])
        decomposed_circuit = circuit.decompose()
        self.assertTrue(len(decomposed_circuit) > len(circuit))
        self.assertTrue(Operator(circuit).equiv(Operator(decomposed_circuit), atol=1e-7))

    def test_pauligate_repeat(self):
        """Test `repeat` method for `PauliGate`."""
        gate = PauliGate("XYZ")
        operator = Operator(gate)
        self.assertTrue(np.allclose(Operator(gate.repeat(2)), operator @ operator))

    def test_ucpaulirotgate_repeat(self):
        """Test `repeat` method for `UCPauliRotGate`."""
        gate = UCPauliRotGate([0.3, 0.5], "X")
        operator = Operator(gate)
        self.assertTrue(np.allclose(Operator(gate.repeat(2)), operator @ operator))

    def test_linear_function_definition(self):
        """Test LinearFunction gate matrix and definition."""
        circ = QuantumCircuit(3)
        circ.append(LinearFunction([[1, 1], [0, 1]]), [0, 2])
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_full_adder_definition(self):
        """Test FullAdder gate matrix and definition."""
        circ = QuantumCircuit(4)
        circ.append(FullAdderGate(1), [0, 1, 2, 3])
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_half_adder_definition(self):
        """Test HalfAdder gate matrix and definition."""
        circ = QuantumCircuit(3)
        circ.append(HalfAdderGate(1), [0, 1, 2])
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_modular_adder_definition(self):
        """Test ModularAdder gate matrix and definition."""
        circ = QuantumCircuit(2)
        circ.append(ModularAdderGate(1), [0, 1])
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))

    def test_multiplier_gate_definition(self):
        """Test Multiplier gate matrix and definition."""
        circ = QuantumCircuit(4)
        circ.append(MultiplierGate(1), [0, 1, 2, 3])
        decomposed_circ = circ.decompose()
        self.assertTrue(Operator(circ).equiv(Operator(decomposed_circ)))


@ddt
class TestStandardGates(QiskitTestCase):
    """Standard Extension Test."""

    @unpack
    @data(
        *inspect.getmembers(
            standard_gates,
            predicate=lambda value: (inspect.isclass(value) and issubclass(value, Gate)),
        )
    )
    def test_definition_parameters(self, class_name, gate_class):
        """Verify definitions from standard library include correct parameters."""

        free_params = _get_free_params(gate_class)
        n_params = len(free_params)
        param_vector = ParameterVector("th", n_params)

        if class_name in ("MCPhaseGate", "MCU1Gate"):
            param_vector = param_vector[:-1]
            gate = gate_class(*param_vector, num_ctrl_qubits=2)
        elif class_name == "MCXGate":
            num_ctrl_qubits = 2
            param_vector = param_vector[:-1]
            gate = gate_class(num_ctrl_qubits, *param_vector)
        elif class_name in ("MCXGrayCode", "MCXRecursive", "MCXVChain"):
            num_ctrl_qubits = 2
            param_vector = param_vector[:-1]
            with self.assertWarns(DeprecationWarning):
                gate = gate_class(num_ctrl_qubits, *param_vector)
        elif class_name == "MSGate":
            num_qubits = 2
            param_vector = param_vector[:-1]
            gate = gate_class(num_qubits, *param_vector)
        else:
            gate = gate_class(*param_vector)

        if gate.definition is not None:
            self.assertEqual(gate.definition.parameters, set(param_vector))

    @unpack
    @data(
        *inspect.getmembers(
            standard_gates,
            predicate=lambda value: (inspect.isclass(value) and issubclass(value, Gate)),
        )
    )
    def test_inverse(self, class_name, gate_class):
        """Verify self-inverse pair yield identity for all standard gates."""

        free_params = _get_free_params(gate_class)
        n_params = len(free_params)
        float_vector = [0.1 + 0.1 * i for i in range(n_params)]

        if class_name in ("MCPhaseGate", "MCU1Gate"):
            float_vector = float_vector[:-1]
            gate = gate_class(*float_vector, num_ctrl_qubits=2)
        elif class_name == "MCXGate":
            num_ctrl_qubits = 3
            float_vector = float_vector[:-1]
            gate = gate_class(num_ctrl_qubits, *float_vector)
        elif class_name in ("MCXGrayCode", "MCXRecursive", "MCXVChain"):
            num_ctrl_qubits = 3
            float_vector = float_vector[:-1]
            with self.assertWarns(DeprecationWarning):
                gate = gate_class(num_ctrl_qubits, *float_vector)
        elif class_name == "PauliGate":
            pauli_string = "IXYZ"
            gate = gate_class(pauli_string)
        else:
            gate = gate_class(*float_vector)

        from qiskit.quantum_info.operators.predicates import is_identity_matrix

        self.assertTrue(is_identity_matrix(Operator(gate).dot(gate.inverse()).data))

        if gate.definition is not None:
            self.assertTrue(is_identity_matrix(Operator(gate).dot(gate.definition.inverse()).data))
            self.assertTrue(is_identity_matrix(Operator(gate).dot(gate.inverse().definition).data))


@ddt
class TestGateEquivalenceEqual(QiskitTestCase):
    """Test the decomposition of a gate in terms of other gates
    yields the same matrix as the hardcoded matrix definition."""

    class_list = (
        SingletonGate.__subclasses__()
        + SingletonControlledGate.__subclasses__()
        + Gate.__subclasses__()
        + ControlledGate.__subclasses__()
    )
    exclude = {
        "ControlledGate",
        "DiagonalGate",
        "UCGate",
        "MCGupDiag",
        "MCU1Gate",
        "UnitaryGate",
        "HamiltonianGate",
        "MCPhaseGate",
        "UCPauliRotGate",
        "SingleQubitUnitary",
        "MCXGate",
        "MCMTGate",
        "VariadicZeroParamGate",
        "StatePreparation",
        "UniformSuperpositionGate",
        "LinearFunction",
        "PermutationGate",
        "Commuting2qBlock",
        "PauliEvolutionGate",
        "PauliProductRotationGate",
        "SingletonGate",
        "SingletonControlledGate",
        "_U0Gate",
        "_DefinedGate",
        "_SingletonGateOverrides",
        "_SingletonControlledGateOverrides",
        "QFTGate",
        "ModularAdderGate",
        "HalfAdderGate",
        "FullAdderGate",
        "MultiplierGate",
        "GraphStateGate",
        "AndGate",
        "OrGate",
        "BitwiseXorGate",
        "InnerProductGate",
        "IntegerComparatorGate",
        "PolynomialPauliRotationsGate",
        "PiecewiseLinearPauliRotationsGate",
        "PiecewisePolynomialPauliRotationsGate",
        "PiecewiseChebyshevGate",
        "ExactReciprocalGate",
        "LinearPauliRotationsGate",
        "LinearAmplitudeFunctionGate",
        "WeightedSumGate",
    }

    # Amazingly, Python's scoping rules for class bodies means that this is the closest we can get
    # to a "natural" comprehension or functional iterable definition:
    #   https://docs.python.org/3/reference/executionmodel.html#resolution-of-names
    @idata(filter(lambda x, exclude=exclude: x.__name__ not in exclude, class_list))
    def test_equivalence_phase(self, gate_class):
        """Test that the equivalent circuits from the equivalency_library
        have equal matrix representations"""
        n_params = len(_get_free_params(gate_class))
        params = [0.1 * i for i in range(1, n_params + 1)]
        if gate_class.__name__ == "RXXGate":
            params = [np.pi / 2]
        if gate_class.__name__ in ["MSGate"]:
            params[0] = 2
        if gate_class.__name__ in ["PauliGate"]:
            params = ["IXYZ"]
        if gate_class.__name__ in ["BooleanExpression", "BitFlipOracleGate", "PhaseOracleGate"]:
            params = ["x | y"]

        gate = gate_class(*params)
        equiv_lib_list = std_eqlib.get_entry(gate)
        for ieq, equivalency in enumerate(equiv_lib_list):
            with self.subTest(msg=gate.name + "_" + str(ieq)):
                op1 = Operator(gate)
                op2 = Operator(equivalency)
                msg = (
                    f"Equivalence entry from '{gate.name}' to:\n"
                    f"{equivalency.draw('text')!s}\nfailed"
                )
                self.assertEqual(op1, op2, msg)


@ddt
class TestStandardEquivalenceLibrary(QiskitTestCase):
    """Standard Extension Test."""

    @data(
        HGate,
        CHGate,
        IGate,
        RGate,
        RXGate,
        CRXGate,
        RYGate,
        CRYGate,
        RZGate,
        CRZGate,
        SGate,
        SdgGate,
        CSwapGate,
        TGate,
        TdgGate,
        U1Gate,
        CU1Gate,
        U2Gate,
        U3Gate,
        CU3Gate,
        XGate,
        CXGate,
        ECRGate,
        CCXGate,
        YGate,
        CYGate,
        ZGate,
        CZGate,
        RYYGate,
        PhaseGate,
        CPhaseGate,
        UGate,
        CUGate,
        SXGate,
        SXdgGate,
        CSXGate,
    )
    def test_definition_parameters(self, gate_class):
        """Verify decompositions from standard equivalence library match definitions."""
        n_params = len(_get_free_params(gate_class))
        param_vector = ParameterVector("th", n_params)
        float_vector = [0.1 * i for i in range(n_params)]

        param_gate = gate_class(*param_vector)
        float_gate = gate_class(*float_vector)

        param_entry = std_eqlib.get_entry(param_gate)
        float_entry = std_eqlib.get_entry(float_gate)

        if not param_gate.definition or not param_gate.definition.data:
            return

        self.assertGreaterEqual(len(param_entry), 1)
        self.assertGreaterEqual(len(float_entry), 1)

        param_qc = QuantumCircuit(param_gate.num_qubits)
        float_qc = QuantumCircuit(float_gate.num_qubits)

        param_qc.append(param_gate, param_qc.qubits)
        float_qc.append(float_gate, float_qc.qubits)

        self.assertTrue(any(equiv == param_qc.decompose() for equiv in param_entry))
        self.assertTrue(any(equiv == float_qc.decompose() for equiv in float_entry))
