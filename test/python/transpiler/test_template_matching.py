# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the TemplateOptimization pass."""

import unittest

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator
from qiskit.circuit.library.templates.nct import template_nct_2a_2, template_nct_5a_3
from qiskit.circuit.library.templates.clifford import (
    clifford_2_1,
    clifford_2_2,
    clifford_2_3,
    clifford_2_4,
    clifford_3_1,
    clifford_4_1,
    clifford_4_2,
)
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.converters.circuit_to_dagdependency import circuit_to_dagdependency
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import TemplateOptimization
from qiskit.circuit.library.templates import rzx
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.utils import optionals
from test.python.quantum_info.operators.symplectic.test_clifford import (
    random_clifford_circuit,
)
from test import QiskitTestCase


def _ry_to_rz_template_pass(parameter: Parameter = None, extra_costs=None):
    """Create a simple pass manager that runs a template optimization with a single transformation.
    It turns ``RX(pi/2).RY(parameter).RX(-pi/2)`` into the equivalent virtual ``RZ`` rotation, where
    if ``parameter`` is given, it will be the instance used in the template."""
    if parameter is None:
        parameter = Parameter("_ry_rz_template_inner")
    template = QuantumCircuit(1)
    template.rx(-np.pi / 2, 0)
    template.ry(parameter, 0)
    template.rx(np.pi / 2, 0)
    template.rz(-parameter, 0)

    costs = {"rx": 16, "ry": 16, "rz": 0}
    if extra_costs is not None:
        costs.update(extra_costs)

    return PassManager(TemplateOptimization([template], user_cost_dict=costs))


@unittest.skipUnless(optionals.HAS_SYMPY, "sympy required for template optimization")
class TestTemplateMatching(QiskitTestCase):
    """Test the TemplateOptimization pass."""

    def test_pass_cx_cancellation_no_template_given(self):
        """
        Check the cancellation of CX gates for the apply of the three basic
        template x-x, cx-cx. ccx-ccx.
        """
        qr = QuantumRegister(3)
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])

        pass_manager = PassManager()
        pass_manager.append(TemplateOptimization())
        circuit_in_opt = pass_manager.run(circuit_in)

        circuit_out = QuantumCircuit(qr)
        circuit_out.h(qr[0])
        circuit_out.h(qr[0])

        self.assertEqual(circuit_in_opt, circuit_out)

    def test_pass_cx_cancellation_own_template(self):
        """
        Check the cancellation of CX gates for the apply of a self made template cx-cx.
        """
        qr = QuantumRegister(2, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])
        dag_in = circuit_to_dag(circuit_in)

        qrt = QuantumRegister(2, "qrc")
        qct = QuantumCircuit(qrt)
        qct.cx(0, 1)
        qct.cx(0, 1)

        template_list = [qct]
        pass_ = TemplateOptimization(template_list)
        dag_opt = pass_.run(dag_in)

        circuit_expected = QuantumCircuit(qr)
        circuit_expected.h(qr[0])
        circuit_expected.h(qr[0])
        dag_expected = circuit_to_dag(circuit_expected)

        self.assertEqual(dag_opt, dag_expected)

    def test_pass_cx_cancellation_template_from_library(self):
        """
        Check the cancellation of CX gates for the apply of the library template cx-cx (2a_2).
        """
        qr = QuantumRegister(2, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])
        dag_in = circuit_to_dag(circuit_in)

        template_list = [template_nct_2a_2()]
        pass_ = TemplateOptimization(template_list)
        dag_opt = pass_.run(dag_in)

        circuit_expected = QuantumCircuit(qr)
        circuit_expected.h(qr[0])
        circuit_expected.h(qr[0])
        dag_expected = circuit_to_dag(circuit_expected)

        self.assertEqual(dag_opt, dag_expected)

    def test_pass_template_nct_5a(self):
        """
        Verify the result of template matching and substitution with the template 5a_3.
        q_0: ───────■─────────■────■──
                  ┌─┴─┐     ┌─┴─┐  │
        q_1: ──■──┤ X ├──■──┤ X ├──┼──
             ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
        q_2: ┤ X ├─────┤ X ├─────┤ X ├
             └───┘     └───┘     └───┘
        The circuit before optimization is:
              ┌───┐               ┌───┐
        qr_0: ┤ X ├───────────────┤ X ├─────
              └─┬─┘     ┌───┐┌───┐└─┬─┘
        qr_1: ──┼────■──┤ X ├┤ Z ├──┼────■──
                │    │  └─┬─┘└───┘  │    │
        qr_2: ──┼────┼────■────■────■────┼──
                │    │  ┌───┐┌─┴─┐  │    │
        qr_3: ──■────┼──┤ H ├┤ X ├──■────┼──
                │  ┌─┴─┐└───┘└───┘     ┌─┴─┐
        qr_4: ──■──┤ X ├───────────────┤ X ├
                   └───┘               └───┘

        The match is given by [0,1][1,2][2,7], after substitution the circuit becomes:

              ┌───┐               ┌───┐
        qr_0: ┤ X ├───────────────┤ X ├
              └─┬─┘     ┌───┐┌───┐└─┬─┘
        qr_1: ──┼───────┤ X ├┤ Z ├──┼──
                │       └─┬─┘└───┘  │
        qr_2: ──┼────■────■────■────■──
                │    │  ┌───┐┌─┴─┐  │
        qr_3: ──■────┼──┤ H ├┤ X ├──■──
                │  ┌─┴─┐└───┘└───┘
        qr_4: ──■──┤ X ├───────────────
                   └───┘
        """
        qr = QuantumRegister(5, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.ccx(qr[3], qr[4], qr[0])
        circuit_in.cx(qr[1], qr[4])
        circuit_in.cx(qr[2], qr[1])
        circuit_in.h(qr[3])
        circuit_in.z(qr[1])
        circuit_in.cx(qr[2], qr[3])
        circuit_in.ccx(qr[2], qr[3], qr[0])
        circuit_in.cx(qr[1], qr[4])
        dag_in = circuit_to_dag(circuit_in)

        template_list = [template_nct_5a_3()]
        pass_ = TemplateOptimization(template_list)
        dag_opt = pass_.run(dag_in)

        # note: cx(2, 1) commutes both with ccx(3, 4, 0) and with cx(2, 4),
        # so there is no real difference with the circuit drawn on the picture above.
        circuit_expected = QuantumCircuit(qr)
        circuit_expected.cx(qr[2], qr[1])
        circuit_expected.ccx(qr[3], qr[4], qr[0])
        circuit_expected.cx(qr[2], qr[4])
        circuit_expected.z(qr[1])
        circuit_expected.h(qr[3])
        circuit_expected.cx(qr[2], qr[3])
        circuit_expected.ccx(qr[2], qr[3], qr[0])

        dag_expected = circuit_to_dag(circuit_expected)

        self.assertEqual(dag_opt, dag_expected)

    def test_pass_template_wrong_type(self):
        """
        If a template is not equivalent to the identity, it raises an error.
        """
        qr = QuantumRegister(2, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.h(qr[0])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[0], qr[1])
        circuit_in.cx(qr[1], qr[0])
        circuit_in.cx(qr[1], qr[0])
        dag_in = circuit_to_dag(circuit_in)

        qrt = QuantumRegister(2, "qrc")
        qct = QuantumCircuit(qrt)
        qct.cx(0, 1)
        qct.x(0)
        qct.h(1)

        template_list = [qct]
        pass_ = TemplateOptimization(template_list)

        self.assertRaises(TranspilerError, pass_.run, dag_in)

    def test_accept_dagdependency(self):
        """
        Check that users can supply DAGDependency in the template list.
        """
        circuit_in = QuantumCircuit(2)
        circuit_in.cx(0, 1)
        circuit_in.cx(0, 1)

        templates = [circuit_to_dagdependency(circuit_in)]

        pass_ = TemplateOptimization(template_list=templates)
        circuit_out = PassManager(pass_).run(circuit_in)

        # these are NOT equal if template optimization works
        self.assertNotEqual(circuit_in, circuit_out)

        # however these are equivalent if the operators are the same
        self.assertTrue(Operator(circuit_in).equiv(circuit_out))

    def test_parametric_template(self):
        """
        Check matching where template has parameters.
             ┌───────────┐                  ┌────────┐
        q_0: ┤ P(-1.0*β) ├──■────────────■──┤0       ├
             ├───────────┤┌─┴─┐┌──────┐┌─┴─┐│  CU(2β)│
        q_1: ┤ P(-1.0*β) ├┤ X ├┤ P(β) ├┤ X ├┤1       ├
             └───────────┘└───┘└──────┘└───┘└────────┘
        First test try match on
             ┌───────┐
        q_0: ┤ P(-2) ├──■────────────■─────────────────────────────
             ├───────┤┌─┴─┐┌──────┐┌─┴─┐┌───────┐
        q_1: ┤ P(-2) ├┤ X ├┤ P(2) ├┤ X ├┤ P(-3) ├──■────────────■──
             ├───────┤└───┘└──────┘└───┘└───────┘┌─┴─┐┌──────┐┌─┴─┐
        q_2: ┤ P(-3) ├───────────────────────────┤ X ├┤ P(3) ├┤ X ├
             └───────┘                           └───┘└──────┘└───┘
        Second test try match on
             ┌───────┐
        q_0: ┤ P(-2) ├──■────────────■────────────────────────────
             ├───────┤┌─┴─┐┌──────┐┌─┴─┐┌──────┐
        q_1: ┤ P(-2) ├┤ X ├┤ P(2) ├┤ X ├┤ P(3) ├──■────────────■──
             └┬──────┤└───┘└──────┘└───┘└──────┘┌─┴─┐┌──────┐┌─┴─┐
        q_2: ─┤ P(3) ├──────────────────────────┤ X ├┤ P(3) ├┤ X ├
              └──────┘                          └───┘└──────┘└───┘
        """

        beta = Parameter("β")
        template = QuantumCircuit(2)
        template.p(-beta, 0)
        template.p(-beta, 1)
        template.cx(0, 1)
        template.p(beta, 1)
        template.cx(0, 1)
        template.cu(0, 2.0 * beta, 0, 0, 0, 1)

        def count_cx(qc):
            """Counts the number of CX gates for testing."""
            return qc.count_ops().get("cx", 0)

        circuit_in = QuantumCircuit(3)
        circuit_in.p(-2, 0)
        circuit_in.p(-2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(-3, 1)
        circuit_in.p(-3, 2)
        circuit_in.cx(1, 2)
        circuit_in.p(3, 2)
        circuit_in.cx(1, 2)

        pass_ = TemplateOptimization(
            template_list=[template],
            user_cost_dict={"cx": 6, "p": 0, "cu": 8},
        )
        circuit_out = PassManager(pass_).run(circuit_in)

        np.testing.assert_almost_equal(Operator(circuit_out).data[3, 3], np.exp(-4.0j))
        np.testing.assert_almost_equal(Operator(circuit_out).data[7, 7], np.exp(-10.0j))
        self.assertEqual(count_cx(circuit_out), 0)  # Two matches => no CX gates.
        np.testing.assert_almost_equal(Operator(circuit_in).data, Operator(circuit_out).data)

        circuit_in = QuantumCircuit(3)
        circuit_in.p(-2, 0)
        circuit_in.p(-2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(2, 1)
        circuit_in.cx(0, 1)
        circuit_in.p(3, 1)
        circuit_in.p(3, 2)
        circuit_in.cx(1, 2)
        circuit_in.p(3, 2)
        circuit_in.cx(1, 2)

        pass_ = TemplateOptimization(
            template_list=[template],
            user_cost_dict={"cx": 6, "p": 0, "cu": 8},
        )
        circuit_out = PassManager(pass_).run(circuit_in)

        # these are NOT equal if template optimization works
        self.assertNotEqual(circuit_in, circuit_out)

        # however these are equivalent if the operators are the same
        self.assertTrue(Operator(circuit_in).equiv(circuit_out))

    def test_optimizer_does_not_replace_unbound_partial_match(self):
        """
        Test that partial matches with parameters will not raise errors.
        This tests that if parameters are still in the temporary template after
        _attempt_bind then they will not be used.
        """

        beta = Parameter("β")
        template = QuantumCircuit(2)
        template.cx(1, 0)
        template.cx(1, 0)
        template.p(beta, 1)
        template.cu(0, 0, 0, -beta, 0, 1)

        circuit_in = QuantumCircuit(2)
        circuit_in.cx(1, 0)
        circuit_in.cx(1, 0)
        pass_ = TemplateOptimization(
            template_list=[template],
            user_cost_dict={"cx": 6, "p": 0, "cu": 8},
        )

        circuit_out = PassManager(pass_).run(circuit_in)

        # The template optimization should not have replaced anything, because
        # that would require it to leave dummy parameters in place without
        # binding them.
        self.assertEqual(circuit_in, circuit_out)

    def test_unbound_parameters_in_rzx_template(self):
        """
        Test that rzx template ('zz2') functions correctly for a simple
        circuit with an unbound ParameterExpression. This uses the same
        Parameter (theta) as the template, so this also checks that template
        substitution handle this correctly.
        """

        theta = Parameter("ϴ")
        circuit_in = QuantumCircuit(2)
        circuit_in.cx(0, 1)
        circuit_in.p(2 * theta, 1)
        circuit_in.cx(0, 1)

        pass_ = TemplateOptimization(
            template_list=[rzx.rzx_zz2()],
            user_cost_dict={"rzx": 0, "cx": 6, "rz": 0, "sx": 1, "p": 0, "h": 1, "rx": 1, "ry": 1},
        )

        circuit_out = PassManager(pass_).run(circuit_in)

        # these are NOT equal if template optimization works
        self.assertNotEqual(circuit_in, circuit_out)

        # however these are equivalent if the operators are the same
        theta_set = 0.42
        self.assertTrue(
            Operator(circuit_in.assign_parameters({theta: theta_set})).equiv(
                circuit_out.assign_parameters({theta: theta_set})
            )
        )

    def test_two_parameter_template(self):
        """
        Test a two-Parameter template based on rzx.rzx_zz3()

                                ┌───┐┌───────┐┌───┐┌────────────┐»
        q_0: ──■─────────────■──┤ X ├┤ Rz(φ) ├┤ X ├┤ Rz(-1.0*φ) ├»
             ┌─┴─┐┌───────┐┌─┴─┐└─┬─┘└───────┘└─┬─┘└────────────┘»
        q_1: ┤ X ├┤ Rz(θ) ├┤ X ├──■─────────────■────────────────»
             └───┘└───────┘└───┘
        «     ┌─────────┐┌─────────┐┌─────────┐┌───────────┐┌──────────────┐»
        «q_0: ┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├┤ Rx(1.0*φ) ├┤1             ├»
        «     └─────────┘└─────────┘└─────────┘└───────────┘│  Rzx(-1.0*φ) │»
        «q_1: ──────────────────────────────────────────────┤0             ├»
        «                                                   └──────────────┘»
        «      ┌─────────┐  ┌─────────┐┌─────────┐                        »
        «q_0: ─┤ Rz(π/2) ├──┤ Rx(π/2) ├┤ Rz(π/2) ├────────────────────────»
        «     ┌┴─────────┴─┐├─────────┤├─────────┤┌─────────┐┌───────────┐»
        «q_1: ┤ Rz(-1.0*θ) ├┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├┤ Rx(1.0*θ) ├»
        «     └────────────┘└─────────┘└─────────┘└─────────┘└───────────┘»
        «     ┌──────────────┐
        «q_0: ┤0             ├─────────────────────────────────
        «     │  Rzx(-1.0*θ) │┌─────────┐┌─────────┐┌─────────┐
        «q_1: ┤1             ├┤ Rz(π/2) ├┤ Rx(π/2) ├┤ Rz(π/2) ├
        «     └──────────────┘└─────────┘└─────────┘└─────────┘

        correctly template matches into a unique circuit, but that it is
        equivalent to the input circuit when the Parameters are bound to floats
        and checked with Operator equivalence.
        """
        theta = Parameter("θ")
        phi = Parameter("φ")

        template = QuantumCircuit(2)
        template.cx(0, 1)
        template.rz(theta, 1)
        template.cx(0, 1)
        template.cx(1, 0)
        template.rz(phi, 0)
        template.cx(1, 0)
        template.rz(-phi, 0)
        template.rz(np.pi / 2, 0)
        template.rx(np.pi / 2, 0)
        template.rz(np.pi / 2, 0)
        template.rx(phi, 0)
        template.rzx(-phi, 1, 0)
        template.rz(np.pi / 2, 0)
        template.rz(-theta, 1)
        template.rx(np.pi / 2, 0)
        template.rz(np.pi / 2, 1)
        template.rz(np.pi / 2, 0)
        template.rx(np.pi / 2, 1)
        template.rz(np.pi / 2, 1)
        template.rx(theta, 1)
        template.rzx(-theta, 0, 1)
        template.rz(np.pi / 2, 1)
        template.rx(np.pi / 2, 1)
        template.rz(np.pi / 2, 1)

        alpha = Parameter("$\\alpha$")
        beta = Parameter("$\\beta$")

        circuit_in = QuantumCircuit(2)
        circuit_in.cx(0, 1)
        circuit_in.rz(2 * alpha, 1)
        circuit_in.cx(0, 1)
        circuit_in.cx(1, 0)
        circuit_in.rz(3 * beta, 0)
        circuit_in.cx(1, 0)

        pass_ = TemplateOptimization(
            [template],
            user_cost_dict={"cx": 6, "rz": 0, "rx": 1, "rzx": 0},
        )
        circuit_out = PassManager(pass_).run(circuit_in)

        # these are NOT equal if template optimization works
        self.assertNotEqual(circuit_in, circuit_out)

        # however these are equivalent if the operators are the same
        alpha_set = 0.37
        beta_set = 0.42
        self.assertTrue(
            Operator(circuit_in.assign_parameters({alpha: alpha_set, beta: beta_set})).equiv(
                circuit_out.assign_parameters({alpha: alpha_set, beta: beta_set})
            )
        )

    def test_exact_substitution_numeric_parameter(self):
        """Test that a template match produces the expected value for numeric parameters."""
        circuit_in = QuantumCircuit(1)
        circuit_in.rx(-np.pi / 2, 0)
        circuit_in.ry(1.45, 0)
        circuit_in.rx(np.pi / 2, 0)
        circuit_out = _ry_to_rz_template_pass().run(circuit_in)

        expected = QuantumCircuit(1)
        expected.rz(1.45, 0)
        self.assertEqual(circuit_out, expected)

    def test_exact_substitution_symbolic_parameter(self):
        """Test that a template match produces the expected value for numeric parameters."""
        a_circuit = Parameter("a")
        circuit_in = QuantumCircuit(1)
        circuit_in.h(0)
        circuit_in.rx(-np.pi / 2, 0)
        circuit_in.ry(a_circuit, 0)
        circuit_in.rx(np.pi / 2, 0)
        circuit_out = _ry_to_rz_template_pass(extra_costs={"h": 1}).run(circuit_in)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.rz(a_circuit, 0)
        self.assertEqual(circuit_out, expected)

    def test_naming_clash(self):
        """Test that the template matching works and correctly replaces a template if there is a
        naming clash between it and the circuit.  This should include binding a partial match with a
        parameter."""
        # Two instances of parameters with the same name---this is how naming clashes might occur.
        a_template = Parameter("a")
        a_circuit = Parameter("a")
        circuit_in = QuantumCircuit(1)
        circuit_in.h(0)
        circuit_in.rx(-np.pi / 2, 0)
        circuit_in.ry(a_circuit, 0)
        circuit_in.rx(np.pi / 2, 0)
        circuit_out = _ry_to_rz_template_pass(a_template, extra_costs={"h": 1}).run(circuit_in)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.rz(a_circuit, 0)
        self.assertEqual(circuit_out, expected)
        # Ensure that the bound parameter in the output is referentially the same as the one we put
        # in the input circuit..
        self.assertEqual(len(circuit_out.parameters), 1)
        self.assertEqual(circuit_in.parameters[0], a_circuit)
        self.assertEqual(circuit_out.parameters[0], a_circuit)

    def test_naming_clash_in_expression(self):
        """Test that the template matching works and correctly replaces a template if there is a
        naming clash between it and the circuit.  This should include binding a partial match with a
        parameter."""
        a_template = Parameter("a")
        a_circuit = Parameter("a")
        circuit_in = QuantumCircuit(1)
        circuit_in.h(0)
        circuit_in.rx(-np.pi / 2, 0)
        circuit_in.ry(2 * a_circuit, 0)
        circuit_in.rx(np.pi / 2, 0)
        circuit_out = _ry_to_rz_template_pass(a_template, extra_costs={"h": 1}).run(circuit_in)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.rz(2 * a_circuit, 0)
        self.assertEqual(circuit_out, expected)
        # Ensure that the bound parameter in the output is referentially the same as the one we put
        # in the input circuit..
        self.assertEqual(len(circuit_out.parameters), 1)
        self.assertEqual(circuit_in.parameters[0], a_circuit)
        self.assertEqual(circuit_out.parameters[0], a_circuit)

    def test_template_match_with_uninvolved_parameter(self):
        """Test that the template matching algorithm succeeds at matching a circuit that contains an
        unbound parameter that is not involved in the subcircuit that matches."""
        b_circuit = Parameter("b")
        circuit_in = QuantumCircuit(2)
        circuit_in.rz(b_circuit, 0)
        circuit_in.rx(-np.pi / 2, 1)
        circuit_in.ry(1.45, 1)
        circuit_in.rx(np.pi / 2, 1)
        circuit_out = _ry_to_rz_template_pass().run(circuit_in)

        expected = QuantumCircuit(2)
        expected.rz(b_circuit, 0)
        expected.rz(1.45, 1)
        self.assertEqual(circuit_out, expected)

    def test_multiple_numeric_matches_same_template(self):
        """Test that the template matching will change both instances of a partial match within a
        longer circuit."""
        circuit_in = QuantumCircuit(2)
        # Qubit 0
        circuit_in.rx(-np.pi / 2, 0)
        circuit_in.ry(1.32, 0)
        circuit_in.rx(np.pi / 2, 0)
        # Qubit 1
        circuit_in.rx(-np.pi / 2, 1)
        circuit_in.ry(2.54, 1)
        circuit_in.rx(np.pi / 2, 1)
        circuit_out = _ry_to_rz_template_pass().run(circuit_in)

        expected = QuantumCircuit(2)
        expected.rz(1.32, 0)
        expected.rz(2.54, 1)
        self.assertEqual(circuit_out, expected)

    def test_multiple_symbolic_matches_same_template(self):
        """Test that the template matching will change both instances of a partial match within a
        longer circuit."""
        a, b = Parameter("a"), Parameter("b")
        circuit_in = QuantumCircuit(2)
        # Qubit 0
        circuit_in.rx(-np.pi / 2, 0)
        circuit_in.ry(a, 0)
        circuit_in.rx(np.pi / 2, 0)
        # Qubit 1
        circuit_in.rx(-np.pi / 2, 1)
        circuit_in.ry(b, 1)
        circuit_in.rx(np.pi / 2, 1)
        circuit_out = _ry_to_rz_template_pass().run(circuit_in)

        expected = QuantumCircuit(2)
        expected.rz(a, 0)
        expected.rz(b, 1)
        self.assertEqual(circuit_out, expected)

    def test_template_match_multiparameter(self):
        """Test that the template matching works on instructions that take more than one
        parameter."""
        a = Parameter("a")
        b = Parameter("b")
        template = QuantumCircuit(1)
        template.u(0, a, b, 0)
        template.rz(-a - b, 0)

        circuit_in = QuantumCircuit(1)
        circuit_in.u(0, 1.23, 2.45, 0)
        pm = PassManager(TemplateOptimization([template], user_cost_dict={"u": 16, "rz": 0}))
        circuit_out = pm.run(circuit_in)

        expected = QuantumCircuit(1)
        expected.rz(1.23 + 2.45, 0)

        self.assertEqual(circuit_out, expected)

    def test_naming_clash_multiparameter(self):
        """Test that the naming clash prevention mechanism works with instructions that take
        multiple parameters."""
        a_template = Parameter("a")
        b_template = Parameter("b")
        template = QuantumCircuit(1)
        template.u(0, a_template, b_template, 0)
        template.rz(-a_template - b_template, 0)

        a_circuit = Parameter("a")
        b_circuit = Parameter("b")
        circuit_in = QuantumCircuit(1)
        circuit_in.u(0, a_circuit, b_circuit, 0)
        pm = PassManager(TemplateOptimization([template], user_cost_dict={"u": 16, "rz": 0}))
        circuit_out = pm.run(circuit_in)

        expected = QuantumCircuit(1)
        expected.rz(a_circuit + b_circuit, 0)

        self.assertEqual(circuit_out, expected)

    def test_consecutive_templates_apply(self):
        """Test the scenario where one template optimization creates an opportunity for
        another template optimization.

        This is the original circuit:

             ┌───┐
        q_0: ┤ X ├──■───X───────■─
             └─┬─┘┌─┴─┐ │ ┌───┐ │
        q_1: ──■──┤ X ├─X─┤ H ├─■─
                  └───┘   └───┘

        The clifford_4_1 template allows to replace the two CNOTs followed by the SWAP by a
        single CNOT:

        q_0: ──■────────■─
             ┌─┴─┐┌───┐ │
        q_1: ┤ X ├┤ H ├─■─
             └───┘└───┘

        At these point, the clifford_4_2 template allows to replace the circuit by a single
        Hadamard gate:

        q_0: ─────
             ┌───┐
        q_1: ┤ H ├
             └───┘

        The second optimization would not have been possible without the applying the first
        optimization.
        """
        qc = QuantumCircuit(2)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.swap(0, 1)
        qc.h(1)
        qc.cz(0, 1)

        qc_expected = QuantumCircuit(2)
        qc_expected.h(1)

        costs = {"h": 1, "cx": 2, "cz": 2, "swap": 3}

        # Check that consecutively applying both templates leads to the expected circuit.
        qc_opt = TemplateOptimization(
            template_list=[clifford_4_1(), clifford_4_2()], user_cost_dict=costs
        )(qc)
        self.assertEqual(qc_opt, qc_expected)

        # Also check that applying the second template by itself does not do anything.
        qc_non_opt = TemplateOptimization(template_list=[clifford_4_2()], user_cost_dict=costs)(qc)
        self.assertEqual(qc, qc_non_opt)

    def test_consecutive_templates_do_not_apply(self):
        """Test that applying one template optimization does not allow incorrectly
        applying other templates (which could happen if the DagDependency graph is
        not constructed correctly after the optimization).
        """
        template_list = [
            clifford_2_2(),
            clifford_2_3(),
        ]
        pm = PassManager(TemplateOptimization(template_list=template_list))
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(0, 1)
        qc.h(0)
        qc.swap(0, 1)
        qc.h(0)
        qc_opt = pm.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_opt))

    def test_clifford_templates(self):
        """Tests TemplateOptimization pass on several larger examples."""
        template_list = [
            clifford_2_1(),
            clifford_2_2(),
            clifford_2_3(),
            clifford_2_4(),
            clifford_3_1(),
        ]
        pm = PassManager(TemplateOptimization(template_list=template_list))
        for seed in range(10):
            qc = random_clifford_circuit(
                num_qubits=5,
                num_gates=100,
                gates=["x", "y", "z", "h", "s", "sdg", "cx", "cz", "swap"],
                seed=seed,
            )
            qc_opt = pm.run(qc)
            self.assertEqual(Operator(qc), Operator(qc_opt))

    def test_circuit_global_phase_preserved_after_single_and_multiple_template_match(self):
        """Test that circuit global_phase survives template optimization (#14537)."""

        circuit_in = QuantumCircuit(2, global_phase=np.pi / 4)
        circuit_in.cx(0, 1)
        circuit_in.cx(0, 1)

        circuit_in_mult = QuantumCircuit(2, global_phase=np.pi / 3)
        # Two independent pairs of CX gates — the template will match twice.
        circuit_in_mult.cx(0, 1)
        circuit_in_mult.cx(0, 1)
        circuit_in_mult.cx(0, 1)
        circuit_in_mult.cx(0, 1)

        template = QuantumCircuit(2)
        template.cx(0, 1)
        template.cx(0, 1)

        result = TemplateOptimization([template])(circuit_in)

        result_mult = TemplateOptimization([template])(circuit_in_mult)

        with self.subTest(msg="single template match"):
            self.assertAlmostEqual(float(result.global_phase), np.pi / 4)
            self.assertEqual(result.count_ops(), {})

        with self.subTest(msg="multiple template matches"):
            self.assertAlmostEqual(float(result_mult.global_phase), np.pi / 3)
            self.assertEqual(result_mult.count_ops(), {})

    def test_template_nonzero_global_phase_applied_to_circuit(self):
        """Test the template's global phase is respected (#14537)."""
        template = QuantumCircuit(1)
        template.h(0)
        template.s(0)
        template.h(0)
        template.s(0)
        template.h(0)
        template.s(0)
        template.global_phase = -np.pi / 4

        qr = QuantumRegister(1, "qr")
        circuit_in = QuantumCircuit(qr, global_phase=np.pi / 4)
        circuit_in.h(qr[0])
        circuit_in.s(qr[0])
        circuit_in.h(qr[0])
        circuit_in.s(qr[0])

        result = TemplateOptimization([template])(circuit_in)

        self.assertEqual(Operator(circuit_in), Operator(result))

    def test_circuit_and_template_both_have_nonzero_global_phase(self):
        """Test that circuit and template global phases both contribute to the result (#14537)."""
        template = QuantumCircuit(1)
        template.h(0)
        template.s(0)
        template.h(0)
        template.s(0)
        template.h(0)
        template.s(0)
        template.global_phase = -np.pi / 4

        qr = QuantumRegister(1, "qr")
        circuit_in = QuantumCircuit(qr)
        circuit_in.h(qr[0])
        circuit_in.s(qr[0])
        circuit_in.h(qr[0])
        circuit_in.s(qr[0])
        circuit_in.h(qr[0])
        circuit_in.s(qr[0])
        circuit_in.global_phase = np.pi / 3

        result = TemplateOptimization([template])(circuit_in)

        # All gates cancelled; total phase = circuit phase + template compensation
        # = pi/3 + pi/4 = 7*pi/12.
        self.assertAlmostEqual(result.global_phase, 7 * np.pi / 12)
        self.assertEqual(result.count_ops(), {})
        self.assertEqual(Operator(circuit_in), Operator(result))


if __name__ == "__main__":
    unittest.main()
