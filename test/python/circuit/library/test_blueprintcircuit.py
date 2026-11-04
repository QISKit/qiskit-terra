# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the blueprint circuit."""

import math
import unittest
from ddt import ddt, data

from qiskit.circuit import (
    QuantumRegister,
    Parameter,
    QuantumCircuit,
    Gate,
    Instruction,
    CircuitInstruction,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import BlueprintCircuit, XGate, EfficientSU2
from test import QiskitTestCase


class MockBlueprint(BlueprintCircuit):
    """A mock blueprint class."""

    def __init__(self, num_qubits):
        super().__init__(name="mock")
        self.num_qubits = num_qubits

    @property
    def num_qubits(self):
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits):
        self._invalidate()
        self._num_qubits = num_qubits
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]

    def _check_configuration(self, raise_on_failure=True):
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits was not set.")

        if self.num_qubits < 1:
            valid = False
            if raise_on_failure:
                raise ValueError("The number of qubits must at least be 1.")

        return valid

    def _build(self):
        super()._build()

        self.rx(Parameter("angle"), 0)
        self.h(self.qubits)


@ddt
class TestBlueprintCircuit(QiskitTestCase):
    """Test the blueprint circuit."""

    def test_invalidate_rebuild(self):
        """Test that invalidate and build reset and set _data and _parameter_table."""
        with self.assertWarns(DeprecationWarning):
            # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
            # BlueprintCircuit gets removed in 3.0
            mock = MockBlueprint(5)
        mock._build()

        with self.subTest(msg="after building"):
            self.assertGreater(len(mock._data), 0)
            self.assertEqual(mock._data.num_parameters(), 1)

        mock._invalidate()
        with self.subTest(msg="after invalidating"):
            self.assertFalse(mock._is_built)
            self.assertEqual(mock._data.num_parameters(), 0)

        mock._build()
        with self.subTest(msg="after re-building"):
            self.assertGreater(len(mock._data), 0)
            self.assertEqual(mock._data.num_parameters(), 1)

    def test_calling_attributes_works(self):
        """Test that the circuit is constructed when attributes are called."""
        properties = ["data"]
        for prop in properties:
            with self.subTest(prop=prop):
                with self.assertWarns(DeprecationWarning):
                    # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed
                    # when BlueprintCircuit gets removed in 3.0
                    circuit = MockBlueprint(3)
                getattr(circuit, prop)
                self.assertGreater(len(circuit._data), 0)

        methods = [
            "qasm",
            "count_ops",
            "num_connected_components",
            "num_nonlocal_gates",
            "depth",
            "__len__",
            "copy",
            "inverse",
        ]
        for method in methods:
            with self.subTest(method=method):
                with self.assertWarns(DeprecationWarning):
                    # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be
                    # removed when BlueprintCircuit gets removed in 3.0
                    circuit = MockBlueprint(3)
                if method == "qasm":
                    continue  # raises since parameterized circuits produce invalid qasm 2.0.
                getattr(circuit, method)()
                self.assertGreater(len(circuit._data), 0)

        with self.subTest(method="__get__[0]"):
            with self.assertWarns(DeprecationWarning):
                # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
                # BlueprintCircuit gets removed in 3.0
                circuit = MockBlueprint(3)
            _ = circuit[2]
            self.assertGreater(len(circuit._data), 0)

    def test_compose_works(self):
        """Test that the circuit is constructed when compose is called."""
        qc = QuantumCircuit(3)
        qc.x([0, 1, 2])
        with self.assertWarns(DeprecationWarning):
            # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
            # BlueprintCircuit gets removed in 3.0
            circuit = MockBlueprint(3)
        circuit.compose(qc, inplace=True)

        reference = QuantumCircuit(3)
        reference.rx(list(circuit.parameters)[0], 0)
        reference.h([0, 1, 2])
        reference.x([0, 1, 2])

        self.assertEqual(reference, circuit)

    def test_compose_respects_inline_captures(self):
        """BlueprintCircuit.compose should forward inline_captures to QuantumCircuit.compose."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Uint(8))
        d = expr.Stretch.new("d")

        class VarBlueprint(BlueprintCircuit):
            def _check_configuration(self, raise_on_failure=True):
                return True

            def _build(self):
                super()._build()
                self.add_input(a)
                self.add_input(b)
                self.add_var(c, 255)
                self.add_stretch(d)
                self.store(a, expr.logic_or(a, b))

        with self.assertWarns(DeprecationWarning):
            blueprint = VarBlueprint()

        other = QuantumCircuit(captures=[a, b, c, d])
        other.store(c, 254)
        other.store(b, expr.logic_or(a, b))

        composed = blueprint.compose(other, inline_captures=True)

        expected = QuantumCircuit(inputs=[a, b])
        expected.add_var(c, 255)
        expected.add_stretch(d)
        expected.store(a, expr.logic_or(a, b))
        expected.store(c, 254)
        expected.store(b, expr.logic_or(a, b))

        self.assertEqual(composed, expected)

    @data("gate", "instruction")
    def test_to_gate_and_instruction(self, method):
        """Test calling to_gate and to_instruction works without calling _build first."""
        with self.assertWarns(DeprecationWarning):
            # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
            # BlueprintCircuit gets removed in 3.0
            circuit = MockBlueprint(2)

        if method == "gate":
            gate = circuit.to_gate()
            self.assertIsInstance(gate, Gate)
        else:
            gate = circuit.to_instruction()
            self.assertIsInstance(gate, Instruction)

    def test_build_before_appends(self):
        """Test that both forms of direct append (public and semi-public) function correctly."""

        class DummyBlueprint(BlueprintCircuit):
            """Dummy circuit."""

            def _check_configuration(self, raise_on_failure=True):
                return True

            def _build(self):
                super()._build()
                self.z(0)

        expected = QuantumCircuit(2)
        expected.z(0)
        expected.x(0)

        qr = QuantumRegister(2, "q")
        with self.assertWarns(DeprecationWarning):
            # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
            # BlueprintCircuit gets removed in 3.0
            mock = DummyBlueprint()
        mock.add_register(qr)
        mock.append(XGate(), [qr[0]], [])
        self.assertEqual(expected, mock)

        with self.assertWarns(DeprecationWarning):
            # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
            # BlueprintCircuit gets removed in 3.0
            mock = DummyBlueprint()
        mock.add_register(qr)
        mock._append(CircuitInstruction(XGate(), (qr[0],), ()))
        self.assertEqual(expected, mock)

    def test_global_phase_copied(self):
        """Test that a global-phase parameter is correctly propagated through."""

        class DummyBlueprint(BlueprintCircuit):
            """Dummy circuit."""

            def _check_configuration(self, raise_on_failure=True):
                return True

            def _build(self):
                # We don't need to do anything, we just need `_build` to be non-abstract.

                return super()._build()

        with self.assertWarns(DeprecationWarning):
            # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
            # BlueprintCircuit gets removed in 3.0
            base = DummyBlueprint()
        base.global_phase = math.pi / 2

        self.assertEqual(base.copy_empty_like().global_phase, math.pi / 2)
        self.assertEqual(base.copy().global_phase, math.pi / 2)

        # Verify that a parametric global phase can be assigned after the copy.
        a = Parameter("a")
        with self.assertWarns(DeprecationWarning):
            # Subclassing BlueprintCircuit is deprecated in 2.0 and the full test can be removed when
            # BlueprintCircuit gets removed in 3.0
            parametric = DummyBlueprint()
        parametric.global_phase = a

        self.assertEqual(
            parametric.copy_empty_like().assign_parameters({a: math.pi / 2}).global_phase,
            math.pi / 2,
        )
        self.assertEqual(
            parametric.copy().assign_parameters({a: math.pi / 2}).global_phase,
            math.pi / 2,
        )

    def test_copy_empty_like(self):
        """Test copying empty-like.

        Regression test of #13535.
        """
        with self.assertWarns(DeprecationWarning):
            # Use the function qiskit.circuit.library.efficient_su2 instead.
            circuit = EfficientSU2(2)
        circuit.global_phase = -2
        circuit.metadata = {"my_legacy": "i was a blueprintcircuit once"}

        cpy = circuit.copy_empty_like()
        cpy.x(0)  # if it were a BlueprintCircuit, this would trigger the building

        expected = QuantumCircuit(
            circuit.num_qubits, global_phase=circuit.global_phase, metadata=circuit.metadata
        )
        expected.x(0)

        self.assertEqual(cpy, expected)
        self.assertNotIsInstance(cpy, BlueprintCircuit)

    def test_copy_empty_like_post_build(self):
        """Test copying empty-like after building the circuit."""
        with self.assertWarns(DeprecationWarning):
            # Use the function qiskit.circuit.library.efficient_su2 instead.
            circuit = EfficientSU2(2)
        num_params = circuit.num_parameters  # trigger building the circuit

        cpy = circuit.copy_empty_like()
        cpy_num_params = cpy.num_parameters  # if it were a BP circuit, this would trigger the build

        circuit.num_qubits = 3  # change the original circuit

        expected = QuantumCircuit(2)  # we still expect an empty 2-qubit circuit
        self.assertEqual(cpy, expected)
        self.assertEqual(cpy_num_params, 0)
        self.assertEqual(num_params, 16)


if __name__ == "__main__":
    unittest.main()
