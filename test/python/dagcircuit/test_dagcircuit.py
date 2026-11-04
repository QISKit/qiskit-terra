# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test for the DAGCircuit object"""

from __future__ import annotations

from collections import Counter
import math
import pickle
import copy
import io
import unittest

from ddt import ddt, data
import numpy as np
from numpy import pi

from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGInNode, DAGOutNode, DAGCircuitError
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    Clbit,
    Qubit,
    Measure,
    Delay,
    Reset,
    Gate,
    Instruction,
    Parameter,
    Barrier,
    SwitchCaseOp,
    IfElseOp,
    WhileLoopOp,
    CASE_DEFAULT,
    Store,
    BoxOp,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import (
    IGate,
    HGate,
    CXGate,
    CZGate,
    XGate,
    YGate,
    U1Gate,
    RXGate,
    CSGate,
    CHGate,
    UnitaryGate,
    PauliProductRotationGate,
    PauliProductMeasurement,
)
from qiskit.quantum_info import Pauli
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class DAGTest(QiskitTestCase):
    """Base test class with helper methods."""

    def raise_if_dagcircuit_invalid(self, dag):
        """Validates the internal consistency of a DAGCircuit._multi_graph.
        Intended for use in testing.

        Raises:
           DAGCircuitError: if DAGCircuit._multi_graph is inconsistent.
        """

        if not dag._is_dag():
            raise DAGCircuitError("multi_graph is not a DAG.")

        # Every node should be of type in, out, or op.
        # All input/output nodes should be present in input_map/output_map.
        for node in dag.nodes():
            if isinstance(node, DAGInNode):
                self.assertEqual(node, dag.input_map[node.wire])
            elif isinstance(node, DAGOutNode):
                self.assertEqual(node, dag.output_map[node.wire])
            elif isinstance(node, DAGOpNode):
                continue
            else:
                raise DAGCircuitError(f"Found node of unexpected type: {type(node)}")

        # Shape of node.op should match shape of node.
        for node in dag.op_nodes():
            self.assertEqual(len(node.qargs), node.op.num_qubits)
            self.assertEqual(len(node.cargs), node.op.num_clbits)

        # Every edge should be labled with a known wire.
        edges_outside_wires = [
            edge_data for edge_data in dag._edges() if edge_data not in dag.wires
        ]
        if edges_outside_wires:
            raise DAGCircuitError(
                f"multi_graph contains one or more edges ({edges_outside_wires}) "
                f"not found in DAGCircuit.wires ({dag.wires})."
            )

        # Every wire should have exactly one input node and one output node.
        for wire in dag.wires:
            in_node = dag.input_map[wire]
            out_node = dag.output_map[wire]

            self.assertEqual(in_node.wire, wire)
            self.assertEqual(out_node.wire, wire)
            self.assertIsInstance(in_node, DAGInNode)
            self.assertIsInstance(out_node, DAGOutNode)

        # Every wire should be propagated by exactly one edge between nodes.
        for wire in dag.wires:
            cur_node_id = dag.input_map[wire]._node_id
            out_node_id = dag.output_map[wire]._node_id

            while cur_node_id != out_node_id:
                out_edges = dag._out_edges(cur_node_id)
                edges_to_follow = [
                    (src, dest, data) for (src, dest, data) in out_edges if data == wire
                ]

                self.assertEqual(len(edges_to_follow), 1)
                cur_node_id = edges_to_follow[0][1]

        # Wires can only terminate at input/output nodes.
        op_counts = Counter()
        for op_node in dag.op_nodes():
            self.assertEqual(
                sum(1 for _ in dag.predecessors(op_node)), sum(1 for _ in dag.successors(op_node))
            )
            op_counts[op_node.name] += 1
        # The _op_names attribute should match the counted op names
        self.assertEqual(op_counts, dag.count_ops())

        # Node input/output edges should match node qarg/carg/condition.
        for node in dag.op_nodes():
            in_wires = set(dag._in_wires(node._node_id))
            out_wires = set(dag._out_wires(node._node_id))

            node_cond_bits = set(
                node.condition[0][:] if getattr(node, "condition", None) is not None else []
            )
            node_qubits = set(node.qargs)
            node_clbits = set(node.cargs)

            all_bits = node_qubits | node_clbits | node_cond_bits

            self.assertEqual(
                in_wires, all_bits, f"In-edge wires {in_wires} != node bits {all_bits}"
            )
            self.assertEqual(
                out_wires, all_bits, f"Out-edge wires {out_wires} != node bits {all_bits}"
            )


class TestDagRegisters(DAGTest):
    """Test qreg and creg inside the dag"""

    def test_add_qreg_creg(self):
        """add_qreg() and  add_creg() methods"""
        dag = DAGCircuit()
        dag.add_qreg(QuantumRegister(2, "qr"))
        dag.add_creg(ClassicalRegister(1, "cr"))
        self.assertDictEqual(dag.qregs, {"qr": QuantumRegister(2, "qr")})
        self.assertDictEqual(dag.cregs, {"cr": ClassicalRegister(1, "cr")})

    def test_dag_get_qubits(self):
        """get_qubits() method"""
        dag = DAGCircuit()
        dag.add_qreg(QuantumRegister(1, "qr1"))
        dag.add_qreg(QuantumRegister(1, "qr10"))
        dag.add_qreg(QuantumRegister(1, "qr0"))
        dag.add_qreg(QuantumRegister(1, "qr3"))
        dag.add_qreg(QuantumRegister(1, "qr4"))
        dag.add_qreg(QuantumRegister(1, "qr6"))
        self.assertListEqual(
            dag.qubits,
            [
                QuantumRegister(1, "qr1")[0],
                QuantumRegister(1, "qr10")[0],
                QuantumRegister(1, "qr0")[0],
                QuantumRegister(1, "qr3")[0],
                QuantumRegister(1, "qr4")[0],
                QuantumRegister(1, "qr6")[0],
            ],
        )

    def test_pickle_bit_locations_with_reg(self):
        """Test bit locations preserved through pickle."""
        dag = DAGCircuit()
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        dag.add_qreg(qr)
        dag.add_creg(cr)
        self.assertEqual(dag.find_bit(dag.qubits[1]).index, 1)
        self.assertEqual(dag.find_bit(dag.qubits[1]).registers, [(qr, 1)])
        self.assertEqual(dag.find_bit(dag.clbits[0]).index, 0)
        self.assertEqual(dag.find_bit(dag.clbits[0]).registers, [(cr, 0)])
        with io.BytesIO() as buf:
            pickle.dump(dag, buf)
            buf.seek(0)
            output = pickle.load(buf)
        self.assertEqual(output.find_bit(output.qubits[1]).index, 1)
        self.assertEqual(output.find_bit(output.qubits[1]).registers, [(qr, 1)])
        self.assertEqual(output.find_bit(output.clbits[0]).index, 0)
        self.assertEqual(output.find_bit(output.clbits[0]).registers, [(cr, 0)])

    def test_deepcopy_bit_locations_with_reg(self):
        """Test bit locations preserved through deepcopy."""
        dag = DAGCircuit()
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(1, "cr")
        dag.add_qreg(qr)
        dag.add_creg(cr)
        self.assertEqual(dag.find_bit(dag.qubits[1]).index, 1)
        self.assertEqual(dag.find_bit(dag.qubits[1]).registers, [(qr, 1)])
        self.assertEqual(dag.find_bit(dag.clbits[0]).index, 0)
        self.assertEqual(dag.find_bit(dag.clbits[0]).registers, [(cr, 0)])
        output = copy.deepcopy(dag)
        self.assertEqual(output.find_bit(output.qubits[1]).index, 1)
        self.assertEqual(output.find_bit(output.qubits[1]).registers, [(qr, 1)])
        self.assertEqual(output.find_bit(output.clbits[0]).index, 0)
        self.assertEqual(output.find_bit(output.clbits[0]).registers, [(cr, 0)])

    def test_pickle_bit_locations_with_no_reg(self):
        """Test bit locations with no registers preserved through pickle."""
        dag = DAGCircuit()
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit()]
        dag.add_qubits(qubits)
        dag.add_clbits(clbits)
        self.assertEqual(dag.find_bit(dag.qubits[1]).index, 1)
        self.assertEqual(dag.find_bit(dag.qubits[1]).registers, [])
        self.assertEqual(dag.find_bit(dag.clbits[0]).index, 0)
        self.assertEqual(dag.find_bit(dag.clbits[0]).registers, [])
        with io.BytesIO() as buf:
            pickle.dump(dag, buf)
            buf.seek(0)
            output = pickle.load(buf)
        self.assertEqual(output.find_bit(output.qubits[1]).index, 1)
        self.assertEqual(output.find_bit(output.qubits[1]).registers, [])
        self.assertEqual(output.find_bit(output.clbits[0]).index, 0)
        self.assertEqual(output.find_bit(output.clbits[0]).registers, [])

    def test_deepcopy_bit_locations_with_no_reg(self):
        """Test bit locations with no registers preserved through deepcopy."""
        dag = DAGCircuit()
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit()]
        dag.add_qubits(qubits)
        dag.add_clbits(clbits)
        self.assertEqual(dag.find_bit(dag.qubits[1]).index, 1)
        self.assertEqual(dag.find_bit(dag.qubits[1]).registers, [])
        self.assertEqual(dag.find_bit(dag.clbits[0]).index, 0)
        self.assertEqual(dag.find_bit(dag.clbits[0]).registers, [])
        output = copy.deepcopy(dag)
        self.assertEqual(output.find_bit(output.qubits[1]).index, 1)
        self.assertEqual(output.find_bit(output.qubits[1]).registers, [])
        self.assertEqual(output.find_bit(output.clbits[0]).index, 0)
        self.assertEqual(output.find_bit(output.clbits[0]).registers, [])

    def test_add_reg_duplicate(self):
        """add_qreg with the same register twice is not allowed."""
        dag = DAGCircuit()
        qr = QuantumRegister(2)
        dag.add_qreg(qr)
        self.assertRaises(DAGCircuitError, dag.add_qreg, qr)

    def test_add_reg_duplicate_name(self):
        """Adding quantum registers with the same name is not allowed."""
        dag = DAGCircuit()
        qr1 = QuantumRegister(3, "qr")
        dag.add_qreg(qr1)
        qr2 = QuantumRegister(2, "qr")
        self.assertRaises(DAGCircuitError, dag.add_qreg, qr2)

    def test_add_reg_bad_type(self):
        """add_qreg with a classical register is not allowed."""
        dag = DAGCircuit()
        cr = ClassicalRegister(2)
        self.assertRaises((DAGCircuitError, TypeError), dag.add_qreg, cr)

    def test_add_qubits_invalid_qubits(self):
        """Verify we raise if pass not a Qubit."""
        dag = DAGCircuit()

        with self.assertRaisesRegex(DAGCircuitError, "not a Qubit instance"):
            dag.add_qubits([Clbit()])

        with self.assertRaisesRegex(DAGCircuitError, "not a Qubit instance"):
            dag.add_qubits([Qubit(), Clbit(), Qubit()])

    def test_add_qubits_invalid_clbits(self):
        """Verify we raise if pass not a Clbit."""
        dag = DAGCircuit()

        with self.assertRaisesRegex(DAGCircuitError, "not a Clbit instance"):
            dag.add_clbits([Qubit()])

        with self.assertRaisesRegex(DAGCircuitError, "not a Clbit instance"):
            dag.add_clbits([Clbit(), Qubit(), Clbit()])

    def test_raise_if_bits_already_present(self):
        """Verify we raise when attempting to add a Bit already in the DAG."""
        dag = DAGCircuit()
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]

        dag.add_qubits(qubits)
        dag.add_clbits(clbits)

        with self.assertRaisesRegex(DAGCircuitError, "duplicate qubits"):
            dag.add_qubits(qubits)

        with self.assertRaisesRegex(DAGCircuitError, "duplicate clbits "):
            dag.add_clbits(clbits)

    def test_raise_if_bits_already_present_from_register(self):
        """Verify we raise when attempting to add a Bit already in the DAG."""
        dag = DAGCircuit()
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        dag.add_creg(cr)
        dag.add_qreg(qr)

        with self.assertRaisesRegex(DAGCircuitError, "duplicate qubits"):
            dag.add_qubits(qr[:])

        with self.assertRaisesRegex(DAGCircuitError, "duplicate clbits "):
            dag.add_clbits(cr[:])

    def test_adding_individual_bit(self):
        """Verify we can add a individual bits to a DAG."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)

        new_bit = Qubit()

        dag.add_qubits([new_bit])

        self.assertEqual(dag.qubits, list(qr) + [new_bit])
        self.assertEqual(list(dag.qregs.values()), [qr])

    def test_find_bit_with_registers(self):
        """Test find_bit with a register."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        res = dag.find_bit(qr[2])
        self.assertEqual(res.index, 2)
        self.assertEqual(res.registers, [(qr, 2)])

    def test_find_bit_with_registers_and_standalone(self):
        """Test find_bit with a register and standalone bit."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        new_bit = Qubit()
        dag.add_qubits([new_bit])
        res = dag.find_bit(qr[2])
        self.assertEqual(res.index, 2)
        bit_res = dag.find_bit(new_bit)
        self.assertEqual(bit_res.index, 3)
        self.assertEqual(bit_res.registers, [])

    def test_find_bit_with_classical_registers_and_standalone(self):
        """Test find_bit with a register and standalone bit."""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "C")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr)
        new_bit = Qubit()
        new_clbit = Clbit()
        dag.add_qubits([new_bit])
        dag.add_clbits([new_clbit])
        res = dag.find_bit(qr[2])
        self.assertEqual(res.index, 2)
        bit_res = dag.find_bit(new_bit)
        self.assertEqual(bit_res.index, 3)
        self.assertEqual(bit_res.registers, [])
        classical_res = dag.find_bit(cr[2])
        self.assertEqual(classical_res.index, 2)
        self.assertEqual(classical_res.registers, [(cr, 2)])
        single_cl_bit_res = dag.find_bit(new_clbit)
        self.assertEqual(single_cl_bit_res.index, 3)
        self.assertEqual(single_cl_bit_res.registers, [])

    def test_find_bit_missing(self):
        """Test error when find_bit is called with missing bit."""
        qr = QuantumRegister(3, "qr")
        dag = DAGCircuit()
        dag.add_qreg(qr)
        new_bit = Qubit()
        with self.assertRaises(DAGCircuitError):
            dag.find_bit(new_bit)

    def test_remove_qubits_determinism(self):
        """Regression test of gh-16655."""

        def build() -> DAGCircuit:
            in_reg = QuantumRegister(4, "f")
            in_dag = DAGCircuit()
            in_dag.add_qreg(in_reg)

            out_reg = QuantumRegister(4, "q")
            out_dag = in_dag.copy_empty_like()
            out_dag.add_qreg(out_reg)
            out_dag.remove_qregs(in_reg)
            out_dag.remove_qubits(*in_reg)
            out_dag.apply_operation_back(XGate(), (out_dag.qubits[0],), ())
            out_dag.apply_operation_back(XGate(), (out_dag.qubits[1],), ())
            return out_dag

        base = build()
        dags = [build() for _ in range(10)]
        self.assertEqual([base.structurally_equal(dag) for dag in dags], [True] * len(dags))

    def test_remove_clbits_determinism(self):
        """Regression test of gh-16655."""

        def build() -> DAGCircuit:
            qr = QuantumRegister(4, "q")
            in_reg = ClassicalRegister(4, "f")
            in_dag = DAGCircuit()
            in_dag.add_qreg(qr)
            in_dag.add_creg(in_reg)

            out_reg = ClassicalRegister(4, "c")
            out_dag = in_dag.copy_empty_like()
            out_dag.add_creg(out_reg)
            out_dag.remove_cregs(in_reg)
            out_dag.remove_clbits(*in_reg)
            out_dag.apply_operation_back(Measure(), (out_dag.qubits[0],), (out_dag.clbits[0],))
            out_dag.apply_operation_back(Measure(), (out_dag.qubits[1],), (out_dag.clbits[1],))
            return out_dag

        base = build()
        dags = [build() for _ in range(10)]
        self.assertEqual([base.structurally_equal(dag) for dag in dags], [True] * len(dags))


class TestDagWireRemoval(DAGTest):
    """Test removal of registers and idle wires."""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        self.dag.name = "Name"
        self.dag.metadata = "Metadata"
        qreg = QuantumRegister(3, "qr")
        creg0 = ClassicalRegister(2, "c0")
        creg1 = ClassicalRegister(2, "c1")
        creg2 = ClassicalRegister(name="c2", bits=list(creg1))
        clbit = Clbit()
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg0)
        self.dag.add_creg(creg1)
        self.dag.add_creg(creg2)
        self.dag.add_clbits([clbit])

        self.qreg = qreg
        self.creg0 = creg0
        self.creg1 = creg1
        self.creg1_alias = creg2
        self.individual_clbit = clbit

        self.original_cregs = [self.creg0, self.creg1, self.creg1_alias]

        # skip adding bits of creg1_alias since it's just an alias for creg1
        self.original_clbits = [
            b for reg in self.original_cregs if reg != self.creg1_alias for b in reg
        ] + [self.individual_clbit]

    def assert_cregs_equal(self, cregs, excluding=None):
        """Assert test DAG cregs match the expected values.

        Args:
            cregs (Iterable(ClassicalRegister)): the classical registers to expect
            excluding (Set(ClassicalRegister)): classical registers to remove from
            ``cregs`` before the comparison.
        """
        if excluding is None:
            excluding = set()
        self.assertEqual(
            self.dag.cregs, {creg.name: creg for creg in cregs if creg not in excluding}
        )

    def assert_clbits_equal(self, clbits, excluding=None):
        """Assert test DAG clbits match the expected values.

        Args:
            clbits (Iterable(Clbit)): the classical bits to expect
            excluding (Set(ClassicalRegister)): classical bits to remove from
            ``clbits`` before the comparison.
        """
        if excluding is None:
            excluding = set()
        self.assertEqual(self.dag.clbits, [b for b in clbits if b not in excluding])

    def test_remove_idle_creg(self):
        """Removing an idle classical register removes just the register."""
        self.dag.remove_cregs(self.creg0)

        self.assert_cregs_equal(self.original_cregs, excluding={self.creg0})
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_busy_creg(self):
        """Classical registers with both busy and idle underlying bits
        can be removed, keeping underlying bits."""
        self.dag.apply_operation_back(Measure(), [self.qreg[0]], [self.creg0[0]])
        self.dag.remove_cregs(self.creg0)

        # creg removal always succeeds
        self.assert_cregs_equal(self.original_cregs, excluding={self.creg0})

        # original bits remain
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_cregs_shared_bits(self):
        """Removing a classical register does not remove other classical
        registers that are simply aliases of it."""
        self.dag.remove_cregs(self.creg1)

        # Only creg1 should be deleted, not its alias
        self.assert_cregs_equal(self.original_cregs, excluding={self.creg1})
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_unknown_creg(self):
        """Classical register removal of unknown registers raises."""
        unknown_creg = ClassicalRegister(1)
        with self.assertRaisesRegex(DAGCircuitError, ".*creg not in circuit.*"):
            self.dag.remove_cregs(unknown_creg)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_idle_clbit(self):
        """Idle classical bits not referenced by any register can be removed."""
        self.dag.remove_clbits(self.individual_clbit)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits, excluding={self.individual_clbit})

    def test_copy_empty_like(self):
        """Copy dag circuit metadata with copy_empty_like."""
        result_dag = self.dag.copy_empty_like()
        self.assertEqual(self.dag.name, result_dag.name)
        self.assertEqual(self.dag.metadata, result_dag.metadata)
        self.assertEqual(self.dag.clbits, result_dag.clbits)
        self.assertEqual(self.dag.qubits, result_dag.qubits)
        self.assertEqual(self.dag.cregs, result_dag.cregs)
        self.assertEqual(self.dag.qregs, result_dag.qregs)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(self.dag.duration, result_dag.duration)
            self.assertEqual(self.dag.unit, result_dag.unit)

    def test_copy_empty_like_vars(self):
        """Variables should be part of the empty copy."""
        dag = DAGCircuit()
        dag.add_input_var(expr.Var.new("a", types.Bool()))
        dag.add_input_var(expr.Var.new("b", types.Uint(8)))
        dag.add_declared_var(expr.Var.new("c", types.Bool()))
        dag.add_declared_var(expr.Var.new("d", types.Uint(8)))
        dag.add_declared_stretch(expr.Stretch.new("e"))
        self.assertEqual(dag, dag.copy_empty_like())

        dag = DAGCircuit()
        dag.add_captured_var(expr.Var.new("a", types.Bool()))
        dag.add_captured_var(expr.Var.new("b", types.Uint(8)))
        dag.add_declared_stretch(expr.Stretch.new("c"))
        dag.add_declared_var(expr.Var.new("d", types.Bool()))
        dag.add_declared_var(expr.Var.new("e", types.Uint(8)))
        dag.add_declared_stretch(expr.Stretch.new("f"))
        self.assertEqual(dag, dag.copy_empty_like())

    def test_copy_empty_like_vars_captures(self):
        """Variables can be converted to captures as part of the empty copy."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Uint(8))
        e = expr.Stretch.new("e")
        all_captures = DAGCircuit()
        for var in [a, b, c, d]:
            all_captures.add_captured_var(var)
        all_captures.add_captured_stretch(e)

        dag = DAGCircuit()
        dag.add_input_var(a)
        dag.add_input_var(b)
        dag.add_declared_var(c)
        dag.add_declared_var(d)
        dag.add_declared_stretch(e)
        self.assertEqual(all_captures, dag.copy_empty_like(vars_mode="captures"))

        dag = DAGCircuit()
        dag.add_captured_var(a)
        dag.add_captured_var(b)
        dag.add_declared_var(c)
        dag.add_declared_var(d)
        dag.add_declared_stretch(e)
        self.assertEqual(all_captures, dag.copy_empty_like(vars_mode="captures"))

    def test_copy_empty_like_vars_drop(self):
        """Variables can be dropped as part of the empty copy."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Uint(8))
        e = expr.Stretch.new("e")

        dag = DAGCircuit()
        dag.add_input_var(a)
        dag.add_input_var(b)
        dag.add_declared_var(c)
        dag.add_declared_var(d)
        dag.add_declared_stretch(e)
        self.assertEqual(DAGCircuit(), dag.copy_empty_like(vars_mode="drop"))

        dag = DAGCircuit()
        dag.add_captured_var(a)
        dag.add_captured_var(b)
        dag.add_declared_var(c)
        dag.add_declared_var(d)
        dag.add_captured_stretch(e)
        self.assertEqual(DAGCircuit(), dag.copy_empty_like(vars_mode="drop"))

    def test_remove_busy_clbit(self):
        """Classical bit removal of busy classical bits raises."""
        self.dag.apply_operation_back(Measure(), [self.qreg[0]], [self.individual_clbit])

        with self.assertRaisesRegex(DAGCircuitError, ".*clbits not idle.*"):
            self.dag.remove_clbits(self.individual_clbit)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_referenced_clbit(self):
        """Classical bit removal removes registers that reference a removed bit,
        even if they have other bits that aren't removed."""
        self.dag.remove_clbits(self.creg0[0])

        self.assert_cregs_equal(self.original_cregs, excluding={self.creg0})
        self.assert_clbits_equal(self.original_clbits, excluding={self.creg0[0]})

    def test_remove_multi_reg_referenced_clbit(self):
        """Classical bit removal removes all registers that reference a removed bit."""
        self.dag.remove_clbits(*self.creg1)

        self.assert_cregs_equal(self.original_cregs, excluding={self.creg1, self.creg1_alias})
        self.assert_clbits_equal(self.original_clbits, excluding=set(self.creg1))

    def test_remove_unknown_clbit(self):
        """Classical bit removal of unknown bits raises."""
        unknown_clbit = Clbit()
        with self.assertRaisesRegex(DAGCircuitError, ".*clbits not in circuit.*"):
            self.dag.remove_clbits(unknown_clbit)

        self.assert_cregs_equal(self.original_cregs)
        self.assert_clbits_equal(self.original_clbits)

    def test_remove_clbit_with_control_flow(self):
        """Test clbit removal in the middle of clbits with control flow."""
        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2, "a")
        cr2 = ClassicalRegister(2, "b")
        clbit = Clbit()
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        dag.add_clbits([clbit])

        inner = QuantumCircuit(1)
        inner.h(0)
        inner.z(0)

        op = IfElseOp(expr.logic_and(expr.equal(cr1, 3), expr.logic_not(clbit)), inner, None)
        dag.apply_operation_back(op, qr, ())
        dag.remove_clbits(*cr2)
        self.assertEqual(dag.clbits, list(cr1) + [clbit])
        self.assertEqual(dag.cregs, {"a": cr1})

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_creg(cr1)
        expected.add_clbits([clbit])

        op = IfElseOp(expr.logic_and(expr.equal(cr1, 3), expr.logic_not(clbit)), inner, None)
        expected.apply_operation_back(op, qr, ())

        self.assertEqual(dag, expected)

    def test_remove_first_node_and_depth(self):
        """Test a nondeterministic panic in rustworkx-core cased by a hole for the first node index

        This caused a non-deterministic panic when using rustworkx-core
        0.18.0 (fixed in rustworkx 0.18.1)
        """
        dag = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        qreg_two = QuantumRegister(3, "qr_two")
        dag.add_qreg(qreg)
        dag.add_qreg(qreg_two)
        dag.remove_qubits(*qreg)
        dag.apply_operation_back(XGate(), (dag.qubits[0],), ())
        dag.apply_operation_back(XGate(), (dag.qubits[1],), ())
        dag.apply_operation_back(XGate(), (dag.qubits[2],), ())
        depth = dag.depth()
        self.assertEqual(depth, 1)


@ddt
class TestDagApplyOperation(DAGTest):
    """Test adding an op node to a dag."""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg)

        self.qubit0 = qreg[0]
        self.qubit1 = qreg[1]
        self.qubit2 = qreg[2]
        self.clbit0 = creg[0]
        self.clbit1 = creg[1]
        self.condition = (creg, 3)

    def test_apply_operation_back(self):
        """The apply_operation_back() method."""
        x_gate = XGate()
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(x_gate, [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0], [self.clbit0])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.assertEqual(len(list(self.dag.nodes())), 16)
        self.assertEqual(len(list(self.dag.edges())), 15)

    def test_edges(self):
        """Test that DAGCircuit.edges() behaves as expected with ops."""
        x_gate = XGate()
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(x_gate, [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0], [self.clbit0])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        out_edges = self.dag.edges(self.dag.output_map.values())
        self.assertEqual(list(out_edges), [])
        in_edges = self.dag.edges(self.dag.input_map.values())
        # number of edges for input nodes should be the same as number of wires
        self.assertEqual(len(list(in_edges)), 5)

    def test_apply_operation_front(self):
        """The apply_operation_front() method"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_front(Reset(), [self.qubit0], [])
        h_node = self.dag.op_nodes(op=HGate).pop()
        reset_node = self.dag.op_nodes(op=Reset).pop()

        self.assertIn(reset_node, set(self.dag.predecessors(h_node)))

    @data("front", "back")
    def test_apply_operation_duplicate_wires(self, direction):
        """The apply-front and apply-back methods should only attempt to add a single edge, even if
        a classical variable is mentioned more than once."""
        qc = QuantumCircuit()
        a = qc.add_input("a", types.Bool())

        dag = circuit_to_dag(qc)
        op = IfElseOp(expr.logic_and(a, a), QuantumCircuit(), None)
        if direction == "front":
            dag.apply_operation_front(op, [], [])
        else:
            dag.apply_operation_back(op, [], [])
        self.assertEqual(len(list(dag.edges())), 2)

    @data("front", "back")
    def test_apply_operation_determinism(self, direction):
        """When adding a node with many qubits and clbits, the order of the edge ids should be
        deterministic."""
        qc = QuantumCircuit(100)
        qubits = qc.qubits
        if direction == "front":
            apply = DAGCircuit.apply_operation_front
        else:
            apply = DAGCircuit.apply_operation_back
        left = circuit_to_dag(qc)
        apply(left, Barrier(len(qubits)), qubits, [])
        right = circuit_to_dag(qc)
        apply(right, Barrier(len(qubits)), qubits, [])
        self.assertTrue(left.structurally_equal(right))

    def test_apply_operation_expr_condition(self):
        """Test that the operation-applying functions correctly handle wires implied from `Expr`
        nodes in the `condition` field of `ControlFlowOp` instances."""
        inner = QuantumCircuit(1)
        inner.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2, "a")
        cr2 = ClassicalRegister(2, "b")
        clbit = Clbit()

        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        dag.add_clbits([clbit])

        # Note that 'cr2' is not in either condition.
        expected_wires = set(qr) | set(cr1) | {clbit}

        op = IfElseOp(expr.logic_and(expr.equal(cr1, 3), expr.logic_not(clbit)), inner, None)
        node = dag.apply_operation_back(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.condition, expr.Expr)
        self.assertEqual(test_wires, expected_wires)

        op = WhileLoopOp(expr.logic_or(expr.less(2, cr1), clbit), inner)
        node = dag.apply_operation_front(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.condition, expr.Expr)
        self.assertEqual(test_wires, expected_wires)

    def test_apply_operation_expr_target(self):
        """Test that the operation-applying functions correctly handle wires implied from `Expr`
        nodes in the `target` field of `SwitchCaseOp`."""
        case_1 = QuantumCircuit(1)
        case_1.x(0)
        case_2 = QuantumCircuit(1)
        case_2.y(0)
        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2, "a")
        cr2 = ClassicalRegister(2, "b")
        # Note that 'cr2' is not in the condition.
        op = SwitchCaseOp(expr.bit_and(cr1, 2), [(1, case_1), (2, case_2)])

        expected_wires = set(qr) | set(cr1)

        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)

        node = dag.apply_operation_back(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.target, expr.Expr)
        self.assertEqual(test_wires, expected_wires)

        node = dag.apply_operation_front(op, qr, ())
        test_wires = {wire for _source, _dest, wire in dag.edges(node)}
        self.assertIsInstance(node.op.target, expr.Expr)
        self.assertEqual(test_wires, expected_wires)


class TestDagNodeSelection(DAGTest):
    """Test methods that select certain dag nodes"""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg)

        self.qubit0 = qreg[0]
        self.qubit1 = qreg[1]
        self.qubit2 = qreg[2]
        self.clbit0 = creg[0]
        self.clbit1 = creg[1]
        self.condition = (creg, 3)

    def test_front_layer(self):
        """The method dag.front_layer() returns first layer"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.front_layer()
        self.assertEqual(len(op_nodes), 1)
        self.assertIsInstance(op_nodes[0].op, HGate)

    def test_get_op_nodes_all(self):
        """The method dag.op_nodes() returns all op nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.op_nodes()
        self.assertEqual(len(op_nodes), 3)

        for node in op_nodes:
            self.assertIsInstance(node.op, Instruction)

    def test_node_representations(self):
        """Test the __repr__ methods of the DAG Nodes"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        # test OpNode
        cx_node = self.dag.named_nodes("cx")[0]
        self.assertEqual(
            repr(cx_node),
            f"DAGOpNode(op={cx_node.op}, qargs={cx_node.qargs}, cargs={cx_node.cargs})",
        )
        # test InNode
        predecessor_cnot = self.dag.quantum_predecessors(cx_node)
        predecessor = next(predecessor_cnot)
        self.assertEqual(repr(predecessor), f"DAGInNode(wire={predecessor.wire})")
        # test OutNode
        successor_cnot = self.dag.quantum_successors(cx_node)
        successor = next(successor_cnot)
        self.assertEqual(repr(successor), f"DAGOutNode(wire={successor.wire})")

    def test_get_op_nodes_particular(self):
        """The method dag.gates_nodes(op=AGate) returns all the AGate nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])

        op_nodes = self.dag.op_nodes(op=HGate)
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()

        self.assertIsInstance(op_node_1.op, HGate)
        self.assertIsInstance(op_node_2.op, HGate)

    def test_get_op_nodes_multiple_types(self):
        """dag.op_nodes(op=iterable_of_types) returns any of the requested types."""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])

        nodes = self.dag.op_nodes(op={Measure, Reset})
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(isinstance(node.op, (Measure, Reset)) for node in nodes))

        # Also accept other iterables (ordering doesn't matter, only membership).
        nodes = self.dag.op_nodes(op=(Measure, Reset))
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(isinstance(node.op, (Measure, Reset)) for node in nodes))

    def test_get_op_nodes_multiple_types_empty(self):
        """An empty iterable should match nothing."""
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])
        self.assertEqual(self.dag.op_nodes(op=()), [])

    def test_get_op_nodes_multiple_types_invalid_element(self):
        """Non-type elements should error."""
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])
        with self.assertRaises(TypeError):
            self.dag.op_nodes(op=(Reset, 123))

    def test_get_op_nodes_with_custom_gate(self):
        """Iterable filters match custom gates and preserve subclass matching."""

        class CustomXGate(XGate):
            _standard_gate = False

        self.dag.apply_operation_back(CustomXGate(), [self.qubit0], [])
        self.dag.apply_operation_back(XGate(), [self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        nodes = self.dag.op_nodes(op=(CustomXGate, HGate))
        self.assertEqual(len(nodes), 2)
        self.assertTrue(all(isinstance(node.op, (CustomXGate, HGate)) for node in nodes))
        self.assertEqual(len(self.dag.op_nodes(op=XGate)), 2)

    def test_quantum_successors(self):
        """The method dag.quantum_successors() returns successors connected by quantum edges"""

        # q_0: |0>─────■───|0>─
        #         ┌─┐┌─┴─┐
        # q_1: |0>┤M├┤ X ├─────
        #         └╥┘└───┘
        #  c_0: 0 ═╬═══════════
        #          ║
        #  c_1: 0 ═╩═══════════

        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        successor_measure = self.dag.quantum_successors(self.dag.named_nodes("measure").pop())
        cnot_node = next(successor_measure)
        with self.assertRaises(StopIteration):
            next(successor_measure)

        self.assertIsInstance(cnot_node.op, CXGate)

        successor_cnot = self.dag.quantum_successors(cnot_node)
        # Ordering between Reset and out[q1] is indeterminate.

        successor1 = next(successor_cnot)
        successor2 = next(successor_cnot)
        with self.assertRaises(StopIteration):
            next(successor_cnot)

        self.assertTrue(
            (isinstance(successor1, DAGOutNode) and isinstance(successor2.op, Reset))
            or (isinstance(successor2, DAGOutNode) and isinstance(successor1.op, Reset))
        )

    def test_is_successor(self):
        """The method dag.is_successor(A, B) checks if node B is a successor of A"""
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        measure_node = self.dag.named_nodes("measure")[0]
        cx_node = self.dag.named_nodes("cx")[0]
        reset_node = self.dag.named_nodes("reset")[0]

        self.assertTrue(self.dag.is_successor(measure_node, cx_node))
        self.assertFalse(self.dag.is_successor(measure_node, reset_node))
        self.assertTrue(self.dag.is_successor(cx_node, reset_node))

    def test_quantum_predecessors(self):
        """The method dag.quantum_predecessors() returns predecessors connected by quantum edges"""

        # q_0: |0>─|0>───■─────
        #              ┌─┴─┐┌─┐
        # q_1: |0>─────┤ X ├┤M├
        #              └───┘└╥┘
        #  c_0: 0 ═══════════╬═
        #                    ║
        #  c_1: 0 ═══════════╩═

        self.dag.apply_operation_back(Reset(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])

        predecessor_measure = self.dag.quantum_predecessors(self.dag.named_nodes("measure").pop())
        cnot_node = next(predecessor_measure)
        with self.assertRaises(StopIteration):
            next(predecessor_measure)

        self.assertIsInstance(cnot_node.op, CXGate)

        predecessor_cnot = self.dag.quantum_predecessors(cnot_node)
        # Ordering between Reset and in[q1] is indeterminate.

        predecessor1 = next(predecessor_cnot)
        predecessor2 = next(predecessor_cnot)
        with self.assertRaises(StopIteration):
            next(predecessor_cnot)

        self.assertTrue(
            (isinstance(predecessor1, DAGInNode) and isinstance(predecessor2.op, Reset))
            or (isinstance(predecessor2, DAGInNode) and isinstance(predecessor1.op, Reset))
        )

    def test_classical_predecessors(self):
        """The method dag.classical_predecessors() returns predecessors connected by classical edges"""

        #       ┌───┐                         ┌───┐
        #  q_0: ┤ H ├──■───────────────────■──┤ H ├
        #       ├───┤┌─┴─┐               ┌─┴─┐├───┤
        #  q_1: ┤ H ├┤ X ├──■─────────■──┤ X ├┤ H ├
        #       └───┘└───┘┌─┴─┐┌───┐┌─┴─┐└───┘└───┘
        #  q_2: ──────────┤ X ├┤ H ├┤ X ├──────────
        #                 └───┘└───┘└───┘
        #  c: 5/═══════════════════════════════════

        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0], [self.clbit0])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])

        predecessor_measure = self.dag.classical_predecessors(self.dag.named_nodes("measure").pop())

        predecessor1 = next(predecessor_measure)

        with self.assertRaises(StopIteration):
            next(predecessor_measure)

        self.assertIsInstance(predecessor1, DAGInNode)
        self.assertIsInstance(predecessor1.wire, Clbit)

    def test_apply_operation_reject_invalid_qarg_carg(self):
        """Test that we can't add a carg to qargs and vice versa on apply methods"""
        with self.assertRaises((KeyError, TypeError)):
            self.dag.apply_operation_back(Measure(), [self.clbit1], [self.qubit1])
        with self.assertRaises((KeyError, TypeError)):
            self.dag.apply_operation_front(Measure(), [self.clbit1], [self.qubit1])

    def test_classical_successors(self):
        """The method dag.classical_successors() returns successors connected by classical edges"""

        #       ┌───┐                         ┌───┐
        #  q_0: ┤ H ├──■───────────────────■──┤ H ├
        #       ├───┤┌─┴─┐               ┌─┴─┐├───┤
        #  q_1: ┤ H ├┤ X ├──■─────────■──┤ X ├┤ H ├
        #       └───┘└───┘┌─┴─┐┌───┐┌─┴─┐└───┘└───┘
        #  q_2: ──────────┤ X ├┤ H ├┤ X ├──────────
        #                 └───┘└───┘└───┘
        #  c: 5/═══════════════════════════════════

        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(Measure(), [self.qubit0], [self.clbit0])
        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])

        successors_measure = self.dag.classical_successors(self.dag.named_nodes("measure").pop())

        successors1 = next(successors_measure)
        with self.assertRaises(StopIteration):
            next(successors_measure)

        self.assertIsInstance(successors1, DAGOutNode)
        self.assertIsInstance(successors1.wire, Clbit)

    def test_is_predecessor(self):
        """The method dag.is_predecessor(A, B) checks if node B is a predecessor of A"""

        self.dag.apply_operation_back(Measure(), [self.qubit1], [self.clbit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        measure_node = self.dag.named_nodes("measure")[0]
        cx_node = self.dag.named_nodes("cx")[0]
        reset_node = self.dag.named_nodes("reset")[0]

        self.assertTrue(self.dag.is_predecessor(cx_node, measure_node))
        self.assertFalse(self.dag.is_predecessor(reset_node, measure_node))
        self.assertTrue(self.dag.is_predecessor(reset_node, cx_node))

    def test_get_gates_nodes(self):
        """The method dag.gate_nodes() returns all gate nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.gate_nodes()
        self.assertEqual(len(op_nodes), 2)

        op_node_1 = op_nodes.pop()
        op_node_2 = op_nodes.pop()

        self.assertIsInstance(op_node_1.op, Gate)
        self.assertIsInstance(op_node_2.op, Gate)

    def test_two_q_gates(self):
        """The method dag.two_qubit_ops() returns all 2Q gate operation nodes"""
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Barrier(2), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(Reset(), [self.qubit0], [])

        op_nodes = self.dag.two_qubit_ops()
        self.assertEqual(len(op_nodes), 1)

        op_node = op_nodes.pop()
        self.assertIsInstance(op_node.op, Gate)
        self.assertEqual(len(op_node.qargs), 2)

    def test_get_named_nodes(self):
        """The get_named_nodes(AName) method returns all the nodes with name AName"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        # The ordering is not assured, so we only compare the output (unordered) sets.
        # We use tuples because lists aren't hashable.
        named_nodes = self.dag.named_nodes("cx")

        node_qargs = {tuple(node.qargs) for node in named_nodes}

        expected_qargs = {
            (self.qubit0, self.qubit1),
            (self.qubit2, self.qubit1),
            (self.qubit0, self.qubit2),
        }
        self.assertEqual(expected_qargs, node_qargs)

    def test_topological_nodes(self):
        """The topological_nodes() method"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        named_nodes = self.dag.topological_nodes()

        qr = self.dag.qregs["qr"]
        cr = self.dag.cregs["cr"]
        expected = [
            qr[0],
            qr[1],
            ("cx", (self.qubit0, self.qubit1)),
            ("h", (self.qubit0,)),
            qr[2],
            ("cx", (self.qubit2, self.qubit1)),
            ("cx", (self.qubit0, self.qubit2)),
            ("h", (self.qubit2,)),
            cr[0],
            cr[1],
            qr[0],
            qr[1],
            qr[2],
            cr[0],
            cr[1],
        ]
        self.assertEqual(
            [((i.op.name, i.qargs) if isinstance(i, DAGOpNode) else i.wire) for i in named_nodes],
            expected,
        )

    def test_topological_op_nodes(self):
        """The topological_op_nodes() method"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        named_nodes = self.dag.topological_op_nodes()

        expected = [
            ("cx", (self.qubit0, self.qubit1)),
            ("h", (self.qubit0,)),
            ("cx", (self.qubit2, self.qubit1)),
            ("cx", (self.qubit0, self.qubit2)),
            ("h", (self.qubit2,)),
        ]
        self.assertEqual(expected, [(i.op.name, i.qargs) for i in named_nodes])

    def test_topological_nodes_reversals(self):
        """Test topological_nodes in reverse order following the example pattern."""

        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2])
        self.dag.apply_operation_back(XGate(), [self.qubit2])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1])
        self.dag.apply_operation_back(HGate(), [self.qubit0])

        nodes_in_reverse = self.dag.topological_nodes(reverse=True)

        qr = self.dag.qregs["qr"]
        cr = self.dag.cregs["cr"]

        expected = [
            qr[0],
            ("h", (self.qubit0,)),
            qr[1],
            ("cx", (self.qubit0, self.qubit1)),
            qr[0],
            qr[2],
            ("x", (self.qubit2,)),
            ("cx", (self.qubit1, self.qubit2)),
            qr[1],
            qr[2],
            cr[0],
            cr[0],
            cr[1],
            cr[1],
        ]

        self.assertEqual(
            [
                ((i.op.name, i.qargs) if isinstance(i, DAGOpNode) else i.wire)
                for i in nodes_in_reverse
            ],
            expected,
        )

    def test_topological_op_nodes_reversals(self):
        """Test topological_op_nodes in reverse order following the example pattern."""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2], [])
        self.dag.apply_operation_back(HGate(), [self.qubit2], [])

        op_nodes_in_reverse = list(self.dag.topological_op_nodes(reverse=True))

        expected = [
            ("h", (self.qubit2,)),
            ("cx", (self.qubit0, self.qubit2)),
            ("h", (self.qubit0,)),
            ("cx", (self.qubit2, self.qubit1)),
            ("cx", (self.qubit0, self.qubit1)),
        ]

        self.assertEqual(
            [(i.op.name, i.qargs) for i in op_nodes_in_reverse],
            expected,
        )

    def test_dag_nodes_on_wire(self):
        """Test that listing the gates on a qubit/classical bit gets the correct gates"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])

        qbit = self.dag.qubits[0]
        self.assertEqual([0, 10, 11, 1], [i._node_id for i in self.dag.nodes_on_wire(qbit)])
        self.assertEqual(
            [10, 11], [i._node_id for i in self.dag.nodes_on_wire(qbit, only_ops=True)]
        )

        cbit = self.dag.clbits[0]
        self.assertEqual([6, 7], [i._node_id for i in self.dag.nodes_on_wire(cbit)])
        self.assertEqual([], [i._node_id for i in self.dag.nodes_on_wire(cbit, only_ops=True)])

        with self.assertRaises(DAGCircuitError):
            next(self.dag.nodes_on_wire(QuantumRegister(5, "qr")[4]))

    def test_dag_nodes_on_wire_multiple_successors(self):
        """
        Test that if an DAGOpNode has multiple successors in the DAG along one wire, they are all
        retrieved in order. This could be the case for a circuit such as

        .. code-block:: text

                q0_0: |0>──■─────────■──
                         ┌─┴─┐┌───┐┌─┴─┐
                q0_1: |0>┤ X ├┤ H ├┤ X ├
                         └───┘└───┘└───┘
        Both the 2nd CX gate and the H gate follow the first CX gate in the DAG, so they
        both must be returned but in the correct order.
        """
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(HGate(), [self.qubit1], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])

        nodes = self.dag.nodes_on_wire(self.dag.qubits[1], only_ops=True)
        node_names = [nd.op.name for nd in nodes]

        self.assertEqual(node_names, ["cx", "h", "cx"])

    def test_remove_op_node(self):
        """Test remove_op_node method."""
        self.dag.apply_operation_back(HGate(), [self.qubit0])

        op_nodes = self.dag.gate_nodes()
        h_gate = op_nodes.pop()
        self.dag.remove_op_node(h_gate)

        self.assertEqual(len(self.dag.gate_nodes()), 0)

    def test_remove_op_node_longer(self):
        """Test remove_op_node method in a "longer" dag"""
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit2])
        self.dag.apply_operation_back(HGate(), [self.qubit2])

        op_nodes = list(self.dag.topological_op_nodes())
        self.dag.remove_op_node(op_nodes[0])

        expected = [
            ("h", (self.qubit0,)),
            ("cx", (self.qubit2, self.qubit1)),
            ("cx", (self.qubit0, self.qubit2)),
            ("h", (self.qubit2,)),
        ]
        self.assertEqual(expected, [(i.op.name, i.qargs) for i in self.dag.topological_op_nodes()])

    def test_remove_non_op_node(self):
        """Try to remove a non-op node with remove_op_node method."""
        self.dag.apply_operation_back(HGate(), [self.qubit0])

        in_node = next(self.dag.topological_nodes())
        self.assertRaises(DAGCircuitError, self.dag.remove_op_node, in_node)

    def test_dag_collect_runs(self):
        """Test the collect_runs method with 3 different gates."""
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2])
        self.dag.apply_operation_back(HGate(), [self.qubit2])
        collected_runs = self.dag.collect_runs(["u1", "cx", "h"])
        self.assertEqual(len(collected_runs), 3)
        for run in collected_runs:
            if run[0].op.name == "cx":
                self.assertEqual(len(run), 2)
                self.assertEqual(["cx"] * 2, [x.op.name for x in run])
                self.assertEqual(
                    [(self.qubit2, self.qubit1), (self.qubit1, self.qubit2)], [x.qargs for x in run]
                )
            elif run[0].op.name == "h":
                self.assertEqual(len(run), 1)
                self.assertEqual(["h"], [x.op.name for x in run])
                self.assertEqual([(self.qubit2,)], [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1"] * 3, [x.op.name for x in run])
                self.assertEqual(
                    [(self.qubit0,), (self.qubit0,), (self.qubit0,)], [x.qargs for x in run]
                )
            else:
                self.fail("Unknown run encountered")

    def test_dag_collect_1q_runs(self):
        """Test the collect_1q_runs method with 3 different gates."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(CXGate(), [self.qubit2, self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit1, self.qubit2])
        self.dag.apply_operation_back(HGate(), [self.qubit2])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 2)
        for run in collected_runs:
            if run[0].op.name == "h":
                self.assertEqual(len(run), 1)
                self.assertEqual(["h"], [x.op.name for x in run])
                self.assertEqual([(self.qubit2,)], [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1"] * 3, [x.op.name for x in run])
                self.assertEqual(
                    [(self.qubit0,), (self.qubit0,), (self.qubit0,)], [x.qargs for x in run]
                )
            else:
                self.fail("Unknown run encountered")

    def test_dag_collect_1q_runs_with_parameterized_gate(self):
        """Test collect 1q splits on parameterized gates."""
        theta = Parameter("theta")
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(theta), [self.qubit0])
        self.dag.apply_operation_back(XGate(), [self.qubit0])
        self.dag.apply_operation_back(XGate(), [self.qubit0])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 2)
        run_gates = [[x.op.name for x in run] for run in collected_runs]
        self.assertIn(["h", "h"], run_gates)
        self.assertIn(["x", "x"], run_gates)
        self.assertNotIn("u1", [x.op.name for run in collected_runs for x in run])

    def test_dag_collect_1q_runs_with_cx_in_middle(self):
        """Test collect_1q_runs_with a cx in the middle of the run."""
        self.dag.apply_operation_back(Reset(), [self.qubit0])
        self.dag.apply_operation_back(Delay(100), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(HGate(), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit0])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit1])
        self.dag.apply_operation_back(U1Gate(3.14), [self.qubit1])
        self.dag.apply_operation_back(HGate(), [self.qubit1])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1])
        self.dag.apply_operation_back(YGate(), [self.qubit0])
        self.dag.apply_operation_back(YGate(), [self.qubit0])
        self.dag.apply_operation_back(XGate(), [self.qubit1])
        self.dag.apply_operation_back(XGate(), [self.qubit1])
        collected_runs = self.dag.collect_1q_runs()
        self.assertEqual(len(collected_runs), 4)
        for run in collected_runs:
            if run[0].op.name == "h":
                self.assertEqual(len(run), 3)
                self.assertEqual(["h", "h", "u1"], [x.op.name for x in run])
                self.assertEqual([(self.qubit0,)] * 3, [x.qargs for x in run])
            elif run[0].op.name == "u1":
                self.assertEqual(len(run), 3)
                self.assertEqual(["u1", "u1", "h"], [x.op.name for x in run])
                self.assertEqual([(self.qubit1,)] * 3, [x.qargs for x in run])
            elif run[0].op.name == "x":
                self.assertEqual(len(run), 2)
                self.assertEqual(["x", "x"], [x.op.name for x in run])
                self.assertEqual([(self.qubit1,)] * 2, [x.qargs for x in run])
            elif run[0].op.name == "y":
                self.assertEqual(len(run), 2)
                self.assertEqual(["y", "y"], [x.op.name for x in run])
                self.assertEqual([(self.qubit0,)] * 2, [x.qargs for x in run])
            else:
                self.fail("Unknown run encountered")


class TestDagLayers(DAGTest):
    """Test finding layers on the dag"""

    def test_layers_basic(self):
        """The layers() method returns a list of layers, each of them with a list of nodes."""
        qreg = QuantumRegister(2, "qr")
        creg = ClassicalRegister(2, "cr")
        qubit0 = qreg[0]
        qubit1 = qreg[1]
        clbit0 = creg[0]
        clbit1 = creg[1]
        x_gate = XGate()
        dag = DAGCircuit()
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qubit0], [])
        dag.apply_operation_back(CXGate(), [qubit0, qubit1], [])
        dag.apply_operation_back(Measure(), [qubit1], [clbit1])
        dag.apply_operation_back(x_gate, [qubit1], [])
        dag.apply_operation_back(Measure(), [qubit0], [clbit0])
        dag.apply_operation_back(Measure(), [qubit1], [clbit1])

        layers = list(dag.layers())
        self.assertEqual(5, len(layers))

        name_layers = [
            [node.op.name for node in layer["graph"].nodes() if isinstance(node, DAGOpNode)]
            for layer in layers
        ]

        self.assertEqual([["h"], ["cx"], ["measure", "measure"], ["x"], ["measure"]], name_layers)

    def test_layers_maintains_order(self):
        """Test that the layers method doesn't mess up the order of the DAG as
        reported in #2698"""
        qr = QuantumRegister(1, "q0")

        # the order the nodes should be in
        truth = [
            (DAGInNode, qr[0], 0),
            (DAGOpNode, "x", 2),
            (DAGOpNode, "id", 3),
            (DAGOutNode, qr[0], 1),
        ]

        # this only occurred sometimes so has to be run more than once
        # (10 times seemed to always be enough for this bug to show at least once)
        for _ in range(10):
            qc = QuantumCircuit(qr)
            qc.x(0)
            dag = circuit_to_dag(qc)
            dag1 = list(dag.layers())[0]["graph"]
            dag1.apply_operation_back(IGate(), [qr[0]], [])

            comp = [
                (type(nd), nd.op.name if isinstance(nd, DAGOpNode) else nd.wire, nd._node_id)
                for nd in dag1.topological_nodes()
            ]
            self.assertEqual(comp, truth)

    def test_serial_layers_do_not_copy_global_phase(self):
        """serial_layers() should not copy the parent DAG global phase."""
        qc = QuantumCircuit(2, global_phase=math.pi / 7)
        qc.h(0)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)

        layers = list(dag.serial_layers())

        self.assertEqual(len(layers), 2)
        self.assertEqual(dag.global_phase, math.pi / 7)
        self.assertEqual(layers[0]["graph"].global_phase, 0.0)
        self.assertEqual(layers[1]["graph"].global_phase, 0.0)

    def test_layers_do_not_copy_global_phase(self):
        """layers() should not copy the parent DAG global phase."""
        qc = QuantumCircuit(2, global_phase=math.pi / 7)
        qc.h(0)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)

        layers = list(dag.layers())

        self.assertEqual(len(layers), 2)
        self.assertEqual(dag.global_phase, math.pi / 7)
        self.assertEqual(layers[0]["graph"].global_phase, 0.0)
        self.assertEqual(layers[1]["graph"].global_phase, 0.0)


def _sort_key(indices: dict[Qubit, int]):
    """Return key function for sorting DAGCircuits, given a global qubit mapping."""

    def _min_active_qubit_id(dag):
        """Transform a DAGCircuit into its minimum active qubit index."""
        try:
            first_op = next(dag.topological_op_nodes())
        except StopIteration:
            return -1
        return min(indices[q] for q in first_op.qargs)

    return _min_active_qubit_id


class TestCircuitProperties(DAGTest):
    """DAGCircuit properties test."""

    def setUp(self):
        #       ┌───┐                ┌───┐
        # q0_0: ┤ H ├────────────────┤ X ├──────────
        #       └───┘                └─┬─┘     ┌───┐
        # q0_1: ───────────────────────┼───────┤ H ├
        #                 ┌───┐        │  ┌───┐└─┬─┘
        # q0_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #       ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # q0_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #       └───┘└──────────────┘  │
        # q1_0: ───────────────────────■────────────
        #                              │
        # q1_1: ───────────────────────■────────────
        super().setUp()
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[2], qr1[3])
        circ.h(qr1[2])
        circ.t(qr1[2])
        circ.ch(qr1[2], qr1[1])
        circ.u(0.0, 0.1, 0.2, qr1[3])
        circ.ccx(qr2[0], qr2[1], qr1[0])

        self.dag = circuit_to_dag(circ)

    def test_circuit_size(self):
        """Test total number of operations in circuit."""
        self.assertEqual(self.dag.size(), 7)

    def test_circuit_depth(self):
        """Test circuit depth."""
        self.assertEqual(self.dag.depth(), 4)

    def test_circuit_width(self):
        """Test number of qubits + clbits in circuit."""
        self.assertEqual(self.dag.width(), 6)

    def test_circuit_num_qubits(self):
        """Test number of qubits in circuit."""
        self.assertEqual(self.dag.num_qubits(), 6)

    def test_circuit_operations(self):
        """Test circuit operations breakdown by kind of op."""
        operations = {"h": 2, "t": 1, "u": 1, "cx": 1, "ch": 1, "ccx": 1}

        self.assertDictEqual(self.dag.count_ops(), operations)

    def test_circuit_factors(self):
        """Test number of separable factors in circuit."""
        self.assertEqual(self.dag.num_tensor_factors(), 2)

    def test_separable_circuits(self):
        """Test separating disconnected sets of qubits in a circuit."""
        # Empty case
        dag = DAGCircuit()
        self.assertEqual(dag.separable_circuits(), [])

        # 3 disconnected qubits
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(2, "c")
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qreg[0]], [])
        dag.apply_operation_back(HGate(), [qreg[1]], [])
        dag.apply_operation_back(HGate(), [qreg[2]], [])
        dag.apply_operation_back(HGate(), [qreg[2]], [])

        comp_dag1 = DAGCircuit()
        comp_dag1.add_qubits([qreg[0]])
        comp_dag1.add_creg(creg)
        comp_dag1.apply_operation_back(HGate(), [qreg[0]], [])
        comp_dag2 = DAGCircuit()
        comp_dag2.add_qubits([qreg[1]])
        comp_dag2.add_creg(creg)
        comp_dag2.apply_operation_back(HGate(), [qreg[1]], [])
        comp_dag3 = DAGCircuit()
        comp_dag3.add_qubits([qreg[2]])
        comp_dag3.add_creg(creg)
        comp_dag3.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag3.apply_operation_back(HGate(), [qreg[2]], [])

        compare_dags = [comp_dag1, comp_dag2, comp_dag3]

        # Get a mapping from qubit to qubit id in original circuit
        indices = {bit: i for i, bit in enumerate(dag.qubits)}

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(remove_idle_qubits=True), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

        # 2 sets of disconnected qubits
        dag.apply_operation_back(CXGate(), [qreg[1], qreg[2]], [])

        comp_dag1 = DAGCircuit()
        comp_dag1.add_qubits([qreg[0]])
        comp_dag1.add_creg(creg)
        comp_dag1.apply_operation_back(HGate(), [qreg[0]], [])
        comp_dag2 = DAGCircuit()
        comp_dag2.add_qubits([qreg[1], qreg[2]])
        comp_dag2.add_creg(creg)
        comp_dag2.apply_operation_back(HGate(), [qreg[1]], [])
        comp_dag2.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag2.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag2.apply_operation_back(CXGate(), [qreg[1], qreg[2]], [])

        compare_dags = [comp_dag1, comp_dag2]

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(remove_idle_qubits=True), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

        # One connected component
        dag.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])

        comp_dag1 = DAGCircuit()
        comp_dag1.add_qreg(qreg)
        comp_dag1.add_creg(creg)
        comp_dag1.apply_operation_back(HGate(), [qreg[0]], [])
        comp_dag1.apply_operation_back(HGate(), [qreg[1]], [])
        comp_dag1.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag1.apply_operation_back(HGate(), [qreg[2]], [])
        comp_dag1.apply_operation_back(CXGate(), [qreg[1], qreg[2]], [])
        comp_dag1.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])

        compare_dags = [comp_dag1]

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(remove_idle_qubits=True), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

    def test_separable_circuits_w_measurements(self):
        """Test separating disconnected sets of qubits in a circuit with measurements."""
        # Test circuit ordering with measurements
        dag = DAGCircuit()
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(1, "c")
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(HGate(), [qreg[0]], [])
        dag.apply_operation_back(XGate(), [qreg[0]], [])
        dag.apply_operation_back(XGate(), [qreg[1]], [])
        dag.apply_operation_back(HGate(), [qreg[1]], [])
        dag.apply_operation_back(YGate(), [qreg[2]], [])
        dag.apply_operation_back(HGate(), [qreg[2]], [])
        dag.apply_operation_back(Measure(), [qreg[0]], [creg[0]])

        qc1 = QuantumCircuit(3, 1)
        qc1.h(0)
        qc1.x(0)
        qc1.measure(0, 0)
        qc2 = QuantumCircuit(3, 1)
        qc2.x(1)
        qc2.h(1)
        qc3 = QuantumCircuit(3, 1)
        qc3.y(2)
        qc3.h(2)
        qcs = [qc1, qc2, qc3]
        compare_dags = [circuit_to_dag(qc) for qc in qcs]

        # Get a mapping from qubit to qubit id in original circuit
        indices = {bit: i for i, bit in enumerate(dag.qubits)}

        # Don't rely on separable_circuits outputs to be in any order. We sort by min active qubit id
        dags = sorted(dag.separable_circuits(), key=_sort_key(indices))

        self.assertEqual(dags, compare_dags)

    def test_default_metadata_value(self):
        """Test that the default DAGCircuit metadata is valid QuantumCircuit metadata."""
        qc = QuantumCircuit(1)
        qc.metadata = self.dag.metadata
        self.assertEqual(qc.metadata, {})


class TestCircuitControlFlowProperties(DAGTest):
    """Properties tests of DAGCircuit with control-flow instructions."""

    def setUp(self):
        super().setUp()
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        qc.measure(0, 0)
        # The depth of an if-else is the path through the longest block (regardless of the
        # condition).  The size is the sum of both blocks (mostly for optimization-target purposes).
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.x(1)
            qc.cx(2, 3)
        # The longest depth goes through this else_ - the block contributes 13 in size and 12 in
        # depth (because the x(1) happens concurrently with the for).
        with else_:
            qc.x(1)
            # This for loop contributes 3x to size and depth.  Its total size (and weight in the
            # depth) is 12 (as 3 * (1 + 3))
            with qc.for_loop(range(3)):
                qc.z(2)
                # This for loop contributes 3x to size and depth.
                with qc.for_loop((4, 0, 1)):
                    qc.z(2)
        # While loops contribute 1x to both size and depth, so this
        with qc.while_loop((qc.clbits[0], True)):
            qc.h(0)
            qc.measure(0, 0)

        self.dag = circuit_to_dag(qc)

    def test_circuit_size(self):
        """Test total number of operations in circuit."""
        # The size sums all branches of control flow, with `for` loops weighted by their number of
        # iterations (so the outer `for_loop` has a size 12 in total).  The whole `if_else` has size
        # 15, and the while-loop body counts once.
        self.assertEqual(self.dag.size(recurse=True), 19)
        with self.assertRaisesRegex(DAGCircuitError, "Size with control flow is ambiguous"):
            self.dag.size(recurse=False)

    def test_circuit_depth(self):
        """Test circuit depth."""
        # The longest depth path goes through the `h, measure`, then the `else`, then the `while`.
        # Within the `else`, the `x(1)` happens concurrently with the `for` loop.  Because each for
        # loop has 3 values, the total weight of the else is 12.
        self.assertEqual(self.dag.depth(recurse=True), 16)
        with self.assertRaisesRegex(DAGCircuitError, "Depth with control flow is ambiguous"):
            self.dag.depth(recurse=False)

    def test_circuit_width(self):
        """Test number of qubits + clbits in circuit."""
        self.assertEqual(self.dag.width(), 6)

    def test_circuit_num_qubits(self):
        """Test number of qubits in circuit."""
        self.assertEqual(self.dag.num_qubits(), 5)

    def test_circuit_operations(self):
        """Test circuit operations breakdown by kind of op."""

        self.assertDictEqual(
            self.dag.count_ops(recurse=False), {"h": 1, "measure": 1, "if_else": 1, "while_loop": 1}
        )
        self.assertDictEqual(
            self.dag.count_ops(recurse=True),
            {
                "h": 2,
                "measure": 2,
                "if_else": 1,
                "x": 2,
                "cx": 1,
                "for_loop": 2,
                "z": 2,
                "while_loop": 1,
            },
        )


class TestCircuitSpecialCases(DAGTest):
    """DAGCircuit test for special cases, usually for regression."""

    def test_circuit_depth_with_repetition(self):
        """When cx repeat, they are not "the same".
        See https://github.com/Qiskit/qiskit-terra/issues/1994
        """
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[1], qr2[1])
        circ.cx(qr1[1], qr2[1])
        circ.h(qr2[0])
        dag = circuit_to_dag(circ)

        self.assertEqual(dag.depth(), 2)


class TestDagEquivalence(DAGTest):
    """DAGCircuit equivalence check."""

    def setUp(self):
        #        ┌───┐                ┌───┐
        # qr1_0: ┤ H ├────────────────┤ X ├──────────
        #        └───┘                └─┬─┘     ┌───┐
        # qr1_1: ───────────────────────┼───────┤ H ├
        #                  ┌───┐        │  ┌───┐└─┬─┘
        # qr1_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #        ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #        └───┘└──────────────┘  │
        # qr2_0: ───────────────────────■────────────
        #                               │
        # qr2_1: ───────────────────────■────────────
        super().setUp()
        self.qr1 = QuantumRegister(4, "qr1")
        self.qr2 = QuantumRegister(2, "qr2")
        circ1 = QuantumCircuit(self.qr1, self.qr2)
        circ1.h(self.qr1[0])
        circ1.cx(self.qr1[2], self.qr1[3])
        circ1.h(self.qr1[2])
        circ1.t(self.qr1[2])
        circ1.ch(self.qr1[2], self.qr1[1])
        circ1.u(0.0, 0.1, 0.2, self.qr1[3])
        circ1.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        self.dag1 = circuit_to_dag(circ1)

    def test_unitary_gate_eq(self):
        """Test equality of equal DAGOpNodes containing unitary gates.
        See: https://github.com/Qiskit/qiskit-terra/issues/14047
        """
        # Create the unitary matrix
        unitary_matrix = np.array(
            [
                [0.65328148 - 0.27059805j, 0.0 + 0.0j, 0.0 + 0.0j, 0.65328148 - 0.27059805j],
                [0.0 + 0.0j, 0.65328148 - 0.27059805j, -0.65328148 + 0.27059805j, 0.0 + 0.0j],
                [0.27059805 - 0.65328148j, 0.0 + 0.0j, 0.0 + 0.0j, -0.27059805 + 0.65328148j],
                [0.0 + 0.0j, 0.27059805 - 0.65328148j, 0.27059805 - 0.65328148j, 0.0 + 0.0j],
            ]
        )

        # Create the instruction
        instruction = Instruction(
            name="unitary", num_qubits=2, num_clbits=0, params=[unitary_matrix]
        )

        # Create the quantum register and qubits
        qreg = QuantumRegister(9, "q")
        qubit1 = Qubit(qreg, 7)
        qubit2 = Qubit(qreg, 5)

        # Create the DAGOpNode
        dag_op_node = DAGOpNode(op=instruction, qargs=(qubit1, qubit2), cargs=())

        # Create the DAGOpNode2
        dag_op_node_2 = DAGOpNode(op=instruction, qargs=(qubit1, qubit2), cargs=())

        self.assertEqual(dag_op_node, dag_op_node_2)

    def test_unitary_gate_variants_eq(self):
        """Test equality of equal DAGOpNodes containing unitary gates."""
        # Create the instructions.
        # Note that our rust code will interpret mat1 as ArrayType::NDArray and
        # mat2 as ArrayType::TwoQ.
        mat1 = np.asarray(CSGate().to_matrix(), dtype=complex)
        mat2 = np.asarray(CSGate().to_matrix(), dtype=complex, order="f")
        instruction1 = Instruction(name="unitary", num_qubits=2, num_clbits=0, params=[mat1])
        instruction2 = Instruction(name="unitary", num_qubits=2, num_clbits=0, params=[mat2])

        # Create the quantum register and qubits
        qreg = QuantumRegister(9, "q")
        qubit1 = Qubit(qreg, 7)
        qubit2 = Qubit(qreg, 5)

        # Create DAGOpNodes
        node1 = DAGOpNode(op=instruction1, qargs=(qubit1, qubit2), cargs=())
        node2 = DAGOpNode(op=instruction2, qargs=(qubit1, qubit2), cargs=())

        self.assertEqual(node1, node2)

    def test_boxop_neq(self):
        """Test non equality of nonequal DAGOpNodes and QuantumCircuits containing BoxOp gates."""
        qc = QuantumCircuit()

        box1 = BoxOp(body=qc, duration=300.0, unit="dt")
        box2 = BoxOp(body=qc, duration=300.0, unit="ms")
        node1 = DAGOpNode(op=box1, qargs=(), cargs=())
        node2 = DAGOpNode(op=box2, qargs=(), cargs=())
        self.assertNotEqual(node1, node2)

        qc1 = QuantumCircuit()
        qc2 = QuantumCircuit()
        qc1.append(box1)
        qc2.append(box2)
        self.assertNotEqual(qc1, qc2)

        box1 = BoxOp(body=qc, duration=300.0, unit="dt")
        box2 = BoxOp(body=qc, duration=200.0, unit="dt")
        node1 = DAGOpNode(op=box1, qargs=(), cargs=())
        node2 = DAGOpNode(op=box2, qargs=(), cargs=())
        self.assertNotEqual(node1, node2)

        qc1 = QuantumCircuit()
        qc2 = QuantumCircuit()
        qc1.append(box1)
        qc2.append(box2)
        self.assertNotEqual(qc1, qc2)

    def test_dag_eq(self):
        """DAG equivalence check: True."""
        #        ┌───┐                ┌───┐
        # qr1_0: ┤ H ├────────────────┤ X ├──────────
        #        └───┘                └─┬─┘     ┌───┐
        # qr1_1: ───────────────────────┼───────┤ H ├
        #                  ┌───┐        │  ┌───┐└─┬─┘
        # qr1_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #        ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #        └───┘└──────────────┘  │
        # qr2_0: ───────────────────────■────────────
        #                               │
        # qr2_1: ───────────────────────■────────────
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u(0.0, 0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[2], self.qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertEqual(self.dag1, dag2)

    def test_dag_neq_topology(self):
        """DAG equivalence check: False. Different topology."""
        #        ┌───┐                     ┌───┐
        # qr1_0: ┤ H ├───────■─────────────┤ X ├
        #        └───┘     ┌─┴─┐           └─┬─┘
        # qr1_1: ──────────┤ H ├─────────────┼──
        #                  ├───┤      ┌───┐  │
        # qr1_2: ──■───────┤ H ├──────┤ T ├──┼──
        #        ┌─┴─┐┌────┴───┴─────┐└───┘  │
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├───────┼──
        #        └───┘└──────────────┘       │
        # qr2_0: ────────────────────────────■──
        #                                    │
        # qr2_1: ────────────────────────────■──
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u(0.0, 0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.ch(self.qr1[0], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertNotEqual(self.dag1, dag2)

    def test_dag_eq_multiple_delays(self):
        """DAG Equivalence with multiple delays of different unit types."""
        #      ┌───┐┌────────────────┐     ┌──────────────────┐
        # q_0: ┤ H ├┤ Delay(100[dt]) ├──■──┤ Delay(100.0[ms]) ├
        #      └───┘└────────────────┘┌─┴─┐└┬───────────────┬─┘
        # q_1: ───────────────────────┤ X ├─┤ Delay(1.0[s]) ├──
        #                             └───┘ └───────────────┘
        circ = QuantumCircuit(2, 0)
        circ.h(0)
        circ.delay(100, 0, "dt")
        circ.cx(0, 1)
        circ.delay(1, 1, "s")
        circ.delay(100, 0, "ms")

        expected = DAGCircuit()
        expected.add_qreg(QuantumRegister(2, "q"))
        expected.apply_operation_back(HGate(), [expected.qubits[0]], [])
        expected.apply_operation_back(Delay(100, "dt"), [expected.qubits[0]], [])
        expected.apply_operation_back(CXGate(), [expected.qubits[0], expected.qubits[1]], [])
        expected.apply_operation_back(Delay(1, "s"), [expected.qubits[1]], [])
        expected.apply_operation_back(Delay(100, "ms"), [expected.qubits[0]], [])

        obtained = circuit_to_dag(circ)
        for node1, node2 in zip(expected.op_nodes(), obtained.op_nodes()):
            # Compare node by node to make sure it compares correctly.
            self.assertEqual(node1, node2)
        self.assertEqual(obtained, expected)

    def test_dag_neq_same_topology(self):
        """DAG equivalence check: False. Same topology."""
        #        ┌───┐                ┌───┐
        # qr1_0: ┤ H ├────────────────┤ X ├──────────
        #        └───┘                └─┬─┘     ┌───┐
        # qr1_1: ───────────────────────┼───────┤ X ├
        #                  ┌───┐        │  ┌───┐└─┬─┘
        # qr1_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #        ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # qr1_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #        └───┘└──────────────┘  │
        # qr2_0: ───────────────────────■────────────
        #                               │
        # qr2_1: ───────────────────────■────────────
        circ2 = QuantumCircuit(self.qr1, self.qr2)
        circ2.cx(self.qr1[2], self.qr1[3])
        circ2.u(0.0, 0.1, 0.2, self.qr1[3])
        circ2.h(self.qr1[0])
        circ2.h(self.qr1[2])
        circ2.t(self.qr1[2])
        circ2.cx(self.qr1[2], self.qr1[1])  # <--- The difference: ch(qr1[2], qr1[1])
        circ2.ccx(self.qr2[0], self.qr2[1], self.qr1[0])
        dag2 = circuit_to_dag(circ2)

        self.assertNotEqual(self.dag1, dag2)

    def test_dag_neq_reversed_unitary(self):
        """Test that reversing the order of qubits of a unitary gate leads to a different circuit."""
        unitary = UnitaryGate(CHGate())

        qc1 = QuantumCircuit(4)
        qc1.append(unitary, [0, 1])

        qc2 = QuantumCircuit(4)
        qc2.append(unitary, [1, 0])

        self.assertNotEqual(qc1, qc2)
        self.assertNotEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

    def test_dag_neq_reversed_ppr(self):
        """Test that reversing the order of qubits of a PPR gate leads to a different
        circuit."""
        ppr = PauliProductRotationGate(Pauli("ZX"), 0.1)

        qc1 = QuantumCircuit(4)
        qc1.append(ppr, [0, 1])

        qc2 = QuantumCircuit(4)
        qc2.append(ppr, [1, 0])

        self.assertNotEqual(qc1, qc2)
        self.assertNotEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

    def test_dag_neq_reversed_ppm(self):
        """Test that reversing the order of qubits of a PPM instruction leads to a different
        circuit."""
        ppm = PauliProductMeasurement(Pauli("ZX"))

        qc1 = QuantumCircuit(4, 1)
        qc1.append(ppm, [0, 1], [0])

        qc2 = QuantumCircuit(4, 1)
        qc2.append(ppm, [1, 0], [0])

        self.assertNotEqual(qc1, qc2)
        self.assertNotEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

    def test_node_params_equal_unequal(self):
        """Test node params are equal or unequal."""
        qc1 = QuantumCircuit(1)
        qc2 = QuantumCircuit(1)
        qc3 = QuantumCircuit(1)
        qc1.p(pi / 4, 0)
        dag1 = circuit_to_dag(qc1)
        qc2.p(pi / 4, 0)
        dag2 = circuit_to_dag(qc2)
        qc3.p(pi / 2, 0)
        dag3 = circuit_to_dag(qc3)
        self.assertEqual(dag1, dag2)
        self.assertNotEqual(dag2, dag3)

    def test_semantic_expr(self):
        """Test that the semantic equality is applied to the bits in `Expr` components as well."""
        cr = ClassicalRegister(3, "c1")
        clbit1 = Clbit()
        clbit2 = Clbit()

        body = QuantumCircuit(1)
        body.x(0)

        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc1.if_test(expr.logic_not(clbit1), body, [0], [])
        qc2 = QuantumCircuit(cr, [Qubit(), clbit2])
        qc2.if_test(expr.logic_not(clbit2), body, [0], [])
        self.assertEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc2 = QuantumCircuit(cr, [Qubit(), clbit2])
        qc1.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        qc2.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        self.assertEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

        # Order of bits not the same.
        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc1.if_test(expr.logic_not(clbit1), body, [0], [])
        qc2 = QuantumCircuit([Qubit(), clbit2], cr)
        qc2.if_test(expr.logic_not(clbit2), body, [0], [])

        qc1 = QuantumCircuit(cr, [Qubit(), clbit1])
        qc1.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        qc2 = QuantumCircuit([Qubit(), clbit2], cr)
        qc2.switch(expr.bit_and(cr, 5), [(1, body)], [0], [])
        self.assertNotEqual(circuit_to_dag(qc1), circuit_to_dag(qc2))

    def test_present_vars(self):
        """The vars should be compared whether or not they're used."""
        a_bool = expr.Var.new("a", types.Bool())
        a_u8 = expr.Var.new("a", types.Uint(8))
        a_stretch = expr.Stretch.new("a")
        a_u8_other = expr.Var.new("a", types.Uint(8))
        b_bool = expr.Var.new("b", types.Bool())
        b_stretch = expr.Stretch.new("b")

        left = DAGCircuit()
        left.add_input_var(a_bool)
        left.add_input_var(b_bool)
        self.assertEqual(left.num_input_vars, 2)
        self.assertEqual(left.num_captured_vars, 0)
        self.assertEqual(left.num_declared_vars, 0)
        self.assertEqual(left.num_vars, 2)

        right = DAGCircuit()
        right.add_input_var(a_bool)
        right.add_input_var(b_bool)
        self.assertEqual(right.num_input_vars, 2)
        self.assertEqual(right.num_captured_vars, 0)
        self.assertEqual(right.num_declared_vars, 0)
        self.assertEqual(left.num_vars, 2)
        self.assertEqual(left, right)

        right = DAGCircuit()
        right.add_input_var(a_u8)
        right.add_input_var(b_bool)
        self.assertEqual(right.num_input_vars, 2)
        self.assertEqual(right.num_captured_vars, 0)
        self.assertEqual(right.num_declared_vars, 0)
        self.assertEqual(right.num_vars, 2)
        self.assertNotEqual(left, right)

        right = DAGCircuit()
        self.assertEqual(right.num_input_vars, 0)
        self.assertEqual(right.num_captured_vars, 0)
        self.assertEqual(right.num_declared_vars, 0)
        self.assertEqual(right.num_vars, 0)
        self.assertNotEqual(left, right)

        right = DAGCircuit()
        right.add_captured_var(a_bool)
        right.add_captured_var(b_bool)
        self.assertEqual(right.num_input_vars, 0)
        self.assertEqual(right.num_captured_vars, 2)
        self.assertEqual(right.num_declared_vars, 0)
        self.assertEqual(right.num_vars, 2)
        self.assertNotEqual(left, right)

        right = DAGCircuit()
        right.add_captured_stretch(a_stretch)
        right.add_captured_stretch(b_stretch)
        self.assertEqual(right.num_input_vars, 0)
        self.assertEqual(right.num_captured_vars, 0)
        self.assertEqual(right.num_declared_vars, 0)
        self.assertEqual(right.num_captured_stretches, 2)
        self.assertEqual(right.num_declared_stretches, 0)
        self.assertEqual(right.num_stretches, 2)
        self.assertNotEqual(left, right)

        right = DAGCircuit()
        right.add_declared_var(a_bool)
        right.add_declared_var(b_bool)
        self.assertEqual(right.num_input_vars, 0)
        self.assertEqual(right.num_captured_vars, 0)
        self.assertEqual(right.num_declared_vars, 2)
        self.assertEqual(right.num_vars, 2)
        self.assertNotEqual(left, right)

        right = DAGCircuit()
        right.add_declared_stretch(a_stretch)
        right.add_declared_stretch(b_stretch)
        self.assertEqual(right.num_input_vars, 0)
        self.assertEqual(right.num_captured_vars, 0)
        self.assertEqual(right.num_declared_vars, 0)
        self.assertEqual(right.num_captured_stretches, 0)
        self.assertEqual(right.num_declared_stretches, 2)
        self.assertEqual(right.num_stretches, 2)
        self.assertNotEqual(left, right)

        left = DAGCircuit()
        left.add_captured_var(a_u8)
        left.add_captured_stretch(b_stretch)

        right = DAGCircuit()
        right.add_captured_var(a_u8)
        right.add_captured_stretch(b_stretch)
        self.assertEqual(left, right)

        right = DAGCircuit()
        right.add_captured_var(a_u8_other)
        self.assertNotEqual(left, right)

    def test_pickle_vars(self):
        """Test vars preserved through pickle."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        # Check inputs.
        dag = DAGCircuit()
        dag.add_input_var(a)

        self.assertEqual(dag.num_vars, 1)
        self.assertEqual(dag.num_input_vars, 1)

        with io.BytesIO() as buf:
            pickle.dump(dag, buf)
            buf.seek(0)
            output = pickle.load(buf)

        self.assertEqual(output.num_vars, 1)
        self.assertEqual(output.num_input_vars, 1)
        self.assertEqual(output, dag)

        # Check captures and declarations.
        dag = DAGCircuit()
        dag.add_declared_var(a)
        dag.add_captured_var(b)

        self.assertEqual(dag.num_vars, 2)
        self.assertEqual(dag.num_captured_vars, 1)
        self.assertEqual(dag.num_declared_vars, 1)

        with io.BytesIO() as buf:
            pickle.dump(dag, buf)
            buf.seek(0)
            output = pickle.load(buf)

        self.assertEqual(output.num_vars, 2)
        self.assertEqual(output.num_captured_vars, 1)
        self.assertEqual(output.num_declared_vars, 1)
        self.assertEqual(output, dag)

    def test_deepcopy_vars(self):
        """Test vars preserved through deepcopy."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        # Check inputs.
        dag = DAGCircuit()
        dag.add_input_var(a)

        self.assertEqual(dag.num_vars, 1)
        self.assertEqual(dag.num_input_vars, 1)

        output = copy.deepcopy(dag)

        self.assertEqual(output.num_vars, 1)
        self.assertEqual(output.num_input_vars, 1)
        self.assertEqual(output, dag)

        # Check captures and declarations.
        dag = DAGCircuit()
        dag.add_declared_var(a)
        dag.add_captured_var(b)

        self.assertEqual(dag.num_vars, 2)
        self.assertEqual(dag.num_captured_vars, 1)
        self.assertEqual(dag.num_declared_vars, 1)

        output = copy.deepcopy(dag)

        self.assertEqual(output.num_vars, 2)
        self.assertEqual(output.num_captured_vars, 1)
        self.assertEqual(output.num_declared_vars, 1)
        self.assertEqual(output, dag)

    def test_wires_added_for_simple_classical_vars(self):
        """Var uses should be represented in the wire structure."""
        a = expr.Var.new("a", types.Bool())
        dag = DAGCircuit()
        dag.add_input_var(a)
        self.assertEqual(list(dag.iter_vars()), [a])
        self.assertEqual(list(dag.iter_input_vars()), [a])
        self.assertEqual(list(dag.iter_captured_vars()), [])
        self.assertEqual(list(dag.iter_declared_vars()), [])

        expected_nodes = [dag.input_map[a], dag.output_map[a]]
        self.assertEqual(list(dag.topological_nodes()), expected_nodes)
        self.assertTrue(dag.is_successor(dag.input_map[a], dag.output_map[a]))

        op_mid = dag.apply_operation_back(Store(a, expr.lift(True)), (), ())
        self.assertTrue(dag.is_successor(dag.input_map[a], op_mid))
        self.assertTrue(dag.is_successor(op_mid, dag.output_map[a]))
        self.assertFalse(dag.is_successor(dag.input_map[a], dag.output_map[a]))

        op_front = dag.apply_operation_front(Store(a, expr.logic_not(a)), (), ())
        self.assertTrue(dag.is_successor(dag.input_map[a], op_front))
        self.assertTrue(dag.is_successor(op_front, op_mid))
        self.assertFalse(dag.is_successor(dag.input_map[a], op_mid))

        op_back = dag.apply_operation_back(Store(a, expr.logic_not(a)), (), ())
        self.assertTrue(dag.is_successor(op_mid, op_back))
        self.assertTrue(dag.is_successor(op_back, dag.output_map[a]))
        self.assertFalse(dag.is_successor(op_mid, dag.output_map[a]))

    def test_wires_added_for_var_control_flow_condition(self):
        """Vars used in if/else or while conditionals should be added to the wire structure."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        dag = DAGCircuit()
        dag.add_declared_var(a)
        dag.add_input_var(b)

        op_store = dag.apply_operation_back(Store(a, expr.lift(False)), (), ())
        op_if = dag.apply_operation_back(IfElseOp(a, QuantumCircuit()), (), ())
        op_while = dag.apply_operation_back(
            WhileLoopOp(expr.logic_or(a, b), QuantumCircuit()), (), ()
        )

        expected_edges = {
            (dag.input_map[a], op_store, a),
            (op_store, op_if, a),
            (op_if, op_while, a),
            (op_while, dag.output_map[a], a),
            (dag.input_map[b], op_while, b),
            (op_while, dag.output_map[b], b),
        }
        self.assertEqual(set(dag.edges()), expected_edges)

    def test_wires_added_for_var_control_flow_target(self):
        """Vars used in switch targets should be added to the wire structure."""
        a = expr.Var.new("a", types.Uint(8))
        b = expr.Var.new("b", types.Uint(8))
        dag = DAGCircuit()
        dag.add_declared_var(a)
        dag.add_input_var(b)

        op_store = dag.apply_operation_back(Store(a, expr.lift(3, a.type)), (), ())
        op_switch = dag.apply_operation_back(
            SwitchCaseOp(expr.bit_xor(a, b), [(CASE_DEFAULT, QuantumCircuit())]), (), ()
        )

        expected_edges = {
            (dag.input_map[a], op_store, a),
            (op_store, op_switch, a),
            (op_switch, dag.output_map[a], a),
            (dag.input_map[b], op_switch, b),
            (op_switch, dag.output_map[b], b),
        }
        self.assertEqual(set(dag.edges()), expected_edges)

    def test_wires_added_for_control_flow_captures(self):
        """Vars captured in control-flow blocks should be in the wire structure."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Uint(8))
        dag = DAGCircuit()
        dag.add_input_var(a)
        dag.add_input_var(b)
        dag.add_declared_var(c)
        dag.add_input_var(d)
        op_store = dag.apply_operation_back(Store(c, expr.lift(False)), (), ())
        op_if = dag.apply_operation_back(IfElseOp(a, QuantumCircuit(captures=[b])), (), ())
        op_switch = dag.apply_operation_back(
            SwitchCaseOp(
                d,
                [
                    (0, QuantumCircuit(captures=[b])),
                    (CASE_DEFAULT, QuantumCircuit(captures=[c])),
                ],
            ),
            (),
            (),
        )

        expected_edges = {
            # a
            (dag.input_map[a], op_if, a),
            (op_if, dag.output_map[a], a),
            # b
            (dag.input_map[b], op_if, b),
            (op_if, op_switch, b),
            (op_switch, dag.output_map[b], b),
            # c
            (dag.input_map[c], op_store, c),
            (op_store, op_switch, c),
            (op_switch, dag.output_map[c], c),
            # d
            (dag.input_map[d], op_switch, d),
            (op_switch, dag.output_map[d], d),
        }
        self.assertEqual(set(dag.edges()), expected_edges)

    def test_get_var_by_name(self):
        """Look up input, captured, and declared variables by name."""
        input_var = expr.Var.new("input", types.Bool())
        dag = DAGCircuit()
        dag.add_input_var(input_var)
        self.assertEqual(dag.get_var("input"), input_var)
        self.assertIsNone(dag.get_var("missing"))

        captured_var = expr.Var.new("captured", types.Bool())
        declared_var = expr.Var.new("declared", types.Bool())
        dag = DAGCircuit()
        dag.add_captured_var(captured_var)
        dag.add_declared_var(declared_var)
        self.assertEqual(dag.get_var("captured"), captured_var)
        self.assertEqual(dag.get_var("declared"), declared_var)
        self.assertIsNone(dag.get_var("missing"))

    def test_get_stretch_by_name(self):
        """Look up captured and declared stretches by name."""
        captured = expr.Stretch.new("captured")
        declared = expr.Stretch.new("declared")
        dag = DAGCircuit()
        dag.add_captured_stretch(captured)
        dag.add_declared_stretch(declared)
        self.assertEqual(dag.get_stretch("captured"), captured)
        self.assertEqual(dag.get_stretch("declared"), declared)
        self.assertIsNone(dag.get_stretch("missing"))

    def test_forbid_mixing_captures_inputs(self):
        """Test that a DAG can't have both captures and inputs."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Stretch.new("c")

        dag = DAGCircuit()
        dag.add_input_var(a)
        with self.assertRaisesRegex(
            DAGCircuitError, "circuits with input variables cannot be enclosed"
        ):
            dag.add_captured_var(b)
        with self.assertRaisesRegex(
            DAGCircuitError, "circuits with input variables cannot be enclosed"
        ):
            dag.add_captured_stretch(c)

        dag = DAGCircuit()
        dag.add_captured_var(a)
        with self.assertRaisesRegex(
            DAGCircuitError, "circuits to be enclosed with captures cannot have input variables"
        ):
            dag.add_input_var(b)

        dag = DAGCircuit()
        dag.add_captured_stretch(c)
        with self.assertRaisesRegex(
            DAGCircuitError, "circuits to be enclosed with captures cannot have input variables"
        ):
            dag.add_input_var(a)

    def test_forbid_adding_nonstandalone_var(self):
        """Temporary "wrapping" vars aren't standalone and can't be tracked separately."""
        dag = DAGCircuit()
        with self.assertRaisesRegex(DAGCircuitError, "cannot add variables that wrap"):
            dag.add_input_var(expr.lift(ClassicalRegister(4, "c")))
        with self.assertRaisesRegex(DAGCircuitError, "cannot add variables that wrap"):
            dag.add_declared_var(expr.lift(Clbit()))

    def test_forbid_adding_conflicting_vars(self):
        """Can't re-add a variable that exists, nor a shadowing variable in the same scope."""
        a1 = expr.Var.new("a", types.Bool())
        a2 = expr.Var.new("a", types.Bool())
        a3 = expr.Stretch.new("a")
        dag = DAGCircuit()
        dag.add_declared_var(a1)
        with self.assertRaisesRegex(DAGCircuitError, "already present in the circuit"):
            dag.add_declared_var(a1)
        with self.assertRaisesRegex(DAGCircuitError, "cannot add .* as its name shadows"):
            dag.add_declared_var(a2)
        with self.assertRaisesRegex(DAGCircuitError, "cannot add .* as its name shadows"):
            dag.add_declared_stretch(a3)

        dag = DAGCircuit()
        dag.add_declared_stretch(a3)
        with self.assertRaisesRegex(DAGCircuitError, "already present in the circuit"):
            dag.add_captured_stretch(a3)
        with self.assertRaisesRegex(DAGCircuitError, "cannot add .* as its name shadows"):
            dag.add_declared_var(a1)
        with self.assertRaisesRegex(DAGCircuitError, "cannot add .* as its name shadows"):
            dag.add_declared_var(a2)

    def test_pickle_stretches(self):
        """Test stretches preserved through pickle."""
        a = expr.Stretch.new("a")
        b = expr.Stretch.new("b")

        # Check captures and declarations.
        dag = DAGCircuit()
        dag.add_declared_stretch(a)
        dag.add_captured_stretch(b)

        self.assertEqual(dag.num_stretches, 2)
        self.assertEqual(dag.num_captured_stretches, 1)
        self.assertEqual(dag.num_declared_stretches, 1)

        with io.BytesIO() as buf:
            pickle.dump(dag, buf)
            buf.seek(0)
            output = pickle.load(buf)

        self.assertEqual(output.num_stretches, 2)
        self.assertEqual(output.num_captured_stretches, 1)
        self.assertEqual(output.num_declared_stretches, 1)
        self.assertEqual(output, dag)

    def test_deepcopy_stretches(self):
        """Test stretches preserved through deepcopy."""
        a = expr.Stretch.new("a")
        b = expr.Stretch.new("b")

        # Check captures and declarations.
        dag = DAGCircuit()
        dag.add_declared_stretch(a)
        dag.add_captured_stretch(b)

        self.assertEqual(dag.num_stretches, 2)
        self.assertEqual(dag.num_captured_stretches, 1)
        self.assertEqual(dag.num_declared_stretches, 1)

        output = copy.deepcopy(dag)

        self.assertEqual(output.num_stretches, 2)
        self.assertEqual(output.num_captured_stretches, 1)
        self.assertEqual(output.num_declared_stretches, 1)
        self.assertEqual(output, dag)


class TestDagSubstitute(DAGTest):
    """Test substituting a dag node with a sub-dag"""

    def setUp(self):
        super().setUp()
        self.dag = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg)

        self.qubit0 = qreg[0]
        self.qubit1 = qreg[1]
        self.qubit2 = qreg[2]
        self.clbit0 = creg[0]
        self.clbit1 = creg[1]
        self.condition = (creg, 3)
        self.dag.apply_operation_back(HGate(), [self.qubit0], [])
        self.dag.apply_operation_back(CXGate(), [self.qubit0, self.qubit1], [])
        self.dag.apply_operation_back(XGate(), [self.qubit1], [])

    def test_substitute_circuit_one_middle(self):
        """The method substitute_node_with_dag() replaces a in-the-middle node with a DAG."""
        cx_node = self.dag.op_nodes(op=CXGate).pop()

        flipped_cx_circuit = DAGCircuit()
        v = QuantumRegister(2, "v")
        flipped_cx_circuit.add_qreg(v)
        flipped_cx_circuit.apply_operation_back(HGate(), [v[0]], [])
        flipped_cx_circuit.apply_operation_back(HGate(), [v[1]], [])
        flipped_cx_circuit.apply_operation_back(CXGate(), [v[1], v[0]], [])
        flipped_cx_circuit.apply_operation_back(HGate(), [v[0]], [])
        flipped_cx_circuit.apply_operation_back(HGate(), [v[1]], [])

        self.dag.substitute_node_with_dag(cx_node, flipped_cx_circuit, wires=[v[0], v[1]])

        self.assertEqual(self.dag.count_ops()["h"], 5)
        expected = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        expected.add_qreg(qreg)
        expected.add_creg(creg)
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[1]], [])
        expected.apply_operation_back(CXGate(), [qreg[1], qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(HGate(), [qreg[1]], [])
        expected.apply_operation_back(XGate(), [qreg[1]], [])
        self.assertEqual(self.dag, expected)

    def test_substitute_circuit_one_front(self):
        """The method substitute_node_with_dag() replaces a leaf-in-the-front node with a DAG."""
        circuit = DAGCircuit()
        v = QuantumRegister(1, "v")
        circuit.add_qreg(v)
        circuit.apply_operation_back(HGate(), [v[0]], [])
        circuit.apply_operation_back(XGate(), [v[0]], [])

        self.dag.substitute_node_with_dag(next(self.dag.topological_op_nodes()), circuit)
        expected = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        expected.add_qreg(qreg)
        expected.add_creg(creg)
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(XGate(), [qreg[0]], [])
        expected.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])
        expected.apply_operation_back(XGate(), [qreg[1]], [])
        self.assertEqual(self.dag, expected)

    def test_substitute_circuit_one_back(self):
        """The method substitute_node_with_dag() replaces a leaf-in-the-back node with a DAG."""
        circuit = DAGCircuit()
        v = QuantumRegister(1, "v")
        circuit.add_qreg(v)
        circuit.apply_operation_back(HGate(), [v[0]], [])
        circuit.apply_operation_back(XGate(), [v[0]], [])

        self.dag.substitute_node_with_dag(list(self.dag.topological_op_nodes())[2], circuit)
        expected = DAGCircuit()
        qreg = QuantumRegister(3, "qr")
        creg = ClassicalRegister(2, "cr")
        expected.add_qreg(qreg)
        expected.add_creg(creg)
        expected.apply_operation_back(HGate(), [qreg[0]], [])
        expected.apply_operation_back(CXGate(), [qreg[0], qreg[1]], [])
        expected.apply_operation_back(HGate(), [qreg[1]], [])
        expected.apply_operation_back(XGate(), [qreg[1]], [])

        self.assertEqual(self.dag, expected)

    def test_substitute_dag_if_else_expr(self):
        """Test that an `IfElseOp` with an `Expr` condition can be substituted for a DAG that
        contains an `IfElseOp` with an `Expr` condition."""

        body_rep = QuantumCircuit(1)
        body_rep.z(0)

        q_rep = QuantumRegister(1)
        c_rep = ClassicalRegister(2)
        replacement = DAGCircuit()
        replacement.add_qreg(q_rep)
        replacement.add_creg(c_rep)
        replacement.apply_operation_back(XGate(), [q_rep[0]], [])
        replacement.apply_operation_back(
            IfElseOp(expr.logic_not(expr.bit_and(c_rep, 1)), body_rep, None), [q_rep[0]], []
        )

        true_src = QuantumCircuit(1)
        true_src.x(0)
        true_src.z(0)
        false_src = QuantumCircuit(1)
        false_src.x(0)
        q_src = QuantumRegister(4)
        c1_src = ClassicalRegister(2)
        c2_src = ClassicalRegister(2)
        src = DAGCircuit()
        src.add_qreg(q_src)
        src.add_creg(c1_src)
        src.add_creg(c2_src)
        node = src.apply_operation_back(
            IfElseOp(expr.logic_not(expr.bit_and(c1_src, 1)), true_src, false_src), [q_src[2]], []
        )

        wires = {q_rep[0]: q_src[2], c_rep[0]: c1_src[0], c_rep[1]: c1_src[1]}
        src.substitute_node_with_dag(node, replacement, wires=wires)

        expected = DAGCircuit()
        expected.add_qreg(q_src)
        expected.add_creg(c1_src)
        expected.add_creg(c2_src)
        expected.apply_operation_back(XGate(), [q_src[2]], [])
        expected.apply_operation_back(
            IfElseOp(expr.logic_not(expr.bit_and(c1_src, 1)), body_rep, None), [q_src[2]], []
        )

        self.assertEqual(src, expected)

    def test_substitute_dag_switch_expr(self):
        """Test that a `SwitchCaseOp` with an `Expr` target can be substituted for a DAG that
        contains another `SwitchCaseOp` with an `Expr` target."""

        case_rep = QuantumCircuit(1)
        case_rep.z(0)

        q_rep = QuantumRegister(1)
        c_rep = ClassicalRegister(2)
        replacement = DAGCircuit()
        replacement.add_qreg(q_rep)
        replacement.add_creg(c_rep)
        replacement.apply_operation_back(XGate(), [q_rep[0]], [])
        # This could logically be an `IfElseOp`, but the point is just to test the `target`.
        replacement.apply_operation_back(
            SwitchCaseOp(expr.equal(c_rep[0], c_rep[1]), [(True, case_rep)]), [q_rep[0]], []
        )

        same_src = QuantumCircuit(1)
        same_src.x(0)
        same_src.z(0)
        diff_src = QuantumCircuit(1)
        diff_src.x(0)
        q_src = QuantumRegister(4)
        c1_src = ClassicalRegister(2)
        c2_src = ClassicalRegister(2)
        src = DAGCircuit()
        src.add_qreg(q_src)
        src.add_creg(c1_src)
        src.add_creg(c2_src)
        node = src.apply_operation_back(
            SwitchCaseOp(expr.lift(c1_src), [((1, 2), diff_src), ((0, 3), same_src)]),
            [q_src[2]],
            [],
        )

        wires = {q_rep[0]: q_src[2], c_rep[0]: c1_src[0], c_rep[1]: c1_src[1]}
        src.substitute_node_with_dag(node, replacement, wires=wires)

        expected = DAGCircuit()
        expected.add_qreg(q_src)
        expected.add_creg(c1_src)
        expected.add_creg(c2_src)
        expected.apply_operation_back(XGate(), [q_src[2]], [])
        node = expected.apply_operation_back(
            SwitchCaseOp(expr.equal(c1_src[0], c1_src[1]), [(True, case_rep)]), [q_src[2]], []
        )

        self.assertEqual(src, expected)

    def test_substitute_dag_vars(self):
        """Should be possible to replace a node with a DAG acting on the same wires."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Bool())

        dag = DAGCircuit()
        dag.add_input_var(a)
        dag.add_input_var(b)
        dag.add_input_var(c)
        dag.apply_operation_back(Store(c, expr.lift(False)), (), ())
        node = dag.apply_operation_back(Store(a, expr.logic_or(expr.logic_or(a, b), c)), (), ())
        dag.apply_operation_back(Store(b, expr.lift(True)), (), ())

        replace = DAGCircuit()
        replace.add_captured_var(a)
        replace.add_captured_var(b)
        replace.add_captured_var(c)
        replace.apply_operation_back(Store(a, expr.logic_or(a, b)), (), ())
        replace.apply_operation_back(Store(a, expr.logic_or(a, c)), (), ())

        expected = DAGCircuit()
        expected.add_input_var(a)
        expected.add_input_var(b)
        expected.add_input_var(c)
        expected.apply_operation_back(Store(c, expr.lift(False)), (), ())
        expected.apply_operation_back(Store(a, expr.logic_or(a, b)), (), ())
        expected.apply_operation_back(Store(a, expr.logic_or(a, c)), (), ())
        expected.apply_operation_back(Store(b, expr.lift(True)), (), ())

        dag.substitute_node_with_dag(node, replace, wires={})

        self.assertEqual(dag, expected)

    def test_substitute_dag_if_else_expr_var(self):
        """Test that substitution works with if/else ops with standalone Vars."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        body_rep = QuantumCircuit(1)
        body_rep.z(0)

        q_rep = QuantumRegister(1)
        c_rep = ClassicalRegister(2)
        replacement = DAGCircuit()
        replacement.add_qreg(q_rep)
        replacement.add_creg(c_rep)
        replacement.add_captured_var(b)
        replacement.apply_operation_back(XGate(), [q_rep[0]], [])
        replacement.apply_operation_back(
            IfElseOp(expr.logic_and(b, expr.equal(c_rep, 1)), body_rep, None), [q_rep[0]], []
        )

        true_src = QuantumCircuit(1)
        true_src.x(0)
        true_src.z(0)
        false_src = QuantumCircuit(1)
        false_src.x(0)
        q_src = QuantumRegister(4)
        c1_src = ClassicalRegister(2)
        c2_src = ClassicalRegister(2)
        src = DAGCircuit()
        src.add_qreg(q_src)
        src.add_creg(c1_src)
        src.add_creg(c2_src)
        src.add_input_var(a)
        src.add_input_var(b)
        node = src.apply_operation_back(
            IfElseOp(expr.logic_and(b, expr.equal(c1_src, 1)), true_src, false_src), [q_src[2]], []
        )

        wires = {q_rep[0]: q_src[2], c_rep[0]: c1_src[0], c_rep[1]: c1_src[1]}
        src.substitute_node_with_dag(node, replacement, wires=wires)

        expected = DAGCircuit()
        expected.add_qreg(q_src)
        expected.add_creg(c1_src)
        expected.add_creg(c2_src)
        expected.add_input_var(a)
        expected.add_input_var(b)
        expected.apply_operation_back(XGate(), [q_src[2]], [])
        expected.apply_operation_back(
            IfElseOp(expr.logic_and(b, expr.equal(c1_src, 1)), body_rep, None), [q_src[2]], []
        )
        self.assertEqual(src, expected)

    def test_contract_var_use_to_nothing(self):
        """The replacement DAG can drop wires."""
        a = expr.Var.new("a", types.Bool())

        src = DAGCircuit()
        src.add_input_var(a)
        node = src.apply_operation_back(Store(a, a), (), ())
        replace = DAGCircuit()
        src.substitute_node_with_dag(node, replace, {})

        expected = DAGCircuit()
        expected.add_input_var(a)
        self.assertEqual(src, expected)

    def test_raise_if_var_mismatch(self):
        """The DAG can't add more wires."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        src = DAGCircuit()
        src.add_input_var(a)
        node = src.apply_operation_back(Store(a, a), (), ())

        replace = DAGCircuit()
        replace.add_input_var(a)
        replace.add_input_var(b)
        replace.apply_operation_back(Store(a, b), (), ())

        with self.assertRaisesRegex(DAGCircuitError, "Cannot replace a node with a DAG with more"):
            src.substitute_node_with_dag(node, replace, wires={})

    def test_substitute_node_with_dag_transfers_captured_and_declared_vars(self):
        """Test that substitute_node_with_dag transfers captured and declared variables.
        Regression test for gh-15509."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        # Create a base DAG with a simple X gate
        base_dag = DAGCircuit()
        qr = QuantumRegister(1)
        base_dag.add_qreg(qr)
        x_node = base_dag.apply_operation_back(XGate(), [qr[0]], [])

        # Create a replacement DAG with captured and declared variables
        replacement_dag = DAGCircuit()
        replacement_dag.add_qubits([qr[0]])
        replacement_dag.add_captured_var(a)
        replacement_dag.add_declared_var(b)
        replacement_dag.apply_operation_back(XGate(), [qr[0]], [])

        # Perform the substitution
        base_dag.substitute_node_with_dag(x_node, replacement_dag, wires=[qr[0]])

        # Verify the variables were transferred
        self.assertEqual(base_dag.num_captured_vars, 1)
        self.assertEqual(base_dag.num_declared_vars, 1)
        self.assertEqual(list(base_dag.iter_captured_vars()), [a])
        self.assertEqual(list(base_dag.iter_declared_vars()), [b])

    def test_substitute_dag_transfers_input_vars(self):
        """substitute_node_with_dag should transfer DAG-level input vars from replacement."""
        # Base DAG with a simple X gate
        base_qc = QuantumCircuit(1)
        base_qc.x(0)
        base_dag = circuit_to_dag(base_qc)

        # Node to replace
        x_node = list(base_dag.op_nodes())[0]

        # Replacement DAG that declares an input var and uses it in a control-flow op
        replacement_dag = DAGCircuit()
        replacement_dag.add_qubits(x_node.qargs)
        replacement_dag.add_clbits(x_node.cargs)

        condition_var = expr.Var.new("condition", types.Bool())
        replacement_dag.add_input_var(condition_var)

        if_block = QuantumCircuit(1)
        if_block.x(0)
        if_else_op = IfElseOp(condition_var, if_block, None)
        replacement_dag.apply_operation_back(
            if_else_op, replacement_dag.qubits, replacement_dag.clbits
        )

        # Perform substitution and verify input vars transferred
        base_dag.substitute_node_with_dag(
            x_node, replacement_dag, wires=x_node.qargs + x_node.cargs
        )

        self.assertEqual(base_dag.num_input_vars, 1)
        self.assertEqual(list(base_dag.iter_input_vars()), [condition_var])

    def test_substitute_node_with_dag_transfers_captured_and_declared_stretches(self):
        """substitute_node_with_dag should transfer captured and declared stretches."""
        a = expr.Stretch.new("a")
        b = expr.Stretch.new("b")

        # Create a base DAG with a simple X gate
        base_dag = DAGCircuit()
        qr = QuantumRegister(1)
        base_dag.add_qreg(qr)
        x_node = base_dag.apply_operation_back(XGate(), [qr[0]], [])

        # Create a replacement DAG with captured and declared stretches
        replacement_dag = DAGCircuit()
        replacement_dag.add_qubits([qr[0]])
        replacement_dag.add_captured_stretch(a)
        replacement_dag.add_declared_stretch(b)
        replacement_dag.apply_operation_back(XGate(), [qr[0]], [])

        # Perform the substitution
        base_dag.substitute_node_with_dag(x_node, replacement_dag, wires=[qr[0]])

        # Verify the stretches were transferred
        self.assertEqual(base_dag.num_captured_stretches, 1)
        self.assertEqual(base_dag.num_declared_stretches, 1)
        self.assertEqual(list(base_dag.iter_captured_stretches()), [a])
        self.assertEqual(list(base_dag.iter_declared_stretches()), [b])


@ddt
class TestDagSubstituteNode(DAGTest):
    """Test substituting a dagnode with a node."""

    def test_substituting_node_with_wrong_width_node_raises(self):
        """Verify replacing a node with one of a different shape raises."""
        dag = DAGCircuit()
        qr = QuantumRegister(2)
        dag.add_qreg(qr)
        node_to_be_replaced = dag.apply_operation_back(CXGate(), [qr[0], qr[1]])

        with self.assertRaises(DAGCircuitError) as _:
            dag.substitute_node(node_to_be_replaced, Measure())

    @data(True, False)
    def test_substituting_io_node_raises(self, inplace):
        """Verify replacing an io node raises."""
        dag = DAGCircuit()
        qr = QuantumRegister(1)
        dag.add_qreg(qr)

        io_node = next(dag.nodes())

        with self.assertRaises(DAGCircuitError) as _:
            dag.substitute_node(io_node, HGate(), inplace=inplace)

    @data(True, False)
    def test_substituting_node_preserves_parents_children(self, inplace):
        """Verify parents and children are preserved by a substitution."""
        qc = QuantumCircuit(3, 2)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(0.1, 2)
        qc.cx(1, 2)
        qc.cx(0, 1)
        dag = circuit_to_dag(qc)
        node_to_be_replaced = dag.named_nodes("rz")[0]
        predecessors = set(dag.predecessors(node_to_be_replaced))
        successors = set(dag.successors(node_to_be_replaced))
        ancestors = dag.ancestors(node_to_be_replaced)
        descendants = dag.descendants(node_to_be_replaced)

        replacement_node = dag.substitute_node(node_to_be_replaced, U1Gate(0.1), inplace=inplace)

        self.raise_if_dagcircuit_invalid(dag)
        self.assertEqual(set(dag.predecessors(replacement_node)), predecessors)
        self.assertEqual(set(dag.successors(replacement_node)), successors)
        self.assertEqual(dag.ancestors(replacement_node), ancestors)
        self.assertEqual(dag.descendants(replacement_node), descendants)
        self.assertEqual(replacement_node is node_to_be_replaced, inplace)

    @data(True, False)
    def test_replace_if_else_op_with_another(self, inplace):
        """Test that one `IfElseOp` can be replaced with another."""
        body = QuantumCircuit(1)
        body.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(IfElseOp(expr.logic_not(cr1), body.copy(), None), qr, [])
        self.assertEqual(dag.num_blocks(), 1)  # Sanity check.
        dag.substitute_node(node, IfElseOp(expr.equal(cr1, 0), body.copy(), None), inplace=inplace)

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_creg(cr1)
        expected.add_creg(cr2)
        expected.apply_operation_back(IfElseOp(expr.equal(cr1, 0), body.copy(), None), qr, [])

        self.assertEqual(dag, expected)
        # If the below assertion fails, `substitute_node` most likely failed to track the refcounts
        # correctly / failed to free a block after the substitution.
        self.assertEqual(dag.num_blocks(), 1)

    @data(True, False)
    def test_reject_replace_if_else_op_with_other_resources(self, inplace):
        """Test that the resources in the `condition` of a `IfElseOp` are checked against those in
        the node to be replaced."""
        body = QuantumCircuit(1)
        body.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(IfElseOp(expr.logic_not(cr1), body.copy(), None), qr, [])

        with self.assertRaisesRegex(DAGCircuitError, "does not span the same wires"):
            dag.substitute_node(
                node, IfElseOp(expr.logic_not(cr2), body.copy(), None), inplace=inplace
            )

    @data(True, False)
    def test_replace_switch_with_another(self, inplace):
        """Test that one `SwitchCaseOp` can be replaced with another."""
        case = QuantumCircuit(1)
        case.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(
            SwitchCaseOp(expr.lift(cr1), [((1, 3), case.copy())]), qr, []
        )
        dag.substitute_node(
            node, SwitchCaseOp(expr.bit_and(cr1, 1), [(1, case.copy())]), inplace=inplace
        )

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_creg(cr1)
        expected.add_creg(cr2)
        expected.apply_operation_back(
            SwitchCaseOp(expr.bit_and(cr1, 1), [(1, case.copy())]), qr, []
        )

        self.assertEqual(dag, expected)

    @data(True, False)
    def test_reject_replace_switch_with_other_resources(self, inplace):
        """Test that the resources in the `target` of a `SwitchCaseOp` are checked against those in
        the node to be replaced."""
        case = QuantumCircuit(1)
        case.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(2)
        cr2 = ClassicalRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        node = dag.apply_operation_back(
            SwitchCaseOp(expr.lift(cr1), [((1, 3), case.copy())]), qr, []
        )

        with self.assertRaisesRegex(DAGCircuitError, "does not span the same wires"):
            dag.substitute_node(
                node, SwitchCaseOp(expr.lift(cr2), [((1, 3), case.copy())]), inplace=inplace
            )

    @data(True, False)
    def test_replace_switch_case_standalone_var(self, inplace):
        """Replace a standalone-Var switch/case with another."""
        a = expr.Var.new("a", types.Uint(8))
        b = expr.Var.new("b", types.Uint(8))

        case = QuantumCircuit(1)
        case.x(0)

        qr = QuantumRegister(1)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_input_var(a)
        dag.add_input_var(b)
        node = dag.apply_operation_back(SwitchCaseOp(a, [((1, 3), case.copy())]), qr, [])
        dag.substitute_node(
            node, SwitchCaseOp(expr.bit_and(a, 1), [(1, case.copy())]), inplace=inplace
        )

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_input_var(a)
        expected.add_input_var(b)
        expected.apply_operation_back(SwitchCaseOp(expr.bit_and(a, 1), [(1, case.copy())]), qr, [])

        self.assertEqual(dag, expected)

    @data(True, False)
    def test_replace_store_standalone_var(self, inplace):
        """Replace a standalone-Var Store with another."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        qr = QuantumRegister(1)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_input_var(a)
        dag.add_input_var(b)
        node = dag.apply_operation_back(Store(a, a), (), ())
        dag.substitute_node(node, Store(a, expr.logic_not(a)), inplace=inplace)

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_input_var(a)
        expected.add_input_var(b)
        expected.apply_operation_back(Store(a, expr.logic_not(a)), (), ())

        self.assertEqual(dag, expected)


class TestReplaceBlock(DAGTest):
    """Test replacing a block of nodes in a DAG."""

    def setUp(self):
        super().setUp()
        self.qc = QuantumCircuit(2)
        self.qc.cx(0, 1)
        self.qc.h(1)
        self.qc.cx(0, 1)
        self.dag = circuit_to_dag(self.qc)

    def test_cycle_check(self):
        """Validate that by default introducing a cycle errors."""
        nodes = list(self.dag.topological_op_nodes())
        with self.assertRaises(DAGCircuitError):
            self.dag.replace_block_with_op(
                [nodes[0], nodes[-1]],
                CZGate(),
                {bit: idx for (idx, bit) in enumerate(self.dag.qubits)},
            )

    def test_empty(self):
        """Test that an empty node block raises."""
        with self.assertRaises(DAGCircuitError):
            self.dag.replace_block_with_op(
                [], CZGate(), {bit: idx for (idx, bit) in enumerate(self.dag.qubits)}
            )

    def test_single_node_block(self):
        """Test that a single node as the block works."""
        qr = QuantumRegister(2)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        node = dag.apply_operation_back(HGate(), [qr[0]])
        new_node = dag.replace_block_with_op(
            [node], XGate(), {bit: idx for (idx, bit) in enumerate(dag.qubits)}
        )

        expected_dag = DAGCircuit()
        expected_dag.add_qreg(qr)
        expected_dag.apply_operation_back(XGate(), [qr[0]])

        self.assertEqual(expected_dag, dag)
        self.assertEqual(expected_dag.count_ops(), dag.count_ops())
        self.assertIsInstance(new_node.op, XGate)

    def test_invalid_replacement_size(self):
        """Test inserting an operation on a wrong number of qubits raises."""

        # two X gates, normal circuit
        qc = QuantumCircuit(2)
        qc.x(range(2))

        # mutilate the DAG
        dag = circuit_to_dag(qc)
        to_replace = list(dag.op_nodes())
        new_node = XGate()
        idx_map = {node.qargs[0]: i for i, node in enumerate(to_replace)}

        with self.assertRaises(DAGCircuitError):
            dag.replace_block_with_op(to_replace, new_node, idx_map)

    def test_replace_control_flow_block(self):
        """Test that we can replace a block of control-flow nodes with a single one."""
        body = QuantumCircuit(1)
        body.x(0)

        qr = QuantumRegister(1)
        cr1 = ClassicalRegister(3)
        cr2 = ClassicalRegister(3)
        dag = DAGCircuit()
        dag.add_qreg(qr)
        dag.add_creg(cr1)
        dag.add_creg(cr2)
        nodes = [
            dag.apply_operation_back(IfElseOp(expr.logic_not(cr1[0]), body.copy(), None), qr, []),
            dag.apply_operation_back(
                SwitchCaseOp(expr.lift(cr2), [((0, 1, 2, 3), body.copy())]), qr, []
            ),
        ]

        dag.replace_block_with_op(
            nodes,
            IfElseOp(expr.logic_or(expr.logic_not(cr1[0]), expr.less(cr2, 4)), body.copy(), None),
            {qr[0]: 0},
        )

        expected = DAGCircuit()
        expected.add_qreg(qr)
        expected.add_creg(cr1)
        expected.add_creg(cr2)
        expected.apply_operation_back(
            IfElseOp(expr.logic_or(expr.logic_not(cr1[0]), expr.less(cr2, 4)), body.copy(), None),
            qr,
            [],
        )

        self.assertEqual(dag, expected)

    def test_contract_stores(self):
        """Test that contraction over nodes with `Var` wires works."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())
        c = expr.Var.new("c", types.Bool())

        dag = DAGCircuit()
        dag.add_input_var(a)
        dag.add_input_var(b)
        dag.add_input_var(c)
        dag.apply_operation_back(Store(c, expr.lift(False)), (), ())
        nodes = [
            dag.apply_operation_back(Store(a, expr.logic_or(a, b)), (), ()),
            dag.apply_operation_back(Store(a, expr.logic_or(a, c)), (), ()),
        ]
        dag.apply_operation_back(Store(b, expr.lift(True)), (), ())
        dag.replace_block_with_op(nodes, Store(a, expr.logic_or(expr.logic_or(a, b), c)), {})

        expected = DAGCircuit()
        expected.add_input_var(a)
        expected.add_input_var(b)
        expected.add_input_var(c)
        expected.apply_operation_back(Store(c, expr.lift(False)), (), ())
        expected.apply_operation_back(Store(a, expr.logic_or(expr.logic_or(a, b), c)), (), ())
        expected.apply_operation_back(Store(b, expr.lift(True)), (), ())

        self.assertEqual(dag, expected)


class TestDagProperties(DAGTest):
    """Test the DAG properties."""

    def setUp(self):
        #       ┌───┐                ┌───┐
        # q0_0: ┤ H ├────────────────┤ X ├──────────
        #       └───┘                └─┬─┘     ┌───┐
        # q0_1: ───────────────────────┼───────┤ H ├
        #                 ┌───┐        │  ┌───┐└─┬─┘
        # q0_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #       ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # q0_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #       └───┘└──────────────┘  │
        # q1_0: ───────────────────────■────────────
        #                              │
        # q1_1: ───────────────────────■────────────
        super().setUp()
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[2], qr1[3])
        circ.h(qr1[2])
        circ.t(qr1[2])
        circ.ch(qr1[2], qr1[1])
        circ.u(0.0, 0.1, 0.2, qr1[3])
        circ.ccx(qr2[0], qr2[1], qr1[0])

        self.dag = circuit_to_dag(circ)

    def test_dag_size(self):
        """Test total number of operations in dag."""
        self.assertEqual(self.dag.size(), 7)

    def test_dag_depth(self):
        """Test dag depth."""
        self.assertEqual(self.dag.depth(), 4)

    def test_dag_width(self):
        """Test number of qubits + clbits in dag."""
        self.assertEqual(self.dag.width(), 6)

    def test_dag_num_qubits(self):
        """Test number of qubits in dag."""
        self.assertEqual(self.dag.num_qubits(), 6)

    def test_dag_operations(self):
        """Test dag operations breakdown by kind of op."""
        operations = {"h": 2, "t": 1, "u": 1, "cx": 1, "ch": 1, "ccx": 1}

        self.assertDictEqual(self.dag.count_ops(), operations)

    def test_dag_factors(self):
        """Test number of separable factors in circuit."""
        self.assertEqual(self.dag.num_tensor_factors(), 2)

    def test_dag_depth_empty(self):
        """Empty circuit DAG is zero depth"""
        q = QuantumRegister(5, "q")
        qc = QuantumCircuit(q)
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 0)

    def test_dag_idle_wires(self):
        """Test dag idle_wires."""
        wires = list(self.dag.idle_wires())
        self.assertEqual(len(wires), 0)
        wires = list(self.dag.idle_wires(["u", "cx"]))
        self.assertEqual(len(wires), 1)

    def test_dag_depth1(self):
        """Test DAG depth #1"""
        #       ┌───┐
        # q1_0: ┤ H ├──■────■─────────────────
        #       ├───┤  │  ┌─┴─┐
        # q1_1: ┤ H ├──┼──┤ X ├──■────────────
        #       ├───┤  │  └───┘  │  ┌───┐
        # q1_2: ┤ H ├──┼─────────┼──┤ X ├──■──
        #       ├───┤┌─┴─┐       │  └─┬─┘┌─┴─┐
        # q2_0: ┤ H ├┤ X ├───────┼────┼──┤ X ├
        #       ├───┤└─┬─┘     ┌─┴─┐  │  └───┘
        # q2_1: ┤ H ├──■───────┤ X ├──■───────
        #       └───┘          └───┘
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(2, "q2")
        c = ClassicalRegister(5, "c")
        qc = QuantumCircuit(qr1, qr2, c)
        qc.h(qr1[0])
        qc.h(qr1[1])
        qc.h(qr1[2])
        qc.h(qr2[0])
        qc.h(qr2[1])
        qc.ccx(qr2[1], qr1[0], qr2[0])
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr2[1])
        qc.cx(qr2[1], qr1[2])
        qc.cx(qr1[2], qr2[0])
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 6)

    def test_dag_depth2(self):
        """Test barrier increases DAG depth"""

        #      ┌───┐                     ░
        # q_0: ┤ H ├──■──────────────────░────
        #      └───┘  │            ┌───┐ ░ ┌─┐
        # q_1: ───────┼────────────┤ X ├─░─┤M├
        #      ┌───┐  │  ┌───┐┌───┐└─┬─┘ ░ └╥┘
        # q_2: ┤ X ├──┼──┤ X ├┤ X ├──┼───░──╫─
        #      └───┘  │  └───┘└───┘  │   ░  ║
        # q_3: ───────┼──────────────┼───░──╫─
        #           ┌─┴─┐┌───┐       │   ░  ║
        # q_4: ─────┤ X ├┤ X ├───────■───░──╫─
        #           └───┘└───┘           ░  ║
        # c: 1/═════════════════════════════╩═
        #                                   0
        q = QuantumRegister(5, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[4])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[2])
        qc.x(q[4])
        qc.cx(q[4], q[1])
        qc.barrier(q)
        qc.measure(q[1], c[0])
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 6)

    def test_dag_depth3(self):
        """Test DAG depth for silly circuit."""

        #      ┌───┐       ░    ░       ┌─┐
        # q_0: ┤ H ├──■────░────░───────┤M├─────
        #      └───┘┌─┴─┐  ░    ░       └╥┘
        # q_1: ─────┤ X ├──■─────────────╫──────
        #           └───┘┌─┴─┐           ║
        # q_2: ──────────┤ X ├──■────────╫──────
        #                └───┘┌─┴─┐      ║
        # q_3: ───────────────┤ X ├──■───╫──────
        #                     └───┘┌─┴─┐ ║
        # q_4: ────────────────────┤ X ├─╫───■──
        #                          └───┘ ║ ┌─┴─┐
        # q_5: ──────────────────────────╫─┤ X ├
        #                                ║ └───┘
        # c: 1/══════════════════════════╩══════
        #                                0
        q = QuantumRegister(6, "q")
        c = ClassicalRegister(1, "c")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.cx(q[0], q[1])
        qc.cx(q[1], q[2])
        qc.cx(q[2], q[3])
        qc.cx(q[3], q[4])
        qc.cx(q[4], q[5])
        qc.barrier(q[0])
        qc.barrier(q[0])
        qc.measure(q[0], c[0])
        dag = circuit_to_dag(qc)
        self.assertEqual(dag.depth(), 6)


class TestSwapNodes(DAGTest):
    """Test Swapping connected nodes."""

    def test_1q_swap_fully_connected(self):
        """Test swapping single qubit gates"""
        dag = DAGCircuit()
        qreg = QuantumRegister(1)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(RXGate(pi / 2), [qreg[0]], [])
        op_node1 = dag.apply_operation_back(RXGate(pi), [qreg[0]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(RXGate(pi), [qreg[0]], [])
        expected.apply_operation_back(RXGate(pi / 2), [qreg[0]], [])

        self.assertEqual(dag, expected)

    def test_2q_swap_fully_connected(self):
        """test swapping full connected 2q gates"""
        dag = DAGCircuit()
        qreg = QuantumRegister(2)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[0, 1]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 0]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(CZGate(), qreg[[1, 0]], [])
        expected.apply_operation_back(CXGate(), qreg[[0, 1]], [])

        self.assertEqual(dag, expected)

    def test_2q_swap_partially_connected(self):
        """test swapping 2q partially connected gates"""
        dag = DAGCircuit()
        qreg = QuantumRegister(3)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        expected.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        self.assertEqual(dag, expected)

    def test_4q_swap_partially_connected(self):
        """test swapping 4q partially connected gates"""
        from qiskit.circuit.library.standard_gates import C3XGate

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 3]], [])
        op_node1 = dag.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 4]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 4]], [])
        expected.apply_operation_back(C3XGate(), qreg[[0, 1, 2, 3]], [])
        self.assertEqual(dag, expected)

    def test_2q_swap_non_connected_raises(self):
        """test swapping 2q non-connected gates raises an exception"""
        dag = DAGCircuit()
        qreg = QuantumRegister(4)
        dag.add_qreg(qreg)
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[0, 2]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 3]], [])
        self.assertRaises(DAGCircuitError, dag.swap_nodes, op_node0, op_node1)

    def test_2q_swap_partially_connected_pre_spectators(self):
        """test swapping 2q partially connected gates when in the presence of pre
        spectator ops."""
        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        dag.add_qreg(qreg)
        dag.apply_operation_back(XGate(), qreg[[0]])
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(XGate(), qreg[[0]])
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        expected.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        self.assertEqual(dag, expected)

    def test_2q_swap_partially_connected_prepost_spectators(self):
        """test swapping 2q partially connected gates when in the presence of pre
        and post spectator ops."""
        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        dag.add_qreg(qreg)
        dag.apply_operation_back(XGate(), qreg[[0]])
        op_node0 = dag.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        op_node1 = dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        dag.swap_nodes(op_node0, op_node1)

        expected = DAGCircuit()
        expected.add_qreg(qreg)
        expected.apply_operation_back(XGate(), qreg[[0]])
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        expected.apply_operation_back(CXGate(), qreg[[1, 0]], [])
        expected.apply_operation_back(CZGate(), qreg[[1, 2]], [])
        self.assertEqual(dag, expected)


class TestDagCausalCone(DAGTest):
    """Test `get_causal_node` function"""

    def test_causal_cone_regular_circuit(self):
        """Test causal cone with a regular circuit"""

        # q_0: ───────■─────────────
        #             │
        # q_1: ──■────┼───■─────────
        #      ┌─┴─┐  │   │
        # q_2: ┤ X ├──┼───┼──■──────
        #      └───┘┌─┴─┐ │  │
        # q_3: ─────┤ X ├─┼──┼───■──
        #           └───┘ │  │ ┌─┴─┐
        # q_4: ───────────■──■─┤ X ├
        #                      └───┘
        # c: 5/═════════════════════

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(CXGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CXGate(), qreg[[0, 3]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 4]], [])
        dag.apply_operation_back(CZGate(), qreg[[2, 4]], [])
        dag.apply_operation_back(CXGate(), qreg[[3, 4]], [])

        # Get causal cone of qubit at index 0
        result = dag.quantum_causal_cone(qreg[0])

        # Expected result
        expected = set(qreg[[0, 3]])
        self.assertEqual(result, expected)

    def test_causal_cone_invalid_qubit(self):
        """Test causal cone with invalid qubit"""

        # q_0: ───────■─────────────
        #             │
        # q_1: ──■────┼───■─────────
        #      ┌─┴─┐  │   │
        # q_2: ┤ X ├──┼───┼──■──────
        #      └───┘┌─┴─┐ │  │
        # q_3: ─────┤ X ├─┼──┼───■──
        #           └───┘ │  │ ┌─┴─┐
        # q_4: ───────────■──■─┤ X ├
        #                      └───┘
        # c: 5/═════════════════════

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(CXGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CXGate(), qreg[[0, 3]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 4]], [])
        dag.apply_operation_back(CZGate(), qreg[[2, 4]], [])
        dag.apply_operation_back(CXGate(), qreg[[3, 4]], [])

        # Raise error due to invalid index
        self.assertRaises(DAGCircuitError, dag.quantum_causal_cone, Qubit())

    def test_causal_cone_no_neighbor(self):
        """Test causal cone with no neighbor"""

        # q_0: ───────────

        # q_1: ──■───■────
        #      ┌─┴─┐ │
        # q_2: ┤ X ├─┼──■─
        #      ├───┤ │  │
        # q_3: ┤ X ├─┼──┼─
        #      └───┘ │  │
        # q_4: ──────■──■─

        # c: 5/═══════════

        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)
        dag.apply_operation_back(CXGate(), qreg[[1, 2]], [])
        dag.apply_operation_back(CZGate(), qreg[[1, 4]], [])
        dag.apply_operation_back(CZGate(), qreg[[2, 4]], [])
        dag.apply_operation_back(XGate(), qreg[[3]], [])

        # Get causal cone of Qubit at index 3.
        result = dag.quantum_causal_cone(qreg[3])
        # Expect only a set with Qubit at index 3
        expected = set(qreg[[3]])
        self.assertEqual(result, expected)

    def test_causal_cone_empty_circuit(self):
        """Test causal cone for circuit with no operations"""
        dag = DAGCircuit()
        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        dag.add_qreg(qreg)
        dag.add_creg(creg)

        # Get causal cone of qubit at index 4
        result = dag.quantum_causal_cone(qreg[4])
        # Expect only a set with Qubit at index 4
        expected = set(qreg[[4]])

        self.assertEqual(result, expected)

    def test_causal_cone_barriers(self):
        """Test causal cone for circuit with barriers"""

        #       ┌───┐      ░               ░
        # q1_0: ┤ X ├──■───░───────────────░───────────
        #       └───┘┌─┴─┐ ░               ░
        # q1_1: ─────┤ X ├─░───────■───■───░───────────
        #            └───┘ ░ ┌───┐ │   │   ░ ┌───┐
        # q1_2: ───────────░─┤ H ├─■───┼───░─┤ Y ├─────
        #                  ░ └───┘   ┌─┴─┐ ░ └─┬─┘┌───┐
        # q1_3: ───────────░─────────┤ Y ├─░───■──┤ X ├
        #                  ░         └───┘ ░      └─┬─┘
        # q1_4: ───────────░───────────────░────────■──
        #                  ░               ░

        # Build the circuit:
        qreg = QuantumRegister(5)
        qc = QuantumCircuit(qreg)
        qc.x(0)
        qc.cx(0, 1)
        qc.barrier()

        qc.h(2)
        qc.cz(2, 1)
        qc.cy(1, 3)
        qc.barrier()

        qc.cy(3, 2)
        qc.cx(4, 3)

        # Transform into a dag:
        dag = circuit_to_dag(qc)

        # Compute result:
        result = dag.quantum_causal_cone(qreg[1])
        # Expected:
        expected = {qreg[0], qreg[1], qreg[2], qreg[3]}

        self.assertEqual(result, expected)

    def test_causal_cone_more_barriers(self):
        """Test causal cone for circuit with barriers. This example shows
        why barriers may need to be examined multiple times."""

        # q0_0: ──■────────░────────────────────────
        #       ┌─┴─┐      ░
        # q0_1: ┤ X ├──────░───■────────────────────
        #       ├───┤      ░ ┌─┴─┐┌───┐┌───┐┌───┐
        # q0_2: ┤ H ├──────░─┤ X ├┤ H ├┤ H ├┤ H ├─X─
        #       ├───┤┌───┐ ░ └───┘└───┘└───┘└───┘ │
        # q0_3: ┤ H ├┤ X ├─░──────────────────────X─
        #       ├───┤└─┬─┘ ░
        # q0_4: ┤ X ├──■───░────────────────────────
        #       └─┬─┘      ░
        # q0_5: ──■────────░────────────────────────

        qreg = QuantumRegister(6)
        qc = QuantumCircuit(qreg)
        qc.cx(0, 1)
        qc.h(2)
        qc.cx(5, 4)
        qc.h(3)
        qc.cx(4, 3)
        qc.barrier()
        qc.cx(1, 2)

        qc.h(2)
        qc.h(2)
        qc.h(2)
        qc.swap(2, 3)

        dag = circuit_to_dag(qc)

        result = dag.quantum_causal_cone(qreg[2])
        expected = {qreg[0], qreg[1], qreg[2], qreg[3], qreg[4], qreg[5]}

        self.assertEqual(result, expected)

    def test_causal_cone_measure(self):
        """Test causal cone with measures."""

        #         ┌───┐ ░ ┌─┐
        #    q_0: ┤ H ├─░─┤M├────────────
        #         ├───┤ ░ └╥┘┌─┐
        #    q_1: ┤ H ├─░──╫─┤M├─────────
        #         ├───┤ ░  ║ └╥┘┌─┐
        #    q_2: ┤ H ├─░──╫──╫─┤M├──────
        #         ├───┤ ░  ║  ║ └╥┘┌─┐
        #    q_3: ┤ H ├─░──╫──╫──╫─┤M├───
        #         ├───┤ ░  ║  ║  ║ └╥┘┌─┐
        #    q_4: ┤ H ├─░──╫──╫──╫──╫─┤M├
        #         └───┘ ░  ║  ║  ║  ║ └╥┘
        #    c: 5/═════════╬══╬══╬══╬══╬═
        #                  ║  ║  ║  ║  ║
        # meas: 5/═════════╩══╩══╩══╩══╩═
        #                  0  1  2  3  4

        qreg = QuantumRegister(5)
        creg = ClassicalRegister(5)
        circuit = QuantumCircuit(qreg, creg)
        for i in range(5):
            circuit.h(i)
        circuit.measure_all()

        dag = circuit_to_dag(circuit)

        result = dag.quantum_causal_cone(dag.qubits[1])
        expected = {qreg[1]}
        self.assertEqual(result, expected)

    def test_reconvergent_paths(self):
        """Test circuit with reconvergent paths."""

        # q0_0: ──■─────────■─────────■─────────■─────────■─────────■───────
        #       ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     ┌─┴─┐
        # q0_1: ┤ X ├──■──┤ X ├──■──┤ X ├──■──┤ X ├──■──┤ X ├──■──┤ X ├──■──
        #       └───┘┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
        # q0_2: ──■──┤ X ├──■──┤ X ├──■──┤ X ├──■──┤ X ├──■──┤ X ├──■──┤ X ├
        #       ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐└───┘
        # q0_3: ┤ X ├─────┤ X ├─────┤ X ├─────┤ X ├─────┤ X ├─────┤ X ├─────
        #       └───┘     └───┘     └───┘     └───┘     └───┘     └───┘
        # q0_4: ────────────────────────────────────────────────────────────

        qreg = QuantumRegister(5)
        circuit = QuantumCircuit(qreg)

        for _ in range(6):
            circuit.cx(0, 1)
            circuit.cx(2, 3)
            circuit.cx(1, 2)

        dag = circuit_to_dag(circuit)

        result = dag.quantum_causal_cone(dag.qubits[1])
        expected = {qreg[0], qreg[1], qreg[2], qreg[3]}
        self.assertEqual(result, expected)


class TestDAGMakePhysical(DAGTest):
    """Tests of `make_physical`."""

    def test_basic(self):
        """Test basic usage of `make_physical`."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Bool())

        def from_qubit_circuit(qc: QuantumCircuit):
            qc.add_register(ClassicalRegister(5, "c"))
            qc.add_input(a)
            qc.add_var(b, expr.lift(False))
            qc.h(0)
            qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])
            return circuit_to_dag(qc, copy_operations=False)

        base = from_qubit_circuit(QuantumCircuit(QuantumRegister(5, "virtuals")))
        base.make_physical()
        self.assertEqual(base, from_qubit_circuit(QuantumCircuit(QuantumRegister(5, "q"))))

        expanded = from_qubit_circuit(QuantumCircuit([Qubit() for _ in range(3)]))
        expanded.make_physical(10)
        self.assertEqual(expanded, from_qubit_circuit(QuantumCircuit(QuantumRegister(10, "q"))))

    def test_after_substitution(self):
        """Make sure that having a nonsequential node ordering in the graph doesn't matter."""
        base = DAGCircuit()
        base.add_qreg(QuantumRegister(1, "virtuals"))
        base.apply_operation_back(XGate(), [base.qubits[0]], [])
        mid = base.apply_operation_back(HGate(), [base.qubits[0]], [])
        base.apply_operation_back(XGate(), [base.qubits[0]], [])

        replacement = QuantumCircuit(1, global_phase=-math.pi / 4)
        replacement.s(0)
        replacement.sx(0)
        replacement.s(0)

        base.substitute_node_with_dag(mid, circuit_to_dag(replacement))
        base.make_physical(5)

        expected = QuantumCircuit(QuantumRegister(5, "q"), global_phase=-math.pi / 4)
        expected.x(0)
        expected.s(0)
        expected.sx(0)
        expected.s(0)
        expected.x(0)

        self.raise_if_dagcircuit_invalid(base)
        self.assertEqual(base, circuit_to_dag(expected))

    def test_empty(self):
        """An empty DAGCircuit."""
        empty_0q = DAGCircuit()
        empty_0q.make_physical()
        expected = DAGCircuit()
        expected.add_qreg(QuantumRegister(0, "q"))
        self.assertEqual(empty_0q, expected)

        empty_8q = DAGCircuit()
        empty_8q.make_physical(8)
        expanded = DAGCircuit()
        expanded.add_qreg(QuantumRegister(8, "q"))
        self.assertEqual(empty_8q, expanded)

    def test_0q_with_classical(self):
        """Classical data should still work fine even if there are no qubits."""
        cr = ClassicalRegister(3, "cr")
        qc = QuantumCircuit(cr)
        a = qc.add_input("a", types.Bool())
        b = qc.add_var("b", expr.lift(False))
        qc.store(b, a)

        def expected(num_qubits):
            dag = DAGCircuit()
            dag.add_qreg(QuantumRegister(num_qubits, "q"))
            dag.add_creg(cr)
            dag.add_input_var(a)
            dag.add_declared_var(b)
            dag.apply_operation_back(Store(b, expr.lift(False)), [], [])
            dag.apply_operation_back(Store(b, a), [], [])
            return dag

        dag = circuit_to_dag(qc)
        dag.make_physical()
        self.assertEqual(dag, expected(0))

        dag = circuit_to_dag(qc)
        dag.make_physical(156)
        self.assertEqual(dag, expected(156))


class TestStructurallyEqual(DAGTest):
    """Test structural-equality checks."""

    def test_deterministic_circuit(self):
        """Two DAGs created from the same circuit should be structurally equal."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.z(0)
        qc.z(1)
        qc.measure([0, 1], [0, 1])
        left = circuit_to_dag(qc)
        right = circuit_to_dag(qc)
        # If this fails, it might be a problem with `circuit_to_dag`.
        self.assertTrue(left.structurally_equal(right))

    def test_name_does_not_matter(self):
        left = circuit_to_dag(QuantumCircuit(name="left"))
        right = circuit_to_dag(QuantumCircuit(name="right"))
        self.assertTrue(left.structurally_equal(right))

    def test_mismatch_global_phase(self):
        # Explicit floats and bound parameter expressions should also be considered different,
        # because they mean we must have done different things.
        a = Parameter("a")
        left = circuit_to_dag(QuantumCircuit(global_phase=2.0))
        right = circuit_to_dag(QuantumCircuit(global_phase=a.bind({a: 2.0})))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_qubits(self):
        left = circuit_to_dag(QuantumCircuit([Qubit() for _ in range(4)]))
        right = circuit_to_dag(QuantumCircuit(left.qubits[::-1]))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_qregs(self):
        left = circuit_to_dag(QuantumCircuit(QuantumRegister(4, "q")))
        # ... but no register.
        right = circuit_to_dag(QuantumCircuit(left.qubits))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_clbits(self):
        left = circuit_to_dag(QuantumCircuit([Clbit() for _ in range(4)]))
        right = circuit_to_dag(QuantumCircuit(left.clbits[::-1]))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_cregs(self):
        left = circuit_to_dag(QuantumCircuit(ClassicalRegister(4, "c")))
        # ... but no register.
        right = circuit_to_dag(QuantumCircuit(left.clbits))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_vars_types(self):
        left = circuit_to_dag(QuantumCircuit(inputs=[expr.Var.new("a", types.Bool())]))
        right = circuit_to_dag(QuantumCircuit(inputs=[expr.Var.new("a", types.Uint(1))]))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_vars_names(self):
        left = circuit_to_dag(QuantumCircuit(inputs=[expr.Var.new("a", types.Bool())]))
        right = circuit_to_dag(QuantumCircuit(inputs=[expr.Var.new("b", types.Bool())]))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_vars_storage(self):
        a = expr.Var.new("a", types.Bool())
        left = circuit_to_dag(QuantumCircuit(inputs=[a]))
        right = circuit_to_dag(QuantumCircuit(captures=[a]))
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_vars_uuid(self):
        # The two calls to `Var.new` produce different UUIDs.
        left = circuit_to_dag(QuantumCircuit(inputs=[expr.Var.new("a", types.Bool())]))
        right = circuit_to_dag(QuantumCircuit(inputs=[expr.Var.new("a", types.Bool())]))
        self.assertFalse(left.structurally_equal(right))

    def test_matching_stretches(self):
        left = QuantumCircuit()
        a = left.add_stretch("a")
        left = circuit_to_dag(left)
        right = QuantumCircuit()
        right.add_stretch(a)
        right = circuit_to_dag(right)
        self.assertTrue(left.structurally_equal(right))

    def test_mismatch_stretch_names(self):
        left = QuantumCircuit()
        left.add_stretch("a")
        left = circuit_to_dag(left)
        right = QuantumCircuit()
        right.add_stretch("b")
        right = circuit_to_dag(right)
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_stretch_uuids(self):
        left = QuantumCircuit()
        left.add_stretch("a")
        left = circuit_to_dag(left)
        right = QuantumCircuit()
        # A new UUID.
        right.add_stretch("a")
        right = circuit_to_dag(right)
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_stretch_orders(self):
        left = QuantumCircuit()
        a = left.add_stretch("a")
        b = left.add_stretch("b")
        left = circuit_to_dag(left)
        right = QuantumCircuit()
        # Wrong order of stretches; this matters.
        right.add_stretch(b)
        right.add_stretch(a)
        right = circuit_to_dag(right)
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_node_order(self):
        left = circuit_to_dag(QuantumCircuit(2))
        right = circuit_to_dag(QuantumCircuit(2))
        self.assertTrue(left.structurally_equal(right))

        left.apply_operation_back(XGate(), [left.qubits[0]], [])
        left.apply_operation_back(XGate(), [left.qubits[1]], [])

        # Exact same gates, but the modification order is different.
        right.apply_operation_back(XGate(), [right.qubits[1]], [])
        right.apply_operation_back(XGate(), [right.qubits[0]], [])

        # Semantic equality should be fine.
        self.assertEqual(left, right)
        # ... but not structural.
        self.assertFalse(left.structurally_equal(right))

    def test_mismatch_with_node_removal(self):
        left = circuit_to_dag(QuantumCircuit(2))
        right = circuit_to_dag(QuantumCircuit(2))
        self.assertTrue(left.structurally_equal(right))

        left.apply_operation_back(XGate(), [left.qubits[0]], [])
        left.apply_operation_back(XGate(), [left.qubits[1]], [])

        right.apply_operation_back(XGate(), [right.qubits[0]], [])
        to_remove = right.apply_operation_back(XGate(), [right.qubits[0]], [])
        right.apply_operation_back(XGate(), [right.qubits[1]], [])
        right.remove_op_node(to_remove)

        # Semantic equality should be fine.
        self.assertEqual(left, right)
        # ... but not structural.  If the node we removed was the final node, it _might_ be ok, but
        # that'd still be an implementation detail.
        self.assertFalse(left.structurally_equal(right))


if __name__ == "__main__":
    unittest.main()
