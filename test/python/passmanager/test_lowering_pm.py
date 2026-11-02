# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the lowering pass manager exposed from Rust."""

from test import QiskitTestCase

from collections.abc import Iterable
from typing import Any

from qiskit import capi
from qiskit.circuit import QuantumCircuit, CircuitData
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import (
    IR,
    LoweringPassManager,
    LoweringPassManagerError,
    PassContextHandle,
    Pass,
    PassManagerState,
    PropertySet,
    WorkflowStatus,
    Task,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.transpiler import generate_preset_pass_manager, CouplingMap, TranspileLayout, Target
from qiskit.transpiler.passes import RemoveIdentityEquivalent


class CountsIR(IR):
    """A count-based IR."""

    _qiskit_ir_name_ = "CountsIR"

    def __init__(self, data: dict[str, int]) -> None:
        super().__init__()
        self.data = data

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CountsIR) and self.data == other.data


class AlsoCountsIR(CountsIR):
    """A subclassed custom IR."""


class CliffordCountsIR(CountsIR):
    """An IR deriving from CountsIR."""

    _qiskit_ir_name_ = "CliffordCountsIR"
    _qiskit_ir_base_ = None  # we want this to be a **new** IR!

    def __init__(self, data: dict[str, int]) -> None:
        clifford_names = set(get_clifford_gate_names())
        if any(key not in clifford_names for key in data):
            raise ValueError("Non-Clifford detected!")

        super().__init__(data)


class NonIRBase:
    """A base-class that does not implement `IR`."""


class IRBase(NonIRBase, IR):
    """Inherit from a non-IR base class, *and* IR."""


class IRBasePass(Pass[IRBase]):
    """A pass on `IRBase`."""

    _qiskit_pass_ir_in_ = IRBase
    _qiskit_pass_ir_out_ = IRBase

    def _qiskit_pass_run_(self, ir, context):
        return ir


class CliffordCircuit(QuantumCircuit):
    """An IR deriving the builtin circuit IR.

    This is not actually doing any Clifford checks and is just for testing purposes.
    """


class CircuitToDag(Pass[QuantumCircuit, DAGCircuit]):
    """A lowering pass from quantum circuit to DAG circuit.

    This does *not* copy the operations.
    """

    _qiskit_pass_ir_in_ = QuantumCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def _qiskit_pass_run_(self, ir, context):
        context.ir_modified = True
        return ir.to_dag(copy_operations=False)


class DagToCircuit(Pass[DAGCircuit, QuantumCircuit]):
    """A lowering pass from quantum circuit to DAG circuit.

    This does *not* copy the operations.
    """

    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = QuantumCircuit

    def __init__(self, with_layout: bool = False):
        super().__init__()
        self.with_layout = with_layout

    def _qiskit_pass_run_(self, ir, context):
        context.ir_modified = True
        circuit = ir.to_circuit(copy_operations=False)

        if self.with_layout:
            property_set = LegacyDagPass._state_from_context(context).property_set
            circuit._layout = TranspileLayout.from_property_set(ir, property_set)
        return circuit


class CircuitToCircuitData(Pass[QuantumCircuit, CircuitData]):
    """Convert a `QuantumCircuit` to the inner `CircuitData`."""

    _qiskit_pass_ir_in_ = QuantumCircuit
    _qiskit_pass_ir_out_ = CircuitData

    def _qiskit_pass_run_(self, ir, context):
        return ir._data


class CircuitDataToDag(Pass[CircuitData, DAGCircuit]):
    """Convert a `CircuitData` to a `DAGCircuit`."""

    _qiskit_pass_ir_in_ = CircuitData
    _qiskit_pass_ir_out_ = DAGCircuit

    def _qiskit_pass_run_(self, ir, context):
        return QuantumCircuit.from_circuit_data(ir).to_dag(copy_operations=False)


class EstimateFidelity(Pass[CircuitData]):
    """Run a fidelity estimation given a target."""

    _qiskit_pass_ir_in_ = CircuitData
    _qiskit_pass_ir_out_ = CircuitData

    def __init__(self, target):
        super().__init__()
        self.target = target

    def _qiskit_pass_run_(self, ir, context):
        c_target = capi.qk_target_borrow_from_python(self.target)
        c_circuit = capi.qk_circuit_borrow_from_python(ir)

        context["fidelity"] = capi.qk_circuit_estimate_fidelity(c_circuit, c_target)
        context.ir_modified = False
        return ir


class RemoveIdentities(Pass[DAGCircuit]):
    """Remove close-to-identity gates in a DAGCircuit."""

    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def __init__(self):
        super().__init__()
        self._pass = RemoveIdentityEquivalent()

    def _qiskit_pass_run_(self, ir, context):
        return self._pass.run(ir)


class CountGates(Pass[DAGCircuit]):
    """Store the gate count in the pass context."""

    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def _qiskit_pass_run_(self, ir, context):
        context.ir_modified = False
        context["counts"] = ir.count_ops()
        return ir


class LyingBuiltinOutput(Pass[QuantumCircuit]):
    _qiskit_pass_ir_in_ = QuantumCircuit
    _qiskit_pass_ir_out_ = DAGCircuit  # define DAGCircuit as return, but we return a QC

    def _qiskit_pass_run_(self, ir, context):
        return ir


class VerifyContext(Pass[DAGCircuit]):
    """A pass raising an error if a defined key is not in the pass context."""

    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def __init__(
        self, expect: dict[str, Any] = {}, present: Iterable[str] = (), absent: Iterable[str] = ()
    ) -> None:
        """
        Args:
            expect: A dict of {key, value} elements to expect in the context.
            present: A set of keys that should be in the context.
            absent: A set of keys that should not be in the context.
        """
        super().__init__()
        self.expect = expect
        self.present = present
        self.absent = absent

    def _qiskit_pass_run_(self, ir, context):
        for key, expected_value in self.expect.items():
            if (value := context.get(key)) is None:
                raise ValueError(f"{key} not found!")

            if value != expected_value:
                raise ValueError("value did not match expectation")

        for key in self.present:
            if context.get(key, "___empty") == "___empty":
                raise ValueError(f"{key} was empty")

        for key in self.absent:
            if context.get(key, "___empty") != "___empty":
                raise ValueError(f"{key} was not empty")

        context.ir_modified = False
        return ir


class DeleteKeys(Pass[DAGCircuit]):
    """A pass deleting a key in the context."""

    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = keys

    def _qiskit_pass_run_(self, ir, context):
        for key in self.keys:
            del context[key]

        return ir


class StealContext(Pass[DAGCircuit]):
    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def __init__(self):
        self.handle = None

    def _qiskit_pass_run_(self, ir, context):
        self.handle = context
        context.ir_modified = False
        return ir


class IncompletePass(Pass):
    """An incomplete pass that does not specify the IRs it is acting on."""

    def _qiskit_pass_run_(self, ir, context):
        return ir


class InvalidPass(Pass[int, int]):
    """A pass that specifies IRs that are not `IR`s."""

    _qiskit_pass_ir_in_ = int
    _qiskit_pass_ir_out_ = int

    def _qiskit_pass_run_(self, ir, context):
        return ir


class NamelessIR(IR):
    """A nameless IR."""


class AnotherNamelessIR(IR):
    """Another nameless IR, which should not be the same as `NamelessIR`."""


class NamelessPass(Pass[NamelessIR]):
    """A pass on a nameless (invalid) IR."""

    _qiskit_pass_ir_in_ = NamelessIR
    _qiskit_pass_ir_out_ = NamelessIR

    def _qiskit_pass_run_(self, ir, context):
        return ir


class DagToCounts(Pass[DAGCircuit, CountsIR]):
    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = CountsIR

    def _qiskit_pass_run_(self, ir, context):
        context.ir_modified = True
        return CountsIR(ir.count_ops())


class PopCounts(Pass[CountsIR]):
    """Remove gate counts from the `CountsIR`."""

    _qiskit_pass_ir_in_ = CountsIR
    _qiskit_pass_ir_out_ = CountsIR

    def __init__(self, to_pop: Iterable[str]) -> None:
        super().__init__()
        self.to_pop = to_pop

    def _qiskit_pass_run_(self, ir, context):
        for key in self.to_pop:
            ir.data.pop(key)
            context.ir_modified = True

        return ir


class LegacyDagPass(Pass[DAGCircuit]):
    """A wrapper for a legacy Python pass."""

    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    # the key in the pass context under which we store the written keys in the context
    # -- this is used to convert from/to property sets, since the PassContextHandle does
    # not have a way to iterate over the (Python) keys
    _legacy_keys = "___legacy_keys"

    def __init__(self, legacy_task: Task[DAGCircuit, DAGCircuit]):
        super().__init__()
        self._task = legacy_task

    def _qiskit_pass_run_(self, ir, context):
        state = self._state_from_context(context)
        out, out_state = self._task.execute(ir, state, None)
        self._update_context(context, state, out_state)

        return out

    @staticmethod
    def _state_from_context(context: PassContextHandle) -> PassManagerState:
        dummy_status = WorkflowStatus()
        property_set = PropertySet(
            {key: context[key] for key in context.get(LegacyDagPass._legacy_keys, ())}
        )
        return PassManagerState(dummy_status, property_set)

    @staticmethod
    def _update_context(
        context: PassContextHandle, old_state: PassManagerState, new_state: PassManagerState
    ):
        context[LegacyDagPass._legacy_keys] = []
        for key, value in new_state.property_set.items():
            context[LegacyDagPass._legacy_keys].append(key)
            context[key] = value

        for deleted_key in set(old_state.property_set.keys()).difference(
            new_state.property_set.keys()
        ):
            del context[deleted_key]


class TestLoweringPassManager(QiskitTestCase):
    """Tests for the lowering pass manager."""

    def test_reuse(self):
        """Test re-using the same pass manager."""
        pm = LoweringPassManager([CircuitToDag(), RemoveIdentities(), DagToCircuit()])

        circuit1 = QuantumCircuit(2)
        circuit1.cx(0, 1)
        circuit1.rz(1e-10, 0)
        circuit1.rx(1, 1)

        expected1 = QuantumCircuit(2)
        expected1.cx(0, 1)
        expected1.rx(1, 1)

        circuit2 = QuantumCircuit(3)
        circuit2.rxx(1e-9, 1, 0)
        circuit2.ryy(-0.2, 0, 1)

        expected2 = QuantumCircuit(3)
        expected2.ryy(-0.2, 0, 1)

        for circuit, expected in [(circuit1, expected1), (circuit2, expected2)]:
            with self.subTest(circuit=circuit):
                self.assertEqual(expected, pm.run(circuit))

    def test_empty_pm(self):
        """Test that an empty pass manager works and acts trivially."""
        pm = LoweringPassManager()

        circuit = QuantumCircuit(11)
        circuit.mcx(list(range(10)), 10)
        with self.subTest(ir=circuit):
            self.assertEqual(pm.run(circuit), circuit)

        dag = circuit.to_dag()
        with self.subTest(ir=dag):
            self.assertEqual(pm.run(dag), dag)

        counts = CountsIR({"x": 42_000})
        with self.subTest(ir=counts):
            self.assertEqual(pm.run(counts), counts)

        # even an empty PM does require `IR` input types
        with self.subTest(ir=1):
            with self.assertRaisesRegex(TypeError, "does not implement `IR`"):
                _ = pm.run(1)

    def test_input_ir_mismatch(self):
        """Test calling the pass manager on a different IR than it is defined on raises."""
        pm = LoweringPassManager([RemoveIdentities()])
        with self.assertRaisesRegex(
            TypeError, "incoming IR of type .*QuantumCircuit.* does not match .*"
        ):
            _ = pm.run(QuantumCircuit(1))

    def test_invalid_pipeline(self):
        """Test constructing a type-mismatched pipeline raises."""
        expected = "cannot append the task: .*outputs DAGCircuit.*takes QuantumCircuit"
        with self.assertRaisesRegex(TypeError, expected):
            _ = LoweringPassManager([CircuitToDag(), CircuitToDag()])

        pm = LoweringPassManager([CircuitToDag()])
        pass_ = CircuitToDag()
        with self.assertRaisesRegex(TypeError, expected):
            pm.append(pass_)

        # check the pass_ we failed to put is still the same object
        self.assertIsInstance(pass_, CircuitToDag)

        # the PM should still be runnable with the previously valid path
        circuit = QuantumCircuit(3)
        circuit.ccz(0, 2, 1)
        self.assertEqual(circuit.to_dag(), pm.run(circuit))

    def test_pass_context_access(self):
        """Test writing and accessing the pass context."""

        circuit = QuantumCircuit(3)
        circuit.swap(0, 1)
        for _ in range(8):
            circuit.sx(0)

        with self.subTest(msg="verify setting"):
            pm = LoweringPassManager(
                [CircuitToDag(), CountGates(), VerifyContext({"counts": circuit.count_ops()})]
            )
            dag = pm.run(circuit)
            self.assertIsInstance(dag, DAGCircuit)

        with self.subTest(msg="verify removal"):
            pm.append(DeleteKeys(["counts"]))
            pm.append(VerifyContext({}, absent=["counts"]))
            dag = pm.run(circuit, copy=False)  # we can consume the IR here
            self.assertIsInstance(dag, DAGCircuit)

    def test_pass_context_is_invalidated(self):
        thief = StealContext()
        pm = LoweringPassManager([CircuitToDag(), CountGates(), thief])
        _ = pm.run(QuantumCircuit(1))

        # the handle should exist but invalid to read from
        self.assertIsInstance(thief.handle, PassContextHandle)
        with self.assertRaisesRegex(RuntimeError, "attempted to use handle after pass returned"):
            _ = thief.handle.ir_modified
        with self.assertRaisesRegex(RuntimeError, "attempted to use handle after pass returned"):
            _ = thief.handle.get("counts")
        with self.assertRaisesRegex(RuntimeError, "attempted to use handle after pass returned"):
            _ = thief.handle["counts"]
        with self.assertRaisesRegex(RuntimeError, "attempted to use handle after pass returned"):
            thief.handle["new"] = "thing"
        with self.assertRaisesRegex(RuntimeError, "attempted to use handle after pass returned"):
            del thief.handle["counts"]

    def test_invalid_delete(self):
        """Test trying to delete a key twice (or an inexisting key)."""
        circuit = QuantumCircuit(2)
        circuit.t([0, 1])

        for missing_key, pm in [
            (
                "counts",
                LoweringPassManager([CountGates(), DeleteKeys(["counts"]), DeleteKeys(["counts"])]),
            ),
            ("idonotexist", LoweringPassManager([CountGates(), DeleteKeys(["idonotexist"])])),
        ]:
            with self.assertRaises(LoweringPassManagerError) as ctx:
                _ = pm.run(circuit.to_dag())

            cause = ctx.exception.__cause__
            self.assertIsInstance(cause, KeyError)
            self.assertRegex(str(cause), missing_key)

    def test_custom_ir(self):
        pm = LoweringPassManager([CircuitToDag(), DagToCounts(), PopCounts(["h", "cx"])])

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.t(1)

        out = pm.run(circuit)
        self.assertEqual(CountsIR({"t": 1}), out)

    def test_name_collision(self):
        pm = LoweringPassManager([NamelessPass()])
        self.assertIsInstance(pm.run(NamelessIR()), NamelessIR)

        with self.assertRaisesRegex(
            TypeError,
            "incoming IR of type.*AnotherNamelessIR.* does not match expected .*NamelessIR.*",
        ):
            _ = pm.run(AnotherNamelessIR())

    def test_invalid_pass(self):
        """Test adding a lowering pass that didn't specify the IRs."""
        with self.assertRaisesRegex(TypeError, "does not implement `IR`"):
            _ = LoweringPassManager([InvalidPass()])

        pm = LoweringPassManager()
        with self.assertRaisesRegex(TypeError, "does not implement `IR`"):
            pm.append(InvalidPass())

    def test_incomplete_pass(self):
        """Test adding a lowering pass that didn't specify the IRs."""
        with self.assertRaises(AttributeError):
            _ = LoweringPassManager([IncompletePass()])

        pm = LoweringPassManager()
        with self.assertRaises(AttributeError):
            pm.append(IncompletePass())

    def test_mismatched_builtin_output(self):
        """Test a pipeline where a pass returns another IR than it specified.

        Note that we're not currently actually guaranteeing this check to happen is passes
        lie about their input or output types. This test is sanity-checking the current state
        and might be changed in the future. This is not an API-documented behavior.
        """
        for pm in [
            LoweringPassManager([LyingBuiltinOutput(), DagToCircuit()]),
            LoweringPassManager([LyingBuiltinOutput()]),
        ]:
            circuit = QuantumCircuit(2)
            circuit.cry(0.123, 0, 1)

            with self.assertRaises(LoweringPassManagerError) as ctx:
                _ = pm.run(circuit)

            cause = ctx.exception.__cause__
            self.assertIsInstance(cause, TypeError)
            self.assertRegex(str(cause), ".* is not an instance of .*")

    def test_pass_error(self):
        """Test the error a pass is raising is propagated."""
        pm = LoweringPassManager([VerifyContext({"vacation?": True})])
        with self.assertRaises(LoweringPassManagerError) as ctx:
            _ = pm.run(DAGCircuit())

        cause = ctx.exception.__cause__
        self.assertIsInstance(cause, ValueError)
        self.assertRegex(str(cause), "vacation\\? not found!")

    def test_legacy_pass(self):
        """Test a pass that runs by wrapping a legacy pass manager pass functions."""
        pm = LoweringPassManager(
            [CircuitToDag(), LegacyDagPass(RemoveIdentityEquivalent()), DagToCircuit()]
        )

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(1e-10, 0)
        circuit.rx(1, 1)

        out = pm.run(circuit)
        self.assertEqual(out.count_ops(), {"cx": 1, "rx": 1})

    def test_legacy_pipeline(self):
        """Test running the legacy pipeline in the new system."""
        backend = GenericBackendV2(25, coupling_map=CouplingMap.from_grid(5, 5))
        legacy_pm = generate_preset_pass_manager(backend=backend, seed_transpiler=23)
        pm = LoweringPassManager(
            [CircuitToDag()]
            + [LegacyDagPass(task) for task in legacy_pm.to_flow_controller().tasks]
            + [DagToCircuit(with_layout=True)]
        )

        circuit = QuantumCircuit(3)
        circuit.cx(0, 2)
        circuit.rz(1e-10, 0)
        circuit.rx(1, 1)

        expect = legacy_pm.run(circuit)
        out = pm.run(circuit)

        self.assertEqual(expect, out)
        self.assertIsInstance(out.layout, TranspileLayout)

    def test_circuit_data_ir(self):
        """Test running a pass on the Rust-native `CircuitData` IR."""
        target = Target.from_configuration(basis_gates=["x", "sx", "cx"])
        pm = LoweringPassManager(
            [
                CircuitToCircuitData(),
                EstimateFidelity(target),
                CircuitDataToDag(),
                VerifyContext(present=["fidelity"]),
            ]
        )

        circuit = QuantumCircuit(2)
        circuit.sx(0)
        circuit.x(0)
        circuit.cx(0, 1)

        out = pm.run(circuit, copy=False)
        self.assertIsInstance(out, DAGCircuit)

    def test_base_ir(self):
        """Test running a pass on an IR defined on another IR base."""

        pm = LoweringPassManager([PopCounts(["h"])])
        ir = CliffordCountsIR({"h": 2, "x": 12})

        with self.assertRaisesRegex(
            TypeError, "incoming IR of type .*CliffordCountsIR.* does not match .*CountsIR.*"
        ):
            _ = pm.run(ir)

    def test_subclassed_custom_ir(self):
        """Test running a pass on the parent class on the subclass."""

        pm = LoweringPassManager([PopCounts(["h"])])
        ir = AlsoCountsIR({"h": 2, "t": 12})
        out = pm.run(ir)
        self.assertEqual(out.data, {"t": 12})

    def test_subclassed_builtin_ir(self):
        """Test running a pass on the parent class on the subclass."""

        pm = LoweringPassManager([CircuitToDag()])
        ir = CliffordCircuit(10)
        ir.h(range(9))
        ir.cx(range(9), 9)

        out = pm.run(ir)
        self.assertIsInstance(out, DAGCircuit)

    def test_subclassed_on_non_ir(self):
        """Test a subclass that has a non-IR as parent."""
        pm = LoweringPassManager([IRBasePass()])
        self.assertIsInstance(pm.run(IRBase()), IRBase)
