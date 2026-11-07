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

from collections.abc import Iterable, MutableMapping, MutableSequence
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
    LoweringTask,
    TaskSequence,
    PropertySet,
    TaskStages,
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


class AlsoCountsPass(Pass[AlsoCountsIR]):
    """A pass on a subclass of `CountsIR` that is not a new IR."""

    _qiskit_pass_ir_in_ = AlsoCountsIR
    _qiskit_pass_ir_out_ = AlsoCountsIR

    def _qiskit_pass_run_(self, ir, context):
        return ir


class CliffordCountsPass(Pass[CliffordCountsIR]):
    """A pass on a subclass of `CountsIR` that declared itself a new IR."""

    _qiskit_pass_ir_in_ = CliffordCountsIR
    _qiskit_pass_ir_out_ = CliffordCountsIR

    def _qiskit_pass_run_(self, ir, context):
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

    def test_append_accepts_subclassed_ir(self):
        """Test that a pass on a subclass of an IR can follow a pass outputting the base."""
        pm = LoweringPassManager([CircuitToDag(), DagToCounts(), AlsoCountsPass()])
        self.assertIsInstance(pm.run(QuantumCircuit(1)), CountsIR)

    def test_append_rejects_declared_new_ir(self):
        """Test that a subclass declaring itself a new IR does not follow the base."""
        with self.assertRaisesRegex(TypeError, "CountsIR.*CliffordCountsIR"):
            _ = LoweringPassManager([CircuitToDag(), DagToCounts(), CliffordCountsPass()])


class TestTasks(QiskitTestCase):
    """Tests for composite tasks."""

    def test_empty_task_is_accepted(self):
        """Test that a task with no typed subtask imposes no type constraint."""
        for task in [TaskSequence(), TaskStages(), TaskSequence([TaskStages()])]:
            with self.subTest(task=task):
                pm = LoweringPassManager([task])
                circuit = QuantumCircuit(1)
                self.assertEqual(circuit, pm.run(circuit))

    def test_empty_children_do_not_break_type_chain(self):
        """Test that empty children neither interrupt nor mask the type chain."""
        pm = LoweringPassManager([TaskSequence(), CircuitToDag(), TaskStages()])
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        self.assertEqual(circuit.to_dag(), pm.run(circuit))

        # the empty trailing child does not mask the preceding pass's output type
        with self.assertRaises(TypeError):
            pm.append(CircuitToDag())

    def test_nested_task_runs(self):
        """Test running a pass manager built from nested tasks."""
        stages = TaskStages(
            [
                ("init", TaskSequence([RemoveIdentities()])),
                ("placeholder", TaskSequence()),
                ("count", TaskSequence([CountGates()])),
            ]
        )
        pm = LoweringPassManager([CircuitToDag(), stages, DagToCircuit()])

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(1e-10, 0)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        self.assertEqual(expected, pm.run(circuit))

    def test_ill_typed_task_is_constructible(self):
        """Test that a task may be built ill-typed, and is only rejected when it is appended."""
        task = TaskSequence([CircuitToDag(), CircuitToDag()])
        self.assertEqual(len(task), 2)
        with self.assertRaisesRegex(TypeError, "ill-typed"):
            _ = LoweringPassManager([task])

    def test_append_rejects_ill_typed_task(self):
        """Test that appending an internally ill-typed task raises and changes nothing."""
        pm = LoweringPassManager([CircuitToDag()])
        task = TaskSequence([RemoveIdentities(), CircuitToDag()])
        with self.assertRaisesRegex(TypeError, "ill-typed"):
            pm.append(task)

        # the task is handed back untouched, and the pass manager still runs
        self.assertEqual(len(task), 2)
        circuit = QuantumCircuit(3)
        circuit.ccz(0, 2, 1)
        self.assertEqual(circuit.to_dag(), pm.run(circuit))

    def test_append_rejects_boundary_mismatch(self):
        """Test that appending a task whose input does not follow on raises."""
        pm = LoweringPassManager([CircuitToDag()])
        with self.assertRaisesRegex(TypeError, "DAGCircuit.*QuantumCircuit"):
            pm.append(TaskSequence([CircuitToDag()]))

    def test_append_reports_nested_path(self):
        """Test that the error locates a mismatch nested inside a tree."""
        inner = TaskSequence([RemoveIdentities(), DagToCircuit(), RemoveIdentities()])
        task = TaskSequence([TaskStages([("layout", inner)])])
        with self.assertRaisesRegex(TypeError, r"\[0\]\['layout'\]\[2\]"):
            _ = LoweringPassManager([CircuitToDag(), task])

    def test_append_rejects_non_task(self):
        """Test that appending something that is not a task raises."""
        with self.assertRaisesRegex(TypeError, "expected a Pass, TaskSequence or TaskStages"):
            LoweringPassManager().append("nope")
        with self.assertRaisesRegex(TypeError, "expected a Pass, TaskSequence or TaskStages"):
            _ = LoweringPassManager(["nope"])

    def test_append_rejects_foreign_task_subclass(self):
        """Test that a subclass of the task interface is not executable."""

        class Custom(LoweringTask):
            """A task that is not one of the three native variants."""

        with self.assertRaisesRegex(TypeError, "expected a Pass, TaskSequence or TaskStages"):
            LoweringPassManager().append(Custom())

    def test_append_rejects_nested_non_task(self):
        """Test that a non-task nested in a container raises when the container is appended."""
        task_sequence = TaskSequence([CircuitToDag()])
        task_sequence.append("nope")
        with self.assertRaisesRegex(TypeError, "expected a Pass, TaskSequence or TaskStages"):
            _ = LoweringPassManager([task_sequence])

    def test_append_rejects_cyclic_task(self):
        """Test that a container holding itself raises rather than overflowing the stack."""
        task_sequence = TaskSequence()
        task_sequence.append(task_sequence)
        with self.assertRaises(ValueError):
            _ = LoweringPassManager([task_sequence])

        stages = TaskStages()
        stages["me"] = stages
        with self.assertRaises(ValueError):
            _ = LoweringPassManager([stages])

    def test_init_rejects_task_stages(self):
        """Test that passing task stages directly raises, since they iterate names."""
        stages = TaskStages([("lower", CircuitToDag())])
        with self.assertRaisesRegex(TypeError, "expected a Pass, TaskSequence or TaskStages"):
            _ = LoweringPassManager(stages)

        # nesting it as a single task is what was meant
        pm = LoweringPassManager([stages])
        self.assertEqual(QuantumCircuit(1).to_dag(), pm.run(QuantumCircuit(1)))

    def test_task_is_reusable(self):
        """Test that the same task may be given to more than one pass manager."""
        task = TaskSequence([CircuitToDag(), RemoveIdentities(), DagToCircuit()])
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)

        for _ in range(2):
            self.assertEqual(circuit, LoweringPassManager([task]).run(circuit))

    def test_edit_after_append_does_not_affect_pm(self):
        """Test that editing a task after it was appended does not change the pass manager."""
        placeholder = TaskSequence()
        pm = LoweringPassManager([CircuitToDag(), placeholder, DagToCircuit()])
        placeholder.append(RemoveIdentities())

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(1e-10, 0)

        # the pass manager lowered the empty placeholder, so the identity gate survives
        self.assertEqual(circuit, pm.run(circuit))

        # the edit is not visible in the pass manager, nor in a task sequence read back from it
        self.assertEqual([], pm.to_task_sequence()[1].tasks)
        self.assertEqual(circuit, LoweringPassManager(pm.to_task_sequence()).run(circuit))


class TestTaskSequenceProtocol(QiskitTestCase):
    """Tests for the sequence protocol of `TaskSequence`."""

    def setUp(self):
        super().setUp()
        self.a, self.b, self.c = RemoveIdentities(), CountGates(), RemoveIdentities()
        self.task_sequence = TaskSequence([self.a, self.b, self.c])

    def test_index(self):
        """Test indexing by position, including from the end."""
        self.assertIs(self.a, self.task_sequence[0])
        self.assertIs(self.c, self.task_sequence[-1])

    def test_index_out_of_range(self):
        """Test that an out-of-range position raises."""
        with self.assertRaises(IndexError):
            _ = self.task_sequence[3]

    def test_slice_returns_task_sequence(self):
        """Test that a slice returns a new task sequence of those tasks."""
        sliced = self.task_sequence[0:2]
        self.assertIsInstance(sliced, TaskSequence)
        self.assertEqual([self.a, self.b], sliced.tasks)

        # the slice is a new object, so editing it leaves the original alone
        sliced.append(RemoveIdentities())
        self.assertEqual(3, len(self.task_sequence))

    def test_slice_with_step(self):
        """Test that a strided slice is honoured."""
        self.assertEqual([self.c, self.b, self.a], self.task_sequence[::-1].tasks)

    def test_index_with_list(self):
        """Test that a list of positions returns those tasks in that order."""
        picked = self.task_sequence[[2, 0]]
        self.assertIsInstance(picked, TaskSequence)
        self.assertEqual([self.c, self.a], picked.tasks)

    def test_index_with_list_of_negatives(self):
        """Test that a list of positions accepts negative indices."""
        self.assertEqual([self.c, self.b], self.task_sequence[[-1, -2]].tasks)

    def test_index_with_empty_list(self):
        """Test that an empty list of positions gives an empty task sequence."""
        self.assertEqual([], self.task_sequence[[]].tasks)

    def test_index_with_repeated_position(self):
        """Test that a repeated position repeats the task."""
        self.assertEqual([self.a, self.a], self.task_sequence[[0, 0]].tasks)

    def test_index_with_list_out_of_range(self):
        """Test that an out-of-range position in a list raises."""
        with self.assertRaises(IndexError):
            _ = self.task_sequence[[0, 9]]

    def test_add(self):
        """Test that two task sequences concatenate into a new one."""
        out = TaskSequence([self.a]) + TaskSequence([self.b])
        self.assertEqual([self.a, self.b], out.tasks)

    def test_add_accepts_any_iterable(self):
        """Test that concatenation accepts any iterable of tasks."""
        self.assertEqual([self.a, self.b], (TaskSequence([self.a]) + [self.b]).tasks)
        self.assertEqual([self.a, self.b], (TaskSequence([self.a]) + iter([self.b])).tasks)

    def test_radd(self):
        """Test that an iterable of tasks concatenates with a task sequence on the right."""
        out = [self.a] + TaskSequence([self.b])
        self.assertIsInstance(out, TaskSequence)
        self.assertEqual([self.a, self.b], out.tasks)

    def test_add_defers_non_task_rejection(self):
        """Test that a non-task concatenates in and is rejected when it is appended."""
        task_sequence = TaskSequence([self.a]) + ["nope"]
        self.assertEqual([self.a, "nope"], task_sequence.tasks)
        with self.assertRaisesRegex(TypeError, "expected a Pass, TaskSequence or TaskStages"):
            _ = LoweringPassManager([task_sequence])

    def test_add_rejects_non_iterable(self):
        """Test that concatenating a non-iterable raises."""
        with self.assertRaises(TypeError):
            _ = TaskSequence([self.a]) + 5

    def test_add_task_stages_takes_names(self):
        """Test that concatenating task stages takes their names, since they iterate names."""
        stages = TaskStages([("init", self.b)])
        self.assertEqual([self.a, "init"], (TaskSequence([self.a]) + stages).tasks)

        # their tasks concatenate, and they nest as a single task
        self.assertEqual([self.a, self.b], (TaskSequence([self.a]) + stages.values()).tasks)
        self.assertEqual([self.a, stages], TaskSequence([self.a, stages]).tasks)

    def test_iadd_accepts_any_iterable(self):
        """Test that in-place concatenation extends from any iterable of tasks."""
        task_sequence = TaskSequence([self.a])
        task_sequence += [self.b]
        task_sequence += TaskSequence([self.c])
        self.assertEqual([self.a, self.b, self.c], task_sequence.tasks)

    def test_extend(self):
        """Test extending a task sequence from an iterable."""
        task_sequence = TaskSequence([self.a])
        task_sequence.extend([self.b, self.c])
        self.assertEqual([self.a, self.b, self.c], task_sequence.tasks)

    def test_copy_construct(self):
        """Test that a task sequence can be built from another task sequence."""
        self.assertEqual(self.task_sequence.tasks, TaskSequence(self.task_sequence).tasks)

    def test_is_mutable_sequence(self):
        """Test that a task sequence registers as a mutable sequence."""
        self.assertIsInstance(self.task_sequence, MutableSequence)

    def test_insert(self):
        """Test inserting a task before a position."""
        self.task_sequence.insert(1, CountGates())
        self.assertEqual([self.a, self.task_sequence[1], self.b, self.c], self.task_sequence.tasks)

    def test_delitem_and_remove(self):
        """Test deleting a task by position and by value."""
        del self.task_sequence[0]
        self.assertEqual([self.b, self.c], self.task_sequence.tasks)
        self.task_sequence.remove(self.c)
        self.assertEqual([self.b], self.task_sequence.tasks)

    def test_setitem(self):
        """Test replacing a task by position."""
        self.task_sequence[0] = self.b
        self.assertEqual([self.b, self.b, self.c], self.task_sequence.tasks)

    def test_eq(self):
        """Test that task sequences compare by their tasks."""
        self.assertEqual(TaskSequence([self.a]), TaskSequence([self.a]))
        self.assertNotEqual(TaskSequence([self.a]), TaskSequence([self.b]))
        self.assertNotEqual(TaskSequence([self.a]), TaskSequence())

    def test_eq_ignores_other_types(self):
        """Test that a task sequence does not compare equal to a plain list of the same tasks."""
        self.assertNotEqual(TaskSequence([self.a]), [self.a])

    def test_unhashable(self):
        """Test that a task sequence is unhashable, since it is mutable."""
        with self.assertRaises(TypeError):
            _ = hash(TaskSequence())


class TestTaskStagesMapping(QiskitTestCase):
    """Tests for the mapping protocol of `TaskStages`."""

    def setUp(self):
        super().setUp()
        self.a, self.b, self.c = RemoveIdentities(), CountGates(), RemoveIdentities()
        self.stages = TaskStages([("init", self.a), ("layout", self.b), ("opt", self.c)])

    def test_index_by_name(self):
        """Test indexing a stage by name."""
        self.assertIs(self.b, self.stages["layout"])

    def test_index_by_position(self):
        """Test that a position reaches the same task as the name."""
        self.assertIs(self.stages["init"], self.stages[0])
        self.assertIs(self.stages["opt"], self.stages[-1])

    def test_index_missing_name(self):
        """Test that an absent name raises."""
        with self.assertRaises(KeyError):
            _ = self.stages["nope"]

    def test_slice_returns_task_stages(self):
        """Test that a slice returns new task stages, keeping the names."""
        sliced = self.stages[1:3]
        self.assertIsInstance(sliced, TaskStages)
        self.assertEqual([("layout", self.b), ("opt", self.c)], list(sliced.items()))

    def test_index_with_list_of_positions(self):
        """Test that a list of positions returns those stages in that order."""
        picked = self.stages[[2, 0]]
        self.assertIsInstance(picked, TaskStages)
        self.assertEqual([("opt", self.c), ("init", self.a)], list(picked.items()))

    def test_index_with_list_of_names(self):
        """Test that a list of names returns those stages in that order."""
        self.assertEqual(
            [("opt", self.c), ("init", self.a)], list(self.stages[["opt", "init"]].items())
        )

    def test_index_with_mixed_list(self):
        """Test that a list may mix names and positions, including negative ones."""
        self.assertEqual(
            [("opt", self.c), ("init", self.a), ("layout", self.b)],
            list(self.stages[["opt", 0, -2]].items()),
        )

    def test_index_with_empty_list(self):
        """Test that an empty list gives empty task stages."""
        self.assertEqual([], list(self.stages[[]].items()))

    def test_index_with_list_leaves_original_alone(self):
        """Test that a list index returns new task stages."""
        picked = self.stages[[0]]
        picked["extra"] = self.b
        self.assertEqual(["init", "layout", "opt"], list(self.stages))

    def test_index_with_repeated_stage_raises(self):
        """Test that naming the same stage twice raises, since names must be unique."""
        with self.assertRaisesRegex(ValueError, "stage names must be unique"):
            _ = self.stages[["init", "init"]]
        with self.assertRaisesRegex(ValueError, "stage names must be unique"):
            _ = self.stages[["init", 0]]

    def test_index_with_list_rejects_bad_positions(self):
        """Test that an absent name or out-of-range position in a list raises."""
        with self.assertRaisesRegex(KeyError, "no stage named 'nope'"):
            _ = self.stages[["nope"]]
        with self.assertRaisesRegex(IndexError, "out of range"):
            _ = self.stages[[9]]

    def test_copy_construct(self):
        """Test that task stages can be built from other stages, and from a dict."""
        self.assertEqual(list(self.stages.items()), list(TaskStages(self.stages).items()))
        self.assertEqual(dict(self.stages), dict(self.stages.items()))

    def test_keywords_follow_positional_stages(self):
        """Test that keyword stages are appended after the positional ones."""
        stages = TaskStages([("init", self.a)], opt=self.c)
        self.assertEqual([("init", self.a), ("opt", self.c)], list(stages.items()))

    def test_keyword_repeating_positional_name_raises(self):
        """Test that a name given both positionally and by keyword raises."""
        with self.assertRaisesRegex(ValueError, "stage names must be unique"):
            TaskStages({"x": self.a}, x=self.b)

    def test_is_mutable_mapping(self):
        """Test that task stages register as a mutable mapping."""
        self.assertIsInstance(self.stages, MutableMapping)

    def test_delitem(self):
        """Test deleting a stage by name."""
        del self.stages["layout"]
        self.assertEqual(["init", "opt"], list(self.stages))

    def test_duplicate_name_raises(self):
        """Test that a repeated stage name raises."""
        with self.assertRaisesRegex(ValueError, "stage names must be unique"):
            TaskStages([("x", self.a), ("x", self.b)])

    def test_eq(self):
        """Test that task stages compare by their names and tasks."""
        self.assertEqual(TaskStages([("x", self.a)]), TaskStages([("x", self.a)]))
        self.assertNotEqual(TaskStages([("x", self.a)]), TaskStages([("y", self.a)]))
        self.assertNotEqual(TaskStages([("x", self.a)]), TaskStages([("x", self.b)]))

    def test_eq_ignores_other_mappings(self):
        """Test that task stages do not compare equal to a plain dict of their stages."""
        self.assertNotEqual(TaskStages([("x", self.a)]), {"x": self.a})

    def test_eq_respects_order(self):
        """Test that stage order is part of equality, unlike for a plain dict."""
        one = TaskStages([("x", self.a), ("y", self.b)])
        other = TaskStages([("y", self.b), ("x", self.a)])
        self.assertNotEqual(one, other)
        self.assertEqual(dict(one), dict(other))

    def test_unhashable(self):
        """Test that task stages are unhashable, since they are mutable."""
        with self.assertRaises(TypeError):
            _ = hash(TaskStages())

    def test_add(self):
        """Test that two sets of task stages concatenate into a new one."""
        out = TaskStages([("x", self.a)]) + TaskStages([("y", self.b)])
        self.assertEqual([("x", self.a), ("y", self.b)], list(out.items()))

    def test_add_with_repeated_name_keeps_position(self):
        """Test that a name in both operands keeps its position and takes the incoming task."""
        out = TaskStages([("x", self.a), ("z", self.c)]) + TaskStages([("x", self.b)])
        self.assertEqual([("x", self.b), ("z", self.c)], list(out.items()))

    def test_add_accepts_mapping_or_pairs(self):
        """Test that concatenation accepts a mapping or an iterable of pairs."""
        base = TaskStages([("x", self.a)])
        expected = [("x", self.a), ("y", self.b)]
        self.assertEqual(expected, list((base + [("y", self.b)]).items()))
        self.assertEqual(expected, list((base + {"y": self.b}).items()))

    def test_radd(self):
        """Test that pairs concatenate with task stages on the right."""
        out = [("x", self.a)] + TaskStages([("y", self.b)])
        self.assertIsInstance(out, TaskStages)
        self.assertEqual([("x", self.a), ("y", self.b)], list(out.items()))

    def test_add_rejects_bad_shapes(self):
        """Test that concatenating a badly-typed name or a non-pair raises."""
        base = TaskStages([("x", self.a)])
        with self.assertRaisesRegex(TypeError, "mapping or an iterable of"):
            _ = base + [(5, self.b)]
        with self.assertRaisesRegex(TypeError, "mapping or an iterable of"):
            _ = base + [self.b]
        with self.assertRaises(TypeError):
            _ = base + 5

    def test_add_defers_non_task_rejection(self):
        """Test that a non-task concatenates in and is rejected when it is appended."""
        stages = TaskStages([("x", CircuitToDag())]) + [("y", "nope")]
        self.assertEqual("nope", stages["y"])
        with self.assertRaisesRegex(TypeError, "expected a Pass, TaskSequence or TaskStages"):
            _ = LoweringPassManager([stages])

    def test_iadd_accepts_any_iterable(self):
        """Test that in-place concatenation extends from any iterable of pairs."""
        stages = TaskStages([("x", self.a)])
        stages += [("y", self.b)]
        self.assertEqual([("x", self.a), ("y", self.b)], list(stages.items()))

    def test_update(self):
        """Test updating task stages from an iterable of pairs."""
        stages = TaskStages([("x", self.a)])
        stages.update([("y", self.b), ("z", self.c)])
        self.assertEqual(["x", "y", "z"], list(stages))

    def test_insert_after_by_name(self):
        """Test inserting directly after a named stage."""
        out = self.stages.insert_after("init", ("new", self.b))
        self.assertEqual(["init", "new", "layout", "opt"], list(out))

    def test_insert_after_by_position(self):
        """Test inserting directly after a position, including from the end."""
        self.assertEqual(
            ["init", "new", "layout", "opt"], list(self.stages.insert_after(0, ("new", self.b)))
        )
        self.assertEqual(
            ["init", "layout", "opt", "new"], list(self.stages.insert_after(-1, ("new", self.b)))
        )

    def test_insert_before_by_name(self):
        """Test inserting directly before a named stage."""
        out = self.stages.insert_before("layout", ("new", self.b))
        self.assertEqual(["init", "new", "layout", "opt"], list(out))

    def test_insert_before_by_position(self):
        """Test inserting directly before a position, including from the end."""
        self.assertEqual(
            ["new", "init", "layout", "opt"], list(self.stages.insert_before(0, ("new", self.b)))
        )
        self.assertEqual(
            ["init", "layout", "new", "opt"], list(self.stages.insert_before(-1, ("new", self.b)))
        )

    def test_insert_at_the_ends(self):
        """Test inserting after the last stage and before the first."""
        self.assertEqual(
            ["init", "layout", "opt", "new"], list(self.stages.insert_after("opt", ("new", self.b)))
        )
        self.assertEqual(
            ["new", "init", "layout", "opt"],
            list(self.stages.insert_before("init", ("new", self.b))),
        )

    def test_insert_chains(self):
        """Test inserting several stages by chaining the calls."""
        out = self.stages.insert_after("init", ("m", self.b)).insert_after("m", ("n", self.c))
        self.assertEqual(["init", "m", "n", "layout", "opt"], list(out))

    def test_insert_moves_an_existing_name(self):
        """Test that inserting a name already present moves it rather than duplicating it."""
        out = self.stages.insert_after("init", ("opt", self.b))
        self.assertEqual(["init", "opt", "layout"], list(out))
        self.assertIs(self.b, out["opt"])

    def test_insert_leaves_the_original_alone(self):
        """Test that inserting returns new task stages."""
        out = self.stages.insert_after("init", ("new", self.b))
        self.assertIsNot(self.stages, out)
        self.assertEqual(["init", "layout", "opt"], list(self.stages))

    def test_insert_rejects_bad_position(self):
        """Test that an absent name or out-of-range position raises."""
        with self.assertRaisesRegex(KeyError, "no stage named 'nope'"):
            self.stages.insert_after("nope", ("new", self.b))
        with self.assertRaisesRegex(IndexError, "out of range"):
            self.stages.insert_after(3, ("new", self.b))
        with self.assertRaisesRegex(IndexError, "out of range"):
            self.stages.insert_before(-4, ("new", self.b))

    def test_insert_rejects_bad_shapes(self):
        """Test that a badly-typed name or a non-pair raises."""
        with self.assertRaisesRegex(TypeError, r"expected a \(name, task\) pair"):
            self.stages.insert_after("init", (5, self.b))
        with self.assertRaisesRegex(TypeError, r"expected a \(name, task\) pair"):
            self.stages.insert_after("init", [("m", self.b), ("n", self.c)])

    def test_inserted_stage_runs(self):
        """Test running a pass manager whose stage was inserted."""
        stages = TaskStages([("lower", CircuitToDag()), ("raise", DagToCircuit())])
        stages = stages.insert_after("lower", ("clean", RemoveIdentities()))
        pm = LoweringPassManager([stages])

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(1e-10, 0)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        self.assertEqual(expected, pm.run(circuit))

    def test_sliced_stage_runs(self):
        """Test running a pass manager built from a slice of task stages."""
        pm = LoweringPassManager([CircuitToDag(), self.stages[0:1], DagToCircuit()])

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(1e-10, 0)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        self.assertEqual(expected, pm.run(circuit))


class TestRepr(QiskitTestCase):
    """Tests for the reprs of tasks and pass managers."""

    def test_repr_runs(self):
        """Test that taking the repr of any task or pass manager neither raises nor hangs."""
        stages = TaskStages(lower=TaskSequence([CircuitToDag()]), raise_=DagToCircuit())
        for index, obj in enumerate(
            [
                CircuitToDag(),
                TaskSequence(),
                TaskStages(),
                stages,
                TaskSequence([stages]),
                LoweringPassManager(),
                LoweringPassManager([stages]),
            ]
        ):
            with self.subTest(index=index):
                _ = repr(obj)

    def test_repr_of_cycle_terminates(self):
        """Test that a container holding itself does not have a repr that recurses forever."""
        task_sequence = TaskSequence()
        task_sequence.append(task_sequence)
        self.assertLess(len(repr(task_sequence)), 1000)

        stages = TaskStages()
        stages["me"] = stages
        self.assertLess(len(repr(stages)), 1000)


class TestToTaskSequence(QiskitTestCase):
    """Tests for reading a pass manager's tasks back out."""

    def test_round_trips(self):
        """Test that `to_task_sequence` round-trips a pass manager's tasks."""
        tasks = [CircuitToDag(), RemoveIdentities(), DagToCircuit()]
        pm = LoweringPassManager(tasks)
        task_sequence = pm.to_task_sequence()
        self.assertEqual(task_sequence.tasks, tasks)

        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        self.assertEqual(pm.run(circuit), LoweringPassManager(task_sequence).run(circuit))

    def test_reflects_what_runs(self):
        """Test that the read-back shows the task sequence rather than the appended objects."""
        placeholder = TaskSequence()
        pm = LoweringPassManager([CircuitToDag(), placeholder, DagToCircuit()])
        placeholder.append(RemoveIdentities())

        # the pass manager holds the empty placeholder it lowered, not the edited one
        read_back = pm.to_task_sequence()
        self.assertEqual([], read_back[1].tasks)
        self.assertIsNot(placeholder, read_back[1])
        self.assertEqual(
            "TaskSequence([CircuitToDag(), TaskSequence([]), DagToCircuit()])", repr(read_back)
        )

    def test_preserves_pass_identity(self):
        """Test that the read-back returns the original pass objects."""
        a, b = CircuitToDag(), DagToCircuit()
        pm = LoweringPassManager([a, TaskSequence([b])])
        read_back = pm.to_task_sequence()
        self.assertIs(a, read_back[0])
        self.assertIs(b, read_back[1][0])

    def test_rebuilds_nested_containers(self):
        """Test that the read-back rebuilds task sequences and stages at any depth."""
        raise_ = DagToCircuit()
        pm = LoweringPassManager(
            [TaskStages([("lower", TaskSequence([CircuitToDag()])), ("raise", raise_)])]
        )
        outer = pm.to_task_sequence()[0]
        self.assertIsInstance(outer, TaskStages)
        self.assertEqual(["lower", "raise"], list(outer))
        self.assertIsInstance(outer["lower"], TaskSequence)
        self.assertIs(raise_, outer["raise"])

    def test_does_not_alias(self):
        """Test that appending to the returned task sequence leaves the pass manager alone."""
        pm = LoweringPassManager([CircuitToDag()])
        task_sequence = pm.to_task_sequence()
        task_sequence.append(RemoveIdentities())

        self.assertEqual(1, len(pm.to_task_sequence()))
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.rz(1e-10, 0)
        self.assertEqual(circuit.to_dag(), pm.run(circuit))
