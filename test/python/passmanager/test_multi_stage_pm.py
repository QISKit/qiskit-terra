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


"""Test the multi-stage pass manager."""

from qiskit.passmanager.multistage_passmanager import PropertySet
from test import QiskitTestCase
from copy import copy

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveIdentityEquivalent
from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import MultiStagePassManager
from qiskit.passmanager.base_tasks import GenericPass
from qiskit.passmanager import BasePassManager
from qiskit.transpiler import generate_preset_pass_manager, CouplingMap


class PauliIR:
    """A bare minimum Pauli-string IR keeping a dense list of global Paulis to apply."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.instructions = []

    def apply(self, pauli: str):
        """Append a Pauli. Must have global size."""
        if len(pauli) != self.num_qubits:
            raise ValueError("Incompatible number of qubits")
        self.instructions.append(pauli)


class PauliPM(BasePassManager):
    """A pass manager for the Pauli IR. Has trivial conversions."""

    def _passmanager_frontend(self, input_program, **kwargs):
        return input_program

    def _passmanager_backend(self, passmanager_ir, in_program, **kwargs):
        return passmanager_ir


class IdentifyIdentities(GenericPass[PauliIR, PauliIR]):
    """An analysis pass identifying Paul identities."""

    def run(self, passmanager_ir: PauliIR) -> PauliIR:
        to_remove = []
        for i, pauli in enumerate(passmanager_ir.instructions):
            if all(p == "I" for p in pauli):
                to_remove.append(i)

        # store it in the property set
        self.property_set["pauli_identities"] = to_remove


class RemovePauliIdentities(GenericPass[PauliIR, PauliIR]):
    """A pass removing Pauli identities."""

    def __init__(self):
        super().__init__()
        self.requires = [IdentifyIdentities()]

    def run(self, passmanager_ir):
        # remove from the back to keep indices valid
        for i in reversed(self.property_set["pauli_identities"]):
            del passmanager_ir.instructions[i]

        return passmanager_ir


class PauliToCircuit(GenericPass[PauliIR, QuantumCircuit]):
    """Lower PauliIR to QuantumCircuit."""

    def run(self, passmanager_ir):
        circuit = QuantumCircuit(passmanager_ir.num_qubits)
        for pauli in passmanager_ir.instructions:
            circuit.pauli(pauli, circuit.qubits)
        return circuit


class CircuitAnalysis(GenericPass[QuantumCircuit, QuantumCircuit]):
    """A task counting the number of operations and storing them in the property set."""

    def run(self, passmanager_ir):
        self.property_set["ops"] = passmanager_ir.count_ops()
        return passmanager_ir


class CircuitToDAG(GenericPass[QuantumCircuit, DAGCircuit]):
    """A lowering task from circuit to DAG."""

    def run(self, passmanager_ir):
        return circuit_to_dag(passmanager_ir)


class CircuitDecomposer(GenericPass[QuantumCircuit, QuantumCircuit]):
    """A pass decomposing the circuit one level."""

    def run(self, passmanager_ir):
        return passmanager_ir.decompose()


class RequireKey(GenericPass[QuantumCircuit, QuantumCircuit]):
    """A task that raises if a required key is absent from the property set."""

    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def run(self, passmanager_ir):
        if self.key not in self.property_set:
            raise ValueError(f"Required property ({self.key}) is not set.")
        return passmanager_ir


class TestMultiStagePM(QiskitTestCase):
    """Test the multi-stage pass manager."""

    def test_stage_names(self):
        """Test getting the stage names."""
        opt1 = CircuitDecomposer()
        lower = CircuitToDAG()
        opt2 = RemoveIdentityEquivalent()

        staged_pm = MultiStagePassManager(first=opt1, lower=lower, last=opt2)
        expected = ("first", "lower", "last")
        with self.subTest(msg="stage names"):
            self.assertEqual(expected, staged_pm.stages)

        with self.subTest(msg="getting stages"):
            self.assertEqual(opt1, getattr(staged_pm, expected[0]))
            self.assertEqual(lower, getattr(staged_pm, expected[1]))
            self.assertEqual(opt2, getattr(staged_pm, expected[2]))

    def test_multi_ir_flow(self):
        """Run a non-trivial multi-IR example."""
        pauli = PauliPM([IdentifyIdentities(), RemovePauliIdentities()])
        pauli_to_circuit = PauliToCircuit()
        circuit = CircuitDecomposer()
        qc_to_dag = CircuitToDAG()

        staged_pm = MultiStagePassManager(
            pauli=pauli,
            pauli_to_circuit=pauli_to_circuit,
            circuit=circuit,
            circuit_to_dag=qc_to_dag,
        )
        input_program = PauliIR(3)
        input_program.apply("XYZ")
        input_program.apply("ZZI")
        input_program.apply("III")
        input_program.apply("IYI")

        output_program = staged_pm.run(input_program)

        expected = QuantumCircuit(3)
        expected.x(2)
        expected.y(1)
        expected.z(0)

        expected.z(2)
        expected.z(1)

        expected.y(1)

        self.assertEqual(circuit_to_dag(expected), output_program)

    def test_generate_preset_pm_compatibility(self):
        """Check the existing pipeline can be a stage."""

        pauli = PauliPM([IdentifyIdentities(), RemovePauliIdentities()])
        pauli_to_circuit = PauliToCircuit()
        qc_to_dag = CircuitToDAG()
        dag = generate_preset_pass_manager(
            coupling_map=CouplingMap.from_line(10), basis_gates=["sx", "x", "rz", "cx"]
        )

        staged_pm = MultiStagePassManager(
            pauli=pauli,
            pauli_to_circuit=pauli_to_circuit,
            circuit_to_dag=qc_to_dag,
            dag=dag,
        )
        input_program = PauliIR(3)
        input_program.apply("ZZI")
        input_program.apply("ZZI")
        input_program.apply("III")
        input_program.apply("IXI")

        output_program = staged_pm.run(input_program)

        self.assertEqual(output_program.count_ops(), {"x": 1})
        self.assertEqual(output_program.num_qubits(), 10)

    def test_stage_replacement(self):
        """Test replacing stages."""
        staged_pm = MultiStagePassManager(qc_to_dag=CircuitToDAG(), dag_opt=[])

        circuit = QuantumCircuit(1)
        circuit.rz(1e-10, 0)

        with self.subTest(msg="no op"):
            out = staged_pm.run(circuit)
            self.assertEqual(circuit_to_dag(circuit), out)

        # change the optimization stage to remove close-to-identity gates
        staged_pm.dag_opt = RemoveIdentityEquivalent()

        with self.subTest(msg="remove identities"):
            out = staged_pm.run(circuit)
            self.assertEqual(0, sum(out.count_ops().values()))

    def test_callback(self):
        """Test the callback works."""
        circuit = [CircuitDecomposer(), CircuitAnalysis()]
        lower = CircuitToDAG()
        dag = RemoveIdentityEquivalent()

        staged_pm = MultiStagePassManager(circuit=circuit, circuit_to_dag=lower, dag=dag)

        # Store all data from the callback to verify it has been passed correctly. This should
        # contain 4 tasks: CircuitNoOp, CircuitAnalysis, CircuitToDag, and DAGRemoveIdentity
        properties = []

        def callback(task, passmanager_ir, property_set, running_time, count):
            properties.append(
                (
                    task.__class__.__name__,
                    copy(passmanager_ir),
                    copy(property_set),
                    running_time,
                    count,
                )
            )

        input_program = QuantumCircuit(3)
        input_program.h(0)
        input_program.rz(1e-10, 0)
        input_program.cswap(0, 1, 2)

        out = staged_pm.run(input_program, callback=callback)

        # Test after task 0
        decomp = input_program.decompose()
        self.assertEqual(properties[0][0], "CircuitDecomposer")
        self.assertEqual(properties[0][1], decomp)
        self.assertEqual(properties[0][2], {})  # empty property set

        # Test after task 1
        self.assertEqual(properties[1][0], "CircuitAnalysis")
        self.assertEqual(properties[1][1], decomp)
        self.assertEqual(properties[1][2]["ops"], decomp.count_ops())

        # Test after task 2
        self.assertEqual(properties[2][0], "CircuitToDAG")
        self.assertEqual(properties[2][1], circuit_to_dag(decomp))
        self.assertEqual(properties[2][2]["ops"], properties[1][2]["ops"])  # property set remains

        # Test after task 3
        self.assertEqual(properties[3][0], "RemoveIdentityEquivalent")
        self.assertEqual(properties[3][1], out)
        self.assertEqual(properties[3][2]["ops"], properties[1][2]["ops"])

        # All recorded runtimes should be non-negative, and the tasks should be counted
        for i, prop in enumerate(properties):
            self.assertGreaterEqual(prop[3], 0.0)  # runtime
            self.assertEqual(prop[4], i)  # task count

    def test_initial_property_set(self):
        """Test that a pre-populated property set is visible to tasks."""
        pm = MultiStagePassManager(circuit_opt=RequireKey("seed"))
        initial = {"seed": 42}
        circuit = QuantumCircuit(1)

        with self.subTest(msg="valid initial set"):
            out = pm.run(circuit, property_set=initial)
            self.assertEqual(out, QuantumCircuit(1))

        with self.subTest(msg="missing initial set"):
            with self.assertRaises(ValueError):
                _ = pm.run(QuantumCircuit(1))

    def test_nesting(self):
        """Test nesting a multi-stage inside a multi-stage via FlowControllerLinear."""
        inner = MultiStagePassManager(
            pauli=RemovePauliIdentities(), to_circuit=PauliToCircuit(), circuit=CircuitDecomposer()
        )

        outer = MultiStagePassManager(
            pauli_to_circuit=inner.to_flow_controller(), to_dag=CircuitToDAG()
        )

        input_program = PauliIR(3)
        input_program.apply("ZZI")

        out = outer.run(input_program)

        expected = QuantumCircuit(3)
        expected.z([2, 1])

        self.assertEqual(circuit_to_dag(expected), out)

    def test_run_on_multiple_inputs(self):
        pm = MultiStagePassManager(
            pauli=RemovePauliIdentities(), to_circuit=PauliToCircuit(), circuit=CircuitDecomposer()
        )
        programs = []
        expected = []
        for pauli in ("ZZI", "IZZ", "ZIZ"):
            program = PauliIR(3)
            program.apply(pauli)

            circuit = QuantumCircuit(3)
            circuit.z([2 - i for i, p in enumerate(pauli) if p == "Z"])

            programs.append(program)
            expected.append(circuit)

        self.assertEqual(pm.run(programs), expected)

    def test_run_on_multiple_inputs_with_property_set(self):
        pm = MultiStagePassManager(
            pauli=RemovePauliIdentities(),
            to_circuit=PauliToCircuit(),
            circuit=CircuitDecomposer(),
        )
        programs = []
        for pauli in ("ZZI", "IZZ", "ZIZ"):
            program = PauliIR(3)
            program.apply(pauli)

            programs.append(program)

        with self.assertRaisesRegex(
            ValueError, "a 'property_set' cannot be provided when passing multiple input programs"
        ):
            pm.run(programs, property_set=PropertySet())
