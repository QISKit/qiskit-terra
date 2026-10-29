# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest

from qiskit import QuantumCircuit
from qiskit._accelerate import mir
from qiskit.dagcircuit import DAGCircuit
from qiskit.passmanager import Pass, LoweringPassManager
from qiskit.transpiler import generate_preset_pbc_pass_manager
from qiskit.transpiler.passes import LitinskiTransformation


class CircuitToDag(Pass[QuantumCircuit, DAGCircuit]):
    _qiskit_pass_ir_in_ = QuantumCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def _qiskit_pass_run_(self, ir, context):
        return ir.to_dag(copy_operations=False)


class LegacyDagPass(Pass[DAGCircuit]):
    _qiskit_pass_ir_in_ = DAGCircuit
    _qiskit_pass_ir_out_ = DAGCircuit

    def __init__(self, task):
        super().__init__()
        self._task = task

    def _qiskit_pass_run_(self, ir, context):
        return self._task.run(ir)


def make_bell_circuit():
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


class TestBellCircuitMIR(unittest.TestCase):
    """Unit tests for lowering Bell circuit to MIR."""
    
    def test_bell_circuit(self):
        print("\n--- Running without Litinski Transformation ---")
        qc = make_bell_circuit()
        pm = LoweringPassManager(
            [CircuitToDag()]
            + [LegacyDagPass(pass_) for pass_ in generate_preset_pbc_pass_manager().to_flow_controller().tasks]
            + [mir.DagToMirPass()]
        )
        mir_instance = pm.run(qc)
        print(mir_instance)
        self.assertIsInstance(mir_instance, mir.Mir)

    def test_bell_circuit_with_litinski(self):
        print("\n--- Running with Litinski Transformation ---")
        qc = make_bell_circuit()
        pm = LoweringPassManager(
            [CircuitToDag()]
            + [LegacyDagPass(pass_) for pass_ in generate_preset_pbc_pass_manager().to_flow_controller().tasks]
            + [LegacyDagPass(LitinskiTransformation(fix_clifford=False))]
            + [mir.DagToMirPass()]
        )
        mir_instance = pm.run(qc)
        print(mir_instance)
        self.assertIsInstance(mir_instance, mir.Mir)


if __name__ == "__main__":
    unittest.main()