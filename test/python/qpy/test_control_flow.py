# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0.

import io
import unittest

from qiskit import QuantumCircuit, qpy


class TestQPYControlFlow(unittest.TestCase):
    """Regression tests for QPY control-flow deserialization."""

    def test_while_loop_preserves_bit_identity(self):
        """WhileLoopOp should deserialize without invalid bit references."""

        qc = QuantumCircuit(1, 1)

        with qc.while_loop((qc.clbits[0], True)):
            qc.x(0)

        buffer = io.BytesIO()
        qpy.dump(qc, buffer)
        buffer.seek(0)

        new_circuit = qpy.load(buffer)[0]

        # This must not raise CircuitError
        new_circuit.draw(output="text")
