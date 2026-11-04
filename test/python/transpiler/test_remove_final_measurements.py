# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test RemoveFinalMeasurements pass"""

import unittest

from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.circuit import Clbit
from qiskit.transpiler.passes import RemoveFinalMeasurements
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes.utils.remove_final_measurements import calc_final_ops
from test import QiskitTestCase


class TestRemoveFinalMeasurements(QiskitTestCase):
    """Test removing final measurements."""

    def test_multi_bit_register_removed_with_clbits(self):
        """Remove register when all clbits removed."""

        def expected_dag():
            q0 = QuantumRegister(2, "q0")
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)

        q0 = QuantumRegister(2, "q0")
        c0 = ClassicalRegister(2, "c0")
        qc = QuantumCircuit(q0, c0)

        # measure into all clbits of c0
        qc.measure(0, 0)
        qc.measure(1, 1)

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_overlapping_register_removal(self):
        """Only registers that become idle directly as a result of
        final op removal are removed. In this test, a 5-bit creg
        is implicitly created with its own bits, along with cregs
        ``c0_lower_3`` and ``c0_upper_3`` which reuse those underlying bits.
        ``c0_lower_3`` and ``c0_upper_3`` reference only 1 bit in common.
        A final measure is performed into a bit that exists in ``c0_lower_3``
        but not in ``c0_upper_3``, and subsequently is removed. Consequently,
        both ``c0_lower_3`` and the 5-bit register are removed, because they
        have become unused as a result of the final measure removal.
        ``c0_upper_3`` remains, because it was idle beforehand, not as a
        result of the measure removal, along with all of its bits,
        including the bit shared with ``c0_lower_3``."""

        def expected_dag():
            q0 = QuantumRegister(3, "q0")
            c0 = ClassicalRegister(5, "c0")
            c0_upper_3 = ClassicalRegister(name="c0_upper_3", bits=c0[2:])

            # note c0 is *not* added to circuit!
            qc = QuantumCircuit(q0, c0_upper_3)
            return circuit_to_dag(qc)

        q0 = QuantumRegister(3, "q0")
        c0 = ClassicalRegister(5, "c0")
        qc = QuantumCircuit(q0, c0)

        c0_lower_3 = ClassicalRegister(name="c0_lower_3", bits=c0[:3])
        c0_upper_3 = ClassicalRegister(name="c0_upper_3", bits=c0[2:])
        # Only qc.clbits[2] is shared between the two.

        qc.add_register(c0_lower_3)
        qc.add_register(c0_upper_3)

        qc.measure(0, c0_lower_3[0])

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertListEqual(list(dag.cregs.values()), [c0_upper_3])
        self.assertListEqual(dag.clbits, list(c0_upper_3))
        self.assertEqual(dag, expected_dag())

    def test_multi_bit_register_removed_if_all_bits_idle(self):
        """A multibit register is removed when all bits are idle."""

        def expected_dag():
            q0 = QuantumRegister(1, "q0")
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)

        q0 = QuantumRegister(1, "q0")
        c0 = ClassicalRegister(2, "c0")
        qc = QuantumCircuit(q0, c0)

        # measure into single bit c0[0] of c0
        qc.measure(0, 0)

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_multi_reg_shared_bits_removed(self):
        """All registers sharing removed bits should be removed."""

        def expected_dag():
            q0 = QuantumRegister(2, "q0")
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)

        q0 = QuantumRegister(2, "q0")
        c0 = ClassicalRegister(2, "c0")
        qc = QuantumCircuit(q0, c0)

        # Create reg with shared bits (same as c0)
        c1 = ClassicalRegister(name="c1", bits=qc.clbits)
        qc.add_register(c1)

        # measure into all clbits of c0
        qc.measure(0, c0[0])
        qc.measure(1, c0[1])

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_final_measures_share_dest(self):
        """Multiple final measurements use the same clbit."""

        def expected_dag():
            qc = QuantumCircuit(QuantumRegister(2, "q0"))
            return circuit_to_dag(qc)

        rq = QuantumRegister(2, "q0")
        rc = ClassicalRegister(1, "c0")
        qc = QuantumCircuit(rq, rc)

        qc.measure(0, 0)
        qc.measure(1, 0)

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertEqual(dag, expected_dag())

    def test_remove_chained_final_measurements(self):
        """Remove successive final measurements."""

        def expected_dag():
            q0 = QuantumRegister(1, "q0")
            q1 = QuantumRegister(1, "q1")
            c0 = ClassicalRegister(1, "c0")
            qc = QuantumCircuit(q0, c0, q1)

            qc.measure(q0, c0)
            qc.measure(q0, c0)
            qc.barrier()
            qc.h(q1)

            return circuit_to_dag(qc)

        q0 = QuantumRegister(1, "q0")
        q1 = QuantumRegister(1, "q1")
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        qc = QuantumCircuit(q0, c0, q1, c1)

        qc.measure(q0, c0)
        qc.measure(q0, c0)
        qc.barrier()
        qc.h(q1)
        qc.measure(q1, c1)
        qc.measure(q0, c1)

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertEqual(dag, expected_dag())

    def test_remove_clbits_without_register(self):
        """clbits of final measurements not in a register are removed."""

        def expected_dag():
            q0 = QuantumRegister(1, "q0")
            qc = QuantumCircuit(q0)
            return circuit_to_dag(qc)

        q0 = QuantumRegister(1, "q0")
        qc = QuantumCircuit(q0)

        # Add clbit without adding register
        qc.add_bits([Clbit()])

        self.assertFalse(qc.cregs)

        # Measure to regless clbit
        qc.measure(0, 0)

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertFalse(dag.cregs)
        self.assertFalse(dag.clbits)
        self.assertEqual(dag, expected_dag())

    def test_final_barriers_and_measures_complex(self):
        """Test complex final barrier and measure removal."""

        def expected_dag():
            q0 = QuantumRegister(5, "q0")
            c1 = ClassicalRegister(1, "c1")
            qc = QuantumCircuit(q0, c1)
            qc.h(q0[0])
            return circuit_to_dag(qc)

        #       ┌───┐┌─┐ ░     ░ ┌─┐
        # q0_0: ┤ H ├┤M├─░─────░─┤M├───────────────
        #       └┬─┬┘└╥┘ ░     ░ └╥┘┌─┐
        # q0_1: ─┤M├──╫──░─────░──╫─┤M├────────────
        #        └╥┘  ║  ░  ░  ░  ║ └╥┘┌─┐
        # q0_2: ──╫───╫──░──░──░──╫──╫─┤M├─────────
        #         ║   ║  ░  ░  ░  ║  ║ └╥┘┌─┐
        # q0_3: ──╫───╫──░──░──░──╫──╫──╫─┤M├──────
        #         ║   ║  ░  ░  ░  ║  ║  ║ └╥┘┌─┐ ░
        # q0_4: ──╫───╫──░─────░──╫──╫──╫──╫─┤M├─░─
        #         ║   ║  ░     ░  ║  ║  ║  ║ └╥┘ ░
        # c0: 1/══╩═══╩═══════════╬══╬══╬══╬══╬════
        #         0   0           ║  ║  ║  ║  ║
        #                         ║  ║  ║  ║  ║
        # c1: 1/══════════════════╬══╬══╬══╬══╬════
        #                         ║  ║  ║  ║  ║
        # meas: 5/════════════════╩══╩══╩══╩══╩════
        #                         0  1  2  3  4
        q0 = QuantumRegister(5, "q0")
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        qc = QuantumCircuit(q0, c0, c1)

        qc.measure(q0[1], c0)
        qc.h(q0[0])
        qc.measure(q0[0], c0[0])
        qc.barrier()
        qc.barrier(q0[2], q0[3])
        qc.measure_all()
        qc.barrier(q0[4])

        dag = circuit_to_dag(qc)
        dag = RemoveFinalMeasurements().run(dag)

        self.assertEqual(dag, expected_dag())

    def test_calc_final_ops(self):
        """Test method to find the final operations in a circuit."""
        q0 = QuantumRegister(5, "q0")
        c0 = ClassicalRegister(1, "c0")
        c1 = ClassicalRegister(1, "c1")
        qc = QuantumCircuit(q0, c0, c1)

        #       ┌───┐┌─┐ ░
        # q0_0: ┤ H ├┤M├─░────
        #       └┬─┬┘└╥┘ ░
        # q0_1: ─┤M├──╫──░────
        #        └╥┘  ║  ░
        # q0_2: ──╫───╫──░────
        #         ║   ║  ░ ┌─┐
        # q0_3: ──╫───╫──░─┤M├
        #         ║   ║  ░ └╥┘
        # q0_4: ──╫───╫──░──╫─
        #         ║   ║  ░  ║
        # c0: 1/══╩═══╩═════╬═
        #         0   0     ║
        # c1: 1/════════════╩═
        #                   0

        qc.measure(q0[1], c0)
        qc.h(q0[0])
        qc.measure(q0[0], c0)
        qc.barrier()
        qc.measure(q0[3], c1)
        final_measurements = [
            node.qargs for node in calc_final_ops(circuit_to_dag(qc), {"measure"})
        ]
        self.assertEqual([(q0[3],)], final_measurements)

        final_measurements_barriers = calc_final_ops(circuit_to_dag(qc), {"measure", "barrier"})
        self.assertEqual(len(final_measurements_barriers), 4)

        final_barriers = calc_final_ops(circuit_to_dag(qc), {"barrier"})
        self.assertEqual(final_barriers, [])


if __name__ == "__main__":
    unittest.main()
