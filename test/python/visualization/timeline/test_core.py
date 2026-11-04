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

"""Tests for core modules of timeline drawer."""

from qiskit import QuantumCircuit, transpile
from qiskit.visualization.timeline import core, stylesheet, generators, layouts
from qiskit.transpiler.target import Target, InstructionProperties
from qiskit.circuit import Delay, Parameter, Measure
from qiskit.circuit.library import HGate, CXGate
from test import QiskitTestCase


class TestCanvas(QiskitTestCase):
    """Test for canvas."""

    def setUp(self):
        super().setUp()

        self.style = stylesheet.QiskitTimelineStyle()
        circ = QuantumCircuit(4)
        circ.h(0)
        circ.barrier()
        circ.cx(0, 2)
        circ.cx(1, 3)

        self.target = Target(num_qubits=4, dt=1e-7)
        self.target.add_instruction(HGate(), {(0,): InstructionProperties(duration=200 * 1e-7)})
        self.target.add_instruction(
            CXGate(),
            {
                (0, 2): InstructionProperties(duration=1000 * 1e-7),
                (1, 3): InstructionProperties(duration=1000 * 1e-7),
            },
        )
        self.target.add_instruction(Delay(Parameter("t")))

        self.circ = transpile(
            circ,
            scheduling_method="alap",
            target=self.target,
            dt=1e-7,
            optimization_level=0,
        )

    def test_time_range(self):
        """Test calculating time range."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.formatter = {"margin.left_percent": 0.1, "margin.right_percent": 0.1}
        canvas.time_range = (0, 100)

        ref_range = [-10.0, 110.0]
        self.assertListEqual(list(canvas.time_range), ref_range)

    def test_load_program(self):
        """Test loading program."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.generator = {
            "gates": [generators.gen_sched_gate],
            "bits": [],
            "barriers": [],
            "gate_links": [],
        }
        canvas.layout = {
            "bit_arrange": layouts.qreg_creg_ascending,
            "time_axis_map": layouts.time_map_in_dt,
        }

        canvas.load_program(self.circ, target=self.target)
        canvas.update()
        drawings_tested = list(canvas.collections)

        self.assertEqual(len(drawings_tested), 8)

        ref_coord = {
            self.circ.qregs[0][0]: -1.0,
            self.circ.qregs[0][1]: -2.0,
            self.circ.qregs[0][2]: -3.0,
            self.circ.qregs[0][3]: -4.0,
        }
        self.assertDictEqual(canvas.assigned_coordinates, ref_coord)

    def test_gate_link_overlap(self):
        """Test shifting gate link overlap."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.formatter.update(
            {
                "margin.link_interval_percent": 0.01,
                "margin.left_percent": 0,
                "margin.right_percent": 0,
            }
        )
        canvas.generator = {
            "gates": [],
            "bits": [],
            "barriers": [],
            "gate_links": [generators.gen_gate_link],
        }
        canvas.layout = {
            "bit_arrange": layouts.qreg_creg_ascending,
            "time_axis_map": layouts.time_map_in_dt,
        }

        canvas.load_program(self.circ, self.target)
        canvas.update()
        drawings_tested = list(canvas.collections)

        self.assertEqual(len(drawings_tested), 2)

        self.assertListEqual(drawings_tested[0][1].xvals, [706.0])
        self.assertListEqual(drawings_tested[1][1].xvals, [694.0])

        ref_keys = list(canvas._collections.keys())
        self.assertEqual(drawings_tested[0][0], ref_keys[0])
        self.assertEqual(drawings_tested[1][0], ref_keys[1])

    def test_object_outside_xlimit(self):
        """Test eliminating drawings outside the horizontal limit."""
        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.generator = {
            "gates": [generators.gen_sched_gate],
            "bits": [generators.gen_bit_name, generators.gen_timeslot],
            "barriers": [],
            "gate_links": [],
        }
        canvas.layout = {
            "bit_arrange": layouts.qreg_creg_ascending,
            "time_axis_map": layouts.time_map_in_dt,
        }
        canvas.load_program(self.circ, target=self.target)
        canvas.set_time_range(t_start=400, t_end=600)
        canvas.update()
        drawings_tested = list(canvas.collections)

        self.assertEqual(len(drawings_tested), 12)

    def test_non_transpiled_delay_circuit(self):
        """Test non-transpiled circuit containing instruction which is trivial on duration."""
        circ = QuantumCircuit(1)
        circ.delay(10, 0)

        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.generator = {
            "gates": [generators.gen_sched_gate],
            "bits": [],
            "barriers": [],
            "gate_links": [],
        }

        with self.assertWarns(DeprecationWarning):
            canvas.load_program(circ, target=self.target)
        self.assertEqual(len(canvas._collections), 4)

    def test_multi_measurement_with_clbit_not_shown(self):
        """Test generating bit link drawings of measurements when clbits is disabled."""
        circ = QuantumCircuit(2, 2)
        circ.measure(0, 0)
        circ.measure(1, 1)

        target = Target(num_qubits=2, dt=1e-7)
        target.add_instruction(
            Measure(),
            {
                (0,): InstructionProperties(duration=2000 * 1e-7),
                (1,): InstructionProperties(duration=2000 * 1e-7),
            },
        )

        circ = transpile(
            circ,
            scheduling_method="alap",
            dt=1e-7,
            target=target,
            optimization_level=0,
        )

        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.formatter.update({"control.show_clbits": False})
        canvas.layout = {
            "bit_arrange": layouts.qreg_creg_ascending,
            "time_axis_map": layouts.time_map_in_dt,
        }
        canvas.generator = {
            "gates": [],
            "bits": [],
            "barriers": [],
            "gate_links": [generators.gen_gate_link],
        }

        canvas.load_program(circ, target)
        canvas.update()
        self.assertEqual(len(canvas._output_dataset), 0)

    def test_multi_measurement_with_clbit_shown(self):
        """Test generating bit link drawings of measurements when clbits is enabled."""
        circ = QuantumCircuit(2, 2)
        circ.measure(0, 0)
        circ.measure(1, 1)

        target = Target(num_qubits=2, dt=1e-7)
        target.add_instruction(
            Measure(),
            {
                (0,): InstructionProperties(duration=2000 * 1e-7),
                (1,): InstructionProperties(duration=2000 * 1e-7),
            },
        )

        circ = transpile(
            circ, scheduling_method="alap", dt=1e-7, optimization_level=0, target=target
        )

        canvas = core.DrawerCanvas(stylesheet=self.style)
        canvas.formatter.update({"control.show_clbits": True})
        canvas.layout = {
            "bit_arrange": layouts.qreg_creg_ascending,
            "time_axis_map": layouts.time_map_in_dt,
        }
        canvas.generator = {
            "gates": [],
            "bits": [],
            "barriers": [],
            "gate_links": [generators.gen_gate_link],
        }

        canvas.load_program(circ, target)
        canvas.update()
        self.assertEqual(len(canvas._output_dataset), 2)
