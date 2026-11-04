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


"""Tests for generator of timeline drawer."""

import qiskit
from qiskit.visualization.timeline import generators, types, stylesheet
from qiskit.circuit import library, Delay
from test import QiskitTestCase


class TestGates(QiskitTestCase):
    """Tests for generator.gates."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.qubit = next(iter(qiskit.QuantumRegister(1, name="foo")))

        self.u1 = types.ScheduledGate(
            t0=100, operand=library.U1Gate(0), duration=0, bits=[self.qubit], bit_position=0
        )

        self.u3 = types.ScheduledGate(
            t0=100, operand=library.U3Gate(0, 0, 0), duration=20, bits=[self.qubit], bit_position=0
        )

        self.delay = types.ScheduledGate(
            t0=100, operand=Delay(20), duration=20, bits=[self.qubit], bit_position=0
        )

        style = stylesheet.QiskitTimelineStyle()
        self.formatter = style.formatter

    def test_gen_sched_gate_with_finite_duration(self):
        """Test test_gen_sched_gate generator with finite duration gate."""
        drawing_obj = generators.gen_sched_gate(self.u3, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.BoxType.SCHED_GATE.value))
        self.assertListEqual(list(drawing_obj.xvals), [100, 120])
        self.assertListEqual(
            list(drawing_obj.yvals),
            [-0.5 * self.formatter["box_height.gate"], 0.5 * self.formatter["box_height.gate"]],
        )
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        ref_meta = {
            "name": "u3",
            "label": "n/a",
            "bits": [self.qubit],
            "t0": 100,
            "duration": 20,
            "parameters": "0, 0, 0",
        }
        self.assertDictEqual(ref_meta, drawing_obj.meta)

        ref_styles = {
            "zorder": self.formatter["layer.gate"],
            "facecolor": self.formatter["color.gates"]["u3"],
            "alpha": self.formatter["alpha.gate"],
            "linewidth": self.formatter["line_width.gate"],
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

    def test_gen_sched_gate_with_zero_duration(self):
        """Test test_gen_sched_gate generator with zero duration gate."""
        drawing_obj = generators.gen_sched_gate(self.u1, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.SymbolType.FRAME.value))
        self.assertListEqual(list(drawing_obj.xvals), [100])
        self.assertListEqual(list(drawing_obj.yvals), [0])
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        self.assertEqual(drawing_obj.text, self.formatter["unicode_symbol.frame_change"])
        self.assertEqual(drawing_obj.latex, self.formatter["latex_symbol.frame_change"])

        ref_styles = {
            "zorder": self.formatter["layer.frame_change"],
            "color": self.formatter["color.gates"]["u1"],
            "size": self.formatter["text_size.frame_change"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

    def test_gen_sched_gate_with_delay(self):
        """Test test_gen_sched_gate generator with delay."""
        drawing_obj = generators.gen_sched_gate(self.delay, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.BoxType.DELAY.value))

    def test_gen_full_gate_name_with_finite_duration(self):
        """Test gen_full_gate_name generator with finite duration gate."""
        drawing_obj = generators.gen_full_gate_name(self.u3, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LabelType.GATE_NAME.value))
        self.assertListEqual(list(drawing_obj.xvals), [110.0])
        self.assertListEqual(list(drawing_obj.yvals), [0.0])
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        self.assertEqual(drawing_obj.text, "u3(0.00, 0.00, 0.00)[20]")
        ref_latex = f"{self.formatter['latex_symbol.gates']['u3']}(0.00, 0.00, 0.00)[20]"
        self.assertEqual(drawing_obj.latex, ref_latex)

        ref_styles = {
            "zorder": self.formatter["layer.gate_name"],
            "color": self.formatter["color.gate_name"],
            "size": self.formatter["text_size.gate_name"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

    def test_gen_full_gate_name_with_zero_duration(self):
        """Test gen_full_gate_name generator with zero duration gate."""
        drawing_obj = generators.gen_full_gate_name(self.u1, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LabelType.GATE_NAME.value))
        self.assertListEqual(list(drawing_obj.xvals), [100.0])
        self.assertListEqual(list(drawing_obj.yvals), [self.formatter["label_offset.frame_change"]])
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        self.assertEqual(drawing_obj.text, "u1(0.00)")
        ref_latex = f"{self.formatter['latex_symbol.gates']['u1']}(0.00)"
        self.assertEqual(drawing_obj.latex, ref_latex)

        ref_styles = {
            "zorder": self.formatter["layer.gate_name"],
            "color": self.formatter["color.gate_name"],
            "size": self.formatter["text_size.gate_name"],
            "va": "bottom",
            "ha": "center",
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

    def test_gen_full_gate_name_with_delay(self):
        """Test gen_full_gate_name generator with delay."""
        drawing_obj = generators.gen_full_gate_name(self.delay, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LabelType.DELAY.value))

    def test_gen_short_gate_name_with_finite_duration(self):
        """Test gen_short_gate_name generator with finite duration gate."""
        drawing_obj = generators.gen_short_gate_name(self.u3, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LabelType.GATE_NAME.value))
        self.assertListEqual(list(drawing_obj.xvals), [110.0])
        self.assertListEqual(list(drawing_obj.yvals), [0.0])
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        self.assertEqual(drawing_obj.text, "u3")
        ref_latex = f"{self.formatter['latex_symbol.gates']['u3']}"
        self.assertEqual(drawing_obj.latex, ref_latex)

        ref_styles = {
            "zorder": self.formatter["layer.gate_name"],
            "color": self.formatter["color.gate_name"],
            "size": self.formatter["text_size.gate_name"],
            "va": "center",
            "ha": "center",
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

    def test_gen_short_gate_name_with_zero_duration(self):
        """Test gen_short_gate_name generator with zero duration gate."""
        drawing_obj = generators.gen_short_gate_name(self.u1, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LabelType.GATE_NAME.value))
        self.assertListEqual(list(drawing_obj.xvals), [100.0])
        self.assertListEqual(list(drawing_obj.yvals), [self.formatter["label_offset.frame_change"]])
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        self.assertEqual(drawing_obj.text, "u1")
        ref_latex = f"{self.formatter['latex_symbol.gates']['u1']}"
        self.assertEqual(drawing_obj.latex, ref_latex)

        ref_styles = {
            "zorder": self.formatter["layer.gate_name"],
            "color": self.formatter["color.gate_name"],
            "size": self.formatter["text_size.gate_name"],
            "va": "bottom",
            "ha": "center",
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

    def test_gen_short_gate_name_with_delay(self):
        """Test gen_short_gate_name generator with delay."""
        drawing_obj = generators.gen_short_gate_name(self.delay, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LabelType.DELAY.value))


class TestTimeslot(QiskitTestCase):
    """Tests for generator.bits."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.program = qiskit.QuantumCircuit(qiskit.QuantumRegister(1, "bar"))
        self.program._op_start_times = []
        self.qubit = self.program.qubits[0]

        style = stylesheet.QiskitTimelineStyle()
        self.formatter = style.formatter

    def test_gen_timeslot(self):
        """Test gen_timeslot generator."""
        drawing_obj = generators.gen_timeslot(self.qubit, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.BoxType.TIMELINE.value))
        self.assertListEqual(
            list(drawing_obj.xvals), [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        )
        self.assertListEqual(
            list(drawing_obj.yvals),
            [
                -0.5 * self.formatter["box_height.timeslot"],
                0.5 * self.formatter["box_height.timeslot"],
            ],
        )
        self.assertListEqual(drawing_obj.bits, [self.qubit])

        ref_styles = {
            "zorder": self.formatter["layer.timeslot"],
            "alpha": self.formatter["alpha.timeslot"],
            "linewidth": self.formatter["line_width.timeslot"],
            "facecolor": self.formatter["color.timeslot"],
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)

    def test_gen_bit_name(self):
        """Test gen_bit_name generator."""
        with self.assertWarnsRegex(UserWarning, "bits cannot be accurately named"):
            generators.gen_bit_name(self.qubit, self.formatter)

        drawing_obj = generators.gen_bit_name(self.qubit, self.formatter, program=self.program)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LabelType.BIT_NAME.value))
        self.assertListEqual(list(drawing_obj.xvals), [types.AbstractCoordinate.LEFT])
        self.assertListEqual(list(drawing_obj.yvals), [0])
        self.assertListEqual(drawing_obj.bits, [self.qubit])
        self.assertEqual(drawing_obj.text, "bar")
        # pylint: disable-next=consider-using-f-string
        ref_latex = r"{{\rm {register}}}_{{{index}}}".format(register="q", index="0")
        self.assertEqual(drawing_obj.latex, ref_latex)

        ref_styles = {
            "zorder": self.formatter["layer.bit_name"],
            "color": self.formatter["color.bit_name"],
            "size": self.formatter["text_size.bit_name"],
            "va": "center",
            "ha": "right",
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)


class TestBarrier(QiskitTestCase):
    """Tests for generator.barriers."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.qubits = list(qiskit.QuantumRegister(3))
        self.barrier = types.Barrier(t0=100, bits=self.qubits, bit_position=1)

        style = stylesheet.QiskitTimelineStyle()
        self.formatter = style.formatter

    def test_gen_barrier(self):
        """Test gen_barrier generator."""
        drawing_obj = generators.gen_barrier(self.barrier, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LineType.BARRIER.value))
        self.assertListEqual(list(drawing_obj.xvals), [100, 100])
        self.assertListEqual(list(drawing_obj.yvals), [-0.5, 0.5])
        self.assertListEqual(drawing_obj.bits, [self.qubits[1]])

        ref_styles = {
            "alpha": self.formatter["alpha.barrier"],
            "zorder": self.formatter["layer.barrier"],
            "linewidth": self.formatter["line_width.barrier"],
            "linestyle": self.formatter["line_style.barrier"],
            "color": self.formatter["color.barrier"],
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)


class TestGateLink(QiskitTestCase):
    """Tests for generator.gate_links."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.qubits = list(qiskit.QuantumRegister(2))
        self.gate_link = types.GateLink(t0=100, opname="cx", bits=self.qubits)

        style = stylesheet.QiskitTimelineStyle()
        self.formatter = style.formatter

    def gen_bit_link(self):
        """Test gen_bit_link generator."""
        drawing_obj = generators.gen_gate_link(self.gate_link, self.formatter)[0]

        self.assertEqual(drawing_obj.data_type, str(types.LineType.GATE_LINK.value))
        self.assertListEqual(list(drawing_obj.xvals), [100])
        self.assertListEqual(list(drawing_obj.yvals), [0])
        self.assertListEqual(drawing_obj.bits, self.qubits)

        ref_styles = {
            "alpha": self.formatter["alpha.bit_link"],
            "zorder": self.formatter["layer.bit_link"],
            "linewidth": self.formatter["line_width.bit_link"],
            "linestyle": self.formatter["line_style.bit_link"],
            "color": self.formatter["color.gates"]["cx"],
        }
        self.assertDictEqual(ref_styles, drawing_obj.styles)
