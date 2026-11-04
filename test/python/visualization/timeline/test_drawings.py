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

"""Tests for drawing of timeline drawer."""

import numpy as np

import qiskit
from qiskit.visualization.timeline import drawings, types
from test import QiskitTestCase


class TestDrawingObjects(QiskitTestCase):
    """Tests for drawings."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        # bits
        self.qubits = list(qiskit.QuantumRegister(2))

        # metadata
        self.meta1 = {"val1": 0, "val2": 1}
        self.meta2 = {"val1": 2, "val2": 3}

        # style data
        self.style1 = {"property1": 0, "property2": 1}
        self.style2 = {"property1": 2, "property2": 3}

    def test_line_data_equivalent(self):
        """Test LineData equivalent check."""
        xs = list(np.arange(10))
        ys = list(np.ones(10))

        obj1 = drawings.LineData(
            data_type=types.LineType.BARRIER,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta1,
            styles=self.style1,
        )

        obj2 = drawings.LineData(
            data_type=types.LineType.BARRIER,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta2,
            styles=self.style2,
        )

        self.assertEqual(obj1, obj2)

    def test_line_data_equivalent_abstract_coord(self):
        """Test LineData equivalent check with abstract coordinate."""
        xs = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.BOTTOM, types.AbstractCoordinate.TOP]

        obj1 = drawings.LineData(
            data_type=types.LineType.BARRIER,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta1,
            styles=self.style1,
        )

        obj2 = drawings.LineData(
            data_type=types.LineType.BARRIER,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta2,
            styles=self.style2,
        )

        self.assertEqual(obj1, obj2)

    def test_box_data_equivalent(self):
        """Test BoxData equivalent check."""
        xs = [0, 1]
        ys = [0, 1]

        obj1 = drawings.BoxData(
            data_type=types.BoxType.SCHED_GATE,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta1,
            styles=self.style1,
        )

        obj2 = drawings.BoxData(
            data_type=types.BoxType.SCHED_GATE,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta2,
            styles=self.style2,
        )

        self.assertEqual(obj1, obj2)

    def test_box_data_equivalent_abstract_coord(self):
        """Test BoxData equivalent check with abstract coordinate."""
        xs = [types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT]
        ys = [types.AbstractCoordinate.BOTTOM, types.AbstractCoordinate.TOP]

        obj1 = drawings.BoxData(
            data_type=types.BoxType.SCHED_GATE,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta1,
            styles=self.style1,
        )

        obj2 = drawings.BoxData(
            data_type=types.BoxType.SCHED_GATE,
            bit=self.qubits[0],
            xvals=xs,
            yvals=ys,
            meta=self.meta2,
            styles=self.style2,
        )

        self.assertEqual(obj1, obj2)

    def test_text_data_equivalent(self):
        """Test TextData equivalent check."""
        obj1 = drawings.TextData(
            data_type=types.LabelType.GATE_NAME,
            bit=self.qubits[0],
            xval=0,
            yval=0,
            text="test",
            latex="test",
            meta=self.meta1,
            styles=self.style1,
        )

        obj2 = drawings.TextData(
            data_type=types.LabelType.GATE_NAME,
            bit=self.qubits[0],
            xval=0,
            yval=0,
            text="test",
            latex="test",
            meta=self.meta2,
            styles=self.style2,
        )

        self.assertEqual(obj1, obj2)

    def test_text_data_equivalent_abstract_coord(self):
        """Test TextData equivalent check with abstract coordinate."""
        obj1 = drawings.TextData(
            data_type=types.LabelType.GATE_NAME,
            bit=self.qubits[0],
            xval=types.AbstractCoordinate.LEFT,
            yval=types.AbstractCoordinate.BOTTOM,
            text="test",
            latex="test",
            meta=self.meta1,
            styles=self.style1,
        )

        obj2 = drawings.TextData(
            data_type=types.LabelType.GATE_NAME,
            bit=self.qubits[0],
            xval=types.AbstractCoordinate.LEFT,
            yval=types.AbstractCoordinate.BOTTOM,
            text="test",
            latex="test",
            meta=self.meta2,
            styles=self.style2,
        )

        self.assertEqual(obj1, obj2)

    def test_bit_link_data_equivalent(self):
        """Test BitLinkData equivalent check."""
        obj1 = drawings.GateLinkData(
            bits=[self.qubits[0], self.qubits[1]], xval=0, styles=self.style1
        )

        obj2 = drawings.GateLinkData(
            bits=[self.qubits[0], self.qubits[1]], xval=0, styles=self.style2
        )

        self.assertEqual(obj1, obj2)

    def test_bit_link_data_equivalent_abstract_coord(self):
        """Test BitLinkData equivalent check with abstract coordinate."""
        obj1 = drawings.GateLinkData(
            bits=[self.qubits[0], self.qubits[1]],
            xval=types.AbstractCoordinate.LEFT,
            styles=self.style1,
        )

        obj2 = drawings.GateLinkData(
            bits=[self.qubits[0], self.qubits[1]],
            xval=types.AbstractCoordinate.LEFT,
            styles=self.style2,
        )

        self.assertEqual(obj1, obj2)
