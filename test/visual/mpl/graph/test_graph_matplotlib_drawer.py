# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for graph MPL drawer"""

import unittest
import os
from contextlib import contextmanager
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.utils import optionals
from qiskit.visualization.state_visualization import state_drawer
from qiskit.visualization.counts_visualization import plot_histogram
from qiskit.visualization.gate_map import plot_gate_map, plot_coupling_map
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Statevector
from test import QiskitTestCase
from test.visual import VisualTestUtilities
from test.python.legacy_cmaps import (
    YORKTOWN_CMAP,
    LAGOS_CMAP,
    RUESCHLIKON_CMAP,
    MUMBAI_CMAP,
    MANHATTAN_CMAP,
)

if optionals.HAS_MATPLOTLIB:
    from matplotlib.pyplot import close as mpl_close
else:
    raise ImportError('Must have Matplotlib installed. To install, run "pip install matplotlib".')

BASE_DIR = Path(__file__).parent
RESULT_DIR = Path(BASE_DIR) / "graph_results"
TEST_REFERENCE_DIR = Path(BASE_DIR) / "references"
FAILURE_DIFF_DIR = Path(BASE_DIR).parent / "visual_test_failures"
FAILURE_PREFIX = "graph_failure_"


@contextmanager
def cwd(path):
    """A context manager to run in a particular path"""
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


class TestGraphMatplotlibDrawer(QiskitTestCase):
    """Graph MPL visualization"""

    def setUp(self):
        super().setUp()
        self.graph_state_drawer = VisualTestUtilities.save_data_wrap(
            state_drawer, str(self), RESULT_DIR
        )
        self.graph_count_drawer = VisualTestUtilities.save_data_wrap(
            plot_histogram, str(self), RESULT_DIR
        )
        self.graph_plot_gate_map = VisualTestUtilities.save_data_wrap(
            plot_gate_map, str(self), RESULT_DIR
        )
        self.graph_plot_coupling_map = VisualTestUtilities.save_data_wrap(
            plot_coupling_map, str(self), RESULT_DIR
        )

        if not os.path.exists(FAILURE_DIFF_DIR):
            os.makedirs(FAILURE_DIFF_DIR)

        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

    def tearDown(self):
        super().tearDown()
        mpl_close("all")

    @staticmethod
    def _image_path(image_name):
        return os.path.join(RESULT_DIR, image_name)

    @staticmethod
    def _reference_path(image_name):
        return os.path.join(TEST_REFERENCE_DIR, image_name)

    def test_plot_bloch_multivector(self):
        """test bloch sphere
        See https://github.com/Qiskit/qiskit-terra/issues/6397.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "bloch_multivector.png"
        self.graph_state_drawer(state=state, output="bloch", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.98, msg=fname)

    def test_plot_state_hinton(self):
        """test plot_state_hinton"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "hinton.png"
        self.graph_state_drawer(state=state, output="hinton", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_state_qsphere(self):
        """test for plot_state_qsphere"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "qsphere.png"
        self.graph_state_drawer(state=state, output="qsphere", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_entangled_state_qsphere(self):
        """test for plot_state_qsphere"""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.z(0)
        circuit.h(0)
        circuit.cx(0, 1)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "entangled_qsphere.png"
        self.graph_state_drawer(state=state, output="qsphere", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_state_city(self):
        """test for plot_state_city"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "state_city.png"
        self.graph_state_drawer(state=state, output="city", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_state_paulivec(self):
        """test for plot_state_paulivec"""
        circuit = QuantumCircuit(1)
        circuit.x(0)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "paulivec.png"
        self.graph_state_drawer(state=state, output="paulivec", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram(self):
        """for testing the plot_histogram"""
        # specifying counts because we do not want oscillation of
        # result until a changes is made to plot_histogram

        counts = {"11": 500, "00": 500}

        fname = "histogram.png"
        self.graph_count_drawer(counts, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_with_rest(self):
        """test plot_histogram with 2 datasets and number_to_keep"""
        data = [{"00": 3, "01": 5, "10": 6, "11": 12}]

        fname = "histogram_with_rest.png"
        self.graph_count_drawer(data, number_to_keep=2, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_2_sets_with_rest(self):
        """test plot_histogram with 2 datasets and number_to_keep"""
        data = [
            {"00": 3, "01": 5, "10": 6, "11": 12},
            {"00": 5, "01": 7, "10": 6, "11": 12},
        ]

        fname = "histogram_2_sets_with_rest.png"
        self.graph_count_drawer(data, number_to_keep=2, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_color(self):
        """Test histogram with single color"""

        counts = {"00": 500, "11": 500}

        fname = "histogram_color.png"
        self.graph_count_drawer(data=counts, color="#204940", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_multiple_colors(self):
        """Test histogram with multiple custom colors"""

        counts = [
            {"00": 10, "01": 15, "10": 20, "11": 25},
            {"00": 25, "01": 20, "10": 15, "11": 10},
        ]

        fname = "histogram_multiple_colors.png"
        self.graph_count_drawer(
            data=counts,
            color=["#204940", "#c26219"],
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_hamming(self):
        """Test histogram with hamming distance"""

        counts = {"101": 500, "010": 500, "001": 500, "100": 500}

        fname = "histogram_hamming.png"
        self.graph_count_drawer(data=counts, sort="hamming", target_string="101", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_value_sort(self):
        """Test histogram with sorting by value"""

        counts = {"101": 300, "010": 240, "001": 80, "100": 150, "110": 160, "000": 280, "111": 60}

        fname = "histogram_value_sort.png"
        self.graph_count_drawer(data=counts, sort="value", target_string="000", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_desc_value_sort(self):
        """Test histogram with sorting by descending value"""

        counts = {"101": 150, "010": 50, "001": 180, "100": 10, "110": 190, "000": 80, "111": 260}

        fname = "histogram_desc_value_sort.png"
        self.graph_count_drawer(
            data=counts,
            sort="value_desc",
            target_string="000",
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_legend(self):
        """Test histogram with legend"""

        counts = [{"0": 50, "1": 30}, {"0": 30, "1": 40}]

        fname = "histogram_legend.png"
        self.graph_count_drawer(
            data=counts,
            legend=["first", "second"],
            filename=fname,
            figsize=(15, 5),
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_histogram_title(self):
        """Test histogram with title"""

        counts = [{"0": 50, "1": 30}, {"0": 30, "1": 40}]

        fname = "histogram_title.png"
        self.graph_count_drawer(data=counts, title="My Histogram", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_1_qubit_gate_map(self):
        """Test plot_gate_map using 1 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=1, basis_gates=["id", "rz", "sx", "x"])

        fname = "1_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_5_qubit_gate_map(self):
        """Test plot_gate_map using 5 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP)

        fname = "5_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_7_qubit_gate_map(self):
        """Test plot_gate_map using 7 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP)

        fname = "7_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_plot_16_qubit_gate_map(self):
        """Test plot_gate_map using 16 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=16, coupling_map=RUESCHLIKON_CMAP)

        fname = "16_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_plot_27_qubit_gate_map(self):
        """Test plot_gate_map using 27 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=27, coupling_map=MUMBAI_CMAP)

        fname = "27_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_65_qubit_gate_map(self):
        """test for plot_gate_map using 65 qubit backend"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=65, coupling_map=MANHATTAN_CMAP)

        fname = "65_qubit_gate_map.png"
        self.graph_plot_gate_map(backend=backend, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_figsize(self):
        """Test figsize parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP)

        fname = "figsize.png"
        self.graph_plot_gate_map(backend=backend, figsize=(10, 10), filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_qubit_size(self):
        """Test qubit_size parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=5, coupling_map=YORKTOWN_CMAP)

        fname = "qubit_size.png"
        self.graph_plot_gate_map(backend=backend, qubit_size=38, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_qubit_color(self):
        """Test qubit_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP)

        fname = "qubit_color.png"
        self.graph_plot_gate_map(backend=backend, qubit_color=["#ff0000"] * 7, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99, msg=fname)

    def test_qubit_labels(self):
        """Test qubit_labels parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP)

        fname = "qubit_labels.png"
        self.graph_plot_gate_map(
            backend=backend, qubit_labels=list(range(10, 17, 1)), filename=fname
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_line_color(self):
        """Test line_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=65, coupling_map=MANHATTAN_CMAP)

        fname = "line_color.png"
        self.graph_plot_gate_map(backend=backend, line_color=["#00ff00"] * 144, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_font_color(self):
        """Test font_color parameter of plot_gate_map"""
        # getting the mock backend from FakeProvider

        backend = GenericBackendV2(num_qubits=65, coupling_map=MANHATTAN_CMAP)

        fname = "font_color.png"
        self.graph_plot_gate_map(backend=backend, font_color="#ff00ff", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_coupling_map(self):
        """Test plot_coupling_map"""

        num_qubits = 5
        qubit_coordinates = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]
        coupling_map = [[1, 0], [1, 2], [1, 3], [3, 4]]

        fname = "coupling_map.png"
        self.graph_plot_coupling_map(
            num_qubits=num_qubits,
            qubit_coordinates=qubit_coordinates,
            coupling_map=coupling_map,
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.99)

    def test_plot_bloch_multivector_figsize_improvements(self):
        """test bloch sphere figsize, font_size, title_font_size and title_pad
        See https://github.com/Qiskit/qiskit-terra/issues/7263
        and https://github.com/Qiskit/qiskit-terra/pull/7264.
        """
        circuit = QuantumCircuit(3)
        circuit.h(1)
        circuit.sxdg(2)

        # getting the state using quantum_info
        state = Statevector(circuit)

        fname = "bloch_multivector_figsize_improvements.png"
        self.graph_state_drawer(
            state=state,
            output="bloch",
            figsize=(3, 2),
            font_size=10,
            title="|0+R> state",
            title_font_size=14,
            title_pad=8,
            filename=fname,
        )
        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, 0.98, msg=fname)


if __name__ == "__main__":
    unittest.main(verbosity=1)
