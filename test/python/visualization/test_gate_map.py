# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A test for visualizing device coupling maps"""
import unittest

from io import BytesIO
from ddt import ddt, data
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.visualization import (
    plot_gate_map,
    plot_coupling_map,
    plot_circuit_layout,
    plot_error_map,
)
from qiskit.utils import optionals
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.layout import Layout, TranspileLayout
from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase
from ..legacy_cmaps import KYOTO_CMAP, MUMBAI_CMAP, YORKTOWN_CMAP, ALMADEN_CMAP, LAGOS_CMAP

if optionals.HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt
if optionals.HAS_PIL:
    from PIL import Image


@ddt
@unittest.skipUnless(optionals.HAS_MATPLOTLIB, "matplotlib not available.")
@unittest.skipUnless(optionals.HAS_PIL, "PIL not available")
@unittest.skipUnless(optionals.HAS_SEABORN, "seaborn not available")
class TestGateMap(QiskitVisualizationTestCase):
    """visual tests for plot_gate_map"""

    backend_configs = {
        "yorktown": (5, YORKTOWN_CMAP),
        "almaden": (20, ALMADEN_CMAP),
        "lagos": (7, LAGOS_CMAP),
    }

    @data(*backend_configs)
    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    def test_plot_gate_map(self, backend_name):
        """tests plotting of gate map of a device (20 qubit, 7 qubit, and 5 qubit)"""
        n, coupling_map = self.backend_configs[backend_name]
        backend = GenericBackendV2(num_qubits=n, coupling_map=coupling_map, seed=0)
        n = backend.num_qubits
        img_ref = path_to_diagram_reference(str(n) + "bit_quantum_computer.png")
        fig = plot_gate_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)

    @data(*backend_configs)
    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    def test_plot_circuit_layout(self, backend_name):
        """tests plot_circuit_layout for each device"""
        n, coupling_map = self.backend_configs[backend_name]
        backend = GenericBackendV2(num_qubits=n, coupling_map=coupling_map, seed=0)
        layout_length = int(backend.num_qubits / 2)
        qr = QuantumRegister(layout_length, "qr")
        circuit = QuantumCircuit(qr)
        circuit._layout = TranspileLayout(
            Layout({qr[i]: i * 2 for i in range(layout_length)}),
            {qubit: index for index, qubit in enumerate(circuit.qubits)},
        )
        circuit._layout.initial_layout.add_register(qr)
        n = backend.num_qubits
        img_ref = path_to_diagram_reference(str(n) + "_plot_circuit_layout.png")
        fig = plot_circuit_layout(circuit, backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.1)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    def test_plot_gate_map_no_backend(self):
        """tests plotting of gate map without a device"""
        n_qubits = 8
        coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
        qubit_coordinates = [[0, 1], [1, 1], [1, 0], [1, 2], [2, 0], [2, 2], [2, 1], [3, 1]]
        img_ref = path_to_diagram_reference(str(n_qubits) + "qubits.png")
        fig = plot_coupling_map(
            num_qubits=n_qubits, qubit_coordinates=qubit_coordinates, coupling_map=coupling_map
        )
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(optionals.HAS_SEABORN, "Seaborn not installed")
    def test_plot_error_map_backend_v2(self):
        """Test plotting error map with fake backend v2."""
        coupling_map = MUMBAI_CMAP
        backend = GenericBackendV2(
            num_qubits=27,
            coupling_map=coupling_map,
        )
        img_ref = path_to_diagram_reference("fake_27_q_v2_error.png")
        fig = plot_error_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.05)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(optionals.HAS_SEABORN, "Seaborn not installed")
    def test_plot_error_map_over_100_qubit(self):
        """Test plotting error map with large fake backend."""
        coupling_map = KYOTO_CMAP
        backend = GenericBackendV2(num_qubits=127, coupling_map=coupling_map, seed=42)
        img_ref = path_to_diagram_reference("fake_127_q_error.png")
        fig = plot_error_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.05)
        plt.close(fig)

    @unittest.skipIf(not optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(optionals.HAS_SEABORN, "Seaborn not installed")
    def test_plot_error_map_over_100_qubit_backend_v2(self):
        """Test plotting error map with large fake backendv2."""
        coupling_map = [
            [0, 1],
            [0, 14],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [4, 5],
            [4, 15],
            [5, 4],
            [5, 6],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [8, 7],
            [8, 16],
            [9, 10],
            [10, 9],
            [10, 11],
            [11, 10],
            [11, 12],
            [12, 11],
            [12, 13],
            [12, 17],
            [13, 12],
            [14, 0],
            [14, 18],
            [15, 4],
            [15, 22],
            [16, 8],
            [16, 26],
            [17, 12],
            [17, 30],
            [18, 14],
            [18, 19],
            [19, 18],
            [19, 20],
            [20, 19],
            [20, 21],
            [20, 33],
            [21, 20],
            [21, 22],
            [22, 15],
            [22, 21],
            [22, 23],
            [23, 22],
            [23, 24],
            [24, 23],
            [24, 25],
            [24, 34],
            [25, 24],
            [25, 26],
            [26, 16],
            [26, 25],
            [26, 27],
            [27, 26],
            [27, 28],
            [28, 27],
            [28, 29],
            [28, 35],
            [29, 28],
            [29, 30],
            [30, 17],
            [30, 29],
            [30, 31],
            [31, 30],
            [31, 32],
            [32, 31],
            [32, 36],
            [33, 20],
            [33, 39],
            [34, 24],
            [34, 43],
            [35, 28],
            [35, 47],
            [36, 32],
            [36, 51],
            [37, 38],
            [37, 52],
            [38, 37],
            [38, 39],
            [39, 33],
            [39, 38],
            [39, 40],
            [40, 39],
            [40, 41],
            [41, 40],
            [41, 42],
            [41, 53],
            [42, 41],
            [42, 43],
            [43, 34],
            [43, 42],
            [43, 44],
            [44, 43],
            [44, 45],
            [45, 44],
            [45, 46],
            [45, 54],
            [46, 45],
            [46, 47],
            [47, 35],
            [47, 46],
            [47, 48],
            [48, 47],
            [48, 49],
            [49, 48],
            [49, 50],
            [49, 55],
            [50, 49],
            [50, 51],
            [51, 36],
            [51, 50],
            [52, 37],
            [52, 56],
            [53, 41],
            [53, 60],
            [54, 45],
            [54, 64],
            [55, 49],
            [55, 68],
            [56, 52],
            [56, 57],
            [57, 56],
            [57, 58],
            [58, 57],
            [58, 59],
            [58, 71],
            [59, 58],
            [59, 60],
            [60, 53],
            [60, 59],
            [60, 61],
            [61, 60],
            [61, 62],
            [62, 61],
            [62, 63],
            [62, 72],
            [63, 62],
            [63, 64],
            [64, 54],
            [64, 63],
            [64, 65],
            [65, 64],
            [65, 66],
            [66, 65],
            [66, 67],
            [66, 73],
            [67, 66],
            [67, 68],
            [68, 55],
            [68, 67],
            [68, 69],
            [69, 68],
            [69, 70],
            [70, 69],
            [70, 74],
            [71, 58],
            [71, 77],
            [72, 62],
            [72, 81],
            [73, 66],
            [73, 85],
            [74, 70],
            [74, 89],
            [75, 76],
            [75, 90],
            [76, 75],
            [76, 77],
            [77, 71],
            [77, 76],
            [77, 78],
            [78, 77],
            [78, 79],
            [79, 78],
            [79, 80],
            [79, 91],
            [80, 79],
            [80, 81],
            [81, 72],
            [81, 80],
            [81, 82],
            [82, 81],
            [82, 83],
            [83, 82],
            [83, 84],
            [83, 92],
            [84, 83],
            [84, 85],
            [85, 73],
            [85, 84],
            [85, 86],
            [86, 85],
            [86, 87],
            [87, 86],
            [87, 88],
            [87, 93],
            [88, 87],
            [88, 89],
            [89, 74],
            [89, 88],
            [90, 75],
            [90, 94],
            [91, 79],
            [91, 98],
            [92, 83],
            [92, 102],
            [93, 87],
            [93, 106],
            [94, 90],
            [94, 95],
            [95, 94],
            [95, 96],
            [96, 95],
            [96, 97],
            [96, 109],
            [97, 96],
            [97, 98],
            [98, 91],
            [98, 97],
            [98, 99],
            [99, 98],
            [99, 100],
            [100, 99],
            [100, 101],
            [100, 110],
            [101, 100],
            [101, 102],
            [102, 92],
            [102, 101],
            [102, 103],
            [103, 102],
            [103, 104],
            [104, 103],
            [104, 105],
            [104, 111],
            [105, 104],
            [105, 106],
            [106, 93],
            [106, 105],
            [106, 107],
            [107, 106],
            [107, 108],
            [108, 107],
            [108, 112],
            [109, 96],
            [110, 100],
            [110, 118],
            [111, 104],
            [111, 122],
            [112, 108],
            [112, 126],
            [113, 114],
            [114, 113],
            [114, 115],
            [115, 114],
            [115, 116],
            [116, 115],
            [116, 117],
            [117, 116],
            [117, 118],
            [118, 110],
            [118, 117],
            [118, 119],
            [119, 118],
            [119, 120],
            [120, 119],
            [120, 121],
            [121, 120],
            [121, 122],
            [122, 111],
            [122, 121],
            [122, 123],
            [123, 122],
            [123, 124],
            [124, 123],
            [124, 125],
            [125, 124],
            [125, 126],
            [126, 112],
            [126, 125],
        ]
        backend = GenericBackendV2(num_qubits=127, coupling_map=coupling_map, seed=42)
        img_ref = path_to_diagram_reference("fake_127_q_v2_error.png")
        fig = plot_error_map(backend)
        with BytesIO() as img_buffer:
            fig.savefig(img_buffer, format="png")
            img_buffer.seek(0)
            self.assertImagesAreEqual(Image.open(img_buffer), img_ref, 0.2)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
