# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for plot_histogram."""

import unittest
from io import BytesIO
from collections import Counter

from qiskit.visualization import plot_histogram
from qiskit.utils import optionals
from .visualization import QiskitVisualizationTestCase

if optionals.HAS_MATPLOTLIB:
    import matplotlib as mpl
if optionals.HAS_PIL:
    from PIL import Image


@unittest.skipUnless(optionals.HAS_MATPLOTLIB, "matplotlib not available.")
class TestPlotHistogram(QiskitVisualizationTestCase):
    """Qiskit plot_histogram tests."""

    def test_different_counts_lengths(self):
        """Test plotting two different length dists works"""
        exact_dist = {
            "000000": 0.015624999999999986,
            "000001": 0.015624999999999986,
            "000011": 0.031249999999999965,
            "000111": 0.06249999999999992,
            "100000": 0.015624999999999986,
            "100001": 0.015624999999999986,
            "100011": 0.031249999999999965,
            "100111": 0.06249999999999992,
            "110000": 0.031249999999999965,
            "110001": 0.031249999999999965,
            "110011": 0.06249999999999992,
            "110111": 0.12499999999999982,
            "111111": 0.4999999999999991,
        }

        raw_dist = {
            "000000": 26,
            "000001": 29,
            "010000": 10,
            "010001": 12,
            "010010": 6,
            "010011": 14,
            "010100": 2,
            "010101": 6,
            "010110": 4,
            "010111": 24,
            "011000": 2,
            "011001": 5,
            "011011": 5,
            "011101": 4,
            "011110": 7,
            "011111": 77,
            "000010": 9,
            "100000": 31,
            "100001": 25,
            "100010": 8,
            "100011": 46,
            "100100": 3,
            "100101": 3,
            "100110": 9,
            "100111": 114,
            "101000": 3,
            "101001": 6,
            "101010": 1,
            "101011": 6,
            "101100": 7,
            "101101": 9,
            "101110": 6,
            "101111": 48,
            "000011": 82,
            "110000": 42,
            "110001": 53,
            "110010": 9,
            "110011": 102,
            "110100": 10,
            "110101": 8,
            "110110": 14,
            "110111": 215,
            "111000": 25,
            "111001": 12,
            "111010": 2,
            "111011": 41,
            "111100": 18,
            "111101": 24,
            "111110": 58,
            "111111": 621,
            "000100": 1,
            "000101": 7,
            "000110": 9,
            "000111": 73,
            "001000": 1,
            "001001": 5,
            "001011": 6,
            "001100": 1,
            "001101": 7,
            "001110": 1,
            "001111": 34,
        }

        fig = plot_histogram([raw_dist, exact_dist])
        self.assertIsInstance(fig, mpl.figure.Figure)

    def test_with_number_to_keep(self):
        """Test plotting using number_to_keep"""
        dist = {"00": 3, "01": 5, "11": 8, "10": 11}
        fig = plot_histogram(dist, number_to_keep=2)
        self.assertIsInstance(fig, mpl.figure.Figure)

    def test_with_number_to_keep_multiple_executions(self):
        """Test plotting using number_to_keep with multiple executions"""
        dist = [{"00": 3, "01": 5, "11": 8, "10": 11}, {"00": 3, "01": 7, "10": 11}]
        fig = plot_histogram(dist, number_to_keep=2)
        self.assertIsInstance(fig, mpl.figure.Figure)

    @unittest.skipUnless(optionals.HAS_PIL, "matplotlib not available.")
    def test_with_number_to_keep_multiple_executions_correct_image(self):
        """Test plotting using number_to_keep with multiple executions"""
        data_noisy = {
            "00000": 0.22,
            "00001": 0.003,
            "00010": 0.005,
            "00011": 0.0,
            "00100": 0.004,
            "00101": 0.001,
            "00110": 0.004,
            "00111": 0.001,
            "01000": 0.005,
            "01001": 0.0,
            "01010": 0.002,
            "01011": 0.0,
            "01100": 0.225,
            "01101": 0.001,
            "01110": 0.003,
            "01111": 0.003,
            "10000": 0.012,
            "10001": 0.002,
            "10010": 0.001,
            "10011": 0.001,
            "10100": 0.247,
            "10101": 0.004,
            "10110": 0.003,
            "10111": 0.001,
            "11000": 0.225,
            "11001": 0.005,
            "11010": 0.002,
            "11011": 0.0,
            "11100": 0.015,
            "11101": 0.004,
            "11110": 0.001,
            "11111": 0.0,
        }
        data_ideal = {
            "00000": 0.25,
            "00001": 0,
            "00010": 0,
            "00011": 0,
            "00100": 0,
            "00101": 0,
            "00110": 0,
            "00111": 0.0,
            "01000": 0.0,
            "01001": 0,
            "01010": 0.0,
            "01011": 0.0,
            "01100": 0.25,
            "01101": 0,
            "01110": 0,
            "01111": 0,
            "10000": 0,
            "10001": 0,
            "10010": 0.0,
            "10011": 0.0,
            "10100": 0.25,
            "10101": 0,
            "10110": 0,
            "10111": 0,
            "11000": 0.25,
            "11001": 0,
            "11010": 0,
            "11011": 0,
            "11100": 0.0,
            "11101": 0,
            "11110": 0,
            "11111": 0.0,
        }
        data_ref_noisy = dict(Counter(data_noisy).most_common(5))
        data_ref_noisy["rest"] = sum(data_noisy.values()) - sum(data_ref_noisy.values())
        data_ref_ideal = dict(Counter(data_ideal).most_common(4))  # do not add 0 values
        data_ref_ideal["rest"] = 0
        figure_ref = plot_histogram([data_ref_ideal, data_ref_noisy])
        figure_truncated = plot_histogram([data_ideal, data_noisy], number_to_keep=5)
        with BytesIO() as img_buffer_ref:
            figure_ref.savefig(img_buffer_ref, format="png")
            img_buffer_ref.seek(0)
            with BytesIO() as img_buffer:
                figure_truncated.savefig(img_buffer, format="png")
                img_buffer.seek(0)
                self.assertImagesAreEqual(Image.open(img_buffer_ref), Image.open(img_buffer), 0.2)
        mpl.pyplot.close(figure_ref)
        mpl.pyplot.close(figure_truncated)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "matplotlib not available.")
    def test_number_of_items_in_legend_with_data_starting_with_zero(self):
        """Test legend if there's a 0 value at the first item of the dataset"""
        dist_1 = {"0": 0.369, "1": 0.13975}
        dist_2 = {"0": 0, "1": 0.48784}
        legend = ["lengend_1", "lengend_2"]
        plot = plot_histogram([dist_1, dist_2], legend=legend)
        self.assertEqual(
            len(plot._localaxes[0].legend_.texts),
            2,
            "Plot should have the same number of legend items as defined",
        )

    def test_distribution_with_leading_trailing_int_zeros(self):
        """Test that distributions with int zeros at start/end are detected correctly.

        Regression test for https://github.com/Qiskit/qiskit/issues/11949
        When a probability distribution has integer zeros at the beginning
        (before float probabilities), plot_histogram should still detect it
        as a distribution and not coerce values to integers.
        """
        # This distribution has int zeros at the start and end, with float
        # probabilities in the middle. The old code only checked the first
        # value's type, incorrectly detecting this as "counts" mode.
        dist = {
            0: 0,
            1: 0,
            2: 0,
            3: 0.25,
            4: 0.50,
            5: 0.25,
            6: 0,
            7: 0,
        }
        fig = plot_histogram(dist)
        self.assertIsInstance(fig, mpl.figure.Figure)
        # Verify the y-axis label indicates distribution mode, not counts
        self.assertEqual(fig.axes[0].get_ylabel(), "Quasi-probability")
        mpl.pyplot.close(fig)

    def test_distribution_detection_with_mixed_int_float_zeros(self):
        """Test distribution detection with mix of int 0 and float 0.0 values."""
        # Mix of int 0 and float 0.0 - should be detected as distribution
        dist = {
            "00": 0,      # int zero
            "01": 0.0,    # float zero
            "10": 0.5,    # float
            "11": 0.5,    # float
        }
        fig = plot_histogram(dist)
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertEqual(fig.axes[0].get_ylabel(), "Quasi-probability")
        mpl.pyplot.close(fig)

    def test_pure_counts_still_detected_as_counts(self):
        """Test that pure integer counts are still detected as counts mode."""
        counts = {"00": 50, "01": 25, "10": 15, "11": 10}
        fig = plot_histogram(counts)
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertEqual(fig.axes[0].get_ylabel(), "Count")
        mpl.pyplot.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
