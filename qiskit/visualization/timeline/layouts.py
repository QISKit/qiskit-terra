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

"""
A collection of functions that decide the layout of an output image.
See :py:mod:`~qiskit.visualization.timeline.types` for more info on the required data.

There are 2 types of layout functions in this module.

1. layout.bit_arrange

In this stylesheet entry the input data is a list of `types.Bits` and returns a
sorted list of `types.Bits`.


The function signature of the layout is restricted to:

    ```python

    def my_layout(bits: List[types.Bits]) -> List[types.Bits]:

        # your code here: sort input bits and return list of bits
    ```

2. layout.time_axis_map

In this stylesheet entry the input data is `Tuple[int, int]` that represents horizontal
axis limit of the output image. The layout function returns `types.HorizontalAxis` data
which is consumed by the plotter API to make horizontal axis.

The function signature of the layout is restricted to:

    ```python

    def my_layout(time_window: Tuple[int, int]) -> types.HorizontalAxis:

        # your code here: create and return axis config
    ```

Arbitrary layout function satisfying the above format can be accepted.
"""
import numpy as np

from qiskit import circuit
from qiskit.visualization.timeline import types


def qreg_creg_ascending(bits: list[types.Bits]) -> list[types.Bits]:
    """Sort bits by ascending order.

    Bit order becomes Q0, Q1, ..., Cl0, Cl1, ...

    Args:
        bits: List of bits to sort.

    Returns:
        Sorted bits.
    """
    return [x for x in bits if isinstance(x, circuit.Qubit)] + [
        x for x in bits if isinstance(x, circuit.Clbit)
    ]


def qreg_creg_descending(bits: list[types.Bits]) -> list[types.Bits]:
    """Sort bits by descending order.

    Bit order becomes Q_N, Q_N-1, ..., Cl_N, Cl_N-1, ...

    Args:
        bits: List of bits to sort.

    Returns:
        Sorted bits.
    """
    return [x for x in bits[::-1] if isinstance(x, circuit.Qubit)] + [
        x for x in bits[::-1] if isinstance(x, circuit.Clbit)
    ]


def time_map_in_dt(time_window: tuple[int, int]) -> types.HorizontalAxis:
    """Layout function for the horizontal axis formatting.

    Generate equispaced 6 horizontal axis ticks.

    Args:
        time_window: Left and right edge of this graph.

    Returns:
        Axis formatter object.
    """
    # shift time axis
    t0, t1 = time_window

    # axis label
    axis_loc = np.linspace(max(t0, 0), t1, 6)
    axis_label = axis_loc.copy()

    # consider time resolution
    label = "System cycle time (dt)"

    formatted_label = [f"{val:.0f}" for val in axis_label]

    return types.HorizontalAxis(
        window=(t0, t1), axis_map=dict(zip(axis_loc, formatted_label)), label=label
    )
