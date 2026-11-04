# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A collection of useful functions for post processing results."""

from __future__ import annotations

import numpy as np

from .make_observable import make_dict_observable


def average_data(counts: dict, observable: dict | np.ndarray | list) -> float:
    """Compute the mean value of a diagonal observable.

    Takes in a diagonal observable in dictionary, list or matrix format and then
    calculates the sum_i value(i) P(i) where value(i) is the value of the
    observable for state i.

    Args:
        counts (dict): a dict of outcomes from an experiment
        observable (dict or matrix or list): The observable to be averaged over.
        As an example, ZZ on qubits can be given as:
        * dict: {"00": 1, "11": 1, "01": -1, "10": -1}
        * matrix: [[1, 0, 0, 0], [0, -1, 0, 0, ], [0, 0, -1, 0], [0, 0, 0, 1]]
        * matrix diagonal (list): [1, -1, -1, 1]

    Returns:
        Double: Average of the observable
    """
    if not isinstance(observable, dict):
        observable = make_dict_observable(observable)
    temp = 0
    tot = sum(counts.values())
    for key, count in counts.items():
        if key in observable:
            temp += count * observable[key] / tot
    return temp
