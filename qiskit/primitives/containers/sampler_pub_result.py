# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Sampler Pub result class
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .bit_array import BitArray
from .pub_result import PubResult


class SamplerPubResult(PubResult):
    """Result of Sampler Pub.

    This is a :class:`.SamplerV2`-specific subclass of :class:`.PubResult` that adds helper methods
    to deal with the bit-array like :class:`.BitArray` data directly (implicitly going via the
    :class:`.DataBin` in :attr:`data`).  See :class:`.PubResult` for more documentation on how to
    use this class.
    """

    def join_data(self, names: Iterable[str] | None = None) -> BitArray | np.ndarray:
        """Join data from many registers into one data container.

        Data is joined along the bits axis. For example, for :class:`~.BitArray` data, this corresponds
        to bitstring concatenation.

        Args:
            names: Which registers to join. Their order is maintained, for example, given
                ``["alpha", "beta"]``, the data from register ``alpha`` is placed to the left of the
                data from register ``beta``. When ``None`` is given, this value is set to the
                ordered list of register names, which will have been preserved from the input circuit
                order.

        Returns:
            Joint data.

        Raises:
            ValueError: If specified names are empty.
            ValueError: If specified name does not exist.
            TypeError: If specified data comes from incompatible types.
        """
        if names is None:
            names = list(self.data)
            if not names:
                raise ValueError("No entry exists in the data bin.")
        else:
            names = list(names)
            if not names:
                raise ValueError("An empty name list is given.")
            for name in names:
                if name not in self.data:
                    raise ValueError(f"Name '{name}' does not exist.")

        data = [self.data[name] for name in names]
        if isinstance(data[0], BitArray):
            if not all(isinstance(datum, BitArray) for datum in data):
                raise TypeError("Data comes from incompatible types.")
            joint_data = BitArray.concatenate_bits(data)
        elif isinstance(data[0], np.ndarray):
            if not all(isinstance(datum, np.ndarray) for datum in data):
                raise TypeError("Data comes from incompatible types.")
            joint_data = np.concatenate(data, axis=-1)
        else:
            raise TypeError("Data comes from incompatible types.")
        return joint_data
