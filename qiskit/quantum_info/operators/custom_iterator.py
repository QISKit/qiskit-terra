# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Custom Lazy Iterator class
"""

from __future__ import annotations
from abc import ABC, abstractmethod


class CustomIterator(ABC):
    """Lazy custom iteration and item access."""

    def __init__(self, obj):
        self.obj = obj
        self._iter = 0

    @abstractmethod
    def __getitem__(self, key):
        """Get next item"""
        # This method should be overridden for lazy conversion of
        # iterator only at a given key value

    def __repr__(self):
        return f"<{type(self.obj)}_iterator at {hex(id(self))}>"

    def __len__(self):
        return len(self.obj)

    def __iter__(self):
        self._iter = 0
        return self

    def __next__(self):
        if self._iter >= len(self):
            raise StopIteration
        self._iter += 1
        return self[self._iter - 1]
