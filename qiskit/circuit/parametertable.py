# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Look-up table for variable parameters in QuantumCircuit.
"""

from collections.abc import MappingView


class ParameterView(MappingView):
    """Temporary class to transition from a set return-type to list.

    Derives from a list but implements all set methods, but all set-methods emit deprecation
    warnings.
    """

    def __init__(self, iterable=None):
        if iterable is not None:
            self.data = list(iterable)
        else:
            self.data = []

        super().__init__(self.data)

    def copy(self):
        """Copy the ParameterView."""
        return self.__class__(self.data.copy())

    def isdisjoint(self, x):
        """Check whether self and the input are disjoint."""
        return not any(element in self for element in x)

    def remove(self, x):
        """Remove an existing element from the view."""
        self.data.remove(x)

    def __repr__(self):
        """Format the class as string."""
        return f"ParameterView({self.data})"

    def __getitem__(self, index):
        """Get items."""
        return self.data[index]

    def __and__(self, x):
        """Get the intersection between self and the input."""
        inter = []
        for element in self:
            if element in x:
                inter.append(element)

        return self.__class__(inter)

    def __rand__(self, x):
        """Get the intersection between self and the input."""
        return self.__and__(x)

    def __iand__(self, x):
        """Get the intersection between self and the input in-place."""
        for element in self:
            if element not in x:
                self.remove(element)
        return self

    def __len__(self):
        """Get the length."""
        return len(self.data)

    def __or__(self, x):
        """Get the union of self and the input."""
        return set(self) | set(x)

    def __sub__(self, x):
        """Get the difference between self and the input."""
        return set(self) - set(x)

    def __xor__(self, x):
        """Get the symmetric difference between self and the input."""
        return set(self) ^ set(x)

    def __ne__(self, other):
        return set(other) != set(self)

    def __eq__(self, other):
        return set(other) == set(self)

    def __le__(self, x):
        return all(element in x for element in self)

    def __lt__(self, x):
        if x != self:
            return self <= x
        return False

    def __ge__(self, x):
        return all(element in self for element in x)

    def __gt__(self, x):
        if x != self:
            return self >= x
        return False

    def __iter__(self):
        return iter(self.data)

    def __contains__(self, x):
        return x in self.data

    __hash__: None  # type: ignore
    __ror__ = __or__
