# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
SpecialPolynomial class.
"""

from __future__ import annotations
import itertools
from itertools import combinations
import copy
from functools import reduce
from operator import mul
import numpy as np

from qiskit.exceptions import QiskitError


class SpecialPolynomial:
    """Multivariate polynomial with special form.

    Maximum degree 3, n Z_2 variables, coefficients in Z_8.
    """

    def __init__(self, n_vars):
        """Construct the zero polynomial on n_vars variables."""
        #   1 constant term
        #   n linear terms x_1, ..., x_n
        #   {n choose 2} quadratic terms x_1x_2, x_1x_3, ..., x_{n-1}x_n
        #   {n choose 3} cubic terms x_1x_2x_3, ..., x_{n-2}x_{n-1}x_n
        # and coefficients in Z_8
        if n_vars < 1:
            raise QiskitError("n_vars for SpecialPolynomial is too small.")
        self.n_vars = n_vars
        self.nc2 = int(n_vars * (n_vars - 1) / 2)
        self.nc3 = int(n_vars * (n_vars - 1) * (n_vars - 2) / 6)
        self.weight_0 = 0
        self.weight_1 = np.zeros(n_vars, dtype=np.int8)
        self.weight_2 = np.zeros(self.nc2, dtype=np.int8)
        self.weight_3 = np.zeros(self.nc3, dtype=np.int8)

    def mul_monomial(self, indices):
        """Multiply by a monomial given by indices.

        Returns the product.
        """
        length = len(indices)
        if length >= 4:
            raise QiskitError("There is no term with on more than 3 indices.")
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() and (indices_arr > self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        if length > 1 and (np.diff(indices_arr) <= 0).any():
            raise QiskitError("Indices are non-increasing!")
        result = SpecialPolynomial(self.n_vars)
        if length == 0:
            result = copy.deepcopy(self)
        else:
            terms0 = [[]]
            terms1 = list(combinations(range(self.n_vars), r=1))
            terms2 = list(combinations(range(self.n_vars), r=2))
            terms3 = list(combinations(range(self.n_vars), r=3))
            for term in terms0 + terms1 + terms2 + terms3:
                value = self.get_term(term)
                new_term = list(set(term).union(set(indices)))
                result.set_term(new_term, (result.get_term(new_term) + value) % 8)
        return result

    def __mul__(self, other):
        """Multiply two polynomials."""
        if not isinstance(other, SpecialPolynomial):
            other = int(other)
        result = SpecialPolynomial(self.n_vars)
        if isinstance(other, int):
            result.weight_0 = (self.weight_0 * other) % 8
            result.weight_1 = (self.weight_1 * other) % 8
            result.weight_2 = (self.weight_2 * other) % 8
            result.weight_3 = (self.weight_3 * other) % 8
        else:
            if self.n_vars != other.n_vars:
                raise QiskitError("Multiplication on different n_vars.")
            terms0 = [[]]
            terms1 = list(combinations(range(self.n_vars), r=1))
            terms2 = list(combinations(range(self.n_vars), r=2))
            terms3 = list(combinations(range(self.n_vars), r=3))
            for term in terms0 + terms1 + terms2 + terms3:
                value = other.get_term(term)
                if value != 0:
                    temp = copy.deepcopy(self)
                    temp = temp.mul_monomial(term)
                    temp = temp * value
                    result = result + temp
        return result

    def __rmul__(self, other):
        """Right multiplication.

        This operation is commutative.
        """
        return self.__mul__(other)

    def __add__(self, other):
        """Add two polynomials."""
        if not isinstance(other, SpecialPolynomial):
            raise QiskitError("Element to add is not a SpecialPolynomial.")
        if self.n_vars != other.n_vars:
            raise QiskitError("Addition on different n_vars.")
        result = SpecialPolynomial(self.n_vars)
        result.weight_0 = (self.weight_0 + other.weight_0) % 8
        result.weight_1 = (self.weight_1 + other.weight_1) % 8
        result.weight_2 = (self.weight_2 + other.weight_2) % 8
        result.weight_3 = (self.weight_3 + other.weight_3) % 8
        return result

    def evaluate(self, xval):
        """Evaluate the multinomial at xval.

        if xval is a length n z2 vector, return element of Z8.
        if xval is a length n vector of multinomials, return
        a multinomial. The multinomials must all be on n vars.
        """
        if len(xval) != self.n_vars:
            raise QiskitError("Evaluate on wrong number of variables.")
        check_int = [isinstance(x, int) for x in xval]
        check_poly = [isinstance(x, SpecialPolynomial) for x in xval]
        if False in check_int and False in check_poly:
            raise QiskitError("Evaluate on a wrong type.")
        is_int = False not in check_int
        if not is_int:
            if False in [i.n_vars == self.n_vars for i in xval]:
                raise QiskitError("Evaluate on incompatible polynomials.")
        else:
            xval = xval % 2
        # Examine each term of this polynomial
        terms0 = [[]]
        terms1 = list(combinations(range(self.n_vars), r=1))
        terms2 = list(combinations(range(self.n_vars), r=2))
        terms3 = list(combinations(range(self.n_vars), r=3))
        # Set the initial result and start for each term
        if is_int:
            result = 0
            start = 1
        else:
            result = SpecialPolynomial(self.n_vars)
            start = SpecialPolynomial(self.n_vars)
            start.weight_0 = 1
        # Compute the new terms and accumulate
        for term in terms0 + terms1 + terms2 + terms3:
            value = self.get_term(term)
            if value != 0:
                newterm = reduce(mul, [xval[j] for j in term], start)
                result = result + value * newterm
        if isinstance(result, int):
            result = result % 8
        return result

    def set_pj(self, indices):
        """Set to special form polynomial on subset of variables.

        p_J(x) := sum_{a subseteq J,|a| neq 0} (-2)^{|a|-1}x^a
        """
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        indices = sorted(indices)
        subsets_2 = itertools.combinations(indices, 2)
        subsets_3 = itertools.combinations(indices, 3)
        self.weight_0 = 0
        self.weight_1 = np.zeros(self.n_vars)
        self.weight_2 = np.zeros(self.nc2)
        self.weight_3 = np.zeros(self.nc3)
        for j in indices:
            self.set_term([j], 1)
        for j in subsets_2:
            self.set_term(list(j), 6)
        for j in subsets_3:
            self.set_term(list(j), 4)

    def get_term(self, indices):
        """Get the value of a term given the list of variables.

        Example: indices = [] returns the constant
                 indices = [0] returns the coefficient of x_0
                 indices = [0,3] returns the coefficient of x_0x_3
                 indices = [0,1,3] returns the coefficient of x_0x_1x_3

        If len(indices) > 3 the method fails.
        If the indices are out of bounds the method fails.
        If the indices are not increasing the method fails.
        """
        length = len(indices)
        if length >= 4:
            return 0
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        if length > 1 and (np.diff(indices_arr) <= 0).any():
            raise QiskitError("Indices are non-increasing.")

        if length == 0:
            return self.weight_0
        if length == 1:
            return self.weight_1[indices[0]]
        if length == 2:
            # sum(self.n_vars-j, {j, 1, indices[0]})
            offset_1 = int(indices[0] * self.n_vars - ((indices[0] + 1) * indices[0]) / 2)
            offset_2 = int(indices[1] - indices[0] - 1)
            return self.weight_2[offset_1 + offset_2]

        # handle length = 3
        tmp_1 = self.n_vars - indices[0]
        offset_1 = int((tmp_1 - 3) * (tmp_1 - 2) * (tmp_1 - 1) / 6)
        tmp_2 = self.n_vars - indices[1]
        offset_2 = int((tmp_2 - 2) * (tmp_2 - 1) / 2)
        offset_3 = self.n_vars - indices[2]
        offset = int(
            self.n_vars * (self.n_vars - 1) * (self.n_vars - 2) / 6 - offset_1 - offset_2 - offset_3
        )

        return self.weight_3[offset]

    def set_term(self, indices, value):
        """Set the value of a term given the list of variables.

        Example: indices = [] returns the constant
                 indices = [0] returns the coefficient of x_0
                 indices = [0,3] returns the coefficient of x_0x_3
                 indices = [0,1,3] returns the coefficient of x_0x_1x_3

        If len(indices) > 3 the method fails.
        If the indices are out of bounds the method fails.
        If the indices are not increasing the method fails.
        The value is reduced modulo 8.
        """
        length = len(indices)
        if length >= 4:
            return
        indices_arr = np.array(indices)
        if (indices_arr < 0).any() or (indices_arr >= self.n_vars).any():
            raise QiskitError("Indices are out of bounds.")
        if length > 1 and (np.diff(indices_arr) <= 0).any():
            raise QiskitError("Indices are non-increasing.")

        value = value % 8
        if length == 0:
            self.weight_0 = value
        elif length == 1:
            self.weight_1[indices[0]] = value
        elif length == 2:
            # sum(self.n_vars-j, {j, 1, indices[0]})
            offset_1 = int(indices[0] * self.n_vars - ((indices[0] + 1) * indices[0]) / 2)
            offset_2 = int(indices[1] - indices[0] - 1)
            self.weight_2[offset_1 + offset_2] = value
        else:  # length = 3
            tmp_1 = self.n_vars - indices[0]
            offset_1 = int((tmp_1 - 3) * (tmp_1 - 2) * (tmp_1 - 1) / 6)
            tmp_2 = self.n_vars - indices[1]
            offset_2 = int((tmp_2 - 2) * (tmp_2 - 1) / 2)
            offset_3 = self.n_vars - indices[2]
            offset = int(
                self.n_vars * (self.n_vars - 1) * (self.n_vars - 2) / 6
                - offset_1
                - offset_2
                - offset_3
            )
            self.weight_3[offset] = value

    @property
    def key(self):
        """Return a string representation."""
        tup = (self.weight_0, tuple(self.weight_1), tuple(self.weight_2), tuple(self.weight_3))
        return tup

    def __eq__(self, x):
        """Test equality."""
        return isinstance(x, SpecialPolynomial) and self.key == x.key

    def __str__(self):
        """Return formatted string representation."""
        out = str(self.weight_0)
        for i in range(self.n_vars):
            value = self.get_term([i])
            if value != 0:
                out += " + "
                if value != 1:
                    out += str(value) + "*"
                out += "x_" + str(i)
        for i in range(self.n_vars - 1):
            for j in range(i + 1, self.n_vars):
                value = self.get_term([i, j])
                if value != 0:
                    out += " + "
                    if value != 1:
                        out += str(value) + "*"
                    out += "x_" + str(i) + "*x_" + str(j)
        for i in range(self.n_vars - 2):
            for j in range(i + 1, self.n_vars - 1):
                for k in range(j + 1, self.n_vars):
                    value = self.get_term([i, j, k])
                    if value != 0:
                        out += " + "
                        if value != 1:
                            out += str(value) + "*"
                        out += "x_" + str(i) + "*x_" + str(j) + "*x_" + str(k)
        return out
