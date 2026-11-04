# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Defines bare dataclasses which house polytope information, as well as a specialized data structure
which describes those two-qubit programs accessible to a given sequence of XX-type interactions.
"""

from __future__ import annotations
from copy import copy
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from qiskit.exceptions import QiskitError

from .utilities import EPSILON


@dataclass
class ConvexPolytopeData:
    """
    The raw data underlying a ConvexPolytope.  Describes a single convex
    polytope, specified by families of `inequalities` and `equalities`, each
    entry of which respectively corresponds to

        inequalities[j][0] + sum_i inequalities[j][i] * xi >= 0

    and

        equalities[j][0] + sum_i equalities[j][i] * xi == 0.
    """

    inequalities: list[list[int]]
    equalities: list[list[int]] = field(default_factory=list)
    name: str = ""


@dataclass
class PolytopeData:
    """
    The raw data of a union of convex polytopes.
    """

    convex_subpolytopes: list[ConvexPolytopeData]


def polytope_has_element(polytope, point):
    """
    Tests whether `polytope` contains `point`.
    """
    return all(
        -EPSILON <= inequality[0] + sum(x * y for x, y in zip(point, inequality[1:]))
        for inequality in polytope.inequalities
    ) and all(
        abs(equality[0] + sum(x * y for x, y in zip(point, equality[1:]))) <= EPSILON
        for equality in polytope.equalities
    )


def manual_get_vertex(polytope, seed=42):
    """
    Returns a single random vertex from `polytope`.
    """
    rng = np.random.default_rng(seed)

    if isinstance(polytope, PolytopeData):
        paragraphs = copy(polytope.convex_subpolytopes)
    elif isinstance(polytope, ConvexPolytopeData):
        paragraphs = [polytope]
    else:
        raise TypeError(f"{type(polytope)} is not polytope-like.")

    rng.shuffle(paragraphs)
    for convex_subpolytope in paragraphs:
        sentences = convex_subpolytope.inequalities + convex_subpolytope.equalities
        if len(sentences) == 0:
            continue
        dimension = len(sentences[0]) - 1
        rng.shuffle(sentences)
        for inequalities in combinations(sentences, dimension):
            matrix = np.array([x[1:] for x in inequalities])
            b = np.array([x[0] for x in inequalities])
            try:
                vertex = np.linalg.inv(-matrix) @ b
                if polytope_has_element(convex_subpolytope, vertex):
                    return vertex
            except np.linalg.LinAlgError:
                pass

    raise QiskitError(f"Polytope has no feasible solutions:\n{polytope}")


@dataclass
class XXPolytope:
    """
    Describes those two-qubit programs accessible to a given sequence of XX-type interactions.

    NOTE: Strengths are normalized so that CX corresponds to pi / 4, which differs from Qiskit's
          conventions around RZX elsewhere.
    """

    # NOTE: This is _not_ a subclass of PolytopeData, because we're never going to call slow,
    #       generic PolytopeData functions on it.

    total_strength: float = 0.0
    max_strength: float = 0.0
    place_strength: float = 0.0

    @classmethod
    def from_strengths(cls, *strengths):
        """
        Constructs an XXPolytope from a sequence of strengths.
        """
        total_strength, max_strength, place_strength = 0, 0, 0
        for strength in strengths:
            total_strength += strength
            if strength >= max_strength:
                max_strength, place_strength = strength, max_strength
            elif strength >= place_strength:
                place_strength = strength

        return XXPolytope(
            total_strength=total_strength, max_strength=max_strength, place_strength=place_strength
        )

    def add_strength(self, new_strength: float = 0.0):
        """
        Returns a new XXPolytope with one new XX interaction appended.
        """
        return XXPolytope(
            total_strength=self.total_strength + new_strength,
            max_strength=max(self.max_strength, new_strength),
            place_strength=(
                self.max_strength
                if new_strength > self.max_strength
                else new_strength if new_strength > self.place_strength else self.place_strength
            ),
        )

    @property
    def _offsets(self):
        """
        Returns b with A*x + b ≥ 0 iff x belongs to the XXPolytope.
        """
        return np.array(
            [
                0,
                0,
                0,
                np.pi / 2,
                self.total_strength,
                self.total_strength - 2 * self.max_strength,
                self.total_strength - self.max_strength - self.place_strength,
            ]
        )

    def member(self, point):
        """
        Returns True when `point` is a member of `self`.
        """

        reflected_point = point.copy().reshape(-1, 3)
        rows = reflected_point[:, 0] >= np.pi / 4 + EPSILON
        reflected_point[rows, 0] = np.pi / 2 - reflected_point[rows, 0]
        reflected_point = reflected_point.reshape(point.shape)

        return np.all(
            self._offsets + np.einsum("ij,...j->...i", A, reflected_point) >= -EPSILON, axis=-1
        )

    def nearest(self, point):
        """
        Finds the nearest point (in Euclidean or infidelity distance) to `self`.
        """
        # pylint:disable=invalid-name

        # NOTE: A CAS says that there are no degenerate double intersections, and the only
        #       degenerate triple intersections are
        #
        #           (1, -1, 0), (0, 0, 1), (-1, 1, 1) and (1, 1, 0), (0, 0, 1), (1, 1, 1).
        #
        #       Skipping this pair won't save much work, so we don't bother.

        # A1, A1inv, A2, A2inv, A3, A3inv
        # These global variables contain projection matrices, computed once-and-for-all, which
        # produce the Euclidean-nearest projection.

        if isinstance(point, np.ndarray) and len(point.shape) == 1:
            y0 = point.copy()
        elif isinstance(point, list):
            y0 = np.array(point)
        else:
            raise TypeError(f"Can't handle type of point: {point} ({type(point)})")

        reflected_p = y0[0] > np.pi / 4 + EPSILON
        if reflected_p:
            y0[0] = np.pi / 2 - y0[0]

        # short circuit in codimension 0
        if self.member(y0):
            if reflected_p:
                y0[0] = np.pi / 2 - y0[0]
            return y0

        # codimension 1
        b1 = self._offsets.reshape(7, 1)
        A1y0 = np.einsum("ijk,k->ij", A1, y0)
        nearest1 = np.einsum("ijk,ik->ij", A1inv, b1 + A1y0) - y0

        # codimension 2
        b2 = np.array([*combinations(self._offsets, 2)])
        A2y0 = np.einsum("ijk,k->ij", A2, y0)
        nearest2 = np.einsum("ijk,ik->ij", A2inv, b2 + A2y0) - y0

        # codimension 3
        b3 = np.array([*combinations(self._offsets, 3)])
        nearest3 = np.einsum("ijk,ik->ij", A3inv, b3)

        # pick the nearest
        nearest = -np.concatenate([nearest1, nearest2, nearest3])
        nearest = nearest[self.member(nearest)]
        smallest_index = np.argmin(np.linalg.norm(nearest - y0, axis=1))

        if reflected_p:
            nearest[smallest_index][0] = np.pi / 2 - nearest[smallest_index][0]
        return nearest[smallest_index]


A = np.array(
    [
        [1, -1, 0],  # a ≥ b
        [0, 1, -1],  # b ≥ c
        [0, 0, 1],  # c ≥ 0
        [-1, -1, 0],  # pi/2 ≥ a + b
        [-1, -1, -1],  # strength
        [1, -1, -1],  # slant
        [0, 0, -1],  # frustum
    ]
)
A1 = A.reshape(-1, 1, 3)  # pylint:disable=too-many-function-args
A1inv = np.linalg.pinv(A1)
A2 = np.array([np.array([x, y], dtype=float) for (x, y) in combinations(A, 2)])
A2inv = np.linalg.pinv(A2)
A3 = np.array([np.array([x, y, z], dtype=float) for (x, y, z) in combinations(A, 3)])
A3inv = np.linalg.pinv(A3)
"""
These globals house matrices, computed once-and-for-all, which project onto the Euclidean-nearest
planes spanned by the planes/points/lines defined by the faces of any XX polytope.

See `XXPolytope.nearest`.
"""
