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
Routines for producing right-angled paths through the Weyl alcove.  Consider a set of native
interactions with an associated minimal covering set of minimum-cost circuit polytopes, as well as a
target coordinate.  The coverage set associates to the target coordinate a circuit type
C = (O1 ... On) consisting of a sequence of native interactions Oj.  A _path_ is a sequence
(I P1 ... Pn) of intermediate Weyl points, where Pj is accessible from P(j-1) by Oj.  A path is said
to be _right-angled_ when at each step one coordinate is fixed (up to possible Weyl reflection) when
expressed in canonical coordinates.

The key inputs to our method are:

+ A family of "b coordinates" which describe the target canonical coordinate.
+ A family of "a coordinates" which describe the source canonical coordinate.
+ A sequence of interaction strengths for which the b-coordinate can be modeled, with one selected
  to be stripped from the sequence ("beta").  The others are bundled as the sum of the
  sequence (s+), its maximum value (s1), and its second maximum value (s2).

Given the b-coordinate and a set of intersection strengths, the procedure for backsolving for the
a-coordinates is then extracted from the monodromy polytope.

NOTE: The constants in this file are auto-generated and are not meant to be edited by hand / read.
"""

from __future__ import annotations
import numpy as np

from .polytopes import ConvexPolytopeData, PolytopeData, manual_get_vertex, polytope_has_element


def get_augmented_coordinate(target_coordinate, strengths):
    """
    Assembles a coordinate in the system used by `xx_region_polytope`.
    """
    *strengths, beta = strengths
    strengths = sorted(strengths + [0, 0])
    interaction_coordinate = [sum(strengths), strengths[-1], strengths[-2], beta]
    return [*target_coordinate, *interaction_coordinate]


def decomposition_hop(target_coordinate, strengths):
    """
    Given a `target_coordinate` and a list of interaction `strengths`, produces a new canonical
    coordinate which is one step back along `strengths`.

    `target_coordinate` is taken to be in positive canonical coordinates, and the entries of
    strengths are taken to be [0, pi], so that (sj / 2, 0, 0) is a positive canonical coordinate.
    """

    target_coordinate = [x / (np.pi / 2) for x in target_coordinate]
    strengths = [x / np.pi for x in strengths]

    augmented_coordinate = get_augmented_coordinate(target_coordinate, strengths)
    specialized_polytope = None
    for cp in xx_region_polytope.convex_subpolytopes:
        if not polytope_has_element(cp, augmented_coordinate):
            continue
        if "AF=B1" in cp.name:
            af, _, _ = target_coordinate
        elif "AF=B2" in cp.name:
            _, af, _ = target_coordinate
        elif "AF=B3" in cp.name:
            _, _, af = target_coordinate
        else:
            raise ValueError("Couldn't find a coordinate to fix.")

        raw_convex_polytope = next(
            (cpp for cpp in xx_lift_polytope.convex_subpolytopes if cpp.name == cp.name), None
        )

        coefficient_dict = {}
        for inequality in raw_convex_polytope.inequalities:
            if inequality[1] == 0 and inequality[2] == 0:
                continue
            offset = (
                inequality[0]  # old constant term
                + inequality[3] * af
                + inequality[4] * augmented_coordinate[0]  # b1
                + inequality[5] * augmented_coordinate[1]  # b2
                + inequality[6] * augmented_coordinate[2]  # b3
                + inequality[7] * augmented_coordinate[3]  # s+
                + inequality[8] * augmented_coordinate[4]  # s1
                + inequality[9] * augmented_coordinate[5]  # s2
                + inequality[10] * augmented_coordinate[6]  # beta
            )

            if offset <= coefficient_dict.get((inequality[1], inequality[2]), offset):
                coefficient_dict[(inequality[1], inequality[2])] = offset

        specialized_polytope = PolytopeData(
            convex_subpolytopes=[
                ConvexPolytopeData(
                    inequalities=[[v, h, l_] for ((h, l_), v) in coefficient_dict.items()]
                )
            ]
        )

        break

    if specialized_polytope is None:
        raise ValueError("Failed to match a constrained_polytope summand.")

    ah, al = manual_get_vertex(specialized_polytope)
    return [x * (np.pi / 2) for x in sorted([ah, al, af], reverse=True)]


xx_region_polytope = PolytopeData(
    convex_subpolytopes=[
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 0],
                [1, -1, -1, 0, 0, 0, 0, 0],
                [0, -1, -1, -1, 1, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [0, 1, -1, -1, 1, -2, 0, 1],
                [0, 1, -1, -1, 1, 0, 0, -1],
                [0, 0, 0, -1, 1, -1, 0, 0],
                [0, 0, -1, 0, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            equalities=[],
            name=(
                "I ∩ A alcove ∩ "
                "A unreflected ∩ ah slant ∩ al frustum ∩ B alcove ∩ B unreflected ∩ AF=B3"
            ),
        ),
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 0],
                [1, -1, -1, 0, 0, 0, 0, 0],
                [1, -1, -1, -1, 1, -2, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [-1, 1, -1, -1, 1, 0, 0, 1],
                [1, -1, -1, -1, 1, 0, 0, -1],
                [0, 0, 0, -1, 1, -1, 0, 0],
                [0, 0, -1, 0, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            equalities=[],
            name=(
                "I ∩ A alcove ∩ "
                "A reflected ∩ ah strength ∩ al frustum ∩ B alcove ∩ B reflected ∩ AF=B3"
            ),
        ),
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -2],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 0],
                [0, 1, -1, -1, 1, -2, 0, 1],
                [0, -1, -1, -1, 1, 0, 0, 1],
                [0, 0, 1, -1, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, -1],
                [0, 1, 1, -1, 1, -2, 0, -1],
                [0, -1, 1, -1, 1, 0, 0, -1],
                [0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            equalities=[],
            name=(
                "I ∩ A alcove ∩ "
                "A unreflected ∩ af slant ∩ al frustum ∩ B alcove ∩ B unreflected ∩ AF=B1"
            ),
        ),
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, -2],
                [0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, -1, -1, 0],
                [-1, 1, -1, -1, 1, 0, 0, 1],
                [1, -1, -1, -1, 1, -2, 0, 1],
                [0, 0, 1, -1, 0, 0, 0, 0],
                [1, -1, 1, -1, 0, 0, 0, -1],
                [-1, 1, 1, -1, 1, 0, 0, -1],
                [1, -1, 1, -1, 1, -2, 0, -1],
                [0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            equalities=[],
            name=(
                "I ∩ A alcove ∩ "
                "A reflected ∩ af strength ∩ al frustum ∩ B alcove ∩ B reflected ∩ AF=B1"
            ),
        ),
    ]
)
"""
For any choice of sequence of strengths, the theory of the monodromy polytope yields a polytope P
so that (b1, b2, b3) belongs to P if and only if (b1, b2, b3) is the positive canonical coordinate
of a program expressible by a circuit whose 2Q interactions are the prescribed sequence of
fractional CX strengths, interleaved with arbitrary single-qubit gates.  The polytope above has the
same property as P, but (1) it is parametrized over the sequence of strengths, and (2) it is broken
into a certain sum of convex regions. Each region is tagged with a name, and it is the projection of
the region of `xx_lift_polytope` (see below) with the corresponding name.

The coordinates are [k, b1, b2, b3, s+, s1, s2, beta].
"""


xx_lift_polytope = PolytopeData(
    convex_subpolytopes=[
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, -1, -1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, -1, -1, 0, 0, 0, 1, -2, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, -1, -1, 1, 0, 0, 1],
                [0, 0, 0, 0, 1, -1, -1, 1, -2, 0, 1],
                [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 1],
                [2, -1, -1, 0, -1, -1, 0, 0, 0, 0, -1],
                [0, 1, 1, 0, -1, -1, 0, 0, 0, 0, 1],
                [0, -1, 1, 0, 1, -1, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 1, -1, 0, 0, 0, 0, -1],
            ],
            equalities=[[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]],
            name=(
                "I ∩ A alcove ∩ "
                "A unreflected ∩ ah slant ∩ al frustum ∩ B alcove ∩ B unreflected ∩ AF=B3"
            ),
        ),
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, -1, -1, 0, 0, 0, 1, 0, 0, 0],
                [0, -1, -1, 1, 0, 0, 0, 1, -2, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1, -1, -1, 1, 0, 0, 1],
                [0, 0, 0, 0, 1, -1, -1, 1, -2, 0, 1],
                [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 1],
                [2, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1],
                [0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 1],
                [0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 1, -1, 0, 0, 0, -1],
            ],
            equalities=[[0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]],
            name=(
                "I ∩ A alcove ∩ "
                "A unreflected ∩ af slant ∩ al frustum ∩ B alcove ∩ B unreflected ∩ AF=B1"
            ),
        ),
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, -1, 0, 0, 0, 1, -2, 0, 0],
                [-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1],
                [1, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1],
                [1, 0, 0, 0, -1, -1, -1, 1, -2, 0, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 1],
                [2, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1],
                [0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 1],
                [0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 1, -1, 0, 0, 0, -1],
            ],
            equalities=[[0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]],
            name=(
                "I ∩ A alcove ∩ "
                "A reflected ∩ af strength ∩ al frustum ∩ B alcove ∩ B reflected ∩ AF=B1"
            ),
        ),
        ConvexPolytopeData(
            inequalities=[
                [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                [1, -1, -1, -1, 0, 0, 0, 1, -2, 0, 0],
                [-1, 1, -1, -1, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0],
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1],
                [1, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1],
                [1, 0, 0, 0, -1, -1, -1, 1, -2, 0, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1],
                [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0],
                [0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 1],
                [2, -1, -1, 0, -1, -1, 0, 0, 0, 0, -1],
                [0, 1, 1, 0, -1, -1, 0, 0, 0, 0, 1],
                [0, -1, 1, 0, 1, -1, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 1],
                [0, 1, -1, 0, 1, -1, 0, 0, 0, 0, -1],
            ],
            equalities=[[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]],
            name=(
                "I ∩ A alcove ∩ "
                "A reflected ∩ ah strength ∩ al frustum ∩ B alcove ∩ B reflected ∩ AF=B3"
            ),
        ),
    ]
)
"""
For any choice of sequence of strengths, the theory of the monodromy polytope yields a polytope Q
so that (a1, a2, a3, b1, b2, b3) belongs to Q if and only if:

    * (b1, b2, b3) is the positive canonical coordinate of a program expressible by a circuit whose
      2Q interactions are the prescribed sequence of fractional CX strengths, interleaved with
      arbitrary single-qubit gates.
    * (a1, a2, a3) is the positive canonical coordinate of a program expressible by a circuit whose
      2Q interactions are the prescribed sequence of fractional CX strengths _with the last
      interaction omitted_, interleaved with arbitrary single-qubit gates.
    * (b1, b2, b3) is accessible from (a1, a2, a3) using single-qubit gates and a single application
      of the final fractional CX strength.

The polytope above is a slight variation on Q:

    (1) It is parametrized over the sequence of strengths.
    (2) It is broken into a certain sum of convex regions. If (b1, b2, b3) belongs to the projection
        of one of these convex regions, then that same region's projection to (a1, a2, a3) is
        guaranteed to be nonempty. (To test this condition, see `xx_region_polytope`.)

Points in this region can thus be used to calculate circuits using `decompose_xxyy_into_xxyy_xx`.

The coordinates are [k, ah, al, af, b1, b2, b3, s+, s1, s2, beta].
"""
