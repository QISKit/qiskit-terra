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

"""Module containing template matching methods."""

from .forward_match import ForwardMatch
from .backward_match import BackwardMatch, Match, MatchingScenarios, MatchingScenariosList
from .template_matching import TemplateMatching
from .maximal_matches import MaximalMatches
from .template_substitution import SubstitutionConfig, TemplateSubstitution

__all__ = [
    "BackwardMatch",
    "ForwardMatch",
    "Match",
    "MatchingScenarios",
    "MatchingScenariosList",
    "MaximalMatches",
    "SubstitutionConfig",
    "TemplateMatching",
    "TemplateSubstitution",
]
