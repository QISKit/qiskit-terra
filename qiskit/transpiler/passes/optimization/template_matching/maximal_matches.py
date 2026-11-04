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
It stores all maximal matches from the given matches obtained by the template
matching algorithm.
"""


class Match:
    """
    Class Match is an object to store a list of matches with its qubits and
    clbits configuration.
    """

    def __init__(self, match, qubit, clbit):
        """
        Create a Match with necessary arguments.
        Args:
            match (list): list of a match.
            qubit (list): list of qubits configuration.
            clbit (list): list of clbits configuration.
        """

        self.match = match
        self.qubit = qubit
        self.clbit = clbit


class MaximalMatches:
    """
    Class MaximalMatches allows to sort and store the maximal matches from the list
    of matches obtained with the template matching algorithm.
    """

    def __init__(self, template_matches):
        """
        Initialize MaximalMatches with the necessary arguments.
        Args:
            template_matches (list): list of matches obtained from running the algorithm.
        """
        self.template_matches = template_matches

        self.max_match_list = []

    def run_maximal_matches(self):
        """
        Method that extracts and stores maximal matches in decreasing length order.
        """

        self.max_match_list = [
            Match(
                sorted(self.template_matches[0].match),
                self.template_matches[0].qubit,
                self.template_matches[0].clbit,
            )
        ]

        for matches in self.template_matches[1::]:
            present = False
            for max_match in self.max_match_list:
                for elem in matches.match:
                    if elem in max_match.match and len(matches.match) <= len(max_match.match):
                        present = True
            if not present:
                self.max_match_list.append(
                    Match(sorted(matches.match), matches.qubit, matches.clbit)
                )
