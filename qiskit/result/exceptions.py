# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Exception for errors when there's an error in the Result
"""

from qiskit.exceptions import QiskitError


class ResultError(QiskitError):
    """Exceptions raised due to errors in result output.

    It may be better for the Qiskit API to raise this exception.

    Args:
        error (dict): This is the error record as it comes back from
            the API. The format is like::

                error = {'status': 403,
                         'message': 'Your credits are not enough.',
                         'code': 'MAX_CREDITS_EXCEEDED'}
    """

    def __init__(self, error):
        super().__init__(error["message"])
        self.status = error["status"]
        self.code = error["code"]

    def __str__(self):
        return f"{self.code}: {self.message}"
