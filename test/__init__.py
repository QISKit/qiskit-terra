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

"""With some utils"""

import itertools
from ddt import data, unpack

from .utils.base import QiskitTestCase
from .utils.decorators import slow_test

__all__ = ["QiskitTestCase", "combine", "slow_test"]


class Case(dict):
    """<no description>"""


def generate_cases(docstring, dsc=None, name=None, **kwargs):
    """Combines kwargs in Cartesian product and creates Case with them"""
    ret = []
    keys = kwargs.keys()
    vals = kwargs.values()
    for values in itertools.product(*vals):
        case = Case(zip(keys, values))
        if docstring is not None:
            setattr(case, "__doc__", docstring.format(**case))
        if dsc is not None:
            setattr(case, "__doc__", dsc.format(**case))
        if name is not None:
            setattr(case, "__name__", name.format(**case))
        ret.append(case)
    return ret


def combine(**kwargs):
    """Decorator to create combinations and tests
    @combine(level=[0, 1, 2, 3],
             circuit=[a, b, c, d],
             dsc='Test circuit {circuit.__name__} with level {level}',
             name='{circuit.__name__}_level{level}')
    """

    def deco(func):
        return data(*generate_cases(docstring=func.__doc__, **kwargs))(unpack(func))

    return deco
