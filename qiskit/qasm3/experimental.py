# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Experimental feature flags."""

import enum

# This enumeration is shared between a bunch of parts of the exporter, so it gets its own file to
# avoid cyclic dependencies.


class ExperimentalFeatures(enum.Flag):
    """Flags for experimental features that the OpenQASM 3 exporter supports.

    These are experimental and are more liable to change, because the OpenQASM 3
    specification has not formally accepted them yet, so the syntax may not be finalized."""

    SWITCH_CASE_V1 = enum.auto()
    """Support exporting switch-case statements as proposed by
    https://github.com/openqasm/openqasm/pull/463 at `commit bfa787aa3078
    <https://github.com/openqasm/openqasm/pull/463/commits/bfa787aa3078>`__.

    These have the output format:

    .. code-block::

        switch (i) {
            case 0:
            case 1:
                x $0;
            break;

            case 2: {
                z $0;
            }
            break;

            default: {
                cx $0, $1;
            }
            break;
        }

    This differs from the syntax of the ``switch`` statement as stabilized.  If this flag is not
    passed, then the parser will instead output using the stabilized syntax, which would render the
    same example above as:

    .. code-block::

        switch (i) {
            case 0, 1 {
                x $0;
            }
            case 2 {
                z $0;
            }
            default {
                cx $0, $1;
            }
        }
    """
