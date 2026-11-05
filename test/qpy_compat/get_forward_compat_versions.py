# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""List the (qpy_version, qiskit_version, python_versions) triplets for forward-compatibility QPY testing.

For each QPY format version, output all the qiskit versions supporting loading from that QPY version
"""

from qpy_version_utils import fetch_installable_qiskit_versions

# Minimum Qiskit minor series that can read each supported QPY format version.
MIN_VERSION_STR = "1.4"
QPY_MIN_COMPAT_VERSION = {
    13: (1, 4),
    14: (2, 0),
    15: (2, 1),
    16: (2, 2),
    17: (2, 3),
}


def get_qiskit_versions():
    """For each minor series referenced in QPY_MIN_COMPAT_VERSION, find the latest
    available patch release on PyPI that is installable on this platform."""
    results = {qpy_version: [] for qpy_version in QPY_MIN_COMPAT_VERSION}
    for version, python_version in fetch_installable_qiskit_versions(
        package="qiskit",
        min_version=MIN_VERSION_STR,
        max_version=None,
    ):
        found_minor_version = (version.major, version.minor)
        # Add this version to all QPY versions it can read
        for qpy_version, minimum_minor in QPY_MIN_COMPAT_VERSION.items():
            if found_minor_version >= minimum_minor:
                results[qpy_version].append(version)

    return results


def main():
    results = get_qiskit_versions()
    for qpy_version, supported_versions in results.items():
        for supported_version in supported_versions:
            print(f"{qpy_version} {supported_version}")  # noqa: T201


if __name__ == "__main__":
    main()
