# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""List the versions of Qiskit and Terra that should be tested for QPY compatibility."""

from qpy_version_utils import fetch_installable_qiskit_versions


def available_versions():
    """Get all the versions of Qiskit that support exporting QPY, and are installable with the
    active version of Python on this platform."""
    yield from (
        ("qiskit-terra", version, python_version)
        for version, python_version in fetch_installable_qiskit_versions(
            "qiskit-terra", "0.18.0", "1.0.0"
        )
    )
    yield from (
        ("qiskit", version, python_version)
        for version, python_version in fetch_installable_qiskit_versions("qiskit", "1.0.0")
    )


def main():
    """main"""
    for package, version, python_version in available_versions():
        print(package, version, python_version)  # noqa: T201


if __name__ == "__main__":
    main()
