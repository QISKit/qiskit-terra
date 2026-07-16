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

"""Shared utilities for QPY compatibility testing version management."""

import json
import re
import sys
import urllib.request

import packaging.tags
import packaging.version

import qiskit


def tags_from_wheel_name(wheel: str):
    """Extract the wheel tag from its filename."""
    assert wheel.lower().endswith(".whl")
    # For more information, see:
    # - https://packaging.python.org/en/latest/specifications/binary-distribution-format/
    # - https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/
    #
    # In particular, take note that a wheel's filename can include "compressed tag sets" (and our
    # Linux wheels generally do this), which is why this function returns an iterable of tags.
    _prefix, interpreters, abis, platforms = wheel[:-4].rsplit("-", 3)
    yield from (
        packaging.tags.Tag(interpreter, abi, platform)
        for interpreter in interpreters.split(".")
        for abi in abis.split(".")
        for platform in platforms.split(".")
    )


def fetch_installable_qiskit_versions(
    package="qiskit",
    min_version=None,
    max_version=None,
    no_docker=None,
):
    """Fetch Qiskit versions from PyPI that are installable on this platform.

    Args:
        package: Package name to query ("qiskit" or "qiskit-terra")
        min_version: Minimum version string (inclusive), or None for no minimum
        max_version: Maximum version string (exclusive), or None for no maximum
        no_docker: If None, checks sys.argv for "--no-docker" flag. If True/False, uses that value.

    Yields:
        Tuple of (version, python_version) where:
        - version: packaging.version.Version object
        - python_version: string like "3.12" (empty string in no-docker mode)
    """
    our_version = packaging.version.parse(qiskit.__version__)
    supported_tags = set(packaging.tags.sys_tags())

    if no_docker is None:
        no_docker = "--no-docker" in sys.argv

    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json") as fd:
        data = json.load(fd)

    min_version = min_version and packaging.version.parse(min_version)
    max_version = max_version and packaging.version.parse(max_version)

    for raw_version, payload in data["releases"].items():
        version = packaging.version.parse(raw_version)

        # Apply version filters
        if min_version is not None and version < min_version:
            continue
        if max_version is not None and version >= max_version:
            continue
        if version > our_version:
            continue
        if version.pre is not None and version.pre[0] in ("a", "b"):
            # Skip alpha and beta prereleases, but allow release candidates
            continue

        # Check if version is installable
        if no_docker:
            # Check if any wheel is compatible with current platform
            if not any(
                tag in supported_tags
                for release in payload
                if release["packagetype"] == "bdist_wheel" and not release["yanked"]
                for tag in tags_from_wheel_name(release["filename"])
            ):
                print(  # noqa: T201
                    f"skipping '{version}', which has no installable binary artifacts",
                    file=sys.stderr,
                )
                continue
            python_version = ""
        else:
            # Docker mode: determine which Python version to use
            try:
                python_versions = [
                    release["python_version"]
                    for release in payload
                    if release["packagetype"] == "bdist_wheel" and not release["yanked"]
                ]
                # Convert "cp311" to "3.11"
                python_versions = [re.sub(r"^cp(\d)(\d+)$", r"\1.\2", v) for v in python_versions]
                python_version = max(python_versions, key=lambda s: tuple(map(int, s.split("."))))
            except ValueError:
                print(  # noqa: T201
                    f"skipping '{version}', which has no installable binary artifacts",
                    file=sys.stderr,
                )
                continue

        yield (version, python_version)
