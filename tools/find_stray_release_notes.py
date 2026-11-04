#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to verify qiskit copyright file headers"""

import argparse
import multiprocessing
import subprocess
import sys
import re

# release notes regex
reno = re.compile(r"releasenotes\/notes")
# exact release note regex
exact_reno = re.compile(r"^releasenotes\/notes")


def discover_files():
    """Find all .py, .pyx, .pxd files in a list of trees"""
    cmd = ["git", "ls-tree", "-r", "--name-only", "HEAD"]
    res = subprocess.run(cmd, capture_output=True, check=True, encoding="UTF8")
    files = res.stdout.split("\n")
    return files


def validate_path(file_path):
    """Validate a path in the git tree."""
    if reno.search(file_path) and not exact_reno.search(file_path):
        return file_path
    return None


def _main():
    parser = argparse.ArgumentParser(description="Find any stray release notes.")
    _args = parser.parse_args()
    files = discover_files()
    with multiprocessing.Pool() as pool:
        res = pool.map(validate_path, files)
    failed_files = [x for x in res if x is not None]
    if len(failed_files) > 0:
        for failed_file in failed_files:
            sys.stderr.write(f"{failed_file} is not in the correct location.\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    _main()
