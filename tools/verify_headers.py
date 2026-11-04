#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
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
import os
import pathlib
import sys
import re

# regex for character encoding from PEP 263
pep263 = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-_.a-zA-Z0-9]+)")
line_start = re.compile(r"^(\/\/|#) This code is part of Qiskit.$")
copyright_line = re.compile(r"^(\/\/|#) \(C\) Copyright IBM 20")


def discover_files(code_paths):
    """Find all .py, .rs, .c, and .h files in a list of trees"""
    return [
        file
        for extension in ("py", "rs", "c", "h")
        for path in code_paths
        for file in pathlib.Path(path).glob(f"**/*.{extension}")
        # CMake generates some files inside the tree.
        if not file.is_relative_to("test/c/build")
    ]


def validate_header(file_path):
    """Validate the header for a single file"""
    header = """# This code is part of Qiskit.
#
"""
    apache_text = """#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
    header_slashes = """// This code is part of Qiskit.
//
"""
    apache_text_slashes = """//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
"""

    count = 0
    with open(file_path, encoding="utf8") as fd:
        lines = fd.readlines()
    start = 0
    for index, line in enumerate(lines):
        count += 1
        if count > 5:
            return file_path, False, "Header not found in first 5 lines"
        if count <= 2 and pep263.match(line):
            return file_path, False, "Unnecessary encoding specification (PEP 263, 3120)"
        if line_start.search(line):
            start = index
            break

    if file_path.suffix in (".rs", ".c", ".h"):
        if "".join(lines[start : start + 2]) != header_slashes:
            return (file_path, False, f"Header up to copyright line does not match: {header}")
        if not copyright_line.search(lines[start + 2]):
            return (file_path, False, "Header copyright line not found")
        if "".join(lines[start + 3 : start + 11]) != apache_text_slashes:
            return (file_path, False, f"Header apache text string doesn't match:\n {apache_text}")
    else:  # .py ending
        if "".join(lines[start : start + 2]) != header:
            return (file_path, False, f"Header up to copyright line does not match: {header}")
        if not copyright_line.search(lines[start + 2]):
            return (file_path, False, "Header copyright line not found")
        if "".join(lines[start + 3 : start + 11]) != apache_text:
            return (file_path, False, f"Header apache text string doesn't match:\n {apache_text}")
    return (file_path, True, None)


def _main():
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qiskit"
    )
    parser = argparse.ArgumentParser(description="Check file headers.")
    parser.add_argument(
        "paths",
        type=str,
        nargs="*",
        default=[default_path],
        help="Paths to scan by default uses ../qiskit from the script",
    )
    args = parser.parse_args()
    files = discover_files(args.paths)
    with multiprocessing.Pool() as pool:
        res = pool.map(validate_header, files)
    failed_files = [x for x in res if x[1] is False]
    if len(failed_files) > 0:
        for failed_file in failed_files:
            sys.stderr.write(f"{failed_file[0]} failed header check because:\n")
            sys.stderr.write(f"{failed_file[2]}\n\n")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    _main()
