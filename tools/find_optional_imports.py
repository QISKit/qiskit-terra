#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Utility to check that slow imports are not used in the default path."""

import subprocess
import sys

# This is not unused: importing it sets up sys.modules
import qiskit  # noqa: F401


def _main():
    optional_imports = [
        "networkx",
        "sympy",
        "pydot",
        "pygments",
        "ipywidgets",
        "scipy.linalg",
        "scipy.optimize",
        "scipy.stats",
        "matplotlib",
        "qiskit_aer",
        "qiskit.providers.ibmq",
        "qiskit.ignis",
        "qiskit.aqua",
        "docplex",
    ]

    modules_imported = []
    for mod in optional_imports:
        if mod in sys.modules:
            modules_imported.append(mod)

    if not modules_imported:
        sys.exit(0)

    res = subprocess.run(
        [sys.executable, "-X", "importtime", "-c", "import qiskit"],
        capture_output=True,
        encoding="utf8",
        check=True,
    )

    import_tree = [
        x.split("|")[-1]
        for x in res.stderr.split("\n")
        if "RuntimeWarning" not in x or "warnings.warn" not in x
    ]

    indent = -1
    matched_module = None
    for module in import_tree:
        line_indent = len(module) - len(module.lstrip())
        module_name = module.strip()
        if module_name in modules_imported:
            if indent > 0:
                continue
            indent = line_indent
            matched_module = module_name
        if indent > 0:
            if line_indent < indent:
                print(f"ERROR: {matched_module} is imported via {module_name}")
                indent = -1
                matched_module = None

    sys.exit(len(modules_imported))


if __name__ == "__main__":
    _main()
