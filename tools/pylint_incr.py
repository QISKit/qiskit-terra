#!/usr/bin/env python3

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Run pylint incrementally on only changed files"""

import subprocess
import argparse
import os
import sys

from pylint import lint

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ["SYSTEMROOT", "PATH"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env["LANGUAGE"] = "C"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=os.path.join(os.path.dirname(ROOT_DIR)),
    ) as proc:
        stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr


def _run_pylint(ref, paths, pylint_args):
    code, stdout, stderr = _minimal_ext_cmd(
        [
            "git",
            "diff-index",
            "--name-only",
            "--diff-filter=d",
            "--merge-base",
            "-z",
            ref,
            "--",
            *paths,
        ]
    )
    if code != 0:
        print(
            f"{__file__}: unable to get list of changed files. Git returncode: {code}\n"
            f"Git must be installed, and you need to be in a git tree with a ref `{ref}`\n"
            f"{stderr.strip().decode('ascii')}"
        )
        sys.exit(128)
    changed_paths = [path.decode("ascii") for path in stdout.split(b"\x00") if len(path) > 0]
    if len(changed_paths) == 0:
        print(f"No changed files in {' '.join(paths)}")
        sys.exit(0)
    changed_paths_pretty = "\n    ".join(changed_paths)
    print(f"Running pylint on {len(changed_paths)} changed files:\n    {changed_paths_pretty}")
    lint.Run([*pylint_args, "--", *changed_paths])


def _main():
    parser = argparse.ArgumentParser(
        description="Incremental pylint.",
        epilog="Unknown arguments passed through to pylint",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--paths",
        required=True,
        type=str,
        nargs="+",
        help="Git <pathspec>s to resolve (and pass any changed files to pylint)",
    )
    args, pylint_args = parser.parse_known_args()
    _run_pylint("lint_incr_latest", args.paths, pylint_args)


if __name__ == "__main__":
    _main()
