#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""List deprecated decorators."""
from __future__ import annotations
from typing import cast
from re import findall
from pathlib import Path
from collections import OrderedDict, defaultdict
import ast
from datetime import datetime
import sys
import argparse
import requests


def short_path(path: Path) -> Path | None:
    """shorten the full path, when possible"""
    original_path = path
    while path is not None:
        if Path.cwd().is_relative_to(path):
            return original_path.relative_to(path)
        path = path.parent if path != path.parent else None
    return original_path


class Deprecation:
    """
    Root class for Deprecation classes.
    """

    @property
    def location_str(self) -> str:
        """String with the location of the deprecated decorator <filename>:<line number>"""
        return f"{self.filename}:{self.lineno}"


class DeprecationDecorator(Deprecation):
    """
    Deprecation decorator, representing a single deprecation decorator.

    Args:
        filename: where is the deprecation.
        deprecation_node: AST node of the decorator call.
        func_node: AST node of the decorated call.
    """

    def __init__(
        self, filename: Path, deprecation_node: ast.Call, func_node: ast.FunctionDef
    ) -> None:
        self.filename = filename
        self.deprecation_node = deprecation_node
        self.func_node = func_node
        self._since: str | None = None
        self._pending: bool | None = None

    @property
    def since(self) -> str | None:
        """Version since the deprecation applies."""
        if not self._since:
            for kwarg in self.deprecation_node.keywords:
                if kwarg.arg == "since":
                    self._since = ".".join(
                        str(cast(ast.Constant, kwarg.value).value).split(".")[:2]
                    )
        return self._since

    @property
    def pending(self) -> bool | None:
        """If it is a pending deprecation."""
        if not self._pending:
            self._pending = next(
                (
                    kwarg.value.value
                    for kwarg in self.deprecation_node.keywords
                    if kwarg.arg == "pending"
                ),
                False,
            )
        return self._pending

    @property
    def lineno(self) -> int:
        """Line number of the decorator."""
        return self.deprecation_node.lineno

    @property
    def target(self) -> str:
        """Name of the decorated function/method."""
        return self.func_node.name


class DeprecationCall(Deprecation):
    """
    Deprecation call, representing a single deprecation call.

    Args:
        decorator_call: deprecation call.
    """

    def __init__(self, filename: Path, decorator_call: ast.Call) -> None:
        self.filename = filename
        self.deprecation_node = decorator_call
        self.lineno = decorator_call.lineno
        self.pending: bool | None = None
        self._target: str | None = None
        self._since: str | None = None

    @property
    def target(self) -> str | None:
        """what's deprecated."""
        if not self._target:
            arg = self.deprecation_node.args.__getitem__(0)
            if isinstance(arg, ast.Attribute):
                self._target = f"{arg.value.id}.{arg.attr}"
            if isinstance(arg, ast.Name):
                self._target = arg.id
        return self._target

    @property
    def since(self) -> str | None:
        """Version since the deprecation applies."""
        if not self._since:
            for kwarg in self.deprecation_node.func.keywords:
                if kwarg.arg == "since":
                    self._since = ".".join(cast(ast.Constant, kwarg.value).value.split(".")[:2])
        return self._since


class DeprecationWarn(DeprecationDecorator):
    """
    Deprecation via manual warning

    Args:
        filename: where is the deprecation.
        deprecation_node: AST node of the decorator call.
        func_node: AST node of the decorated call.
    """

    @property
    def since(self) -> str | None:
        if not self._since:
            candidates = []
            for arg in self.deprecation_node.args:
                if isinstance(arg, ast.Constant):
                    candidates += [v.strip(".") for v in findall(r"\s+([\d.]+)", arg.value)]
            self._since = (min(candidates, default=0) or "n/a") + "?"
        return self._since

    @property
    def pending(self) -> bool | None:
        """If it is a pending deprecation."""
        if self._pending is None:
            self._pending = False
            for arg in self.deprecation_node.args:
                if hasattr(arg, "id") and arg.id == "PendingDeprecationWarning":
                    self._pending = True
        return self._pending


class DecoratorVisitor(ast.NodeVisitor):
    """
    Node visitor for finding deprecation decorator
    Args:
        filename: Name of the file to analyze
    """

    def __init__(self, filename: Path):
        self.filename = short_path(filename)
        self.deprecations: list[Deprecation] = []

    @staticmethod
    def is_deprecation_decorator(node: ast.expr) -> bool:
        """Check if a node is a deprecation decorator"""
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("deprecate_")
        )

    @staticmethod
    def is_deprecation_warning(node: ast.expr) -> bool:
        """Check if a node is a deprecation warning"""
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "warn"
        ):
            for arg in node.args:
                if hasattr(arg, "id") and "DeprecationWarning" in arg.id:
                    return True
        return False

    @staticmethod
    def is_deprecation_call(node: ast.expr) -> bool:
        """Check if a node is a deprecation call"""
        return (
            isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Name)
            and node.func.func.id.startswith("deprecate_")
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visitor for function declarations"""
        for d_node in node.decorator_list:
            if DecoratorVisitor.is_deprecation_decorator(d_node):
                self.deprecations.append(
                    DeprecationDecorator(self.filename, cast(ast.Call, d_node), node)
                )
        for stmt in ast.walk(node):
            if DecoratorVisitor.is_deprecation_warning(stmt):
                self.deprecations.append(DeprecationWarn(self.filename, stmt, node))
        ast.NodeVisitor.generic_visit(self, node)

    def visit_Call(self, node: ast.Call) -> None:
        """Visitor for function call"""
        if DecoratorVisitor.is_deprecation_call(node):
            self.deprecations.append(DeprecationCall(self.filename, node))
        ast.NodeVisitor.generic_visit(self, node)


class DeprecationCollection:
    """
    A collection of :class:~.Deprecation

    Args:
        dirname: Directory name that would be checked recursively for deprecations.
    """

    def __init__(self, dirname: Path):
        self.dirname = dirname
        self._deprecations: list[Deprecation] | None = None
        self.grouped: OrderedDict[str, list[Deprecation]] = OrderedDict()

    @property
    def deprecations(self) -> list[Deprecation]:
        """List of deprecation :class:~.Deprecation"""
        if self._deprecations is None:
            self.collect_deprecations()
        return cast(list, self._deprecations)

    def collect_deprecations(self) -> None:
        """Run the :class:~.DecoratorVisitor on `self.dirname` (in place)"""
        self._deprecations = []
        files = [self.dirname] if self.dirname.is_file() else self.dirname.rglob("*.py")
        for filename in files:
            self._deprecations.extend(DeprecationCollection.find_deprecations(filename))

    def group_by(self, attribute_idx: str) -> None:
        """Group :class:~`.Deprecation` in self.deprecations based on the attribute attribute_idx"""
        grouped = defaultdict(list)
        for obj in self.deprecations:
            grouped[getattr(obj, attribute_idx)].append(obj)
        for key in sorted(grouped.keys(), key=str):
            self.grouped[key] = grouped[key]

    @staticmethod
    def find_deprecations(file_name: Path) -> list[Deprecation]:
        """Runs the deprecation finder on file_name"""
        code = Path(file_name).read_text()
        mod = ast.parse(code, file_name)
        decorator_visitor = DecoratorVisitor(file_name)
        decorator_visitor.visit(mod)
        return decorator_visitor.deprecations


def print_main(directory: str, pending: str, format_: str) -> None:
    """Prints output"""
    collection = DeprecationCollection(Path(directory))
    collection.group_by("since")

    DATA_JSON = LAST_TIME_MINOR = DETAILS = None
    try:
        DATA_JSON = requests.get("https://pypi.org/pypi/qiskit/json", timeout=5).json()
    except requests.exceptions.ConnectionError:
        print("https://pypi.org/pypi/qiskit/json timeout...", file=sys.stderr)

    if DATA_JSON:
        LAST_MINOR = ".".join(DATA_JSON["info"]["version"].split(".")[:2])
        LAST_TIME_MINOR = datetime.fromisoformat(
            DATA_JSON["releases"][f"{LAST_MINOR}.0"][0]["upload_time"]
        )

    for since_version, deprecations in collection.grouped.items():
        if DATA_JSON and LAST_TIME_MINOR:
            try:
                release_minor_datetime = datetime.fromisoformat(
                    DATA_JSON["releases"][f"{since_version}.0"][0]["upload_time"]
                )
                release_minor_date = release_minor_datetime.strftime("%B %d, %Y")
                diff_days = (LAST_TIME_MINOR - release_minor_datetime).days
                DETAILS = f"Released in {release_minor_date}"
                if diff_days:
                    DETAILS += f" ({round(diff_days / 30.4)} month since the last minor release)"
            except KeyError:
                DETAILS = "Future release?"
        lines = []
        for deprecation in deprecations:
            if pending == "exclude" and deprecation.pending:
                continue
            if pending == "only" and not deprecation.pending:
                continue
            pending_arg = " - PENDING" if deprecation.pending else ""
            if format_ == "console":
                lines.append(f" - {deprecation.location_str} ({deprecation.target}){pending_arg}")
            if format_ == "md":
                lines.append(f" - `{deprecation.location_str}` (`{deprecation.target}`)")
        if format_ == "md":
            formatted_since_version = f"**{since_version or 'n/a'}**"
            DETAILS = ""
        if lines:
            print(f"\n{formatted_since_version}: {DETAILS}")
            print("\n".join(lines))


def create_parser() -> argparse.ArgumentParser:
    """Creates the ArgumentParser object"""
    default_directory = Path(__file__).joinpath("..", "..", "qiskit").resolve()
    parser = argparse.ArgumentParser(
        prog="find_deprecated",
        description="Finds the deprecation decorators in a path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--directory", default=default_directory, help="directory to search")
    parser.add_argument(
        "-p",
        "--pending",
        choices=["only", "include", "exclude"],
        default="exclude",
        help="show pending deprecations",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["console", "md"],
        default="console",
        help="format the output",
    )
    return parser


if __name__ == "__main__":
    args = create_parser().parse_args()
    print_main(args.directory, args.pending, args.format)
