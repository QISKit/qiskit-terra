# This code is part of Qiskit.
#
# (C) Copyright IBM 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from __future__ import annotations

import datetime
import doctest
import importlib
import inspect
import os
import re
from pathlib import Path

import qiskit

# pylint: disable=invalid-name,missing-function-docstring,missing-module-docstring


project = "Qiskit"
project_copyright = f"2017-{datetime.date.today().year}, Qiskit Development Team"
author = "Qiskit Development Team"

# The short X.Y version
version = ".".join(qiskit.__version__.split(".")[:2])
# The full version, including alpha/beta/rc tags
release = qiskit.__version__

language = "en"

rst_prolog = f".. |version| replace:: {version}"

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    # This is used by qiskit/documentation to generate links to github.com.
    "sphinx.ext.linkcode",
    "matplotlib.sphinxext.plot_directive",
    "reno.sphinxext",
    "sphinxcontrib.katex",
    "breathe",
]

breathe_projects = {"qiskit": "xml/"}
breathe_default_project = "qiskit"
breathe_domain_by_extension = {"h": "c"}
breathe_show_include = False

templates_path = ["_templates"]

# Number figures, tables and code-blocks if they have a caption.
numfig = True
# Available keys are 'figure', 'table', 'code-block' and 'section'.  '%s' is the number.
numfig_format = {"table": "Table %s"}

# Relative to source directory, affects general discovery, and html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]


# This adds the module name to e.g. function API docs. We use the default of True because our
# module pages sometimes have functions from submodules on the page, and we want to make clear
# that you must include the submodule to import it. We should strongly consider reorganizing our
# code to avoid this, i.e. re-exporting the submodule members from the top-level module. Once fixed
# and verified by only having a single `.. currentmodule::` in the file, we can turn this back to
# False.
add_module_names = True

# A list of prefixes that are ignored for sorting the Python module index
# (e.g., if this is set to ['foo.'], then foo.bar is shown under B, not F).
modindex_common_prefix = ["qiskit."]

# ----------------------------------------------------------------------------------
# Intersphinx
# ----------------------------------------------------------------------------------

intersphinx_mapping = {
    "rustworkx": ("https://www.rustworkx.org/", None),
    "qiskit-ibm-runtime": ("https://quantum.cloud.ibm.com/docs/api/qiskit-ibm-runtime/", None),
    "qiskit-aer": ("https://qiskit.github.io/qiskit-aer/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

# ----------------------------------------------------------------------------------
# HTML theme
# ----------------------------------------------------------------------------------

html_theme = "alabaster"
html_last_updated_fmt = "%Y/%m/%d"

# ----------------------------------------------------------------------------------
# Autodoc
# ----------------------------------------------------------------------------------

# Note that setting autodoc defaults here may not have as much of an effect as you may expect; any
# documentation created by autosummary uses a template file (in autosummary in the templates path),
# which likely overrides the autodoc defaults.

# These options impact when using `.. autoclass::` manually.
# They do not impact the `.. autosummary::` templates.
autodoc_default_options = {
    "show-inheritance": True,
}

# Move type hints from signatures to the parameter descriptions (except in overload cases, where
# that's not possible).
autodoc_typehints = "description"
# Tell Sphinx it does not need to attempt to parse `# type:` comments for type hints.  We don't use
# them for documentation, and having this set causes Sphinx to use a custom static-analysis checker
# to find type hints, which (as of Sphinx 9.1.0, at least) is not as reliable at finding type hints
# inherited from parent classes.
autodoc_use_type_comments = False
# Pull class documentation from both the class docstring and the `__init__`.
autoclass_content = "both"
# Some type hints are too long to be understandable. So, we set up aliases to be used instead.
autodoc_type_aliases = {
    "EstimatorPubLike": "EstimatorPubLike",
    "SamplerPubLike": "SamplerPubLike",
}

autosummary_generate = True
autosummary_generate_overwrite = False

# We only use Google-style docstrings, and allowing Napoleon to parse Numpy-style docstrings both
# slows down the build (a little) and can sometimes result in _regular_ section headings in
# module-level documentation being converted into surprising things.
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autosummary generates stub filenames based on the import name.
# Sometimes, two distinct interfaces only differ in capitalization; this
# creates a problem on case-insensitive OS/filesystems like macOS. So,
# we manually avoid the clash by renaming one of the files.
autosummary_filename_map = {
    "qiskit.circuit.library.iqp": "qiskit.circuit.library.iqp_function",
}


# ----------------------------------------------------------------------------------
# Doctest
# ----------------------------------------------------------------------------------

doctest_default_flags = (
    doctest.ELLIPSIS
    | doctest.NORMALIZE_WHITESPACE
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.DONT_ACCEPT_TRUE_FOR_1
)

# Leaving this string empty disables testing of doctest blocks from docstrings.
# Doctest blocks are structures like this one:
# >> code
# output
doctest_test_doctest_blocks = ""


# ----------------------------------------------------------------------------------
# Plot directive
# ----------------------------------------------------------------------------------

plot_html_show_formats = False


# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    module_name = info["module"]
    if "qiskit" not in module_name:
        return None

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    # Unwrap decorators. This requires they used `functools.wrap()`.
    while hasattr(obj, "__wrapped__"):
        obj = getattr(obj, "__wrapped__")

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None:
        return None
    try:
        relative_file_name = Path(full_file_name).resolve().relative_to(REPO_ROOT)
        file_name = re.sub(r"\.tox\/.+\/site-packages\/", "", relative_file_name.as_posix())
    except ValueError:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"

    github_branch = os.environ.get("QISKIT_DOCS_GITHUB_BRANCH_NAME", "main")
    return f"https://github.com/Qiskit/qiskit/tree/{github_branch}/{file_name}{linespec}"
