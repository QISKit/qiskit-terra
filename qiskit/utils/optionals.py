# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
.. currentmodule:: qiskit.utils.optionals

Qiskit has several features that are enabled only if certain *optional* dependencies
are satisfied. This module, :mod:`qiskit.utils.optionals`, has a collection of objects that can
be used to test if certain functionality is available, and optionally raise
:class:`.MissingOptionalLibraryError` if the functionality is not available.


Available Testers
=================

Qiskit Components
-----------------

.. py:data:: HAS_AER

    `Qiskit Aer <https://qiskit.github.io/qiskit-aer/>`__ provides high-performance simulators for
    the quantum circuits constructed within Qiskit.

.. py:data:: HAS_IBMQ

    The :mod:`Qiskit IBMQ Provider <qiskit.providers.ibmq>` is used for accessing IBM Quantum
    hardware in the IBM cloud.

.. py:data:: HAS_IGNIS

    :mod:`Qiskit Ignis <qiskit.ignis>` provides tools for quantum hardware verification, noise
    characterization, and error correction.

.. py:data:: HAS_TOQM

    `Qiskit TOQM <https://github.com/qiskit-toqm/qiskit-toqm>`__ provides transpiler passes
    for the `Time-optimal Qubit mapping algorithm <https://doi.org/10.1145/3445814.3446706>`__.


External Python Libraries
-------------------------

.. py:data:: HAS_CONSTRAINT

    `python-constraint <https://github.com/python-constraint/python-constraint>`__ is a
    constraint satisfaction problem solver, used in the :class:`~.CSPLayout` transpiler pass.

.. py:data:: HAS_CPLEX

    The `IBM CPLEX Optimizer <https://www.ibm.com/analytics/cplex-optimizer>`__ is a
    high-performance mathematical programming solver for linear, mixed-integer and quadratic
    programming. This is no longer used by Qiskit, but it was historically and the optional
    remains for backwards compatibility.

.. py:data:: HAS_CVXPY

    `CVXPY <https://www.cvxpy.org/>`__ is a Python package for solving convex optimization
    problems.  It is required for calculating diamond norms with
    :func:`.quantum_info.diamond_norm`.

.. py:data:: HAS_DOCPLEX

    `IBM Decision Optimization CPLEX Modelling
    <https://ibmdecisionoptimization.github.io/docplex-doc/>`__ is a library for prescriptive
    analysis.  Like CPLEX, this is no longer used by Qiskit, but it was historically and the
    optional remains for backwards compatibility.

.. py:data:: HAS_FIXTURES

    The test suite has additional features that are available if the optional `fixtures
    <https://launchpad.net/python-fixtures>`__ module is installed.  This generally also needs
    :data:`HAS_TESTTOOLS` as well.  This is generally only needed for Qiskit developers.

.. py:data:: HAS_IPYTHON

    If `the IPython kernel <https://ipython.org/>`__ is available, certain additional
    visualizations and line magics are made available.

.. py:data:: HAS_IPYWIDGETS

    Monitoring widgets for jobs running on external backends can be provided if `ipywidgets
    <https://ipywidgets.readthedocs.io/en/latest/>`__ is available.

.. py:data:: HAS_JAX

    Some methods of gradient calculation within :mod:`.opflow.gradients` require `JAX
    <https://github.com/google/jax>`__ for autodifferentiation.

.. py:data:: HAS_JUPYTER

    Some of the tests require a complete `Jupyter <https://jupyter.org/>`__ installation to test
    interactivity features.

.. py:data:: HAS_MATPLOTLIB

    Qiskit provides several visualization tools in the :mod:`.visualization` module.
    Almost all of these are built using `Matplotlib <https://matplotlib.org/>`__, which must
    be installed in order to use them.

.. py:data:: HAS_NETWORKX

    No longer used by Qiskit.  Internally, Qiskit now uses the high-performance `rustworkx
    <https://github.com/Qiskit/rustworkx>`__ library as a core dependency, and during the
    change-over period, it was sometimes convenient to convert things into the Python-only
    `NetworkX <https://networkx.org/>`__ format.  Some tests of application modules, such as
    `Qiskit Nature <https://qiskit-community.github.io/qiskit-nature/>`__ still use NetworkX.

.. py:data:: HAS_NLOPT

    `NLopt <https://nlopt.readthedocs.io/en/latest/>`__ is a nonlinear optimization library,
    used by the global optimizers in the :mod:`.algorithms.optimizers` module.

.. py:data:: HAS_PIL

    PIL is a Python image-manipulation library.  Qiskit actually uses the `pillow
    <https://pillow.readthedocs.io/en/stable/>`__ fork of PIL if it is available when generating
    certain visualizations, for example of both :class:`.QuantumCircuit` and
    :class:`.DAGCircuit` in certain modes.

.. py:data:: HAS_PYDOT

    For some graph visualizations, Qiskit uses `pydot <https://github.com/pydot/pydot>`__ as an
    interface to GraphViz (see :data:`HAS_GRAPHVIZ`).

.. py:data:: HAS_PYGMENTS

    Pygments is a code highlighter and formatter used by many environments that involve rich
    display of code blocks, including Sphinx and Jupyter.  Qiskit uses this when producing rich
    output for these environments.

.. py:data:: HAS_PYLATEX

    Various LaTeX-based visualizations, especially the circuit drawers, need access to the
    `pylatexenc <https://github.com/phfaist/pylatexenc>`__ project to work correctly.

.. py:data:: HAS_QASM3_IMPORT

    The functions :func:`.qasm3.load` and :func:`.qasm3.loads` for importing OpenQASM 3 programs
    into :class:`.QuantumCircuit` instances use `an external importer package
    <https://qiskit.github.io/qiskit-qasm3-import>`__.

.. py:data:: HAS_SEABORN

    Qiskit provides several visualization tools in the :mod:`.visualization` module.  Some
    of these are built using `Seaborn <https://seaborn.pydata.org/>`__, which must be installed
    in order to use them.

.. py:data:: HAS_SKLEARN

    Some of the gradient functions in :mod:`.opflow.gradients` use regularisation methods from
    `Scikit Learn <https://scikit-learn.org/stable/>`__.

.. py:data:: HAS_SKQUANT

    Some of the optimisers in :mod:`.algorithms.optimizers` are based on those found in `Scikit
    Quant <https://github.com/scikit-quant/scikit-quant>`__, which must be installed to use
    them.

.. py:data:: HAS_SQSNOBFIT

    `SQSnobFit <https://pypi.org/project/SQSnobFit/>`__ is a library for the "stable noisy
    optimization by branch and fit" algorithm.  It is used by the :class:`.SNOBFIT` optimizer.

.. py:data:: HAS_SYMENGINE

    `Symengine <https://github.com/symengine/symengine>`__ is a fast C++ backend for the
    symbolic-manipulation library `Sympy <https://www.sympy.org/en/index.html>`__.  This
    dependency is used to load legacy QPY formats, where this package was used to handle
    :class:`~.circuit.Parameter`\\ s.

.. py:data:: HAS_SYMPY

    `SymPy <https://www.sympy.org/en/index.html>`__ is a Python library for symbolic mathematics.
    ``SymPy`` was historically used for the implementation of the :class:`.ParameterExpression`
    class but isn't any longer. However it is needed for some legacy functionality that uses
    :meth:`.ParameterExpression.sympify`. It is also used in some visualization functions
    and template matching.

.. py:data:: HAS_TESTTOOLS

    Qiskit's test suite has more advanced functionality available if the optional
    `testtools <https://pypi.org/project/testtools/>`__ library is installed.  This is generally
    only needed for Qiskit developers.

.. py:data:: HAS_TWEEDLEDUM

    `Tweedledum <https://github.com/boschmitt/tweedledum>`__ is an extension library for
    synthesis and optimization of circuits that may involve classical oracles. In the past
    Qiskit's :class:`.PhaseOracle` used this but it is no longer used by Qiskit.

.. py:data:: HAS_Z3

    `Z3 <https://github.com/Z3Prover/z3>`__ is a theorem prover, used in the
    :class:`.CrosstalkAdaptiveSchedule` and :class:`.HoareOptimizer` transpiler passes.

External Command-Line Tools
---------------------------

.. py:data:: HAS_GRAPHVIZ

    For some graph visualizations, Qiskit uses the `GraphViz <https://graphviz.org/>`__
    visualization tool via its ``pydot`` interface (see :data:`HAS_PYDOT`).

.. py:data:: HAS_PDFLATEX

    Visualization tools that use LaTeX in their output, such as the circuit drawers, require
    ``pdflatex`` to be available.  You will generally need to ensure that you have a working
    LaTeX installation available, and the ``qcircuit.tex`` package.

.. py:data:: HAS_PDFTOCAIRO

    Visualization tools that convert LaTeX-generated files into rasterized images use the
    ``pdftocairo`` tool.  This is part of the `Poppler suite of PDF tools
    <https://poppler.freedesktop.org/>`__.


Lazy Checker Classes
====================

.. currentmodule:: qiskit.utils

Each of the lazy checkers is an instance of :class:`.LazyDependencyManager` in one of its two
subclasses: :class:`.LazyImportTester` and :class:`.LazySubprocessTester`.  These should be imported
from :mod:`.utils` directly if required, such as::

    from qiskit.utils import LazyImportTester

.. autoclass:: qiskit.utils.LazyDependencyManager
    :members:

.. autoclass:: qiskit.utils.LazyImportTester
.. autoclass:: qiskit.utils.LazySubprocessTester
"""

# NOTE: If you're changing this file, sync it with the `optionals-all` dependency group in
# `pyproject.toml` as well.

import logging as _logging

from .lazy_tester import (
    LazyImportTester as _LazyImportTester,
    LazySubprocessTester as _LazySubprocessTester,
)

_logger = _logging.getLogger(__name__)

HAS_AER = _LazyImportTester(
    "qiskit_aer",
    name="Qiskit Aer",
    install="pip install qiskit-aer",
)
HAS_IBMQ = _LazyImportTester(
    "qiskit.providers.ibmq",
    name="IBMQ Provider",
    install="pip install qiskit-ibmq-provider",
)
HAS_IGNIS = _LazyImportTester(
    "qiskit.ignis",
    name="Qiskit Ignis",
    install="pip install qiskit-ignis",
)
HAS_TOQM = _LazyImportTester("qiskit_toqm", name="Qiskit TOQM", install="pip install qiskit-toqm")

HAS_CONSTRAINT = _LazyImportTester(
    "constraint",
    name="python-constraint",
    install="pip install python-constraint",
)

HAS_CPLEX = _LazyImportTester(
    "cplex",
    install="pip install cplex",
    msg="This may not be possible for all Python versions and OSes",
)
HAS_CVXPY = _LazyImportTester("cvxpy", install="pip install cvxpy")
HAS_DOCPLEX = _LazyImportTester(
    {"docplex": (), "docplex.mp.model": ("Model",)},
    install="pip install docplex",
    msg="This may not be possible for all Python versions and OSes",
)
HAS_FIXTURES = _LazyImportTester("fixtures", install="pip install fixtures")
HAS_IPYTHON = _LazyImportTester("IPython", install="pip install ipython")
HAS_IPYWIDGETS = _LazyImportTester("ipywidgets", install="pip install ipywidgets")
HAS_JAX = _LazyImportTester(
    {"jax": ("grad", "jit"), "jax.numpy": ()},
    name="jax",
    install="pip install jax",
)
HAS_JUPYTER = _LazyImportTester(["jupyter", "nbformat", "nbconvert"], install="pip install jupyter")
HAS_MATPLOTLIB = _LazyImportTester(
    ("matplotlib.patches", "matplotlib.pyplot"),
    name="matplotlib",
    install="pip install matplotlib",
)
HAS_NETWORKX = _LazyImportTester("networkx", install="pip install networkx")

HAS_NLOPT = _LazyImportTester("nlopt", name="NLopt Optimizer", install="pip install nlopt")
HAS_PIL = _LazyImportTester("PIL.Image", name="pillow", install="pip install pillow")
HAS_PYDOT = _LazyImportTester("pydot", install="pip install pydot")
HAS_PYGMENTS = _LazyImportTester("pygments", install="pip install pygments")
HAS_PYLATEX = _LazyImportTester(
    {
        "pylatexenc.latex2text": ("LatexNodes2Text",),
        "pylatexenc.latexencode": ("utf8tolatex",),
    },
    name="pylatexenc",
    install="pip install pylatexenc",
)
HAS_QASM3_IMPORT = _LazyImportTester(
    "qiskit_qasm3_import", install="pip install qiskit_qasm3_import"
)
HAS_SEABORN = _LazyImportTester("seaborn", install="pip install seaborn")
HAS_SKLEARN = _LazyImportTester(
    {"sklearn.linear_model": ("Ridge", "Lasso")},
    name="scikit-learn",
    install="pip install scikit-learn",
)
HAS_SKQUANT = _LazyImportTester(
    "skquant.opt",
    name="scikit-quant",
    install="pip install scikit-quant",
)
HAS_SQSNOBFIT = _LazyImportTester("SQSnobFit", install="pip install SQSnobFit")
HAS_SYMENGINE = _LazyImportTester("symengine", install="pip install symengine<0.14")
HAS_SYMPY = _LazyImportTester("sympy", install="pip install sympy")
HAS_TESTTOOLS = _LazyImportTester("testtools", install="pip install testtools")
HAS_TWEEDLEDUM = _LazyImportTester("tweedledum", install="pip install tweedledum")
HAS_Z3 = _LazyImportTester("z3", install="pip install z3-solver")

HAS_GRAPHVIZ = _LazySubprocessTester(
    ("dot", "-V"),
    name="Graphviz",
    msg=(
        "To install, follow the instructions at https://graphviz.org/download/."
        " Qiskit needs the Graphviz binaries, which the 'graphviz' package on pip does not install."
        " You must install the actual Graphviz software"
    ),
)
HAS_PDFLATEX = _LazySubprocessTester(
    ("pdflatex", "-version"),
    msg="You will likely need to install a full LaTeX distribution for your system",
)
HAS_PDFTOCAIRO = _LazySubprocessTester(
    ("pdftocairo", "-v"),
    msg="This is part of the 'poppler' set of PDF utilities",
)
