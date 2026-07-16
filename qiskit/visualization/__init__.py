# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
============================================
Visualizations (:mod:`qiskit.visualization`)
============================================

.. currentmodule:: qiskit.visualization

The visualization module contain functions that visualizes measurement outcome counts, quantum
states, circuits, devices and more.

To use visualization functions, you are required to install visualization optionals to your
development environment:

.. code-block:: bash

   pip install 'qiskit[visualization]'

.. warning::

    In general, the visualization tooling in Qiskit may call system programs on arbitrary user
    input, and should only be used on trusted inputs.  The visualization tooling provided in Qiskit
    is mostly intended for local visualization, and deliberately allows user code injection in
    several places, such as Graphviz node labels like register names in :meth:`.DAGCircuit.draw`, or
    LaTeX instruction labels from :attr:`.Instruction.label` in the ``latex`` mode of
    :meth:`.QuantumCircuit.draw`.

Common Keyword Arguments
========================

Many of the figures created by visualization functions in this module are created by `Matplotlib
<https://matplotlib.org/>`_ and accept a subset of the following common arguments. Consult the
individual documentation for exact details.

* ``title`` (``str``): a text string to use for the plot title.
* ``legend`` (``list``): a list of strings to use for labels of the data.
* ``figsize`` (``tuple``): figure size in inches .
* ``color`` (``list``): a list of strings for plotting.
* ``ax`` (`matplotlib.axes.Axes <https://matplotlib.org/stable/api/axes_api.html>`_): An optional
  ``Axes`` object to be used for the visualization output. If none is specified a new
  `matplotlib.figure.Figure <https://matplotlib.org/stable/api/figure_api.html>`_ will be created
  and used. Additionally, if specified there will be no returned ``Figure`` since it is redundant.
* ``filename`` (``str``): file path to save image to.

The following example demonstrates the common usage of these arguments:

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit.visualization import plot_histogram

   counts1 = {'00': 499, '11': 501}
   counts2 = {'00': 511, '11': 489}

   data = [counts1, counts2]
   plot_histogram(data)

You can specify ``legend``, ``title``, ``figsize`` and ``color`` by passing to the kwargs.

.. plot::
   :alt: Output from the previous code.
   :include-source:
   :context: reset

   from qiskit.visualization import plot_histogram

   counts1 = {'00': 499, '11': 501}
   counts2 = {'00': 511, '11': 489}
   data = [counts1, counts2]

   legend = ['First execution', 'Second execution']
   title = 'New histogram'
   figsize = (10,10)
   color=['crimson','midnightblue']
   plot_histogram(data, legend=legend, title=title, figsize=figsize, color=color)

You can save the figure to file either by passing the file name to ``filename`` kwarg or use
`matplotlib.figure.Figure.savefig
<https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.savefig>`_ method.

.. code-block:: python

   plot_histogram(data, filename='new_hist.png')

   hist = plot_histogram(data)
   hist.savefig('new_hist.png')


Counts Visualizations
=====================

This section contains functions that visualize measurement outcome counts.

.. autosummary::
   :toctree: ../stubs/

   plot_histogram

Example Usage
-------------

Here is an example of using :func:`plot_histogram` to visualize measurement outcome counts:

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit.visualization import plot_histogram

   counts = {"00": 501, "11": 499}
   plot_histogram(counts)

The data can be a dictionary with bit string as key and counts as value, or more commonly a
:class:`~qiskit.result.Counts` object obtained from :meth:`~qiskit.result.Result.get_counts`.

Distribution Visualizations
===========================

This section contains functions that visualize sampled distributions.

.. autosummary::
   :toctree: ../stubs/

   plot_distribution

State Visualizations
====================

This section contains functions that visualize quantum states.

.. autosummary::
   :toctree: ../stubs/

   plot_bloch_vector
   plot_bloch_multivector
   plot_state_city
   plot_state_hinton
   plot_state_paulivec
   plot_state_qsphere

Example Usage
-------------

Here is an example of using :func:`plot_state_city` to visualize a quantum state:

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit.visualization import plot_state_city

   state = [[ 0.75  , 0.433j],
            [-0.433j, 0.25  ]]
   plot_state_city(state)

The state can be array-like list of lists, ``numpy.array``, or more commonly
:class:`~qiskit.quantum_info.Statevector` or :class:`~qiskit.quantum_info.DensityMatrix` objects
obtained from a :class:`~qiskit.circuit.QuantumCircuit`:

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit import QuantumCircuit
   from qiskit.quantum_info import Statevector
   from qiskit.visualization import plot_state_city

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0,1)

   # plot using a Statevector
   state = Statevector(qc)
   plot_state_city(state)

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit import QuantumCircuit
   from qiskit.quantum_info import DensityMatrix
   from qiskit.visualization import plot_state_city

   qc = QuantumCircuit(2)
   qc.h(0)
   qc.cx(0,1)

   # plot using a DensityMatrix
   state = DensityMatrix(qc)
   plot_state_city(state)

You can find code examples for each visualization functions on the individual function API page.

Device Visualizations
=====================

.. autosummary::
   :toctree: ../stubs/

   plot_gate_map
   plot_error_map
   plot_circuit_layout
   plot_coupling_map

Circuit Visualizations
======================

.. autosummary::
   :toctree: ../stubs/

   circuit_drawer

DAG Visualizations
==================

.. autosummary::
   :toctree: ../stubs/

   dag_drawer

Pass Manager Visualizations
===========================

.. autosummary::
   :toctree: ../stubs/

   pass_manager_drawer

Timeline Visualizations
=======================

.. autosummary::
   :toctree: ../stubs/

   timeline_drawer

Single Qubit State Transition Visualizations
============================================

.. autosummary::
   :toctree: ../stubs/

   visualize_transition

Array/Matrix Visualizations
===========================

.. autosummary::
   :toctree: ../stubs/

   array_to_latex

Exceptions
==========

.. autoexception:: VisualizationError
"""


from .array import array_to_latex

from .circuit import circuit_drawer
from .counts_visualization import plot_histogram, plot_distribution
from .state_visualization import (
    plot_state_hinton,
    plot_bloch_vector,
    plot_bloch_multivector,
    plot_state_city,
    plot_state_paulivec,
    plot_state_qsphere,
    state_drawer,
)
from .transition_visualization import visualize_transition
from .dag_visualization import dag_drawer
from .gate_map import plot_gate_map, plot_circuit_layout, plot_error_map, plot_coupling_map
from .pass_manager_visualization import pass_manager_drawer
from .pass_manager_visualization import staged_pass_manager_drawer

from .timeline import draw as timeline_drawer

from .exceptions import VisualizationError

# These modules aren't part of the public interface, and were moved in Terra 0.22.  They're
# re-imported here to allow a backwards compatible path, and should be deprecated in Terra 0.23.
from .circuit import text, matplotlib, latex

__all__ = [
    "VisualizationError",
    "array_to_latex",
    "circuit_drawer",
    "dag_drawer",
    "draw",
    "latex",
    "matplotlib",
    "pass_manager_drawer",
    "plot_bloch_multivector",
    "plot_bloch_vector",
    "plot_circuit_layout",
    "plot_coupling_map",
    "plot_distribution",
    "plot_error_map",
    "plot_gate_map",
    "plot_histogram",
    "plot_state_city",
    "plot_state_hinton",
    "plot_state_paulivec",
    "plot_state_qsphere",
    "staged_pass_manager_drawer",
    "state_drawer",
    "text",
    "timeline_drawer",
    "visualize_transition",
]
