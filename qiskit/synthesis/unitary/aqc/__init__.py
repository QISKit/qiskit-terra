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

r"""
=====================================================================
Approximate Quantum Compiler (:mod:`qiskit.synthesis.unitary.aqc`)
=====================================================================

.. currentmodule:: qiskit.synthesis.unitary.aqc

Implementation of Approximate Quantum Compiler as described in the paper [1].

Interface
=========

The main public interface of this module is reached by passing ``unitary_synthesis_method='aqc'`` to
:func:`~.compiler.transpile`.  This will swap the synthesis method to use
:class:`~.transpiler.passes.synthesis.AQCSynthesisPlugin`.
The individual classes are:

.. autosummary::
    :toctree: ../stubs
    :template: autosummary/class_no_inherited_members.rst

    AQC
    ApproximateCircuit
    ApproximatingObjective
    CNOTUnitCircuit
    CNOTUnitObjective
    DefaultCNOTUnitObjective
    FastCNOTUnitObjective


Mathematical Detail
===================

We are interested in compiling a quantum circuit, which we formalize as finding the best
circuit representation in terms of an ordered gate sequence of a target unitary matrix
:math:`U\in U(d)`, with some additional hardware constraints. In particular, we look at
representations that could be constrained in terms of hardware connectivity, as well
as gate depth, and we choose a gate basis in terms of CNOT and rotation gates.
We recall that the combination of CNOT and rotation gates is universal in :math:`SU(d)` and
therefore it does not limit compilation.

To properly define what we mean by best circuit representation, we define the metric
as the Frobenius norm between the unitary matrix of the compiled circuit :math:`V` and
the target unitary matrix :math:`U`, i.e., :math:`\|V - U\|_{\mathrm{F}}`. This choice
is motivated by mathematical programming considerations, and it is related to other
formulations that appear in the literature. Let's take a look at the problem in more detail.

Let :math:`n` be the number of qubits and :math:`d=2^n`. Given a CNOT structure :math:`ct`
and a vector of rotation angles :math:`\theta`, the parametric circuit forms a matrix
:math:`Vct(\theta)\in SU(d)`. If we are given a target circuit forming a matrix
:math:`U\in SU(d)`, then we would like to compute

.. math::

    \mathrm{argmax}_{\theta}\frac{1}{d}|\langle Vct(\theta),U\rangle|

where the inner product is the Frobenius inner product. Note that
:math:`|\langle V,U\rangle|\leq d` for all unitaries :math:`U` and :math:`V`, so the objective
has range in :math:`[0,1]`.

Our strategy is to maximize

.. math::

    \frac{1}{d}\Re \langle Vct(\theta),U\rangle

using its gradient. We will now discuss the specifics by going through an example.

While the range of :math:`Vct` is a subset of :math:`SU(d)` by construction, the target
circuit may form a general unitary matrix. However, for any :math:`U\in U(d)`,

.. math::

    \frac{\exp(2\pi i k/d)}{\det(U)^{1/d}}U\in SU(d)\text{ for all }k\in\{0,\ldots,d-1\}.

Thus, we should normalize the target circuit by its global phase and then approximately
compile the normalized circuit. We can add the global phase back in afterwards.

In the algorithm let :math:`U'` denote the un-normalized target matrix and :math:`U`
the normalized  target matrix. Now that we have :math:`U`, we give the gradient function
to the Nesterov's method optimizer and compute :math:`\theta`.

To add the global phase back in, we can form the control circuit as

.. math::

    \frac{\langle Vct(\theta),U'\rangle}{|\langle Vct(\theta),U'\rangle|}Vct(\theta).

Note that while we optimized using Nesterov's method in the paper, this was for its convergence
guarantees, not its speed in practice. It is much faster to use L-BFGS which is used as a
default optimizer in this implementation.

A basic usage of the AQC algorithm should consist of the following steps::

    # Define a target circuit as a unitary matrix
    unitary = ...

    # Define a number of qubits for the algorithm, at least 3 qubits
    num_qubits = round(math.log2(unitary.shape[0]))

    # Choose a layout of the CNOT structure for the approximate circuit, e.g. ``spin`` for
    # a linear layout.
    layout = options.get("layout") or "spin"

    # Choose a connectivity type, e.g. ``full`` for full connectivity between qubits.
    connectivity = options.get("connectivity") or "full"

    # Define a targeted depth of the approximate circuit in the number of CNOT units.
    depth = int(options.get("depth") or 0)

    # Generate a network made of CNOT units
    cnots = make_cnot_network(
        num_qubits=num_qubits,
        network_layout=layout,
        connectivity_type=connectivity,
        depth=depth
    )

    # Create an optimizer to be used by AQC
    optimizer = partial(scipy.optimize.minimize, method="L-BFGS-B")

    # Create an instance
    aqc = AQC(optimizer)

    # Create a template circuit that will approximate our target circuit
    approximate_circuit = CNOTUnitCircuit(num_qubits=num_qubits, cnots=cnots)

    # Create an objective that defines our optimization problem
    approximating_objective = DefaultCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)

    # Run optimization process to compile the unitary
    aqc.compile_unitary(
        target_matrix=unitary,
        approximate_circuit=approximate_circuit,
        approximating_objective=approximating_objective
    )

Now ``approximate_circuit`` is a circuit that approximates the target unitary to a certain
degree and can be used instead of the original matrix.

This uses a helper function, :obj:`make_cnot_network`.

.. autofunction:: make_cnot_network

One can take advantage of accelerated version of objective function. It implements the same
mathematical algorithm as the default one ``DefaultCNOTUnitObjective`` but runs several times
faster. Instantiation of accelerated objective function class is similar to the default case:

    # Create an objective that defines our optimization problem
    approximating_objective = FastCNOTUnitObjective(num_qubits=num_qubits, cnots=cnots)

The rest of the code in the above example does not change.

References:

    [1]: Liam Madden, Andrea Simonetto, Best Approximate Quantum Compiling Problems.
        `arXiv:2106.05649 <https://arxiv.org/abs/2106.05649>`_
"""

from .approximate import ApproximateCircuit, ApproximatingObjective
from .aqc import AQC
from .cnot_structures import make_cnot_network
from .cnot_unit_circuit import CNOTUnitCircuit
from .cnot_unit_objective import CNOTUnitObjective, DefaultCNOTUnitObjective
from .fast_gradient.fast_gradient import FastCNOTUnitObjective

__all__ = [
    "AQC",
    "ApproximateCircuit",
    "ApproximatingObjective",
    "CNOTUnitCircuit",
    "CNOTUnitObjective",
    "DefaultCNOTUnitObjective",
    "FastCNOTUnitObjective",
    "make_cnot_network",
]
