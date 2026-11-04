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
================================================================================
Fast implementation of objective function class
(:mod:`qiskit.synthesis.unitary.aqc.fast_gradient`)
================================================================================

.. currentmodule:: qiskit.synthesis.unitary.aqc.fast_gradient

Extension to the implementation of Approximate Quantum Compiler as described in the paper [1].

Interface
=========

The main public class of this module is FastCNOTUnitObjective. It replaces the default objective
function implementation :class:`.DefaultCNOTUnitObjective` for faster computation.
The individual classes include the public one (FastCNOTUnitObjective) and few
internal ones:

.. autosummary::
    :toctree: ../stubs
    :template: autosummary/class_no_inherited_members.rst

    FastCNOTUnitObjective
    LayerBase
    Layer1Q
    Layer2Q
    PMatrix


Mathematical Details
====================

In what follows we briefly outline the main ideas underlying the accelerated implementation
of objective function class.

* The key ingredient of approximate compiling is the efficient optimization procedure
  that minimizes :math:`\|V - U\|_{\mathrm{F}}` on a classical computer, where :math:`U`
  is a given (target) unitary matrix and :math:`V` is a matrix of approximating quantum
  circuit. Alternatively, we maximize the Hilbert-Schmidt product between :math:`U` and
  :math:`V` as outlined in the main part of the documentation.

* The circuit :math:`V` can be represented as a sequence of 2-qubit gates (layers)
  applied one after another. The corresponding matrix takes the form:
  :math:`V = C_0 C_1 \ldots C_{L-1} F`, where :math:`L` is the length of the sequence
  (number of layers). If the total number of qubits :math:`n > 2`, every
  :math:`C_i = C_i(\Theta_i)` is a sparse, :math:`2^n \times 2^n` matrix of 2-qubit gate
  (CNOT unit block) parameterized by a sub-set of parameters :math:`\Theta_i`
  (4 parameters per unit block), and :math:`F` is a matrix that comprises the action
  of all 1-qubit gates in front of approximating circuit. See the paper [1] for details.

* Over the course of optimization we compute the value of objective function and its
  gradient, which implies computation of :math:`V` and its derivatives
  :math:`{\partial V}/{\partial \Theta_i}` for all :math:`i`, given the current estimation
  of all the parameters :math:`\Theta`.

* A naive implementation of the product :math:`V = C_0 C_1 \ldots C_{L-1} F` and its
  derivatives would include computation and memorization of forward and backward partial
  products as required by the backtracking algorithm. This is wasteful in terms of
  performance and resource allocation.

* Minimization of :math:`\|V - U\|_{\mathrm{F}}^2` is equivalent to maximization of
  :math:`\text{Re}\left(\text{Tr}\left(U^{\dagger} V\right)\right)`. By cyclic permutation
  of the sequence of matrices under trace operation, we can avoid memorization of intermediate
  partial products of gate matrices :math:`C_i`. Note, matrix size grows exponentially with
  the number of qubits, quickly becoming prohibitively large.

* Sparse structure of :math:`C_i` can be exploited to speed up matrix-matrix multiplication.
  However, using sparse matrices as such does not give performance gain because sparse patterns
  tend to violate data proximity inside the cache memory of modern CPUs. Instead, we make use
  of special structure of gate matrices :math:`C_i` coupled with permutation ones. Although
  permutation is not cache friendly either, its impact is seemingly less severe than that
  of sparse matrix multiplication (at least in Python implementation).

* On every optimization iteration we, first, compute :math:`V = C_0 C_1 \ldots C_{L-1} F`
  given the current estimation of all the parameters :math:`\Theta`.

* As for the gradient of objective function, it can be shown (by moving cyclically around
  an individual matrices under trace operation) that:

.. math::
    \text{Tr}\left( U^{\dagger} \frac{\partial V}{\partial \Theta_{l,k}} \right) =
    \langle \text{vec}\left(E_l\right), \text{vec}\left(
    \frac{\partial C_l}{\partial \Theta_{l,k}}\right) \rangle,

where :math:`\Theta_{l,k}` is a :math:`k`-th parameter of :math:`l`-th CNOT unit block,
and :math:`E_l=C_{l-1}\left(C_{l-2}\left(\cdots\left(C_0\left(U^{\dagger}V
C_0^{\dagger}\right)C_1^{\dagger}\right) \cdots\right)C_{l-1}^{\dagger}\right)C_l^{\dagger}`
is an intermediate matrix.

* For every :math:`l`-th gradient component, we compute the trace using the matrix
  :math:`E_l`, then this matrix is updated by multiplication on left and on the right
  by corresponding gate matrices :math:`C_l` and :math:`C_{l+1}^{\dagger}` respectively
  and proceed to the next gradient component.

* We save computations and resources by not storing intermediate partial products of
  :math:`C_i`.  Instead, incrementally updated matrix :math:`E_l` keeps all related
  information. Also, vectorization of involved matrices (see the above formula) allows
  us to replace matrix-matrix multiplication by "cheaper" vector-vector one under the
  trace operation.

* The matrices :math:`C_i` are sparse. However, even for relatively small matrices
  (< 1M elements) sparse-dense multiplication can be very slow. Construction of sparse
  matrices takes a time as well.  We should update every gate matrix on each iteration
  of optimization loop.

* In fact, any gate matrix :math:`C_i` can be transformed to what we call a standard
  form: :math:`C_i = P^T \widetilde{C}_i P`, where :math:`P` is an easily computable
  permutation matrix and :math:`\widetilde{C}_i` has a block-diagonal layout:

.. math::
    \widetilde{C}_i = \left(
    \begin{array}{ccc}
    G_{4 \times 4} & \ddots & 0 \\
    \ddots & \ddots & \ddots \\
    0 & \ddots & G_{4 \times 4}
    \end{array}
    \right)

* The 2-qubit gate matrix :math:`G_{4 \times 4}` is repeated along diagonal of the full
  :math:`2^n \times 2^n` :math:`\widetilde{C}_i`.

* We do not actually create neither matrix :math:`\widetilde{C}_i` nor :math:`P`.
  In fact, only :math:`G_{4 \times 4}` and a permutation array (of size :math:`2^n`)
  are kept in memory.

* Consider left-hand side multiplication by some dense, :math:`2^n \times 2^n` matrix :math:`M`:

.. math::
    C_i M = P^T \widetilde{C}_i P M = P^T \left( \widetilde{C}_i \left( P M \right) \right)

* First, we permute rows of :math:`M`, which is equivalent to the product :math:`P M`, but
  without expensive multiplication of two :math:`2^n \times 2^n` matrices.

* Second, we compute :math:`\widetilde{C}_i P M` multiplying every block-diagonal sub-matrix
  :math:`G_{4 \times 4}` by the corresponding rows of :math:`P M`. This is the dense-dense
  matrix multiplication, which is very well optimized on modern CPUs. Important: the total
  number of operations is :math:`O(2^{2 n})` in contrast to :math:`O(2^{3 n})` as in general
  case.

* Third, we permute rows of :math:`\widetilde{C}_i P M` by applying :math:`P^T`.

* Right-hand side multiplication is done in a similar way.

* In summary, we save computational resources by exploiting some properties of 2-qubit gate
  matrices :math:`C_i` and using hardware optimized multiplication of dense matrices. There
  is still a room for further improvement, of course.

References:

    [1]: Liam Madden, Andrea Simonetto, Best Approximate Quantum Compiling Problems.
    `arXiv:2106.05649 <https://arxiv.org/abs/2106.05649>`_
"""
