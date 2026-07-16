.. _capi-obs:

=====
QkObs
=====

.. code-block:: c

   typedef struct QkObs QkObs

An observable over Pauli bases that stores its data in a qubit-sparse format.


Mathematics
===========

This observable represents a sum over strings of the Pauli operators and Pauli-eigenstate
projectors, with each term weighted by some complex number.  That is, the full observable is

.. math::
    \text{\texttt{QkObs}} = \sum_i c_i \bigotimes_n A^{(n)}_i

for complex numbers :math:`c_i` and single-qubit operators acting on qubit :math:`n` from a
restricted alphabet :math:`A^{(n)}_i`.  The sum over :math:`i` is the sum of the individual
terms, and the tensor product produces the operator strings.
The alphabet of allowed single-qubit operators that the :math:`A^{(n)}_i` are drawn from is the
Pauli operators and the Pauli-eigenstate projection operators.  Explicitly, these are:

.. _qkobs-alphabet:
.. table:: Alphabet of single-qubit terms used in ``QkObs``

  +----------------------------------------+--------------------+----------------+
  | Operator                               | ``QkBitTerm``      | Numeric value  |
  +========================================+====================+================+
  | :math:`I` (identity)                   | Not stored.        | Not stored.    |
  +----------------------------------------+--------------------+----------------+
  | :math:`X` (Pauli X)                    | ``QkBitTerm_X``    | ``0b0010`` (2) |
  +----------------------------------------+--------------------+----------------+
  | :math:`Y` (Pauli Y)                    | ``QkBitTerm_Y``    | ``0b0011`` (3) |
  +----------------------------------------+--------------------+----------------+
  | :math:`Z` (Pauli Z)                    | ``QkBitTerm_Z``    | ``0b0001`` (1) |
  +----------------------------------------+--------------------+----------------+
  | :math:`\lvert+\rangle\langle+\rvert`   | ``QkBitTerm_Plus`` | ``0b1010`` (10)|
  | (projector to positive eigenstate of X)|                    |                |
  +----------------------------------------+--------------------+----------------+
  | :math:`\lvert-\rangle\langle-\rvert`   | ``QkBitTerm_Minus``| ``0b0110`` (6) |
  | (projector to negative eigenstate of X)|                    |                |
  +----------------------------------------+--------------------+----------------+
  | :math:`\lvert r\rangle\langle r\rvert` | ``QkBitTerm_Right``| ``0b1011`` (11)|
  | (projector to positive eigenstate of Y)|                    |                |
  +----------------------------------------+--------------------+----------------+
  | :math:`\lvert l\rangle\langle l\rvert` | ``QkBitTerm_Left`` | ``0b0111`` (7) |
  | (projector to negative eigenstate of Y)|                    |                |
  +----------------------------------------+--------------------+----------------+
  | :math:`\lvert0\rangle\langle0\rvert`   | ``QkBitTerm_Zero`` | ``0b1001`` (9) |
  | (projector to positive eigenstate of Z)|                    |                |
  +----------------------------------------+--------------------+----------------+
  | :math:`\lvert1\rangle\langle1\rvert`   | ``QkBitTerm_One``  | ``0b0101`` (5) |
  | (projector to negative eigenstate of Z)|                    |                |
  +----------------------------------------+--------------------+----------------+

Due to allowing both the Paulis and their projectors, the allowed alphabet forms an overcomplete
basis of the operator space.  This means that there is not a unique summation to represent a
given observable. As a consequence, comparison requires additional care and using
``qk_obs_canonicalize`` on two mathematically equivalent observables might not result in the same
representation.

``QkObs`` uses its particular overcomplete basis with the aim of making
"efficiency of measurement" equivalent to "efficiency of representation".  For example, the
observable :math:`{\lvert0\rangle\langle0\rvert}^{\otimes n}` can be efficiently measured on
hardware with simple :math:`Z` measurements, but can only be represented in terms of Paulis
as :math:`{(I + Z)}^{\otimes n}/2^n`, which requires :math:`2^n` stored terms. ``QkObs`` requires
only a single term to store this. The downside to this is that it is impractical to take an
arbitrary matrix and find the *best* ``QkObs`` representation.  You typically will want to construct
a ``QkObs`` directly, rather than trying to decompose into one.


Representation
==============

The internal representation of a ``QkObs`` stores only the non-identity qubit
operators.  This makes it significantly more efficient to represent observables such as
:math:`\sum_{n\in \text{qubits}} Z^{(n)}`; ``QkObs`` requires an amount of
memory linear in the total number of qubits.
The terms are stored compressed, similar in spirit to the compressed sparse row format of sparse
matrices.  In this analogy, the terms of the sum are the "rows", and the qubit terms are the
"columns", where an absent entry represents the identity rather than a zero.  More explicitly,
the representation is made up of four contiguous arrays:

.. _qkobs-arrays:
.. table:: Data arrays used to represent ``QkObs``

  =======================  ===========  ============================================================
  Attribute accessible by  Length       Description
  =======================  ===========  ============================================================
  ``qk_obs_coeffs``        :math:`t`    The complex scalar multiplier for each term.

  ``qk_obs_bit_terms``     :math:`s`    Each of the non-identity single-qubit terms for all of
                                        the operators, in order. These correspond to the
                                        non-identity :math:`A^{(n)}_i` in the sum description,
                                        where the entries are stored in order of increasing
                                        :math:`i` first, and in order of increasing :math:`n`
                                        within each term.

  ``qk_obs_indices``       :math:`s`    The corresponding qubit (:math:`n`) for each of the
                                        bit terms. ``QkObs`` requires that this list is term-wise
                                        sorted, and algorithms can rely on this invariant being
                                        upheld.

  ``qk_obs_boundaries``    :math:`t+1`  The indices that partition the bit terms and indices
                                        into complete terms.  For term number :math:`i`, its
                                        complex coefficient is stored at index ``i``, and its
                                        non-identity single-qubit operators and their corresponding
                                        qubits are in the range ``[boundaries[i], boundaries[i+1])``
                                        in the bit terms and indices, respectively.
                                        The boundaries always have an explicit 0 as their first
                                        element.
  =======================  ===========  ============================================================

The length parameter :math:`t` is the number of terms in the sum and can be queried using
``qk_obs_num_terms``. The parameter :math:`s` is the total number of non-identity single-qubit
terms and can be queried using ``qk_obs_len``.

As illustrative examples:

* in the case of a zero operator, the boundaries are length 1 (a single 0) and all other
  vectors are empty.

* in the case of a fully simplified identity operator, the boundaries are ``{0, 0}``,
  the coefficients have a single entry, and both the bit terms and indices are empty.

* for the operator :math:`Z_2 Z_0 - X_3 Y_1`, the boundaries are ``{0, 2, 4}``,
  the coeffs are ``{1.0, -1.0}``, the bit terms are ``{QkBitTerm_Z, QkBitTerm_Z, QkBitTerm_Y,
  QkBitTerm_X}`` and the indices are ``{0, 2, 1, 3}``.  The operator might act on more than
  four qubits, depending on the the number of qubits (see ``qk_obs_num_qubits``). Note
  that the single-bit terms and indices are sorted into termwise sorted order.

These cases are not special, they're fully consistent with the rules and should not need special
handling.


Canonical ordering
------------------

For any given mathematical observable, there are several ways of representing it with
``QkObs``.  For example, the same set of single-bit terms and their corresponding indices might
appear multiple times in the observable.  Mathematically, this is equivalent to having only a
single term with all the coefficients summed.  Similarly, the terms of the sum in a ``QkObs``
can be in any order while representing the same observable, since addition is commutative
(although while floating-point addition is not associative, ``QkObs`` makes no guarantees about
the summation order).

These two categories of representation degeneracy can cause the operator equality,
``qk_obs_equal``, to claim that two observables are not equal, despite representing the same
object.  In these cases, it can be convenient to define some *canonical form*, which allows
observables to be compared structurally.
You can put a ``QkObs`` in canonical form by using the ``qk_obs_canonicalize`` function.
The precise ordering of terms in canonical ordering is not specified, and may change between
versions of Qiskit.  Within the same version of Qiskit, however, you can compare two observables
structurally by comparing their simplified forms.

.. note::

    If you wish to account for floating-point tolerance in the comparison, it is safest to use
    a recipe such as:

    .. code-block:: c

        bool equivalent(QkObs *left, QkObs *right, double tol) {
          // compare a canonicalized version of left - right to the zero observable
          QkObs *neg_right = qk_obs_multiply(right, &(QkComplex64){-1, 0});
          QkObs *diff = qk_obs_add(left, neg_right);
          QkObs *canonical = qk_obs_canonicalize(diff, tol);

          QkObs *zero = qk_obs_zero(qk_obs_num_qubits(left));
          bool equiv = qk_obs_equal(diff, zero);
          // free all temporary variables
          qk_obs_free(neg_right);
          qk_obs_free(diff);
          qk_obs_free(canonical);
          qk_obs_free(zero);
          return equiv;
        }

.. note::

    The canonical form produced by ``qk_obs_canonicalize`` alone will not universally detect all
    observables that are equivalent due to the over-complete basis alphabet.


Indexing
--------

Individual observable sum terms in ``QkObs`` can be accessed via ``qk_obs_term`` and return
objects of type ``QkObsTerm``. These terms then contain fields with the coefficient of the term,
its bit terms, indices and the number of qubits it is defined on. Together with the information
of the number of terms, you can iterate over all observable terms as

.. code-block:: c

    size_t num_terms = qk_obs_num_terms(obs);  // obs is QkObs*
    for (size_t i = 0; i < num_terms; i++) {
        QkObsTerm term;  // allocate term on stack
        int exit = qk_obs_term(obs, i, &term);  // get the term (exit > 0 upon index errors)
        // do something with the term...
    }

.. warning::

    Populating a ``QkObsTerm`` via ``qk_obs_term`` will reference data of the original
    ``QkObs``. Modifying the bit terms or indices will change the observable and can leave
    it in an incoherent state.


Construction
============

``QkObs`` can be constructed by initializing an empty observable (with ``qk_obs_zero``) and
iteratively adding terms (with ``qk_obs_add_term``). Alternatively, an observable can be
constructed from "raw" data (with ``qk_obs_new``) if all internal data is specified. This requires
care to ensure the data is coherent and results in a valid observable.

.. _qkobs-constructors:
.. table:: Constructors

  ===================  =========================================================================
  Function             Summary
  ===================  =========================================================================
  ``qk_obs_zero``      Construct an empty observable on a given number of qubits.

  ``qk_obs_identity``  Construct the identity observable on a given number of qubits.

  ``qk_obs_new``       Construct an observable from :ref:`the raw data arrays <qkobs-arrays>`.
  ===================  =========================================================================


Mathematical manipulation
=========================

``QkObs`` supports fundamental arithmetic operations in between observables or with scalars.
You can:

* add two observables using ``qk_obs_add`` and ``qk_obs_add_inplace``

* multiply by a complex number with ``qk_obs_multiply`` and ``qk_obs_multiply_inplace``

* compose (multiply) two observables via ``qk_obs_compose`` and ``qk_obs_compose_map``

* compute ``left + scalar * right`` for two observables and a complex scalar with
  ``qk_obs_scaled_add`` and ``qk_obs_scaled_add_inplace``


Functions
=========

.. doxygengroup:: QkObs
   :content-only:

