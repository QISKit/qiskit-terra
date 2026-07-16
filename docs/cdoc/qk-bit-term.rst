.. _capi-bit-term:

=========
QkBitTerm
=========

..
    This is documented manually here because the C-space `enum` is generated
    programmatically from Rust and is not the correct C-level documentation.

.. code-block:: c

   enum QkBitTerm

An enum that to represent each of the single-qubit alphabet terms enumerated below.

Values
------

* ``QkBitTerm_X`` 
   The Pauli :math:`X` operator. 

   Value: 2 (``0b0010``)

* ``QkBitTerm_Y``
   The Pauli :math:`Y` operator.
   
   Value: 3 (``0b0011``)

* ``QkBitTerm_Z``
   The Pauli :math:`Z` operator.
   
   Value: 1 (``0b0001``)

* ``QkBitTerm_Plus``
   The projector :math:`\lvert +\rangle\langle +\rvert` to the positive :math:`X` eigenstate.
   
   Value: 10 (``0b1010``)

* ``QkBitTerm_Minus``
   The projector :math:`\lvert -\rangle\langle -\rvert` to the negative :math:`X` eigenstate.
   
   Value: 6 (``0b0110``)

* ``QkBitTerm_Right``
   The projector :math:`\lvert r\rangle\langle r\rvert` to the positive :math:`Y` eigenstate.

   Value: 11 (``0b1011``)

* ``QkBitTerm_Left``
   The projector :math:`\lvert l\rangle\langle l\rvert` to the negative :math:`Y` eigenstate.
  
   Value: 7 (``0b0111``)

* ``QkBitTerm_Zero``
   The projector :math:`\lvert 0\rangle\langle 0\rvert` to the positive :math:`Z` eigenstate.
   
   Value: 9 (``0b1001``)

* ``QkBitTerm_One``
   The projector :math:`\lvert 1\rangle\langle 1\rvert` to the negative :math:`Z` eigenstate.
   
   Value: 5 (``0b0101``)

Representation
--------------

The enum is stored as single byte, its elements are represented as unsigned 8-bit integer.

.. code-block:: c

   typedef uint8_t QkBitTerm

.. warning::

   Not all ``uint8_t`` values are valid bit terms. Passing invalid values is undefined behavior.

The numeric structure of these is that they are all four-bit values of which the low two
bits are the (phase-less) symplectic representation of the Pauli operator related to the
object, where the low bit denotes a contribution by :math:`Z` and the second lowest a
contribution by :math:`X`, while the upper two bits are ``00`` for a Pauli operator, ``01``
for the negative-eigenstate projector, and ``10`` for the positive-eigenstate projector.

Functions
---------

.. doxygengroup:: QkBitTerm
   :content-only:
