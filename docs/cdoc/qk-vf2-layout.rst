.. _capi-vf2-layout:

=========================
VF2 compiler-pass objects
=========================

VF2 Layout Configuration (:c:type:`QkVF2LayoutConfiguration`)
=============================================================

.. c:type:: struct QkVF2LayoutConfiguration QkVF2LayoutConfiguration

The configuration for the VF2 layout passes.  This is an encapsulated configuration to allow for
changes in the API over time; you create and mutate this using the constructor and setters below.

.. doxygengroup:: QkVF2LayoutConfiguration
   :members:
   :content-only:

VF2 Layout Result (:c:type:`QkVF2LayoutResult`)
===============================================

.. c:type:: struct QkVF2LayoutResult QkVF2LayoutResult

When running the ``qk_transpiler_pass_standalone_vf2_layout`` function it returns its analysis
result as a ``QkVF2LayoutResult`` object. This object contains the outcome of the transpiler pass,
whether the pass was able to find a layout or not, and what the layout selected by the pass was.

Functions
~~~~~~~~~

.. doxygengroup:: QkVF2LayoutResult
   :members:
   :content-only:

