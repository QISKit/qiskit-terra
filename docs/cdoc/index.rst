===========================
Qiskit C API (``qiskit.h``)
===========================

The Qiskit C API is a low level interface to the core data model of Qiskit. It
is designed to provide a high performance interface to Qiskit for compiled
languages and provides a defined ABI to the internal Rust data model that is
used to create the Python API. There are two expected modes of operation for
the C API:

* A standalone shared library for creating and working with Qiskit objects from
  compiled languages without a runtime dependency on Python, and
* For building Python extensions that are using Qiskit but interface directly
  with the Rust objects from the extension code without using Python for better
  performance.

To get started, see `Install the Qiskit C API <https://quantum.cloud.ibm.com/docs/guides/install-c-api>`_.
To combine the C API with custom Python extensions, see
`Extend Qiskit in Python with C <https://quantum.cloud.ibm.com/docs/guides/c-extension-for-python>`_.

As this interface is still new in Qiskit it should be considered experimental
and the interface might change between minor version releases.

---------------
Quantum Circuit
---------------

.. toctree::
   :maxdepth: 1

   qk-circuit
   qk-quantum-register
   qk-classical-register
   qk-param

Dynamic Circuits
++++++++++++++++

.. toctree::
   :maxdepth: 1

   qk-dynamic-circuits
   qk-control-flow
   qk-classical-expressions

Circuit Library
+++++++++++++++

.. toctree::
   :maxdepth: 1

   qk-circuit-library


-------------------
Quantum information
-------------------

.. toctree::
   :maxdepth: 1

   qk-obs
   qk-obs-term
   qk-bit-term
   qk-complex64
   qk-exit-code

----------
Transpiler
----------

Using the transpiler from the C API is intended to only cover
circuits created solely via the C API. If you are in a hybrid mode where
you're using the C API with Python you should invoke the transpiler via
the Python :mod:`qiskit.transpiler` module instead; the functionality is
the same Rust internals they just offer different entrypoints. The C API
for transpilation makes assumptions about the input only using constructs
exposed to the C Quantum Circuit API and you will potentially get incomplete
results transpiling circuits from Python via the C API.

.. toctree::
   :maxdepth: 1

   qk-dag
   qk-transpiler
   qk-target
   qk-target-entry
   qk-neighbors
   qk-transpile-layout
   qk-transpiler-passes
   qk-vf2-layout
   qk-sabre-layout-options


---------
Utilities
---------

.. toctree::
   :maxdepth: 1

   config
   version
