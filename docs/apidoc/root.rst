.. _qiskit-root:

==============================
Root namespace (:mod:`qiskit`)
==============================

.. currentmodule:: qiskit

Most Qiskit functionality is accessed through specific submodules.
You can consult the top-level documentation of :mod:`qiskit` to find the list of modules, such as
:mod:`qiskit.circuit` or :mod:`qiskit.transpiler.passes`.

Several names are re-exported from the repository root, whose canonical public locations are in
submodules.  The re-exports in the root namespace are part of Qiskit's public API.

..
   Unlike other `autosummary` directives in Qiskit, we _don't_ set `:toctree:` because we don't want
   the stub files generated for this table.  This is just a cross-referencing table to other
   modules, which own the data.

Names re-exported from :mod:`qiskit.circuit`:

.. autosummary::
   
   ~circuit.AncillaRegister
   ~circuit.ClassicalRegister
   ~circuit.QuantumCircuit
   ~circuit.QuantumRegister

Names re-exported from :mod:`qiskit.compiler`:

.. autosummary::

   ~compiler.transpile

Names re-exported from :mod:`qiskit.exceptions`:

.. autosummary::

   ~exceptions.MissingOptionalLibraryError
   ~exceptions.QiskitError

Names re-exported from :mod:`qiskit.transpiler`:

.. autosummary::

   ~transpiler.generate_preset_pass_manager
