# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
======================================================
Fake Provider (:mod:`qiskit.providers.fake_provider`)
======================================================

.. currentmodule:: qiskit.providers.fake_provider

Overview
========

The fake provider module in Qiskit contains fake (simulated) backend classes
useful for testing the transpiler and other backend-facing functionality.

Example Usage
-------------

Here is an example of using a simulated backend for transpilation and running.

.. plot::
   :alt: Output from the previous code.
   :include-source:

   from qiskit import QuantumCircuit, transpile
   from qiskit.providers.fake_provider import GenericBackendV2
   from qiskit.visualization import plot_histogram

   # Generate a 5-qubit simulated backend
   backend = GenericBackendV2(num_qubits=5)

   # Create a simple circuit
   circuit = QuantumCircuit(3)
   circuit.h(0)
   circuit.cx(0,1)
   circuit.cx(0,2)
   circuit.measure_all()
   circuit.draw('mpl')

   # Transpile the ideal circuit to a circuit that can be directly executed by the backend
   transpiled_circuit = transpile(circuit, backend)
   transpiled_circuit.draw('mpl')

   # Run the transpiled circuit using the simulated backend
   job = backend.run(transpiled_circuit)
   counts = job.result().get_counts()
   plot_histogram(counts)


V2 Simulated Backends
=====================

.. autosummary::
    :toctree: ../stubs/

    GenericBackendV2

"""
from .generic_backend_v2 import GenericBackendV2

__all__ = ["GenericBackendV2"]
