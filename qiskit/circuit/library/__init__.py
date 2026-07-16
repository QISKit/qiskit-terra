# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


r"""
===============================================
Circuit Library (:mod:`qiskit.circuit.library`)
===============================================

.. currentmodule:: qiskit.circuit.library

The circuit library is a collection of quantum circuits that serve as algorithmic building blocks,
benchmarking tools, and examples that are conjectured to be hard to simulate classically.
The goal is to provide circuits that are well-defined, widely applicable, and easy to reuse across
different workflows.

The criteria for a circuit in the library are:

Well-defined
   The circuit has a clear structure or a mathematical action, supported by a paper or
   reference.

Widely used and relevant
   The circuit is a canonical or best-known choice for a well-defined task, is a standard reference
   or common baseline, or is historically foundational to a substantial subfield of quantum
   computing.

Standalone
   The circuit is self-contained, meaning it can be generated efficiently without new
   external dependencies.

Elements in the circuit library are either :class:`.QuantumCircuit`\ s or
:class:`~.circuit.Instruction`\ s, allowing them to be easily investigated or plugged into other
circuits. This enables fast prototyping and circuit design at higher levels of abstraction.

For example:

.. plot::
   :alt: A circuit implementing a Suzuki-Trotter expansion of a Hamiltonian evolution.
   :include-source:

   from qiskit.circuit import QuantumCircuit
   from qiskit.circuit.library import PauliEvolutionGate
   from qiskit.quantum_info import SparsePauliOp

   hamiltonian = SparsePauliOp(["ZZI", "IZZ", "IXI"], coeffs=[1, 1, -1])
   gate = PauliEvolutionGate(hamiltonian)

   circuit = QuantumCircuit(hamiltonian.num_qubits)
   circuit.append(gate, circuit.qubits)

   circuit.draw("mpl")

This library is organized in different sections:

   * :ref:`Standard gates <standard-gates>`
   * :ref:`Standard directives <standard-directives>`
   * :ref:`Standard operations <standard-operations>`
   * :ref:`Generalized gates <generalized-gates>`
   * :ref:`Arithmetic operations <arithmetic>`
   * :ref:`Basis changes <basis-change>`
   * :ref:`Boolean logic <boolean-logic>`
   * :ref:`Data encoding <data-encoding>`
   * :ref:`Data preparation <data-preparation>`
   * :ref:`Particular operations <particular>`
   * :ref:`N-local circuits <n-local>`
   * :ref:`Oracles <oracles>`
   * :ref:`Template circuits <template>`

We distinguish into different categories of operations:

Standard gates
   These are fundamental quantum gates, a subset of which typically forms a basis gate
   set on a quantum computer. These are unitary operations represented as :class:`.Gate`.
   The library also provides standard compiler directives (a :class:`.Barrier`) and non-unitary
   operations (like :class:`.Measure`).

Abstract operations
   This category includes operations that are defined by a mathematical action, but can be implemented
   with different decompositions. For example, a multi-controlled :class:`.XGate` flips the target
   qubit if all control qubits are :math:`|1\rangle`, and there are a variety of concrete circuits
   implementing this operation using lower-level gates. Such abstract operations are represented as
   :class:`.Gate` or :class:`~.circuit.Instruction`. This allows building the circuit without choosing
   a concrete implementation of each block and, finally, let the compiler (or you as user) choose the
   optimal decomposition. For example:

   .. plot::
      :alt: A circuit with a multi-controlled X gate.
      :include-source:

      from qiskit.circuit.library import MCXGate
      mcx = MCXGate(4)

      from qiskit import QuantumCircuit
      circuit = QuantumCircuit(5)
      circuit.append(mcx, [0, 1, 4, 2, 3])
      circuit.draw("mpl")

   For circuits with abstract operations, the circuit context is taken into account during
   transpilation. For example, if idle qubits are available, they can be used to obtain a shallower
   circuit::

     from qiskit import transpile

     small_circuit = QuantumCircuit(5)  # here we have no idle qubits
     small_circuit.append(mcx, [0, 1, 4, 2, 3])
     small_tqc = transpile(small_circuit, basis_gates=["u", "cx"])
     print("No aux:", small_tqc.count_ops())

     large_circuit = QuantumCircuit(10)  # now we will have 5 idle qubits
     large_circuit.append(mcx, [0, 1, 4, 2, 3])
     large_tqc = transpile(large_circuit, basis_gates=["u", "cx"])
     print("With aux:", large_tqc.count_ops())

   Which prints:

   .. parsed-literal::

      No aux: OrderedDict([('u', 41), ('cx', 36)])
      With aux: OrderedDict([('u', 24), ('cx', 18)])

Structural operations
   These operations have a unique decomposition. As the compiler does not need to reason about
   them on a higher level, they are implemented as functions that return a :class:`.QuantumCircuit`
   object. For example:

   .. plot::
      :alt: The real amplitudes ansatz circuit.
      :include-source:

      from qiskit.circuit.library import real_amplitudes

      ansatz = real_amplitudes(5, entanglement="pairwise")
      ansatz.draw("mpl")

.. _standard-gates:

Standard gates
==============

These operations are reversible unitary gates and they all subclass
:class:`~qiskit.circuit.Gate`. As a consequence, they all have the methods
:meth:`~qiskit.circuit.Gate.to_matrix`, :meth:`~qiskit.circuit.Gate.power`,
and :meth:`~qiskit.circuit.Gate.control`, which we can generally only apply to unitary operations.

For example:

.. plot::
   :alt: The X gate and the matrix, power, and control methods.
   :include-source:
   :nofigs:

    from qiskit.circuit.library import XGate
    gate = XGate()
    print(gate.to_matrix())             # X gate
    print(gate.power(1/2).to_matrix())  # √X gate -- see also the SXGate
    print(gate.control(1).to_matrix())  # CX (controlled X) gate

.. code-block:: text

    [[0.+0.j 1.+0.j]
     [1.+0.j 0.+0.j]]
    [[0.5+0.5j 0.5-0.5j]
     [0.5-0.5j 0.5+0.5j]]
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]]


The function :func:`.get_standard_gate_name_mapping` allows you to see the available standard gates
and operations.

.. autofunction:: get_standard_gate_name_mapping

1-qubit standard gates
----------------------

.. autosummary::
   :toctree: ../stubs/

   HGate
   IGate
   PhaseGate
   RGate
   RXGate
   RYGate
   RZGate
   SGate
   SdgGate
   SXGate
   SXdgGate
   TGate
   TdgGate
   UGate
   U1Gate
   U2Gate
   U3Gate
   XGate
   YGate
   ZGate

2-qubit standard gates
----------------------

.. autosummary::
   :toctree: ../stubs/

   CHGate
   CPhaseGate
   CRXGate
   CRYGate
   CRZGate
   CSGate
   CSdgGate
   CSXGate
   CUGate
   CU1Gate
   CU3Gate
   CXGate
   CYGate
   CZGate
   DCXGate
   ECRGate
   iSwapGate
   RXXGate
   RYYGate
   RZXGate
   RZZGate
   SwapGate
   XXMinusYYGate
   XXPlusYYGate

3+ qubit standard gates
-----------------------

.. autosummary::
   :toctree: ../stubs/

   C3SXGate
   C3XGate
   C4XGate
   CCXGate
   CCZGate
   CSwapGate
   RCCXGate
   RC3XGate

Global standard gates
---------------------

The following gate is global and does not take any qubit arguments.

.. autosummary::
   :toctree: ../stubs/

   GlobalPhaseGate


.. _standard-directives:

Standard Directives
===================

Directives are operations to the quantum stack that are meant to be interpreted by the backend or
the transpiler. In general, the transpiler or backend might optionally ignore them if there is no
implementation for them.

* :class:`~qiskit.circuit.Barrier`


.. _standard-operations:

Standard Operations
===================

Operations are non-reversible changes in the quantum state of the circuit.

* :class:`~qiskit.circuit.Measure`
* :class:`~qiskit.circuit.Reset`

In addition to :class:`.Measure`, there is a generalized joint-Pauli measurement available, used as
a basic operation in Pauli-based computation.

.. autosummary::
    :toctree: ../stubs

    PauliProductMeasurement


.. _generalized-gates:

Generalized Gates
=================

This module extends the standard gates with a broader collection of basic gates. This includes
gates that are variadic, meaning that the number of qubits depends on the input.
For example::

    from qiskit.circuit.library import DiagonalGate

    diagonal = DiagonalGate([1, 1j])
    print(diagonal.num_qubits)

    diagonal = DiagonalGate([1, 1, 1, -1])
    print(diagonal.num_qubits)

which prints:

.. code-block:: text

    1
    2

.. autosummary::
   :toctree: ../stubs/

   DiagonalGate
   PermutationGate
   MCMTGate
   MCPhaseGate
   MCXGate
   MSGate
   RVGate
   PauliGate
   PauliProductRotationGate
   LinearFunction
   Isometry
   UnitaryGate
   UCGate
   UCPauliRotGate
   UCRXGate
   UCRYGate
   UCRZGate

The above objects derive :class:`.Gate` or :class:`~.circuit.Instruction`, which allows the
compiler to reason about them on an abstract level. We therefore suggest using these instead
of the following, which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   Diagonal
   MCMT
   MCMTVChain
   MCXGrayCode
   MCXRecursive
   MCXVChain
   Permutation
   GMS
   GR
   GRX
   GRY
   GRZ

.. _boolean-logic:

Boolean Logic
=============

These :class:`.Gate`\ s implement boolean logic operations, such as the logical
``or`` of a set of qubit states.

.. autosummary::
   :toctree: ../stubs/

   AndGate
   OrGate
   BitwiseXorGate
   InnerProductGate

The above objects derive :class:`.Gate` (or return this type), which allows the
compiler to reason about them on an abstract level. We therefore suggest using these instead
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   AND
   OR
   XOR
   InnerProduct


A random bitwise ``xor`` circuit can be directly generated using:

.. autosummary::
   :toctree: ../stubs/

   random_bitwise_xor

.. _basis-change:

Basis Change
============

These gates perform basis transformations of the qubit states. For example,
in the case of the Quantum Fourier Transform (QFT), it transforms between
the computational basis and the Fourier basis.

.. autosummary::
   :toctree: ../stubs/

   QFTGate

The above object derives :class:`.Gate`, which allows the
compiler to reason about it on an abstract level. We therefore suggest using this instead
of the following which derives :class:`.QuantumCircuit` and is eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   QFT

.. _arithmetic:

Arithmetic
==========

These gates and circuits perform classical arithmetic, such as addition or multiplication.

Adders
------

Adders compute the sum of two :math:`n`-qubit registers, that is

.. math::

   |a\rangle_n |b\rangle_n \mapsto |a\rangle_n |a + b\rangle_{t},

where the size :math:`t` of the output register depends on the type of adder used.

.. autosummary::
   :toctree: ../stubs/

   ModularAdderGate
   HalfAdderGate
   FullAdderGate

The above objects derive :class:`.Gate`, which allows the
compiler to reason about them on an abstract level. We therefore suggest using these instead
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   DraperQFTAdder
   CDKMRippleCarryAdder
   VBERippleCarryAdder

Multipliers
-----------

Multipliers compute the product of two :math:`n`-qubit registers, that is

.. math::

   |a\rangle_n |b\rangle_n |0\rangle_{t} \mapsto |a\rangle_n |b\rangle_n |a \cdot b\rangle_t,

where :math:`t` is the number of bits used to represent the result.

.. autosummary::
   :toctree: ../stubs/

   MultiplierGate

The above object derives :class:`.Gate`, which allows the
compiler to reason about it on an abstract level. We therefore suggest using this instead
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   HRSCumulativeMultiplier
   RGQFTMultiplier

Amplitude Functions
-------------------

An amplitude function approximates a function :math:`f: \{0, ..., 2^n - 1\} \rightarrow [0, 1]`
applied on the amplitudes of :math:`n` qubits. See the class docstring for more detailed information.

.. autosummary::
   :toctree: ../stubs/

   LinearAmplitudeFunctionGate

The above object derives :class:`.Gate`, which allows the
compiler to reason about it on an abstract level. We therefore suggest using this instead
of the following which derives :class:`.QuantumCircuit` and is eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   LinearAmplitudeFunction

Functional Pauli Rotations
--------------------------

Functional Pauli rotations implement operations of the form

.. math::

   |x\rangle |0\rangle \mapsto \cos(f(x))|x\rangle|0\rangle + \sin(f(x))|x\rangle|1\rangle

using Pauli-:math:`Y` rotations for different types of functions :math:`f`, such as linear,
polynomial, or  a piecewise version of these. They are similar to the amplitude functions above, but
without pre- and post-processing for the domain and image of the target function.

.. autosummary::
   :toctree: ../stubs/

   LinearPauliRotationsGate
   PolynomialPauliRotationsGate
   PiecewiseLinearPauliRotationsGate
   PiecewisePolynomialPauliRotationsGate
   PiecewiseChebyshevGate

The above objects derive :class:`.Gate`, which allows the
compiler to reason about them on an abstract level. We therefore suggest using these instead
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   FunctionalPauliRotations
   LinearPauliRotations
   PolynomialPauliRotations
   PiecewiseLinearPauliRotations
   PiecewisePolynomialPauliRotations
   PiecewiseChebyshev


Other arithmetic functions
--------------------------

Here we list additional arithmetic circuits. See the individual class docstrings for more details.

.. autosummary::
   :toctree: ../stubs/

   ExactReciprocalGate
   IntegerComparatorGate
   QuadraticFormGate
   WeightedSumGate

The above objects derive :class:`.Gate`, which allows the
compiler to reason about them on an abstract level. We therefore suggest using these instead
of the following which derive :class:`.QuantumCircuit` and are eagerly constructed.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   ExactReciprocal
   IntegerComparator
   QuadraticForm
   WeightedAdder

.. _particular:

Particular Quantum Circuits
===========================

The following gates and quantum circuits define specific operations of interest:

.. autosummary::
   :toctree: ../stubs/

   fourier_checking
   hidden_linear_function
   iqp
   random_iqp
   quantum_volume
   phase_estimation
   grover_operator
   unitary_overlap
   GraphStateGate
   PauliEvolutionGate
   HamiltonianGate

Below we provide the same operations as classes deriving :class:`.QuantumCircuit`. For better
runtime and compiler performance, however, we suggest using above functions and gates.

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   FourierChecking
   GraphState
   HiddenLinearFunction
   IQP
   QuantumVolume
   PhaseEstimation
   GroverOperator
   UnitaryOverlap

.. _n-local:

N-local circuits
================

The following functions return a parameterized :class:`.QuantumCircuit` to use as ansatz in
a broad set of variational quantum algorithms.

For example, we can build a variational circuit

.. plot::
   :alt: The efficient SU2 ansatz circuit...
   :include-source:
   :context:

   from qiskit.circuit.library import efficient_su2

   num_qubits = 4
   ansatz = efficient_su2(num_qubits, entanglement="pairwise")
   ansatz.draw("mpl")

and combine it with

.. plot::
   :alt: ... combined with the ZZ feature map.
   :include-source:
   :context:

   from qiskit.circuit.library import zz_feature_map

   circuit = zz_feature_map(num_qubits)
   circuit.barrier()
   circuit.compose(ansatz, inplace=True)

   circuit.draw("mpl")

to obtain a circuit for variational quantum classification.

The following functions all construct variational circuits and are optimized for a fast
construction:

.. autosummary::
   :toctree: ../stubs/

   n_local
   efficient_su2
   real_amplitudes
   pauli_two_design
   excitation_preserving
   qaoa_ansatz
   hamiltonian_variational_ansatz
   evolved_operator_ansatz

While we suggest using the above functions, we also continue supporting the following
:class:`.BlueprintCircuit`, which wrap the circuits into a block
and allow for inplace mutations of the circuits:

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   NLocal
   TwoLocal
   PauliTwoDesign
   RealAmplitudes
   EfficientSU2
   EvolvedOperatorAnsatz
   ExcitationPreserving
   QAOAAnsatz


.. _data-encoding:

Data encoding circuits
======================

The following functions return a parameterized :class:`.QuantumCircuit` to use as data
encoding circuits in a series of variational quantum algorithms:

.. autosummary::
   :toctree: ../stubs/

   pauli_feature_map
   z_feature_map
   zz_feature_map

While we suggest using the above functions, we also continue supporting the following
:class:`.BlueprintCircuit`, which wrap the circuits into a block
and allow for inplace mutations of the circuits:

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   PauliFeatureMap
   ZFeatureMap
   ZZFeatureMap


.. _data-preparation:

Data preparation circuits
=========================

The following operations are used for state preparation:

.. autosummary::
   :toctree: ../stubs/

   StatePreparation
   Initialize

.. _oracles:

Oracles
=======

An "oracle" can refer to a variety of black-box operations on quantum states. Here, we consider
oracles implementing boolean functions :math:`f: \{0, ..., 2^n - 1\} \rightarrow \{0, 1\}` via
phase-flips

.. math::

   |x\rangle_n \mapsto (-1)^{f(x)} |x\rangle_n,

or bit-flips

.. math::

   |x\rangle_n |b\rangle \mapsto |x\rangle_n |b \oplus f(x)\rangle.

These are implemented in

.. autosummary::
   :toctree: ../stubs/

   PhaseOracleGate
   BitFlipOracleGate

and an important building block for Grover's algorithm (see :func:`.grover_operator`).

In addition to the :class:`.Gate`-based implementation we also support the
:class:`.QuantumCircuit`-version of the phase flip oracle

.. autosummary::
   :toctree: ../stubs/
   :template: autosummary/class_no_inherited_members.rst

   PhaseOracle


.. _template:

Template circuits
=================

Templates are functions that return circuits that compute the identity. They are used at
circuit optimization where matching part of the template allows the compiler
to replace the match with the inverse of the remainder from the template.

In this example, the identity constant in a template is checked:

.. plot::
   :alt: A Toffoli template circuit.
   :include-source:
   :nofigs:

    from qiskit.circuit.library.templates import template_nct_4b_1
    from qiskit.quantum_info import Operator
    import numpy as np

    template = template_nct_4b_1()

    identity = np.identity(2 ** len(template.qubits), dtype=complex)
    data = Operator(template).data
    np.allclose(data, identity)  # True, template_nct_4b_1 is the identity

NCT (Not-CNOT-Toffoli) template circuits
----------------------------------------

Template circuits for :class:`~qiskit.circuit.library.XGate`,
:class:`~qiskit.circuit.library.CXGate`,
and :class:`~qiskit.circuit.library.CCXGate` (Toffoli) gates.

References:

Maslov, D. and Dueck, G. W. and Miller, D. M.,
Techniques for the synthesis of reversible Toffoli networks, 2007
https://dx.doi.org/10.1145/1278349.1278355

.. currentmodule:: qiskit.circuit.library.templates.nct
.. autofunction:: template_nct_2a_1
.. autofunction:: template_nct_2a_2
.. autofunction:: template_nct_2a_3
.. autofunction:: template_nct_4a_1
.. autofunction:: template_nct_4a_2
.. autofunction:: template_nct_4a_3
.. autofunction:: template_nct_4b_1
.. autofunction:: template_nct_4b_2
.. autofunction:: template_nct_5a_1
.. autofunction:: template_nct_5a_2
.. autofunction:: template_nct_5a_3
.. autofunction:: template_nct_5a_4
.. autofunction:: template_nct_6a_1
.. autofunction:: template_nct_6a_2
.. autofunction:: template_nct_6a_3
.. autofunction:: template_nct_6a_4
.. autofunction:: template_nct_6b_1
.. autofunction:: template_nct_6b_2
.. autofunction:: template_nct_6c_1
.. autofunction:: template_nct_7a_1
.. autofunction:: template_nct_7b_1
.. autofunction:: template_nct_7c_1
.. autofunction:: template_nct_7d_1
.. autofunction:: template_nct_7e_1
.. autofunction:: template_nct_9a_1
.. autofunction:: template_nct_9c_1
.. autofunction:: template_nct_9c_2
.. autofunction:: template_nct_9c_3
.. autofunction:: template_nct_9c_4
.. autofunction:: template_nct_9c_5
.. autofunction:: template_nct_9c_6
.. autofunction:: template_nct_9c_7
.. autofunction:: template_nct_9c_8
.. autofunction:: template_nct_9c_9
.. autofunction:: template_nct_9c_10
.. autofunction:: template_nct_9c_11
.. autofunction:: template_nct_9c_12
.. autofunction:: template_nct_9d_1
.. autofunction:: template_nct_9d_2
.. autofunction:: template_nct_9d_3
.. autofunction:: template_nct_9d_4
.. autofunction:: template_nct_9d_5
.. autofunction:: template_nct_9d_6
.. autofunction:: template_nct_9d_7
.. autofunction:: template_nct_9d_8
.. autofunction:: template_nct_9d_9
.. autofunction:: template_nct_9d_10
.. currentmodule:: qiskit.circuit.library

Clifford template circuits
--------------------------

Template circuits over Clifford gates.

.. autofunction:: clifford_2_1
.. autofunction:: clifford_2_2
.. autofunction:: clifford_2_3
.. autofunction:: clifford_2_4
.. autofunction:: clifford_3_1
.. autofunction:: clifford_4_1
.. autofunction:: clifford_4_2
.. autofunction:: clifford_4_3
.. autofunction:: clifford_4_4
.. autofunction:: clifford_5_1
.. autofunction:: clifford_6_1
.. autofunction:: clifford_6_2
.. autofunction:: clifford_6_3
.. autofunction:: clifford_6_4
.. autofunction:: clifford_6_5
.. autofunction:: clifford_8_1
.. autofunction:: clifford_8_2
.. autofunction:: clifford_8_3

RZXGate template circuits
-------------------------

Template circuits with :class:`~qiskit.circuit.library.RZXGate`.

.. autofunction:: rzx_yz
.. autofunction:: rzx_xz
.. autofunction:: rzx_cy
.. autofunction:: rzx_zz1
.. autofunction:: rzx_zz2
.. autofunction:: rzx_zz3

"""

from .standard_gates import *
from .templates import *
from ..barrier import Barrier
from ..measure import Measure
from ..reset import Reset
from .blueprintcircuit import BlueprintCircuit
from .generalized_gates import *
from .pauli_evolution import PauliEvolutionGate
from .pauli_product_measurement import PauliProductMeasurement
from .hamiltonian_gate import HamiltonianGate
from .boolean_logic import *
from .basis_change import *
from .arithmetic import *
from . import n_local as mod_n_local  # avoid conflict with the function called `n_local`
from .n_local import *
from .data_preparation import *
from .quantum_volume import QuantumVolume, quantum_volume
from .fourier_checking import FourierChecking, fourier_checking
from .graph_state import GraphState, GraphStateGate
from .hidden_linear_function import HiddenLinearFunction, hidden_linear_function
from .iqp import IQP, iqp, random_iqp
from .phase_estimation import PhaseEstimation, phase_estimation
from .grover_operator import GroverOperator, grover_operator
from .phase_oracle import PhaseOracle, PhaseOracleGate
from .bit_flip_oracle import BitFlipOracleGate
from .overlap import UnitaryOverlap, unitary_overlap

from . import (
    standard_gates,
    templates,
    generalized_gates,
    boolean_logic,
    basis_change,
    arithmetic,
    data_preparation,
)

__all__ = [
    "IQP",
    "Barrier",
    "BitFlipOracleGate",
    "BlueprintCircuit",
    "FourierChecking",
    "GraphState",
    "GraphStateGate",
    "GroverOperator",
    "HamiltonianGate",
    "HiddenLinearFunction",
    "Measure",
    "PauliEvolutionGate",
    "PauliProductMeasurement",
    "PhaseEstimation",
    "PhaseOracle",
    "PhaseOracleGate",
    "QuantumVolume",
    "Reset",
    "UnitaryOverlap",
    "fourier_checking",
    "grover_operator",
    "hidden_linear_function",
    "iqp",
    "phase_estimation",
    "quantum_volume",
    "random_iqp",
    "unitary_overlap",
]
__all__ += standard_gates.__all__
__all__ += templates.__all__
__all__ += generalized_gates.__all__
__all__ += boolean_logic.__all__
__all__ += basis_change.__all__
__all__ += arithmetic.__all__
__all__ += mod_n_local.__all__
__all__ += data_preparation.__all__
