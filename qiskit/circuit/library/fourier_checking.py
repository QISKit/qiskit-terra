# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fourier checking circuit."""

from collections.abc import Sequence
import math

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.deprecation import deprecate_func

from .generalized_gates.diagonal import Diagonal, DiagonalGate


class FourierChecking(QuantumCircuit):
    """Fourier checking circuit.

    The circuit for the Fourier checking algorithm, introduced in [1],
    involves a layer of Hadamards, the function :math:`f`, another layer of
    Hadamards, the function :math:`g`, followed by a final layer of Hadamards.
    The functions :math:`f` and :math:`g` are classical functions realized
    as phase oracles (diagonal operators with {-1, 1} on the diagonal).

    The probability of observing the all-zeros string is :math:`p(f,g)`.
    The algorithm solves the promise Fourier checking problem,
    which decides if f is correlated with the Fourier transform
    of g, by testing if :math:`p(f,g) <= 0.01` or :math:`p(f,g) >= 0.05`,
    promised that one or the other of these is true.

    The functions :math:`f` and :math:`g` are currently implemented
    from their truth tables but could be represented concisely and
    implemented efficiently for special classes of functions.

    Fourier checking is a special case of :math:`k`-fold forrelation [2].

    References:

    [1] S. Aaronson, BQP and the Polynomial Hierarchy, 2009 (Section 3.2).
    `arXiv:0910.4698 <https://arxiv.org/abs/0910.4698>`_

    [2] S. Aaronson, A. Ambainis, Forrelation: a problem that
    optimally separates quantum from classical computing, 2014.
    `arXiv:1411.5729 <https://arxiv.org/abs/1411.5729>`_
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use qiskit.circuit.library.fourier_checking instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, f: Sequence[int], g: Sequence[int]) -> None:
        """Create Fourier checking circuit.

        Args:
            f: truth table for f, length 2**n list of {1,-1}.
            g: truth table for g, length 2**n list of {1,-1}.

        Raises:
            CircuitError: if the inputs f and g are not valid.

        Reference Circuit:
            .. plot::
               :alt: Diagram illustrating the previously described circuit.

               from qiskit.circuit.library import FourierChecking
               from qiskit.visualization.library import _generate_circuit_library_visualization
               f = [1, -1, -1, -1]
               g = [1, 1, -1, -1]
               circuit = FourierChecking(f, g)
               _generate_circuit_library_visualization(circuit)
        """
        num_qubits = math.log2(len(f))

        if len(f) != len(g) or num_qubits == 0 or not num_qubits.is_integer():
            raise CircuitError(
                "The functions f and g must be given as truth "
                "tables, each as a list of 2**n entries of "
                "{1, -1}."
            )

        # This definition circuit is not replaced by the circuit produced by fourier_checking,
        # as the latter produces a slightly different circuit, with DiagonalGates instead
        # of Diagonal circuits.
        circuit = QuantumCircuit(int(num_qubits), name=f"fc: {f}, {g}")
        circuit.h(circuit.qubits)
        circuit.compose(Diagonal(f), inplace=True)
        circuit.h(circuit.qubits)
        circuit.compose(Diagonal(g), inplace=True)
        circuit.h(circuit.qubits)
        super().__init__(*circuit.qregs, name=circuit.name)
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)


def fourier_checking(f: Sequence[int], g: Sequence[int]) -> QuantumCircuit:
    """Fourier checking circuit.

    The circuit for the Fourier checking algorithm, introduced in [1],
    involves a layer of Hadamards, the function :math:`f`, another layer of
    Hadamards, the function :math:`g`, followed by a final layer of Hadamards.
    The functions :math:`f` and :math:`g` are classical functions realized
    as phase oracles (diagonal operators with {-1, 1} on the diagonal).

    The probability of observing the all-zeros string is :math:`p(f,g)`.
    The algorithm solves the promise Fourier checking problem,
    which decides if f is correlated with the Fourier transform
    of g, by testing if :math:`p(f,g) <= 0.01` or :math:`p(f,g) >= 0.05`,
    promised that one or the other of these is true.

    The functions :math:`f` and :math:`g` are currently implemented
    from their truth tables but could be represented concisely and
    implemented efficiently for special classes of functions.

    Fourier checking is a special case of :math:`k`-fold forrelation [2].

    Reference Circuit:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit.circuit.library import fourier_checking
       circuit = fourier_checking([1, -1, -1, -1], [1, 1, -1, -1])
       circuit.draw('mpl')

    References:

    [1] S. Aaronson, BQP and the Polynomial Hierarchy, 2009 (Section 3.2).
    `arXiv:0910.4698 <https://arxiv.org/abs/0910.4698>`_

    [2] S. Aaronson, A. Ambainis, Forrelation: a problem that
    optimally separates quantum from classical computing, 2014.
    `arXiv:1411.5729 <https://arxiv.org/abs/1411.5729>`_
    """
    num_qubits = math.log2(len(f))

    if len(f) != len(g) or num_qubits == 0 or not num_qubits.is_integer():
        raise CircuitError(
            "The functions f and g must be given as truth "
            "tables, each as a list of 2**n entries of "
            "{1, -1}."
        )
    num_qubits = int(num_qubits)

    circuit = QuantumCircuit(num_qubits, name=f"fc: {f}, {g}")
    circuit.h(circuit.qubits)
    circuit.append(DiagonalGate(f), range(num_qubits))
    circuit.h(circuit.qubits)
    circuit.append(DiagonalGate(g), range(num_qubits))
    circuit.h(circuit.qubits)
    return circuit
