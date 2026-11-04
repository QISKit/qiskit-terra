# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Module containing multi-controlled phase gate synthesis methods."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.exceptions import QiskitError
from qiskit._accelerate.synthesis.multi_controlled import (
    synth_mcp_noaux_sp22 as synth_mcp_noaux_sp22_rs,
)


def synth_mcp_noaux_v24(num_ctrl_qubits: int, phase: ParameterValueType) -> QuantumCircuit:
    r"""Synthesize a multi-controlled phase gate with no auxiliary qubits.

    This method implements the synthesis of a multi-controlled phase gate
    using multi-controlled RZ gates.

    The circuit depth is :math:`O(n^2)` and the total number of CX gates used
    is :math:`8n^2-16n-60` for :math:`n \ge 7` control qubits,
    which is worse than :func:`synth_mcp_noaux_sp22` for :math:`n \ge 5`,
    but better for :math:`n \le 4`.

    The number of CX gates required can be derived as follows:

    - Each :math:`m`-controlled RZ gate uses at most

      .. math::

        N(m) = 2 \cdot C_{mcx}\left(\lceil \frac{m}{2} \rceil\right)
        + 2 \cdot C_{mcx}\left(\lfloor \frac{m}{2} \rfloor\right)

      CX gates for :math:`m \ge 2` and 2 for :math:`m = 1`, where
      :math:`C_{mcx}(k)` is the number of CX gates required to synthesize a
      :math:`k`-controlled X gate.
    - Synthesizing a :math:`k`-controlled X gate requires

      .. math::

        C_{mcx}(k) = \begin{cases}
        1 & (k = 1) \\
        6 & (k = 2) \\
        14 & (k = 3) \\
        8k - 6 & (k \ge 4)
        \end{cases}

      CX gates, which can be derived from the implementation of :func:`synth_mcx_n_dirty_i15`.
    - Thus, for :math:`1 \le m \le 7`,

      .. math::

        N(m) = \begin{cases}
        2 & (m = 1) \\
        4 & (m = 2) \\
        14 & (m = 3) \\
        24 & (m = 4) \\
        40 & (m = 5) \\
        56 & (m = 6) \\
        80 & (m = 7)
        \end{cases}

      and for :math:`m \ge 8`,

      .. math::

        N(m) = 2 \left( 8 \left\lceil \frac{m}{2} \right\rceil - 6 \right)
        + 2 \left( 8 \left\lfloor \frac{m}{2} \right\rfloor - 6 \right) = 16m-24

    - Multi-controlled phase gates can be synthesized by applying :math:`m`-controlled RZ gates for
      :math:`m = 1, 2, \ldots, n`, where :math:`n` is the number of control qubits [1].
      Therefore, the total number of CX gates used to synthesize a multi-controlled phase
      gate with :math:`n \ge 7` control qubits is

      .. math::

        \sum_{m=1}^{n} N(m) = 2 + 4 + 14 + 24 + 40 + 56 + 80 + \sum_{m=8}^{n} (16m-24)
        = 8n^2 - 16n - 60.

    Args:
        num_ctrl_qubits: The number of control qubits.
        phase: The phase angle for the multi-controlled phase gate.

    Returns:
        A QuantumCircuit implementing the multi-controlled phase gate.

    Raises:
        QiskitError: If the number of control qubits is negative.

    References:
        [1] Vale et. al.,
        Circuit Decomposition of Multicontrolled Special Unitary Single-Qubit Gates,
        IEEE TCAD 43(3) (2024), `arXiv:2302.06377 <https://arxiv.org/abs/2302.06377>`__.

    """
    qc = QuantumCircuit(num_ctrl_qubits + 1)

    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcp_noaux_v24 cannot be called with a negative number of control qubits."
        )
    elif num_ctrl_qubits == 0:
        qc.p(phase, 0)
    elif num_ctrl_qubits == 1:
        qc.cp(phase, 0, 1)
    else:
        q_controls = list(range(num_ctrl_qubits))
        q_target = num_ctrl_qubits
        new_target = q_target
        for k in range(num_ctrl_qubits):
            # Note: it's better *not* to run transpile recursively
            qc.mcrz(phase / (2**k), q_controls, new_target, use_basis_gates=False)
            new_target = q_controls.pop()
        qc.p(phase / (2**num_ctrl_qubits), new_target)

    return qc


def synth_mcp_noaux_sp22(num_ctrl_qubits: int, phase: ParameterValueType) -> QuantumCircuit:
    r"""Synthesize a multi-controlled phase gate with :math:`n` controls based on the paper
    by da Silva et al. [1] and the implementation in qclib [2].

    For :math:`n \ge 2`, the method produces a circuit with :math:`4n^2-4n+2` CX gates
    and requires :math:`O(n)` depth.
    For :math:`n \le 4`, it is more efficient to use :func:`synth_mcp_noaux_v24`,
    which produces a circuit with less CX gates.

    The circuit breaks down into four steps, each applying a specific pattern of controlled gates.

    - Step 1: Apply :math:`n` controlled phase gates and :math:`n(n-1)/2` controlled RX gates.
    - Step 2: Apply :math:`n-1` controlled phase gates and :math:`(n-1)(n-2)/2` controlled RX gates.

      This is the initial phase. It applies angle rotations (e.g., :math:`R_X(\pi/k)`) and
      the :math:`k`-th roots of the target unitary (:math:`U^{1/k}`) in a cascading V-shape pattern.
      This step systematically accumulates the partitioned components of the unitary
      operation on the target qubit based on the control states.

    - Step 3: Apply :math:`n(n-1)/2` controlled RX gates.
    - Step 4: Apply :math:`(n-1)(n-2)/2` controlled RX gates.

      Steps 3 and 4 together constitute the uncomputation process. By applying only the inverse of
      the angle rotation operations in steps 1 and 2, these steps reverse the unwanted
      entanglement and phase shifts generated in the first two steps. This cancellation ensures
      that the target qubit undergoes the full unitary operation :math:`U`
      if and only if all control qubits are in the :math:`|1\rangle` state.

    Each controlled RX gate and controlled phase gate requires two CX gates,
    resulting in a total of :math:`4n^2-4n+2` CX gates.

    Args:
        num_ctrl_qubits: The number of control qubits.
        phase: The phase angle for the multi-controlled phase gate.

    Returns:
        A QuantumCircuit implementing the multi-controlled phase gate.

    Raises:
        QiskitError: If the number of control qubits is negative.

    References:

    [1] A. J. da Silva and D. K. Park,
    Linear-depth quantum circuits for multiqubit controlled gates,
    `Phys. Rev. A 106, 042602
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042602>`__.

    [2] https://github.com/qclib/qclib/blob/master/qclib/gates/ldmcu.py
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcp_noaux_sp22 cannot be called with a negative number of control qubits."
        )
    return QuantumCircuit._from_circuit_data(synth_mcp_noaux_sp22_rs(num_ctrl_qubits, phase))


def synth_mcp_noaux_default(num_ctrl_qubits: int, phase: ParameterValueType) -> QuantumCircuit:
    """Choose the best synthesis code for :class:`.MCPhaseGate` according to the number of control qubits.

    Args:
        num_ctrl_qubits: The number of control qubits.
        phase: The phase angle for the multi-controlled phase gate.

    Returns:
        A QuantumCircuit implementing the multi-controlled phase gate.

    Raises:
        QiskitError: If the number of control qubits is negative.
    """
    qc = QuantumCircuit(num_ctrl_qubits + 1)

    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcp_noaux_default cannot be called with a negative number of control qubits."
        )
    elif num_ctrl_qubits == 0:
        qc.p(phase, 0)
    elif num_ctrl_qubits == 1:
        qc.cp(phase, 0, 1)
    elif num_ctrl_qubits <= 4:
        qc = synth_mcp_noaux_v24(num_ctrl_qubits, phase)
    else:
        qc = synth_mcp_noaux_sp22(num_ctrl_qubits, phase)

    return qc
