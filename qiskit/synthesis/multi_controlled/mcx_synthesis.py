# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing multi-controlled circuits synthesis with and without ancillary qubits."""

from __future__ import annotations
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import HGate, CU1Gate
from qiskit._accelerate.synthesis.multi_controlled import (
    c3x as c3x_rs,
    c4x as c4x_rs,
    synth_mcx_explicit as synth_mcx_explicit_rs,
    synth_mcx_n_dirty_i15 as synth_mcx_n_dirty_i15_rs,
    synth_mcx_n_dirty_m15 as synth_mcx_n_dirty_m15_rs,
    synth_mcx_noaux_hp24 as synth_mcx_noaux_hp24_rs,
    synth_mcx_n_clean_m15 as synth_mcx_n_clean_m15_rs,
    synth_mcx_1_clean_b95 as synth_mcx_1_clean_b95_rs,
    synth_mcx_noaux_sp22 as synth_mcx_noaux_sp22_rs,
    synth_mcx_1_kg24 as synth_mcx_1_kg24_rs,
    synth_mcx_2_kg24 as synth_mcx_2_kg24_rs,
)
from .gray_code import gray_code_chain
from qiskit.synthesis.multi_controlled.mcp_synthesis import (
    synth_mcp_noaux_v24,
)


def synth_mcx_n_dirty_i15(
    num_ctrl_qubits: int,
    relative_phase: bool = False,
    action_only: bool = False,
) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k` controls based on the paper
    by Iten et al. [1].

    For :math:`k\ge 4`, the method uses :math:`k - 2` dirty ancillary qubits, producing a circuit
    with :math:`2 * k - 1` qubits and at most :math:`8 * k - 6` CX gates. For :math:`k\le 3`,
    explicitly constructed efficient circuits that require no ancillary qubits are used instead.

    Args:
        num_ctrl_qubits: The number of control qubits.

        relative_phase: when set to ``True``, the method applies the optimized multi-controlled X gate
            up to a relative phase, in a way that, by lemma 8 of [1], the relative
            phases of the ``action part`` cancel out with the phases of the ``reset part``.

        action_only: when set to ``True``, the method applies only the ``action part`` of lemma 8 of [1].

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Iten et. al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
           `arXiv:1501.06911 <https://arxiv.org/abs/1501.06911>`_
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_n_dirty_i15 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_n_dirty_i15_rs(num_ctrl_qubits, relative_phase, action_only)
    )


def synth_mcx_n_dirty_m15(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k` controls using dirty ancillary
    qubits, following circuit (5) for three controls and Proposition 5 for four or more
    controls in Maslov [1].

    For :math:`k = 3`, the method uses one dirty ancilla and produces a circuit with
    16 T gates and 14 CX gates. For :math:`k \ge 4`, it uses
    :math:`\lceil (k - 2) / 2 \rceil` dirty ancillas and produces a circuit with
    :math:`8k - 8` T gates and :math:`8k - 12` CX gates. For :math:`k \le 2`, no
    ancillas are used.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application
           to multiple control Toffoli optimization*, Phys. Rev. A 93, 022311 (2016),
           `arXiv:1508.03273 <https://arxiv.org/abs/1508.03273>`_
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_n_dirty_m15 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_n_dirty_m15_rs(num_ctrl_qubits), legacy_qubits=True
    )


def _synth_mcx_special_cases(num_ctrl_qubits: int) -> QuantumCircuit:
    """Internal function that produces default MCX circuits when num_ctrl_qubits is <=4."""
    return QuantumCircuit._from_circuit_data(synth_mcx_explicit_rs(num_ctrl_qubits))


def synth_mcx_n_clean_m15(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate following Proposition 4 of Maslov [1], using
    :math:`k\ge 3` controls and
    :math:`\lceil(k - 2) / 2\rceil` clean ancillary qubits, producing a circuit with
    :math:`8 * k - 9` T gates and :math:`6 * k - 6` CX gates.
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application
           to multiple control Toffoli optimization*, Phys. Rev. A 93, 022311 (2016),
           `arXiv:1508.03273 <https://arxiv.org/abs/1508.03273>`_
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_n_clean_m15 cannot be called with a negative number of control qubits."
        )

    circ = QuantumCircuit._from_circuit_data(synth_mcx_n_clean_m15_rs(num_ctrl_qubits))
    return circ


def synth_mcx_1_clean_b95(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using a single
    clean ancillary qubit producing a circuit with :math:`k + 2` qubits and at most
    :math:`16 * k - 24` CX gates, by [1], [2].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Barenco et. al., *Elementary gates for quantum computation*, Phys.Rev. A52 3457 (1995),
           `arXiv:quant-ph/9503016 <https://arxiv.org/abs/quant-ph/9503016>`_
        2. Iten et. al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
           `arXiv:1501.06911 <https://arxiv.org/abs/1501.06911>`_
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_1_clean_b95 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(synth_mcx_1_clean_b95_rs(num_ctrl_qubits))


def synth_mcx_gray_code(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using the Gray code.

    Produces a quantum circuit with :math:`k + 1` qubits. This method
    produces exponentially many CX gates and should be used only for small
    values of :math:`k`.
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    Returns:
        The synthesized quantum circuit.
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_gray_code cannot be called with a negative number of control qubits."
        )

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    num_qubits = num_ctrl_qubits + 1
    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q)
    qc._append(HGate(), [q[-1]], [])
    scaled_lam = np.pi / (2 ** (num_ctrl_qubits - 1))
    bottom_gate = CU1Gate(scaled_lam)
    definition = gray_code_chain(q, num_ctrl_qubits, bottom_gate)
    for instr, qargs, cargs in definition:
        qc._append(instr, qargs, cargs)
    qc._append(HGate(), [q[-1]], [])
    return qc


def synth_mcx_noaux_sp22(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled :class:`.XGate` gate with :math:`k` controls based on
    the implementation for :class:`.MCPhaseGate`.

    In turn, the :class:`.MCPhaseGate` uses the decomposition for multi-controlled
    special unitaries described in the paper by da Silva et al. [1]
    and the implementation in qclib [2].

    Produces a quantum circuit with :math:`k + 1` qubits.
    The number of CX-gates is quadratic in :math:`k`.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        [1] A. J. da Silva and D. K. Park,
        Linear-depth quantum circuits for multiqubit controlled gates,
        `Phys. Rev. A 106, 042602
        <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042602>`__.

        [2] https://github.com/qclib/qclib/blob/master/qclib/gates/ldmcu.py
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_noaux_sp22 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_noaux_sp22_rs(num_ctrl_qubits), legacy_qubits=True
    )


def synth_mcx_noaux_v24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k` controls based on
    the implementation for MCPhaseGate.

    In turn, the MCPhase gate uses the decomposition for multi-controlled
    special unitaries described in [1].

    Produces a quantum circuit with :math:`k + 1` qubits.
    The number of CX-gates is quadratic in :math:`k`.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
           Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
           `arXiv:2302.06377 <https://arxiv.org/abs/2302.06377>`_
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_noaux_v24 cannot be called with a negative number of control qubits."
        )
    circ = QuantumCircuit(num_ctrl_qubits + 1)
    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)
    elif num_ctrl_qubits == 3:
        circ = synth_c3x()
    elif num_ctrl_qubits == 4:
        circ = synth_c4x()
    else:
        circ.h(num_ctrl_qubits)
        circ.compose(
            synth_mcp_noaux_v24(num_ctrl_qubits, phase=np.pi),
            range(num_ctrl_qubits + 1),
            inplace=True,
        )
        circ.h(num_ctrl_qubits)
    return circ


def synth_mcx_noaux_hp24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k` controls based on
    the work by Huang and Palsberg.

    Produces a quantum circuit with :math:`k + 1` qubits. The number of CX-gates
    is linear in :math:`k`.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Huang and Palsberg, *Compiling Conditional Quantum Gates without Using
           Helper Qubits*, PLDI (2024),
           <https://dl.acm.org/doi/10.1145/3656436>`_
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_noaux_hp24 cannot be called with a negative number of control qubits."
        )

    circ = QuantumCircuit._from_circuit_data(synth_mcx_noaux_hp24_rs(num_ctrl_qubits))
    return circ


def synth_mcx_1_kg24(num_ctrl_qubits: int, clean: bool = True) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`1` ancillary
    qubits, producing a circuit with depth :math:`O(k)` as described in Sec. 5 of [1].
    For :math:`k\le 2`, the returned circuit uses no ancillary qubits: it is a single
    X gate for :math:`k = 0`, a single CX gate for :math:`k = 1`, or the elementary-gate
    decomposition of a CCX gate for :math:`k = 2`.


    Args:
        num_ctrl_qubits: The number of control qubits.
        clean: If True, the ancilla is clean, otherwise it is dirty.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_1_kg24 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_1_kg24_rs(num_ctrl_qubits, clean), legacy_qubits=True
    )


def synth_mcx_1_clean_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`1` clean
    ancillary qubit and depth :math:`O(k)`, as described in Sec. 5.1 of [1]. The
    construction is equivalent to :math:`2k-3` Toffoli gates (mostly the cheaper
    relative-phase Toffoli, RCCX, plus one closing CCX); the returned circuit already
    contains their decomposition into elementary single- and two-qubit gates, for a
    total of :math:`6k-6` CX gates.
    For :math:`k\le 2`, the returned circuit uses no ancillary qubits: it is a single
    X gate for :math:`k = 0`, a single CX gate for :math:`k = 1`, or the elementary-gate
    decomposition of a CCX gate for :math:`k = 2`.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_1_clean_kg24 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_1_kg24_rs(num_ctrl_qubits, True), legacy_qubits=True
    )


def synth_mcx_1_dirty_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`1` dirty
    ancillary qubit and depth :math:`O(k)`, as described in Sec. 5.3 of [1]. The
    construction is equivalent to :math:`4k-8` Toffoli gates (mostly RCCX, plus closing
    CCX gates); the returned circuit already contains their decomposition into elementary
    single- and two-qubit gates, for a total of :math:`12k-18` CX gates.
    For :math:`k\le 2`, the returned circuit uses no ancillary qubits: it is a single
    X gate for :math:`k = 0`, a single CX gate for :math:`k = 1`, or the elementary-gate
    decomposition of a CCX gate for :math:`k = 2`.


    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_1_dirty_kg24 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_1_kg24_rs(num_ctrl_qubits, False), legacy_qubits=True
    )


def synth_mcx_2_kg24(num_ctrl_qubits: int, clean: bool = True) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`2` ancillary
    qubits, producing a circuit with depth :math:`O(\log(k))` as described in Sec. 5.2/5.4 of [1].
    For :math:`k\le 2`, the returned circuit uses no ancillary qubits: it is a single
    X gate for :math:`k = 0`, a single CX gate for :math:`k = 1`, or the elementary-gate
    decomposition of a CCX gate for :math:`k = 2`.

    Args:
        num_ctrl_qubits: The number of control qubits.
        clean: If True, both ancillas are clean, otherwise both are dirty.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_2_kg24 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_2_kg24_rs(num_ctrl_qubits, clean), legacy_qubits=True
    )


def synth_mcx_2_clean_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`2` clean
    ancillary qubits and depth :math:`O(\log k)`, as described in Sec. 5.2 of [1]. The
    construction is equivalent to :math:`2k-3` Toffoli gates (mostly RCCX, plus one closing
    CCX); the returned circuit already contains their decomposition into elementary
    single- and two-qubit gates, for a total of :math:`6k-6` CX gates.
    For :math:`k\le 2`, the returned circuit uses no ancillary qubits: it is a single
    X gate for :math:`k = 0`, a single CX gate for :math:`k = 1`, or the elementary-gate
    decomposition of a CCX gate for :math:`k = 2`.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_2_clean_kg24 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_2_kg24_rs(num_ctrl_qubits, True), legacy_qubits=True
    )


def synth_mcx_2_dirty_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`2` dirty
    ancillary qubits and depth :math:`O(\log k)`, as described in Sec. 5.4 of [1]. The
    construction is equivalent to :math:`4k-8` Toffoli gates (mostly RCCX, plus closing
    CCX gates); the returned circuit already contains their decomposition into elementary
    single- and two-qubit gates, for a total of :math:`12k-18` CX gates.
    For :math:`k\le 2`, the returned circuit uses no ancillary qubits: it is a single
    X gate for :math:`k = 0`, a single CX gate for :math:`k = 1`, or the elementary-gate
    decomposition of a CCX gate for :math:`k = 2`.


    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_2_dirty_kg24 cannot be called with a negative number of control qubits."
        )

    return QuantumCircuit._from_circuit_data(
        synth_mcx_2_kg24_rs(num_ctrl_qubits, False), legacy_qubits=True
    )


def synth_c3x() -> QuantumCircuit:
    """Efficient synthesis of 3-controlled X-gate."""
    return QuantumCircuit._from_circuit_data(c3x_rs())


def synth_c4x() -> QuantumCircuit:
    """Efficient synthesis of 4-controlled X-gate."""
    return QuantumCircuit._from_circuit_data(c4x_rs())
