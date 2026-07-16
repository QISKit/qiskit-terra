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
from math import ceil
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import HGate, CU1Gate
from qiskit._accelerate.synthesis.multi_controlled import (
    c3x as c3x_rs,
    c4x as c4x_rs,
    synth_mcx_n_dirty_i15 as synth_mcx_n_dirty_i15_rs,
    synth_mcx_noaux_v24 as synth_mcx_noaux_v24_rs,
    synth_mcx_noaux_hp24 as synth_mcx_noaux_hp24_rs,
)
from .gray_code import gray_code_chain


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


def _synth_mcx_special_cases(num_ctrl_qubits: int) -> QuantumCircuit:
    """Internal function that produces default MCX circuits when num_ctrl_qubits is 0, 1, or 2."""
    if num_ctrl_qubits == 0:
        qc = QuantumCircuit(1)
        qc.x(0)
        return qc

    elif num_ctrl_qubits == 1:
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        return qc

    elif num_ctrl_qubits == 2:
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        return qc

    else:
        raise QiskitError(
            "_synth_mcx_special_cases should be called with only 0, 1, or 2 controls."
        )


def synth_mcx_n_clean_m15(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`k - 2`
    clean ancillary qubits with producing a circuit with :math:`2 * k - 1` qubits
    and at most :math:`6 * k - 6` CX gates, by Maslov [1].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    Raises:
        QiskitError: if ``num_ctrl_qubits`` is illegal.

    References:
        1. Maslov., Phys. Rev. A 93, 022311 (2016),
           `arXiv:1508.03273 <https://arxiv.org/pdf/1508.03273>`_
    """
    if num_ctrl_qubits < 0:
        raise QiskitError(
            "synth_mcx_n_clean_m15 cannot be called with a negative number of control qubits."
        )

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    num_qubits = 2 * num_ctrl_qubits - 1
    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q)
    q_controls = q[:num_ctrl_qubits]
    q_target = q[num_ctrl_qubits]
    q_ancillas = q[num_ctrl_qubits + 1 :]

    qc.rccx(q_controls[0], q_controls[1], q_ancillas[0])
    i = 0
    for j in range(2, num_ctrl_qubits - 1):
        qc.rccx(q_controls[j], q_ancillas[i], q_ancillas[i + 1])

        i += 1

    qc.ccx(q_controls[-1], q_ancillas[i], q_target)

    for j in reversed(range(2, num_ctrl_qubits - 1)):
        qc.rccx(q_controls[j], q_ancillas[i - 1], q_ancillas[i])

        i -= 1

    qc.rccx(q_controls[0], q_controls[1], q_ancillas[i])

    return qc


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

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    if num_ctrl_qubits == 3:
        return synth_c3x()

    elif num_ctrl_qubits == 4:
        return synth_c4x()

    num_qubits = num_ctrl_qubits + 2
    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q)

    num_ctrl_qubits = len(q) - 1
    q_ancilla = q[-1]
    q_target = q[-2]
    middle = ceil(num_ctrl_qubits / 2)

    # The construction involving 4 MCX gates is described in Lemma 7.3 of [1], and also
    # appears as Lemma 9 in [2]. The optimization that the first and third MCX gates
    # can be synthesized up to relative phase follows from Lemma 7 in [2], as a diagonal
    # gate following the first MCX gate commutes with the second MCX gate, and
    # thus cancels with the inverse diagonal gate preceding the third MCX gate. The
    # same optimization cannot be applied to the second MCX gate, since a diagonal
    # gate following the second MCX gate would not satisfy the preconditions of Lemma 7,
    # and would not necessarily commute with the third MCX gate.
    controls1 = [*q[:middle]]
    mcx1 = synth_mcx_n_dirty_i15(num_ctrl_qubits=len(controls1), relative_phase=True)
    qubits1 = [*controls1, q_ancilla, *q[middle : middle + mcx1.num_qubits - len(controls1) - 1]]

    controls2 = [*q[middle : num_ctrl_qubits - 1], q_ancilla]
    mcx2 = synth_mcx_n_dirty_i15(num_ctrl_qubits=len(controls2))
    qc2_qubits = [*controls2, q_target, *q[0 : mcx2.num_qubits - len(controls2) - 1]]

    qc.compose(mcx1, qubits1, inplace=True)
    qc.compose(mcx2, qc2_qubits, inplace=True)
    qc.compose(mcx1.inverse(), qubits1, inplace=True)
    qc.compose(mcx2, qc2_qubits, inplace=True)

    return qc


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

    circ = QuantumCircuit._from_circuit_data(synth_mcx_noaux_v24_rs(num_ctrl_qubits))
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


def _n_parallel_ccx_x(n: int, apply_x: bool = True) -> QuantumCircuit:
    r"""
    Construct a quantum circuit for creating n-conditionally clean ancillae using 3n qubits. This
    implements Fig. 4a of [1]. The circuit applies n relative CCX (RCCX) gates . If apply_x is True,
    each RCCX gate is preceded by an X gate on the target qubit. The order of returned qubits is
    qr_a, qr_b, qr_target.

    Args:
        n: Number of conditionally clean ancillae to create.
        apply_x: If True, apply X gate to the target qubit.

    Returns:
        QuantumCircuit: The quantum circuit for creating n-conditionally clean ancillae.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    n_qubits = 3 * n
    q = QuantumRegister(n_qubits, name="q")
    qc = QuantumCircuit(q, name=f"ccxn_{n}")
    qr_a, qr_b, qr_target = q[:n], q[n : 2 * n], q[2 * n :]

    if apply_x:
        qc.x(qr_target)

    qc.rccx(qr_a, qr_b, qr_target)

    return qc


def _linear_depth_ladder_ops(num_ladder_qubits: int) -> tuple[QuantumCircuit, list[int]]:
    r"""
    Helper function to create linear-depth ladder operations used in Khattar and Gidney's MCX synthesis.
    In particular, this implements Step-1 and Step-2 on Fig. 3 of [1] except for the first and last
    CCX gates.

    Args:
        num_ladder_qubits: No. of qubits involved in the ladder operation.

    Returns:
        A tuple consisting of the linear-depth ladder circuit and the index of control qubit to
        apply the final CCX gate.

    Raises:
        QiskitError: If num_ladder_qubits <= 2.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    if num_ladder_qubits <= 2:
        raise QiskitError("n_ctrls >= 3 to use MCX ladder. Otherwise, use CCX")

    n = num_ladder_qubits + 1
    qc = QuantumCircuit(n)
    qreg = list(range(n))

    # up-ladder
    for i in range(2, n - 2, 2):
        qc.rccx(qreg[i + 1], qreg[i + 2], qreg[i])
        qc.x(qreg[i])

    # down-ladder
    if n % 2 != 0:
        a, b, target = n - 3, n - 5, n - 6
    else:
        a, b, target = n - 1, n - 4, n - 5

    if target > 0:
        qc.rccx(qreg[a], qreg[b], qreg[target])
        qc.x(qreg[target])

    for i in range(target, 2, -2):
        qc.rccx(qreg[i], qreg[i - 1], qreg[i - 2])
        qc.x(qreg[i - 2])

    mid_second_ctrl = 1 + max(0, 6 - n)
    final_ctrl = qreg[mid_second_ctrl] - 1
    return qc, final_ctrl


def synth_mcx_1_kg24(num_ctrl_qubits: int, clean: bool = True) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`1` ancillary qubit as
    described in Sec. 5 of [1].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

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

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    q_controls = QuantumRegister(num_ctrl_qubits, name="ctrl")
    q_target = QuantumRegister(1, name="targ")
    q_ancilla = AncillaRegister(1, name="anc")
    qc = QuantumCircuit(q_controls, q_target, q_ancilla, name="mcx_linear_depth")

    ladder_ops, final_ctrl = _linear_depth_ladder_ops(num_ctrl_qubits)

    qc.rccx(q_controls[0], q_controls[1], q_ancilla[0])  #              # create cond. clean ancilla
    qc.compose(ladder_ops, q_ancilla[:] + q_controls[:], inplace=True)  # up-ladder
    qc.ccx(q_ancilla, q_controls[final_ctrl], q_target)  #              # target
    qc.compose(  #                                                      # down-ladder
        ladder_ops.inverse(),
        q_ancilla[:] + q_controls[:],
        inplace=True,
    )
    qc.rccx(q_controls[0], q_controls[1], q_ancilla[0])  #              # undo cond. clean ancilla

    if not clean:
        # perform toggle-detection if ancilla is dirty
        qc.compose(ladder_ops, q_ancilla[:] + q_controls[:], inplace=True)
        qc.ccx(q_ancilla, q_controls[final_ctrl], q_target)
        qc.compose(ladder_ops.inverse(), q_ancilla[:] + q_controls[:], inplace=True)

    return qc


def synth_mcx_1_clean_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`1` clean
    ancillary qubit producing a circuit with :math:`2k-3` Toffoli gates or :math:`6k-6`
    CX gates and depth :math:`O(k)` as described in Sec. 5.1 of [1].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

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

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    return synth_mcx_1_kg24(num_ctrl_qubits, clean=True)


def synth_mcx_1_dirty_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`1` dirty
    ancillary qubit producing a circuit with :math:`4k-8` Toffoli gates or :math:`12k-18`
    CX gates and depth :math:`O(k)` as described in Sec. 5.3 of [1].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

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

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    return synth_mcx_1_kg24(num_ctrl_qubits, clean=False)


def _build_logn_depth_ccx_ladder(
    ancilla_idx: int, ctrls: list[int], skip_cond_clean: bool = False
) -> tuple[QuantumCircuit, list[int]]:
    r"""
    Helper function to build a log-depth ladder composed of CCX and X gates as shown in Fig. 4b of [1].

    Args:
        ancilla_idx: Index of the ancillary qubit.
        ctrls: List of control qubits.
        skip_cond_clean: If True, do not include the conditionally clean ancilla (step 1 and 5 in
            Fig. 4b of [1]).

    Returns:
        A tuple consisting of the log-depth ladder circuit of conditionally clean ancillae and the
        list of indices of control qubit to apply the linear-depth MCX gate.

    Raises:
        QiskitError: If no. of qubits in parallel CCX + X gates are not the same.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    qc = QuantumCircuit(len(ctrls) + 1)
    anc = [ancilla_idx]
    final_ctrls = []

    while len(ctrls) > 1:
        next_batch_len = min(len(anc) + 1, len(ctrls))
        ctrls, nxt_batch = ctrls[next_batch_len:], ctrls[:next_batch_len]
        new_anc = []
        while len(nxt_batch) > 1:
            ccx_n = len(nxt_batch) // 2
            st = int(len(nxt_batch) % 2)
            ccx_x, ccx_y, ccx_t = (
                nxt_batch[st : st + ccx_n],
                nxt_batch[st + ccx_n :],
                anc[-ccx_n:],
            )
            if not len(ccx_x) == len(ccx_y) == ccx_n >= 1:
                raise QiskitError(
                    f"Invalid CCX gate parameters: {len(ccx_x)=} != {len(ccx_y)=} != {len(ccx_n)=}"
                )
            if ccx_t != [ancilla_idx]:
                qc.compose(_n_parallel_ccx_x(ccx_n), ccx_x + ccx_y + ccx_t, inplace=True)
            elif not skip_cond_clean:
                qc.rccx(ccx_x[0], ccx_y[0], ccx_t[0])  # # create conditionally clean ancilla

            new_anc += nxt_batch[st:]  #                     # newly created cond. clean ancilla
            nxt_batch = ccx_t + nxt_batch[:st]
            anc = anc[:-ccx_n]

        anc = sorted(anc + new_anc)
        final_ctrls += nxt_batch

    final_ctrls += ctrls
    final_ctrls = sorted(final_ctrls)
    return qc, final_ctrls[:-1]  # exclude ancilla


def synth_mcx_2_kg24(num_ctrl_qubits: int, clean: bool = True) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`2` ancillary qubits.
    as described in Sec. 5 of [1].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

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
            "synth_mcx_2_kg24 cannot be called with a negative number of control qubits."
        )

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    q_control = QuantumRegister(num_ctrl_qubits, name="ctrl")
    q_target = QuantumRegister(1, name="targ")
    q_ancilla = AncillaRegister(2, name="anc")
    qc = QuantumCircuit(q_control, q_target, q_ancilla, name="mcx_logn_depth")

    ladder_ops, final_ctrls = _build_logn_depth_ccx_ladder(
        num_ctrl_qubits, list(range(num_ctrl_qubits))
    )
    qc.compose(ladder_ops, q_control[:] + [q_ancilla[0]], inplace=True)
    if len(final_ctrls) == 1:  # Already a toffoli
        qc.ccx(q_ancilla[0], q_control[final_ctrls[0]], q_target)
    else:
        mid_mcx = synth_mcx_1_clean_kg24(len(final_ctrls) + 1)
        qc.compose(
            mid_mcx,
            [q_ancilla[0]]
            + q_control[final_ctrls]
            + q_target[:]
            + [q_ancilla[1]],  # ctrls, targ, anc
            inplace=True,
        )
    qc.compose(ladder_ops.inverse(), q_control[:] + [q_ancilla[0]], inplace=True)

    if not clean:
        # perform toggle-detection if ancilla is dirty
        ladder_ops_new, final_ctrls = _build_logn_depth_ccx_ladder(
            num_ctrl_qubits, list(range(num_ctrl_qubits)), skip_cond_clean=True
        )
        qc.compose(ladder_ops_new, q_control[:] + [q_ancilla[0]], inplace=True)
        if len(final_ctrls) == 1:
            qc.ccx(q_ancilla[0], q_control[final_ctrls[0]], q_target)
        else:
            qc.compose(
                mid_mcx,
                [q_ancilla[0]] + q_control[final_ctrls] + q_target[:] + [q_ancilla[1]],
                inplace=True,
            )
        qc.compose(ladder_ops_new.inverse(), q_control[:] + [q_ancilla[0]], inplace=True)

    return qc


def synth_mcx_2_clean_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`2` clean
    ancillary qubits producing a circuit with :math:`2k-3` Toffoli gates or :math:`6k-6`
    CX gates and depth :math:`O(\log(k))` as described in Sec. 5.2 of [1].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

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

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    return synth_mcx_2_kg24(num_ctrl_qubits, clean=True)


def synth_mcx_2_dirty_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`2` dirty
    ancillary qubits producing a circuit with :math:`4k-8` Toffoli gates or :math:`12k-18` CX
    gates and depth :math:`O(\log(k))` as described in Sec. 5.4 of [1].
    For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
    (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.

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

    if num_ctrl_qubits <= 2:
        return _synth_mcx_special_cases(num_ctrl_qubits)

    return synth_mcx_2_kg24(num_ctrl_qubits, clean=False)


def synth_c3x() -> QuantumCircuit:
    """Efficient synthesis of 3-controlled X-gate."""
    return QuantumCircuit._from_circuit_data(c3x_rs())


def synth_c4x() -> QuantumCircuit:
    """Efficient synthesis of 4-controlled X-gate."""
    return QuantumCircuit._from_circuit_data(c4x_rs())
