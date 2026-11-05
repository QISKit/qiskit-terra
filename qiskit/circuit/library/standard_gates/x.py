# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""X, CX, CCX and multi-controlled X gates."""
from __future__ import annotations
import warnings
import numpy
from qiskit.circuit.annotated_operation import AnnotatedOperation
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
from qiskit._accelerate.circuit import StandardGate
from qiskit.utils.deprecation import deprecate_func

_X_ARRAY = [[0, 1], [1, 0]]
_SX_ARRAY = [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]


@with_gate_array(_X_ARRAY)
class XGate(SingletonGate):
    r"""The single-qubit Pauli-X gate (:math:`\sigma_x`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.x` method.

    Matrix representation:

    .. math::

        X = \begin{pmatrix}
                0 & 1 \\
                1 & 0
            \end{pmatrix}

    Circuit symbol:

    .. code-block:: text

             ┌───┐
        q_0: ┤ X ├
             └───┘

    Equivalent to a :math:`\pi` radian rotation about the X axis.

    .. note::

        A global phase difference exists between the definitions of
        :math:`RX(\pi)` and :math:`X`.

        .. math::

            RX(\pi) = \begin{pmatrix}
                        0 & -i \\
                        -i & 0
                      \end{pmatrix}
                    = -i X

    The gate is equivalent to a classical bit flip.

    .. math::

        |0\rangle \rightarrow |1\rangle \\
        |1\rangle \rightarrow |0\rangle
    """

    _standard_gate = StandardGate.X

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("x", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌──────────┐
        # q: ┤ U(π,0,π) ├
        #    └──────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.X._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the X gate.

        The controlled gate is implemented as :class:`.CXGate` for a single control qubit,
        as :class:`.CCXGate` for two control qubits, and as :class:`.MCXGate`
        for three or more control qubits. In each case, the value of ``annotated``
        is ignored.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        gate = MCXGate(
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _base_label=self.label,
        )
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted X gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            XGate: inverse gate (self-inverse).
        """
        return XGate()  # self-inverse

    def __eq__(self, other):
        return isinstance(other, XGate)


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=1)
class CXGate(SingletonControlledGate):
    r"""Controlled-X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cx` and
    :meth:`~qiskit.circuit.QuantumCircuit.cnot` methods.

    Circuit symbol:

    .. code-block:: text

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ X ├
             └───┘

    Matrix representation:

    .. math::

        CX\ q_0, q_1 =
            I \otimes |0\rangle\langle0| + X \otimes |1\rangle\langle1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──

        .. math::

            CX\ q_1, q_0 =
                |0 \rangle\langle 0| \otimes I + |1 \rangle\langle 1| \otimes X =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1 \\
                    0 & 0 & 1 & 0
                \end{pmatrix}


    In the computational basis, this gate flips the target qubit
    if the control qubit is in the :math:`|1\rangle` state.
    In this sense it is similar to a classical XOR gate.

    .. math::
        `|a, b\rangle \rightarrow |a, a \oplus b\rangle`
    """

    _standard_gate = StandardGate.CX

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CX gate."""
        super().__init__(
            "cx",
            2,
            [],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
            _base_label=_base_label,
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the CX gate.

        The controlled gate is implemented as :class:`.CCXGate` for a single control
        qubit, and as :class:`.MCXGate` for two or more control qubits.
        The value of ``annotated`` is ignored.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(
            num_ctrl_qubits=num_ctrl_qubits + 1,
            label=label,
            ctrl_state=new_ctrl_state,
            _base_label=self.label,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Return inverted CX gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CXGate: inverse gate (self-inverse).
        """
        return CXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CXGate) and self.ctrl_state == other.ctrl_state


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=2, cached_states=(3,))
class CCXGate(SingletonControlledGate):
    r"""CCX gate, also known as Toffoli gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ccx` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ──■──
               │
        q_1: ──■──
             ┌─┴─┐
        q_2: ┤ X ├
             └───┘

    Matrix representation:

    .. math::

        CCX q_0, q_1, q_2 =
            I \otimes I \otimes |0 \rangle \langle 0| + CX \otimes |1 \rangle \langle 1| =
           \begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_2 and q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───┐
            q_0: ┤ X ├
                 └─┬─┘
            q_1: ──■──
                   │
            q_2: ──■──

        .. math::

            CCX\ q_2, q_1, q_0 =
                |0 \rangle \langle 0| \otimes I \otimes I + |1 \rangle \langle 1| \otimes CX =
                \begin{pmatrix}
                    1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                    0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                    0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{pmatrix}

    """

    _standard_gate = StandardGate.CCX

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CCX gate."""
        super().__init__(
            "ccx",
            3,
            [],
            num_ctrl_qubits=2,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=2)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #                                                        ┌───┐
        # q_0: ───────────────────■─────────────────────■────■───┤ T ├───■──
        #                         │             ┌───┐   │  ┌─┴─┐┌┴───┴┐┌─┴─┐
        # q_1: ───────■───────────┼─────────■───┤ T ├───┼──┤ X ├┤ Tdg ├┤ X ├
        #      ┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌┴───┴┐┌─┴─┐├───┤└┬───┬┘└───┘
        # q_2: ┤ H ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├─┤ H ├──────
        #      └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘└───┘ └───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CCX._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the CCX gate.

        The controlled gate is implemented as :class:`.MCXGate`.
        The value of `annotated` is ignored.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(
            num_ctrl_qubits=num_ctrl_qubits + 2,
            label=label,
            ctrl_state=new_ctrl_state,
            _base_label=self.label,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Return an inverted CCX gate (also a CCX).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CCXGate: inverse gate (self-inverse).
        """
        return CCXGate(ctrl_state=self.ctrl_state)  # self-inverse

    def __eq__(self, other):
        return isinstance(other, CCXGate) and self.ctrl_state == other.ctrl_state


@with_gate_array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, -1j],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1j, 0, 0, 0, 0],
    ]
)
class RCCXGate(SingletonGate):
    """The simplified Toffoli gate, also referred to as Margolus gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    This implementation requires three CX gates which is the minimal amount possible,
    as shown in https://arxiv.org/abs/quant-ph/0312225.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the dashed box
    of Fig. 3.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rccx` method.
    """

    _standard_gate = StandardGate.RCCX

    def __init__(self, label: str | None = None):
        """Create a new simplified CCX gate."""
        super().__init__("rccx", 3, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # q_0: ────────────────────────■────────────────────────
        #                              │
        # q_1: ────────────■───────────┼─────────■──────────────
        #      ┌───┐┌───┐┌─┴─┐┌─────┐┌─┴─┐┌───┐┌─┴─┐┌─────┐┌───┐
        # q_2: ┤ H ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ H ├
        #      └───┘└───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.RCCX._get_definition(self.params), legacy_qubits=True
        )

    def __eq__(self, other):
        return isinstance(other, RCCXGate)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ) -> ControlledGate | AnnotatedOperation:
        """Return a controlled version of the RCCX gate.

        For a single control qubit, the controlled gate uses a compact, exact
        decomposition consisting of an :class:`.RC3XGate` and an
        :class:`.CSdgGate`. For a new control qubit :math:`c`, the first RCCX
        control :math:`a`, the second RCCX control :math:`b`, and target
        :math:`t`, it uses the identity

        .. math::

            C(\\mathrm{RCCX})(c,a,b;t)
            = \\mathrm{CS}^{\\dagger}(c,a)\\,\\mathrm{RC3X}(c,a,b;t).

        This avoids separately controlling the T gates in the RCCX decomposition.
        For more than one control qubit, the generic controlled-gate synthesis is
        used.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an
                integer or a bitstring. If ``None``, defaults to the all-ones state.
            annotated: For more than one control qubit, indicates whether to return
                an annotated operation. Ignored for a single control qubit.

        Returns:
            A controlled version of this gate.
        """
        if num_ctrl_qubits != 1:
            return super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )

        from qiskit.circuit import QuantumCircuit

        definition = QuantumCircuit(4)
        definition.append(RC3XGate(), definition.qubits)
        definition.csdg(0, 1)

        return ControlledGate(
            "crccx",
            4,
            [],
            label=label,
            num_ctrl_qubits=1,
            definition=definition,
            ctrl_state=ctrl_state,
            base_gate=RCCXGate(label=self.label),
        )

    def inverse(self, annotated: bool = False):
        """Invert this gate. The RCCX gate is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            RCCXGate: inverse gate (self-inverse).

        """
        return RCCXGate()


@with_controlled_gate_array(_SX_ARRAY, num_ctrl_qubits=3, cached_states=(7,))
class C3SXGate(SingletonControlledGate):
    """The 3-qubit controlled sqrt-X gate.

    This implementation is based on Page 17 of [1].

    References:
        [1] Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
    """

    _standard_gate = StandardGate.C3SX

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create a new 3-qubit controlled sqrt-X gate.

        Args:
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g. ``'110'``), or ``None``. If ``None``, use all 1s.
        """
        from .sx import SXGate

        super().__init__(
            "c3sx",
            4,
            [],
            num_ctrl_qubits=3,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=SXGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=3)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.C3SX._get_definition(self.params), legacy_qubits=True
        )

    def __eq__(self, other):
        return isinstance(other, C3SXGate) and self.ctrl_state == other.ctrl_state


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=3, cached_states=(7,))
class C3XGate(SingletonControlledGate):
    r"""The X gate controlled on 3 qubits.

    This implementation uses :math:`\sqrt{T}` and 14 CNOT gates.
    """

    _standard_gate = StandardGate.C3X

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create a new 3-qubit controlled X gate."""
        super().__init__(
            "mcx",
            4,
            [],
            num_ctrl_qubits=3,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=3)

    # seems like open controls not happening?
    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.C3X._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the C3X gate.

        The controlled gate is implemented as :class:`.MCXGate`.
        The value of ``annotated`` is ignored.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(
            num_ctrl_qubits=num_ctrl_qubits + 3,
            label=label,
            ctrl_state=new_ctrl_state,
            _base_label=self.label,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Invert this gate. The C3X is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            C3XGate: inverse gate (self-inverse).
        """
        return C3XGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, C3XGate) and self.ctrl_state == other.ctrl_state


@with_gate_array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1j, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1j, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
)
class RC3XGate(SingletonGate):
    """The simplified 3-controlled Toffoli gate.

    The simplified Toffoli gate implements the Toffoli gate up to relative phases.
    Note, that the simplified Toffoli is not equivalent to the Toffoli. But can be used in places
    where the Toffoli gate is uncomputed again.

    This concrete implementation is from https://arxiv.org/abs/1508.03273, the complete circuit
    of Fig. 4.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rcccx` method.
    """

    _standard_gate = StandardGate.RC3X

    def __init__(self, label: str | None = None):
        """Create a new RC3X gate."""
        super().__init__("rcccx", 4, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.RC3X._get_definition(self.params), legacy_qubits=True
        )

    def __eq__(self, other):
        return isinstance(other, RC3XGate)


@with_controlled_gate_array(_X_ARRAY, num_ctrl_qubits=4, cached_states=(15,))
class C4XGate(SingletonControlledGate):
    """The 4-qubit controlled X gate.

    This implementation is based on Page 21, Lemma 7.5, of [1], with the use
    of the relative phase version of c3x, the rc3x [2].

    References:
        1. Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
        2. Maslov, 2015. https://arxiv.org/abs/1508.03273
    """

    def __init__(
        self,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create a new 4-qubit controlled X gate."""
        super().__init__(
            "mcx",
            5,
            [],
            num_ctrl_qubits=4,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
        )

    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=4)

    # seems like open controls not happening?
    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from .h import HGate
        from .p import CPhaseGate

        # This is not a standard gate in Rust
        q = QuantumRegister(5, name="q")
        qc = QuantumCircuit(q)
        rules = [
            (HGate(), [q[4]], []),
            (CPhaseGate(numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (RC3XGate(), [q[0], q[1], q[2], q[3]], []),
            (HGate(), [q[4]], []),
            (CPhaseGate(-numpy.pi / 2), [q[3], q[4]], []),
            (HGate(), [q[4]], []),
            (RC3XGate().inverse(), [q[0], q[1], q[2], q[3]], []),
            (C3SXGate(), [q[0], q[1], q[2], q[4]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the C4X gate.

        The controlled gate is implemented as :class:`.MCXGate`.
        The value of ``annotated`` is ignored.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCXGate(
            num_ctrl_qubits=num_ctrl_qubits + 4,
            label=label,
            ctrl_state=new_ctrl_state,
            _base_label=self.label,
        )
        return gate

    def inverse(self, annotated: bool = False):
        """Invert this gate. The C4X is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            C4XGate: inverse gate (self-inverse).
        """
        return C4XGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, C4XGate) and self.ctrl_state == other.ctrl_state


class MCXGate(ControlledGate):
    """The general, multi-controlled X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcx` method.
    """

    def __new__(
        cls,
        num_ctrl_qubits: int | None = None,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create a new MCX instance.

        Depending on the number of controls and which mode of the MCX, this creates an
        explicit CX, CCX, C3X or C4X instance or a generic MCX gate.
        """
        # The CXGate and CCXGate will be implemented for all modes of the MCX, and
        # the C3XGate and C4XGate are handled in the gate definition.
        explicit: dict[int, type[ControlledGate]] = {1: CXGate, 2: CCXGate}
        gate_class = explicit.get(num_ctrl_qubits, None)
        if gate_class is not None:
            gate = gate_class.__new__(
                gate_class, label=label, ctrl_state=ctrl_state, _base_label=_base_label
            )
            # if __new__ does not return the same type as cls, init is not called
            gate.__init__(
                label=label,
                ctrl_state=ctrl_state,
                _base_label=_base_label,
            )
            return gate
        return super().__new__(cls)

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _name="mcx",
        _base_label=None,
    ):
        """Create new MCX gate."""
        if self.__class__ in [MCXGate, MCXGrayCode, MCXRecursive, MCXVChain]:
            # DeprecationWarning for internal subclasses (that are deprecated) is fine. We should
            # still raise warnings for other subclasses out of our control
            # TODO MCXGate, MCXGrayCode, MCXRecursive, MCXVChain are deprecated and this path can be
            #   removed once they get removed:
            #   https://github.com/Qiskit/qiskit/pull/12961
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=r".+qiskit\.circuit\.library\.standard_gates\.x\.MCXGate\."
                    r"get_num_ancilla_qubits.+",
                    module="qiskit",
                )
                num_ancilla_qubits = self.__class__.get_num_ancilla_qubits(num_ctrl_qubits)
        else:
            num_ancilla_qubits = self.__class__.get_num_ancilla_qubits(num_ctrl_qubits)

        # alternative: just remove the above
        super().__init__(
            _name,
            num_ctrl_qubits + 1 + num_ancilla_qubits,
            [],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=XGate(label=_base_label),
        )

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXGate: inverse gate (self-inverse).
        """
        return MCXGate(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    @staticmethod
    @deprecate_func(
        additional_msg=(
            "For an MCXGate it is no longer possible to know the number of ancilla qubits "
            "that would be eventually used by the transpiler when the gate is created. "
            "Instead, it is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "However, if a specific synthesis method using a specific number of ancilla "
            "qubits is required, one can create a custom gate by calling the corresponding "
            "synthesis function directly."
        ),
        since="2.1",
        removal_timeline="in Qiskit 3.0",
    )
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "noancilla") -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        This staticmethod might be necessary to check the number of ancillas before
        creating the gate, or to use the number of ancillas in the initialization.
        """
        if mode == "noancilla":
            return 0
        if mode in ["recursion", "advanced"]:
            return int(num_ctrl_qubits > 4)
        if mode[:7] == "v-chain" or mode[:5] == "basic":
            return max(0, num_ctrl_qubits - 2)
        raise AttributeError(f"Unsupported mode ({mode}) specified!")

    def _define(self):
        """This definition is based on MCPhaseGate implementation."""

        from qiskit.synthesis.multi_controlled import synth_mcx_noaux_sp22

        qc = synth_mcx_noaux_sp22(self.num_ctrl_qubits)
        self.definition = qc

    @property
    def num_ancilla_qubits(self):
        """The number of ancilla qubits."""
        return self.get_num_ancilla_qubits(self.num_ctrl_qubits)

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the MCX gate.

        The controlled gate is implemented as :class:`.MCXGate`.
        The value of ``annotated`` is ignored.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state

        # use __class__ so this works for derived classes
        if self.__class__ in [MCXGrayCode, MCXRecursive, MCXVChain]:
            # DeprecationWarning for internal subclasses (that are deprecated) is fine. We should
            # still raise warnings for other subclasses out of our control
            # TODO MCXGate, MCXGrayCode, MCXRecursive, MCXVChain are deprecated and this path can be
            #   removed once they get removed:
            #   https://github.com/Qiskit/qiskit/pull/12961
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    module="qiskit",
                )
                gate = self.__class__(
                    self.num_ctrl_qubits + num_ctrl_qubits,
                    label=label,
                    ctrl_state=new_ctrl_state,
                )
        else:
            gate = MCXGate(
                self.num_ctrl_qubits + num_ctrl_qubits,
                label=label,
                ctrl_state=new_ctrl_state,
                _base_label=self.label,
            )
        return gate

    def copy(self, name=None):
        """
        Copy of the instruction.

        Args:
            name (str): name to be given to the copied circuit, if ``None`` then the name stays the same.

        Returns:
            qiskit.circuit.Instruction: a copy of the current instruction, with the name updated if it
            was provided
        """
        if self.__class__ in [MCXGrayCode, MCXRecursive, MCXVChain]:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=DeprecationWarning,
                    message=r".*qiskit\.circuit\.library\.standard_gates\.x.*",
                )
                return super().copy(name=name)
        return super().copy(name=name)

    def __deepcopy__(self, memo=None):
        if self.__class__ in [MCXGrayCode, MCXRecursive, MCXVChain]:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=DeprecationWarning, message=".+MCXVChain.+"
                )
                return super().__deepcopy__(memo=memo)
        return super().__deepcopy__(memo=memo)


class MCXGrayCode(MCXGate):
    r"""Implement the multi-controlled X gate using the Gray code.

    This delegates the implementation to the MCU1 gate, since :math:`X = H \cdot U1(\pi) \cdot H`.
    """

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "It is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "If this specific synthesis method is required, one can specify it using the "
            "high-level-synthesis plugin 'gray_code' for MCX gates, or, alternatively, "
            "one can use synth_mcx_gray_code' to construct the gate directly."
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __new__(
        cls,
        num_ctrl_qubits: int | None = None,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create a new MCXGrayCode instance"""
        # if 1 to 4 control qubits, create explicit gates
        explicit = {1: CXGate, 2: CCXGate, 3: C3XGate, 4: C4XGate}
        gate_class = explicit.get(num_ctrl_qubits, None)
        if gate_class is not None:
            gate = gate_class.__new__(
                gate_class,
                label=label,
                ctrl_state=ctrl_state,
                _base_label=_base_label,
            )
            # if __new__ does not return the same type as cls, init is not called
            gate.__init__(
                label=label,
                ctrl_state=ctrl_state,
            )
            return gate
        return super().__new__(cls)

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: str | None = None,
        ctrl_state: int | str | None = None,
    ):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name="mcx_gray")

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXGrayCode: inverse gate (self-inverse).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            inverse = MCXGrayCode(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)
        return inverse

    def _define(self):
        """Define the MCX gate using the Gray code."""

        from qiskit.synthesis.multi_controlled import synth_mcx_gray_code

        qc = synth_mcx_gray_code(self.num_ctrl_qubits)
        self.definition = qc


class MCXRecursive(MCXGate):
    """Implement the multi-controlled X gate using recursion.

    Using a single clean ancilla qubit, the multi-controlled X gate is split into
    four sub-registers, each one of them uses the V-chain method.

    The method is based on Lemma 9 of [2], first shown in Lemma 7.3 of [1].

    References:
        1. Barenco et al., 1995. https://arxiv.org/pdf/quant-ph/9503016.pdf
        2. Iten et al., 2015. https://arxiv.org/abs/1501.06911
    """

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "It is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "If this specific synthesis method is required, one can specify it using the "
            "high-level-synthesis plugin ``'gray_code'`` for MCX gates, or, alternatively, "
            "one can use ``'synth_mcx_1_clean'`` to construct the gate directly."
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __new__(
        cls,
        num_ctrl_qubits: int | None = None,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        return super().__new__(cls, num_ctrl_qubits, label, ctrl_state, _base_label=_base_label)

    def __init__(
        self,
        num_ctrl_qubits: int,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        super().__init__(
            num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _name="mcx_recursive",
            _base_label=None,
        )

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "recursion"):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXRecursive: inverse gate (self-inverse).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
            inverse = MCXRecursive(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)
        return inverse

    def _define(self):
        """Define the MCX gate using recursion."""

        from qiskit.synthesis.multi_controlled import synth_mcx_1_clean_b95

        qc = synth_mcx_1_clean_b95(self.num_ctrl_qubits)
        self.definition = qc


class MCXVChain(MCXGate):
    """Implement the multi-controlled X gate using a V-chain of CX gates."""

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "It is recommended to use MCXGate and let HighLevelSynthesis choose "
            "the best synthesis method depending on the number of ancilla qubits available. "
            "If this specific synthesis method is required, one can specify it using the "
            "high-level-synthesis plugins `n_clean_m15` (using clean ancillas) or "
            "`n_dirty_i15` (using dirty ancillas) for MCX gates. Alternatively, one can "
            "use synth_mcx_n_dirty_i15 and synth_mcx_n_clean_m15 to construct the gate directly."
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __new__(
        cls,
        num_ctrl_qubits: int | None = None,
        dirty_ancillas: bool = False,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
        relative_phase: bool = False,
        action_only: bool = False,
    ):
        """Create a new MCX instance.

        This must be defined anew to include the additional argument ``dirty_ancillas``.
        """
        return super().__new__(
            cls,
            num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _base_label=_base_label,
        )

    def __init__(
        self,
        num_ctrl_qubits: int,
        dirty_ancillas: bool = False,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
        relative_phase: bool = False,
        action_only: bool = False,
    ):
        """
        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            dirty_ancillas: when set to ``True``, the method applies an optimized multicontrolled-X gate
                up to a relative phase using dirty ancillary qubits with the properties of lemmas 7 and 8
                from arXiv:1501.06911, with at most 8*k - 6 CNOT gates.
                For k within the range {1, ..., ceil(n/2)}. And for n representing the total number of
                qubits.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            relative_phase: when set to ``True``, the method applies the optimized multicontrolled-X gate
                up to a relative phase, in a way that, by lemma 7 of arXiv:1501.06911, the relative
                phases of the ``action part`` cancel out with the phases of the ``reset part``.
            action_only: when set to ``True``, the method applies only the action part of lemma 8
                from arXiv:1501.06911.

        """
        super().__init__(
            num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            _name="mcx_vchain",
            _base_label=_base_label,
        )
        self._dirty_ancillas = dirty_ancillas
        self._relative_phase = relative_phase
        self._action_only = action_only
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name="mcx_vchain")

    def inverse(self, annotated: bool = False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXVChain: inverse gate (self-inverse).
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r".+qiskit\.circuit\.library\.standard_gates\.x\.MCXVChain\..+",
            )
            inverse = MCXVChain(
                num_ctrl_qubits=self.num_ctrl_qubits,
                dirty_ancillas=self._dirty_ancillas,
                ctrl_state=self.ctrl_state,
                relative_phase=self._relative_phase,
                action_only=self._action_only,
            )
        return inverse

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str = "v-chain"):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def _define(self):
        """Define the MCX gate using a V-chain of CX gates."""

        if self._dirty_ancillas:

            from qiskit.synthesis.multi_controlled import synth_mcx_n_dirty_i15

            qc = synth_mcx_n_dirty_i15(
                self.num_ctrl_qubits,
                self._relative_phase,
                self._action_only,
            )

        else:  # use clean ancillas

            from qiskit.circuit import QuantumCircuit
            from qiskit.synthesis.multi_controlled import synth_mcx_n_clean_m15

            qc = synth_mcx_n_clean_m15(self.num_ctrl_qubits)
            if qc.num_qubits < self.num_qubits:
                # Preserve the deprecated MCXVChain width while the underlying synthesis now
                # needs fewer clean ancillas.
                padded = QuantumCircuit(self.num_qubits)
                padded.compose(qc, range(qc.num_qubits), inplace=True)
                qc = padded

        self.definition = qc
