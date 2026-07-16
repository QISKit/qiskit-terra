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

"""Tests for swap strategy routers."""

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Qubit, QuantumRegister, Parameter, ParameterVector
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import PassManager, CouplingMap, Layout, TranspilerError
from qiskit.circuit.library import PauliEvolutionGate, CXGate
from qiskit.circuit.library.n_local import QAOAAnsatz
from qiskit.converters import circuit_to_dag
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import SetLayout
from qiskit.transpiler.passes import InverseCancellation
from qiskit.transpiler.passes import Decompose
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
    Commuting2qBlock,
)
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing import (
    SwapStrategy,
    FindCommutingPauliEvolutions,
    Commuting2qGateRouter,
)
from test import QiskitTestCase


@ddt
class TestPauliEvolutionSwapStrategies(QiskitTestCase):
    """A class to test the swap strategies transpiler passes."""

    def setUp(self):
        """Assume a linear coupling map."""
        super().setUp()
        cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strat = SwapStrategy(cmap, swap_layers=[[(0, 1), (2, 3)], [(1, 2)]])

        self.pm_ = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
            ]
        )

    def test_basic_zz(self):
        """Test to decompose a ZZ-based evolution op.

        The expected circuit is:

        ..code-block:: text

                                                           ┌────────────────┐
            q_0: ───────────────────X──────────────────────┤0               ├
                 ┌────────────────┐ │ ┌────────────────┐   │  exp(-i ZZ)(3) │
            q_1: ┤0               ├─X─┤0               ├─X─┤1               ├
                 │  exp(-i ZZ)(1) │   │  exp(-i ZZ)(2) │ │ └────────────────┘
            q_2: ┤1               ├─X─┤1               ├─X───────────────────
                 └────────────────┘ │ └────────────────┘
            q_3: ───────────────────X────────────────────────────────────────


        """

        op = SparsePauliOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        swapped = self.pm_.run(circ)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (1, 2))
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
        expected.swap(1, 2)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 3), (0, 1))

        self.assertEqual(swapped, expected)

    def test_global_phase_preserved_without_commuting_blocks(self):
        """Test routing a circuit with no commuting blocks preserves its global phase."""
        circ = QuantumCircuit(4, global_phase=0.3)
        circ.h(0)
        circ.cx(0, 1)

        swap_strat = SwapStrategy.from_line([0, 1, 2, 3])
        routed = PassManager([Commuting2qGateRouter(swap_strat)]).run(circ)

        self.assertEqual(routed.global_phase, circ.global_phase)
        self.assertEqual(Operator(routed), Operator(circ))

    def test_basic_xx(self):
        """Test to route an XX-based evolution op.

        The op is :code:`[("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)]`.

        The expected circuit is:

        ..code-block:: text

                  ┌────────────────┐
            q_0: ─┤0               ├─X─────────────────────────────────────────
                  │  exp(-i XX)(3) │ │ ┌─────────────────┐
            q_1: ─┤1               ├─X─┤0                ├─X───────────────────
                 ┌┴────────────────┤   │  exp(-i XX)(-6) │ │ ┌────────────────┐
            q_2: ┤0                ├─X─┤1                ├─X─┤0               ├
                 │  exp(-i XX)(-3) │ │ └─────────────────┘   │  exp(-i XX)(6) │
            q_3: ┤1                ├─X───────────────────────┤1               ├
                 └─────────────────┘                         └────────────────┘
        """

        op = SparsePauliOp.from_list([("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)])

        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 3), range(4))

        swapped = self.pm_.run(circ)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(Pauli("XX"), 3), (0, 1))
        expected.append(PauliEvolutionGate(Pauli("XX"), -3), (2, 3))
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected.append(PauliEvolutionGate(Pauli("XX"), -6), (1, 2))
        expected.swap(1, 2)
        expected.append(PauliEvolutionGate(Pauli("XX"), 6), (2, 3))

        self.assertEqual(swapped, expected)

    def test_idle_qubit(self):
        """Test to route on an op that has an idle qubit.

        The op is :code:`[("IIXX", 1), ("IXIX", 2)]`.

        The expected circuit is:

        ..code-block:: text

                 ┌─────────────────┐
            q_0: ┤0                ├─X────────────────────
                 │  exp(-it XX)(3) │ │ ┌─────────────────┐
            q_1: ┤1                ├─X─┤0                ├
                 └─────────────────┘   │  exp(-it XX)(6) │
            q_2: ──────────────────────┤1                ├
                                       └─────────────────┘
            q_3: ─────────────────────────────────────────

        """

        op = SparsePauliOp.from_list([("IIXX", 1), ("IXIX", 2)])

        cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strat = SwapStrategy(cmap, swap_layers=(((0, 1),),))

        pm_ = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter(swap_strat)])

        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 3), range(4))

        swapped = pm_.run(circ)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(Pauli("XX"), 3), (0, 1))
        expected.swap(0, 1)
        expected.append(PauliEvolutionGate(Pauli("XX"), 6), (1, 2))

        self.assertEqual(swapped, expected)

    def test_basic_xx_with_measure(self):
        """Test to route an XX-based evolution op with measures.

        The op is :code:`[("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)]`.

        The expected circuit is:

        ..code-block:: text

                     ┌────────────────┐                                            ░    ┌─┐
               q_0: ─┤0               ├─X──────────────────────────────────────────░────┤M├──────
                     │  exp(-i XX)(3) │ │ ┌─────────────────┐                      ░    └╥┘   ┌─┐
               q_1: ─┤1               ├─X─┤0                ├─X────────────────────░─────╫────┤M├
                    ┌┴────────────────┤   │  exp(-i XX)(-6) │ │ ┌────────────────┐ ░ ┌─┐ ║    └╥┘
               q_2: ┤0                ├─X─┤1                ├─X─┤0               ├─░─┤M├─╫─────╫─
                    │  exp(-i XX)(-3) │ │ └─────────────────┘   │  exp(-i XX)(6) │ ░ └╥┘ ║ ┌─┐ ║
               q_3: ┤1                ├─X───────────────────────┤1               ├─░──╫──╫─┤M├─╫─
                    └─────────────────┘                         └────────────────┘ ░  ║  ║ └╥┘ ║
            meas: 4/══════════════════════════════════════════════════════════════════╩══╩══╩══╩═
                                                                                      0  1  2  3
        """

        op = SparsePauliOp.from_list([("XXII", -1), ("IIXX", 1), ("XIIX", -2), ("IXIX", 2)])

        circ = QuantumCircuit(4, 4)
        circ.append(PauliEvolutionGate(op, 3), range(4))
        circ.barrier()
        for idx in range(4):
            circ.measure(idx, idx)

        swapped = self.pm_.run(circ)

        expected = QuantumCircuit(4, 4)
        expected.append(PauliEvolutionGate(Pauli("XX"), 3), (0, 1))
        expected.append(PauliEvolutionGate(Pauli("XX"), -3), (2, 3))
        expected.swap(0, 1)
        expected.swap(2, 3)
        expected.append(PauliEvolutionGate(Pauli("XX"), -6), (1, 2))
        expected.swap(1, 2)
        expected.append(PauliEvolutionGate(Pauli("XX"), 6), (2, 3))
        expected.barrier()
        expected.measure(2, 0)
        expected.measure(0, 1)
        expected.measure(3, 2)
        expected.measure(1, 3)

        self.assertEqual(swapped, expected)

    def test_qaoa(self):
        """Test the QAOA with a custom mixer.

        This test ensures that single-qubit gates end up on the correct qubits. The mixer
        uses Ry gates and the operator is :code:`[("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)]`.

        ..parsed-literal:

                 ┌───┐                                              ┌──────────────────┐»
            q_0: ┤ H ├─────────────────────X────────────────────────┤0                 ├»
                 ├───┤┌──────────────────┐ │ ┌──────────────────┐   │  exp(-i ZZ)(3.0) │»
            q_1: ┤ H ├┤0                 ├─X─┤0                 ├─X─┤1                 ├»
                 ├───┤│  exp(-i ZZ)(1.0) │   │  exp(-i ZZ)(2.0) │ │ └──────────────────┘»
            q_2: ┤ H ├┤1                 ├─X─┤1                 ├─X─────────────────────»
                 ├───┤└──────────────────┘ │ └──────────────────┘                       »
            q_3: ┤ H ├─────────────────────X────────────────────────────────────────────»
                 └───┘                                                                  »
            «     ┌────────┐                    ┌──────────────────┐         »
            «q_0: ┤ Ry(-1) ├────────────────────┤0                 ├────X────»
            «     ├────────┤┌──────────────────┐│  exp(-i ZZ)(6.0) │    │    »
            «q_1: ┤ Ry(-3) ├┤0                 ├┤1                 ├────X────»
            «     ├────────┤│  exp(-i ZZ)(4.0) │└──────────────────┘         »
            «q_2: ┤ Ry(0)  ├┤1                 ├─────────X───────────────────»
            «     ├────────┤└──────────────────┘         │                   »
            «q_3: ┤ Ry(-2) ├─────────────────────────────X───────────────────»
            «     └────────┘                                                 »
            «                         ┌────────┐
            «q_0: ────────────────────┤ Ry(-3) ├
            «     ┌──────────────────┐├────────┤
            «q_1: ┤0                 ├┤ Ry(-1) ├
            «     │  exp(-i ZZ)(2.0) │├────────┤
            «q_2: ┤1                 ├┤ Ry(-2) ├
            «     └──────────────────┘├────────┤
            «q_3: ────────────────────┤ Ry(0)  ├
            «                         └────────┘
        """

        mixer = QuantumCircuit(4)
        for idx in range(4):
            mixer.ry(-idx, idx)

        op = SparsePauliOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        with self.assertWarns(DeprecationWarning):
            circ = QAOAAnsatz(op, reps=2, mixer_operator=mixer)
        swapped = self.pm_.run(circ.decompose())

        param_dict = {p: idx + 1 for idx, p in enumerate(swapped.parameters)}
        swapped.assign_parameters(param_dict, inplace=True)

        # There is some optionality in the order of two instructions. Both are valid.
        valid_expected = []
        for order in [0, 1]:
            expected = QuantumCircuit(4)
            expected.h(range(4))
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (1, 2))
            expected.swap(0, 1)
            expected.swap(2, 3)
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
            expected.swap(1, 2)
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 3), (0, 1))
            expected.ry(-1, 0)
            expected.ry(-3, 1)
            expected.ry(0, 2)
            expected.ry(-2, 3)
            if order == 0:
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 6), (0, 1))
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 4), (2, 1))
            else:
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 4), (2, 1))
                expected.append(PauliEvolutionGate(Pauli("ZZ"), 6), (0, 1))

            expected.swap(0, 1)
            expected.swap(2, 3)
            expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
            expected.ry(-3, 0)
            expected.ry(-1, 1)
            expected.ry(-2, 2)
            expected.ry(0, 3)

            valid_expected.append(expected == swapped)

        self.assertEqual(set(valid_expected), {True, False})

    def test_enlarge_with_ancilla(self):
        """This pass tests that idle qubits after an embedding are left idle."""

        # Create a four qubit problem.
        op = SparsePauliOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])

        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        # Create a four qubit quantum circuit.
        backend_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (1, 3), (3, 4)])

        swap_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strat = SwapStrategy(swap_cmap, swap_layers=(((0, 1), (2, 3)), ((1, 2),)))

        initial_layout = Layout.from_intlist([0, 1, 3, 4], *circ.qregs)

        pm_pre = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
                SetLayout(initial_layout),
                FullAncillaAllocation(backend_cmap),
                EnlargeWithAncilla(),
                ApplyLayout(),
            ]
        )

        embedded = pm_pre.run(circ)

        expected = QuantumCircuit(5)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (1, 3))
        expected.swap(0, 1)
        expected.swap(3, 4)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 3))
        expected.swap(1, 3)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 3), (0, 1))

        self.assertEqual(embedded, expected)

    def test_ccx(self):
        """Test that extra multi-qubit operations are properly adjusted.

        Here, we test that the circuit

        .. code-block:: text

                 ┌──────────────────────────┐
            q_0: ┤0                         ├──■──
                 │                          │┌─┴─┐
            q_1: ┤1 exp(-it (IZZ + ZIZ))(1) ├┤ X ├
                 │                          │└─┬─┘
            q_2: ┤2                         ├──■──
                 └──────────────────────────┘
            q_3: ─────────────────────────────────

        becomes

        .. code-block:: text

                 ┌─────────────────┐                      ┌───┐
            q_0: ┤0                ├─X────────────────────┤ X ├
                 │  exp(-it ZZ)(1) │ │ ┌─────────────────┐└─┬─┘
            q_1: ┤1                ├─X─┤0                ├──■──
                 └─────────────────┘   │  exp(-it ZZ)(2) │  │
            q_2: ──────────────────────┤1                ├──■──
                                       └─────────────────┘
            q_3: ──────────────────────────────────────────────


        as expected. I.e. the Toffoli is properly adjusted at the end.
        """
        cmap = CouplingMap(couplinglist=[(0, 1), (1, 2)])
        swap_strat = SwapStrategy(cmap, swap_layers=(((0, 1),),))

        pm_ = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
            ]
        )
        op = SparsePauliOp.from_list([("IZZ", 1), ("ZIZ", 2)])
        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(3))
        circ.ccx(0, 2, 1)

        swapped = pm_.run(circ)

        expected = QuantumCircuit(4)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 1), (0, 1))
        expected.swap(0, 1)
        expected.append(PauliEvolutionGate(Pauli("ZZ"), 2), (1, 2))
        expected.ccx(1, 2, 0)

        self.assertEqual(swapped, expected)

    def test_t_device(self):
        """Test the swap strategy to route a complete problem on a T device.

        The coupling map in this test corresponds to

        .. code-block:: text

            0 -- 1 -- 2
                 |
                 3
                 |
                 4

        The problem being routed is a fully connect ZZ graph. It has 10 terms since there are
        five qubits in the coupling map. This test checks that the circuit produced by the
        commuting gate router has the instructions we expect. There are several circuits that are
        valid since some of the Rzz gates commute.
        """
        swaps = (
            ((1, 3),),
            ((0, 1), (3, 4)),
            ((1, 3),),
        )

        cmap = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        cmap.make_symmetric()

        swap_strat = SwapStrategy(cmap, swaps)

        # A dense Pauli op.
        op = SparsePauliOp.from_list(
            [
                ("IIIZZ", 1),
                ("IIZIZ", 2),
                ("IZIIZ", 3),
                ("ZIIIZ", 4),
                ("IIZZI", 5),
                ("IZIZI", 6),
                ("ZIIZI", 7),
                ("IZZII", 8),
                ("ZIZII", 9),
                ("ZZIII", 10),
            ]
        )

        circ = QuantumCircuit(5)
        circ.append(PauliEvolutionGate(op, 1), range(5))

        pm_ = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
            ]
        )

        swapped = circuit_to_dag(pm_.run(circ))

        # The swapped circuit can take on several forms as some of the gates commute.
        # We test that sets of gates are where we expected them in the circuit data
        def inst_info(op, qargs, qreg):
            """Get a tuple we can easily test."""
            param = None
            if len(op.params) > 0:
                param = op.params[0]

            return op.name, param, qreg.index(qargs[0]), qreg.index(qargs[1])

        qreg = swapped.qregs["q"]
        inst_list = [inst_info(node.op, node.qargs, qreg) for node in swapped.op_nodes()]

        # First block has the Rzz gates ("IIIZZ", 1), ("IIZZI", 5), ("IZIZI", 6), ("ZZIII", 10)
        expected = {
            ("PauliEvolution", 1.0, 0, 1),
            ("PauliEvolution", 5.0, 1, 2),
            ("PauliEvolution", 6.0, 1, 3),
            ("PauliEvolution", 10.0, 3, 4),
        }
        self.assertSetEqual(set(inst_list[0:4]), expected)

        # Block 2 is a swap
        self.assertSetEqual({inst_list[4]}, {("swap", None, 1, 3)})

        # Block 3 This combines a swap layer and two layers of Rzz gates.
        expected = {
            ("PauliEvolution", 8.0, 2, 1),
            ("PauliEvolution", 7.0, 3, 4),
            ("PauliEvolution", 3.0, 0, 1),
            ("swap", None, 0, 1),
            ("PauliEvolution", 2.0, 1, 2),
            ("PauliEvolution", 4.0, 1, 3),
            ("swap", None, 3, 4),
        }
        self.assertSetEqual(set(inst_list[5:12]), expected)

        # Test the remaining instructions.
        self.assertSetEqual({inst_list[12]}, {("swap", None, 1, 3)})
        self.assertSetEqual({inst_list[13]}, {("PauliEvolution", 9.0, 2, 1)})

    def test_single_qubit_circuit(self):
        """Test that a circuit with only single qubit gates is left unchanged."""

        op = SparsePauliOp.from_list([("IIIX", 1), ("IIXI", 2), ("IZII", 3), ("XIII", 4)])

        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        self.assertEqual(circ, self.pm_.run(circ))

    @data(
        {(0, 1): 0, (2, 3): 0, (1, 2): 1},  # better coloring for the swap strategy
        {(0, 1): 1, (2, 3): 1, (1, 2): 0},  # worse, i.e., less CX cancellation.
    )
    def test_edge_coloring(self, edge_coloring):
        """Test that the edge coloring works."""

        op = SparsePauliOp.from_list([("IIZZ", 1), ("IZZI", 2), ("ZZII", 3), ("ZIZI", 4)])
        swaps = (((1, 2),),)

        cmap = CouplingMap([[0, 1], [1, 2], [2, 3]])
        cmap.make_symmetric()

        swap_strat = SwapStrategy(cmap, swaps)

        circ = QuantumCircuit(4)
        circ.append(PauliEvolutionGate(op, 1), range(4))

        pm_ = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat, edge_coloring=edge_coloring),
                Decompose(),  # double decompose gets to CX
                Decompose(),
                InverseCancellation([CXGate()]),
            ]
        )

        expected = QuantumCircuit(4)
        if edge_coloring[(0, 1)] == 1:
            expected.cx(1, 2)
            expected.rz(4, 2)
            expected.cx(1, 2)
            expected.cx(0, 1)
            expected.rz(2, 1)
            expected.cx(0, 1)
            expected.cx(2, 3)
            expected.rz(6, 3)
            expected.cx(2, 3)
            expected.cx(1, 2)
            expected.cx(2, 1)
            expected.cx(1, 2)
            expected.cx(2, 3)
            expected.rz(8, 3)
            expected.cx(2, 3)
        else:
            expected.cx(0, 1)
            expected.rz(2, 1)
            expected.cx(0, 1)
            expected.cx(2, 3)
            expected.rz(6, 3)
            expected.cx(2, 3)
            expected.cx(1, 2)
            expected.rz(4, 2)
            expected.cx(2, 1)
            expected.cx(1, 2)
            expected.cx(2, 3)
            expected.rz(8, 3)
            expected.cx(2, 3)

        self.assertEqual(pm_.run(circ), expected)

    def test_permutation_tracking(self):
        """Test that circuit layout permutations are properly tracked in the pass property
        set and returned with the output circuit."""

        # We use the same scenario as the QAOA test above
        mixer = QuantumCircuit(4)
        for idx in range(4):
            mixer.ry(-idx, idx)

        op = SparsePauliOp.from_list([("IZZI", 1), ("ZIIZ", 2), ("ZIZI", 3)])
        with self.assertWarns(DeprecationWarning):
            circ = QAOAAnsatz(op, reps=2, mixer_operator=mixer)

        cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])
        swap_strat = SwapStrategy(cmap, swap_layers=[[(0, 1), (2, 3)], [(1, 2)]])

        # test standalone
        swap_pm = PassManager(
            [
                FindCommutingPauliEvolutions(),
                Commuting2qGateRouter(swap_strat),
            ]
        )
        swapped = swap_pm.run(circ.decompose())

        # test as pre-routing step
        backend = GenericBackendV2(num_qubits=4, coupling_map=[[0, 1], [0, 2], [0, 3]], seed=42)
        pm = generate_preset_pass_manager(
            optimization_level=3, target=backend.target, seed_transpiler=40
        )
        pm.pre_routing = swap_pm
        full = pm.run(circ.decompose())

        self.assertEqual(swapped.layout.routing_permutation(), [3, 1, 2, 0])
        self.assertEqual(full.layout.routing_permutation(), [0, 1, 2, 3])


class TestSwapRouterExceptions(QiskitTestCase):
    """Test that exceptions are properly raised."""

    def setUp(self):
        """Setup useful variables."""
        super().setUp()

        # A fully connected problem.
        op = SparsePauliOp.from_list(
            [("IIZZ", 1), ("IZIZ", 1), ("ZIIZ", 1), ("IZZI", 1), ("ZIZI", 1), ("ZZII", 1)]
        )
        self.circ = QuantumCircuit(4)
        self.circ.append(PauliEvolutionGate(op, 1), range(4))

        self.swap_cmap = CouplingMap(couplinglist=[(0, 1), (1, 2), (2, 3)])

        # This swap strategy does not reach full connectivity.
        self.swap_strat = SwapStrategy(self.swap_cmap, swap_layers=(((0, 1), (2, 3)),))

    def test_no_swap_strategy(self):
        """Test raise on no swap strategy."""

        pm_ = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter()])

        with self.assertRaises(TranspilerError):
            pm_.run(self.circ)

    def test_dangling_qubits(self):
        """Test that dangling qubits are not allowed."""

        loose = [Qubit() for _ in [None] * 5]
        reg = QuantumRegister(5)
        qc = QuantumCircuit(loose, reg)

        message = "Circuit has qubits not contained in the qubit register."
        with self.assertRaisesRegex(TranspilerError, message):
            Commuting2qGateRouter(self.swap_strat).run(circuit_to_dag(qc))

    def test_too_many_registers(self):
        """Check that we raise if there are too many registers."""
        qc = QuantumCircuit(QuantumRegister(5), QuantumRegister(4))

        message = "Commuting2qGateRouter runs on circuits with one quantum register."
        with self.assertRaisesRegex(TranspilerError, message):
            Commuting2qGateRouter(self.swap_strat).run(circuit_to_dag(qc))

    def test_deficient_swap_strategy(self):
        """Test to raise when all edges cannot be implemented."""

        pm_ = PassManager([FindCommutingPauliEvolutions(), Commuting2qGateRouter()])

        with self.assertRaises(TranspilerError):
            pm_.run(self.circ)

    def test_commuting2qblocks_errors(self):
        """Test the errors of the 2q commuting block."""
        circ = QuantumCircuit(3)
        circ.ccx(0, 1, 2)

        with self.assertRaises(QiskitError):
            Commuting2qBlock(circuit_to_dag(circ).op_nodes())

    def test_commuting2qblock_preserves_parameters(self):
        """Test that Commuting2qBlock preserves parameters from parametric gates."""

        c_0 = Parameter("c_0")
        c_1 = Parameter("c_1")
        gamma = Parameter("gamma")

        qc = QuantumCircuit(3)
        qc.rzz(c_0 * gamma, 0, 1)
        qc.rzz(c_1, 1, 2)

        dag = circuit_to_dag(qc)
        nodes = list(dag.topological_op_nodes())
        block = Commuting2qBlock(nodes)

        # Check that both parameters are preserved
        self.assertEqual(len(block.params), 3)
        param_names = {p.name for p in block.params}
        self.assertEqual(param_names, {"c_0", "gamma", "c_1"})

    def test_parameters_when_appending_commuting2qblock_to_a_circuit(self):
        """Test that a circuit appended with Commuting2qBlock knows about its parameters."""

        theta = ParameterVector("theta", 3)
        qc = QuantumCircuit(3)
        qc.rzz(theta[0], 0, 1)
        qc.rzz(theta[1], 1, 2)
        qc.rzz(theta[2], 0, 2)

        dag = circuit_to_dag(qc)
        nodes = list(dag.topological_op_nodes())
        block = Commuting2qBlock(nodes)

        circuit_with_block = QuantumCircuit(3)
        self.assertEqual(circuit_with_block.parameters, {})
        circuit_with_block.append(block, range(3))

        circuit_params = set(circuit_with_block.parameters)
        expected_params = {theta[0], theta[1], theta[2]}
        self.assertEqual(circuit_params, expected_params)

        # Assign parameters and verify the circuit has no parameters after assignment
        param_values = {theta[0]: 0.5, theta[1]: 1.0, theta[2]: 1.5}
        assigned_circuit = circuit_with_block.assign_parameters(param_values)
        self.assertEqual(assigned_circuit.parameters, {})

    def test_commuting2qblock_parameter_reuse(self):
        """Test that reusing the same parameter multiple times within a block works correctly."""

        gamma = Parameter("gamma")
        c_0 = Parameter("c_0")
        qc = QuantumCircuit(3)
        qc.rzz(gamma, 0, 1)
        qc.rzz(2 * gamma, 1, 2)
        qc.rzz(gamma + c_0, 0, 2)  # Reuse gamma

        dag = circuit_to_dag(qc)
        nodes = list(dag.topological_op_nodes())
        block = Commuting2qBlock(nodes)

        self.assertEqual(len(block.params), 2)
        self.assertEqual(set(block.params), {c_0, gamma})

        circuit_with_block = QuantumCircuit(3)
        circuit_with_block.append(block, range(3))

        self.assertEqual(set(circuit_with_block.parameters), {gamma, c_0})

        assigned_circuit = circuit_with_block.assign_parameters({gamma: 0.7})
        self.assertEqual(len(assigned_circuit.parameters), 1)
