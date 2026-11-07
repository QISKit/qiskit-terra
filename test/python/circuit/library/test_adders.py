# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test adder circuits."""

import unittest
import warnings

import numpy as np
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    VBERippleCarryAdder,
    ModularAdderGate,
    HalfAdderGate,
    FullAdderGate,
)
from qiskit.synthesis.arithmetic import (
    adder_ripple_c04,
    adder_ripple_v95,
    adder_ripple_r25,
    adder_qft_d00,
    adder_modular_v17,
)
from qiskit.transpiler.passes import HLSConfig, HighLevelSynthesis
from test import QiskitTestCase

ADDERS = {
    "vbe": adder_ripple_v95,
    "cdkm": adder_ripple_c04,
    "rv": adder_ripple_r25,
    "draper": adder_qft_d00,
    "vrg": adder_modular_v17,
}

ADDER_CIRCUITS = {
    "vbe": VBERippleCarryAdder,
    "cdkm": CDKMRippleCarryAdder,
    "draper": DraperQFTAdder,
}


@ddt
class TestAdder(QiskitTestCase):
    """Test the adder circuits."""

    def assertAdditionIsCorrect(
        self, num_state_qubits: int, adder: QuantumCircuit, inplace: bool, kind: str
    ):
        """Assert that adder correctly implements the summation.

        This test prepares a superposition with a distinct amplitude for every input, then
        performs the addition and checks the output state. The distinct amplitudes ensure
        that both the input-output mapping and relative phases are verified.

        Args:
            num_state_qubits: The number of bits in the numbers that are added.
            adder: The circuit performing the addition of two numbers with ``num_state_qubits``
                bits.
            inplace: If True, compare against an inplace addition where the result is written into
                the second register plus carry qubit. If False, assume that the result is written
                into a third register of appropriate size.
            kind: The kind of adder; "fixed", "half", or "full".
        """
        if kind == "full":
            num_superpos_qubits = 2 * num_state_qubits + 1
        else:
            num_superpos_qubits = 2 * num_state_qubits

        # Use a distinct positive amplitude for each input basis state. Input qubits are
        # the least-significant qubits and all output and helper qubits start in state 0.
        amplitudes = np.arange(1, 2**num_superpos_qubits + 1, dtype=float)
        amplitudes /= np.linalg.norm(amplitudes)
        initial_state = np.zeros(2**adder.num_qubits, dtype=complex)
        initial_state[: len(amplitudes)] = amplitudes
        statevector = Statevector(initial_state).evolve(adder)
        pad = "0" * adder.num_ancillas  # state of the ancillas

        # compute the expected results
        expected_state = np.zeros_like(statevector.data)
        num_bits_sum = num_state_qubits + 1
        # iterate over all possible inputs
        for x in range(2**num_state_qubits):
            for y in range(2**num_state_qubits):
                # compute the sum
                if kind == "full":
                    additions = [x + y, 1 + x + y]
                elif kind == "half":
                    additions = [x + y]
                else:
                    additions = [(x + y) % (2**num_state_qubits)]

                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)

                for i, addition in enumerate(additions):
                    if kind == "full":
                        input_index = i | (x << 1) | (y << (num_state_qubits + 1))
                    else:
                        input_index = x | (y << num_state_qubits)

                    bin_res = bin(addition)[2:].zfill(num_bits_sum)
                    if kind == "full":
                        cin = str(i)
                        bin_index = (
                            pad + bin_res + bin_x + cin
                            if inplace
                            else pad + bin_res + bin_y + bin_x + cin
                        )
                    else:
                        bin_index = (
                            pad + bin_res + bin_x if inplace else pad + bin_res + bin_y + bin_x
                        )

                    index = int(bin_index, 2)
                    expected_state[index] = initial_state[input_index]

        np.testing.assert_array_almost_equal(expected_state, statevector.data)

    @data(
        (3, "cdkm", "half"),
        (5, "cdkm", "half"),
        (3, "cdkm", "fixed"),
        (5, "cdkm", "fixed"),
        (1, "cdkm", "full"),
        (3, "cdkm", "full"),
        (5, "cdkm", "full"),
        (3, "rv", "half"),
        (5, "rv", "half"),
        (3, "draper", "half"),
        (5, "draper", "half"),
        (3, "draper", "fixed"),
        (5, "draper", "fixed"),
        (1, "vbe", "full"),
        (3, "vbe", "full"),
        (5, "vbe", "full"),
        (1, "vbe", "half"),
        (2, "vbe", "half"),
        (5, "vbe", "half"),
        (1, "vbe", "fixed"),
        (2, "vbe", "fixed"),
        (4, "vbe", "fixed"),
        (1, "vrg", "fixed"),
        (3, "vrg", "fixed"),
        (5, "vrg", "fixed"),
    )
    @unpack
    def test_summation(self, num_state_qubits, adder, kind):
        """Test summation for all implemented adders."""
        for use_function in [True, False]:
            with self.subTest(use_function=use_function):
                if use_function:
                    if adder in [
                        "rv",
                        "vrg",
                    ]:  # no kind for this. we still need kind for the test result
                        circuit = ADDERS[adder](num_state_qubits)
                    else:
                        circuit = ADDERS[adder](num_state_qubits, kind)
                else:
                    if adder in ["rv", "vrg"]:  # no adder circuit for this
                        continue
                    with self.assertWarns(DeprecationWarning):
                        circuit = ADDER_CIRCUITS[adder](num_state_qubits, kind)

                self.assertAdditionIsCorrect(num_state_qubits, circuit, True, kind)

    @data(
        ("full", 8, 2),
        ("half", 6, 2),
        ("fixed", 6, 0),
    )
    @unpack
    def test_vbe_relative_phase_toffoli_counts(self, kind, expected_rccx, expected_ccx):
        """Test that only carry-out computations use exact Toffoli gates."""
        circuit = adder_ripple_v95(3, kind=kind).decompose(
            gates_to_decompose=["Carry", "Carry_rp", "Carry_rp_dg", "Sum"]
        )
        counts = circuit.count_ops()
        self.assertEqual(counts.get("rccx", 0), expected_rccx)
        self.assertEqual(counts.get("ccx", 0), expected_ccx)

    @data(
        adder_ripple_c04,
        adder_ripple_v95,
        adder_ripple_r25,
        adder_qft_d00,
        adder_modular_v17,
    )
    def test_raises_on_wrong_num_bits(self, adder):
        """Test an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            _ = adder(-1)

    @data(CDKMRippleCarryAdder, DraperQFTAdder, VBERippleCarryAdder)
    def test_raises_on_wrong_num_bits_legacy(self, adder):
        """Test (legacy) an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            with warnings.catch_warnings(record=True):
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                _ = adder(-1)

    def test_plugins(self):
        """Test calling HLS plugins for various adder types."""

        # all gates with the plugins we check
        modes = {
            "ModularAdder": (
                ModularAdderGate,
                ["ripple_c04", "ripple_v95", "qft_d00", "modular_v17"],
            ),
            "HalfAdder": (HalfAdderGate, ["ripple_c04", "ripple_v95", "ripple_r25", "qft_d00"]),
            "FullAdder": (FullAdderGate, ["ripple_c04", "ripple_v95"]),
        }

        # an operation we expect to be in the circuit with given plugin name
        expected_ops = {
            "ripple_c04": "MAJ",
            "ripple_v95": "Carry_rp",
            "ripple_r25": "ccx",
            "qft_d00": "cp",
            "modular_v17": "rccx",
        }

        num_state_qubits = 3
        max_auxiliaries = num_state_qubits - 1  # V95 needs these
        max_num_qubits = 2 * num_state_qubits + max_auxiliaries + 2

        for name, (adder_cls, plugins) in modes.items():
            for plugin in plugins:
                with self.subTest(name=name, plugin=plugin):
                    adder = adder_cls(num_state_qubits)

                    circuit = QuantumCircuit(max_num_qubits)
                    circuit.append(adder, range(adder.num_qubits))

                    hls_config = HLSConfig(**{name: [plugin]})
                    hls = HighLevelSynthesis(hls_config=hls_config)

                    synth = hls(circuit)
                    ops = set(synth.count_ops().keys())

                    self.assertTrue(expected_ops[plugin] in ops)

    def test_plugins_when_do_not_apply(self):
        """Test that plugins do not do anything when not enough
        clean ancilla qubits are available.
        """
        with self.subTest(name="FullAdder"):
            adder = FullAdderGate(3)
            circuit = QuantumCircuit(9)
            circuit.append(adder, range(adder.num_qubits))
            hls_config = HLSConfig(FullAdder=["ripple_v95"])
            hls = HighLevelSynthesis(hls_config=hls_config)
            synth = hls(circuit)
            self.assertEqual(synth.count_ops(), {"FullAdder": 1})
        with self.subTest(name="HalfAdder"):
            adder = HalfAdderGate(3)
            circuit = QuantumCircuit(8)
            circuit.append(adder, range(adder.num_qubits))
            hls_config = HLSConfig(HalfAdder=["ripple_v95"])
            hls = HighLevelSynthesis(hls_config=hls_config)
            synth = hls(circuit)
            self.assertEqual(synth.count_ops(), {"HalfAdder": 1})
        with self.subTest(name="ModularAdder"):
            adder = ModularAdderGate(3)
            circuit = QuantumCircuit(7)
            circuit.append(adder, range(adder.num_qubits))
            hls_config = HLSConfig(ModularAdder=["ripple_v95"])
            hls = HighLevelSynthesis(hls_config=hls_config)
            synth = hls(circuit)
            self.assertEqual(synth.count_ops(), {"ModularAdder": 1})

    def test_default_plugins(self):
        """Tests covering different branches in the default synthesis plugins."""

        # Test's name indicates which synthesis method should get used.
        with self.subTest(name="HalfAdder_use_ripple_rv_25"):
            adder = HalfAdderGate(3)
            circuit = QuantumCircuit(9)
            circuit.append(adder, range(7))
            hls = HighLevelSynthesis()
            synth = hls(circuit)
            ops = set(synth.count_ops().keys())
            self.assertTrue("ccx" in ops)
        with self.subTest(name="HalfAdder_use_ripple_c04"):
            adder = HalfAdderGate(4)
            circuit = QuantumCircuit(12)
            circuit.append(adder, range(9))
            hls = HighLevelSynthesis()
            synth = hls(circuit)
            ops = set(synth.count_ops().keys())
            self.assertTrue("MAJ" in ops)
        with self.subTest(name="HalfAdder_use_ripple_rv_25"):
            adder = HalfAdderGate(4)
            circuit = QuantumCircuit(9)
            circuit.append(adder, range(9))
            hls = HighLevelSynthesis()
            synth = hls(circuit)
            ops = set(synth.count_ops().keys())
            self.assertTrue("ccx" in ops)

        with self.subTest(name="FullAdder_use_ripple_c04"):
            adder = FullAdderGate(4)
            circuit = QuantumCircuit(10)
            circuit.append(adder, range(10))
            hls = HighLevelSynthesis()
            synth = hls(circuit)
            ops = set(synth.count_ops().keys())
            self.assertTrue("MAJ" in ops)
        with self.subTest(name="FullAdder_use_ripple_v95"):
            adder = FullAdderGate(1)
            circuit = QuantumCircuit(10)
            circuit.append(adder, range(4))
            hls = HighLevelSynthesis()
            synth = hls(circuit)
            ops = set(synth.count_ops().keys())
            self.assertTrue("Carry" in ops)

        with self.subTest(name="ModularAdder_use_qft_d00"):
            adder = ModularAdderGate(4)
            circuit = QuantumCircuit(8)
            circuit.append(adder, range(8))
            hls = HighLevelSynthesis()
            synth = hls(circuit)
            ops = set(synth.count_ops().keys())
            self.assertTrue("cp" in ops)
        with self.subTest(name="ModularAdder_use_v17"):
            adder = ModularAdderGate(6)
            circuit = QuantumCircuit(12)
            circuit.append(adder, range(12))
            hls = HighLevelSynthesis()
            synth = hls(circuit)
            ops = set(synth.count_ops().keys())
            self.assertTrue("rccx" in ops)


if __name__ == "__main__":
    unittest.main()
