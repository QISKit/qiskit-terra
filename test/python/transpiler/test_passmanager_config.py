# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests PassManagerConfig"""

from qiskit import QuantumRegister
from qiskit.providers.backend import Backend
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passmanager_config import PassManagerConfig
from test import QiskitTestCase
from ..legacy_cmaps import ALMADEN_CMAP


class TestPassManagerConfig(QiskitTestCase):
    """Test PassManagerConfig.from_backend()."""

    def test_config_from_backend_v2(self):
        """Test from_backend() with a BackendV2 instance."""
        backend = GenericBackendV2(num_qubits=27, coupling_map=CouplingMap.from_line(27), seed=42)
        config = PassManagerConfig.from_backend(backend)
        self.assertEqual(config.basis_gates, backend.operation_names)
        self.assertEqual(config.coupling_map.get_edges(), backend.coupling_map.get_edges())

    def test_invalid_backend(self):
        """Test from_backend() with an invalid backend."""
        with self.assertRaises(AttributeError):
            PassManagerConfig.from_backend(Backend())

    def test_from_backend_and_user(self):
        """Test from_backend() with a backend and user options.

        `FakeMelbourne` is used in this testcase. This backend does not have
        `defaults` attribute and thus not provide an instruction schedule map.
        """
        qr = QuantumRegister(4, "qr")
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=ALMADEN_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        config = PassManagerConfig.from_backend(
            backend, basis_gates=["user_gate"], initial_layout=initial_layout
        )
        self.assertEqual(config.basis_gates, ["user_gate"])
        self.assertNotEqual(config.basis_gates, backend.operation_names)
        self.assertEqual(str(config.coupling_map), str(CouplingMap(backend.coupling_map)))
        self.assertEqual(config.initial_layout, initial_layout)

    def test_invalid_user_option(self):
        """Test from_backend() with an invalid user option."""
        backend = GenericBackendV2(num_qubits=20, coupling_map=ALMADEN_CMAP, seed=42)
        with self.assertRaises(TypeError):
            PassManagerConfig.from_backend(backend, invalid_option=None)

    def test_str(self):
        """Test string output."""
        pm_config = PassManagerConfig.from_backend(BasicSimulator())
        # For testing remove instruction schedule map, its str output is non-deterministic
        # based on hash seed
        str_out = str(pm_config)
        expected = """Pass Manager Config:
\tinitial_layout: None
\tbasis_gates: ['ccx', 'ccz', 'ch', 'cp', 'crx', 'cry', 'crz', 'cs', 'csdg', 'cswap', 'csx', 'cu', 'cu1', 'cu3', 'cx', 'cy', 'cz', 'dcx', 'delay', 'ecr', 'global_phase', 'h', 'id', 'iswap', 'measure', 'p', 'r', 'rccx', 'reset', 'rx', 'rxx', 'ry', 'ryy', 'rz', 'rzx', 'rzz', 's', 'sdg', 'swap', 'sx', 'sxdg', 't', 'tdg', 'u', 'u1', 'u2', 'u3', 'unitary', 'x', 'xx_minus_yy', 'xx_plus_yy', 'y', 'z']
\tcoupling_map: None
\tlayout_method: None
\trouting_method: None
\ttranslation_method: None
\tscheduling_method: None
\tinstruction_durations:\u0020
\tapproximation_degree: None
\tseed_transpiler: None
\ttiming_constraints: None
\tunitary_synthesis_method: default
\tunitary_synthesis_plugin_config: None
\tqubits_initially_zero: True
\ttarget: Target: Basic Target
\tNumber of qubits: None
\tInstructions:
\t\tccx
\t\tccz
\t\tch
\t\tcp
\t\tcrx
\t\tcry
\t\tcrz
\t\tcs
\t\tcsdg
\t\tcswap
\t\tcsx
\t\tcu
\t\tcu1
\t\tcu3
\t\tcx
\t\tcy
\t\tcz
\t\tdcx
\t\tdelay
\t\tecr
\t\tglobal_phase
\t\th
\t\tid
\t\tiswap
\t\tmeasure
\t\tp
\t\tr
\t\trccx
\t\treset
\t\trx
\t\trxx
\t\try
\t\tryy
\t\trz
\t\trzx
\t\trzz
\t\ts
\t\tsdg
\t\tswap
\t\tsx
\t\tsxdg
\t\tt
\t\ttdg
\t\tu
\t\tu1
\t\tu2
\t\tu3
\t\tunitary
\t\tx
\t\txx_minus_yy
\t\txx_plus_yy
\t\ty
\t\tz
\t
"""
        self.assertEqual(str_out, expected)
