# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test scheduled circuit (quantum circuit with duration)."""
from ddt import ddt, data

from qiskit import QuantumCircuit, QiskitError
from qiskit import transpile
from qiskit.circuit import Parameter
from qiskit.circuit.duration import convert_durations_to_dt
from qiskit.circuit.library import CXGate, HGate
from qiskit.circuit.delay import Delay
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler import InstructionProperties, Target
from qiskit.transpiler.exceptions import TranspilerError
from test import QiskitTestCase


@ddt
class TestScheduledCircuit(QiskitTestCase):
    """Test scheduled circuit (quantum circuit with duration)."""

    def setUp(self):
        super().setUp()
        self.dt = 2.2222222222222221e-10
        self.backend_with_dt = GenericBackendV2(2, seed=42, dt=self.dt)
        self.backend_without_dt = GenericBackendV2(2, seed=42)
        self.backend_without_dt.target.dt = None
        self.simulator_backend = BasicSimulator()

    def test_schedule_circuit_when_backend_tells_dt(self):
        """dt is known to transpiler by backend"""
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit="ms")  # 450000[dt]
        qc.delay(100, 0, unit="ns")  # 450[dt]
        qc.h(0)  # 195[dt]
        qc.h(1)  # 210[dt]

        backend = GenericBackendV2(2, seed=42)

        sc = transpile(qc, backend, scheduling_method="alap", layout_method="trivial")
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(sc.duration, 451095)
            self.assertEqual(sc.unit, "dt")
        self.assertEqual(sc.data[0].operation.name, "delay")
        self.assertEqual(sc.data[0].operation.duration, 450900)
        self.assertEqual(sc.data[0].operation.unit, "dt")
        self.assertEqual(sc.data[1].operation.name, "rz")
        self.assertEqual(sc.data[4].operation.name, "delay")
        self.assertEqual(sc.data[4].operation.duration, 450885)
        self.assertEqual(sc.data[4].operation.unit, "dt")

    def test_schedule_circuit_when_transpile_option_tells_dt(self):
        """dt is known to transpiler by transpile option"""
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit="ms")  # 450000[dt]
        qc.delay(100, 0, unit="ns")  # 450[dt]
        qc.h(0)  # duration: rz(0) + sx(195[dt]) + rz(0)
        qc.h(1)  # duration: rz(0)+ sx(210[dt]) + rz(0)
        sc = transpile(
            qc,
            self.backend_without_dt,
            scheduling_method="alap",
            dt=self.dt,
            layout_method="trivial",
            seed_transpiler=20,
        )
        target_durations = self.backend_with_dt.target.durations()
        with self.assertWarns(DeprecationWarning):
            self.assertAlmostEqual(sc.duration, (450450 + target_durations.get("sx", 0)))
            self.assertEqual(sc.unit, "dt")
        self.assertEqual(sc.data[0].operation.name, "delay")
        self.assertEqual(sc.data[0].operation.duration, 450450)
        self.assertEqual(sc.data[0].operation.unit, "dt")
        self.assertEqual(sc.data[1].operation.name, "rz")
        self.assertEqual(sc.data[4].operation.name, "delay")
        self.assertEqual(
            sc.data[4].operation.duration,
            450450 + target_durations.get("sx", 0) - target_durations.get("sx", 1),
        )
        self.assertEqual(sc.data[4].operation.unit, "dt")

    def test_schedule_circuit_in_sec_when_no_one_tells_dt(self):
        """dt is unknown and all delays and gate times are in SI"""
        qc = QuantumCircuit(2)
        qc.delay(0.1, 0, unit="ms")  # 450000[dt]
        qc.delay(100, 0, unit="ns")  # 450[dt]
        qc.h(0)  # duration: rz(0) + sx(195[dt]) + rz(0)
        qc.h(1)  # duration: rz(0)+ sx(210[dt]) + rz(0)
        sc = transpile(
            qc, self.backend_without_dt, scheduling_method="alap", layout_method="trivial"
        )
        target_durations = self.backend_with_dt.target.durations()

        with self.assertWarns(DeprecationWarning):
            self.assertAlmostEqual(sc.duration, (450450 + target_durations.get("sx", 0)) * self.dt)
            self.assertEqual(sc.unit, "s")
        self.assertEqual(sc.data[0].operation.name, "delay")
        self.assertAlmostEqual(sc.data[0].operation.duration, 1.0e-4 + 1.0e-7)
        self.assertEqual(sc.data[0].operation.unit, "s")
        self.assertEqual(sc.data[1].operation.name, "rz")
        self.assertEqual(sc.data[4].operation.name, "delay")
        self.assertAlmostEqual(sc.data[4].operation.duration, 1.0e-4 + 1.0e-7)
        self.assertEqual(sc.data[4].operation.unit, "s")

    def test_cannot_schedule_circuit_with_mixed_SI_and_dt_when_no_one_tells_dt(self):
        """dt is unknown but delays and gate times have a mix of SI and dt"""
        qc = QuantumCircuit(2)
        qc.delay(100, 0, unit="ns")  # 450[dt]
        qc.delay(30, 0, unit="dt")  # 30[dt]
        qc.h(0)  # duration: rz(0) + sx(195[dt]) + rz(0)
        qc.h(1)  # duration: rz(0)+ sx(210[dt]) + rz(0)
        with self.assertRaises(QiskitError):
            transpile(qc, self.backend_without_dt, scheduling_method="alap")

    def test_transpile_single_delay_circuit(self):
        qc = QuantumCircuit(1)
        qc.delay(1234, 0)
        sc = transpile(qc, backend=self.backend_with_dt, scheduling_method="alap")
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(sc.duration, 1234)
        self.assertEqual(sc.data[0].operation.name, "delay")
        self.assertEqual(sc.data[0].operation.duration, 1234)
        self.assertEqual(sc.data[0].operation.unit, "dt")

    def test_transpile_t1_circuit(self):
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.delay(1000, 0, unit="ns")  # 4500 [dt]
        qc.measure_all()
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method="alap")
        # the x and measure gates get routed to qubit 1
        target_durations = self.backend_with_dt.target.durations()
        expected_scheduled = (
            target_durations.get("x", 1) + 4500 + target_durations.get("measure", 1)
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, expected_scheduled)

    def test_transpile_delay_circuit_with_backend(self):
        qc = QuantumCircuit(2)
        qc.h(0)  # 195 [dt]
        qc.delay(100, 1, unit="ns")  # 450 [dt]
        qc.cx(0, 1)  # 3169 [dt]
        scheduled = transpile(
            qc, backend=self.backend_with_dt, scheduling_method="alap", layout_method="trivial"
        )
        target_durations = self.backend_with_dt.target.durations()
        # The longest path output is 3 sx gates 1 cx(1,) and two delays for
        # alignment
        cx_duration = target_durations.get("cx", (1, 0))
        sx_duration = target_durations.get("sx", (0,))
        expected_duration = sx_duration * 3 + cx_duration + 270 + 15
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, expected_duration)

    def test_transpile_circuit_with_custom_instruction(self):
        """See: https://github.com/Qiskit/qiskit-terra/issues/5154"""
        bell = QuantumCircuit(2, name="bell")
        bell.h(0)
        bell.cx(0, 1)
        bell_instr = bell.to_instruction()
        qc = QuantumCircuit(2)
        qc.delay(500, 1)
        qc.append(bell_instr, [0, 1])

        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(0)})
        target.add_instruction(
            HGate(), {(0,): InstructionProperties(0), (1,): InstructionProperties(0)}
        )
        target.add_instruction(Delay(Parameter("t")), {(0,): None, (1,): None})
        target.add_instruction(
            bell_instr,
            {
                (0, 1): InstructionProperties(1000 * 1e-2),
                (1, 0): InstructionProperties(1000 * 1e-2),
            },
        )
        target.dt = 1e-2
        scheduled = transpile(
            qc,
            scheduling_method="alap",
            target=target,
            dt=1e-2,
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, 1500)

    def test_transpile_delay_circuit_with_dt_but_without_scheduling_method(self):
        qc = QuantumCircuit(1)
        qc.delay(100, 0, unit="ns")
        transpiled = transpile(qc, backend=self.backend_with_dt)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(transpiled.duration, None)  # not scheduled
        self.assertEqual(transpiled.data[0].operation.duration, 450)  # unit is converted ns -> dt

    def test_transpile_delay_circuit_without_scheduling_method_or_durs(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        not_scheduled = transpile(qc)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(not_scheduled.duration, None)

    def test_raise_error_if_transpile_with_scheduling_method_but_without_durations(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        with self.assertRaises(TranspilerError):
            transpile(qc, scheduling_method="alap")

    def test_invalidate_schedule_circuit_if_new_instruction_is_appended(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500 * self.dt, 1, "s")
        qc.cx(0, 1)
        scheduled = transpile(qc, backend=self.backend_with_dt, scheduling_method="alap")
        # append a gate to a scheduled circuit
        scheduled.h(0)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, None)

    def test_unit_seconds_when_using_backend_durations(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500 * self.dt, 1, "s")
        qc.cx(0, 1)
        # usual case
        scheduled = transpile(
            qc, backend=self.backend_with_dt, scheduling_method="alap", layout_method="trivial"
        )
        target_durations = self.backend_with_dt.target.durations()
        cx_duration = target_durations.get("cx", (1, 0))
        sx_duration = target_durations.get("sx", (0,))
        expected_duration = cx_duration + 3 * sx_duration + 320 + 15
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, expected_duration)

    def test_per_qubit_durations(self):
        """Test target with custom instruction_durations"""
        target = GenericBackendV2(
            3,
            coupling_map=[[0, 1], [1, 2]],
            basis_gates=["cx", "h"],
            seed=42,
        ).target
        target.update_instruction_properties("cx", (0, 1), InstructionProperties(0.00001))
        target.update_instruction_properties("cx", (1, 2), InstructionProperties(0.00001))
        target.update_instruction_properties("h", (0,), InstructionProperties(0.000002))
        target.update_instruction_properties("h", (1,), InstructionProperties(0.000002))
        target.update_instruction_properties("h", (2,), InstructionProperties(0.000002))

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.h(1)

        sc = transpile(qc, scheduling_method="alap", target=target)
        self.assertEqual(sc.qubit_start_time(0), 500)
        self.assertEqual(sc.qubit_stop_time(0), 54554)
        self.assertEqual(sc.qubit_start_time(1), 9509)
        self.assertEqual(sc.qubit_stop_time(1), 63563)
        self.assertEqual(sc.qubit_start_time(2), 0)
        self.assertEqual(sc.qubit_stop_time(2), 0)
        self.assertEqual(sc.qubit_start_time(0, 1), 500)
        self.assertEqual(sc.qubit_stop_time(0, 1), 63563)

        qc.measure_all()

        target.update_instruction_properties("measure", (0,), InstructionProperties(0.0001))
        target.update_instruction_properties("measure", (1,), InstructionProperties(0.0001))

        sc = transpile(qc, scheduling_method="alap", target=target)
        q = sc.qubits
        self.assertEqual(sc.qubit_start_time(q[0]), 500)
        self.assertEqual(sc.qubit_stop_time(q[0]), 514013)
        self.assertEqual(sc.qubit_start_time(q[1]), 9509)
        self.assertEqual(sc.qubit_stop_time(q[1]), 514013)
        self.assertEqual(sc.qubit_start_time(q[2]), 63563)
        self.assertEqual(sc.qubit_stop_time(q[2]), 514013)
        self.assertEqual(sc.qubit_start_time(*q), 500)
        self.assertEqual(sc.qubit_stop_time(*q), 514013)

    def test_convert_duration_to_dt(self):
        """Test that circuit duration unit conversion is applied only when necessary.
        Tests fix for bug reported in PR #11782."""

        backend = GenericBackendV2(num_qubits=3, seed=42)

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.measure_all()

        circuit_dt = transpile(circ, backend, scheduling_method="asap")
        with self.assertWarns(DeprecationWarning):
            # reference duration and unit in dt
            ref_duration = circuit_dt.duration
            ref_unit = circuit_dt.unit

            circuit_s = circuit_dt.copy()
            circuit_s.duration *= backend.dt
            circuit_s.unit = "s"

            circuit_ms = circuit_s.copy()
            circuit_ms.duration *= 1000
            circuit_ms.unit = "ms"

            for circuit in [circuit_dt, circuit_s, circuit_ms]:
                with self.subTest(circuit=circuit):
                    converted_circ = convert_durations_to_dt(
                        circuit, dt_in_sec=2.22e-10, inplace=False
                    )
                    self.assertEqual(
                        converted_circ.duration,
                        ref_duration,
                    )
                    self.assertEqual(
                        converted_circ.unit,
                        ref_unit,
                    )

    @data("s", "dt", "f", "p", "n", "u", "µ", "m", "k", "M", "G", "T", "P")
    def test_estimate_duration(self, unit):
        """Test the circuit duration is computed correctly."""
        backend = GenericBackendV2(num_qubits=3, seed=42)

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.measure_all()

        circuit_dt = transpile(circ, backend, scheduling_method="asap")
        duration = circuit_dt.estimate_duration(backend.target, unit=unit)
        expected_in_sec = 1.815516e-06
        expected_val = {
            "s": expected_in_sec,
            "dt": int(expected_in_sec / backend.target.dt),
            "f": expected_in_sec / 1e-15,
            "p": expected_in_sec / 1e-12,
            "n": expected_in_sec / 1e-9,
            "u": expected_in_sec / 1e-6,
            "µ": expected_in_sec / 1e-6,
            "m": expected_in_sec / 1e-3,
            "k": expected_in_sec / 1e3,
            "M": expected_in_sec / 1e6,
            "G": expected_in_sec / 1e9,
            "T": expected_in_sec / 1e12,
            "P": expected_in_sec / 1e15,
        }
        self.assertEqual(duration, expected_val[unit])

    @data("s", "dt", "f", "p", "n", "u", "µ", "m", "k", "M", "G", "T", "P")
    def test_estimate_duration_with_long_delay(self, unit):
        """Test the circuit duration is computed correctly."""
        backend = GenericBackendV2(num_qubits=3, seed=42)

        circ = QuantumCircuit(3)
        circ.cx(0, 1)
        circ.measure_all()
        circ.delay(1e15, 2)

        circuit_dt = transpile(circ, backend, scheduling_method="asap")
        duration = circuit_dt.estimate_duration(backend.target, unit=unit)
        expected_in_sec = 222000.00000139928
        expected_val = {
            "s": expected_in_sec,
            "dt": int(expected_in_sec / backend.target.dt),
            "f": expected_in_sec / 1e-15,
            "p": expected_in_sec / 1e-12,
            "n": expected_in_sec / 1e-9,
            "u": expected_in_sec / 1e-6,
            "µ": expected_in_sec / 1e-6,
            "m": expected_in_sec / 1e-3,
            "k": expected_in_sec / 1e3,
            "M": expected_in_sec / 1e6,
            "G": expected_in_sec / 1e9,
            "T": expected_in_sec / 1e12,
            "P": expected_in_sec / 1e15,
        }
        self.assertEqual(duration, expected_val[unit])

    @data("s", "dt", "f", "p", "n", "u", "µ", "m", "k", "M", "G", "T", "P")
    def test_estimate_duration_with_dt_float(self, unit):
        # This is not a valid use case, but it is still expressible currently
        # since we don't disallow fractional dt values. This should not be assumed
        # to be a part of an api contract. If there is a refactor and this test
        # breaks remove the test it is not valid. This was only added to provide
        # explicit test coverage for a rust code path.
        backend = GenericBackendV2(num_qubits=3, seed=42)

        circ = QuantumCircuit(3)
        circ.cx(0, 1)
        circ.measure_all()
        circ.delay(1.23e15, 2, unit="dt")
        circuit_dt = transpile(circ, backend, scheduling_method="asap")
        duration = circuit_dt.estimate_duration(backend.target, unit=unit)
        expected_in_sec = 273060.0000013993
        expected_val = {
            "s": expected_in_sec,
            "dt": int(expected_in_sec / backend.target.dt),
            "f": expected_in_sec / 1e-15,
            "p": expected_in_sec / 1e-12,
            "n": expected_in_sec / 1e-9,
            "u": expected_in_sec / 1e-6,
            "µ": expected_in_sec / 1e-6,
            "m": expected_in_sec / 1e-3,
            "k": expected_in_sec / 1e3,
            "M": expected_in_sec / 1e6,
            "G": expected_in_sec / 1e9,
            "T": expected_in_sec / 1e12,
            "P": expected_in_sec / 1e15,
        }
        self.assertEqual(duration, expected_val[unit])

    def test_estimate_duration_invalid_unit(self):
        backend = GenericBackendV2(num_qubits=3, seed=42)

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.measure_all()

        circuit_dt = transpile(circ, backend, scheduling_method="asap")
        with self.assertRaises(QiskitError):
            circuit_dt.estimate_duration(backend.target, unit="jiffy")

    def test_delay_circ(self):
        backend = GenericBackendV2(num_qubits=3, seed=42)

        circ = QuantumCircuit(2)
        circ.delay(100, 0, unit="dt")

        circuit_dt = transpile(circ, backend, scheduling_method="asap")
        res = circuit_dt.estimate_duration(backend.target, unit="dt")
        self.assertIsInstance(res, int)
        self.assertEqual(res, 100)

    def test_reset_circ(self):
        backend = GenericBackendV2(num_qubits=2, seed=42)

        circ = QuantumCircuit(1)
        circ.reset(0)

        circuit_reset = transpile(circ, backend, scheduling_method="asap")
        res = circuit_reset.estimate_duration(backend.target, unit="dt")
        self.assertIsInstance(res, int)
        self.assertEqual(res, 9964)

    def test_estimate_duration_control_flow(self):
        backend = GenericBackendV2(num_qubits=3, seed=42, control_flow=True)

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.measure_all()
        with circ.if_test((0, True)):
            circ.x(0)
        with self.assertRaises(QiskitError):
            circ.estimate_duration(backend.target)

    def test_estimate_duration_with_var(self):
        backend = GenericBackendV2(num_qubits=3, seed=42, control_flow=True)

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.measure_all()
        circ.add_var("a", False)
        with self.assertRaises(QiskitError):
            circ.estimate_duration(backend.target)

    def test_estimate_duration_parameterized_delay(self):
        backend = GenericBackendV2(num_qubits=3, seed=42, control_flow=True)

        circ = QuantumCircuit(2)
        circ.cx(0, 1)
        circ.measure_all()
        circ.delay(Parameter("t"), 0)
        with self.assertRaises(QiskitError):
            circ.estimate_duration(backend.target)

    def test_estimate_duration_dt_delay_no_dt(self):
        backend = GenericBackendV2(num_qubits=3, seed=42)
        circ = QuantumCircuit(1)
        circ.delay(100, 0)
        backend.target.dt = None
        with self.assertRaises(QiskitError):
            circ.estimate_duration(backend.target)

    def test_change_dt_in_transpile(self):
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        # default case
        scheduled = transpile(
            qc,
            backend=GenericBackendV2(1, basis_gates=["x"], seed=2, dt=self.dt),
            scheduling_method="asap",
        )
        with self.assertWarns(DeprecationWarning):
            org_duration = scheduled.duration
        # halve dt in sec = double duration in dt
        scheduled = transpile(
            qc,
            backend=GenericBackendV2(1, basis_gates=["x"], seed=2, dt=self.dt),
            scheduling_method="asap",
            dt=self.dt / 2,
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, org_duration * 2)

    @data("asap", "alap")
    def test_duration_on_same_instruction_instance(self, scheduling_method):
        """See: https://github.com/Qiskit/qiskit-terra/issues/5771"""
        backend = GenericBackendV2(3, seed=42, dt=self.dt)
        self.assertNotEqual(
            backend.target.durations().get("cx", qubits=(0, 1), unit="dt"),
            backend.target.durations().get("cx", qubits=(1, 2), unit="dt"),
        )
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        qc.cz(1, 2)
        sc = transpile(qc, backend=backend, scheduling_method=scheduling_method)
        cxs = [inst.operation for inst in sc.data if inst.operation.name == "cx"]
        self.assertEqual(cxs[0], cxs[1])

    # Tests for circuits with parameterized delays
    def test_can_transpile_circuits_after_assigning_parameters(self):
        """Check if not scheduled but duration is converted in dt"""
        idle_dur = Parameter("t")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, "us")
        qc.measure(0, 0)
        qc = qc.assign_parameters({idle_dur: 0.1})
        circ = transpile(qc, self.backend_with_dt)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(circ.duration, None)  # not scheduled
        self.assertEqual(circ.data[1].operation.duration, 450)  # converted in dt

    def test_can_transpile_circuits_with_assigning_parameters_inbetween(self):
        idle_dur = Parameter("t")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, "us")
        qc.measure(0, 0)
        circ = transpile(qc, self.backend_with_dt)
        circ = circ.assign_parameters({idle_dur: 0.1})
        self.assertEqual(circ.data[1].name, "delay")
        self.assertEqual(circ.data[1].params[0], 450)

    def test_can_transpile_circuits_with_unbounded_parameters(self):
        idle_dur = Parameter("t")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, "us")
        qc.measure(0, 0)
        # not assign parameter
        circ = transpile(qc, self.backend_with_dt)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(circ.duration, None)  # not scheduled
        self.assertEqual(circ.data[1].operation.unit, "dt")  # converted in dt
        self.assertEqual(
            circ.data[1].operation.duration, idle_dur * 1e-6 / self.dt
        )  # still parameterized

    @data("asap", "alap")
    def test_can_schedule_circuits_with_bounded_parameters(self, scheduling_method):
        idle_dur = Parameter("t")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, "us")
        qc.measure(0, 0)
        qc = qc.assign_parameters({idle_dur: 0.1})
        circ = transpile(qc, self.backend_with_dt, scheduling_method=scheduling_method)
        with self.assertWarns(DeprecationWarning):
            self.assertIsNotNone(circ.duration)  # scheduled

    @data("asap", "alap")
    def test_fail_to_schedule_circuits_with_unbounded_parameters(self, scheduling_method):
        idle_dur = Parameter("t")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(idle_dur, 0, "us")
        qc.measure(0, 0)

        # unassigned parameter
        with self.assertRaises(TranspilerError):
            transpile(qc, self.backend_with_dt, scheduling_method=scheduling_method)
