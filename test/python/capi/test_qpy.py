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

import ctypes
import io
import os
from pathlib import Path
import tempfile

from qiskit import QuantumCircuit, capi, qpy
from qiskit.circuit.library import PermutationGate, QFTGate, SdgGate
from qiskit.qpy import dump, common as qpy_common
from test import QiskitTestCase


class TestQpyCAPI(QiskitTestCase):
    def test_min_versions(self):
        """The C API exposes its QPY read and write lower bounds at runtime."""
        self.assertEqual(capi.qk_qpy_read_min_version(), qpy.QPY_COMPATIBILITY_VERSION)
        self.assertEqual(capi.qk_qpy_write_min_version(), qpy_common.QPY_RUST_WRITE_MIN_VERSION)

    def test_dump_python_circuit_data(self):
        """The Python ctypes binding can dump multiple Python-owned CircuitData objects."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        second_circuit = QuantumCircuit(1)
        second_circuit.x(0)

        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 2)(
            capi.qk_circuit_borrow_from_python(circuit._data),
            capi.qk_circuit_borrow_from_python(second_circuit._data),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            filename = Path(tmp_dir) / "circuit.qpy"
            result = capi.qk_qpy_dump_file(
                circuit_ptrs, len(circuit_ptrs), os.fsencode(filename), None
            )

            self.assertEqual(result, capi.QkExitCode.Success.value.value)
            self.assertTrue(filename.is_file())
            with filename.open("rb") as qpy_file:
                loaded = qpy.load(qpy_file)

        self.assertEqual(loaded, [circuit, second_circuit])

    def test_dump_python_circuit_data_with_buffer(self):
        """The Python ctypes binding can dump multiple Python-owned CircuitData objects."""
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])

        second_circuit = QuantumCircuit(1)
        second_circuit.x(0)

        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 2)(
            capi.qk_circuit_borrow_from_python(circuit._data),
            capi.qk_circuit_borrow_from_python(second_circuit._data),
        )
        buffer = ctypes.POINTER(ctypes.c_uint8)()
        size = ctypes.c_size_t()
        error = ctypes.POINTER(ctypes.c_char)()

        # should fail due to unsupported version
        result = capi.qk_qpy_dump_buffer(
            circuit_ptrs,
            len(circuit_ptrs),
            ctypes.byref(buffer),
            ctypes.byref(size),
            ctypes.byref(error),
        )
        self.assertEqual(result, capi.QkExitCode.Success.value.value)
        buffer_array = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8 * size.value))
        data = bytes(buffer_array.contents)
        with io.BytesIO(data) as qpy_buf:
            loaded = qpy.load(qpy_buf)

        capi.qk_qpy_free_buffer(buffer, size)
        self.assertEqual(loaded, [circuit, second_circuit])

    def test_dump_error_message(self):
        """QPY serialization errors include an owned diagnostic string."""
        circuit = QuantumCircuit(1)
        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 1)(
            capi.qk_circuit_borrow_from_python(circuit._data)
        )
        buffer = ctypes.POINTER(ctypes.c_uint8)()
        size = ctypes.c_size_t()
        error = ctypes.POINTER(ctypes.c_char)()

        # should fail due to unsupported version
        result = capi.qk_qpy_dump_buffer_with_version(
            circuit_ptrs,
            len(circuit_ptrs),
            ctypes.byref(buffer),
            ctypes.byref(size),
            16,
            ctypes.byref(error),
        )

        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(b"not supported", ctypes.string_at(error))
        capi.qk_str_free(error)

    def test_python_defined_op(self):
        circuit = QuantumCircuit(3)
        circuit.append(PermutationGate([0, 2, 1]), range(3))
        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 1)(
            capi.qk_circuit_borrow_from_python(circuit._data)
        )
        buffer = ctypes.POINTER(ctypes.c_uint8)()
        size = ctypes.c_size_t()
        error = ctypes.POINTER(ctypes.c_char)()

        # should fail due to unsupported version
        result = capi.qk_qpy_dump_buffer(
            circuit_ptrs,
            len(circuit_ptrs),
            ctypes.byref(buffer),
            ctypes.byref(size),
            ctypes.byref(error),
        )
        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(b"is only available when QPY is invoked from Python", ctypes.string_at(error))
        capi.qk_str_free(error)
        with io.BytesIO() as buf:
            dump(circuit, buf)
            buffer = buf.getvalue()
        length = len(buffer)
        array_type = ctypes.c_ubyte * length
        array_data = array_type.from_buffer(bytearray(buffer))
        output = capi.QkQpyLoadedCircuits(None, 0)
        result = capi.qk_qpy_load_buffer(
            ctypes.byref(output), ctypes.POINTER(ctypes.c_ubyte)(array_data), length, error
        )
        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(b"is only available when QPY is invoked from Python", ctypes.string_at(error))
        capi.qk_str_free(error)

    def test_python_defined_op_with_valid_params(self):
        circuit = QuantumCircuit(3)
        circuit.append(QFTGate(3), range(3))
        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 1)(
            capi.qk_circuit_borrow_from_python(circuit._data)
        )
        buffer = ctypes.POINTER(ctypes.c_uint8)()
        size = ctypes.c_size_t()
        error = ctypes.POINTER(ctypes.c_char)()

        # should fail due to unsupported version
        result = capi.qk_qpy_dump_buffer(
            circuit_ptrs,
            len(circuit_ptrs),
            ctypes.byref(buffer),
            ctypes.byref(size),
            ctypes.byref(error),
        )
        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(
            b"'Python defined instruction' is only available when QPY is invoked from Python",
            ctypes.string_at(error),
        )
        capi.qk_str_free(error)
        with io.BytesIO() as buf:
            dump(circuit, buf)
            buffer = buf.getvalue()
        length = len(buffer)
        array_type = ctypes.c_ubyte * length
        array_data = array_type.from_buffer(bytearray(buffer))
        output = capi.QkQpyLoadedCircuits(None, 0)
        result = capi.qk_qpy_load_buffer(
            ctypes.byref(output), ctypes.POINTER(ctypes.c_ubyte)(array_data), length, error
        )
        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(
            b"'Python defined instructions' is only available when QPY is invoked from Python",
            ctypes.string_at(error),
        )

    def test_python_custom_op(self):
        circuit = QuantumCircuit(6)
        gate = SdgGate().control(5, annotated=True)
        circuit.append(gate, range(6))
        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 1)(
            capi.qk_circuit_borrow_from_python(circuit._data)
        )
        buffer = ctypes.POINTER(ctypes.c_uint8)()
        size = ctypes.c_size_t()
        error = ctypes.POINTER(ctypes.c_char)()

        # should fail due to unsupported version
        result = capi.qk_qpy_dump_buffer(
            circuit_ptrs,
            len(circuit_ptrs),
            ctypes.byref(buffer),
            ctypes.byref(size),
            ctypes.byref(error),
        )
        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(
            b"'Python-defined operations' is only available when QPY is invoked from Python",
            ctypes.string_at(error),
        )
        capi.qk_str_free(error)
        with io.BytesIO() as buf:
            dump(circuit, buf)
            buffer = buf.getvalue()
        length = len(buffer)
        array_type = ctypes.c_ubyte * length
        array_data = array_type.from_buffer(bytearray(buffer))
        output = capi.QkQpyLoadedCircuits(None, 0)
        result = capi.qk_qpy_load_buffer(
            ctypes.byref(output), ctypes.POINTER(ctypes.c_ubyte)(array_data), length, error
        )
        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(
            b"'Custom instructions' is only available when QPY is invoked from Python",
            ctypes.string_at(error),
        )
        capi.qk_str_free(error)

    def test_control_flow(self):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.if_test((circuit.clbits[0], 0)):
            circuit.x(1)
            circuit.measure(1, 1)
        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 1)(
            capi.qk_circuit_borrow_from_python(circuit._data)
        )
        buffer = ctypes.POINTER(ctypes.c_uint8)()
        size = ctypes.c_size_t()
        error = ctypes.POINTER(ctypes.c_char)()

        # should fail due to unsupported version
        result = capi.qk_qpy_dump_buffer(
            circuit_ptrs,
            len(circuit_ptrs),
            ctypes.byref(buffer),
            ctypes.byref(size),
            ctypes.byref(error),
        )
        self.assertEqual(result, capi.QkExitCode.QpyError.value.value)
        self.assertIn(
            b"'Control Flow operations' is only available when QPY is invoked from Python",
            ctypes.string_at(error),
        )
        capi.qk_str_free(error)
        with io.BytesIO() as buf:
            dump(circuit, buf)
            buffer = buf.getvalue()
        length = len(buffer)
        array_type = ctypes.c_ubyte * length
        array_data = array_type.from_buffer(bytearray(buffer))
        output = capi.QkQpyLoadedCircuits(None, 0)
        result = capi.qk_qpy_load_buffer(
            ctypes.byref(output), ctypes.POINTER(ctypes.c_ubyte)(array_data), length, error
        )
        self.assertEqual(result, capi.QkExitCode.Success.value.value)
        self.assertEqual(output.len, 1)
        self.assertEqual(
            capi.qk_circuit_to_python_full(capi.qk_circuit_copy(output.data[0])), circuit
        )
        capi.qk_qpy_loaded_circuits_clear(output)

    def test_dt_delay(self):
        circuit = QuantumCircuit(1)
        circuit.delay(5, 0, "dt")
        circuit_ptrs = (ctypes.POINTER(capi.QkCircuit) * 1)(
            capi.qk_circuit_borrow_from_python(circuit._data)
        )
        buffer = ctypes.POINTER(ctypes.c_uint8)()
        size = ctypes.c_size_t()
        error = ctypes.POINTER(ctypes.c_char)()

        # should fail due to unsupported version
        result = capi.qk_qpy_dump_buffer(
            circuit_ptrs,
            len(circuit_ptrs),
            ctypes.byref(buffer),
            ctypes.byref(size),
            ctypes.byref(error),
        )
        self.assertEqual(result, capi.QkExitCode.Success.value.value)
        buffer_array = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8 * size.value))
        data = bytes(buffer_array.contents)
        length = len(data)
        array_type = ctypes.c_ubyte * length
        array_data = array_type.from_buffer(bytearray(data))
        with io.BytesIO(data) as qpy_buf:
            loaded = qpy.load(qpy_buf)

        self.assertEqual(loaded, [circuit])
        output = capi.QkQpyLoadedCircuits(None, 0)
        result = capi.qk_qpy_load_buffer(
            ctypes.byref(output), ctypes.POINTER(ctypes.c_ubyte)(array_data), length, error
        )
        self.assertEqual(result, capi.QkExitCode.Success.value.value)
        self.assertEqual(output.len, 1)
        self.assertEqual(
            capi.qk_circuit_to_python_full(capi.qk_circuit_copy(output.data[0])), circuit
        )
        capi.qk_qpy_free_buffer(buffer, size)
        instruction = capi.QkCircuitInstruction()
        capi.qk_circuit_get_instruction(output.data[0], 0, ctypes.byref(instruction))
        self.assertEqual(
            capi.QkParamKind.Int.value.value, capi.qk_param_kind(instruction.params[0])
        )
        param_val = ctypes.c_int64()
        self.assertTrue(capi.qk_param_as_int(instruction.params[0], ctypes.byref(param_val)))
        self.assertEqual(param_val.value, 5)
        capi.qk_qpy_loaded_circuits_clear(output)
