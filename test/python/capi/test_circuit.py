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

from qiskit import capi
from qiskit.circuit import QuantumCircuit, Parameter
from test import QiskitTestCase
import ctypes


class TestCircuit(QiskitTestCase):
    """Unit tests for the circuit via the C FFI"""

    def test_circuit_with_multiple_dt_units(self):
        circ = QuantumCircuit(2, 0)
        circ.delay(100, 0, "dt")
        circ.h(0)
        circ.delay(734, 0, "ms")
        circ.cx(0, 1)
        circ.delay(Parameter("a"), 1, "ns")

        c_circ = capi.qk_circuit_borrow_from_python(circ._data)

        expected_param_types = [
            capi.QkParamKind.Int.value.value,
            capi.QkParamKind.Real.value.value,
            capi.QkParamKind.ParameterExpression.value.value,
        ]

        param_result_types = [100, 734.0]

        # Test each delay
        for idx, c_idx in enumerate([0, 2, 4]):
            view = capi.QkCircuitInstruction()
            capi.qk_circuit_get_instruction(c_circ, c_idx, ctypes.byref(view))
            param = view.params[0]

            # Compare the element's kind using its integer value.
            param_kind = capi.qk_param_kind(param)
            self.assertEqual(param_kind, expected_param_types[idx])

            # Check if the integer case happens and load the integer within a pointer.
            if param_kind == capi.QkParamKind.Int.value.value:
                val = ctypes.c_int64(0)
                result = capi.qk_param_as_int(param, ctypes.byref(val))
                self.assertTrue(result, "Getting an integer param as an int failed.")
                self.assertEqual(val.value, param_result_types[idx])

            # Process differs with real as we don't really allocate.
            elif param_kind == capi.QkParamKind.Real.value.value:
                val = capi.qk_param_as_real(param)

                self.assertEqual(val, param_result_types[idx])

                # Reserve space for c_int. Will not be overwritten.
                val_as_int = ctypes.c_int64(-1)

                self.assertFalse(capi.qk_param_as_int(param, ctypes.byref(val_as_int)))
                self.assertEqual(val_as_int.value, -1)
            # Only way of seeing a parameter expression
            else:
                self.assertEqual(param_kind, capi.QkParamKind.ParameterExpression.value.value)
                val = capi.qk_param_str(param)
                cast_val = ctypes.cast(val, ctypes.c_char_p)

                exp = ctypes.c_char_p(b"a")

                self.assertEqual(cast_val.value, exp.value)
                capi.qk_str_free(val)

                # Reserve space for c_int. Will not be overwritten.
                val_as_int = ctypes.c_int64(-1)

                self.assertFalse(capi.qk_param_as_int(param, ctypes.byref(val_as_int)))
                self.assertEqual(val_as_int.value, -1)
            capi.qk_circuit_instruction_clear(view)
