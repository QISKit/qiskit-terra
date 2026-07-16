# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Binary IO for any value objects, such as numbers, string, parameters."""

from __future__ import annotations

import collections.abc
import io
import struct
import uuid
from dataclasses import dataclass

import numpy as np

from qiskit.circuit import CASE_DEFAULT, Clbit, ClassicalRegister, Duration
from qiskit.circuit.classical import expr, types
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import (
    ParameterExpression,
    ParameterValueType,
    OpCode,
    op_code_to_method,
)
from qiskit.circuit.parametervector import ParameterVector
from qiskit.qpy import common, formats, exceptions, type_keys
from qiskit.qpy.binary_io.parse_sympy_repr import parse_sympy_repr


@dataclass
class _INSTRUCTION:
    op: OpCode
    lhs: ParameterValueType | None
    rhs: ParameterValueType | None = None


@dataclass
class _SUBS:
    binds: dict
    op: OpCode = OpCode.SUBSTITUTE


def _write_parameter(file_obj, obj):
    name_bytes = obj.name.encode(common.ENCODE)
    file_obj.write(struct.pack(formats.PARAMETER_PACK, len(name_bytes), obj.uuid.bytes))
    file_obj.write(name_bytes)


def _write_parameter_vec(file_obj, obj):
    name_bytes = obj.vector.name.encode(common.ENCODE)
    file_obj.write(
        struct.pack(
            formats.PARAMETER_VECTOR_ELEMENT_PACK,
            len(name_bytes),
            len(obj.vector),
            obj.uuid.bytes,
            obj.index,
        )
    )
    file_obj.write(name_bytes)


def _encode_replay_entry(inst, file_obj, version, r_side=False):
    inst_type = None
    inst_data = None
    if inst is None:
        inst_type = "n"
        inst_data = b"\x00"
    elif isinstance(inst, Parameter):
        inst_type = "p"
        inst_data = inst.uuid.bytes
    elif isinstance(inst, complex):
        inst_type = "c"
        inst_data = struct.pack("!dd", inst.real, inst.imag)
    elif isinstance(inst, float):
        inst_type = "f"
        inst_data = struct.pack("!Qd", 0, inst)
    elif isinstance(inst, int):
        inst_type = "i"
        inst_data = struct.pack("!Qq", 0, inst)
    elif isinstance(inst, ParameterExpression):
        if not r_side:
            entry = struct.pack(
                formats.PARAM_EXPR_ELEM_V13_PACK,
                255,
                b"s",
                b"\x00",
                b"n",
                b"\x00",
            )
        else:
            entry = struct.pack(
                formats.PARAM_EXPR_ELEM_V13_PACK,
                255,
                b"n",
                b"\x00",
                b"s",
                b"\x00",
            )
        file_obj.write(entry)
        _write_parameter_expression_v13(file_obj, inst, version)
        if not r_side:
            entry = struct.pack(
                formats.PARAM_EXPR_ELEM_V13_PACK,
                255,
                b"e",
                b"\x00",
                b"n",
                b"\x00",
            )
        else:
            entry = struct.pack(
                formats.PARAM_EXPR_ELEM_V13_PACK,
                255,
                b"n",
                b"\x00",
                b"e",
                b"\x00",
            )
        file_obj.write(entry)
        inst_type = "n"
        inst_data = b"\x00"
    else:
        raise exceptions.QpyError("Invalid parameter expression type")
    return inst_type, inst_data


def _encode_replay_subs(subs, file_obj, version):
    with io.BytesIO() as mapping_buf:
        if version < 15:
            subs_dict = {k.name: v for k, v in subs.binds.items()}
        else:
            subs_dict = {k.uuid.bytes: v for k, v in subs.binds.items()}
        common.write_mapping(
            mapping_buf, mapping=subs_dict, serializer=dumps_value, version=version
        )
        data = mapping_buf.getvalue()
    entry = struct.pack(
        formats.PARAM_EXPR_ELEM_V13_PACK,
        subs.op,
        b"u",
        struct.pack("!QQ", len(data), 0),
        b"n",
        b"\x00",
    )
    file_obj.write(entry)
    file_obj.write(data)
    return subs.binds


def _write_parameter_expression_v13(file_obj, obj, version):
    # A symbol is `Parameter` or `ParameterVectorElement`.
    # `symbol_map` maps symbols to ParameterExpression (which may be a symbol).
    symbol_map = {}
    for inst in obj._qpy_replay:
        if isinstance(inst, _SUBS):
            symbol_map.update(_encode_replay_subs(inst, file_obj, version))
            continue
        lhs_type, lhs = _encode_replay_entry(inst.lhs, file_obj, version)
        rhs_type, rhs = _encode_replay_entry(inst.rhs, file_obj, version, True)
        entry = struct.pack(
            formats.PARAM_EXPR_ELEM_V13_PACK,
            int(inst.op),
            lhs_type.encode("utf8"),
            lhs,
            rhs_type.encode("utf8"),
            rhs,
        )
        file_obj.write(entry)
    return symbol_map


def _write_parameter_expression(file_obj, obj, use_symengine, *, version):
    extra_symbols = None
    with io.BytesIO() as buf:
        extra_symbols = _write_parameter_expression_v13(buf, obj, version)
        expr_bytes = buf.getvalue()

    parameters = obj.parameters
    symbol_table_len = len(parameters)
    if extra_symbols:
        symbol_table_len += 2 * len(extra_symbols)
    param_expr_header_raw = struct.pack(
        formats.PARAMETER_EXPR_PACK, symbol_table_len, len(expr_bytes)
    )
    file_obj.write(param_expr_header_raw)
    file_obj.write(expr_bytes)
    for symbol in parameters:
        symbol_key = type_keys.Value.assign(symbol)

        # serialize key
        if symbol_key == type_keys.Value.PARAMETER_VECTOR:
            symbol_data = common.data_to_binary(symbol, _write_parameter_vec)
        else:
            symbol_data = common.data_to_binary(symbol, _write_parameter)

        # serialize value
        value_key = symbol_key
        value_data = b""

        elem_header = struct.pack(
            formats.PARAM_EXPR_MAP_ELEM_V3_PACK,
            symbol_key,
            value_key,
            len(value_data),
        )
        file_obj.write(elem_header)
        file_obj.write(symbol_data)
        file_obj.write(value_data)
    if extra_symbols:
        for symbol in extra_symbols:
            symbol_key = type_keys.Value.assign(symbol)
            # serialize key
            if symbol_key == type_keys.Value.PARAMETER_VECTOR:
                symbol_data = common.data_to_binary(symbol, _write_parameter_vec)
            else:
                symbol_data = common.data_to_binary(symbol, _write_parameter)
            # serialize value
            value_key, value_data = dumps_value(
                symbol, version=version, use_symengine=use_symengine
            )

            elem_header = struct.pack(
                formats.PARAM_EXPR_MAP_ELEM_V3_PACK,
                symbol_key,
                value_key,
                len(value_data),
            )
            file_obj.write(elem_header)
            file_obj.write(symbol_data)
            file_obj.write(value_data)


class _ExprWriter(expr.ExprVisitor[None]):
    __slots__ = ("clbit_indices", "file_obj", "standalone_var_indices", "version")

    def __init__(self, file_obj, clbit_indices, standalone_var_indices, version):
        self.file_obj = file_obj
        self.clbit_indices = clbit_indices
        self.standalone_var_indices = standalone_var_indices
        self.version = version

    def _write_expr_type(self, type_, /):
        _write_expr_type(self.file_obj, type_, self.version)

    def visit_generic(self, node, /):
        raise exceptions.QpyError(f"unhandled Expr object '{node}'")

    def visit_var(self, node, /):
        self.file_obj.write(type_keys.Expression.VAR)
        self._write_expr_type(node.type)
        if node.standalone:
            self.file_obj.write(type_keys.ExprVar.UUID)
            self.file_obj.write(
                struct.pack(
                    formats.EXPR_VAR_UUID_PACK,
                    *formats.EXPR_VAR_UUID(self.standalone_var_indices[node]),
                )
            )
        elif isinstance(node.var, Clbit):
            self.file_obj.write(type_keys.ExprVar.CLBIT)
            self.file_obj.write(
                struct.pack(
                    formats.EXPR_VAR_CLBIT_PACK,
                    *formats.EXPR_VAR_CLBIT(self.clbit_indices[node.var]),
                )
            )
        elif isinstance(node.var, ClassicalRegister):
            self.file_obj.write(type_keys.ExprVar.REGISTER)
            self.file_obj.write(
                struct.pack(
                    formats.EXPR_VAR_REGISTER_PACK, *formats.EXPR_VAR_REGISTER(len(node.var.name))
                )
            )
            self.file_obj.write(node.var.name.encode(common.ENCODE))
        else:
            raise exceptions.QpyError(f"unhandled Var object '{node.var}'")

    def visit_stretch(self, node, /):
        self.file_obj.write(type_keys.Expression.STRETCH)
        self._write_expr_type(node.type)
        self.file_obj.write(
            struct.pack(
                formats.EXPRESSION_STRETCH_PACK,
                *formats.EXPRESSION_STRETCH(self.standalone_var_indices[node]),
            )
        )

    def visit_value(self, node, /):
        self.file_obj.write(type_keys.Expression.VALUE)
        self._write_expr_type(node.type)
        if node.value is True or node.value is False:
            self.file_obj.write(type_keys.ExprValue.BOOL)
            self.file_obj.write(
                struct.pack(formats.EXPR_VALUE_BOOL_PACK, *formats.EXPR_VALUE_BOOL(node.value))
            )
        elif isinstance(node.value, int):
            self.file_obj.write(type_keys.ExprValue.INT)
            if node.value == 0:
                num_bytes = 0
                buffer = b""
            else:
                # This wastes a byte for `-(2 ** (8*n - 1))` for natural `n`, but they'll still
                # decode fine so it's not worth another special case.  They'll encode to
                # b"\xff\x80\x00\x00...", but we could encode them to b"\x80\x00\x00...".
                num_bytes = (node.value.bit_length() // 8) + 1
                buffer = node.value.to_bytes(num_bytes, "big", signed=True)
            self.file_obj.write(
                struct.pack(formats.EXPR_VALUE_INT_PACK, *formats.EXPR_VALUE_INT(num_bytes))
            )
            self.file_obj.write(buffer)
        elif isinstance(node.value, float):
            self.file_obj.write(type_keys.ExprValue.FLOAT)
            self.file_obj.write(
                struct.pack(formats.EXPR_VALUE_FLOAT_PACK, *formats.EXPR_VALUE_FLOAT(node.value))
            )
        elif isinstance(node.value, Duration):
            if self.version < 16 and node.value.unit() == "ps":
                raise exceptions.UnsupportedFeatureForVersion(
                    "Duration variant 'Duration.ps'", required=16, target=self.version
                )
            self.file_obj.write(type_keys.ExprValue.DURATION)
            _write_duration(self.file_obj, node.value)
        else:
            raise exceptions.QpyError(f"unhandled Value object '{node.value}'")

    def visit_cast(self, node, /):
        self.file_obj.write(type_keys.Expression.CAST)
        self._write_expr_type(node.type)
        self.file_obj.write(
            struct.pack(formats.EXPRESSION_CAST_PACK, *formats.EXPRESSION_CAST(node.implicit))
        )
        node.operand.accept(self)

    def visit_unary(self, node, /):
        self.file_obj.write(type_keys.Expression.UNARY)
        self._write_expr_type(node.type)
        self.file_obj.write(
            struct.pack(formats.EXPRESSION_UNARY_PACK, *formats.EXPRESSION_UNARY(node.op.value))
        )
        node.operand.accept(self)

    def visit_binary(self, node, /):
        self.file_obj.write(type_keys.Expression.BINARY)
        self._write_expr_type(node.type)
        self.file_obj.write(
            struct.pack(formats.EXPRESSION_BINARY_PACK, *formats.EXPRESSION_BINARY(node.op.value))
        )
        node.left.accept(self)
        node.right.accept(self)

    def visit_index(self, node, /):
        if self.version < 12:
            raise exceptions.UnsupportedFeatureForVersion(
                "the 'Index' expression", required=12, target=self.version
            )
        self.file_obj.write(type_keys.Expression.INDEX)
        self._write_expr_type(node.type)
        node.target.accept(self)
        node.index.accept(self)


def _write_expr(
    file_obj,
    node: expr.Expr,
    clbit_indices: collections.abc.Mapping[Clbit, int],
    standalone_var_indices: collections.abc.Mapping[expr.Var, int],
    version: int,
):
    node.accept(_ExprWriter(file_obj, clbit_indices, standalone_var_indices, version))


def _write_expr_type(file_obj, type_: types.Type, version: int):
    if type_.kind is types.Bool:
        file_obj.write(type_keys.ExprType.BOOL)
    elif type_.kind is types.Uint:
        file_obj.write(type_keys.ExprType.UINT)
        file_obj.write(
            struct.pack(formats.EXPR_TYPE_UINT_PACK, *formats.EXPR_TYPE_UINT(type_.width))
        )
    elif type_.kind is types.Float:
        if version < 14:
            raise exceptions.UnsupportedFeatureForVersion(
                "float-typed expressions", required=14, target=version
            )
        file_obj.write(type_keys.ExprType.FLOAT)
    elif type_.kind is types.Duration:
        if version < 14:
            raise exceptions.UnsupportedFeatureForVersion(
                "duration-typed expressions", required=14, target=version
            )
        file_obj.write(type_keys.ExprType.DURATION)
    else:
        raise exceptions.QpyError(f"unhandled Type object '{type_};")


def _write_duration(file_obj, duration: Duration):
    unit = duration.unit()
    if unit == "dt":
        file_obj.write(type_keys.CircuitDuration.DT)
        file_obj.write(
            struct.pack(formats.DURATION_DT_PACK, *formats.DURATION_DT(duration.value()))
        )
    elif unit == "ps":
        file_obj.write(type_keys.CircuitDuration.PS)
        file_obj.write(
            struct.pack(formats.DURATION_PS_PACK, *formats.DURATION_PS(duration.value()))
        )
    elif unit == "ns":
        file_obj.write(type_keys.CircuitDuration.NS)
        file_obj.write(
            struct.pack(formats.DURATION_NS_PACK, *formats.DURATION_NS(duration.value()))
        )
    elif unit == "us":
        file_obj.write(type_keys.CircuitDuration.US)
        file_obj.write(
            struct.pack(formats.DURATION_US_PACK, *formats.DURATION_US(duration.value()))
        )
    elif unit == "ms":
        file_obj.write(type_keys.CircuitDuration.MS)
        file_obj.write(
            struct.pack(formats.DURATION_MS_PACK, *formats.DURATION_MS(duration.value()))
        )
    elif unit == "s":
        file_obj.write(type_keys.CircuitDuration.S)
        file_obj.write(struct.pack(formats.DURATION_S_PACK, *formats.DURATION_S(duration.value())))
    else:
        raise exceptions.QpyError(f"unhandled Duration object '{duration};")


def _read_parameter(file_obj):
    data = formats.PARAMETER(
        *struct.unpack(formats.PARAMETER_PACK, file_obj.read(formats.PARAMETER_SIZE))
    )
    param_uuid = uuid.UUID(bytes=data.uuid)
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    return Parameter(name, uuid=param_uuid)


def _read_parameter_vec(file_obj, vectors):
    data = formats.PARAMETER_VECTOR_ELEMENT(
        *struct.unpack(
            formats.PARAMETER_VECTOR_ELEMENT_PACK,
            file_obj.read(formats.PARAMETER_VECTOR_ELEMENT_SIZE),
        ),
    )
    # Starting in version 15, the parameter vector root uuid
    # is used as a key instead of the parameter name.
    root_uuid_int = uuid.UUID(bytes=data.uuid).int - data.index
    root_uuid = uuid.UUID(int=root_uuid_int)
    name = file_obj.read(data.vector_name_size).decode(common.ENCODE)
    if (vector := vectors.get(root_uuid, None)) is None:
        vector = vectors[root_uuid] = ParameterVector(name, data.vector_size, uuid=root_uuid)
    return vector[data.index]


def _read_parameter_expression(file_obj):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )

    sympy_str = file_obj.read(data.expr_size).decode(common.ENCODE)
    expr_ = parse_sympy_repr(sympy_str)
    name_map = {}
    for _ in range(data.map_elements):
        elem_data = formats.PARAM_EXPR_MAP_ELEM(
            *struct.unpack(
                formats.PARAM_EXPR_MAP_ELEM_PACK,
                file_obj.read(formats.PARAM_EXPR_MAP_ELEM_SIZE),
            )
        )
        symbol = _read_parameter(file_obj)

        elem_key = type_keys.Value(elem_data.type)
        binary_data = file_obj.read(elem_data.size)
        if elem_key == type_keys.Value.INTEGER:
            value = struct.unpack("!q", binary_data)
        elif elem_key == type_keys.Value.FLOAT:
            value = struct.unpack("!d", binary_data)
        elif elem_key == type_keys.Value.COMPLEX:
            value = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
        elif elem_key == type_keys.Value.PARAMETER:
            value = symbol
        elif elem_key == type_keys.Value.PARAMETER_EXPRESSION:
            value = common.data_from_binary(binary_data, _read_parameter_expression)
        else:
            raise exceptions.QpyError(f"Invalid parameter expression map type: {elem_key}")
        name_map[symbol.name] = value

    return ParameterExpression(name_map, str(expr_))


def _read_parameter_expression_v3(file_obj, vectors, use_symengine):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )

    payload = file_obj.read(data.expr_size)
    if use_symengine:
        expr_ = common.load_symengine_payload(payload)
    else:
        sympy_str = payload.decode(common.ENCODE)
        expr_ = parse_sympy_repr(sympy_str)

    name_map = {}
    for _ in range(data.map_elements):
        elem_data = formats.PARAM_EXPR_MAP_ELEM_V3(
            *struct.unpack(
                formats.PARAM_EXPR_MAP_ELEM_V3_PACK,
                file_obj.read(formats.PARAM_EXPR_MAP_ELEM_V3_SIZE),
            )
        )
        symbol_key = type_keys.Value(elem_data.symbol_type)

        if symbol_key == type_keys.Value.PARAMETER:
            symbol = _read_parameter(file_obj)
        elif symbol_key == type_keys.Value.PARAMETER_VECTOR:
            symbol = _read_parameter_vec(file_obj, vectors)
        else:
            raise exceptions.QpyError(f"Invalid parameter expression map type: {symbol_key}")

        elem_key = type_keys.Value(elem_data.type)
        binary_data = file_obj.read(elem_data.size)
        if elem_key == type_keys.Value.INTEGER:
            value = struct.unpack("!q", binary_data)
        elif elem_key == type_keys.Value.FLOAT:
            value = struct.unpack("!d", binary_data)
        elif elem_key == type_keys.Value.COMPLEX:
            value = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
        elif elem_key in (type_keys.Value.PARAMETER, type_keys.Value.PARAMETER_VECTOR):
            value = symbol
        elif elem_key == type_keys.Value.PARAMETER_EXPRESSION:
            value = common.data_from_binary(
                binary_data,
                _read_parameter_expression_v3,
                vectors=vectors,
                use_symengine=use_symengine,
            )
        else:
            raise exceptions.QpyError(f"Invalid parameter expression map type: {elem_key}")
        name_map[symbol.name] = value

    return ParameterExpression(name_map, str(expr_))


def _read_parameter_expression_v13(file_obj, vectors, version):
    data = formats.PARAMETER_EXPR(
        *struct.unpack(formats.PARAMETER_EXPR_PACK, file_obj.read(formats.PARAMETER_EXPR_SIZE))
    )

    payload = file_obj.read(data.expr_size)

    symbol_map = {}
    for _ in range(data.map_elements):
        elem_data = formats.PARAM_EXPR_MAP_ELEM_V3(
            *struct.unpack(
                formats.PARAM_EXPR_MAP_ELEM_V3_PACK,
                file_obj.read(formats.PARAM_EXPR_MAP_ELEM_V3_SIZE),
            )
        )
        symbol_key = type_keys.Value(elem_data.symbol_type)

        if symbol_key == type_keys.Value.PARAMETER:
            symbol = _read_parameter(file_obj)
        elif symbol_key == type_keys.Value.PARAMETER_VECTOR:
            symbol = _read_parameter_vec(file_obj, vectors)
        elif symbol_key == type_keys.Value.PARAMETER_EXPRESSION:
            symbol = _read_parameter_expression_v13(file_obj, vectors, version)
        else:
            raise exceptions.QpyError(f"Invalid parameter expression map type: {symbol_key}")

        elem_key = type_keys.Value(elem_data.type)

        binary_data = file_obj.read(elem_data.size)
        if elem_key == type_keys.Value.INTEGER:
            value = struct.unpack("!q", binary_data)
        elif elem_key == type_keys.Value.FLOAT:
            value = struct.unpack("!d", binary_data)
        elif elem_key == type_keys.Value.COMPLEX:
            value = complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
        elif elem_key in (type_keys.Value.PARAMETER, type_keys.Value.PARAMETER_VECTOR):
            value = symbol
        elif elem_key == type_keys.Value.PARAMETER_EXPRESSION:
            value = common.data_from_binary(
                binary_data,
                _read_parameter_expression_v13,
                vectors=vectors,
                version=version,
            )
        else:
            raise exceptions.QpyError(f"Invalid parameter expression map type: {elem_key}")
        symbol_map[symbol] = value
    with io.BytesIO(payload) as buf:
        return _read_parameter_expr_v13(buf, symbol_map, version, vectors)


def _read_parameter_expr_v13(buf, symbol_map, version, vectors):
    param_uuid_map = {symbol.uuid: symbol for symbol in symbol_map if isinstance(symbol, Parameter)}
    name_map = {str(v): k for k, v in symbol_map.items()}
    data = buf.read(formats.PARAM_EXPR_ELEM_V13_SIZE)
    stack = []
    while data:
        expression_data = formats.PARAM_EXPR_ELEM_V13._make(
            struct.unpack(formats.PARAM_EXPR_ELEM_V13_PACK, data)
        )
        # LHS
        if expression_data.LHS_TYPE == b"p":
            stack.append(param_uuid_map[uuid.UUID(bytes=expression_data.LHS)])
        elif expression_data.LHS_TYPE == b"f":
            stack.append(struct.unpack("!Qd", expression_data.LHS)[1])
        elif expression_data.LHS_TYPE == b"n":
            pass
        elif expression_data.LHS_TYPE == b"c":
            stack.append(complex(*struct.unpack("!dd", expression_data.LHS)))
        elif expression_data.LHS_TYPE == b"i":
            stack.append(struct.unpack("!Qq", expression_data.LHS)[1])
        elif expression_data.LHS_TYPE == b"s":
            data = buf.read(formats.PARAM_EXPR_ELEM_V13_SIZE)
            continue
        elif expression_data.LHS_TYPE == b"e":
            data = buf.read(formats.PARAM_EXPR_ELEM_V13_SIZE)
            continue
        elif expression_data.LHS_TYPE == b"u":
            size = struct.unpack_from("!QQ", expression_data.LHS)[0]
            subs_map_data = buf.read(size)
            with io.BytesIO(subs_map_data) as mapping_buf:
                mapping = common.read_mapping(
                    mapping_buf,
                    deserializer=loads_value,
                    version=version,
                    vectors=vectors,
                )
            # Starting in version 15, the uuid is used instead of the name
            if version < 15:
                stack.append({name_map[k]: v for k, v in mapping.items()})
            else:
                stack.append({param_uuid_map[k]: v for k, v in mapping.items()})
        else:
            raise exceptions.QpyError(
                "Unknown ParameterExpression operation type {expression_data.LHS_TYPE}"
            )
        # RHS
        if expression_data.RHS_TYPE == b"p":
            stack.append(param_uuid_map[uuid.UUID(bytes=expression_data.RHS)])
        elif expression_data.RHS_TYPE == b"f":
            stack.append(struct.unpack("!Qd", expression_data.RHS)[1])
        elif expression_data.RHS_TYPE == b"n":
            pass
        elif expression_data.RHS_TYPE == b"c":
            stack.append(complex(*struct.unpack("!dd", expression_data.RHS)))
        elif expression_data.RHS_TYPE == b"i":
            stack.append(struct.unpack("!Qq", expression_data.RHS)[1])
        elif expression_data.RHS_TYPE == b"s":
            data = buf.read(formats.PARAM_EXPR_ELEM_V13_SIZE)
            continue
        elif expression_data.RHS_TYPE == b"e":
            data = buf.read(formats.PARAM_EXPR_ELEM_V13_SIZE)
            continue
        else:
            raise exceptions.QpyError(
                f"Unknown ParameterExpression operation type {expression_data.RHS_TYPE}"
            )
        if expression_data.OP_CODE == 255:
            continue
        method_str = op_code_to_method(expression_data.OP_CODE)
        if expression_data.OP_CODE in (0, 1, 2, 3, 4, 13, 15, 18, 19, 20):
            rhs = stack.pop()
            lhs = stack.pop()
            if not isinstance(lhs, ParameterExpression):
                lhs = ParameterExpression._Value(lhs)
            stack.append(getattr(lhs, method_str)(rhs))
        else:
            lhs = stack.pop()
            stack.append(getattr(lhs, method_str)())
        data = buf.read(formats.PARAM_EXPR_ELEM_V13_SIZE)

    return stack.pop()


def _read_expr(
    file_obj,
    clbits: collections.abc.Sequence[Clbit],
    cregs: collections.abc.Mapping[str, ClassicalRegister],
    standalone_vars: collections.abc.Sequence[expr.Var],
) -> expr.Expr:

    type_key = file_obj.read(formats.EXPRESSION_DISCRIMINATOR_SIZE)
    type_ = _read_expr_type(file_obj)
    if type_key == type_keys.Expression.VAR:
        var_type_key = file_obj.read(formats.EXPR_VAR_DISCRIMINATOR_SIZE)
        if var_type_key == type_keys.ExprVar.UUID:
            payload = formats.EXPR_VAR_UUID._make(
                struct.unpack(formats.EXPR_VAR_UUID_PACK, file_obj.read(formats.EXPR_VAR_UUID_SIZE))
            )
            return standalone_vars[payload.var_index]
        if var_type_key == type_keys.ExprVar.CLBIT:
            payload = formats.EXPR_VAR_CLBIT._make(
                struct.unpack(
                    formats.EXPR_VAR_CLBIT_PACK, file_obj.read(formats.EXPR_VAR_CLBIT_SIZE)
                )
            )
            return expr.Var(clbits[payload.index], type_)
        if var_type_key == type_keys.ExprVar.REGISTER:
            payload = formats.EXPR_VAR_REGISTER._make(
                struct.unpack(
                    formats.EXPR_VAR_REGISTER_PACK, file_obj.read(formats.EXPR_VAR_REGISTER_SIZE)
                )
            )
            name = file_obj.read(payload.reg_name_size).decode(common.ENCODE)
            return expr.Var(cregs[name], type_)
        raise exceptions.QpyError("Invalid classical-expression Var key '{var_type_key}'")
    if type_key == type_keys.Expression.STRETCH:
        payload = formats.EXPRESSION_STRETCH._make(
            struct.unpack(
                formats.EXPRESSION_STRETCH_PACK, file_obj.read(formats.EXPRESSION_STRETCH_SIZE)
            )
        )
        return standalone_vars[payload.var_index]
    if type_key == type_keys.Expression.VALUE:
        value_type_key = file_obj.read(formats.EXPR_VALUE_DISCRIMINATOR_SIZE)
        if value_type_key == type_keys.ExprValue.BOOL:
            payload = formats.EXPR_VALUE_BOOL._make(
                struct.unpack(
                    formats.EXPR_VALUE_BOOL_PACK, file_obj.read(formats.EXPR_VALUE_BOOL_SIZE)
                )
            )
            return expr.Value(payload.value, type_)
        if value_type_key == type_keys.ExprValue.INT:
            payload = formats.EXPR_VALUE_INT._make(
                struct.unpack(
                    formats.EXPR_VALUE_INT_PACK, file_obj.read(formats.EXPR_VALUE_INT_SIZE)
                )
            )
            return expr.Value(
                int.from_bytes(file_obj.read(payload.num_bytes), "big", signed=True), type_
            )
        if value_type_key == type_keys.ExprValue.FLOAT:
            payload = formats.EXPR_VALUE_FLOAT._make(
                struct.unpack(
                    formats.EXPR_VALUE_FLOAT_PACK, file_obj.read(formats.EXPR_VALUE_FLOAT_SIZE)
                )
            )
            return expr.Value(payload.value, type_)
        if value_type_key == type_keys.ExprValue.DURATION:
            value = _read_duration(file_obj)
            return expr.Value(value, type_)
        raise exceptions.QpyError("Invalid classical-expression Value key '{value_type_key}'")
    if type_key == type_keys.Expression.CAST:
        payload = formats.EXPRESSION_CAST._make(
            struct.unpack(formats.EXPRESSION_CAST_PACK, file_obj.read(formats.EXPRESSION_CAST_SIZE))
        )
        return expr.Cast(
            _read_expr(file_obj, clbits, cregs, standalone_vars), type_, implicit=payload.implicit
        )
    if type_key == type_keys.Expression.UNARY:
        payload = formats.EXPRESSION_UNARY._make(
            struct.unpack(
                formats.EXPRESSION_UNARY_PACK, file_obj.read(formats.EXPRESSION_UNARY_SIZE)
            )
        )
        return expr.Unary(
            expr.Unary.Op(payload.opcode),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            type_,
        )
    if type_key == type_keys.Expression.BINARY:
        payload = formats.EXPRESSION_BINARY._make(
            struct.unpack(
                formats.EXPRESSION_BINARY_PACK, file_obj.read(formats.EXPRESSION_BINARY_SIZE)
            )
        )
        return expr.Binary(
            expr.Binary.Op(payload.opcode),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            type_,
        )
    if type_key == type_keys.Expression.INDEX:
        return expr.Index(
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            _read_expr(file_obj, clbits, cregs, standalone_vars),
            type_,
        )
    raise exceptions.QpyError(f"Invalid classical-expression Expr key '{type_key}'")


def _read_expr_type(file_obj) -> types.Type:
    type_key = file_obj.read(formats.EXPR_TYPE_DISCRIMINATOR_SIZE)
    if type_key == type_keys.ExprType.BOOL:
        return types.Bool()
    if type_key == type_keys.ExprType.UINT:
        elem = formats.EXPR_TYPE_UINT._make(
            struct.unpack(formats.EXPR_TYPE_UINT_PACK, file_obj.read(formats.EXPR_TYPE_UINT_SIZE))
        )
        return types.Uint(elem.width)
    if type_key == type_keys.ExprType.FLOAT:
        return types.Float()
    if type_key == type_keys.ExprType.DURATION:
        return types.Duration()
    raise exceptions.QpyError(f"Invalid classical-expression Type key '{type_key}'")


def _read_duration(file_obj) -> Duration:
    type_key = file_obj.read(formats.DURATION_DISCRIMINATOR_SIZE)
    if type_key == type_keys.CircuitDuration.DT:
        elem = formats.DURATION_DT._make(
            struct.unpack(formats.DURATION_DT_PACK, file_obj.read(formats.DURATION_DT_SIZE))
        )
        return Duration.dt(elem.value)
    if type_key == type_keys.CircuitDuration.PS:
        elem = formats.DURATION_PS._make(
            struct.unpack(formats.DURATION_PS_PACK, file_obj.read(formats.DURATION_PS_SIZE))
        )
        return Duration.ps(elem.value)
    if type_key == type_keys.CircuitDuration.NS:
        elem = formats.DURATION_NS._make(
            struct.unpack(formats.DURATION_NS_PACK, file_obj.read(formats.DURATION_NS_SIZE))
        )
        return Duration.ns(elem.value)
    if type_key == type_keys.CircuitDuration.US:
        elem = formats.DURATION_US._make(
            struct.unpack(formats.DURATION_US_PACK, file_obj.read(formats.DURATION_US_SIZE))
        )
        return Duration.us(elem.value)
    if type_key == type_keys.CircuitDuration.MS:
        elem = formats.DURATION_MS._make(
            struct.unpack(formats.DURATION_MS_PACK, file_obj.read(formats.DURATION_MS_SIZE))
        )
        return Duration.ms(elem.value)
    if type_key == type_keys.CircuitDuration.S:
        elem = formats.DURATION_S._make(
            struct.unpack(formats.DURATION_S_PACK, file_obj.read(formats.DURATION_S_SIZE))
        )
        return Duration.s(elem.value)
    raise exceptions.QpyError(f"Invalid duration Type key '{type_key}'")


def read_standalone_vars(file_obj, num_vars):
    """Read the ``num_vars`` standalone variable declarations from the file.

    Args:
        file_obj (File): a file-like object to read from.
        num_vars (int): the number of variables to read.

    Returns:
        tuple[dict, list]: the first item is a mapping of the ``ExprVarDeclaration`` type keys to
        the variables defined by that type key, and the second is the total order of variable
        declarations.
    """
    read_vars = {
        type_keys.ExprVarDeclaration.INPUT: [],
        type_keys.ExprVarDeclaration.CAPTURE: [],
        type_keys.ExprVarDeclaration.LOCAL: [],
        type_keys.ExprVarDeclaration.STRETCH_CAPTURE: [],
        type_keys.ExprVarDeclaration.STRETCH_LOCAL: [],
    }
    var_order = []
    for _ in range(num_vars):
        data = formats.EXPR_VAR_DECLARATION._make(
            struct.unpack(
                formats.EXPR_VAR_DECLARATION_PACK,
                file_obj.read(formats.EXPR_VAR_DECLARATION_SIZE),
            )
        )
        type_ = _read_expr_type(file_obj)
        name = file_obj.read(data.name_size).decode(common.ENCODE)
        if data.usage in {
            type_keys.ExprVarDeclaration.STRETCH_CAPTURE,
            type_keys.ExprVarDeclaration.STRETCH_LOCAL,
        }:
            var = expr.Stretch(uuid.UUID(bytes=data.uuid_bytes), name)
        else:
            var = expr.Var(uuid.UUID(bytes=data.uuid_bytes), type_, name=name)
        read_vars[data.usage].append(var)
        var_order.append(var)
    return read_vars, var_order


def _write_standalone_var(file_obj, var, type_key, version):
    name = var.name.encode(common.ENCODE)
    file_obj.write(
        struct.pack(
            formats.EXPR_VAR_DECLARATION_PACK,
            *formats.EXPR_VAR_DECLARATION(var.var.bytes, type_key, len(name)),
        )
    )
    _write_expr_type(file_obj, var.type, version)
    file_obj.write(name)


def write_standalone_vars(file_obj, circuit, version):
    """Write the standalone variables out from a circuit.

    Args:
        file_obj (File): the file-like object to write to.
        circuit (QuantumCircuit): the circuit to take the variables from.
        version (int): the QPY target version.

    Returns:
        dict[expr.Var | expr.Stretch, int]: a mapping of the variables written to the
            index that they were written at.
    """
    index = 0
    out = {}
    for var in circuit.iter_input_vars():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.INPUT, version)
        out[var] = index
        index += 1
    for var in circuit.iter_captured_vars():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.CAPTURE, version)
        out[var] = index
        index += 1
    for var in circuit.iter_declared_vars():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.LOCAL, version)
        out[var] = index
        index += 1
    if version < 14 and circuit.num_stretches:
        raise exceptions.UnsupportedFeatureForVersion(
            "circuits containing stretch variables", required=14, target=version
        )
    for var in circuit.iter_captured_stretches():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.STRETCH_CAPTURE, version)
        out[var] = index
        index += 1
    for var in circuit.iter_declared_stretches():
        _write_standalone_var(file_obj, var, type_keys.ExprVarDeclaration.STRETCH_LOCAL, version)
        out[var] = index
        index += 1
    return out


def dumps_value(
    obj,
    *,
    version,
    index_map=None,
    use_symengine=False,
    standalone_var_indices=None,
):
    """Serialize input value object.

    Args:
        obj (any): Arbitrary value object to serialize.
        version (int): the target QPY version for the dump.
        index_map (dict): Dictionary with two keys, "q" and "c".  Each key has a value that is a
            dictionary mapping :class:`.Qubit` or :class:`.Clbit` instances (respectively) to their
            integer indices.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_var_indices (dict): Dictionary that maps standalone :class:`.expr.Var` entries to
            the index that should be used to refer to them.

    Returns:
        tuple: TypeKey and binary data.

    Raises:
        QpyError: Serializer for given format is not ready.
    """
    type_key = type_keys.Value.assign(obj)

    if type_key == type_keys.Value.INTEGER:
        binary_data = struct.pack("!q", obj)
    elif type_key == type_keys.Value.FLOAT:
        binary_data = struct.pack("!d", obj)
    elif type_key == type_keys.Value.COMPLEX:
        binary_data = struct.pack(formats.COMPLEX_PACK, obj.real, obj.imag)
    elif type_key == type_keys.Value.NUMPY_OBJ:
        binary_data = common.data_to_binary(obj, np.save)
    elif type_key == type_keys.Value.STRING:
        binary_data = obj.encode(common.ENCODE)
    elif type_key in (type_keys.Value.NULL, type_keys.Value.CASE_DEFAULT):
        binary_data = b""
    elif type_key == type_keys.Value.PARAMETER_VECTOR:
        binary_data = common.data_to_binary(obj, _write_parameter_vec)
    elif type_key == type_keys.Value.PARAMETER:
        binary_data = common.data_to_binary(obj, _write_parameter)
    elif type_key == type_keys.Value.PARAMETER_EXPRESSION:
        binary_data = common.data_to_binary(
            obj, _write_parameter_expression, use_symengine=use_symengine, version=version
        )
    elif type_key == type_keys.Value.EXPRESSION:
        clbit_indices = {} if index_map is None else index_map["c"]
        standalone_var_indices = {} if standalone_var_indices is None else standalone_var_indices
        binary_data = common.data_to_binary(
            obj,
            _write_expr,
            clbit_indices=clbit_indices,
            standalone_var_indices=standalone_var_indices,
            version=version,
        )
    else:
        raise exceptions.QpyError(f"Serialization for {type_key} is not implemented in value I/O.")

    return type_key, binary_data


def write_value(
    file_obj, obj, *, version, index_map=None, use_symengine=False, standalone_var_indices=None
):
    """Write a value to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        obj (any): Value to write.
        version (int): the target QPY version for the dump.
        index_map (dict): Dictionary with two keys, "q" and "c".  Each key has a value that is a
            dictionary mapping :class:`.Qubit` or :class:`.Clbit` instances (respectively) to their
            integer indices.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_var_indices (dict): Dictionary that maps standalone :class:`.expr.Var` entries to
            the index that should be used to refer to them.
    """
    type_key, data = dumps_value(
        obj,
        version=version,
        index_map=index_map,
        use_symengine=use_symengine,
        standalone_var_indices=standalone_var_indices,
    )
    common.write_generic_typed_data(file_obj, type_key, data)


def loads_value(
    type_key,
    binary_data,
    version,
    vectors,
    *,
    clbits=(),
    cregs=None,
    use_symengine=False,
    standalone_vars=(),
):
    """Deserialize input binary data to value object.

    Args:
        type_key (ValueTypeKey): Type enum information.
        binary_data (bytes): Data to deserialize.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.
        clbits (Sequence[Clbit]): Clbits in the current scope.
        cregs (Mapping[str, ClassicalRegister]): Classical registers in the current scope.
        use_symengine (bool): If True, symbolic objects will be de-serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_vars (Sequence[Var]): standalone :class:`.expr.Var` nodes in the order that they
            were declared by the circuit header.
    Returns:
        any: Deserialized value object.

    Raises:
        QpyError: Serializer for given format is not ready.
    """

    if isinstance(type_key, bytes):
        type_key = type_keys.Value(type_key)

    if type_key == type_keys.Value.INTEGER:
        return struct.unpack("!q", binary_data)[0]
    if type_key == type_keys.Value.FLOAT:
        return struct.unpack("!d", binary_data)[0]
    if type_key == type_keys.Value.COMPLEX:
        return complex(*struct.unpack(formats.COMPLEX_PACK, binary_data))
    if type_key == type_keys.Value.NUMPY_OBJ:
        return common.data_from_binary(binary_data, np.load)
    if type_key == type_keys.Value.STRING:
        return binary_data.decode(common.ENCODE)
    if type_key == type_keys.Value.NULL:
        return None
    if type_key == type_keys.Value.CASE_DEFAULT:
        return CASE_DEFAULT
    if type_key == type_keys.Value.PARAMETER_VECTOR:
        return common.data_from_binary(
            binary_data,
            _read_parameter_vec,
            vectors=vectors,
        )
    if type_key == type_keys.Value.PARAMETER:
        return common.data_from_binary(binary_data, _read_parameter)
    if type_key == type_keys.Value.PARAMETER_EXPRESSION:
        if version < 3:
            return common.data_from_binary(binary_data, _read_parameter_expression)
        elif version < 13:
            return common.data_from_binary(
                binary_data,
                _read_parameter_expression_v3,
                vectors=vectors,
                use_symengine=use_symengine,
            )
        else:
            return common.data_from_binary(
                binary_data, _read_parameter_expression_v13, vectors=vectors, version=version
            )
    if type_key == type_keys.Value.EXPRESSION:
        return common.data_from_binary(
            binary_data,
            _read_expr,
            clbits=clbits,
            cregs=cregs or {},
            standalone_vars=standalone_vars,
        )

    raise exceptions.QpyError(f"Serialization for {type_key} is not implemented in value I/O.")


def read_value(
    file_obj,
    version,
    vectors,
    *,
    clbits=(),
    cregs=None,
    use_symengine=False,
    standalone_vars=(),
):
    """Read a value from the file like object.

    Args:
        file_obj (File): A file like object to write data.
        version (int): QPY version.
        vectors (dict): ParameterVector in current scope.
        clbits (Sequence[Clbit]): Clbits in the current scope.
        cregs (Mapping[str, ClassicalRegister]): Classical registers in the current scope.
        use_symengine (bool): If True, symbolic objects will be de-serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        standalone_vars (Sequence[expr.Var]): standalone variables in the order they were defined in
            the QPY payload.

    Returns:
        any: Deserialized value object.
    """
    type_key, data = common.read_generic_typed_data(file_obj)

    return loads_value(
        type_key,
        data,
        version,
        vectors,
        clbits=clbits,
        cregs=cregs,
        use_symengine=use_symengine,
        standalone_vars=standalone_vars,
    )
