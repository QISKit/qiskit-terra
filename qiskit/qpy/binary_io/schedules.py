# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Read schedule and schedule instructions.

This module is kept post pulse-removal to allow reading legacy
payloads containing pulse gates without breaking the load flow.
The purpose of the `_read` and `_load` methods below is just to advance
the file handle while consuming pulse data."""
import json
import struct

from io import BytesIO

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.qpy import formats, common, type_keys
from qiskit.qpy.binary_io import value
from qiskit.qpy.exceptions import QpyError


def _read_channel(file_obj, version) -> None:
    common.read_type_key(file_obj)  # read type_key
    value.read_value(file_obj, version, {})  # read index


def _read_waveform(file_obj, version) -> None:
    header = formats.WAVEFORM._make(
        struct.unpack(
            formats.WAVEFORM_PACK,
            file_obj.read(formats.WAVEFORM_SIZE),
        )
    )
    samples_raw = file_obj.read(header.data_size)
    common.data_from_binary(samples_raw, np.load)  # read samples
    value.read_value(file_obj, version, {})  # read name


def _loads_obj(type_key, binary_data, version, vectors):
    """Wraps `value.loads_value` to deserialize binary data to dictionary
    or list objects which are not supported by `value.loads_value`.
    """
    if type_key == b"D":
        with BytesIO(binary_data) as container:
            return common.read_mapping(
                file_obj=container, deserializer=_loads_obj, version=version, vectors=vectors
            )
    elif type_key == b"l":
        with BytesIO(binary_data) as container:
            return common.read_sequence(
                file_obj=container, deserializer=_loads_obj, version=version, vectors=vectors
            )
    else:
        return value.loads_value(type_key, binary_data, version, vectors)


def _read_kernel(file_obj, version) -> None:
    common.read_mapping(
        file_obj=file_obj,
        deserializer=_loads_obj,
        version=version,
        vectors={},
    )
    value.read_value(file_obj, version, {})  # read name


def _read_discriminator(file_obj, version) -> None:
    # read params
    common.read_mapping(
        file_obj=file_obj,
        deserializer=_loads_obj,
        version=version,
        vectors={},
    )
    value.read_value(file_obj, version, {})  # read name


def _read_symbolic_pulse(file_obj, version) -> None:
    make = formats.SYMBOLIC_PULSE._make
    pack = formats.SYMBOLIC_PULSE_PACK
    size = formats.SYMBOLIC_PULSE_SIZE

    header = make(
        struct.unpack(
            pack,
            file_obj.read(size),
        )
    )
    pulse_type = file_obj.read(header.type_size).decode(common.ENCODE)
    file_obj.read(header.envelope_size)  # read envelope
    file_obj.read(header.constraints_size)  # read constraints
    file_obj.read(header.valid_amp_conditions_size)  # read valid amp conditions
    # read parameters
    common.read_mapping(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )

    # In the transition to Qiskit Terra 0.23 (QPY version 6), the representation of library pulses
    # was changed from complex "amp" to float "amp" and "angle". The existing library pulses in
    # previous versions are handled here separately to conform with the new representation. To
    # avoid role assumption for "amp" for custom pulses, only the library pulses are handled this
    # way.

    # List of pulses in the library in QPY version 5 and below:
    legacy_library_pulses = ["Gaussian", "GaussianSquare", "Drag", "Constant"]
    class_name = "SymbolicPulse"  # Default class name, if not in the library

    if pulse_type in legacy_library_pulses:
        class_name = "ScalableSymbolicPulse"

    value.read_value(file_obj, version, {})  # read duration
    value.read_value(file_obj, version, {})  # read name

    if class_name not in {"SymbolicPulse", "ScalableSymbolicPulse"}:
        raise NotImplementedError(f"Unknown class '{class_name}'")


def _read_symbolic_pulse_v6(file_obj, version) -> None:
    make = formats.SYMBOLIC_PULSE_V2._make
    pack = formats.SYMBOLIC_PULSE_PACK_V2
    size = formats.SYMBOLIC_PULSE_SIZE_V2

    header = make(
        struct.unpack(
            pack,
            file_obj.read(size),
        )
    )
    class_name = file_obj.read(header.class_name_size).decode(common.ENCODE)
    file_obj.read(header.type_size).decode(common.ENCODE)  # read pulse type
    file_obj.read(header.envelope_size)  # read envelope
    file_obj.read(header.constraints_size)  # read constraints
    file_obj.read(header.valid_amp_conditions_size)  # read valid_amp_conditions
    # read parameters
    common.read_mapping(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )

    value.read_value(file_obj, version, {})  # read duration
    value.read_value(file_obj, version, {})  # read name

    if class_name not in {"SymbolicPulse", "ScalableSymbolicPulse"}:
        raise NotImplementedError(f"Unknown class '{class_name}'")


def _read_alignment_context(file_obj, version) -> None:
    common.read_type_key(file_obj)

    common.read_sequence(
        file_obj,
        deserializer=value.loads_value,
        version=version,
        vectors={},
    )


def _loads_operand(type_key, data_bytes, version):
    if type_key == type_keys.ScheduleOperand.WAVEFORM:
        return common.data_from_binary(data_bytes, _read_waveform, version=version)
    if type_key == type_keys.ScheduleOperand.SYMBOLIC_PULSE:
        if version < 6:
            return common.data_from_binary(data_bytes, _read_symbolic_pulse, version=version)
        else:
            return common.data_from_binary(data_bytes, _read_symbolic_pulse_v6, version=version)
    if type_key == type_keys.ScheduleOperand.CHANNEL:
        return common.data_from_binary(data_bytes, _read_channel, version=version)
    if type_key == type_keys.ScheduleOperand.OPERAND_STR:
        return data_bytes.decode(common.ENCODE)
    if type_key == type_keys.ScheduleOperand.KERNEL:
        return common.data_from_binary(
            data_bytes,
            _read_kernel,
            version=version,
        )
    if type_key == type_keys.ScheduleOperand.DISCRIMINATOR:
        return common.data_from_binary(
            data_bytes,
            _read_discriminator,
            version=version,
        )

    return value.loads_value(type_key, data_bytes, version, {})


def _read_element(file_obj, version, metadata_deserializer, use_symengine) -> None:
    type_key = common.read_type_key(file_obj)

    if type_key == type_keys.Program.SCHEDULE_BLOCK:
        return read_schedule_block(file_obj, version, metadata_deserializer, use_symengine)

    # read operands
    common.read_sequence(
        file_obj,
        deserializer=_loads_operand,
        version=version,
    )
    # read name
    value.read_value(file_obj, version, {})

    return None


def _loads_reference_item(type_key, data_bytes, metadata_deserializer, version) -> None:
    if type_key == type_keys.Value.NULL:
        return None
    if type_key == type_keys.Program.SCHEDULE_BLOCK:
        return common.data_from_binary(
            data_bytes,
            deserializer=read_schedule_block,
            version=version,
            metadata_deserializer=metadata_deserializer,
        )

    raise QpyError(
        f"Loaded schedule reference item is neither None nor ScheduleBlock. "
        f"Type key {type_key} is not valid data type for a reference items. "
        "This data cannot be loaded. Please check QPY version."
    )


def read_schedule_block(file_obj, version, metadata_deserializer=None, use_symengine=False):
    """Consume a single ScheduleBlock from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        version (int): QPY version.
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.ScheduleBlock.metadata` attribute for a schedule block
            in the file-like object. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
    Returns:
        QuantumCircuit: Returns a dummy QuantumCircuit object, containing just name and metadata.
        This function exists just to allow reading legacy payloads containing pulse information
        without breaking the entire load flow.

    Raises:
        TypeError: If any of the instructions is invalid data format.
        QiskitError: QPY version is earlier than block support.
    """
    if version < 5:
        raise QiskitError(f"QPY version {version} does not support ScheduleBlock.")

    data = formats.SCHEDULE_BLOCK_HEADER._make(
        struct.unpack(
            formats.SCHEDULE_BLOCK_HEADER_PACK,
            file_obj.read(formats.SCHEDULE_BLOCK_HEADER_SIZE),
        )
    )
    file_obj.read(data.name_size).decode(common.ENCODE)  # read name
    metadata_raw = file_obj.read(data.metadata_size)
    json.loads(metadata_raw, cls=metadata_deserializer)  # read metadata
    _read_alignment_context(file_obj, version)

    for _ in range(data.num_elements):
        _read_element(file_obj, version, metadata_deserializer, use_symengine)

    # Load references
    if version >= 7:
        common.read_mapping(
            file_obj=file_obj,
            deserializer=_loads_reference_item,
            version=version,
            metadata_deserializer=metadata_deserializer,
        )
