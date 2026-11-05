# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Common functions across several serialization and deserialization modules.
"""

import io
import struct
import uuid

from qiskit.utils.optionals import HAS_SYMENGINE

from qiskit.qpy import formats, exceptions

QPY_VERSION = 18
QPY_COMPATIBILITY_VERSION = 13
QPY_RUST_READ_MIN_VERSION = 13
QPY_RUST_WRITE_MIN_VERSION = 17
ENCODE = "utf8"


def read_generic_typed_data(file_obj):
    """Read a single data chunk from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.

    Returns:
        tuple: Tuple of type key binary and the bytes object of the single data.
    """
    data = formats.INSTRUCTION_PARAM._make(
        struct.unpack(formats.INSTRUCTION_PARAM_PACK, file_obj.read(formats.INSTRUCTION_PARAM_SIZE))
    )

    return data.type, file_obj.read(data.size)


def read_sequence(file_obj, deserializer, **kwargs):
    """Read a sequence of data from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        deserializer (Callable): Deserializer callback that can handle input object type.
            This must take type key and binary data of the element and return object.
        kwargs: Options set to the deserializer.

    Returns:
        list: Deserialized object.
    """
    sequence = []

    data = formats.SEQUENCE._make(
        struct.unpack(formats.SEQUENCE_PACK, file_obj.read(formats.SEQUENCE_SIZE))
    )
    for _ in range(data.num_elements):
        type_key, datum_bytes = read_generic_typed_data(file_obj)
        sequence.append(deserializer(type_key, datum_bytes, **kwargs))

    return sequence


def read_mapping(file_obj, deserializer, **kwargs):
    """Read a mapping from the file like object.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        deserializer (Callable): Deserializer callback that can handle mapping item.
            This must take type key and binary data of the mapping value and return object.
        kwargs: Options set to the deserializer.

    Returns:
        dict: Deserialized object.
    """
    mapping = {}

    data = formats.SEQUENCE._make(
        struct.unpack(formats.SEQUENCE_PACK, file_obj.read(formats.SEQUENCE_SIZE))
    )
    for _ in range(data.num_elements):
        map_header = formats.MAP_ITEM._make(
            struct.unpack(formats.MAP_ITEM_PACK, file_obj.read(formats.MAP_ITEM_SIZE))
        )
        if kwargs.get("version", 15) < 15:
            key = file_obj.read(map_header.key_size).decode(ENCODE)
        else:
            key = uuid.UUID(bytes=file_obj.read(map_header.key_size))

        datum = deserializer(map_header.type, file_obj.read(map_header.size), **kwargs)
        mapping[key] = datum

    return mapping


def read_type_key(file_obj):
    """Read a type key from the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.

    Returns:
        bytes: Type key.
    """
    return formats.TYPE_KEY._make(
        struct.unpack(formats.TYPE_KEY_PACK, file_obj.read(formats.TYPE_KEY_SIZE))
    ).key


def write_generic_typed_data(file_obj, type_key, data_binary):
    """Write statically typed binary data to the file like object.

    Args:
        file_obj (File): A file like object to write data.
        type_key (Enum): Object type of the data.
        data_binary (bytes): Binary data to write.
    """
    data_header = struct.pack(formats.INSTRUCTION_PARAM_PACK, type_key, len(data_binary))
    file_obj.write(data_header)
    file_obj.write(data_binary)


def write_sequence(file_obj, sequence, serializer, **kwargs):
    """Write a sequence of data in the file like object.

    Args:
        file_obj (File): A file like object to write data.
        sequence (Sequence): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
            This must return type key and binary data of each element.
        kwargs: Options set to the serializer.
    """
    num_elements = len(sequence)

    file_obj.write(struct.pack(formats.SEQUENCE_PACK, num_elements))
    for datum in sequence:
        type_key, datum_bytes = serializer(datum, **kwargs)
        write_generic_typed_data(file_obj, type_key, datum_bytes)


def write_mapping(file_obj, mapping, serializer, **kwargs):
    """Write a mapping in the file like object.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        file_obj (File): A file like object to write data.
        mapping (Mapping): Object to serialize.
        serializer (Callable): Serializer callback that can handle mapping item.
            This must return type key and binary data of the mapping value.
        kwargs: Options set to the serializer.
    """
    num_elements = len(mapping)

    file_obj.write(struct.pack(formats.SEQUENCE_PACK, num_elements))
    for key, datum in mapping.items():
        if kwargs.get("version", 15) < 15:
            key_bytes = key.encode(ENCODE)
        else:
            key_bytes = key
        type_key, datum_bytes = serializer(datum, **kwargs)
        item_header = struct.pack(formats.MAP_ITEM_PACK, len(key_bytes), type_key, len(datum_bytes))
        file_obj.write(item_header)
        file_obj.write(key_bytes)
        file_obj.write(datum_bytes)


def write_type_key(file_obj, type_key):
    """Write a type key in the file like object.

    Args:
        file_obj (File): A file like object that contains the QPY binary data.
        type_key (bytes): Type key to write.
    """
    file_obj.write(struct.pack(formats.TYPE_KEY_PACK, type_key))


def data_to_binary(obj, serializer, **kwargs):
    """Convert object into binary data with specified serializer.

    Args:
        obj (any): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        serializer(container, obj, **kwargs)
        binary_data = container.getvalue()

    return binary_data


def sequence_to_binary(sequence, serializer, **kwargs):
    """Convert sequence into binary data with specified serializer.

    Args:
        sequence (Sequence): Object to serialize.
        serializer (Callable): Serializer callback that can handle input object type.
            This must return type key and binary data of each element.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        write_sequence(container, sequence, serializer, **kwargs)
        binary_data = container.getvalue()

    return binary_data


def mapping_to_binary(mapping, serializer, **kwargs):
    """Convert mapping into binary data with specified serializer.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        mapping (Mapping): Object to serialize.
        serializer (Callable): Serializer callback that can handle mapping item.
            This must return type key and binary data of the mapping value.
        kwargs: Options set to the serializer.

    Returns:
        bytes: Binary data.
    """
    with io.BytesIO() as container:
        write_mapping(container, mapping, serializer, **kwargs)
        binary_data = container.getvalue()

    return binary_data


def data_from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary data with specified deserializer.

    Args:
        binary_data (bytes): Binary data to deserialize.
        deserializer (Callable): Deserializer callback that can handle input object type.
        kwargs: Options set to the deserializer.

    Returns:
        any: Deserialized object.
    """
    with io.BytesIO(binary_data) as container:
        container.seek(0)
        obj = deserializer(container, **kwargs)
    return obj


def sequence_from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary sequence with specified deserializer.

    Args:
        binary_data (bytes): Binary data to deserialize.
        deserializer (Callable): Deserializer callback that can handle input object type.
            This must take type key and binary data of the element and return object.
        kwargs: Options set to the deserializer.

    Returns:
        any: Deserialized sequence.
    """
    with io.BytesIO(binary_data) as container:
        sequence = read_sequence(container, deserializer, **kwargs)

    return sequence


def mapping_from_binary(binary_data, deserializer, **kwargs):
    """Load object from binary mapping with specified deserializer.

    .. note::

        This function must be used to make a binary data of mapping
        which include QPY serialized values.
        It's easier to use JSON serializer followed by encoding for standard data formats.
        This only supports flat dictionary and key must be string.

    Args:
        binary_data (bytes): Binary data to deserialize.
        deserializer (Callable): Deserializer callback that can handle mapping item.
            This must take type key and binary data of the mapping value and return object.
        kwargs: Options set to the deserializer.

    Returns:
        dict: Deserialized object.
    """
    with io.BytesIO(binary_data) as container:
        mapping = read_mapping(container, deserializer, **kwargs)

    return mapping


@HAS_SYMENGINE.require_in_call("QPY versions 10 through 12 with symengine parameter serialization")
def load_symengine_payload(payload: bytes):
    """Load a symengine expression from it's serialized cereal payload."""
    # This is a horrible hack to workaround the symengine version checking
    # it's deserialization does. There were no changes to the serialization
    # format between 0.11 and 0.13 but the deserializer checks that it can't
    # load across a major or minor version boundary. This works around it
    # by just lying about the generating version.
    import symengine
    from symengine.lib.symengine_wrapper import (  # pylint: disable = no-name-in-module
        load_basic,
    )

    symengine_version = symengine.__version__.split(".")
    major = payload[2]
    minor = payload[3]
    if int(symengine_version[1]) != minor:
        if major != 0:
            raise exceptions.QpyError(
                "Qiskit doesn't support loading a symengine payload generated with symengine >= 1.0"
            )
        if minor == 9:
            raise exceptions.QpyError(
                "Qiskit doesn't support loading a historical QPY file with `use_symengine=True` "
                "generated in an environment using symengine 0.9.0. If you need to load this file "
                "you can do so with Qiskit 0.45.x or 0.46.x and re-export the QPY file using "
                "`use_symengine=False`."
            )
        if minor not in (11, 13):
            raise exceptions.QpyError(
                f"Incompatible symengine version {major}.{minor} used to generate the QPY "
                "payload"
            )
        minor_version = int(symengine_version[1])
        if minor_version not in (11, 13):
            raise exceptions.QpyError(
                f"Incompatible installed symengine version {symengine.__version__} to load "
                "this QPY payload"
            )
        payload = bytearray(payload)
        payload[3] = minor_version
        payload = bytes(payload)
    return load_basic(payload)
