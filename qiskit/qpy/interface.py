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

"""User interface of qpy serializer."""

from __future__ import annotations

import gzip
import io
import shutil
from json import JSONEncoder, JSONDecoder
from typing import BinaryIO, TYPE_CHECKING
from collections.abc import Callable
from collections.abc import Iterable, Mapping
import struct
import warnings
import re

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.qpy import formats, common, binary_io, type_keys
from qiskit.qpy.exceptions import QpyError
from qiskit import user_config
from qiskit.version import __version__
from qiskit._accelerate import qpy as _qpy

if TYPE_CHECKING:
    from qiskit.circuit import annotation


QPY_SUPPORTED_TYPES = QuantumCircuit

# Some standard-library types claim to be `IOBase.seekable`, but don't actually support arbitrary
# seeking in write mode.  We won't expand this list for incorrect third-party types, but
# pragmatically, we can workaround trouble in the stdlib.
#
# See https://github.com/Qiskit/qiskit/issues/15157#issuecomment-3389209015 for more detail.
KNOWN_BAD_SEEKERS = (gzip.GzipFile,)

# This version pattern is taken from the pypa packaging project:
# https://github.com/pypa/packaging/blob/21.3/packaging/version.py#L223-L254
# which is dual licensed Apache 2.0 and BSD see the source for the original
# authors and other details
VERSION_PATTERN = (
    "^"
    + r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""
    + "$"
)
VERSION_PATTERN_REGEX = re.compile(VERSION_PATTERN, re.VERBOSE | re.IGNORECASE)


def dump(
    programs: list[QPY_SUPPORTED_TYPES] | QPY_SUPPORTED_TYPES,
    file_obj: BinaryIO,
    metadata_serializer: type[JSONEncoder] | None = None,
    use_symengine: bool = False,
    version: int = common.QPY_VERSION,
    annotation_factories: Mapping[str, Callable[[], annotation.QPYSerializer]] | None = None,
):
    """Write QPY binary data to a file

    This function is used to save a circuit to a file for later use or transfer
    between machines. The QPY format is backwards compatible and can be
    loaded with future versions of Qiskit.

    For example:

    .. plot::
       :include-source:
       :nofigs:
       :context: reset

        from qiskit.circuit import QuantumCircuit
        from qiskit import qpy

        qc = QuantumCircuit(2, name='Bell', metadata={'test': True})
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

    from this you can write the qpy data to a file:

    .. code-block:: python

        with open('bell.qpy', 'wb') as fd:
            qpy.dump(qc, fd)

    or a gzip compressed file:

    .. code-block:: python

        import gzip

        with gzip.open('bell.qpy.gz', 'wb') as fd:
            qpy.dump(qc, fd)

    Which will save the qpy serialized circuit to the provided file.

    Args:
        programs: QPY supported object(s) to store in the specified file like object.
            QPY supports :class:`.QuantumCircuit`.
        file_obj: The file like object to write the QPY data too
        metadata_serializer: An optional JSONEncoder class that
            will be passed the ``.metadata`` attribute for each program in ``programs`` and will be
            used as the ``cls`` kwarg on the `json.dump()`` call to JSON serialize that dictionary.
        use_symengine: This flag is no longer used by QPY versions supported by this function and
            will have no impact on the generated QPY payload except to set a field in a QPY v13 file
            header which is unused.
        version: The QPY format version to emit. By default this defaults to
            the latest supported format of :attr:`~.qpy.QPY_VERSION`, however for
            compatibility reasons if you need to load the generated QPY payload with an older
            version of Qiskit you can also select an older QPY format version down to the minimum
            supported export version, which only can change during a Qiskit major version release,
            to generate an older QPY format version.  You can access the current QPY version and
            minimum compatible version with :attr:`.qpy.QPY_VERSION` and
            :attr:`.qpy.QPY_COMPATIBILITY_VERSION` respectively.

            .. note::

                If specified with an older version of QPY the limitations and potential bugs stemming
                from the QPY format at that version will persist. This should only be used if
                compatibility with loading the payload with an older version of Qiskit is necessary.

        annotation_factories: Mapping of namespaces to functions that create new instances of
            :class:`.annotation.QPUSerializer`, for handling the dumping of custom
            :class:`.Annotation` objects.  The subsequent call to :func:`load` will need to use
            similar serializer objects, that understand the custom output format of those
            serializers.

    Raises:
        TypeError: When invalid data type is input.
        ValueError: When an unsupported version number is passed in for the ``version`` argument.
    """
    if not isinstance(programs, Iterable):
        programs = [programs]

    # dump accepts only QuantumCircuit typed objects
    for program in programs:
        if not issubclass(type(program), QuantumCircuit):
            raise TypeError(f"'{type(program)}' is not a supported data type.")

    if version is None:
        version = common.QPY_VERSION
    elif common.QPY_COMPATIBILITY_VERSION > version or version > common.QPY_VERSION:
        raise ValueError(
            f"Dumping payloads with the specified QPY version ({version}) is not supported by "
            f"this version of Qiskit. Try selecting a version between "
            f"{common.QPY_COMPATIBILITY_VERSION} and {common.QPY_VERSION} for `qpy.dump`."
        )

    use_rust = version >= common.QPY_RUST_WRITE_MIN_VERSION
    if use_rust:
        _qpy.dump(
            programs,
            file_obj,
            metadata_serializer,
            version,
            annotation_factories,
        )
        return

    version_match = VERSION_PATTERN_REGEX.search(__version__)
    version_parts = [int(x) for x in version_match.group("release").split(".")]
    encoding = type_keys.SymExprEncoding.assign(use_symengine)
    header = struct.pack(
        formats.FILE_HEADER_V10_PACK,
        b"QISKIT",
        version,
        version_parts[0],
        version_parts[1],
        version_parts[2],
        len(programs),
        encoding,
    )
    file_obj.write(header)
    common.write_type_key(file_obj, type_keys.Program.CIRCUIT)
    header_bytes_written = formats.FILE_HEADER_V10_SIZE + formats.TYPE_KEY_SIZE

    def _write_circuit(out_stream, circuit):
        binary_io.write_circuit(
            out_stream,
            circuit,
            metadata_serializer=metadata_serializer,
            use_symengine=bool(use_symengine),
            version=version,
            annotation_factories=annotation_factories,
            use_rust=use_rust,
        )

    if version >= 16:
        # We need a circuit table.
        if file_obj.seekable() and not isinstance(file_obj, KNOWN_BAD_SEEKERS):
            # Fast path for properly seekable streams
            file_offsets = []
            table_start = file_obj.tell()
            # Skip past the circuit table to write circuit contents first.
            file_obj.seek(len(programs) * formats.CIRCUIT_TABLE_ENTRY_SIZE, 1)
            for program in programs:
                file_offsets.append(file_obj.tell())
                _write_circuit(file_obj, program)
            # Seek back to the table start and write it out.
            file_obj.seek(table_start)
            for offset in file_offsets:
                file_obj.write(
                    struct.pack(
                        formats.CIRCUIT_TABLE_ENTRY_PACK, *formats.CIRCUIT_TABLE_ENTRY(offset)
                    )
                )
            # Seek to the end of the stream.
            file_obj.seek(0, 2)
        else:
            # We need to create a temporary BytesIO buffer since the input
            # stream isn't seekable.
            buffer_offsets = []
            offset_table_size = len(programs) * formats.CIRCUIT_TABLE_ENTRY_SIZE
            with io.BytesIO() as circuits_buffer:
                for program in programs:
                    buffer_offsets.append(circuits_buffer.tell())
                    _write_circuit(circuits_buffer, program)
                # Write circuit table to input stream, adjusting offsets.
                for offset in buffer_offsets:
                    file_obj.write(
                        struct.pack(
                            formats.CIRCUIT_TABLE_ENTRY_PACK,
                            *formats.CIRCUIT_TABLE_ENTRY(
                                header_bytes_written + offset_table_size + offset
                            ),
                        )
                    )
                # Write circuits to the input stream.
                circuits_buffer.seek(0)
                shutil.copyfileobj(circuits_buffer, file_obj)
    else:
        # No circuit table needed, just write the circuits sequentially.
        for program in programs:
            _write_circuit(file_obj, program)


def load(
    file_obj: BinaryIO,
    metadata_deserializer: type[JSONDecoder] | None = None,
    annotation_factories: Mapping[str, Callable[[], annotation.QPYSerializer]] | None = None,
) -> list[QPY_SUPPORTED_TYPES]:
    """Load a QPY binary file

    This function is used to load a serialized QPY Qiskit program file and create
    :class:`~qiskit.circuit.QuantumCircuit` objects from its contents.
    For example:

    .. code-block:: python

        from qiskit import qpy

        with open('bell.qpy', 'rb') as fd:
            circuits = qpy.load(fd)

    or with a gzip compressed file:

    .. code-block:: python

        import gzip
        from qiskit import qpy

        with gzip.open('bell.qpy.gz', 'rb') as fd:
            circuits = qpy.load(fd)

    which will read the contents of the qpy and return a list of
    :class:`~qiskit.circuit.QuantumCircuit` objects from the file.

    Args:
        file_obj: A file like object that contains the QPY binary
            data for a circuit.
        metadata_deserializer: An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the ``.metadata`` attribute for any programs in the QPY file.
            If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
        annotation_factories: Mapping of namespaces to functions that create new instances of
            :class:`.annotation.QPUSerializer`, for handling the loading of custom
            :class:`.Annotation` objects.

    Returns:
        The list of Qiskit programs contained in the QPY data.
        A list is always returned, even if there is only 1 program in the QPY data.

    Raises:
        QiskitError: if ``file_obj`` is not a valid QPY file
        TypeError: When invalid data type is loaded.
        MissingOptionalLibraryError: If the ``symengine`` engine library is
            not installed when loading a QPY version 10, 11, or 12 payload
            that is using symengine symbolic encoding and contains
            :class:`.ParameterExpression` instances.
        QpyError: if known but unsupported data type is loaded.
    """

    # identify file header version
    version = struct.unpack("!6sB", file_obj.read(7))[1]
    file_obj.seek(0)

    if version > common.QPY_VERSION:
        raise QiskitError(
            f"The QPY format version being read, {version}, isn't supported by "
            "this Qiskit version. Please upgrade your version of Qiskit to load this QPY payload"
        )
    config = user_config.get_config()
    min_qpy_version = config.get("min_qpy_version")
    if min_qpy_version is not None and version < min_qpy_version:
        raise QpyError(
            f"QPY version {version} is lower than the configured minimum "
            f"version {min_qpy_version}."
        )
    use_rust = version >= common.QPY_RUST_READ_MIN_VERSION
    if use_rust:
        return _qpy.load(file_obj, metadata_deserializer, annotation_factories)

    if version < 10:
        data = formats.FILE_HEADER._make(
            struct.unpack(
                formats.FILE_HEADER_PACK,
                file_obj.read(formats.FILE_HEADER_SIZE),
            )
        )
    else:
        data = formats.FILE_HEADER_V10._make(
            struct.unpack(
                formats.FILE_HEADER_V10_PACK,
                file_obj.read(formats.FILE_HEADER_V10_SIZE),
            )
        )

    if data.preface.decode(common.ENCODE) != "QISKIT":
        raise QiskitError("Input file is not a valid QPY file")
    version_match = VERSION_PATTERN_REGEX.search(__version__)
    env_qiskit_version = [int(x) for x in version_match.group("release").split(".")]

    qiskit_version = (data.major_version, data.minor_version, data.patch_version)

    if (
        env_qiskit_version[0] < qiskit_version[0]
        or (
            env_qiskit_version[0] == qiskit_version[0] and qiskit_version[1] > env_qiskit_version[1]
        )
        or (
            env_qiskit_version[0] == qiskit_version[0]
            and qiskit_version[1] == env_qiskit_version[1]
            and qiskit_version[2] > env_qiskit_version[2]
        )
    ):
        warnings.warn(
            "The qiskit version used to generate the provided QPY "
            f"file, {'.'.join([str(x) for x in qiskit_version])}, "
            f"is newer than the current qiskit version {__version__}. "
            "This may result in an error if the QPY file uses "
            "instructions not present in this current qiskit "
            "version"
        )

    if data.qpy_version < 5:
        type_key = type_keys.Program.CIRCUIT
    else:
        type_key = common.read_type_key(file_obj)

    if type_key == type_keys.Program.SCHEDULE_BLOCK:
        raise QpyError(
            "Payloads of type `ScheduleBlock` cannot be loaded as of Qiskit 2.0. "
            "Use an earlier version of Qiskit if you want to load `ScheduleBlock` payloads."
        )
    if type_key != type_keys.Program.CIRCUIT:
        raise TypeError(f"Invalid payload format data kind '{type_key}'.")

    if data.qpy_version < 10:
        use_symengine = False
    else:
        use_symengine = data.symbolic_encoding == type_keys.SymExprEncoding.SYMENGINE

    if data.qpy_version >= 16:
        # Obtain the byte offsets for each program
        program_offsets = []
        for _ in range(data.num_programs):
            program_offsets.append(
                formats.CIRCUIT_TABLE_ENTRY(
                    *struct.unpack(
                        formats.CIRCUIT_TABLE_ENTRY_PACK,
                        file_obj.read(formats.CIRCUIT_TABLE_ENTRY_SIZE),
                    )
                ).offset
            )

    programs = []
    for i in range(data.num_programs):
        if data.qpy_version >= 16:
            # Deserialize each program using their byte offsets
            file_obj.seek(program_offsets[i])
        programs.append(
            binary_io.read_circuit(
                file_obj,
                data.qpy_version,
                metadata_deserializer=metadata_deserializer,
                use_symengine=bool(use_symengine),
                annotation_factories=annotation_factories,
                use_rust=use_rust,
            )
        )
    return programs


def get_qpy_version(
    file_obj: BinaryIO,
) -> int:
    """This function identifies the QPY version of the file.

    This function will read the header of ``file_obj`` and will
    return the QPY format version. It will **not** advance the
    cursor of ``file_obj``. If you are using this for a subsequent
    read, such as to call :func:`.load`, you can pass ``file_obj``
    directly. For example::

        from qiskit import qpy

        qpy_version = qpy.get_qpy_version(qpy_file)
        if qpy_version > 12:
            qpy.load(qpy_file)

    Args:
        file_obj: A file like object that contains the QPY binary
            data for a circuit.

    Returns:
        The QPY version of the specified file.
    """

    version = struct.unpack("!6sB", file_obj.read(7))[1]
    file_obj.seek(-7, 1)
    return version
