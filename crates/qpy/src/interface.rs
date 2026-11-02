// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// QPY interface module providing high-level dump/load functions
//
// This module provides the main entry points for serializing and deserializing
// quantum circuits to/from QPY format. It handles the complete file structure
// including headers, circuit tables, and multiple circuits.

use binrw::{BinRead, Endian, VecArgs};
use pyo3::PyResult;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::converters::QuantumCircuitData;

use crate::annotations::{AnnotationHandler, NativeDeserializers, NativeSerializers};
use crate::bytes::Bytes;
use crate::circuit_reader::unpack_circuit;
use crate::circuit_writer::{pack_circuit, pack_layout};
use crate::error::QpyError;
use crate::formats::{LayoutV2Pack, QPYCircuit, QPYFileHeader};
use crate::py_methods::{py_circuit_data_to_quantum_circuit, serialize_metadata};
use crate::value::{
    ProgramType, QpyCaller, SymbolicEncoding, deserialize, deserialize_with_args, serialize,
    serialize_with_args,
};
use crate::{QPY_READ_MIN_VERSION, QPY_WRITE_MIN_VERSION};

use std::io::{Cursor, Seek};

/// A circuit loaded from QPY, before the Python-only parts of circuit construction.
///
/// The packed circuit is retained because it contains the metadata, name, and layout that are
/// applied when converting the native [`CircuitData`] into a Python `QuantumCircuit`.
pub struct LoadedCircuit {
    /// The native circuit data reconstructed from the payload.
    pub circuit_data: CircuitData,
    /// The packed representation containing data needed for Python circuit construction.
    pub packed_circuit: QPYCircuit,
}

/// Data associated with a circuit that is not stored in native [`CircuitData`].
pub struct ExtraCircuitData {
    /// The circuit name.
    pub name: Option<String>,
    /// The serialized Python metadata mapping.
    pub metadata: Bytes,
    /// The transpiler layout data.
    pub layout: Option<LayoutV2Pack>,
}

/// Parses an ASCII decimal string as a `u8` during compile-time evaluation.
///
/// This is used for Cargo's package-version components, which are guaranteed to
/// contain decimal digits.
const fn parse_u8_from_ascii(s: &str) -> u8 {
    let bytes = s.as_bytes();
    let mut result: u8 = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] < b'0' || bytes[i] > b'9' {
            panic!("Invalid character in version string");
        }
        let digit = bytes[i] - b'0';
        result = result * 10 + digit;
        i += 1;
    }
    result
}

/// Returns this crate's Cargo package version as three numeric components.
const fn parse_version() -> (u8, u8, u8) {
    let major = parse_u8_from_ascii(env!("CARGO_PKG_VERSION_MAJOR"));
    let minor = parse_u8_from_ascii(env!("CARGO_PKG_VERSION_MINOR"));
    let patch = parse_u8_from_ascii(env!("CARGO_PKG_VERSION_PATCH"));
    (major, minor, patch)
}

const QISKIT_VERSION: (u8, u8, u8) = parse_version();
const QPY_VERSION: u8 = 18;

/// Serializes native circuits into a complete binary QPY payload.
/// # Arguments
///
/// * circuits: The rust `CircuitData` of the circuits to serialize. Mutability is required because some implicit
///   gates have to be instantiated in order to be serialized.
/// * extra_data: The extra data associated with each circuit, including name, metadata, and layout. Since this
///   is currently Python-only, metadata and layout should be already serialized into `Bytes` objects.
/// * qpy_version: The QPY version to use for serialization. Must be >= QPY_WRITE_MIN_VERSION.
/// * annotation_handler: The annotation handler to use for serializing annotations. If None, the native handler is used.
///
/// Returns:
/// A `Bytes` object containing the complete QPY payload.
pub fn dump_qpy<'a, C>(
    circuits: C,
    qpy_version: u8,
    annotation_handler: Option<AnnotationHandler>,
    caller: Option<QpyCaller>,
) -> Result<Bytes, QpyError>
where
    C: ExactSizeIterator<Item = Result<(&'a CircuitData, ExtraCircuitData), QpyError>>,
{
    if qpy_version < QPY_WRITE_MIN_VERSION {
        Err(QpyError::UnsupportedFeatureForVersion {
            feature: "Rust QPY".to_string(),
            version: qpy_version,
            min_version: QPY_WRITE_MIN_VERSION,
        })?;
    }
    let caller = caller.unwrap_or(QpyCaller::Native);
    let annotation_handler = annotation_handler.unwrap_or(AnnotationHandler::native(
        Vec::new(),
        NativeSerializers::default(),
        NativeDeserializers::default(),
    ));
    let serialized_circuits: Vec<Bytes> = circuits
        .map(|data| {
            let (circuit, extra) = data?;
            serialize_with_args::<QPYCircuit, (u8,)>(
                &pack_circuit(
                    circuit,
                    extra,
                    qpy_version,
                    annotation_handler.child()?,
                    caller,
                )?,
                (qpy_version,),
            )
        })
        .collect::<Result<Vec<Bytes>, QpyError>>()?;
    // Since QPY doesn't use symengine anymore, we default to SymbolicEncoding::Sympy
    let qpy_header = QPYFileHeader {
        qpy_version,
        qiskit_version: QISKIT_VERSION,
        num_programs: serialized_circuits.len() as u64,
        symbolic_encoding: SymbolicEncoding::Sympy,
        type_key: ProgramType::Circuit, //for now, no other value type is used
    };
    let serialized_qpy_header = serialize(&qpy_header)?;

    // At this point we have collected all the relevant data
    // But still need to create the offset table and put everything together
    let header_size = serialized_qpy_header.len();
    let offset_table_size = serialized_circuits.len() * 8; // 8 bytes per u64
    let circuits_start_offset = header_size + offset_table_size;
    // Build the offset table
    let mut offset_table: Vec<u64> = Vec::with_capacity(serialized_circuits.len());
    let mut current_offset = circuits_start_offset as u64;

    for circuit_bytes in &serialized_circuits {
        offset_table.push(current_offset);
        current_offset += circuit_bytes.len() as u64;
    }

    let mut output = Vec::<u8>::with_capacity(current_offset as usize);

    output.extend_from_slice(&serialized_qpy_header);
    for offset in offset_table {
        output.extend_from_slice(&offset.to_be_bytes());
    }
    for circuit_bytes in serialized_circuits {
        output.extend_from_slice(&circuit_bytes);
    }

    Ok(Bytes::from(output))
}

#[pyfunction]
#[pyo3(name = "dump")]
/// Python entry point for serializing circuits to binary and writing to file.
///
/// # Arguments
///
/// * py: The GIL handle.
/// * programs: The list of QuantumCircuit objects to serialize.
/// * file_obj: A writable file-like object to which the serialized QPY data will be written.
/// * metadata_serializer: An optional Python callable that takes a metadata dictionary and returns a serialized bytes object.
/// * version: The QPY version to use for serialization. Must be >= QPY_WRITE_MIN_VERSION.
/// * annotation_factories: An optional dictionary for annotation handlers
pub fn py_dump_qpy(
    py: Python,
    programs: &Bound<PyAny>,
    file_obj: &Bound<PyAny>,
    metadata_serializer: Option<Bound<PyAny>>,
    version: u8,
    annotation_factories: Option<Bound<PyDict>>,
) -> PyResult<()> {
    let annotation_factories = annotation_factories.unwrap_or(PyDict::new(py));
    let annotation_handler = AnnotationHandler::python(&annotation_factories.clone().unbind())?;
    let circuits: Vec<QuantumCircuitData> = programs.extract()?;
    let data = circuits.iter().map(|circuit| {
        let metadata = serialize_metadata(&circuit.metadata, metadata_serializer.as_ref())?;
        let layout = pack_layout(circuit.transpile_layout.clone(), &circuit.data, version)?;
        Ok::<_, QpyError>((
            &circuit.data,
            ExtraCircuitData {
                name: circuit.name.clone(),
                metadata,
                layout: Some(layout),
            },
        ))
    });
    let serialized_qpy = dump_qpy(
        data,
        version,
        Some(annotation_handler),
        Some(QpyCaller::Python),
    )?;
    file_obj.call_method1("write", (pyo3::types::PyBytes::new(py, &serialized_qpy),))?;
    Ok(())
}

/// Reads a QPY circuit-offset table and splits the remaining payload into circuits.
///
/// The cursor must be positioned immediately before the table. Circuit sizes
/// are derived from adjacent offsets; the final circuit consumes all remaining
/// bytes.
pub fn read_raw_circuits(
    cursor: &mut Cursor<&[u8]>,
    num_programs: usize,
) -> Result<Vec<Bytes>, QpyError> {
    let circuit_table = Vec::<u64>::read_options(
        cursor,
        Endian::Big,
        VecArgs {
            count: num_programs,
            inner: (),
        },
    )?;

    // Read circuits using offset differences to determine sizes
    let mut circuits = Vec::new();
    circuits
        .try_reserve_exact(num_programs)
        .map_err(QpyError::AllocationError)?;

    for i in 0..num_programs {
        let size = if i + 1 < circuit_table.len() {
            if circuit_table[i] > circuit_table[i + 1] {
                return Err(QpyError::InvalidFormat(
                    "Circuit offset table invalid".to_string(),
                ));
            }
            (circuit_table[i + 1] - circuit_table[i]) as usize
        } else {
            // Last circuit: read remaining bytes
            let current_pos = cursor.stream_position()?;
            let end_pos = cursor.seek(std::io::SeekFrom::End(0))?;
            cursor.seek(std::io::SeekFrom::Start(current_pos))?;
            (end_pos - current_pos) as usize
        };

        let circuit = Bytes::read_options(
            cursor,
            Endian::Big,
            VecArgs::<Vec<u8>> {
                count: size,
                inner: Vec::new(),
            },
        )?;
        circuits.push(circuit);
    }
    Ok(circuits)
}

/// Serialize native circuits using empty metadata and no transpiler layout.
///
/// This is a non-Python convenience interface used by the C API.
pub fn native_dump_qpy(
    circuits: &[&CircuitData],
    qpy_version: Option<u8>,
) -> Result<Vec<u8>, QpyError> {
    let data = circuits.iter().copied().map(|circuit| {
        Ok((
            circuit,
            ExtraCircuitData {
                name: None,
                // The default Python QPY metadata codec is JSON, so an empty mapping must still be
                // represented by valid JSON for files produced through non-Python interfaces.
                metadata: Bytes::from("{}"),
                layout: None,
            },
        ))
    });
    dump_qpy(
        data,
        qpy_version.unwrap_or(QPY_VERSION),
        None,
        Some(QpyCaller::Native),
    )
    .map(|bytes| bytes.0)
}

/// Deserializes native circuits from a complete QPY payload.
///
/// # Arguments
///
/// * data: The complete QPY payload as a byte slice.
/// * annotation_handler: An optional annotation handler for deserializing annotations. If None, the native (dummy) handler is used.
///
/// # Returns
/// A vector of [`LoadedCircuit`] values, each containing the native `CircuitData` and packed data needed for any later Python-only construction.
pub fn load_qpy(
    data: &Bytes,
    annotation_handler: Option<AnnotationHandler>,
    caller: Option<QpyCaller>,
) -> Result<Vec<LoadedCircuit>, QpyError> {
    // Every QPY file begins with "QISKIT" followed by a version byte.
    // Since the header might be effected by the version, we begin by explicitly extracting the version.
    let qpy_version: u8 = *data.get(6).ok_or_else(|| {
        QpyError::ConversionError("QPY payload is empty; cannot read version".to_string())
    })?;
    if qpy_version < QPY_READ_MIN_VERSION {
        Err(QpyError::UnsupportedFeatureForVersion {
            feature: "Rust QPY".to_string(),
            version: qpy_version,
            min_version: QPY_READ_MIN_VERSION,
        })?;
    }
    let caller = caller.unwrap_or(QpyCaller::Native);
    let annotation_handler = annotation_handler.unwrap_or(AnnotationHandler::native(
        Vec::new(),
        NativeSerializers::default(),
        NativeDeserializers::default(),
    ));
    let (qpy_file_header, header_size) = deserialize::<QPYFileHeader>(data)?;
    // Verify the type key is for circuits
    if qpy_file_header.type_key == ProgramType::Schedule {
        return Err(QpyError::PayloadTypeError(
            "Payloads of type `Schedule` cannot be loaded as of Qiskit 2.0. \nUse an earlier version of Qiskit if you want to load `Schedule` payloads.".to_string()
        ));
    }
    if qpy_file_header.type_key != ProgramType::Circuit {
        return Err(QpyError::PayloadTypeError(format!(
            "Invalid payload format data kind '{}'",
            qpy_file_header.type_key
        )));
    }
    let num_programs = qpy_file_header.num_programs as usize;
    let use_symengine = matches!(
        qpy_file_header.symbolic_encoding,
        SymbolicEncoding::Symengine
    );
    let mut circuits = Vec::new();
    circuits
        .try_reserve_exact(num_programs)
        .map_err(QpyError::AllocationError)?;
    let mut cursor = Cursor::new(data as &[u8]);
    cursor.seek(std::io::SeekFrom::Start(header_size as u64))?;
    if qpy_file_header.qpy_version >= 16 {
        let qpy_raw_circuits = read_raw_circuits(&mut cursor, num_programs)?;
        for raw_circuit in &qpy_raw_circuits {
            let (packed_circuit, _) = deserialize_with_args::<QPYCircuit, (u8,)>(
                raw_circuit,
                (qpy_file_header.qpy_version,),
            )?;
            let circuit_data = unpack_circuit(
                &packed_circuit,
                qpy_file_header.qpy_version,
                use_symengine,
                annotation_handler.child()?,
                caller,
            )?;
            circuits.push(LoadedCircuit {
                circuit_data,
                packed_circuit,
            });
        }
    } else {
        // QPY version < 16, no offset table
        let packed_qpy_circuits = Vec::<QPYCircuit>::read_options(
            &mut cursor,
            Endian::Big,
            VecArgs {
                count: num_programs,
                inner: (qpy_file_header.qpy_version,),
            },
        )?;
        for packed_circuit in packed_qpy_circuits {
            let circuit_data = unpack_circuit(
                &packed_circuit,
                qpy_file_header.qpy_version,
                use_symengine,
                annotation_handler.child()?,
                caller,
            )?;
            circuits.push(LoadedCircuit {
                circuit_data,
                packed_circuit,
            });
        }
    }
    Ok(circuits)
}

#[pyfunction]
#[pyo3(name = "load")]
/// Python entry point for loading circuits from a readable file-like object.
///
/// This reads the complete payload from `file_obj`, reconstructs each native
/// circuit, and then applies Python-only metadata, layout, and annotation data
/// to create Python `QuantumCircuit` objects.
///
/// # Arguments
///
/// * py: The GIL handle.
/// * file_obj: The file-like object to read the QPY payload from
/// * metadata_deserializer: An optional Python callable that takes a serialized bytes object and returns a metadata dictionary.
/// * annotation_factories: An optional dictionary for annotation handlers
///
/// Returns
/// A list of Python `QuantumCircuit` objects reconstructed from the QPY payload.
pub fn py_load_qpy<'py>(
    py: Python<'py>,
    file_obj: &Bound<PyAny>,
    metadata_deserializer: Option<Bound<PyAny>>,
    annotation_factories: Option<Bound<PyDict>>,
) -> Result<Vec<Bound<'py, PyAny>>, QpyError> {
    let annotation_factories = annotation_factories.unwrap_or(PyDict::new(py));
    // Read all data from file object
    let data: Bytes = file_obj.call_method0("read")?.extract()?;

    let annotation_handler = AnnotationHandler::python(&annotation_factories.clone().unbind())?;
    load_qpy(&data, Some(annotation_handler), Some(QpyCaller::Python))?
        .into_iter()
        .map(|loaded| {
            py_circuit_data_to_quantum_circuit(
                py,
                loaded.circuit_data,
                &loaded.packed_circuit,
                metadata_deserializer.as_ref().map(Bound::as_ref),
            )
        })
        .collect()
}

/// Deserialize native circuits from a complete QPY payload.
pub fn native_load_qpy(data: &[u8]) -> Result<Vec<CircuitData>, QpyError> {
    load_qpy(&Bytes::from(data), None, Some(QpyCaller::Native)).map(|loaded| {
        loaded
            .into_iter()
            .map(|loaded| loaded.circuit_data)
            .collect()
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::operations::{DelayUnit, OperationRef, Param, StandardInstruction};
    use qiskit_circuit::packed_instruction::PackedOperation;
    use smallvec::smallvec;

    /// Builds the [`ExtraCircuitData`] required to serialize a native circuit that carries no
    /// Python-only metadata or transpiler layout.
    fn native_extra_data(circuit: &CircuitData, name: &str, version: u8) -> ExtraCircuitData {
        ExtraCircuitData {
            name: Some(name.to_string()),
            // Circuit metadata is always decoded with `json.loads`, and older Qiskit
            // releases require the decoded value to be a dictionary.
            metadata: "{}".into(),
            layout: Some(pack_layout(None, circuit, version).unwrap()),
        }
    }

    /// A circuit containing a single `Delay` measured in `dt` should survive a native
    /// QPY dump/load round trip: the instruction, its `dt` unit, and its integer-typed
    /// duration are all preserved.
    #[test]
    fn delay_in_dt_roundtrip() {
        let version = QPY_WRITE_MIN_VERSION;

        // Build a 1-qubit circuit with a `Delay` instruction of 13 dt value.
        let circuit = CircuitData::from_packed_operations(
            1,
            0,
            [Ok((
                PackedOperation::from_standard_instruction(StandardInstruction::Delay(
                    DelayUnit::DT,
                )),
                smallvec![Param::Int(13)],
                vec![Qubit(0)],
                Vec::with_capacity(0),
            ))],
            0.0.into(),
        )
        .unwrap();

        // Round trip through the native QPY dump/load entry points.
        let circuit_array = [circuit];
        let data = circuit_array.iter().map(|circuit| {
            Ok((
                circuit,
                native_extra_data(circuit, "delay_dt_circuit", version),
            ))
        });
        let payload = dump_qpy(data, version, None, None).unwrap();
        let loaded = load_qpy(&payload, None, None).unwrap();

        // Exactly one circuit, with exactly one instruction.
        assert_eq!(loaded.len(), 1);
        let loaded_circuit = &loaded[0].circuit_data;
        assert_eq!(loaded_circuit.num_qubits(), 1);
        assert_eq!(loaded_circuit.len(), 1);

        // That instruction is a `Delay` whose unit round-tripped as `dt`.
        let inst = &loaded_circuit.data()[0];
        let OperationRef::StandardInstruction(StandardInstruction::Delay(unit)) = inst.op.view()
        else {
            panic!(
                "expected a standard Delay instruction, got {:?}",
                inst.op.view()
            );
        };
        assert_eq!(unit, DelayUnit::DT);

        // The duration parameter is preserved.
        assert!(matches!(inst.params_view(), [Param::Int(d)] if *d == 13));
    }
}
