// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::PySequence;
use pyo3::types::PyString;
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{
    Param, StandardGate, StandardInstruction, add_param, multiply_param, multiply_params,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Clbit, Qubit};
use qiskit_synthesis::pauli_evolution::sparse_term_evolution;
use smallvec::{SmallVec, smallvec};
use std::f64::consts::PI;

use crate::QiskitError;
use crate::entanglement;

type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Build a Pauli feature map circuit.
///
/// Args:
///     feature_dimension: The feature dimension (i.e. the number of qubits).
///     parameters: A parameter vector with ``feature_dimension`` elements. Taken as input
///         here to avoid a call to Python constructing the vector.
///     reps: The number of repetitions of Hadamard + evolution layers.
///     entanglement: The entanglement, given as Python string or None (defaults to "full").
///     paulis: The Pauli strings as list of strings or None (default to ["z", "zz"]).
///     alpha: A float multiplier for rotation angles.
///     insert_barriers: Whether to insert barriers in between the Hadamard and evolution layers.
///     data_map_func: An accumulation function that takes as input a vector of parameters the
///         current gate acts on and returns a scalar.
///
/// Returns:
///     The ``CircuitData`` to construct the Pauli feature map.
#[pyfunction]
#[pyo3(signature = (feature_dimension, parameters, *, reps=1, entanglement=None, paulis=None, alpha=2.0, insert_barriers=false, data_map_func=None))]
#[allow(clippy::too_many_arguments)]
pub fn pauli_feature_map(
    py: Python,
    feature_dimension: u32,
    parameters: Bound<PyAny>,
    reps: usize,
    entanglement: Option<&Bound<PyAny>>,
    paulis: Option<&Bound<PySequence>>,
    alpha: f64,
    insert_barriers: bool,
    data_map_func: Option<&Bound<PyAny>>,
) -> PyResult<PyCircuitData> {
    // normalize the Pauli strings
    let pauli_strings = _get_paulis(feature_dimension, paulis)?;

    // set the default value for entanglement
    let default = PyString::new(py, "full");
    let entanglement = entanglement.unwrap_or(&default);

    // extract the parameters from the input variable ``parameters``
    let parameter_vector = parameters
        .try_iter()?
        .map(|el| Param::extract_no_coerce(el?.as_borrowed()))
        .collect::<PyResult<Vec<Param>>>()?;

    // construct a Barrier object Python side to (possibly) add to the circuit
    let packed_barrier = if insert_barriers {
        Some((
            PackedOperation::from_standard_instruction(StandardInstruction::Barrier(
                feature_dimension,
            )),
            smallvec![],
            (0..feature_dimension).map(Qubit).collect(),
            vec![] as Vec<Clbit>,
        ))
    } else {
        None
    };

    // Main work: construct the circuit instructions as iterator. Each repetition is constituted
    // by a layer of Hadamards and the Pauli evolutions of the specified Paulis.
    // Note that we eagerly trigger errors, since the final CircuitData::from_packed_operations
    // does not allow Result objects in the iterator.
    let mut packed_insts: Vec<Instruction> = Vec::new();
    for rep in 0..reps {
        // add H layer
        packed_insts.extend(_get_h_layer(feature_dimension));

        if insert_barriers {
            packed_insts.push(packed_barrier.clone().unwrap());
        }

        // add evolutions
        let evo_layer = _get_evolution_layer(
            feature_dimension,
            rep,
            alpha,
            &parameter_vector,
            &pauli_strings,
            entanglement,
            data_map_func,
        )?;
        packed_insts.extend(evo_layer);

        // add barriers, if necessary
        if insert_barriers && rep < reps - 1 {
            packed_insts.push(packed_barrier.clone().unwrap());
        }
    }

    Ok(CircuitData::from_packed_operations(
        feature_dimension,
        0,
        packed_insts.into_iter().map(Ok),
        Param::Float(0.0),
    )?
    .into())
}

fn _get_h_layer(feature_dimension: u32) -> impl Iterator<Item = Instruction> {
    (0..feature_dimension).map(|i| {
        (
            StandardGate::H.into(),
            smallvec![],
            vec![Qubit(i)],
            vec![] as Vec<Clbit>,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn _get_evolution_layer<'a>(
    feature_dimension: u32,
    rep: usize,
    alpha: f64,
    parameter_vector: &'a [Param],
    pauli_strings: &'a [String],
    entanglement: &'a Bound<PyAny>,
    data_map_func: Option<&'a Bound<PyAny>>,
) -> PyResult<Vec<Instruction>> {
    let mut insts: Vec<Instruction> = Vec::new();

    for pauli in pauli_strings {
        let block_size = pauli.len() as u32;
        let entanglement =
            entanglement::get_entanglement(feature_dimension, block_size, entanglement, rep)?;

        for indices in entanglement {
            let indices = indices?;
            let active_parameters: Vec<Param> = indices
                .clone()
                .iter()
                .map(|i| parameter_vector[*i as usize].clone())
                .collect();

            let angle = match data_map_func {
                Some(fun) => fun.call1((active_parameters,))?.extract()?,
                None => _default_reduce(active_parameters),
            };

            // Get the pauli evolution and map it into
            //   (PackedOperation, SmallVec<[Params; 3]>, Vec<Qubit>, Vec<Clbit>)
            // to call CircuitData::from_packed_operations. This is needed since we might
            // have to interject barriers, which are not a standard gate and prevents us
            // from using CircuitData::from_standard_gates.
            let evo = sparse_term_evolution(
                pauli,
                indices.into_iter().rev().collect(),
                multiply_param(&angle, alpha),
                true,
                false,
            );
            insts.extend(evo);
        }
    }

    Ok(insts)
}

/// The default data_map_func for Pauli feature maps. For a parameter vector (x1, ..., xN), this
/// implements
///   (pi - x1) (pi - x2) ... (pi - xN)
/// unless there is only one parameter, in which case it returns just the value.
fn _default_reduce(parameters: Vec<Param>) -> Param {
    if parameters.len() == 1 {
        parameters[0].clone()
    } else {
        let acc = parameters.iter().fold(Param::Float(1.0), |acc, param| {
            multiply_params(acc, add_param(param, -PI))
        });
        if parameters.len().is_multiple_of(2) {
            acc
        } else {
            multiply_param(&acc, -1.0) // take care of parity
        }
    }
}

/// Normalize the Pauli strings to a Vec<String>. We first define the default, which is
/// ["z", "zz"], unless we only have a single qubit, in which case we default to ["z"].
/// Then, ``pauli_strings`` is either set to the default, or we try downcasting to a
/// PyString->String, followed by a check whether the feature dimension is large enough
/// for the Pauli (e.g. we cannot implement a "zzz" Pauli on a 2 qubit circuit).
fn _get_paulis(
    feature_dimension: u32,
    paulis: Option<&Bound<PySequence>>,
) -> PyResult<Vec<String>> {
    let default_pauli: Vec<String> = if feature_dimension == 1 {
        vec!["z".to_string()]
    } else {
        vec!["z".to_string(), "zz".to_string()]
    };

    paulis.map_or_else(
        || Ok(default_pauli), // use Ok() since we might raise an error in the other arm
        |v| {
            let v = PySequenceMethods::to_list(v)?; // sequence to list
            v.iter() // iterate over the list of Paulis
                .map(|el| {
                    // Get the string and check whether it fits the feature dimension
                    let as_string = (*el.cast::<PyString>()?).to_string();
                    if as_string.len() > feature_dimension as usize {
                        Err(QiskitError::new_err(format!(
                            "feature_dimension ({feature_dimension}) smaller than the Pauli ({as_string})"
                        )))
                    } else {
                        Ok(as_string)
                    }
                })
                .collect::<PyResult<Vec<String>>>()
        },
    )
}
