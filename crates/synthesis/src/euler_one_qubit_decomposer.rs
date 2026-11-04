// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#![allow(clippy::too_many_arguments)]
#![allow(clippy::upper_case_acronyms)]

use hashbrown::HashMap;
use num_complex::{Complex64, ComplexFloat};
use smallvec::{SmallVec, smallvec};
use std::cmp::Ordering;
use std::f64::consts::PI;
use std::str::FromStr;

use pyo3::IntoPyObjectExt;
use pyo3::Python;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyo3::wrap_pyfunction;

use ndarray::prelude::*;
use numpy::PyReadonlyArray2;
use pyo3::pybacked::PyBackedStr;

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::dag_node::DAGOpNode;
use qiskit_circuit::operations::{Operation, Param, StandardGate};
use qiskit_circuit::{Qubit, impl_intopyobject_for_copy_pyclass};
use qiskit_util::complex::c64;
use qiskit_util::py::{PySequenceIndex, SequenceIndex};

pub const ANGLE_ZERO_EPSILON: f64 = 1e-12;

#[pyclass(module = "qiskit._accelerate.euler_one_qubit_decomposer")]
pub struct OneQubitGateErrorMap {
    error_map: Vec<HashMap<String, f64>>,
}

#[pymethods]
impl OneQubitGateErrorMap {
    #[new]
    #[pyo3(signature=(num_qubits=None))]
    fn new(num_qubits: Option<usize>) -> Self {
        OneQubitGateErrorMap {
            error_map: match num_qubits {
                Some(n) => Vec::with_capacity(n),
                None => Vec::new(),
            },
        }
    }

    fn add_qubit(&mut self, error_map: HashMap<String, f64>) {
        self.error_map.push(error_map);
    }

    fn __getstate__(&self) -> Vec<HashMap<String, f64>> {
        self.error_map.clone()
    }

    fn __setstate__(&mut self, state: Vec<HashMap<String, f64>>) {
        self.error_map = state;
    }
}

#[derive(Debug)]
#[pyclass(sequence)]
pub struct OneQubitGateSequence {
    pub gates: Vec<(StandardGate, SmallVec<[f64; 3]>)>,
    #[pyo3(get)]
    pub global_phase: f64,
}

type OneQubitGateSequenceState = (Vec<(StandardGate, SmallVec<[f64; 3]>)>, f64);

#[pymethods]
impl OneQubitGateSequence {
    #[new]
    fn new() -> Self {
        OneQubitGateSequence {
            gates: Vec::new(),
            global_phase: 0.,
        }
    }
    fn __getstate__(&self) -> OneQubitGateSequenceState {
        (self.gates.clone(), self.global_phase)
    }

    fn __setstate__(&mut self, state: OneQubitGateSequenceState) {
        self.gates = state.0;
        self.global_phase = state.1;
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.gates.len())
    }

    fn __getitem__(&self, py: Python, idx: PySequenceIndex) -> PyResult<Py<PyAny>> {
        match idx.with_len(self.gates.len())? {
            SequenceIndex::Int(idx) => Ok((&self.gates[idx]).into_py_any(py)?),
            indices => Ok(PyList::new(
                py,
                indices
                    .iter()
                    .map(|pos| (&self.gates[pos]).into_pyobject(py).unwrap()),
            )?
            .into_any()
            .unbind()),
        }
    }
}

fn circuit_kak(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    k_gate: StandardGate,
    a_gate: StandardGate,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut lam = lam;
    let mut theta = theta;
    let mut phi = phi;
    let mut circuit: Vec<(StandardGate, SmallVec<[f64; 3]>)> = Vec::with_capacity(3);
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    let mut global_phase = phase - (phi + lam) / 2.;
    if theta.abs() < atol {
        lam += phi;
        // NOTE: The following normalization is safe, because the gphase correction below
        // fixes a particular diagonal entry to 1, which prevents any potential phase
        // slippage coming from _mod_2pi injecting multiples of 2pi.
        lam = mod_2pi(lam, atol);
        if lam.abs() > atol {
            circuit.push((k_gate, smallvec![lam]));
            global_phase += lam / 2.;
        }
        return OneQubitGateSequence {
            gates: circuit,
            global_phase,
        };
    }
    if (theta - PI).abs() < atol {
        global_phase += phi;
        lam -= phi;
        phi = 0.;
    }
    if mod_2pi(lam + PI, atol).abs() < atol || mod_2pi(phi + PI, atol).abs() < atol {
        lam += PI;
        theta = -theta;
        phi += PI;
    }
    lam = mod_2pi(lam, atol);
    if lam.abs() > atol {
        global_phase += lam / 2.;
        circuit.push((k_gate, smallvec![lam]));
    }
    circuit.push((a_gate, smallvec![theta]));
    phi = mod_2pi(phi, atol);
    if phi.abs() > atol {
        global_phase += phi / 2.;
        circuit.push((k_gate, smallvec![phi]));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase,
    }
}

fn circuit_u3(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if simplify && theta.abs() < atol {
        // At zero theta the gate is diagonal and only depends on `phi + lam`.
        let tot = mod_2pi(phi + lam, atol);
        if tot.abs() > atol {
            circuit.push((StandardGate::U3, smallvec![0., 0., tot]));
        }
    } else {
        let phi = mod_2pi(phi, atol);
        let lam = mod_2pi(lam, atol);
        circuit.push((StandardGate::U3, smallvec![theta, phi, lam]));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

fn circuit_u321(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    if theta.abs() < atol {
        let tot = mod_2pi(phi + lam, atol);
        if tot.abs() > atol {
            circuit.push((StandardGate::U1, smallvec![tot]));
        }
    } else if (theta - PI / 2.).abs() < atol {
        circuit.push((
            StandardGate::U2,
            smallvec![mod_2pi(phi, atol), mod_2pi(lam, atol)],
        ));
    } else {
        circuit.push((
            StandardGate::U3,
            smallvec![theta, mod_2pi(phi, atol), mod_2pi(lam, atol)],
        ));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

fn circuit_u(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    if theta.abs() < atol {
        // At zero theta the gate is diagonal and only depends on `phi + lam`.
        let tot = mod_2pi(phi + lam, atol);
        if tot.abs() > atol {
            circuit.push((StandardGate::U, smallvec![0., 0., tot]));
        }
    } else {
        let phi = mod_2pi(phi, atol);
        let lam = mod_2pi(lam, atol);
        circuit.push((StandardGate::U, smallvec![theta, phi, lam]));
    }
    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

fn circuit_psx_gen<F, P, X>(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
    mut pfun: P,
    mut xfun: F,
    xpifun: Option<X>,
) -> OneQubitGateSequence
where
    F: FnMut(&mut OneQubitGateSequence),
    P: FnMut(&mut OneQubitGateSequence, f64),
    X: FnOnce(&mut OneQubitGateSequence),
{
    let mut phi = phi;
    let mut lam = lam;
    let mut theta = theta;
    let mut circuit = OneQubitGateSequence {
        gates: Vec::new(),
        global_phase: phase,
    };
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }
    // Early return for zero SX decomposition
    if theta.abs() < atol {
        pfun(&mut circuit, lam + phi);
        return circuit;
    }
    // Early return for single SX decomposition
    if (theta - PI / 2.).abs() < atol {
        pfun(&mut circuit, lam - PI / 2.);
        xfun(&mut circuit);
        pfun(&mut circuit, phi + PI / 2.);
        return circuit;
    }
    // General double SX decomposition
    if (theta - PI).abs() < atol {
        circuit.global_phase += lam;
        phi -= lam;
        lam = 0.;
    }
    if mod_2pi(lam + PI, atol).abs() < atol || mod_2pi(phi, atol).abs() < atol {
        lam += PI;
        theta = -theta;
        phi += PI;
        circuit.global_phase -= theta;
    }
    // Shift theta and phi to turn the decomposition from
    // RZ(phi).RY(theta).RZ(lam) = RZ(phi).RX(-pi/2).RZ(theta).RX(pi/2).RZ(lam)
    // into RZ(phi+pi).SX.RZ(theta+pi).SX.RZ(lam).
    theta += PI;
    phi += PI;
    circuit.global_phase -= PI / 2.;
    // emit circuit
    pfun(&mut circuit, lam);
    match xpifun {
        Some(xpifun) if mod_2pi(theta, atol).abs() < atol => xpifun(&mut circuit),
        _ => {
            xfun(&mut circuit);
            pfun(&mut circuit, theta);
            xfun(&mut circuit);
        }
    };
    pfun(&mut circuit, phi);
    circuit
}

fn circuit_rr(
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> OneQubitGateSequence {
    let mut circuit = Vec::new();
    let mut atol = match atol {
        Some(atol) => atol,
        None => ANGLE_ZERO_EPSILON,
    };
    if !simplify {
        atol = -1.0;
    }

    if mod_2pi((phi + lam) / 2., atol).abs() < atol {
        // This can be expressed as a single R gate
        if theta.abs() > atol {
            circuit.push((
                StandardGate::R,
                smallvec![theta, mod_2pi(PI / 2. + phi, atol)],
            ));
        }
    } else {
        // General case: use two R gates
        if (theta - PI).abs() > atol {
            circuit.push((
                StandardGate::R,
                smallvec![theta - PI, mod_2pi(PI / 2. - lam, atol)],
            ));
        }
        circuit.push((
            StandardGate::R,
            smallvec![PI, mod_2pi(0.5 * (phi - lam + PI), atol)],
        ));
    }

    OneQubitGateSequence {
        gates: circuit,
        global_phase: phase,
    }
}

#[pyfunction]
#[pyo3(signature=(target_basis, theta, phi, lam, phase, simplify, atol=None))]
pub fn generate_circuit(
    target_basis: &EulerBasis,
    theta: f64,
    phi: f64,
    lam: f64,
    phase: f64,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<OneQubitGateSequence> {
    let res = match target_basis {
        EulerBasis::ZYZ => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RZ,
            StandardGate::RY,
            simplify,
            atol,
        ),
        EulerBasis::ZXZ => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RZ,
            StandardGate::RX,
            simplify,
            atol,
        ),
        EulerBasis::XZX => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RX,
            StandardGate::RZ,
            simplify,
            atol,
        ),
        EulerBasis::XYX => circuit_kak(
            theta,
            phi,
            lam,
            phase,
            StandardGate::RX,
            StandardGate::RY,
            simplify,
            atol,
        ),
        EulerBasis::U3 => circuit_u3(theta, phi, lam, phase, simplify, atol),
        EulerBasis::U321 => circuit_u321(theta, phi, lam, phase, simplify, atol),
        EulerBasis::U => circuit_u(theta, phi, lam, phase, simplify, atol),
        EulerBasis::PSX => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((StandardGate::Phase, smallvec![phi]));
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::SX, SmallVec::new()));
            };

            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                None::<Box<dyn FnOnce(&mut OneQubitGateSequence)>>,
            )
        }
        EulerBasis::ZSX => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((StandardGate::RZ, smallvec![phi]));
                    circuit.global_phase += phi / 2.;
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::SX, SmallVec::new()));
            };
            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                None::<Box<dyn FnOnce(&mut OneQubitGateSequence)>>,
            )
        }
        EulerBasis::U1X => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((StandardGate::U1, smallvec![phi]));
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.global_phase += PI / 4.;
                circuit.gates.push((StandardGate::RX, smallvec![PI / 2.]));
            };
            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                None::<Box<dyn FnOnce(&mut OneQubitGateSequence)>>,
            )
        }
        EulerBasis::ZSXX => {
            let mut inner_atol = match atol {
                Some(atol) => atol,
                None => ANGLE_ZERO_EPSILON,
            };
            if !simplify {
                inner_atol = -1.0;
            }
            let fnz = |circuit: &mut OneQubitGateSequence, phi: f64| {
                let phi = mod_2pi(phi, inner_atol);
                if phi.abs() > inner_atol {
                    circuit.gates.push((StandardGate::RZ, smallvec![phi]));
                    circuit.global_phase += phi / 2.;
                }
            };
            let fnx = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::SX, SmallVec::new()));
            };
            let fnxpi = |circuit: &mut OneQubitGateSequence| {
                circuit.gates.push((StandardGate::X, SmallVec::new()));
            };
            circuit_psx_gen(
                theta,
                phi,
                lam,
                phase,
                simplify,
                atol,
                fnz,
                fnx,
                Some(fnxpi),
            )
        }
        EulerBasis::RR => circuit_rr(theta, phi, lam, phase, simplify, atol),
    };
    Ok(res)
}

const EULER_BASIS_SIZE: usize = 12;

pub static EULER_BASES: [&[&str]; EULER_BASIS_SIZE] = [
    &["u3"],
    &["u3", "u2", "u1"],
    &["u"],
    &["p", "sx"],
    &["u1", "rx"],
    &["r"],
    &["rz", "ry"],
    &["rz", "rx"],
    &["rz", "rx"],
    &["rx", "ry"],
    &["rz", "sx", "x"],
    &["rz", "sx"],
];
pub static EULER_BASIS_NAMES: [EulerBasis; EULER_BASIS_SIZE] = [
    EulerBasis::U3,
    EulerBasis::U321,
    EulerBasis::U,
    EulerBasis::PSX,
    EulerBasis::U1X,
    EulerBasis::RR,
    EulerBasis::ZYZ,
    EulerBasis::ZXZ,
    EulerBasis::XZX,
    EulerBasis::XYX,
    EulerBasis::ZSXX,
    EulerBasis::ZSX,
];

/// A structure containing a set of supported `EulerBasis` for running 1q synthesis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EulerBasisSet {
    basis: [bool; EULER_BASIS_SIZE],
    initialized: bool,
}

impl EulerBasisSet {
    // Instantiate a new EulerBasisSet
    pub fn new() -> Self {
        EulerBasisSet {
            basis: [false; EULER_BASIS_SIZE],
            initialized: false,
        }
    }

    /// Get the set of supported bases given a function that marks whether each basis gate (by name)
    /// is supported.
    ///
    /// The `is_supported` function may be called more than once for the same gate name.
    pub fn from_support(mut is_supported: impl FnMut(&str) -> bool) -> Self {
        let mut out = Self {
            basis: EULER_BASES.map(|basis| basis.iter().all(|gate| is_supported(gate))),
            initialized: true,
        };
        if out.basis_supported(EulerBasis::U321) && out.basis_supported(EulerBasis::U3) {
            out.remove(EulerBasis::U3);
        }
        if out.basis_supported(EulerBasis::ZSXX) && out.basis_supported(EulerBasis::ZSX) {
            out.remove(EulerBasis::ZSX);
        }
        out
    }

    /// Return true if this has been initialized any basis is supported
    pub fn initialized(&self) -> bool {
        self.initialized
    }

    /// Add a basis to the set
    pub fn add_basis(&mut self, basis: EulerBasis) {
        self.basis[basis as usize] = true;
        self.initialized = true;
    }

    /// Get an iterator of all the supported EulerBasis
    pub fn get_bases(&self) -> impl Iterator<Item = EulerBasis> + '_ {
        self.basis
            .iter()
            .enumerate()
            .filter_map(|(index, supported)| {
                if *supported {
                    Some(EULER_BASIS_NAMES[index])
                } else {
                    None
                }
            })
    }

    /// Check if a basis is supported by this set
    pub fn basis_supported(&self, basis: EulerBasis) -> bool {
        self.basis[basis as usize]
    }

    /// Modify this set to support all EulerBasis
    pub fn support_all(&mut self) {
        self.basis = [true; 12];
        self.initialized = true;
    }

    /// Remove an basis from the set
    pub fn remove(&mut self, basis: EulerBasis) {
        self.basis[basis as usize] = false;
    }
}

impl Default for EulerBasisSet {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Copy, Eq, Hash, PartialEq)]
#[pyclass(
    module = "qiskit._accelerate.euler_one_qubit_decomposer",
    eq,
    eq_int,
    from_py_object
)]
pub enum EulerBasis {
    U3 = 0,
    U321 = 1,
    U = 2,
    PSX = 3,
    U1X = 4,
    RR = 5,
    ZYZ = 6,
    ZXZ = 7,
    XZX = 8,
    XYX = 9,
    ZSXX = 10,
    ZSX = 11,
}
impl_intopyobject_for_copy_pyclass!(EulerBasis);

impl EulerBasis {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U321 => "U321",
            Self::U3 => "U3",
            Self::U => "U",
            Self::PSX => "PSX",
            Self::ZSX => "ZSX",
            Self::ZSXX => "ZSXX",
            Self::U1X => "U1X",
            Self::RR => "RR",
            Self::ZYZ => "ZYZ",
            Self::ZXZ => "ZXZ",
            Self::XYX => "XYX",
            Self::XZX => "XZX",
        }
    }
}

#[pymethods]
impl EulerBasis {
    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (py.get_type::<Self>(), (PyString::new(py, self.as_str()),)).into_py_any(py)
    }

    #[new]
    pub fn __new__(input: &str) -> PyResult<Self> {
        Self::from_str(input)
            .map_err(|_| PyValueError::new_err(format!("Invalid target basis '{input}'")))
    }
}

impl FromStr for EulerBasis {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "U321" => Ok(EulerBasis::U321),
            "U3" => Ok(EulerBasis::U3),
            "U" => Ok(EulerBasis::U),
            "PSX" => Ok(EulerBasis::PSX),
            "ZSX" => Ok(EulerBasis::ZSX),
            "ZSXX" => Ok(EulerBasis::ZSXX),
            "U1X" => Ok(EulerBasis::U1X),
            "RR" => Ok(EulerBasis::RR),
            "ZYZ" => Ok(EulerBasis::ZYZ),
            "ZXZ" => Ok(EulerBasis::ZXZ),
            "XYX" => Ok(EulerBasis::XYX),
            "XZX" => Ok(EulerBasis::XZX),
            _ => Err(()),
        }
    }
}

#[inline]
pub fn angles_from_unitary(unitary: ArrayView2<Complex64>, target_basis: EulerBasis) -> [f64; 4] {
    match target_basis {
        EulerBasis::U321 => params_u3_inner(unitary),
        EulerBasis::U3 => params_u3_inner(unitary),
        EulerBasis::U => params_u3_inner(unitary),
        EulerBasis::PSX => params_u1x_inner(unitary),
        EulerBasis::ZSX => params_u1x_inner(unitary),
        EulerBasis::ZSXX => params_u1x_inner(unitary),
        EulerBasis::U1X => params_u1x_inner(unitary),
        EulerBasis::RR => params_zyz_inner(unitary),
        EulerBasis::ZYZ => params_zyz_inner(unitary),
        EulerBasis::ZXZ => params_zxz_inner(unitary),
        EulerBasis::XYX => params_xyx_inner(unitary),
        EulerBasis::XZX => params_xzx_inner(unitary),
    }
}

#[inline]
fn compare_error_fn(
    circuit: &OneQubitGateSequence,
    error_map: &Option<&OneQubitGateErrorMap>,
    qubit: usize,
) -> (f64, usize) {
    match error_map {
        Some(global_err_map) => {
            let err_map = &global_err_map.error_map[qubit];
            let fidelity_product: f64 = circuit
                .gates
                .iter()
                .map(|gate| 1. - err_map.get(gate.0.name()).unwrap_or(&0.))
                .product();
            (1. - fidelity_product, circuit.gates.len())
        }
        None => (circuit.gates.len() as f64, circuit.gates.len()),
    }
}

fn compute_error_term(gate: &str, error_map: &OneQubitGateErrorMap, qubit: usize) -> f64 {
    1. - error_map.error_map[qubit].get(gate).unwrap_or(&0.)
}

fn compute_error_str(
    gates: &[(String, SmallVec<[f64; 3]>)],
    error_map: Option<&OneQubitGateErrorMap>,
    qubit: usize,
) -> (f64, usize) {
    match error_map {
        Some(err_map) => {
            let num_gates = gates.len();
            let gate_fidelities: f64 = gates
                .iter()
                .map(|gate| compute_error_term(gate.0.as_str(), err_map, qubit))
                .product();
            (1. - gate_fidelities, num_gates)
        }
        None => (gates.len() as f64, gates.len()),
    }
}

#[pyfunction]
#[pyo3(signature=(circuit, qubit, error_map=None))]
pub fn compute_error_list(
    circuit: Vec<PyRef<DAGOpNode>>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
) -> (f64, usize) {
    let circuit_list: Vec<(String, SmallVec<[f64; 3]>)> = circuit
        .iter()
        .map(|node| {
            (
                node.instruction.operation.name().to_string(),
                smallvec![], // Params not needed in this path
            )
        })
        .collect();
    compute_error_str(&circuit_list, error_map, qubit)
}

#[pyfunction]
#[pyo3(signature = (unitary, target_basis_list, qubit, error_map=None, simplify=true, atol=None))]
pub fn unitary_to_gate_sequence(
    unitary: PyReadonlyArray2<Complex64>,
    target_basis_list: Vec<PyBackedStr>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<Option<OneQubitGateSequence>> {
    let mut target_basis_set = EulerBasisSet::new();
    for basis in target_basis_list
        .iter()
        .map(|basis| EulerBasis::__new__(basis))
    {
        target_basis_set.add_basis(basis?);
    }
    Ok(unitary_to_gate_sequence_inner(
        unitary.as_array(),
        &target_basis_set,
        qubit,
        error_map,
        simplify,
        atol,
    ))
}

#[inline]
pub fn unitary_to_gate_sequence_inner(
    unitary_mat: ArrayView2<Complex64>,
    target_basis_list: &EulerBasisSet,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
    simplify: bool,
    atol: Option<f64>,
) -> Option<OneQubitGateSequence> {
    target_basis_list
        .get_bases()
        .map(|target_basis| {
            let [theta, phi, lam, phase] = angles_from_unitary(unitary_mat, target_basis);
            generate_circuit(&target_basis, theta, phi, lam, phase, simplify, atol).unwrap()
        })
        .min_by(|a, b| {
            let error_a = compare_error_fn(a, &error_map, qubit);
            let error_b = compare_error_fn(b, &error_map, qubit);
            error_a.partial_cmp(&error_b).unwrap_or(Ordering::Equal)
        })
}

#[pyfunction]
#[pyo3(signature = (unitary, target_basis_list, qubit, error_map=None, simplify=true, atol=None))]
pub fn unitary_to_circuit(
    unitary: PyReadonlyArray2<Complex64>,
    target_basis_list: Vec<PyBackedStr>,
    qubit: usize,
    error_map: Option<&OneQubitGateErrorMap>,
    simplify: bool,
    atol: Option<f64>,
) -> PyResult<Option<PyCircuitData>> {
    let mut target_basis_set = EulerBasisSet::new();
    for basis in target_basis_list
        .iter()
        .map(|basis| EulerBasis::__new__(basis))
    {
        target_basis_set.add_basis(basis?);
    }
    let circuit_sequence = unitary_to_gate_sequence_inner(
        unitary.as_array(),
        &target_basis_set,
        qubit,
        error_map,
        simplify,
        atol,
    );
    Ok(circuit_sequence.map(|seq| {
        CircuitData::from_standard_gates(
            1,
            seq.gates.into_iter().map(|(gate, params)| {
                (
                    gate,
                    params.into_iter().map(Param::Float).collect(),
                    smallvec![Qubit(0)],
                )
            }),
            Param::Float(seq.global_phase),
        )
        .map(Into::into)
        .expect("Unexpected Qiskit python bug")
    }))
}

#[inline]
pub fn det_one_qubit(mat: ArrayView2<Complex64>) -> Complex64 {
    mat[[0, 0]] * mat[[1, 1]] - mat[[0, 1]] * mat[[1, 0]]
}

/// Wrap angle into interval [-π,π). If within atol of the endpoint, clamp to -π
#[inline]
pub fn mod_2pi(angle: f64, atol: f64) -> f64 {
    // f64::rem_euclid() isn't exactly the same as Python's % operator, but because
    // the RHS here is a constant and positive it is effectively equivalent for
    // this case
    let wrapped = (angle + PI).rem_euclid(2. * PI) - PI;
    if (wrapped - PI).abs() < atol {
        -PI
    } else {
        wrapped
    }
}

fn params_zyz_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let det_arg = det_one_qubit(mat).arg();
    let phase = 0.5 * det_arg;
    let theta = 2. * mat[[1, 0]].abs().atan2(mat[[0, 0]].abs());
    let ang1 = mat[[1, 1]].arg();
    let ang2 = mat[[1, 0]].arg();
    let phi = ang1 + ang2 - det_arg;
    let lam = ang1 - ang2;
    [theta, phi, lam, phase]
}

pub fn params_zxz_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let [theta, phi, lam, phase] = params_zyz_inner(mat);
    [theta, phi + PI / 2., lam - PI / 2., phase]
}

#[pyfunction]
pub fn params_zyz(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_zyz_inner(mat)
}

fn params_u3_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    // The determinant of U3 gate depends on its params
    // via det(u3(theta, phi, lam)) = exp(1j*(phi+lam))
    // Since the phase is wrt to a SU matrix we must rescale
    // phase to correct this
    let [theta, phi, lam, phase] = params_zyz_inner(mat);
    [theta, phi, lam, phase - 0.5 * (phi + lam)]
}

#[pyfunction]
pub fn params_u3(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_u3_inner(mat)
}

fn params_u1x_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    // The determinant of this decomposition depends on its params
    // Since the phase is wrt to a SU matrix we must rescale
    // phase to correct this
    let [theta, phi, lam, phase] = params_zyz_inner(mat);
    [theta, phi, lam, phase - 0.5 * (theta + phi + lam)]
}

#[pyfunction]
pub fn params_u1x(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_u1x_inner(mat)
}

fn params_xyx_inner(mat: ArrayView2<Complex64>) -> [f64; 4] {
    let mat_zyz = arr2(&[
        [
            0.5 * (mat[[0, 0]] + mat[[0, 1]] + mat[[1, 0]] + mat[[1, 1]]),
            0.5 * (mat[[0, 0]] - mat[[0, 1]] + mat[[1, 0]] - mat[[1, 1]]),
        ],
        [
            0.5 * (mat[[0, 0]] + mat[[0, 1]] - mat[[1, 0]] - mat[[1, 1]]),
            0.5 * (mat[[0, 0]] - mat[[0, 1]] - mat[[1, 0]] + mat[[1, 1]]),
        ],
    ]);
    let [theta, phi, lam, phase] = params_zyz_inner(mat_zyz.view());
    let new_phi = mod_2pi(phi + PI, 0.);
    let new_lam = mod_2pi(lam + PI, 0.);
    [
        theta,
        new_phi,
        new_lam,
        phase + (new_phi + new_lam - phi - lam) / 2.,
    ]
}

#[pyfunction]
pub fn params_xyx(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_xyx_inner(mat)
}

fn params_xzx_inner(umat: ArrayView2<Complex64>) -> [f64; 4] {
    let det = det_one_qubit(umat);
    let phase = det.ln().im / 2.;
    let sqrt_det = det.sqrt();
    let mat_zyz = arr2(&[
        [
            c64((umat[[0, 0]] / sqrt_det).re, (umat[[1, 0]] / sqrt_det).im),
            c64((umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
        ],
        [
            c64(-(umat[[1, 0]] / sqrt_det).re, (umat[[0, 0]] / sqrt_det).im),
            c64((umat[[0, 0]] / sqrt_det).re, -(umat[[1, 0]] / sqrt_det).im),
        ],
    ]);
    let [theta, phi, lam, phase_zxz] = params_zxz_inner(mat_zyz.view());
    [theta, phi, lam, phase + phase_zxz]
}

#[pyfunction]
pub fn params_xzx(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let umat = unitary.as_array();
    params_xzx_inner(umat)
}

#[pyfunction]
pub fn params_zxz(unitary: PyReadonlyArray2<Complex64>) -> [f64; 4] {
    let mat = unitary.as_array();
    params_zxz_inner(mat)
}

pub fn euler_one_qubit_decomposer(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(params_zyz))?;
    m.add_wrapped(wrap_pyfunction!(params_xyx))?;
    m.add_wrapped(wrap_pyfunction!(params_xzx))?;
    m.add_wrapped(wrap_pyfunction!(params_zxz))?;
    m.add_wrapped(wrap_pyfunction!(params_u3))?;
    m.add_wrapped(wrap_pyfunction!(params_u1x))?;
    m.add_wrapped(wrap_pyfunction!(generate_circuit))?;
    m.add_wrapped(wrap_pyfunction!(unitary_to_gate_sequence))?;
    m.add_wrapped(wrap_pyfunction!(unitary_to_circuit))?;
    m.add_wrapped(wrap_pyfunction!(compute_error_list))?;
    m.add_class::<OneQubitGateSequence>()?;
    m.add_class::<OneQubitGateErrorMap>()?;
    m.add_class::<EulerBasis>()?;
    Ok(())
}
