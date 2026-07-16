// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use pyo3::{PyResult, Python};
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError, PyCircuitData};
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyInstruction, PyOpKind, StandardGate, multiply_param,
};
use qiskit_circuit::{BlocksMode, imports};
use qiskit_circuit::{Clbit, Qubit, VarsMode};
use smallvec::SmallVec;

use qiskit_circuit::instruction::Parameters;
use std::f64::consts::PI;

use crate::QiskitError;
const PI2: f64 = PI / 2.0;

/// Definition circuit for CCX.
pub fn ccx() -> CircuitData {
    StandardGate::CCX.definition(&[]).unwrap()
}

/// Definition circuit for C3X.
#[pyfunction]
pub fn c3x() -> PyCircuitData {
    StandardGate::C3X.definition(&[]).unwrap().into()
}

/// Definition circuit for RCCX.
fn rccx() -> CircuitData {
    StandardGate::RCCX.definition(&[]).unwrap()
}

/// Definition circuit for RC3X.
fn rc3x() -> CircuitData {
    StandardGate::RC3X.definition(&[]).unwrap()
}

/// Definition circuit for C3SX.
fn c3sx() -> CircuitData {
    StandardGate::C3SX.definition(&[]).unwrap()
}

/// Convenience methods to add gates to the circuit.
///
/// This trait is **not** intended to be user-facing. It defines utility functions
/// that make the code easier to read and that are used only for synthesis.
trait CircuitDataForSynthesis {
    /// Appends H to the circuit.
    fn h(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends X to the circuit.
    #[allow(dead_code)]
    fn x(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends T to the circuit.
    fn t(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends Tdg to the circuit.
    fn tdg(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends Phase to the circuit.
    #[allow(dead_code)]
    fn p(&mut self, theta: f64, q: u32) -> Result<(), CircuitDataError>;

    /// Appends CX to the circuit.
    fn cx(&mut self, q1: u32, q2: u32) -> Result<(), CircuitDataError>;

    /// Appends CPhase to the circuit.
    fn cp(&mut self, theta: f64, q1: u32, q2: u32) -> Result<(), CircuitDataError>;

    /// Appends CCPhase to the circuit.
    fn ccp(&mut self, theta: f64, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError>;

    /// Appends CCX to the circuit.
    fn ccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError>;

    /// Appends RCCX to the circuit.
    fn rccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError>;

    /// Compose ``other`` into ``self``, while remapping the qubits
    /// over which ``other`` is defined. The operations are added in-place.
    fn compose(
        &mut self,
        other: &Self,
        qargs_map: &[Qubit],
        cargs_map: &[Clbit],
    ) -> Result<(), CircuitDataError>;

    /// Construct the inverse circuit
    fn inverse(&self) -> Result<CircuitData, CircuitDataError>;
}

impl CircuitDataForSynthesis for CircuitData {
    /// Appends H to the circuit.
    #[inline]
    fn h(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::H, &[], &[Qubit(q)])
    }

    /// Appends X to the circuit.
    #[inline]
    fn x(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::X, &[], &[Qubit(q)])
    }

    /// Appends T to the circuit.
    #[inline]
    fn t(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::T, &[], &[Qubit(q)])
    }

    /// Appends Tdg to the circuit.
    #[inline]
    fn tdg(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::Tdg, &[], &[Qubit(q)])
    }

    /// Appends Phase to the circuit.
    #[inline]
    fn p(&mut self, theta: f64, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::Phase, &[Param::Float(theta)], &[Qubit(q)])
    }

    /// Appends CX to the circuit.
    #[inline]
    fn cx(&mut self, q1: u32, q2: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::CX, &[], &[Qubit(q1), Qubit(q2)])
    }

    /// Appends CPhase to the circuit.
    #[inline]
    fn cp(&mut self, theta: f64, q1: u32, q2: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(
            StandardGate::CPhase,
            &[Param::Float(theta)],
            &[Qubit(q1), Qubit(q2)],
        )
    }

    fn ccp(&mut self, theta: f64, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError> {
        self.cx(q1, q3)?;
        self.p(-theta / 4., q3)?;
        self.cx(q2, q3)?;
        self.p(theta / 4., q3)?;
        self.cx(q1, q3)?;
        self.p(-theta / 4., q3)?;
        self.cx(q2, q3)?;
        self.p(theta / 4., q3)?;
        self.p(theta / 4., q1)?;
        self.p(theta / 4., q2)?;
        self.cx(q1, q2)?;
        self.p(-theta / 4., q2)?;
        self.cx(q1, q2)?;
        Ok(())
    }

    /// Appends the decomposition of the CCX to the circuit.
    fn ccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError> {
        self.compose(&ccx(), &[Qubit(q1), Qubit(q2), Qubit(q3)], &[])
    }

    /// Appends RCCX to the circuit.
    fn rccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError> {
        self.compose(&rccx(), &[Qubit(q1), Qubit(q2), Qubit(q3)], &[])
    }

    /// Compose ``other`` into ``self``, while remapping the qubits over which ``other`` is defined.
    /// The operations are added in-place.
    fn compose(
        &mut self,
        other: &Self,
        qargs_map: &[Qubit],
        cargs_map: &[Clbit],
    ) -> Result<(), CircuitDataError> {
        if other.num_qubits() > self.num_qubits() {
            panic!("Cannot compose a larger circuit onto a smaller circuit.");
        }

        for inst in other.data() {
            let remapped_qubits: Vec<Qubit> = other
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| qargs_map[q.index()])
                .collect();
            let remapped_clbits: Vec<Clbit> = other
                .get_cargs(inst.clbits)
                .iter()
                .map(|c| cargs_map[c.index()])
                .collect();

            self.push_packed_operation(
                inst.op.clone(),
                inst.params.as_deref().cloned(),
                &remapped_qubits,
                &remapped_clbits,
            )?;
        }

        self.add_global_phase(other.global_phase())?;
        Ok(())
    }

    /// Construct the inverse circuit
    fn inverse(&self) -> Result<CircuitData, CircuitDataError> {
        let inverse_global_phase = multiply_param(self.global_phase(), -1.0);

        let mut inverse_circuit =
            CircuitData::copy_empty_like(self, VarsMode::Alike, BlocksMode::Keep)?;
        inverse_circuit.set_global_phase_param(inverse_global_phase)?;

        let data = self.data();

        for i in 0..data.len() {
            let inst = &data[data.len() - 1 - i];

            let inverse_inst: Option<(StandardGate, SmallVec<[Param; 3]>)> = match &inst.op.view() {
                OperationRef::StandardGate(gate) => gate.inverse(inst.params_view()),
                _ => None,
            };

            if inverse_inst.is_none() {
                panic!(
                    "The circuit cannot be inverted: {} is not a standard gate.",
                    inst.op.name()
                );
            }

            let (inverse_op, inverse_op_params) = inverse_inst.unwrap();

            inverse_circuit.push_packed_operation(
                inverse_op.into(),
                Some(Parameters::Params(inverse_op_params)),
                self.get_qargs(inst.qubits),
                self.get_cargs(inst.clbits),
            )?;
        }
        Ok(inverse_circuit)
    }
}

/// Efficient synthesis for 4-controlled X-gate.
#[pyfunction]
pub fn c4x() -> Result<PyCircuitData, CircuitDataError> {
    let mut circuit = CircuitData::with_capacity(5, 0, 0, Param::Float(0.0))?;
    circuit.h(4)?;
    circuit.cp(PI2, 3, 4)?;
    circuit.h(4)?;
    circuit.compose(&rc3x(), &[Qubit(0), Qubit(1), Qubit(2), Qubit(3)], &[])?;
    circuit.h(4)?;
    circuit.cp(-PI2, 3, 4)?;
    circuit.h(4)?;
    circuit.compose(
        &rc3x().inverse()?,
        &[Qubit(0), Qubit(1), Qubit(2), Qubit(3)],
        &[],
    )?;
    circuit.compose(&c3sx(), &[Qubit(0), Qubit(1), Qubit(2), Qubit(4)], &[])?;
    Ok(circuit.into())
}

/// Adds gates of the "action gadget" to the circuit
fn add_action_gadget(
    circuit: &mut CircuitData,
    q0: u32,
    q1: u32,
    q2: u32,
) -> Result<(), CircuitDataError> {
    circuit.h(q2)?;
    circuit.t(q2)?;
    circuit.cx(q0, q2)?;
    circuit.tdg(q2)?;
    circuit.cx(q1, q2)
}

/// Adds gates of the "reset gadget" to the circuit
fn add_reset_gadget(
    circuit: &mut CircuitData,
    q0: u32,
    q1: u32,
    q2: u32,
) -> Result<(), CircuitDataError> {
    circuit.cx(q1, q2)?;
    circuit.t(q2)?;
    circuit.cx(q0, q2)?;
    circuit.tdg(q2)?;
    circuit.h(q2)
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on the paper
/// by Iten et al. [1].
///
/// For :math:`k\ge 4` the method uses :math:`k - 2` dirty ancillary qubits, producing a circuit
/// with :math:`2 * k - 1` qubits and at most :math:`8 * k - 6` CX gates. For :math:`k\le 3`
/// explicit efficient circuits are used instead.
///
/// # Arguments
/// - num_controls: the number of control qubits.
/// - relative_phase: when set to `true`, the method applies the optimized multi-controlled
///   X gate up to a relative phase, in a way that the relative phases of the `action part`
///   cancel out with the relative phases of the `reset part`.
/// - action_only: when set to `true`, the methods applies only the `action part`.
///
/// # References
///
/// 1. Iten et al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
///    [arXiv:1501.06911] (https://arxiv.org/abs/1501.06911).
pub fn synth_mcx_n_dirty_i15(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> Result<CircuitData, CircuitDataError> {
    if num_controls == 0 {
        let mut circuit = CircuitData::with_capacity(1, 0, 1, Param::Float(0.0))?;
        circuit.x(0)?;
        Ok(circuit)
    } else if num_controls == 1 {
        let mut circuit = CircuitData::with_capacity(2, 0, 1, Param::Float(0.0))?;
        circuit.cx(0, 1)?;
        Ok(circuit)
    } else if num_controls == 2 {
        Ok(ccx())
    } else if num_controls == 3 && !relative_phase {
        Ok(c3x().into())
    } else {
        let num_ancillas = num_controls - 2;
        let num_qubits = num_controls + 1 + num_ancillas;
        let mut circuit = CircuitData::with_capacity(num_qubits as u32, 0, 0, Param::Float(0.0))?;

        let controls: Vec<u32> = (0..num_controls).map(|q| q as u32).collect();
        let target = num_controls as u32;
        let ancillas: Vec<u32> = ((num_controls + 1)..num_qubits).map(|q| q as u32).collect();

        for j in 0..2 {
            if !relative_phase {
                circuit.ccx(
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
            } else if j == 0 {
                add_action_gadget(
                    &mut circuit,
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
            } else if j == 1 {
                add_reset_gadget(
                    &mut circuit,
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
            }

            // action part
            for i in (0..num_controls - 3).rev() {
                add_action_gadget(&mut circuit, controls[i + 2], ancillas[i], ancillas[i + 1])?;
            }

            circuit.rccx(controls[0], controls[1], ancillas[0])?;

            // reset part
            for i in 0..num_controls - 3 {
                add_reset_gadget(&mut circuit, controls[i + 2], ancillas[i], ancillas[i + 1])?;
            }

            if action_only {
                circuit.ccx(
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
                break;
            }
        }
        Ok(circuit)
    }
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on
/// the implementation for `MCPhaseGate`.
///
/// In turn, the MCPhase gate uses the decomposition for multi-controlled
/// special unitaries described in [1].
///
/// # Arguments
/// - num_controls: the number of control qubits.
///
/// # Returns
///
/// A quantum circuit with :math:`k + 1` qubits. The number of CX-gates is
/// quadratic in :math:`k`.
///
/// # References
///
/// 1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
///    Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
///    [arXiv:2302.06377] (https://arxiv.org/abs/2302.06377).
pub fn synth_mcx_noaux_v24(
    py: Python,
    num_controls: usize,
) -> Result<CircuitData, CircuitDataError> {
    if num_controls == 0 {
        let mut circuit = CircuitData::with_capacity(1, 0, 1, Param::Float(0.0))?;
        circuit.x(0)?;
        Ok(circuit)
    } else if num_controls == 1 {
        let mut circuit = CircuitData::with_capacity(2, 0, 1, Param::Float(0.0))?;
        circuit.cx(0, 1)?;
        Ok(circuit)
    } else if num_controls == 2 {
        Ok(ccx())
    } else if num_controls == 3 {
        Ok(c3x().into())
    } else if num_controls == 4 {
        c4x().map(Into::into)
    } else {
        let num_qubits = (num_controls + 1) as u32;
        let target = num_controls as u32;

        let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;
        circuit.h(target)?;

        let mcphase_cls = imports::MCPHASE_GATE.get_bound(py);
        let mcphase_gate = mcphase_cls
            .call1((PI, num_controls))
            .map_err(CircuitDataError::ErrorFromPython)?;

        let inst = PyInstruction {
            qubits: num_qubits,
            clbits: 0,
            params: 1,
            op_name: "mcphase".to_string(),
            ob: mcphase_gate.into(),
            kind: PyOpKind::Gate,
        };
        circuit.push_packed_operation(
            inst.into(),
            None,
            &(0..num_qubits).map(Qubit).collect::<Vec<Qubit>>(),
            &[],
        )?;

        circuit.h(target)?;

        Ok(circuit)
    }
}

// The following synth_mcx_noaux_hp24 algorithm is based on the work by Huang and Palsberg.
//
// # References
//
// 1. Huang and Palsberg, *Compiling Conditional Quantum Gates without Using Helper Qubits*, PLDI (2024),
// https://dl.acm.org/doi/10.1145/3656436.
// 2. The supplementary material for [1] that can be downloaded from the link above.
// 3. Python implementation, available at https://github.com/Keli-Huang/Qulin_Large_Toffoli

/// Synthesize the :math:`n`-qubit increment gate using :math:`n` dirty ancilla qubits.
///
/// The construction appears in Fig. 18 in the supplementary material [2].
///
/// Best suitable when n is large.
fn increment_n_dirty_large(n: u32) -> PyResult<CircuitData> {
    // U_x^3-gate from Fig. 22 in [2].
    fn ux(circuit: &mut CircuitData, q1: u32, q2: u32, q3: u32) -> PyResult<()> {
        circuit.cx(q1, q3)?;
        circuit.cx(q1, q2)?;
        circuit.ccx(q2, q3, q1)?;
        Ok(())
    }

    // U_z^3-gate from Fig. 24 in [2].
    fn uz(circuit: &mut CircuitData, q1: u32, q2: u32, q3: u32) -> PyResult<()> {
        circuit.ccx(q2, q3, q1)?;
        circuit.cx(q1, q2)?;
        circuit.cx(q2, q3)?;
        Ok(())
    }

    let mut circuit = CircuitData::with_capacity(2 * n, 0, 0, Param::Float(0.0))?;
    let qubits: Vec<u32> = (0..n).collect();
    let ancillas: Vec<u32> = (n..2 * n).collect();

    circuit.x(ancillas[0])?;
    for q in qubits.iter() {
        circuit.cx(ancillas[0], *q)?;
    }
    circuit.x(ancillas[0])?;

    // This implements U^{n}_{z+y+x} in Fig.19 and Fig.23 from the supplementary material for [1].
    for i in 0..n - 1 {
        ux(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }

    circuit.cx(ancillas[0], qubits[(n - 1) as usize])?;
    for i in (0..n - 1).rev() {
        uz(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }

    for i in 0..n - 1 {
        circuit.x(ancillas[(i + 1) as usize])?;
    }

    // This implements U^{n}_{z+y+x} in Fig.19 and Fig.23 from the supplementary material for [1].
    for i in 0..n - 1 {
        ux(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }
    circuit.cx(ancillas[0], qubits[(n - 1) as usize])?;
    for i in (0..n - 1).rev() {
        uz(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }
    for i in 0..n - 1 {
        circuit.x(ancillas[(i + 1) as usize])?;
    }

    circuit.x(qubits[(n - 1) as usize])?;
    circuit.x(ancillas[0])?;
    for q in qubits.iter() {
        circuit.cx(ancillas[0], *q)?;
    }
    circuit.x(ancillas[0])?;

    Ok(circuit)
}

/// Synthesize the :math:`n`-qubit increment gate using :math:`n` dirty ancilla qubits.
///
/// The construction appears in Fig. 10 in the main paper [1].
///
/// Best suitable for when n is small.
fn increment_n_dirty_small(n: u32) -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(2 * n, 0, 0, Param::Float(0.0))?;

    for k in (1..n).rev() {
        let k_mcx = synth_mcx_n_dirty_i15(k as usize, false, false)?;
        let k_mcx_qubits: Vec<Qubit> = (0..k + 1).chain(n + 1..2 * n).map(Qubit).collect();
        circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;
    }
    circuit.x(0)?;

    Ok(circuit)
}

/// Synthesize the :math:`n`-qubit increment gate using :math:`n` dirty ancilla qubits.
///
/// # Arguments
/// - :math:`n`: the number of qubits in the increment gate.
///
/// # Returns
///
/// A quantum circuit with :math:`2 * n` qubits.
fn increment_n_dirty(n: u32) -> PyResult<CircuitData> {
    if n <= 10 {
        increment_n_dirty_small(n)
    } else {
        increment_n_dirty_large(n)
    }
}

/// Synthesize a relative MCX gate without any auxiliary qubits.
///
/// The construction appears as Fig. 10 in the main paper [1].
///
/// Best suitable for when `num_controls` is small.
fn synth_relative_mcx(num_controls: usize) -> Result<CircuitData, CircuitDataError> {
    let num_qubits = (num_controls + 1) as u32;
    let target = num_controls as u32;
    let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;

    match num_controls {
        0 => {
            panic!("synth_relative_mcx requires at least 1 control qubit.");
        }
        1 => {
            circuit.cx(0, 1)?;
        }
        2 => {
            circuit.rccx(0, 1, 2)?;
        }
        3.. => {
            // splits the control qubits into 3 blocks of approximately equal size
            let num3 = num_controls / 3;
            let num2 = (num_controls - num3) / 2;
            let num1 = num_controls - num3 - num2;

            let qubits1: Vec<Qubit> = (0..num1 as u32)
                .chain(std::iter::once(target))
                .map(Qubit)
                .collect();
            let qubits2: Vec<Qubit> = ((num1 as u32)..(num1 + num2) as u32)
                .chain(std::iter::once(target))
                .map(Qubit)
                .collect();
            let qubits3: Vec<Qubit> = ((num1 + num2) as u32..num_controls as u32)
                .chain(std::iter::once(target))
                .map(Qubit)
                .collect();

            let circuit1 = synth_relative_mcx(num1)?;
            let circuit2 = synth_relative_mcx(num2)?;
            let circuit3 = synth_relative_mcx(num3)?;

            circuit.h(target)?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit2, &qubits2, &[])?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit1, &qubits1, &[])?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit2, &qubits2, &[])?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit1, &qubits1, &[])?;
            circuit.h(target)?;
        }
    }
    Ok(circuit)
}

/// Synthesize a relative MCX gate using up to `num_controls` dirty ancilla qubits.
fn synth_relative_mcx_n_dirty(num_controls: usize) -> Result<CircuitData, CircuitDataError> {
    // For small values of num_controls, it is more efficient to use a relative MCX
    // gate that does not require any auxiliary qubits, while for large values it is
    // not efficient to construct the true MCX gate that uses num_controls ancillas.
    // An interesting question is whether there are relative-MCX implementations that
    // use ancilla qubits.
    if num_controls < 11 {
        synth_relative_mcx(num_controls)
    } else {
        synth_mcx_n_dirty_i15(num_controls, false, false)
    }
}

/// Synthesize the :math:`n`-qubit increment/decrement gate using :math:`1` dirty
/// ancilla qubit for odd values of :math:`n`.
///
/// Returns an error when :math:`n` is even.
///
/// # Arguments
/// - :math:`n`: the number of qubits in the increment/decrement gate.
/// - flag_add: whether to increment by 1 or to decrement by 1.
///
/// # Returns
///
/// A quantum circuit with :math:`n+1` qubits.
fn increment_1_dirty(n: u32, flag_add: bool) -> PyResult<CircuitData> {
    if n.is_multiple_of(2) {
        return Err(QiskitError::new_err(
            "increment_1_dirty_large requires an odd number of qubits.",
        ));
    }

    let k = n.div_ceil(2);

    // This construction is described in Fig. 6 of [1].
    //
    // For n = 2k-1 odd, it reduces an n-increment gate with 1 ancilla to
    // * three k-increment gates that can use up to k ancillas, and
    // * two relative-MCX(k) gates that can use up to k-1 ancillas.

    // We say that qubits 0..k are "first half" qubits, qubits k+1..n are "second-half" qubits.
    let ancilla = n;

    let mut circuit = CircuitData::with_capacity(n + 1, 0, 0, Param::Float(0.0))?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    let k_incrementer = increment_n_dirty(k)?;

    // The first two instances of k_incrementer are defined over ancilla + "second-half" qubits,
    // and use "first-half" qubits as own ancillas
    let k12_incrementer_qubits: Vec<Qubit> = std::iter::once(ancilla)
        .chain(k..n)
        .chain(0..k)
        .map(Qubit)
        .collect();

    circuit.compose(&k_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla)?;

    for q in k..n {
        circuit.cx(ancilla, q)?;
    }

    let k_mcx = synth_relative_mcx_n_dirty(k as usize)?;

    // The two instances of the relative MCX gate have "first half" qubits as controls, ancilla as target, and
    // "second-half qubits" as own ancillas
    let k_mcx_qubits: Vec<Qubit> = (0..k)
        .chain(std::iter::once(ancilla))
        .chain(k..n)
        .map(Qubit)
        .collect();
    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    circuit.compose(&k_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla)?;

    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    for q in k..n {
        circuit.cx(ancilla, q)?;
    }

    let k3_incrementer = increment_n_dirty(k)?;

    // The instance of k3_incrementer is defined over "first-half" qubits,
    // and use "first-half" qubits as own ancillas
    let k3_incrementer_qubits: Vec<Qubit> = (0..k)
        .chain(k..n)
        .chain(std::iter::once(ancilla))
        .map(Qubit)
        .collect();

    circuit.compose(&k3_incrementer, &k3_incrementer_qubits, &[])?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    Ok(circuit)
}

/// Synthesize the :math:`n`-qubit increment/decrement gate using :math:`2` dirty
/// ancilla qubit both for odd and even values of :math:`n`.
///
/// Returns an error when :math:`n` is even.
///
/// # Arguments
/// - :math:`n`: the number of qubits in the increment/decrement gate.
/// - flag_add: whether to increment by 1 or to decrement by 1.
///
/// # Returns
///
/// A quantum circuit with :math:`n+2` qubits.
fn increment_2_dirty(n: u32, flag_add: bool) -> PyResult<CircuitData> {
    let k = (n + 2) / 2; // same as (n+1)/2 for n odd, and (n+2)/2 for n even

    // This construction is a slight modification of the construction described
    // in Fig. 6 of [1].
    //
    // For n = 2k-1 odd, it reduces an n-increment gate with 2 ancillas to
    // * three k-increment gates with k ancillas, and
    // * two relative-MCX(k) gates with k-1 ancillas;
    //
    // For n = 2k-2 even, it reduces an n-increment gate with 2 ancillas to
    // * two (k-1)-increment gates with k ancillas,
    // * one k-increment gate with k-1 ancillas, and
    // * two relative-MCX gates with k-1 ancillas.

    // We say that qubits 0..k are "first half" qubits, qubits k+1..n are "second-half" qubits.
    let ancilla1 = n;
    let ancilla2 = n + 1;

    let mut circuit = CircuitData::with_capacity(n + 2, 0, 0, Param::Float(0.0))?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    let k12_incrementer = increment_n_dirty(1 + n - k)?;

    // The two instances of k12_incrementer are defined over ancilla1 + "second-half" qubits,
    // and use "first-half" qubits + ancilla2 as own ancillas
    let k12_incrementer_qubits: Vec<Qubit> = std::iter::once(ancilla1)
        .chain(k..n)
        .chain(0..k)
        .chain(std::iter::once(ancilla2))
        .map(Qubit)
        .collect();

    circuit.compose(&k12_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla1)?;

    for q in k..n {
        circuit.cx(ancilla1, q)?;
    }

    let k_mcx = synth_relative_mcx_n_dirty(k as usize)?;

    // The two instances of the relative MCX gate have "first half" qubits as controls, ancilla1 as target, and
    // "second-half qubits" + ancilla2 as own ancillas
    let k_mcx_qubits: Vec<Qubit> = (0..k)
        .chain(std::iter::once(ancilla1))
        .chain(k..n)
        .chain(std::iter::once(ancilla2))
        .map(Qubit)
        .collect();
    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    circuit.compose(&k12_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla1)?;

    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    for q in k..n {
        circuit.cx(ancilla1, q)?;
    }

    let k3_incrementer = increment_n_dirty(k)?;

    // The instance of k3_incrementer is defined over "first-half" qubits,
    // and use "first-half" qubits + ancilla1 + ancilla2 as own ancillas
    let k3_incrementer_qubits: Vec<Qubit> = (0..k)
        .chain(k..n)
        .chain(std::iter::once(ancilla1))
        .chain(std::iter::once(ancilla2))
        .map(Qubit)
        .collect();

    circuit.compose(&k3_incrementer, &k3_incrementer_qubits, &[])?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    Ok(circuit)
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on the paper
/// by Huang and Palsberg [1].
///
/// # Arguments
///
/// - num_controls: the number of control qubits.
///
/// # Returns
///
/// A quantum circuit with :math:`k + 1` qubits.
/// The number of CX-gates is linear in :math:`k`.
///
/// # References:
///
/// 1. Huang and Palsberg, *Compiling Conditional Quantum Gates without Using Helper Qubits*, PLDI (2024),
///    https://dl.acm.org/doi/10.1145/3656436.
pub fn synth_mcx_noaux_hp24(num_controls: usize) -> PyResult<CircuitData> {
    let n = num_controls + 1;

    let mut circuit = CircuitData::with_capacity(n as u32, 0, 0, Param::Float(0.0))?;

    // Handle small cases explicitly
    if n == 1 {
        circuit.x(0)?;
    } else if n == 2 {
        circuit.cx(0, 1)?;
    } else {
        circuit.h(num_controls as u32)?;

        // The construction described in Fig.7 of the paper only works for even values of n.
        // The construction described in Fig.8 of the paper works for all values of n and is better than the one
        // in Fig.7 when n<23.

        if n.is_multiple_of(2) && (n >= 23) {
            // This implements C^{n-1}(V) in Fig.7.

            // These implement U^{n-1}_{+1} and U^{n-1}_{-1} (last qubit is ancilla)
            let increment_plus_1 = increment_1_dirty((n - 1) as u32, true)?;
            let increment_minus_1 = increment_1_dirty((n - 1) as u32, false)?;
            let increment_qubits: Vec<Qubit> = (0..n).map(|q| Qubit(q as u32)).collect();

            circuit.compose(&increment_plus_1, &increment_qubits, &[])?;

            // This implements U^{n-1}_{V_{n-1}}(-1)
            let mut phi = -PI;
            for q in (1..n - 1).rev() {
                phi /= 2.;
                circuit.cp(phi, q as u32, (n - 1) as u32)?;
            }

            circuit.compose(&increment_minus_1, &increment_qubits, &[])?;

            // This implements U^{n-1}_{V_{n-1}}(-1)
            let mut phi = PI;
            for q in (1..n - 1).rev() {
                phi /= 2.;
                circuit.cp(phi, q as u32, (n - 1) as u32)?;
            }

            circuit.cp(phi, 0, (n - 1) as u32)?;
        } else {
            // This implements C^{n-1}(V) in Fig.8.

            // These implement U^{n-1}_{+1} and U^{n-1}_{-1} (last qubit is ancilla)
            let increment_plus_1 = increment_2_dirty((n - 2) as u32, true)?;
            let increment_minus_1 = increment_2_dirty((n - 2) as u32, false)?;
            let increment_qubits: Vec<Qubit> = (0..n).map(|q| Qubit(q as u32)).collect();

            circuit.compose(&increment_plus_1, &increment_qubits, &[])?;

            // This implements C(U^{n-2}_{V_{n-2}}(-1)) in Fig.8.
            let mut phi = -PI;
            for q in (1..n - 2).rev() {
                phi /= 2.;
                circuit.ccp(phi, q as u32, (n - 2) as u32, (n - 1) as u32)?;
            }

            circuit.compose(&increment_minus_1, &increment_qubits, &[])?;

            // This implements C(U^{n-2}_{V_{n-2}}(1)) in Fig.8.
            let mut phi = PI;
            for q in (1..n - 2).rev() {
                phi /= 2.;
                circuit.ccp(phi, q as u32, (n - 2) as u32, (n - 1) as u32)?;
            }

            circuit.ccp(phi, 0, (n - 2) as u32, (n - 1) as u32)?;
        }

        circuit.h(num_controls as u32)?;
    }

    Ok(circuit)
}

#[cfg(all(test, not(miri)))]
mod test {
    use crate::matrix::sim::sim_unitary_circuit;
    use approx::abs_diff_eq;

    use super::{increment_n_dirty_large, increment_n_dirty_small};

    #[test]
    fn test_increment_n_dirty() {
        // Check that both methods to implement the :math:`n`-qubit increment gate using
        // :math:`n` dirty ancilla qubits produce the same matrix (for small number of qubits).
        for nq in 1..4 {
            let circuit1 = increment_n_dirty_small(nq).unwrap();
            let mat1 = sim_unitary_circuit(&circuit1).unwrap();

            let circuit2 = increment_n_dirty_large(nq).unwrap();
            let mat2 = sim_unitary_circuit(&circuit2).unwrap();

            const EPS: f64 = 1e-10;
            let close = abs_diff_eq!(mat1, mat2, epsilon = EPS);
            assert!(close);
        }
    }
}
