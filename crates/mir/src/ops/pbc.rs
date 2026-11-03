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

//! This module contains operations that are used in Pauli-Based Computations.

use std::{f64, fmt};

///  Basic Pauli enum
///
/// TODO: Decide whether we want to use existing from Qiskit or define one
/// specifically for MIR. Currently I'm inclined to create a new enum here,
/// specifically for use in MIR, as we can add some helper functions here that
/// will be needed throughout MIR (e.g., determine commuting/anti-commuting
/// properties, and maybe some conversion functions to/from string). That said,
/// I have not yet looked at the existing implementation pwdin Qiskit to know
/// whether the same could be done with them.
#[derive(Debug, Copy, Clone, Default, PartialEq)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub enum Pauli {
    #[default]
    I,
    X,
    Z,
    Y,
}

impl fmt::Display for Pauli {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// New type for PauliStrings.
///
/// This is to be used primarily for operations in MIR. The current
/// implementation is meant to be quick and dirty to add the necessary
/// functionality. This will be optimized in future work (see [Issue
/// 17326](https://github.com/Qiskit/qiskit/issues/17326)).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct PauliString(Vec<Pauli>);

impl PauliString {
    pub fn new(paulis: Vec<Pauli>) -> Self {
        PauliString(paulis)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// A rotation angle in units of `rad / pi`.
///
/// # Storage
///
/// These are internally stored in units of `rad / pi`.  Explicitly, in conjunction with the `exp(-i
/// P angle / 2)` conventions of Pauli rotations, this means, for example:
///
/// * a stored value `1.0` is equivalent (up to phase) of an application of the Pauli.
/// * a stored value `4.0` is a _complete_ Pauli rotation (including phase)
/// * a stored value `0.25` is a T-like rotation.
///
/// The internal storage of [`f64`] makes integer multiples of binary fractions exact, and IEEE-754
/// requires the basic arithmetic operations to produce the float nearest to the "true" mathematical
/// answer.  Together, these mean that the Clifford rotations (and T rotations) are stored exactly,
/// and simple arithmetic on them is exact.  This is not the case with raw radians, which incur
/// round-off error due to the irrational `pi`.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct AngleType(f64);

impl AngleType {
    /// With the Pauli convention, the angle corresponding to an absolute identity rotation.
    pub const FULL_TURN: Self = Self(4.0);
    /// With the Pauli convention, the angle corresponding to an identity-and-phase rotation (i.e.
    /// `-I`).
    pub const HALF_TURN: Self = Self(2.0);
    /// With the Pauli convention, the angle corresponding to (up to phase) an application of a
    /// Pauli (like the `Z` gate).
    pub const PAULI_LIKE: Self = Self(1.0);
    /// With the Pauli convention, the angle corresponding to (up to phase) an application of a
    /// square-root Pauli (like the `S` gate with respect to `Z`).
    pub const S_LIKE: Self = Self(0.5);
    /// With the Pauli convention, the angle corresponding to (up to phase) an application of the
    /// primitive non-Clifford rotation (like the `T` gate with respect to `Z`).
    pub const T_LIKE: Self = Self(0.25);
    pub const ZERO: Self = Self(0.0);

    /// Create a new angle from "native" (`rad / pi`) units.
    pub const fn new(angle: f64) -> Self {
        AngleType(angle)
    }

    /// Create a new angle explicitly from radians.
    ///
    /// Note that this conversion incurs potential round-off limitations due to the inexact
    /// representation of `pi`; you should prefer to work with angles in the "native" units (see
    /// [`new`](Self::new)) if possible.
    pub const fn from_rad(rad: f64) -> Self {
        AngleType(rad * f64::consts::FRAC_1_PI)
    }

    /// Get the angle in native units.
    pub const fn get(&self) -> f64 {
        self.0
    }
    /// Get the angle in radians.
    ///
    /// Note that this incurs potential round-off limitations due to the inexact representation of
    /// `pi`.
    pub const fn as_rad(&self) -> f64 {
        self.0 * f64::consts::PI
    }
}

/// An arbitrary-qubit Pauli rotation.
///
/// This uses the convention `exp(-i P angle / 2)`, where `P` is the Pauli string and `angle` is the
/// value (in radians) of [`angle`](Self::angle).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "mem_dbg", derive(mem_dbg::MemSize, mem_dbg::MemDbg))]
pub struct PauliRotation {
    paulis: PauliString,
    angle: AngleType,
}

impl PauliRotation {
    pub fn new(paulis: PauliString, angle: AngleType) -> Self {
        Self { paulis, angle }
    }

    pub fn paulis(&self) -> &PauliString {
        &self.paulis
    }

    pub fn angle(&self) -> AngleType {
        self.angle
    }
}

impl fmt::Display for PauliRotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "pauli_rotation({:?}, {:?}", self.paulis, self.angle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_pauli_rotation() {
        let pauli_rotation = PauliRotation {
            paulis: PauliString(vec![Pauli::I; 8]),
            angle: AngleType(0.5),
        };

        assert_eq!(*pauli_rotation.paulis(), PauliString(vec![Pauli::I; 8]));
        assert_eq!(pauli_rotation.angle(), AngleType(0.5));
    }
}
