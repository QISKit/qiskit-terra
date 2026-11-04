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

use super::errors::TargetError;
use smallvec::SmallVec;

/// Model bounds on angle parameters for a gate
///
/// `None` represents no bound, while `Some([f64; 2])` represents an inclusive bound set
/// on the lower and upper allowed values respectively.
#[derive(Clone, Debug)]
pub(crate) struct AngleBound(SmallVec<[Option<[f64; 2]>; 3]>);

impl AngleBound {
    pub fn bounds(&self) -> &[Option<[f64; 2]>] {
        &self.0
    }

    pub fn new(bounds: SmallVec<[Option<[f64; 2]>; 3]>) -> Result<Self, TargetError> {
        for [low, high] in bounds.iter().flatten() {
            if low >= high {
                return Err(TargetError::InvalidBounds {
                    low: *low,
                    high: *high,
                });
            }
        }
        Ok(Self(bounds))
    }

    pub fn angles_supported(&self, angles: &[f64]) -> bool {
        angles
            .iter()
            .zip(&self.0)
            .all(|(angle, bound)| match bound {
                Some([low, high]) => !(angle < low || angle > high),
                None => true,
            })
    }
}
