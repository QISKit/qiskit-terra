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

use crate::data_tree::DataTree;
use crate::program::QuantumProgram;
use crate::tensor::Tensor;

pub struct RunResult {
    data: DataTree<Tensor>,
}

pub trait Job {
    type JobError;

    fn result(&self) -> Result<RunResult, Self::JobError>;
}

pub trait BackendV3 {
    type JobError;
    fn run(&self, program: &QuantumProgram) -> impl Job<JobError = Self::JobError>;
}
