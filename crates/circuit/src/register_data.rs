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

use std::sync::OnceLock;
use std::{fmt::Debug, marker::PhantomData};

use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict, PyList};

use crate::{bit::Register, circuit_data::CircuitError};

/// Error thrown when adding a register using strict mode
/// and it's already present.
#[derive(Debug, thiserror::Error)]
#[error("register name \"{0}\" already exists")]
pub struct RegisterAlreadyExists(String);

impl From<RegisterAlreadyExists> for PyErr {
    fn from(value: RegisterAlreadyExists) -> Self {
        CircuitError::new_err(value.to_string())
    }
}

/// Represents the location in which the register is stored within [RegisterData].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RegisterIndex<R: Register> {
    index: u32,
    _marker: PhantomData<R>,
}

impl<R: Register + Clone> Copy for RegisterIndex<R> {}

impl<R: Register> From<usize> for RegisterIndex<R> {
    fn from(value: usize) -> Self {
        Self {
            index: value
                .try_into()
                .unwrap_or_else(|_| panic!("'{value}' is too big to be converted to u32")),
            _marker: PhantomData,
        }
    }
}

impl<R: Register> From<u32> for RegisterIndex<R> {
    fn from(value: u32) -> Self {
        Self {
            index: value,
            _marker: PhantomData,
        }
    }
}

impl<R: Register> RegisterIndex<R> {
    pub fn index(&self) -> usize {
        self.index as usize
    }
}

#[derive(Debug, Clone)]
pub struct RegisterData<R: Register> {
    reg_index: HashMap<String, RegisterIndex<R>>,
    registers: Vec<R>,
    cached_registers: OnceLock<Py<PyDict>>,
}

impl<R> Default for RegisterData<R>
where
    R: Debug + Clone + Register,
{
    fn default() -> Self {
        Self::new()
    }
}
impl<R: Register + PartialEq> PartialEq for RegisterData<R> {
    fn eq(&self, other: &Self) -> bool {
        (self.registers == other.registers) && (self.reg_index == other.reg_index)
    }
}
impl<R: Register + Eq> Eq for RegisterData<R> {}

impl<R> RegisterData<R>
where
    R: Debug + Clone + Register,
{
    /// Creates an empty instance of [RegisterData]
    pub fn new() -> Self {
        Self {
            reg_index: HashMap::new(),
            registers: Vec::new(),
            cached_registers: OnceLock::new(),
        }
    }

    /// Creates an empty instance of [RegisterData] with enough capacity
    /// for the specified registers.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            reg_index: HashMap::with_capacity(capacity),
            registers: Vec::new(),
            cached_registers: OnceLock::new(),
        }
    }

    /// Inserts a [Register] into the circuit. If it exists, the index of where
    /// it sits will be returned.
    ///
    /// _**Note**_ if `strict` is passed, the insertion will fail.
    pub fn add_register(
        &mut self,
        register: R,
        strict: bool,
    ) -> Result<bool, RegisterAlreadyExists> {
        if self
            .reg_index
            .try_insert(register.name().to_string(), self.registers.len().into())
            .is_ok()
        {
            self.registers.push(register);
            self.cached_registers.take();
            Ok(true)
        } else if strict {
            Err(RegisterAlreadyExists(register.name().to_string()))
        } else {
            Ok(false)
        }
    }

    /// Return the number of registers stored
    pub fn len(&self) -> usize {
        self.reg_index.len()
    }

    /// Check if this [RegisterData] instance is empty.
    pub fn is_empty(&self) -> bool {
        self.reg_index.is_empty()
    }

    /// Checks if a [Register] exists within the circuit
    pub fn contains(&self, register: &R) -> bool {
        self.reg_index.contains_key(register.name())
    }

    /// Checks if a [Register] exists within the circuit by name
    pub fn contains_key(&self, key: &str) -> bool {
        self.reg_index.contains_key(key)
    }

    /// Retrieves register by name, returns none if not found.
    pub fn get(&self, register: &str) -> Option<&R> {
        self.reg_index
            .get(register)
            .map(|idx| &self.registers[idx.index()])
    }

    /// Retrieves register by index, returns none if not found.
    pub fn get_index(&self, index: RegisterIndex<R>) -> Option<&R> {
        self.registers.get(index.index())
    }

    /// Removes register by name and returns it, returns `None` if it
    /// doesn't exist.
    ///
    /// __**Note:** This operation is performed at `O(n)` times in the worst case.__
    pub fn remove(&mut self, register: &str) -> Option<R> {
        self.cached_registers.take();
        if let Some(index) = self.reg_index.remove(register) {
            let bit = self.registers.remove(index.index());
            // Update indices.
            for i in index.index()..self.registers.len() {
                self.reg_index
                    .insert(self.registers[i].name().to_string(), i.into());
            }
            Some(bit)
        } else {
            None
        }
    }

    /// Removes the registers with the provided names. Skips registers that are
    /// absent from this instance.
    pub fn remove_registers<I>(&mut self, indices: I)
    where
        I: IntoIterator<Item = String>,
    {
        let mut indices_sorted: Vec<usize> = indices
            .into_iter()
            .filter_map(|i| self.reg_index.get(&i).map(|idx| idx.index()))
            .collect();
        indices_sorted.sort();
        self.cached_registers.take();
        for index in indices_sorted.into_iter().rev() {
            let bit = self.registers.remove(index);
            self.reg_index.remove(bit.name());
        }
        // Update indices.
        for (i, registers) in self.registers.iter().enumerate() {
            self.reg_index
                .insert(registers.name().to_string(), i.into());
        }
    }

    /// Returns a slice of all the [Register] instances.
    pub fn registers(&self) -> &[R] {
        &self.registers
    }

    /// Called during Python garbage collection, only!.
    /// Note: INVALIDATES THIS INSTANCE.
    pub fn dispose(&mut self) {
        self.reg_index.clear();
        self.registers.clear();
        self.cached_registers.take();
    }

    /// Create an instance of [RegisterData] from an existing mapping
    /// of `String` and the specified register type `R`.
    pub fn from_mapping<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = (String, R)>,
    {
        let mut reg_index = HashMap::new();
        let mut registers = Vec::new();
        for (index, (name, reg)) in iter.into_iter().enumerate() {
            registers.push(reg);
            reg_index.insert(name, index.into());
        }
        Self {
            registers,
            reg_index,
            cached_registers: OnceLock::new(),
        }
    }
}

impl<R> RegisterData<R>
where
    R: Debug + Clone + Register + for<'py> IntoPyObject<'py>,
{
    /// Returns the dictionary mapping the register names and the register instances.
    pub fn cached(&self, py: Python) -> &Py<PyDict> {
        self.cached_registers.get_or_init(|| {
            self.registers
                .iter()
                .map(|reg| (reg.name(), reg.clone()))
                .into_py_dict(py)
                .unwrap()
                .into()
        })
    }

    /// Return list of registers in the circuit.
    pub fn cached_list<'py>(&'py self, py: Python<'py>) -> Bound<'py, PyList> {
        self.cached_registers
            .get_or_init(|| {
                self.registers
                    .iter()
                    .map(|reg| (reg.name(), reg.clone()))
                    .into_py_dict(py)
                    .unwrap()
                    .into()
            })
            .bind(py)
            .values()
    }

    pub fn cached_raw(&self) -> Option<&Py<PyDict>> {
        self.cached_registers.get()
    }
}
