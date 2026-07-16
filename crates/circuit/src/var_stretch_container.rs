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

use crate::classical::expr;
use crate::object_registry::ObjectRegistry;
use crate::{Stretch, Var};
use qiskit_util::IndexMap;
use thiserror::Error;

use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;
use pyo3::{IntoPyObjectExt, prelude::*};

/// The scope type associated with a given variable in the container.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum VarType {
    Input = 0,
    Capture = 1,
    Declare = 2,
}

/// The scope type associated with a given stretch in the container.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum StretchType {
    Capture = 0,
    Declare = 1,
}

type VarStretchState = (Vec<(String, Py<PyAny>)>, Vec<expr::Var>, Vec<expr::Stretch>);

/// Errors related to [VarStretchContainer] which may occur when adding
/// variables or stretches to the container.
#[derive(Error, Debug)]
pub enum VarStretchContainerError {
    #[error("already present in the circuit")]
    DuplicateStretch,
    #[error("already present in the circuit")]
    DuplicateVar,
    #[error("cannot add var as its name shadows an existing identifier")]
    VarShadowing,
    #[error("cannot add variables that wrap `Clbit` or `ClassicalRegister` instances")]
    VarWrappingClbit,
    #[error("circuits with input variables cannot be enclosed, so they cannot be closures")]
    CaptureWithInputVars,
    #[error("circuits to be enclosed with captures cannot have input variables")]
    InputWithCaptureVars,
    #[error("cannot add stretch as its name shadows an existing identifier")]
    StretchShadowing,
}

/// A container for variables and stretches used by [`crate::circuit_data::CircuitData`]
/// and [`crate::dag_circuit::DAGCircuit`].
///
/// Variables (and stretches) are stored in the `vars` (and `stretches`) fields as
/// [`expr::Var`] (or [`expr::Stretch`]) objects, together with their corresponding
/// [`Var`] (or [`Stretch`]) indices, using [`ObjectRegistry`] fields.
///
/// The `identifier_info` field maintains a mapping from variable and stretch names to
/// an enum that encodes the type and index of each identifier in the container.
///
/// The indices for each type of variable and stretch are stored in the `var_indices`
/// and `stretch_indices` fields, respectively, enabling efficient iteration and
/// size queries for each type.
#[derive(Clone, Debug)]
pub struct VarStretchContainer {
    // Variables registered in the container
    vars: ObjectRegistry<Var, expr::Var>,
    // Stretches registered in the container
    stretches: ObjectRegistry<Stretch, expr::Stretch>,
    // Variable identifiers, in order of their addition to the container
    identifier_info: IndexMap<String, IdentifierInfo>,

    // Var indices stored in the container, in order of insertion for each type
    var_indices: [Vec<Var>; 3],

    // Stretch indices stored in the container, in order of insertion for each type
    stretch_indices: [Vec<Stretch>; 2],
}

impl VarStretchContainer {
    pub fn new() -> Self {
        VarStretchContainer {
            vars: ObjectRegistry::new(),
            stretches: ObjectRegistry::new(),
            identifier_info: IndexMap::default(),
            var_indices: [Vec::new(), Vec::new(), Vec::new()],
            stretch_indices: [Vec::new(), Vec::new()],
        }
    }

    pub fn with_capacity(num_vars: Option<usize>, num_stretches: Option<usize>) -> Self {
        let num_vars = num_vars.unwrap_or_default();
        let num_stretches = num_stretches.unwrap_or_default();
        VarStretchContainer {
            vars: ObjectRegistry::with_capacity(num_vars),
            stretches: ObjectRegistry::with_capacity(num_stretches),
            identifier_info: IndexMap::with_capacity_and_hasher(
                num_vars + num_stretches,
                foldhash::fast::RandomState::default(),
            ),
            var_indices: [Vec::new(), Vec::new(), Vec::new()],
            stretch_indices: [Vec::new(), Vec::new()],
        }
    }

    /// Creates a clone of this container, converting all contained variables and stretches into captures.
    pub fn clone_as_captures(&self) -> Self {
        let mut res = VarStretchContainer {
            vars: ObjectRegistry::with_capacity(self.vars.len()),
            stretches: ObjectRegistry::with_capacity(self.stretches.len()),
            identifier_info: IndexMap::with_capacity_and_hasher(
                self.identifier_info.len(),
                foldhash::fast::RandomState::default(),
            ),
            var_indices: [Vec::new(), Vec::with_capacity(self.vars.len()), Vec::new()],
            stretch_indices: [Vec::with_capacity(self.stretches.len()), Vec::new()],
        };

        for var in self.vars.objects() {
            res.add_var(var.clone(), VarType::Capture)
                .expect("Self should be in a consistent state");
        }
        for stretch in self.stretches.objects() {
            res.add_stretch(stretch.clone(), StretchType::Capture)
                .expect("Self should be in a consistent state");
        }

        res
    }

    /// Returns an immutable view of the variables stored in the container.
    pub fn vars(&self) -> &ObjectRegistry<Var, expr::Var> {
        &self.vars
    }

    /// Returns an immutable view of the stretches stored in the container.
    pub fn stretches(&self) -> &ObjectRegistry<Stretch, expr::Stretch> {
        &self.stretches
    }

    /// Adds a new [expr::Var] to the container.
    ///
    /// # Arguments:
    ///
    /// * var: the new variable to add.
    /// * var_type: the type the variable should have in the container.
    ///
    /// # Returns:
    ///
    /// The [Var] index of the variable in the container.
    pub fn add_var(
        &mut self,
        var: expr::Var,
        var_type: VarType,
    ) -> Result<Var, VarStretchContainerError> {
        let expr::Var::Standalone { name, .. } = &var else {
            return Err(VarStretchContainerError::VarWrappingClbit);
        };

        match self.identifier_info.get(name) {
            Some(IdentifierInfo::Var(info)) if Some(&var) == self.vars.get(info.var) => {
                return Err(VarStretchContainerError::DuplicateVar);
            }
            Some(_) => {
                return Err(VarStretchContainerError::VarShadowing);
            }
            _ => {}
        }

        match var_type {
            VarType::Input
                if self.num_vars(VarType::Capture) > 0
                    || self.num_stretches(StretchType::Capture) > 0 =>
            {
                return Err(VarStretchContainerError::InputWithCaptureVars);
            }
            VarType::Capture if self.num_vars(VarType::Input) > 0 => {
                return Err(VarStretchContainerError::CaptureWithInputVars);
            }
            _ => {}
        }

        let name = name.clone();
        let idx = self
            .vars
            .add(var)
            .expect("Var should not exist in the registry");
        self.var_indices[var_type as usize].push(idx);

        self.identifier_info.insert(
            name,
            IdentifierInfo::Var(VarInfo {
                var: idx,
                type_: var_type,
            }),
        );

        Ok(idx)
    }

    /// Adds a new [expr::Stretch] to the container.
    ///
    /// # Arguments:
    ///
    /// * stretch: the new stretch to add.
    /// * stretch_type: the type the stretch should have in the .
    ///
    /// # Returns:
    ///
    /// The [Stretch] index of the stretch in the container.
    pub fn add_stretch(
        &mut self,
        stretch: expr::Stretch,
        stretch_type: StretchType,
    ) -> Result<Stretch, VarStretchContainerError> {
        match self.identifier_info.get(&stretch.name) {
            Some(IdentifierInfo::Stretch(info))
                if Some(&stretch) == self.stretches.get(info.stretch) =>
            {
                return Err(VarStretchContainerError::DuplicateStretch);
            }
            Some(_) => {
                return Err(VarStretchContainerError::StretchShadowing);
            }
            _ => {}
        }

        if let StretchType::Capture = stretch_type {
            if self.num_vars(VarType::Input) > 0 {
                return Err(VarStretchContainerError::CaptureWithInputVars);
            }
        }

        let name = stretch.name.clone();
        let idx = self
            .stretches
            .add(stretch)
            .expect("Stretch should not exist in the registry");
        self.stretch_indices[stretch_type as usize].push(idx);

        self.identifier_info.insert(
            name,
            IdentifierInfo::Stretch(StretchInfo {
                stretch: idx,
                type_: stretch_type,
            }),
        );

        Ok(idx)
    }

    /// Returns the [expr::Var] object corresponding to the specified name, or None if no such variable exists in the container.
    #[inline]
    pub fn get_var(&self, name: &str) -> Option<&expr::Var> {
        if let Some(IdentifierInfo::Var(var)) = self.identifier_info.get(name) {
            self.vars.get(var.var)
        } else {
            None
        }
    }

    /// Returns the [expr::Stretch] object corresponding to the specified name, or None if no such stretch exists in the container.
    #[inline]
    pub fn get_stretch(&self, name: &str) -> Option<&expr::Stretch> {
        if let Some(IdentifierInfo::Stretch(stretch)) = self.identifier_info.get(name) {
            self.stretches.get(stretch.stretch)
        } else {
            None
        }
    }

    /// Returns an iterator over the contained [expr::Var] objects of the specified variable type.
    pub fn iter_vars(&self, var_type: VarType) -> impl ExactSizeIterator<Item = &expr::Var> {
        self.var_indices[var_type as usize].iter().map(|idx| {
            self.vars
                .get(*idx)
                .expect("A variable with this index should be registered")
        })
    }

    /// Returns an iterator over the contained [expr::Stretch] objects of the specified stretch type.
    pub fn iter_stretches(
        &self,
        stretch_type: StretchType,
    ) -> impl ExactSizeIterator<Item = &expr::Stretch> {
        self.stretch_indices[stretch_type as usize]
            .iter()
            .map(|idx| {
                self.stretches
                    .get(*idx)
                    .expect("A stretch with this index should be registered")
            })
    }

    /// Returns the number of contained variables of the specified type.
    #[inline(always)]
    pub fn num_vars(&self, var_type: VarType) -> usize {
        self.var_indices[var_type as usize].len()
    }

    /// Returns the number of contained stretches of the specified type.
    #[inline(always)]
    pub fn num_stretches(&self, stretch_type: StretchType) -> usize {
        self.stretch_indices[stretch_type as usize].len()
    }

    /// Returns the number of contained identifiers, i.e. total vars and stretches of all types.
    #[inline(always)]
    pub fn num_identifiers(&self) -> usize {
        self.identifier_info.len()
    }

    /// Returns `true` if a variable or a stretch with the specified name exists in the container.
    #[inline(always)]
    pub fn has_identifier(&self, name: &str) -> bool {
        self.identifier_info.contains_key(name)
    }

    /// Returns `true` is a variable with the specified name exists in the container.
    #[inline(always)]
    pub fn has_var(&self, name: &str) -> bool {
        matches!(self.identifier_info.get(name), Some(IdentifierInfo::Var(_)))
    }

    /// Returns `true` is a variable with the specified name and type exists in the container.
    #[inline(always)]
    pub fn has_var_by_type(&self, name: &str, var_type: VarType) -> bool {
        if let Some(IdentifierInfo::Var(info)) = self.identifier_info.get(name) {
            info.type_ == var_type
        } else {
            false
        }
    }

    /// Returns `true` is a stretch with the specified name exists in the container.
    #[inline(always)]
    pub fn has_stretch(&self, name: &str) -> bool {
        matches!(
            self.identifier_info.get(name),
            Some(IdentifierInfo::Stretch(_))
        )
    }

    /// Returns `true` is a stretch with the specified name and type exists in the container.
    #[inline(always)]
    pub fn has_stretch_by_type(&self, name: &str, stretch_type: StretchType) -> bool {
        if let Some(IdentifierInfo::Stretch(info)) = self.identifier_info.get(name) {
            info.type_ == stretch_type
        } else {
            false
        }
    }

    /// Checks whether this container is structurally equal to the other container.
    ///
    /// Two `VarStretchContainer`s are considered structurally equal if and only if
    /// their `vars`, `stretches` and `identifier_info` fields are identical.
    pub fn structurally_equal(&self, other: &VarStretchContainer) -> bool {
        if self.vars != other.vars || self.stretches != other.stretches {
            return false;
        }

        self.identifier_info
            .iter()
            .zip(other.identifier_info.iter())
            .all(|(i1, i2)| i1 == i2)
    }

    /// Returns cloned copies of identifier info, variable objects and stretch objects.
    pub fn to_pickle(&self, py: Python) -> VarStretchState {
        (
            self.identifier_info
                .iter()
                .map(|(k, v)| (k.clone(), v.clone().to_pickle(py).unwrap()))
                .collect::<Vec<(String, Py<PyAny>)>>(),
            self.vars.objects().clone(),
            self.stretches.objects().clone(),
        )
    }

    /// Constructs Self given identifier info, variables and stretch objects previously serialized with [Self::to_pickle()]
    pub fn from_pickle(py: Python, state: VarStretchState) -> PyResult<Self> {
        let mut res = VarStretchContainer::with_capacity(Some(state.1.len()), Some(state.2.len()));

        for identifier_info in state.0 {
            let id_info = IdentifierInfo::from_pickle(identifier_info.1.bind(py))?;
            match &id_info {
                IdentifierInfo::Stretch(stretch_info) => {
                    res.stretch_indices[stretch_info.type_ as usize].push(stretch_info.stretch);
                }
                IdentifierInfo::Var(var_info) => {
                    res.var_indices[var_info.type_ as usize].push(var_info.var);
                }
            }

            res.identifier_info.insert(identifier_info.0, id_info);
        }

        for var in state.1 {
            res.vars.add(var)?;
        }

        for stretch in state.2 {
            res.stretches.add(stretch)?;
        }

        Ok(res)
    }
}

impl Default for VarStretchContainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Two `VarStretchContainer` containers are considered equal if and only if they contain the same
/// variables and stretches, and the declaration order of declared stretches matches in both containers.
impl PartialEq for VarStretchContainer {
    fn eq(&self, other: &Self) -> bool {
        if self.num_vars(VarType::Input) != other.num_vars(VarType::Input)
            || self.num_vars(VarType::Capture) != other.num_vars(VarType::Capture)
            || self.num_vars(VarType::Declare) != other.num_vars(VarType::Declare)
            || self.num_stretches(StretchType::Capture) != other.num_stretches(StretchType::Capture)
            || self.num_stretches(StretchType::Declare) != other.num_stretches(StretchType::Declare)
        {
            return false;
        }

        let mut prev_rhs_stretch_idx = 0usize;
        for (id_name, lhs_id_info) in &self.identifier_info {
            let Some(rhs_id_info) = other.identifier_info.get(id_name) else {
                return false; // Identifier does not exist in other
            };

            match (lhs_id_info, rhs_id_info) {
                (IdentifierInfo::Var(lhs_var_info), IdentifierInfo::Var(rhs_var_info)) => {
                    if lhs_var_info.type_ != rhs_var_info.type_
                        || !other
                            .vars
                            .contains(self.vars.get(lhs_var_info.var).unwrap())
                    {
                        return false; // Not the same var type or UUID
                    }
                }
                (
                    IdentifierInfo::Stretch(lhs_stretch_info),
                    IdentifierInfo::Stretch(rhs_stretch_info),
                ) => {
                    if lhs_stretch_info.type_ != rhs_stretch_info.type_
                        || !other
                            .stretches
                            .contains(self.stretches.get(lhs_stretch_info.stretch).unwrap())
                    {
                        return false; // Not the same stretch type or UUID
                    };

                    // Check whether the declared stretches in the other container follow the same order of
                    // declaration as in self. This is done by verifying that the indices of the declared stretches
                    // in `identifier_info` of the other container - which match the stretches encountered during the
                    // iteration here - are monotonically increasing.
                    if let StretchType::Declare = rhs_stretch_info.type_ {
                        let rhs_stretch_idx = other.identifier_info.get_index_of(id_name).unwrap();
                        if rhs_stretch_idx < prev_rhs_stretch_idx {
                            return false;
                        }
                        prev_rhs_stretch_idx = rhs_stretch_idx;
                    }
                }
                _ => {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq)]
struct VarInfo {
    var: Var,
    type_: VarType,
}

impl VarInfo {
    fn to_pickle(&self, py: Python) -> PyResult<Py<PyAny>> {
        (self.var.0, self.type_ as u8).into_py_any(py)
    }

    fn from_pickle(ob: &Bound<PyAny>) -> PyResult<Self> {
        let val_tuple = ob.cast::<PyTuple>()?;
        Ok(VarInfo {
            var: Var(val_tuple.get_item(0)?.extract()?),
            type_: match val_tuple.get_item(1)?.extract::<u8>()? {
                0 => VarType::Input,
                1 => VarType::Capture,
                2 => VarType::Declare,
                _ => return Err(PyValueError::new_err("Invalid var type")),
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
struct StretchInfo {
    stretch: Stretch,
    type_: StretchType,
}

impl StretchInfo {
    fn to_pickle(&self, py: Python) -> PyResult<Py<PyAny>> {
        (self.stretch.0, self.type_ as u8).into_py_any(py)
    }

    fn from_pickle(ob: &Bound<PyAny>) -> PyResult<Self> {
        let val_tuple = ob.cast::<PyTuple>()?;
        Ok(StretchInfo {
            stretch: Stretch(val_tuple.get_item(0)?.extract()?),
            type_: match val_tuple.get_item(1)?.extract::<u8>()? {
                0 => StretchType::Capture,
                1 => StretchType::Declare,
                _ => return Err(PyValueError::new_err("Invalid stretch type")),
            },
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
enum IdentifierInfo {
    Stretch(StretchInfo),
    Var(VarInfo),
}

impl IdentifierInfo {
    fn to_pickle(&self, py: Python) -> PyResult<Py<PyAny>> {
        match self {
            IdentifierInfo::Stretch(info) => (0, info.to_pickle(py)?).into_py_any(py),
            IdentifierInfo::Var(info) => (1, info.to_pickle(py)?).into_py_any(py),
        }
    }

    fn from_pickle(ob: &Bound<PyAny>) -> PyResult<Self> {
        let val_tuple = ob.cast::<PyTuple>()?;
        match val_tuple.get_item(0)?.extract::<u8>()? {
            0 => Ok(IdentifierInfo::Stretch(StretchInfo::from_pickle(
                &val_tuple.get_item(1)?,
            )?)),
            1 => Ok(IdentifierInfo::Var(VarInfo::from_pickle(
                &val_tuple.get_item(1)?,
            )?)),
            _ => Err(PyValueError::new_err("Invalid identifier info type")),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        bit::{ClassicalRegister, ShareableClbit},
        classical::types::Type,
    };
    use uuid::Uuid;

    fn new_var(name: &str) -> expr::Var {
        expr::Var::Standalone {
            uuid: Uuid::new_v4().as_u128(),
            name: name.to_string(),
            ty: Type::Bool,
        }
    }

    fn new_stretch(name: &str) -> expr::Stretch {
        expr::Stretch {
            uuid: Uuid::new_v4().as_u128(),
            name: name.to_string(),
        }
    }

    #[test]
    fn test_type_variants_order() {
        // Make sure the numerical values of the type variants are kept static as they are used for slice indexing.
        // Currently changing the variants order will only affect performance-wise the [clone_as_captures]
        // function as it assumes specific indices for the captured vars and stretches vectors inside var_indices and
        // var_stretches, respectively.
        assert_eq!(VarType::Input as usize, 0);
        assert_eq!(VarType::Capture as usize, 1);
        assert_eq!(VarType::Declare as usize, 2);
        assert_eq!(StretchType::Capture as usize, 0);
        assert_eq!(StretchType::Declare as usize, 1);
    }

    #[test]
    fn test_clone_as_captures1() -> Result<(), VarStretchContainerError> {
        // Test that all variable/stretch types are converted to captures
        let mut container = VarStretchContainer::new();
        container.add_var(new_var("in1"), VarType::Input)?;
        container.add_var(new_var("v1"), VarType::Declare)?;
        container.add_stretch(new_stretch("s1"), StretchType::Declare)?;

        let cloned = container.clone_as_captures();

        assert_eq!(cloned.num_vars(VarType::Capture), 2);
        assert_eq!(cloned.num_vars(VarType::Declare), 0);
        assert_eq!(cloned.num_vars(VarType::Input), 0);

        assert_eq!(cloned.num_stretches(StretchType::Capture), 1);
        assert_eq!(cloned.num_stretches(StretchType::Declare), 0);

        Ok(())
    }

    #[test]
    fn test_clone_as_captures2() -> Result<(), VarStretchContainerError> {
        // Test that all variable/stretch types are converted to captures
        let mut container = VarStretchContainer::new();
        container.add_var(new_var("v1"), VarType::Declare)?;
        container.add_var(new_var("v2"), VarType::Capture)?;
        container.add_stretch(new_stretch("s1"), StretchType::Declare)?;
        container.add_stretch(new_stretch("s2"), StretchType::Capture)?;

        let cloned = container.clone_as_captures();

        assert_eq!(cloned.num_vars(VarType::Capture), 2);
        assert_eq!(cloned.num_vars(VarType::Declare), 0);
        assert_eq!(cloned.num_vars(VarType::Input), 0);

        assert_eq!(cloned.num_stretches(StretchType::Capture), 2);
        assert_eq!(cloned.num_stretches(StretchType::Declare), 0);

        Ok(())
    }

    #[test]
    fn test_add_var() -> Result<(), VarStretchContainerError> {
        // Test name and scope checking logic in add_var
        let mut container = VarStretchContainer::new();

        // Cannot add Clbit
        assert!(
            container
                .add_var(
                    expr::Var::Bit {
                        bit: ShareableClbit::new_anonymous(),
                    },
                    VarType::Capture,
                )
                .is_err()
        );

        // Cannot add ClassicalRegister
        assert!(
            container
                .add_var(
                    expr::Var::Register {
                        register: ClassicalRegister::new_owning("r1", 1),
                        ty: Type::Bool
                    },
                    VarType::Declare
                )
                .is_err()
        );

        container.add_var(new_var("in1"), VarType::Input)?;

        // Cannot add an already existing var
        assert!(container.add_var(new_var("in1"), VarType::Input).is_err());
        // Cannot shadow an existing identifier
        assert!(
            container
                .add_stretch(new_stretch("in1"), StretchType::Declare)
                .is_err()
        );

        // Cannot add captured vars if input vars already exist
        assert!(container.add_var(new_var("c1"), VarType::Capture).is_err());

        let mut container = container.clone_as_captures();

        // Cannot add input vars if there exist captured vars
        assert!(container.add_var(new_var("in2"), VarType::Input).is_err());

        Ok(())
    }

    #[test]
    fn test_add_strech() -> Result<(), VarStretchContainerError> {
        // Test name and scope checking logic in add_stretch
        let mut container = VarStretchContainer::new();
        container.add_var(new_var("in1"), VarType::Input)?;
        container.add_stretch(new_stretch("s1"), StretchType::Declare)?;

        // Cannot add a stretch which already exists
        assert!(
            container
                .add_stretch(new_stretch("s1"), StretchType::Declare)
                .is_err()
        );

        // Cannot add an identifier which shadows another identifier
        assert!(
            container
                .add_stretch(new_stretch("in1"), StretchType::Declare)
                .is_err()
        );

        // Cannot a captured stretch if inputs already exist
        assert!(
            container
                .add_stretch(new_stretch("s2"), StretchType::Capture)
                .is_err()
        );

        Ok(())
    }

    #[test]
    fn test_eq() -> Result<(), VarStretchContainerError> {
        // Test the PartialEq trait implementation
        let mut container1 = VarStretchContainer::new();
        let mut container2 = VarStretchContainer::new();

        container1.add_var(new_var("v1"), VarType::Input)?;
        assert_ne!(container1, container2); // not the same number of inputs vars

        container1 = VarStretchContainer::new();
        container1.add_var(new_var("v1"), VarType::Declare)?;
        assert_ne!(container1, container2); // not the same number of declared vars

        container1 = VarStretchContainer::new();
        container1.add_var(new_var("v1"), VarType::Capture)?;
        assert_ne!(container1, container2); // not the same number of captured vars

        container1 = VarStretchContainer::new();
        container1.add_stretch(new_stretch("s1"), StretchType::Capture)?;
        assert_ne!(container1, container2); // not the same number of captured stretches

        container1 = VarStretchContainer::new();
        container1.add_stretch(new_stretch("s1"), StretchType::Declare)?;
        assert_ne!(container1, container2); // not the same number of declared stretches

        container1.add_var(new_var("v1"), VarType::Capture)?;
        container1.add_var(new_var("v2"), VarType::Declare)?;
        container1.add_stretch(new_stretch("s2"), StretchType::Capture)?;
        assert_eq!(container1, container1.clone()); // trivial equality

        let v1 = new_var("v1");
        container1 = VarStretchContainer::new();
        container1.add_var(v1, VarType::Declare)?;
        container2 = VarStretchContainer::new();
        container2.add_var(new_var("v1"), VarType::Declare)?;
        assert_ne!(container1, container2); // not the same UUID for "v1"

        let s3 = new_stretch("s3");
        let s4 = new_stretch("s4");
        container2 = container1.clone();
        container1.add_stretch(s3.clone(), StretchType::Declare)?;
        container1.add_stretch(s4.clone(), StretchType::Declare)?;
        container2.add_stretch(s4, StretchType::Declare)?;
        container2.add_stretch(s3, StretchType::Declare)?;
        assert_ne!(container1, container2); // not the same order of declared stretches

        Ok(())
    }
}
