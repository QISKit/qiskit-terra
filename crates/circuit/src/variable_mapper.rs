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

use crate::bit::{ClassicalRegister, Register, ShareableClbit};
use crate::classical::expr;
use crate::operations::{Condition, SwitchTarget};
use hashbrown::{HashMap, HashSet};
use num_bigint::BigUint;
use num_traits::Num;
use std::cell::RefCell;
use std::error::Error;

pub(crate) struct VariableMapper {
    target_cregs: Vec<ClassicalRegister>,
    register_map: RefCell<HashMap<String, ClassicalRegister>>,
    bit_map: HashMap<ShareableClbit, ShareableClbit>,
    var_map: HashMap<expr::Var, expr::Var>,
    stretch_map: HashMap<expr::Stretch, expr::Stretch>,
}

impl VariableMapper {
    /// Constructs a new mapper.
    ///
    /// The `stretch_map` is only used for direct calls to [VariableMapper::map_expr]
    /// since `condition`s and `target`s expressions are never durations. Provide
    /// [None] if you don't need this.
    pub fn new(
        target_cregs: Vec<ClassicalRegister>,
        bit_map: HashMap<ShareableClbit, ShareableClbit>,
        var_map: HashMap<expr::Var, expr::Var>,
        stretch_map: Option<HashMap<expr::Stretch, expr::Stretch>>,
    ) -> Self {
        Self {
            target_cregs,
            register_map: RefCell::default(),
            bit_map,
            var_map,
            stretch_map: stretch_map.unwrap_or_default(),
        }
    }

    /// Map the given `condition` so that it only references variables in the destination
    /// circuit (as given to this class on initialization).
    ///
    /// If `allow_reorder` is `true`, then when a legacy condition (the two-tuple form) is made
    /// on a register that has a counterpart in the destination with all the same (mapped) bits but
    /// in a different order, then that register will be used and the value suitably modified to
    /// make the equality condition work.  This is maintaining legacy (tested) behavior of
    /// [DAGCircuit::compose]; nowhere else does this, and in general this would require *far*
    /// more complex classical rewriting than Qiskit needs to worry about in the full expression
    /// era.
    pub fn map_condition<F, E>(
        &self,
        condition: &Condition,
        allow_reorder: bool,
        mut add_register: F,
    ) -> Result<Condition, E>
    where
        F: FnMut(&ClassicalRegister) -> Result<(), E>,
        E: Error,
    {
        Ok(match condition {
            Condition::Bit(target, value) => {
                Condition::Bit(self.bit_map.get(target).unwrap().clone(), *value)
            }
            Condition::Expr(e) => Condition::Expr(self.map_expr(e, &mut add_register)?),
            Condition::Register(target, value) => {
                if !allow_reorder {
                    return Ok(Condition::Register(
                        self.map_register(target, &mut add_register)?,
                        value.clone(),
                    ));
                }
                // This is maintaining the legacy behavior of `DAGCircuit.compose`.  We don't
                // attempt to speed-up this lookup with a cache, since that would just make the more
                // standard cases more annoying to deal with.

                let mapped_bits_order = target
                    .bits()
                    .map(|b| self.bit_map.get(&b).unwrap().clone())
                    .collect::<Vec<_>>();
                let mapped_bits_set: HashSet<ShareableClbit> =
                    mapped_bits_order.iter().cloned().collect();

                let mapped_theirs = self
                    .target_cregs
                    .iter()
                    .find(|register| {
                        let register_set: HashSet<ShareableClbit> = register.iter().collect();
                        mapped_bits_set == register_set
                    })
                    .cloned()
                    .map(Ok::<_, E>)
                    .unwrap_or_else(|| {
                        let mapped_theirs =
                            ClassicalRegister::new_alias(None, mapped_bits_order.clone());
                        add_register(&mapped_theirs)?;
                        Ok(mapped_theirs)
                    })?;

                let new_order: HashMap<ShareableClbit, usize> = mapped_bits_order
                    .into_iter()
                    .enumerate()
                    .map(|(i, bit)| (bit, i))
                    .collect();

                // Build the little-endian bit string
                let value_bits: Vec<char> = format!("{:0width$b}", value, width = target.len())
                    .chars()
                    .rev()
                    .collect();

                // Reorder bits and reverse again to go back to big-endian for final conversion
                let mapped_str: String = mapped_theirs
                    .iter() // TODO: we should probably not need to collect to Vec here. Why do we?
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|bit| value_bits[*new_order.get(&bit).unwrap()])
                    .rev()
                    .collect();

                Condition::Register(
                    mapped_theirs,
                    BigUint::from_str_radix(&mapped_str, 2).unwrap(),
                )
            }
        })
    }

    /// Map the real-time variables in a `target` of a `SwitchCaseOp` to the new
    /// circuit.
    pub fn map_target<F, E>(
        &self,
        target: &SwitchTarget,
        mut add_register: F,
    ) -> Result<SwitchTarget, E>
    where
        F: FnMut(&ClassicalRegister) -> Result<(), E>,
        E: Error,
    {
        Ok(match target {
            SwitchTarget::Bit(bit) => SwitchTarget::Bit(self.bit_map.get(bit).cloned().unwrap()),
            SwitchTarget::Register(register) => {
                SwitchTarget::Register(self.map_register(register, &mut add_register)?)
            }
            SwitchTarget::Expr(expr) => SwitchTarget::Expr(self.map_expr(expr, &mut add_register)?),
        })
    }

    /// Map the variables in an [expr::Expr] node to the new circuit.
    pub fn map_expr<F, E>(&self, expr: &expr::Expr, mut add_register: F) -> Result<expr::Expr, E>
    where
        F: FnMut(&ClassicalRegister) -> Result<(), E>,
        E: Error,
    {
        let mut mapped = expr.clone();
        mapped.visit_mut(|e| match e {
            expr::ExprRefMut::Var(var) => match var {
                expr::Var::Standalone { .. } => {
                    if let Some(mapping) = self.var_map.get(var).cloned() {
                        *var = mapping;
                    }
                    Ok(())
                }
                expr::Var::Bit { bit } => {
                    let bit = self.bit_map.get(bit).cloned().unwrap();
                    *var = expr::Var::Bit { bit };
                    Ok(())
                }
                expr::Var::Register { register, ty } => {
                    let ty = *ty;
                    let register = self.map_register(register, &mut add_register)?;
                    *var = expr::Var::Register { register, ty };
                    Ok(())
                }
            },
            expr::ExprRefMut::Stretch(stretch) => {
                if let Some(mapping) = self.stretch_map.get(stretch).cloned() {
                    *stretch = mapping;
                }
                Ok(())
            }
            _ => Ok(()),
        })?;
        Ok(mapped)
    }

    /// Map the target's registers to suitable equivalents in the destination, adding an
    /// extra one if there's no exact match.
    fn map_register<F, E>(
        &self,
        theirs: &ClassicalRegister,
        mut add_register: F,
    ) -> Result<ClassicalRegister, E>
    where
        F: FnMut(&ClassicalRegister) -> Result<(), E>,
        E: Error,
    {
        if let Some(mapped_theirs) = self.register_map.borrow().get(theirs.name()) {
            return Ok(mapped_theirs.clone());
        }

        let mapped_bits: Vec<_> = theirs.iter().map(|b| self.bit_map[&b].clone()).collect();
        let mapped_theirs = self
            .target_cregs
            .iter()
            .find(|register| {
                let register: Vec<_> = register.bits().collect();
                mapped_bits == register
            })
            .cloned()
            .map(Ok::<_, E>)
            .unwrap_or_else(|| {
                let mapped_theirs = ClassicalRegister::new_alias(None, mapped_bits.clone());
                add_register(&mapped_theirs)?;
                Ok(mapped_theirs)
            })?;

        self.register_map
            .borrow_mut()
            .insert(theirs.name().to_string(), mapped_theirs.clone());
        Ok(mapped_theirs)
    }
}
