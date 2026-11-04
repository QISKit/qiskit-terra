// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use super::symbol_expr::{Symbol, SymbolExpr};

pub struct IterSymbols<'a>(Vec<&'a SymbolExpr>);
impl<'a> IterSymbols<'a> {
    pub fn new(expr: &'a SymbolExpr) -> Self {
        Self(vec![expr])
    }
}

impl<'a> Iterator for IterSymbols<'a> {
    type Item = &'a Symbol;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(mut cur) = self.0.pop() {
            loop {
                cur = match cur {
                    SymbolExpr::Symbol(e) => return Some(e),
                    SymbolExpr::Value(_) => break,
                    SymbolExpr::Unary { op: _, expr } => expr,
                    SymbolExpr::Binary { op: _, lhs, rhs } => {
                        self.0.push(rhs);
                        lhs
                    }
                };
            }
        }
        None
    }
}
