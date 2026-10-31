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

//! Implementation of the Symbol Table for Qiskit Midle-End Intermediate Representation.

use std::fmt;

/// Errors encountered when dealing with Symbols and the SymbolTable.
#[derive(thiserror::Error, Clone, Debug)]
#[error("{kind}")]
pub struct SymbolTableError {
    kind: SymbolTableErrorKind,
    /// Symbol is optional because not all errors may be related to specific symbols.
    /// But when an error *is* related to a specific symbol, it will be good to include it
    /// in the error message to help with debugging.
    sym: Option<Symbol>,
}

impl SymbolTableError {
    /// View the [`Symbol`] attached to this error (if there is one).
    #[inline]
    pub fn symbol(&self) -> Option<&Symbol> {
        self.sym.as_ref()
    }
    /// Retrieve the ownership of the Symbol.
    #[inline]
    pub fn into_symbol(self) -> Option<Symbol> {
        self.sym
    }
    /// Get the underlying error.
    #[inline]
    pub fn kind(&self) -> &SymbolTableErrorKind {
        &self.kind
    }
}

/// The kind of error that has been triggered.
///
/// This will grow over time as we add more functionality to the SymbolTable,
/// and possibly verification and sanity checks.
#[derive(thiserror::Error, Clone, Debug, PartialEq)]
pub enum SymbolTableErrorKind {
    #[error("Symbol ID already set: {0}")]
    AlreadyInitialized(SymbolId),
}

impl SymbolTableErrorKind {
    pub fn with_symbol(self, sym: Symbol) -> SymbolTableError {
        SymbolTableError {
            kind: self,
            sym: Some(sym),
        }
    }
}

/// Symbol represents various types of classical information. This will start
/// with variables, but likely grow to include other things over time.
#[derive(Clone, Debug, PartialEq)]
pub struct Symbol {
    /// Symbol ID used in the IR. This field should be set when adding to the
    /// Symbol Table. This field is intentionally private and does not have a
    /// set method because it should not be set outside of this module.
    sym_id: Option<SymbolId>,
    name: String,
}

impl Symbol {
    /// Create a new (empty) symbol.
    pub const fn new(name: String) -> Self {
        Self { sym_id: None, name }
    }

    /// Get the name for this symbol.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the symbol id field for this symbol.
    ///
    /// Note that it is expected for the SymbolId field to be None before being
    /// added to the symbol table, and contain a valid Id after being added to
    /// the table (i.e., the Symbol Table manages and provides the Symbol Ids).
    pub fn id(&self) -> Option<SymbolId> {
        self.sym_id
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Symbol: {:?}, {}", self.sym_id, self.name)
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct SymbolId(usize);

impl fmt::Display for SymbolId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Default, Debug)]
pub struct SymbolTable {
    sym_table: Vec<Symbol>,
}

impl SymbolTable {
    /// Add a symbol to the symbol table.
    ///
    /// It is assumed that the provided symbol does *not* have an Id set. If it
    /// does, it is assumed to be already in the symbol table and an error is
    /// returned.
    ///
    /// Note: the symbol id is derived from the number of symbols currently in
    /// the table. This will not work once we support removing symbols from the
    /// table. At that point, we will want to look at a new backing data
    /// structure (something that can efficiently remove symbols and keep the
    /// existing SymbolId valid) and update this to track the next ID
    /// differently. That has intentionally been left for later.
    pub fn add(&mut self, mut sym: Symbol) -> Result<SymbolId, SymbolTableError> {
        if let Some(sym_id) = sym.sym_id {
            return Err(SymbolTableErrorKind::AlreadyInitialized(sym_id).with_symbol(sym));
        }

        let this_sym_id = SymbolId(self.sym_table.len());
        sym.sym_id = Some(this_sym_id);
        self.sym_table.push(sym);

        Ok(this_sym_id)
    }

    /// Get a reference to a symbol in the symbol table.
    ///
    /// This will panic if the specified SymbolId is *not* in the table.
    pub fn get(&self, sym_id: SymbolId) -> Option<&Symbol> {
        self.sym_table.get(sym_id.0)
    }

    /// Get a mutable reference to the specified symbol.
    ///
    /// This will panic if the SymbolId is not in the table.
    pub fn get_mut(&mut self, sym_id: SymbolId) -> Option<&mut Symbol> {
        self.sym_table.get_mut(sym_id.0)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SymbolIdList(Vec<SymbolId>);

impl SymbolIdList {
    pub fn new(syms: Vec<SymbolId>) -> Self {
        SymbolIdList(syms)
    }
    pub fn inner(&self) -> &Vec<SymbolId> {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_symbol() {
        let new_sym = Symbol::new(String::from("test"));

        assert_eq!(new_sym.sym_id, None);
        assert_eq!(new_sym.name, "test");
    }

    #[test]
    fn test_sym_table_add_symbols() {
        let new_sym = Symbol::new(String::from("test"));
        let mut new_sym2 = Symbol::new(String::from("bad"));

        new_sym2.sym_id = Some(SymbolId(1));

        let mut sym_table = SymbolTable::default();

        let sym_id = sym_table.add(new_sym).unwrap();

        assert_eq!(sym_id, SymbolId(0));

        assert_eq!(
            *sym_table.add(new_sym2).unwrap_err().kind(),
            SymbolTableErrorKind::AlreadyInitialized(SymbolId(1))
        );
    }

    #[test]
    fn test_get_symbol() {
        let new_sym = Symbol::new(String::from("test"));
        let mut copy_sym = new_sym.clone();

        let mut sym_table = SymbolTable::default();

        let sym_id = sym_table.add(new_sym).unwrap();
        copy_sym.sym_id = Some(sym_id);

        assert_eq!(*sym_table.get(sym_id).unwrap(), copy_sym);

        assert!(sym_table.get(SymbolId(1)).is_none());
    }
}
