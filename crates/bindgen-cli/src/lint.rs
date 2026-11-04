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

use crate::SlotsLists;
use qiskit_bindgen::{EXPORT_PREFIX, EXPORT_RENAME, fn_attrs};
use qiskit_util::{IndexMap, IndexSet};
use std::fmt::Write;

/// Tracked failures from the linting of a resolved vtable listing against a set of extracted
/// `cbindgen` bindings.
#[derive(Clone, Debug)]
pub struct Failures {
    /// Functions that are in the `cbindgen` bindings but not in any of the listed slots (or
    /// explicitly skipped).
    pub missing: Vec<String>,
    /// Functions that appear more than once in the slots listing.
    pub duplicates: IndexMap<String, Vec<(String, usize)>>,
    /// Functions that are marked as "skipped" in the bindings but aren't actually skipped.
    pub skipped_but_exported: IndexMap<String, (String, usize)>,
    /// Entries in `EXPORT_RENAME` whose source type was never found by cbindgen, meaning the
    /// rename had no effect and is likely a leftover from a deleted or renamed type.
    pub dead_renames: Vec<String>,
}
impl Failures {
    pub fn is_empty(&self) -> bool {
        self.missing.is_empty()
            && self.duplicates.is_empty()
            && self.skipped_but_exported.is_empty()
            && self.dead_renames.is_empty()
    }

    pub fn explain(&self) -> String {
        let mut explanation = String::from("Found discrepancies between bindings and slots.");
        if !self.missing.is_empty() {
            explanation.push_str("\n\nExported functions without a slot assignment or skip:");
            for missing in self.missing.iter() {
                write!(explanation, "\n* {missing}").unwrap();
            }
        }
        if !self.duplicates.is_empty() {
            explanation.push_str("\n\nFunctions exported more than once:");
            for (name, locations) in self.duplicates.iter() {
                write!(explanation, "\n* {name}").unwrap();
                for (api, slot) in locations.iter() {
                    write!(explanation, "\n - ({api}, {slot})").unwrap()
                }
            }
        }
        if !self.skipped_but_exported.is_empty() {
            explanation.push_str("\n\nFunctions marked as vtable skipped, but still present:");
            for (fname, (api, slot)) in self.skipped_but_exported.iter() {
                write!(explanation, "\n* {fname} at ({api}, {slot})").unwrap();
            }
        }
        if !self.dead_renames.is_empty() {
            explanation
                .push_str("\n\nDead entries in EXPORT_RENAME (source type not found by cbindgen):");
            for entry in self.dead_renames.iter() {
                write!(explanation, "\n* {entry}").unwrap();
            }
        }
        explanation
    }
}
/// Lint for simple errors between an extract set of exported functions (`bindings`) and a static
/// listing of slots (`slots`).
///
/// See [`Failures`] for the types of lints performed here.
pub fn lint(
    bindings: &cbindgen::Bindings,
    slots: &SlotsLists,
) -> anyhow::Result<Result<(), Failures>> {
    let mut seen = IndexMap::<&str, (&str, usize)>::default();
    let mut duplicates = IndexMap::<String, Vec<(String, usize)>>::default();
    for (api_name, list) in slots.slots.iter() {
        for (slot, export) in list.iter_names() {
            if let Some((prev_api, prev_slot)) = seen.insert(export, (api_name, slot)) {
                // We put everything into `duplicates` here, then take it out again later if we
                // discover post cbindgen-parsing that that's allowed.
                duplicates
                    .entry(export.to_owned())
                    .or_insert_with(|| vec![(String::from(prev_api), prev_slot)])
                    .push((String::from(api_name), slot));
            }
        }
    }
    let mut skipped_but_exported = IndexMap::default();
    let mut missing = Vec::new();
    for func in bindings.functions.iter() {
        let fname = func.path.name();
        let attrs = fn_attrs(func)?;
        match (seen.get(fname), attrs.skipped) {
            (Some((api, slot)), true) => {
                skipped_but_exported.insert(String::from(fname), (String::from(*api), *slot));
            }
            (None, false) => missing.push(String::from(fname)),
            (Some(_), false) | (None, true) => (),
        }
        if attrs.allow_duplicate {
            duplicates.swap_remove(fname);
        }
    }
    // Check for dead entries in EXPORT_RENAME: entries whose source type was never encountered
    // by cbindgen, meaning the rename silently did nothing.
    let emitted_names: IndexSet<&str> = bindings
        .items
        .iter()
        .map(|item| item.deref().export_name())
        .collect();
    let dead_renames = EXPORT_RENAME
        .iter()
        .filter_map(|&(src, dst)| {
            let expected = format!("{EXPORT_PREFIX}{dst}");
            if !emitted_names.contains(expected.as_str()) {
                Some(format!("{src} -> {expected}"))
            } else {
                None
            }
        })
        .collect();

    let failures = Failures {
        missing,
        duplicates,
        skipped_but_exported,
        dead_renames,
    };
    if failures.is_empty() {
        Ok(Ok(()))
    } else {
        Ok(Err(failures))
    }
}
