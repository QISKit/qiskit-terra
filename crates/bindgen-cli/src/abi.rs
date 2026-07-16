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

use std::fmt::{self, Write};

use anyhow::{anyhow, bail};
use itertools::EitherOrBoth;
use qiskit_cext_vtable::ExportedFunction;
use qiskit_util::IndexMap;

/// What type of change is permitted in the ABI between the two versions?
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub enum SemVer {
    Bugfix,
    Feature,
    Breaking,
}
impl SemVer {
    fn from_versions(old: &Version, new: &Version) -> anyhow::Result<Self> {
        if (old.major, old.minor, old.patch) > (new.major, new.minor, new.patch) {
            bail!("'old' version {} is actually newer than {}", old, new);
        }
        if old.major < new.major {
            Ok(Self::Breaking)
        } else if old.minor < new.minor {
            Ok(Self::Feature)
        } else {
            // This also includes two equal versions, which seems possible for developing, and
            // comparing two `-dev` versions that aren't actually equal.
            Ok(Self::Bugfix)
        }
    }
}

/// The minimal semver components extracted from a version that might also have suffixes.
#[derive(Clone, Debug)]
pub struct Version {
    major: usize,
    minor: usize,
    patch: usize,
    suffix: Option<String>,
}
impl Version {
    /// Parse a version out of a `<major>.<minor>.<patch>(-<suffix>)?` form.
    fn try_parse(val: &str) -> anyhow::Result<Version> {
        let (val, suffix) = match val.split_once("-") {
            Some((val, suffix)) => (val, Some(suffix.to_owned())),
            None => (val, None),
        };
        let mut parts = val.split(".");
        let mut part = || -> anyhow::Result<usize> {
            Ok(parts
                .next()
                .ok_or_else(|| anyhow!("not enough version parts"))?
                .parse()?)
        };
        Ok(Version {
            major: part()?,
            minor: part()?,
            patch: part()?,
            suffix,
        })
    }
}
impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(suffix) = self.suffix.as_ref() {
            write!(f, "-{suffix}")?;
        }
        Ok(())
    }
}

/// Complete set of exported slots and version for comparison.
pub struct SlotsLists {
    pub api_version: Version,
    /// Association of names to concrete vtables.
    pub slots: IndexMap<String, SlotsList>,
}
impl SlotsLists {
    /// The slots lists that are defined in the depended-on `qiskit_cext_vtable`.
    pub fn ours() -> Self {
        let names = |funcs: Vec<Option<ExportedFunction>>| {
            SlotsList(
                funcs
                    .into_iter()
                    .map(|f| f.map(|f| f.name.to_owned()))
                    .collect(),
            )
        };
        Self {
            api_version: Version::try_parse(env!("CARGO_PKG_VERSION"))
                .expect("our version should be valid"),
            slots: [
                ("circuit", &qiskit_cext_vtable::FUNCTIONS_CIRCUIT),
                ("transpile", &qiskit_cext_vtable::FUNCTIONS_TRANSPILE),
                ("qi", &qiskit_cext_vtable::FUNCTIONS_QI),
            ]
            .iter()
            .map(|(name, funcs)| (String::from(*name), names(funcs.slots())))
            .collect(),
        }
    }

    /// Parse a slots list out of a string.
    ///
    /// The expected format of the string is a literal output of our own `Display` implementation,
    /// which in turn is the same as a call to `qiskit-bindgen-cli show-slots`.
    pub fn try_parse(val: &str) -> anyhow::Result<Self> {
        // This isn't fancy or trying to cleanly recover - we're mostly just expected the input to
        // be correct already.
        let mut lines = val.trim().lines();
        let Some(line) = lines.next() else {
            bail!("missing __version__");
        };
        let Some(version) = line.strip_prefix("__version__ =").map(str::trim) else {
            bail!("missing __version__");
        };
        let Some(version) = version.strip_prefix('"').and_then(|v| v.strip_suffix('"')) else {
            bail!("malformed __version__: not a string");
        };
        let mut slots_lists = IndexMap::default();
        while let Some(intro) = lines.next() {
            let Some((name, rest)) = intro.split_once(" = [") else {
                bail!("didn't find expected '<slots_name> = [' opener");
            };
            match rest.trim() {
                "" => (),
                "]" => {
                    slots_lists.insert(name.to_owned(), SlotsList(vec![]));
                    continue;
                }
                _ => bail!("unexpected line after '{name}': '{rest}'"),
            }
            let mut slots = Vec::new();
            for line in lines.by_ref().map(str::trim) {
                if line == "]" {
                    break;
                }
                let Some(func_name) = line
                    .strip_prefix('"')
                    .and_then(|line| line.strip_suffix("\","))
                else {
                    bail!("failed to parse slot line '{line}'");
                };
                slots.push((!func_name.is_empty()).then(|| String::from(func_name)));
            }
            slots_lists.insert(name.to_owned(), SlotsList(slots));
        }
        Ok(Self {
            api_version: Version::try_parse(version)?,
            slots: slots_lists,
        })
    }
}
impl fmt::Display for SlotsLists {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "__version__ = \"{}\"", self.api_version)?;
        for (name, slots) in self.slots.iter() {
            writeln!(f, "{name} = {slots}")?;
        }
        Ok(())
    }
}

/// An individual concrete vtable, but written in terms of the function name instead of a pointer.
pub struct SlotsList(Vec<Option<String>>);
impl SlotsList {
    /// Iterate over the slot indices and the name stored there in order.
    pub fn iter_names(&self) -> impl Iterator<Item = (usize, &str)> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(i, fname)| fname.as_deref().map(|fname| (i, fname)))
    }
}
impl fmt::Display for SlotsList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // This is meant to be easy both to parse back in, and to be read by humans.  As written,
        // this `impl` also produces a valid Python list of strings that can be read by
        // `ast.literal_eval`.
        if self.0.is_empty() {
            write!(f, "[]")
        } else {
            writeln!(f, "[")?;
            for slot in self.0.iter() {
                writeln!(f, "    \"{}\",", slot.as_deref().unwrap_or_default())?;
            }
            write!(f, "]")
        }
    }
}

#[derive(Clone, Debug)]
pub enum SlotBreakage {
    Deleted(String),
    Changed { from: String, to: String },
}

#[derive(Clone, Debug, Default)]
pub struct Breakages {
    pub deleted_tables: Vec<String>,
    pub changed_slots: IndexMap<String, Vec<(usize, SlotBreakage)>>,
}
impl Breakages {
    pub fn is_empty(&self) -> bool {
        self.deleted_tables.is_empty() && self.changed_slots.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Features {
    pub new_tables: Vec<String>,
    pub new_slots: IndexMap<String, Vec<(usize, String)>>,
}
impl Features {
    pub fn is_empty(&self) -> bool {
        self.new_tables.is_empty() && self.new_slots.is_empty()
    }
}

/// Tracked failures from comparison of two slots lists.
#[derive(Clone, Debug)]
pub struct Changes {
    pub version_prev: Version,
    pub version_cur: Version,
    pub semver_allowed: SemVer,
    pub breakages: Breakages,
    pub features: Features,
}
impl Changes {
    pub fn semver_actual(&self) -> SemVer {
        if !self.breakages.is_empty() {
            SemVer::Breaking
        } else if !self.features.is_empty() {
            SemVer::Feature
        } else {
            SemVer::Bugfix
        }
    }

    pub fn is_allowed(&self) -> bool {
        self.semver_actual() <= self.semver_allowed
    }

    pub fn explain(&self) -> String {
        if self.is_allowed() {
            return format!(
                "Current slots list for version {} is compatible with previous version {}.",
                self.version_cur, self.version_prev,
            );
        }
        let mut explanation = format!(
            "Current slots list for version {} is incompatible with previous version {}.",
            self.version_cur, self.version_prev,
        );
        write!(
            explanation,
            " Allowed {:?}, but actual change is {:?}.",
            self.semver_allowed,
            self.semver_actual()
        )
        .unwrap();
        if !self.breakages.deleted_tables.is_empty() {
            write!(explanation, "\n\n[API break] Deleted tables:").unwrap();
            for table in &self.breakages.deleted_tables {
                write!(explanation, "\n* {table}").unwrap();
            }
        }
        if !self.breakages.changed_slots.is_empty() {
            write!(explanation, "\n\n[API break] Changed tables:").unwrap();
            for (table, changes) in &self.breakages.changed_slots {
                write!(explanation, "\n* {table}").unwrap();
                for (slot, change) in changes {
                    write!(explanation, "\n - {slot}: ").unwrap();
                    match change {
                        SlotBreakage::Deleted(name) => {
                            write!(explanation, "deleted '{name}'").unwrap()
                        }
                        SlotBreakage::Changed { from, to } => {
                            write!(explanation, "changed '{from}' to '{to}'").unwrap()
                        }
                    }
                }
            }
        }
        if !self.features.new_tables.is_empty() {
            write!(explanation, "\n\n[New feature] Added tables:").unwrap();
            for table in &self.features.new_tables {
                write!(explanation, "\n* {table}").unwrap();
            }
        }
        if !self.features.new_slots.is_empty() {
            write!(explanation, "\n\n[New feature] Added slots:").unwrap();
            for (table, slots) in &self.features.new_slots {
                write!(explanation, "\n* {table}").unwrap();
                for (slot, added) in slots {
                    write!(explanation, "\n - {slot}: {added}").unwrap();
                }
            }
        }
        explanation
    }
}

pub fn check(old: &SlotsLists, new: &SlotsLists) -> anyhow::Result<Changes> {
    let semver_allowed = SemVer::from_versions(&old.api_version, &new.api_version)?;
    // TODO: this is a very basic checker that's just to get a pass/fail check running in CI.  The
    // pass/fail should be entirely accurate, but its descriptions of _what_ changed are wildly
    // overcomplex in most cases.  What we actually want is a modification of a line-diff algorithm
    // (a Levenshtein-distance minimisation), because the actual most likely failure is a slot
    // _insertion_, which is the exact situation that this algorithm will produce the worst
    // explanations for (there's no concept of "moved" here).
    let mut breakages = Breakages::default();
    let mut features = Features::default();
    for (table, slots_old) in old.slots.iter() {
        let Some(slots_new) = new.slots.get(table) else {
            breakages.deleted_tables.push(table.clone());
            continue;
        };
        let mut changed_slots = Vec::new();
        let mut new_slots = Vec::new();
        for slot in itertools::merge_join_by(
            slots_old.iter_names(),
            slots_new.iter_names(),
            |(i, _), (j, _)| (*i).cmp(j),
        ) {
            match slot {
                EitherOrBoth::Left((slot, name)) => {
                    changed_slots.push((slot, SlotBreakage::Deleted(name.to_owned())));
                }
                EitherOrBoth::Right((slot, name)) => {
                    new_slots.push((slot, name.to_owned()));
                }
                EitherOrBoth::Both((slot, from), (_, to)) => {
                    if from != to {
                        changed_slots.push((
                            slot,
                            SlotBreakage::Changed {
                                from: from.to_owned(),
                                to: to.to_owned(),
                            },
                        ));
                    }
                }
            }
        }
        if !changed_slots.is_empty() {
            breakages
                .changed_slots
                .insert(table.to_owned(), changed_slots);
        }
        if !new_slots.is_empty() {
            features.new_slots.insert(table.to_owned(), new_slots);
        }
    }
    features.new_tables.extend(
        new.slots
            .keys()
            .filter(|table| !old.slots.contains_key(*table))
            .cloned(),
    );
    Ok(Changes {
        version_prev: old.api_version.clone(),
        version_cur: new.api_version.clone(),
        semver_allowed,
        breakages,
        features,
    })
}
