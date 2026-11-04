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

mod abi;
mod lint;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::anyhow;
use clap::{Parser, Subcommand};

use crate::abi::SlotsLists;

/// Toolkit for working with and installing the C header files.
#[derive(Parser, Debug)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}
#[derive(Subcommand, Debug)]
enum Command {
    /// Install the header files into a given directory.
    Install {
        /// Path to the `cext` sources to generate headers for.
        #[arg(short, long)]
        cext_path: PathBuf,
        /// Where to install the header files to.
        #[arg(short, long)]
        output_path: PathBuf,
    },
    /// Print out a representation of all the slots in use.
    ShowSlots,
    /// Check for correctness between the slots tables and the list of exported functions for the
    /// current version of Qiskit.
    LintSlots {
        /// Path to the `cext` sources to generate headers for.
        #[arg(short, long)]
        cext_path: PathBuf,
    },
    GeneratePyo3 {
        /// Path to the `cext` sources to generate headers for.
        #[arg(short, long)]
        cext_path: PathBuf,
        /// Path to write the output crate.
        #[arg(short, long)]
        output_path: PathBuf,
    },
    /// Check that the semantic version constraints are upheld between two sets of slots.
    ///
    /// This does not check the function-pointer types of any given symbol, only the names and
    /// offsets.
    CheckAbi {
        /// An filepath to an export of a previous Qiskit version's slots.
        old: PathBuf,
        /// If givne, a filepath to an export of a newer Qiskit version's slots.  If not given,
        /// compare against the current vtable definitions.
        new: Option<PathBuf>,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match &args.command {
        Command::Install {
            cext_path,
            output_path,
        } => {
            let mut bindings = qiskit_bindgen::generate_bindings(cext_path)?;
            qiskit_bindgen::install_c_headers(&mut bindings, output_path)?;
            Ok(())
        }
        #[allow(clippy::print_stdout)]
        Command::ShowSlots => {
            println!("{}", SlotsLists::ours());
            Ok(())
        }
        Command::LintSlots { cext_path } => {
            let bindings = qiskit_bindgen::generate_bindings(cext_path)?;
            lint::lint(&bindings, &SlotsLists::ours())?.map_err(|fails| anyhow!(fails.explain()))
        }
        Command::GeneratePyo3 {
            cext_path,
            output_path,
        } => {
            let bindings = qiskit_bindgen::generate_bindings(cext_path)?;
            qiskit_bindgen::install_rust_pyo3_ffi(&bindings, output_path)?;
            Ok(())
        }
        Command::CheckAbi { old, new } => {
            let load = |path: &Path| SlotsLists::try_parse(&fs::read_to_string(path)?);
            let old = load(old)?;
            let new = new
                .as_deref()
                .map(load)
                .transpose()?
                .unwrap_or_else(SlotsLists::ours);
            let changes = abi::check(&old, &new)?;
            if changes.is_allowed() {
                Ok(())
            } else {
                Err(anyhow!(changes.explain()))
            }
        }
    }
}
