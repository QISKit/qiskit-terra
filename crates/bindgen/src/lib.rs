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

pub mod render;
pub mod simple_ir;

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const CBINDGEN_ATTRIBUTE_NAME: &str = "qk-vtable-rules";
pub const CBINDGEN_SKIP: &str = "no-export";
pub const CBINDGEN_DUPLICATE: &str = "allow-duplicate";

/// Structured set of attributes that can be set on functions in `cbindgen`.
#[derive(Clone, Copy, Debug, Default)]
pub struct FnAttributes {
    /// The function should be skipped and not present in any vtable slots list.
    pub skipped: bool,
    /// The function is permitted to be exported in more than one slot.
    pub allow_duplicate: bool,
}
impl FnAttributes {
    /// Set the field corresponding to a given attribute.
    pub fn set(&mut self, attr: &str) -> anyhow::Result<()> {
        match attr {
            CBINDGEN_SKIP => self.skipped = true,
            CBINDGEN_DUPLICATE => self.allow_duplicate = true,
            _ => anyhow::bail!("unknown attribute: {attr}"),
        }
        Ok(())
    }
}

pub static SCOPED_INCLUDE_DIR: &str = "qiskit";
pub static GENERATED_FILE_TYPES: &str = "types.h";
pub static GENERATED_FILE_FUNCS: &str = "funcs.h";

pub static PYTHON_BINDING_FEATURE: &str = "python_binding";
pub static PYTHON_BINDING_DEFINE: &str = "QISKIT_C_PYTHON_INTERFACE";

pub static COPYRIGHT: &str = "\
This code is part of Qiskit.

(C) Copyright IBM 2026

This code is licensed under the Apache License, Version 2.0. You may
obtain a copy of this license in the LICENSE.txt file in the root directory
of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.

Any modifications or derivative works of this code must retain this
copyright notice, and modified files need to carry a notice indicating
that they have been altered from the originals.
";
pub fn copyright_with_line_comments(comment: &str) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    for line in COPYRIGHT.lines() {
        if line.is_empty() {
            _ = writeln!(out, "{comment}");
        } else {
            _ = writeln!(out, "{comment} {line}");
        }
    }
    out
}

/// Crates that contain definitions of objects that are exposed through the C API.
pub static QISKIT_PUBLIC_API_CRATES: &[&str] =
    &["qiskit-quantum-info", "qiskit-circuit", "qiskit-transpiler"];

pub static EXPORT_PREFIX: &str = "Qk";
pub static EXPORT_RENAME: &[(&str, &str)] = &[
    ("CBlocksMode", "BlocksMode"),
    ("CDagNeighbors", "DagNeighbors"),
    ("CDagNodeType", "DagNodeType"),
    ("CDelayUnit", "DelayUnit"),
    ("CInstruction", "CircuitInstruction"),
    ("CInstructionProperties", "InstructionProperties"),
    ("CNeighbors", "Neighbors"),
    ("COperationKind", "OperationKind"),
    ("CPauliProductRotation", "PauliProductRotation"),
    ("CPauliProductMeasurement", "PauliProductMeasurement"),
    ("CSparseTerm", "ObsTerm"),
    ("CTargetOp", "TargetOp"),
    ("CVarsMode", "VarsMode"),
    ("CircuitData", "Circuit"),
    ("DAGCircuit", "Dag"),
    ("SparseObservable", "Obs"),
    ("StandardGate", "Gate"),
    // Classical expression types
    ("Expr", "ExprNode"),
    ("CExprNodeKind", "ExprNodeKind"),
    ("CBinaryExprInfo", "BinaryExprInfo"),
    ("CBinaryOpType", "BinaryOpType"),
    ("CUnaryExprInfo", "UnaryExprInfo"),
    ("CUnaryOpType", "UnaryOpType"),
    ("CCastExprInfo", "CastExprInfo"),
    ("CIndexExprInfo", "IndexExprInfo"),
    ("CExprType", "ExprType"),
    ("CExprTypeInfo", "ExprTypeInfo"),
    ("CDurationType", "DurationType"),
    ("CDurationInfo", "DurationInfo"),
    ("CDurationValue", "DurationValue"),
    // Control flow types
    ("CControlFlowInstruction", "ControlFlowInstruction"),
    ("CControlFlowKind", "ControlFlowKind"),
    ("CConditionType", "ConditionType"),
    ("CConditionBitInfo", "ConditionBitInfo"),
    ("CBoxDurationKind", "BoxDurationKind"),
    ("CSwitchCaseLabels", "SwitchCaseLabels"),
    ("CSymbolType", "SymbolType"),
    ("CSymbolInfo", "SymbolInfo"),
    ("CLoopCollectionType", "LoopCollectionType"),
    ("CLoopParamKind", "LoopParamKind"),
    ("CLoopElements", "LoopElements"),
];
pub static EXPORT_VERBATIM: &[&str] = &["PyObject"];

// Defined in `qiskit/attributes.h`.
pub static FN_DEPRECATED: &str = "Qk_DEPRECATED_FN";
pub static FN_DEPRECATED_WITH_NOTE: &str = "Qk_DEPRECATED_FN_NOTE({})";

/// Mapping of Rust `#[cfg(feature = <key>)]` keys to C `#ifdef <value>` values.
pub static CFG_FEATURE_DEFINES: &[(&str, &str)] =
    &[(PYTHON_BINDING_FEATURE, PYTHON_BINDING_DEFINE)];

#[inline]
fn to_vec_string(slice: &[&str]) -> Vec<String> {
    slice.iter().map(|s| String::from(*s)).collect()
}

#[inline]
fn manual_include_dir() -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "include"]
        .iter()
        .collect::<PathBuf>()
}
/// Get the manually written include files, relative to the result of [`manual_include_dir`].
fn manual_include_files() -> anyhow::Result<Vec<PathBuf>> {
    fn recurse(
        relative_path: &Path,
        base_dir: &Path,
        mut out: Vec<PathBuf>,
    ) -> anyhow::Result<Vec<PathBuf>> {
        for entry in base_dir.read_dir()? {
            let entry = entry?;
            let name = entry.file_name();
            let path = entry.path();
            if entry.metadata()?.is_dir() {
                out = recurse(&relative_path.join(&name), &base_dir.join(&name), out)?;
            } else {
                // Only propagate files that look like C/C++ header files.
                match path.extension().and_then(|s| s.to_str()) {
                    Some("h") | Some("hpp") => out.push(relative_path.join(&name)),
                    Some(_) | None => (),
                }
            }
        }
        Ok(out)
    }
    recurse(&PathBuf::new(), &manual_include_dir(), Vec::new())
}

/// Get the Qiskit configuration
fn get_config() -> anyhow::Result<cbindgen::Config> {
    // We need to include the `attributes.h` file in all generated files to make sure Doxygen can
    // understand the deprecated attributes (even though `qiskit.h` is organised to include it).
    let includes = vec![
        Path::new(SCOPED_INCLUDE_DIR)
            .join("attributes.h")
            .to_str()
            .expect("our paths should always be valid utf-8")
            .to_owned(),
    ];
    let enumeration = cbindgen::EnumConfig {
        prefix_with_name: true,
        ..Default::default()
    };
    // For the export configuration, we need certain names ("EXPORT_VERBATIM") to stay as they are,
    // so we instead set `renaming_overrides_prefixing: true`, and manually propagate the prefix
    // onto all the renames, so we can re-use the "renames" mapping to force the verbatim exports
    // too.
    #[expect(clippy::disallowed_types)] // used in the cbindgen API.
    let mut rename: std::collections::HashMap<_, _> = EXPORT_RENAME
        .iter()
        .map(|&(k, v)| (String::from(k), format!("{EXPORT_PREFIX}{v}")))
        .collect();
    rename.extend(
        EXPORT_VERBATIM
            .iter()
            .map(|&k| (String::from(k), String::from(k))),
    );
    let export = cbindgen::ExportConfig {
        prefix: Some(EXPORT_PREFIX.into()),
        rename,
        renaming_overrides_prefixing: true,
        ..Default::default()
    };
    let function = cbindgen::FunctionConfig {
        deprecated: Some(FN_DEPRECATED.into()),
        deprecated_with_note: Some(FN_DEPRECATED_WITH_NOTE.into()),
        ..Default::default()
    };
    let parse = cbindgen::ParseConfig {
        parse_deps: true,
        include: Some(to_vec_string(QISKIT_PUBLIC_API_CRATES)),
        ..Default::default()
    };
    let defines = CFG_FEATURE_DEFINES
        .iter()
        .map(|&(cfg, def)| (format!("feature = {cfg}"), String::from(def)))
        .collect();
    Ok(cbindgen::Config {
        header: Some(copyright_with_line_comments("//")),
        language: cbindgen::Language::C,
        includes,
        include_version: true,
        style: cbindgen::Style::Type,
        cpp_compat: true,
        usize_is_size_t: true,
        defines,
        enumeration,
        export,
        function,
        parse,
        ..Default::default()
    })
}

/// Is a given function marked with one of our special attributes?
///
/// Returns an error if there are unknown attributes used in the list.
pub fn fn_attrs(func: &cbindgen::ir::Function) -> anyhow::Result<FnAttributes> {
    func.annotations
        .list(CBINDGEN_ATTRIBUTE_NAME)
        .map_or(Ok(FnAttributes::default()), |attrs| {
            let mut out = FnAttributes::default();
            for attr in attrs {
                out.set(&attr)?;
            }
            Ok(out)
        })
}

/// Generate the cbindgen bindings object for the C-extensions crate.
pub fn generate_bindings(cext_path: impl AsRef<Path>) -> anyhow::Result<cbindgen::Bindings> {
    cbindgen::Builder::new()
        .with_crate(cext_path)
        .with_config(get_config()?)
        .generate()
        .map_err(|e| e.into())
}

/// Install the complete stand-alone C include path into the given directory.
///
/// This takes `&mut Bindings` only as an internal implementation detail of how separated-out
/// include files are written; the bindings will always be returned to their original state when
/// the function returns.
pub fn install_c_headers(
    bindings: &mut cbindgen::Bindings,
    install_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let install_path = install_path.as_ref();
    let scoped_install_path = install_path.join(SCOPED_INCLUDE_DIR);
    fs::create_dir_all(&scoped_install_path)?;
    // _Probably_ globals and constants can be handled just by putting them in the types file
    // (because they're likely shared between all access modes to the header file), but since we
    // haven't got any yet, we just stay safe and check when some appear.
    assert!(bindings.globals.is_empty(), "globals not handled yet");
    let mut buf = Vec::<u8>::new();
    {
        // First, write out only the types and constants into one file.
        let functions = ::std::mem::take(&mut bindings.functions);
        bindings.write(&mut buf);
        bindings.functions = functions;
        fs::File::create(scoped_install_path.join(GENERATED_FILE_TYPES))?.write_all(&buf)?;
        buf.clear();
    }
    {
        // Now, write out the functions into the part of the generated file that's only read when
        // we're not in Python-extension mode.
        let items = ::std::mem::take(&mut bindings.items);
        let constants = ::std::mem::take(&mut bindings.constants);
        bindings.write(&mut buf);
        bindings.items = items;
        bindings.constants = constants;
        fs::File::create(scoped_install_path.join(GENERATED_FILE_FUNCS))?.write_all(&buf)?;
        buf.clear();
    }
    let manual_path = manual_include_dir();
    for file in manual_include_files()? {
        if let Some(parent) = file.parent() {
            fs::create_dir_all(install_path.join(parent))?;
        }
        fs::copy(manual_path.join(&file), install_path.join(&file))?;
    }
    Ok(())
}

fn rust_pyo3_ffi_root() -> PathBuf {
    [env!("CARGO_MANIFEST_DIR"), "pyo3-ffi"].iter().collect()
}

fn rust_pyo3_ffi_files() -> anyhow::Result<Vec<PathBuf>> {
    let root = rust_pyo3_ffi_root();
    let output = Command::new("cargo")
        .arg("package")
        .arg("--list") // Only list the files in the output, don't do any packaging.
        .arg("--allow-dirty") // Permit `git` to be dirty, since this isn't a true packaging op.
        .current_dir(&root)
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "'cargo package' listing failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut out = Vec::new();
    // Technically the paths are probably `OsString`, but that's just fiddly to handle properly.  We
    // can fix it if we need to.
    for line in String::from_utf8(output.stdout)?.lines() {
        let path = line.parse::<PathBuf>()?;
        // Some of the files listed by `cargo package` are ones that would be generated by the
        // packaging command, so if they don't exist, don't add them.
        if root.join(&path).is_file() {
            out.push(path);
        }
    }
    Ok(out)
}

pub fn install_rust_pyo3_ffi(
    bindings: &cbindgen::Bindings,
    install_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let mut items = render::rust::Items::default();
    items.add_from_cbindgen(bindings)?;

    let install_path = install_path.as_ref();
    fs::create_dir_all(install_path)?;

    // First copy over the template repository verbatim...
    let crate_root = rust_pyo3_ffi_root();
    for file in rust_pyo3_ffi_files()? {
        if let Some(parent) = file.parent() {
            fs::create_dir_all(install_path.join(parent))?;
        }
        fs::copy(crate_root.join(&file), install_path.join(&file))?;
    }

    // ... and then overwrite the generated file.
    items.export(fs::File::create(install_path.join("src").join("ffi.rs"))?)?;
    Ok(())
}
