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

use crate::simple_ir;
use anyhow::anyhow;
use cbindgen::bindgen::ir;

pub const FN_MACRO: &str = "declare_fn";

#[derive(Clone, Copy, Debug)]
pub enum Primitive {
    PyObject,
    Complex64,
    Bool,
    I8,
    I16,
    I32,
    I64,
    I128,
    Isize,
    U8,
    U16,
    U32,
    U64,
    U128,
    Usize,
    F32,
    F64,
    /// The Rust 'char'.
    Char32,
    /// The C 'char', not the Rust one.
    Char,
    SChar,
    UChar,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    Void,
}
impl Primitive {
    /// A fully-qualified identifier for the type name.
    pub const fn qualname(&self) -> &'static str {
        /// Helper wrapper around `stringify` that triggers a compiler error if the path is not
        /// valid.  This is just a self-documenting test that there aren't typos in the arms
        /// (because we're unlikely to have API coverage of all of them).
        macro_rules! valid_type {
            ($x:path) => {{
                let _: $x;
                stringify!($x)
            }};
        }

        match self {
            Self::Bool => valid_type!(bool),
            Self::I8 => valid_type!(i8),
            Self::I16 => valid_type!(i16),
            Self::I32 => valid_type!(i32),
            Self::I64 => valid_type!(i64),
            Self::I128 => valid_type!(i128),
            Self::Isize => valid_type!(isize),
            Self::U8 => valid_type!(u8),
            Self::U16 => valid_type!(u16),
            Self::U32 => valid_type!(u32),
            Self::U64 => valid_type!(u64),
            Self::U128 => valid_type!(u128),
            Self::Usize => valid_type!(usize),
            Self::F32 => valid_type!(f32),
            Self::F64 => valid_type!(f64),
            Self::Char32 => valid_type!(char),
            Self::Void => valid_type!(::std::ffi::c_void),
            Self::Char => valid_type!(::std::ffi::c_char),
            Self::SChar => valid_type!(::std::ffi::c_schar),
            Self::UChar => valid_type!(::std::ffi::c_uchar),
            Self::Short => valid_type!(::std::ffi::c_short),
            Self::UShort => valid_type!(::std::ffi::c_ushort),
            Self::Int => valid_type!(::std::ffi::c_int),
            Self::UInt => valid_type!(::std::ffi::c_uint),
            Self::Long => valid_type!(::std::ffi::c_long),
            Self::ULong => valid_type!(::std::ffi::c_ulong),
            Self::LongLong => valid_type!(::std::ffi::c_longlong),
            Self::ULongLong => valid_type!(::std::ffi::c_ulonglong),
            // Can't use `valid_type!` on these because we don't depend on these crates here.
            Self::PyObject => "::pyo3::ffi::PyObject",
            Self::Complex64 => "::num_complex::Complex64",
        }
    }

    /// Convert to an equivalent `unsigned` type, if there is one.
    pub const fn to_unsigned(self) -> Option<Self> {
        match self {
            Self::PyObject
            | Self::Complex64
            | Self::Void
            | Self::Bool
            | Self::F32
            | Self::F64
            | Self::Char32 => None,
            Self::Char | Self::SChar | Self::UChar => Some(Self::UChar),
            Self::I8 | Self::U8 => Some(Self::U8),
            Self::I16 | Self::U16 => Some(Self::U16),
            Self::I32 | Self::U32 => Some(Self::U32),
            Self::I64 | Self::U64 => Some(Self::U64),
            Self::I128 | Self::U128 => Some(Self::U128),
            Self::Isize | Self::Usize => Some(Self::Usize),
            Self::Short | Self::UShort => Some(Self::UShort),
            Self::Int | Self::UInt => Some(Self::UInt),
            Self::Long | Self::ULong => Some(Self::ULong),
            Self::LongLong | Self::ULongLong => Some(Self::ULongLong),
        }
    }

    pub const fn from_cbindgen_intkind(kind: ir::IntKind, signed: bool) -> Self {
        let signed_ty = match kind {
            ir::IntKind::Short => Self::Short,
            ir::IntKind::Int => Self::Int,
            ir::IntKind::Long => Self::Long,
            ir::IntKind::LongLong => Self::LongLong,
            ir::IntKind::SizeT => Self::Isize,
            ir::IntKind::Size => Self::Isize,
            ir::IntKind::B8 => Self::I8,
            ir::IntKind::B16 => Self::I16,
            ir::IntKind::B32 => Self::I32,
            ir::IntKind::B64 => Self::I64,
        };
        if signed {
            signed_ty
        } else {
            signed_ty
                .to_unsigned()
                .expect("all integer types have unsigned variants")
        }
    }

    pub fn try_from_cbindgen_primitive(ty: &ir::PrimitiveType) -> anyhow::Result<Self> {
        match ty {
            ir::PrimitiveType::Void => Ok(Self::Void),
            ir::PrimitiveType::Bool => Ok(Self::Bool),
            ir::PrimitiveType::Char => Ok(Self::Char),
            ir::PrimitiveType::SChar => Ok(Self::SChar),
            ir::PrimitiveType::UChar => Ok(Self::UChar),
            ir::PrimitiveType::Char32 => Ok(Self::Char32),
            ir::PrimitiveType::Float => Ok(Self::F32),
            ir::PrimitiveType::Double => Ok(Self::F64),
            ir::PrimitiveType::VaList => Err(anyhow!("variadic arguments not handled")), // come on.
            ir::PrimitiveType::PtrDiffT => Ok(Self::Isize),
            ir::PrimitiveType::Integer {
                zeroable: _,
                signed,
                kind,
            } => Ok(Self::from_cbindgen_intkind(*kind, *signed)),
        }
    }
}

mod parse {
    use super::Primitive;
    use crate::simple_ir::{self, PtrKind, TypeKind};
    use anyhow::bail;
    use cbindgen::bindgen::ir;

    pub fn r#type(mut ty: &ir::Type) -> anyhow::Result<simple_ir::Type<Primitive>> {
        let mut ptrs = Vec::new();
        let base = loop {
            match ty {
                ir::Type::Ptr {
                    ty: inner,
                    is_const,
                    ..
                } => {
                    ptrs.push(if *is_const {
                        PtrKind::Const
                    } else {
                        PtrKind::Mut
                    });
                    ty = inner;
                }
                ir::Type::Path(p) => {
                    break match p.export_name() {
                        "PyObject" => TypeKind::Builtin(Primitive::PyObject),
                        "QkComplex64" => TypeKind::Builtin(Primitive::Complex64),
                        name => TypeKind::Custom(name.to_owned()),
                    };
                }
                ir::Type::Primitive(ty) => {
                    break TypeKind::Builtin(Primitive::try_from_cbindgen_primitive(ty)?);
                }
                ir::Type::Array(..) => bail!("array types not yet handled"),
                ir::Type::FuncPtr { .. } => bail!("funcptrs not yet handled"),
            }
        };
        Ok(simple_ir::Type { ptrs, base })
    }

    pub fn function(func: &ir::Function) -> anyhow::Result<simple_ir::Function<Primitive>> {
        let name = func.path.name().to_owned();
        let args = func
            .args
            .iter()
            .map(|arg| {
                if arg.array_length.is_some() {
                    bail!("function array arguments not handled yet");
                }
                Ok(simple_ir::FunctionArg {
                    name: arg.name.clone(),
                    ty: r#type(&arg.ty)?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        // `void` in return position is modelled by `()`, but by `std::ffi::c_void` when a pointee.
        let ret = (func.ret != ir::Type::Primitive(ir::PrimitiveType::Void))
            .then(|| r#type(&func.ret))
            .transpose()?;
        Ok(simple_ir::Function { name, args, ret })
    }

    pub fn r#struct(val: &ir::Struct) -> anyhow::Result<simple_ir::Struct<Primitive>> {
        let fields = val
            .fields
            .iter()
            .map(|field| -> anyhow::Result<_> {
                Ok(simple_ir::StructField {
                    name: field.name.clone(),
                    ty: r#type(&field.ty)?,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(simple_ir::Struct {
            name: val.export_name.clone(),
            fields: Some(fields),
        })
    }

    pub fn r#union(val: &ir::Union) -> anyhow::Result<simple_ir::Union<Primitive>> {
        let fields = val
            .fields
            .iter()
            .map(|field| -> anyhow::Result<_> {
                Ok(simple_ir::StructField {
                    name: field.name.clone(),
                    ty: r#type(&field.ty)?,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(simple_ir::Union {
            name: val.export_name.clone(),
            fields,
        })
    }

    /// Extract all objects from a set of `cbindgen::Bindings`, adding them to ourselves.
    ///
    /// This fails if the bindings contain any unsupported constructs.
    pub fn add_items(
        items: &mut simple_ir::Items<Primitive>,
        bindings: &cbindgen::Bindings,
    ) -> anyhow::Result<()> {
        for item in bindings.items.iter() {
            match item {
                ir::ItemContainer::Enum(item) => {
                    items.enums.push(simple_ir::Enum::try_from_cbindgen(item)?)
                }
                ir::ItemContainer::OpaqueItem(item) => items
                    .structs
                    .push(simple_ir::Struct::opaque(item.export_name.clone())),
                ir::ItemContainer::Struct(item) => items.structs.push(r#struct(item)?),
                ir::ItemContainer::Union(item) => items.unions.push(r#union(item)?),
                ir::ItemContainer::Constant(_)
                | ir::ItemContainer::Static(_)
                | ir::ItemContainer::Typedef(_) => {
                    bail!("unhandled item: {item:?}");
                }
            }
        }

        for func in &bindings.functions {
            items.functions.push(function(func)?);
        }

        Ok(())
    }
}

mod export {
    use super::{FN_MACRO, Primitive};
    use crate::simple_ir::{self, PtrKind, TypeKind};
    use anyhow::anyhow;
    use hashbrown::HashMap;

    fn render_type(ty: &simple_ir::Type<Primitive>, out: &mut String) {
        for ptr in ty.ptrs.iter().rev() {
            match ptr {
                PtrKind::Mut => out.push_str("*mut "),
                PtrKind::Const => out.push_str("*const "),
            }
        }
        match &ty.base {
            TypeKind::Builtin(ty) => out.push_str(ty.qualname()),
            TypeKind::Custom(name) => out.push_str(name),
        }
    }

    fn r#enum(val: &simple_ir::Enum) -> String {
        let repr = Primitive::from_cbindgen_intkind(val.repr.kind, val.repr.signed).qualname();
        let name = &val.name;
        let mut out = format!(
            "
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr({repr})]
pub enum {name} {{
"
        );
        for variant in &val.variants {
            out.push_str("    ");
            out.push_str(&variant.name);
            out.push_str(" = ");
            out.push_str(&variant.value);
            out.push_str(",\n");
        }
        out.push('}');
        out
    }

    fn function(
        val: &simple_ir::Function<Primitive>,
        macro_name: &str,
        vtable: &str,
        offset: usize,
    ) -> String {
        // TODO: this function doesn't actually define the function at all; we delegate that to a
        // macro definition in the _receiving_ crate.  This is done so that the complex Rust code of
        // the function is written in an actual Rust file as opposed to being in a string literal
        // inside this file, but we probably want a better story for that in the future.
        let name = &val.name;
        let mut out = format!("{macro_name}!({vtable}[{offset}]; {name}(");
        let mut counter = 0usize..;
        let mut next_arg = || format!("__arg_{}", counter.next().expect("functionally infinite"));
        let mut args = val.args.as_slice();
        if let Some(first) = args.split_off_first() {
            out.push_str(&first.name.clone().unwrap_or_else(&mut next_arg));
            out.push_str(": ");
            render_type(&first.ty, &mut out);
        }
        for arg in args {
            out.push_str(", ");
            out.push_str(&arg.name.clone().unwrap_or_else(&mut next_arg));
            out.push_str(": ");
            render_type(&arg.ty, &mut out);
        }
        out.push(')');
        if let Some(ret) = val.ret.as_ref() {
            out.push_str(" -> ");
            render_type(ret, &mut out);
        }
        out.push_str(");");
        out
    }

    fn r#struct(val: &simple_ir::Struct<Primitive>) -> String {
        let name = &val.name;
        let Some(fields) = val.fields.as_deref() else {
            // In the absence of Rust-stable `extern type`, this is the Nomicon-approved way of
            // defining an opaque type:
            //      https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
            return format!(
                "
#[repr(C)]
pub struct {name} {{
    _private: ::core::marker::PhantomData<(*mut u8, ::core::marker::PhantomPinned)>,
}}"
            );
        };
        let mut out = format!(
            "
#[repr(C)]
pub struct {name} {{"
        );
        for field in fields {
            out.push_str("\n    pub ");
            out.push_str(&field.name);
            out.push_str(": ");
            render_type(&field.ty, &mut out);
            out.push(',');
        }
        out.push_str("\n}");
        out
    }

    fn r#union(val: &simple_ir::Union<Primitive>) -> String {
        let name = &val.name;
        let mut out = format!(
            "
#[repr(C)]
pub union {name} {{"
        );
        for field in &val.fields {
            out.push_str("\n    pub ");
            out.push_str(&field.name);
            out.push_str(": ");
            render_type(&field.ty, &mut out);
            out.push(',');
        }
        out.push_str("\n}");

        out
    }

    pub fn items(
        items: &simple_ir::Items<Primitive>,
        mut out: impl std::io::Write,
    ) -> anyhow::Result<()> {
        writeln!(out, "{}", crate::copyright_with_line_comments("//"))?;
        writeln!(
            out,
            "\
// =====================================
// This file is automatically generated.
// =====================================

use crate::{FN_MACRO};"
        )?;
        for item in &items.enums {
            writeln!(out, "{}", r#enum(item))?;
        }
        for item in &items.structs {
            writeln!(out, "{}", r#struct(item))?;
        }
        for item in &items.unions {
            writeln!(out, "{}", r#union(item))?;
        }
        writeln!(out)?;
        let functions = items
            .functions
            .iter()
            .map(|func| (func.name.clone(), func))
            .collect::<HashMap<_, _>>();
        let capsules = [
            (
                "crate::QK_FFI_CIRCUIT",
                &qiskit_cext_vtable::FUNCTIONS_CIRCUIT,
            ),
            (
                "crate::QK_FFI_TRANSPILE",
                &qiskit_cext_vtable::FUNCTIONS_TRANSPILE,
            ),
            ("crate::QK_FFI_QI", &qiskit_cext_vtable::FUNCTIONS_QI),
        ];
        for (vtable_name, vtable) in capsules {
            for entry in vtable.exports(0) {
                let item = functions
                    .get(entry.name)
                    .ok_or_else(|| anyhow!("failed to find {} in bindings", entry.name))?;
                writeln!(
                    out,
                    "{}",
                    function(item, super::FN_MACRO, vtable_name, entry.slot)
                )?;
            }
        }
        Ok(())
    }
}

pub type Items = simple_ir::Items<Primitive>;
impl Items {
    pub fn add_from_cbindgen(&mut self, bindings: &cbindgen::Bindings) -> anyhow::Result<()> {
        parse::add_items(self, bindings)
    }

    pub fn export(&self, out: impl std::io::Write) -> anyhow::Result<()> {
        export::items(self, out)
    }
}
