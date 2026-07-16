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

pub const REQUIRED_IMPORTS: &[&str] = &["ctypes", "enum"];

/// Numeric primitive types that are representable by `ctypes`.
#[derive(Clone, Copy, Debug)]
pub enum Primitive {
    PyObject,
    VoidP,
    Bool,
    Char,
    SByte,
    UByte,
    Single,
    Double,
    Short,
    UShort,
    Int,
    UInt,
    Long,
    ULong,
    LongLong,
    ULongLong,
    Ssize,
    Size,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}
impl Primitive {
    /// The (module-qualified) `ctypes` constructor function of this primitive.
    pub const fn qualname(&self) -> &'static str {
        match self {
            Self::PyObject => "ctypes.py_object",
            Self::VoidP => "ctypes.c_void_p",
            Self::Bool => "ctypes.c_bool",
            Self::Char => "ctypes.c_char",
            Self::SByte => "ctypes.c_byte",
            Self::UByte => "ctypes.c_ubyte",
            Self::Single => "ctypes.c_float",
            Self::Double => "ctypes.c_double",
            Self::Short => "ctypes.c_short",
            Self::UShort => "ctypes.c_ushort",
            Self::Int => "ctypes.c_int",
            Self::UInt => "ctypes.c_uint",
            Self::Long => "ctypes.c_long",
            Self::ULong => "ctypes.c_ulong",
            Self::LongLong => "ctypes.c_longlong",
            Self::ULongLong => "ctypes.c_ulonglong",
            Self::Ssize => "ctypes.c_ssize_t",
            Self::Size => "ctypes.c_size_t",
            Self::I8 => "ctypes.c_int8",
            Self::U8 => "ctypes.c_uint8",
            Self::I16 => "ctypes.c_int16",
            Self::U16 => "ctypes.c_uint16",
            Self::I32 => "ctypes.c_int32",
            Self::U32 => "ctypes.c_uint32",
            Self::I64 => "ctypes.c_int64",
            Self::U64 => "ctypes.c_uint64",
        }
    }

    /// Convert to an equivalent `unsigned` type, if there is one in `ctypes.`
    pub const fn to_unsigned(self) -> Option<Self> {
        match self {
            Self::PyObject
            | Self::VoidP
            | Self::Bool
            | Self::Single
            | Self::Double
            | Self::Char => None,
            Self::SByte | Self::UByte => Some(Self::UByte),
            Self::Short | Self::UShort => Some(Self::UShort),
            Self::Int | Self::UInt => Some(Self::UInt),
            Self::Long | Self::ULong => Some(Self::ULong),
            Self::LongLong | Self::ULongLong => Some(Self::ULongLong),
            Self::Ssize | Self::Size => Some(Self::Size),
            Self::I8 | Self::U8 => Some(Self::U8),
            Self::I16 | Self::U16 => Some(Self::U16),
            Self::I32 | Self::U32 => Some(Self::U32),
            Self::I64 | Self::U64 => Some(Self::U64),
        }
    }

    pub const fn from_cbindgen_intkind(kind: ir::IntKind, signed: bool) -> Self {
        let signed_ty = match kind {
            ir::IntKind::Short => Self::Short,
            ir::IntKind::Int => Self::Int,
            ir::IntKind::Long => Self::Long,
            ir::IntKind::LongLong => Self::LongLong,
            ir::IntKind::SizeT => Self::Ssize,
            // We _should_ dispatch on `config.usize_is_size_t`, but we can't represent the
            // `intptr_t`/`uintptr_t` types that cbindgen uses as the alternative.
            ir::IntKind::Size => Self::Size,
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
            ir::PrimitiveType::Void => Err(anyhow!("cannot represent `void` without `void *`")),
            ir::PrimitiveType::Bool => Ok(Self::Bool),
            ir::PrimitiveType::Char => Ok(Self::Char),
            ir::PrimitiveType::SChar => Ok(Self::SByte),
            ir::PrimitiveType::UChar => Ok(Self::UByte),
            // Rust's `char` has no direct equivalent, so this the best we can do.
            ir::PrimitiveType::Char32 => Ok(Self::U32),
            ir::PrimitiveType::Float => Ok(Self::Single),
            ir::PrimitiveType::Double => Ok(Self::Double),
            ir::PrimitiveType::VaList => Err(anyhow!("variadic arguments not handled")), // come on.
            // ctypes doesn't actually give us access to a `ptrdiff_t` type , which is obviously the
            // most natural conversion.  We also don't get `intptr_t`, which is the next most
            // likely, so this is a bit of a guess...
            ir::PrimitiveType::PtrDiffT => Ok(Self::Ssize),
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
    use hashbrown::HashMap;

    static VOID: ir::Type = ir::Type::Primitive(ir::PrimitiveType::Void);

    pub fn r#type(
        mut ty: &ir::Type,
        mut override_fn: impl FnMut(&str) -> Option<Primitive>,
    ) -> anyhow::Result<simple_ir::Type<Primitive>> {
        let mut ptrs = Vec::new();
        let base = loop {
            match ty {
                ir::Type::Ptr {
                    ty: inner,
                    is_const,
                    ..
                } => {
                    // ctypes special-cases void pointers to avoid needing a standalone `void`, and of
                    // course `PyObject *` is handled specially.
                    if **inner == VOID {
                        break TypeKind::Builtin(Primitive::VoidP);
                    }
                    if matches!(&**inner, ir::Type::Path(p) if p.name() == "PyObject") {
                        break TypeKind::Builtin(Primitive::PyObject);
                    }
                    ptrs.push(if *is_const {
                        PtrKind::Const
                    } else {
                        PtrKind::Mut
                    });
                    ty = inner;
                }
                ir::Type::Path(p) => {
                    let name = p.export_name();
                    if let Some(primitive) = override_fn(name) {
                        break TypeKind::Builtin(primitive);
                    } else {
                        break TypeKind::Custom(name.to_owned());
                    }
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

    pub fn function(
        func: &ir::Function,
        mut override_fn: impl FnMut(&str) -> Option<Primitive>,
    ) -> anyhow::Result<simple_ir::Function<Primitive>> {
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
                    ty: r#type(&arg.ty, &mut override_fn)?,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let ret = (func.ret != VOID)
            .then(|| r#type(&func.ret, override_fn))
            .transpose()?;
        Ok(simple_ir::Function { name, args, ret })
    }

    pub fn r#struct(
        val: &ir::Struct,
        mut override_fn: impl FnMut(&str) -> Option<Primitive>,
    ) -> anyhow::Result<simple_ir::Struct<Primitive>> {
        let fields = val
            .fields
            .iter()
            .map(|field| -> anyhow::Result<_> {
                Ok(simple_ir::StructField {
                    name: field.name.clone(),
                    ty: r#type(&field.ty, &mut override_fn)?,
                })
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(simple_ir::Struct {
            name: val.export_name.clone(),
            fields: Some(fields),
        })
    }

    pub fn r#union(
        val: &ir::Union,
        mut override_fn: impl FnMut(&str) -> Option<Primitive>,
    ) -> anyhow::Result<simple_ir::Union<Primitive>> {
        let fields = val
            .fields
            .iter()
            .map(|field| -> anyhow::Result<_> {
                Ok(simple_ir::StructField {
                    name: field.name.clone(),
                    ty: r#type(&field.ty, &mut override_fn)?,
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
        let constructor =
            |repr: &ir::ReprType| Primitive::from_cbindgen_intkind(repr.kind, repr.signed);
        let mut overrides = items
            .enums
            .iter()
            .map(|val| (val.name.clone(), constructor(&val.repr)))
            .collect::<HashMap<_, _>>();

        for item in bindings.items.iter() {
            match item {
                ir::ItemContainer::Enum(item) => {
                    let val = simple_ir::Enum::try_from_cbindgen(item)?;
                    overrides.insert(val.name.clone(), constructor(&val.repr));
                    items.enums.push(val);
                }
                ir::ItemContainer::OpaqueItem(item) => items
                    .structs
                    .push(simple_ir::Struct::opaque(item.export_name.clone())),
                ir::ItemContainer::Struct(item) => items
                    .structs
                    .push(r#struct(item, |path| overrides.get(path).copied())?),
                ir::ItemContainer::Union(item) => items
                    .unions
                    .push(r#union(item, |path| overrides.get(path).copied())?),
                ir::ItemContainer::Constant(_)
                | ir::ItemContainer::Static(_)
                | ir::ItemContainer::Typedef(_) => {
                    bail!("unhandled item: {item:?}");
                }
            }
        }

        for func in &bindings.functions {
            items
                .functions
                .push(function(func, |path| overrides.get(path).copied())?);
        }

        Ok(())
    }
}

mod export {
    use super::Primitive;
    use crate::simple_ir::{self, TypeKind};

    /// Write this type into an existing string.
    fn render_type(ty: &simple_ir::Type<Primitive>, out: &mut String) {
        let name = match &ty.base {
            TypeKind::Builtin(p) => p.qualname(),
            TypeKind::Custom(c) => c.as_str(),
        };
        for _ in &ty.ptrs {
            out.push_str("ctypes.POINTER(");
        }
        out.push_str(name);
        for _ in &ty.ptrs {
            out.push(')');
        }
    }

    /// Get a string that declares this `Enum` as a Python class.
    ///
    /// We export a corresponding `class <Name>(enum.Enum)` to Python with a named attribute for each of
    /// the variants for ease of use in Python space, but in actual function interfaces we have to use
    /// the raw `ctypes` backing type to get the ABI correct.
    pub fn r#enum(val: &simple_ir::Enum) -> String {
        let mut out = format!("\nclass {}(enum.Enum):", val.name);
        if val.variants.is_empty() {
            out.push_str("\n    pass");
            return out;
        }
        let ctypes_repr = Primitive::from_cbindgen_intkind(val.repr.kind, val.repr.signed);
        let constructor = ctypes_repr.qualname();
        for simple_ir::EnumVariant { name, value } in &val.variants {
            out.push_str(&format!("\n    {} = {}({})", name, constructor, value));
        }
        out
    }

    /// Declare the argument and return types of this function.
    ///
    /// If `dllname` name is given, we assume the function is an attribute on an object with that
    /// name.
    pub fn function(val: &simple_ir::Function<Primitive>, dllname: &str) -> String {
        let prefix = format!("{}.{}", dllname, val.name);
        let mut out = format!("\n{}.argtypes = [", prefix);
        if let Some((first, rest)) = val.args.split_first() {
            render_type(&first.ty, &mut out);
            for arg in rest {
                out.push_str(", ");
                render_type(&arg.ty, &mut out);
            }
        }
        out.push_str("]\n");
        out.push_str(&prefix);
        out.push_str(".restype = ");
        match val.ret.as_ref() {
            Some(ret) => render_type(ret, &mut out),
            None => out.push_str("None"),
        }
        out.push('\n');

        // Re-export into main namespace.
        out.push_str(&val.name);
        out.push_str(" = ");
        out.push_str(&prefix);
        out
    }

    /// Get a string representing the declaration of this `struct` as a Python `ctypes.Structure`.
    pub fn r#struct(val: &simple_ir::Struct<Primitive>) -> String {
        // TODO: this doesn't handle the case of (mutually) recursive `struct` definitions; we
        // assume that all the `_fields_` will refer to fully defined `ctypes` objects.
        let mut out = format!("\nclass {}(ctypes.Structure):\n", val.name);
        let Some(fields) = val.fields.as_ref() else {
            out.push_str("    pass\n");
            return out;
        };
        out.push_str("    _fields_ = [\n");
        for simple_ir::StructField { name, ty } in fields {
            out.push_str("        (\"");
            out.push_str(name);
            out.push_str("\", ");
            render_type(ty, &mut out);
            out.push_str("),\n");
        }
        out.push_str("    ]");
        out
    }

    /// Get a string representing the declaration of this `union`` as a Python `ctypes.Union`.
    pub fn r#union(val: &simple_ir::Union<Primitive>) -> String {
        let mut out = format!("\nclass {}(ctypes.Union):\n", val.name);
        out.push_str("    _fields_ = [\n");

        for simple_ir::StructField { name, ty } in &val.fields {
            out.push_str("        (\"");
            out.push_str(name);
            out.push_str("\", ");
            render_type(ty, &mut out);
            out.push_str("),\n");
        }
        out.push_str("    ]");
        out
    }

    /// Export this complete set of objects, subject to the given configuration.
    ///
    /// This includes an `__all__`, a suitable set of `import` statements, a `PyDLL`, and then all
    /// the objects to declare.
    pub fn items(
        items: &simple_ir::Items<Primitive>,
        config: &super::Config,
        mut out: impl std::io::Write,
    ) -> anyhow::Result<()> {
        if let Some(header) = config.header.as_deref() {
            writeln!(out, "{}", header)?;
        }
        writeln!(out)?;

        writeln!(out, "__all__ = [")?;
        writeln!(out, "    \"{}\",", config.dll.name)?;
        for val in &items.enums {
            writeln!(out, "    \"{}\",", val.name)?;
        }
        for val in &items.structs {
            writeln!(out, "    \"{}\",", val.name)?;
        }
        for val in &items.functions {
            writeln!(out, "    \"{}\",", val.name)?;
        }
        writeln!(out, "]")?;
        writeln!(out)?;

        for import in super::REQUIRED_IMPORTS {
            writeln!(out, "import {import}")?;
        }
        for import in &config.imports {
            if !super::REQUIRED_IMPORTS.contains(&import.as_str()) {
                writeln!(out, "import {import}")?;
            }
        }

        writeln!(out)?;
        writeln!(
            out,
            "{} = ctypes.PyDLL({})",
            config.dll.name, config.dll.expr
        )?;

        writeln!(out)?;
        for item in &items.enums {
            writeln!(out, "{}", r#enum(item))?;
        }
        for item in &items.unions {
            writeln!(out, "{}", r#union(item))?;
        }
        for item in &items.structs {
            writeln!(out, "{}", r#struct(item))?;
        }
        for item in &items.functions {
            writeln!(out, "{}", function(item, &config.dll.name))?;
        }

        Ok(())
    }
}

/// Configuration of how to declare a `PyDLL` object containing the `ctypes` bindings.
#[derive(Clone, Debug)]
pub struct DllDeclaration {
    /// The name of the attribute to assign the `PyDLL` to.
    pub name: String,
    /// An expression that results in a filename that can be given to `ctypes.PyDLL`.
    pub expr: String,
}

/// Configuration of the `ctypes` export process.
#[derive(Clone, Debug)]
pub struct Config {
    /// An arbitrary string to write at the start of the file.
    pub header: Option<String>,
    /// Any additional imports to define, such as those needed for the [`DllDeclaration::expr`].
    pub imports: Vec<String>,
    pub dll: DllDeclaration,
}

pub type Items = simple_ir::Items<Primitive>;
impl Items {
    pub fn implicit() -> Self {
        let field = |name: &str| simple_ir::StructField {
            name: name.to_owned(),
            ty: simple_ir::Type {
                ptrs: Vec::new(),
                base: simple_ir::TypeKind::Builtin(Primitive::Double),
            },
        };
        let complex = simple_ir::Struct {
            name: "QkComplex64".to_owned(),
            fields: Some(vec![field("re"), field("im")]),
        };
        Items {
            enums: vec![],
            structs: vec![complex],
            functions: vec![],
            unions: vec![],
        }
    }

    pub fn add_from_cbindgen(&mut self, bindings: &cbindgen::Bindings) -> anyhow::Result<()> {
        parse::add_items(self, bindings)
    }

    pub fn export(&self, config: &Config, out: impl std::io::Write) -> anyhow::Result<()> {
        export::items(self, config, out)
    }
}
