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

use cbindgen::bindgen::ir;
use hashbrown::HashMap;

/// Render a given type object into a string representing it in C.
fn render_type(ty: &ir::Type, config: &cbindgen::Config) -> String {
    fn render(ty: &ir::Type, config: &cbindgen::Config, acc: &mut String) {
        match ty {
            ir::Type::Ptr {
                ty,
                is_const,
                is_nullable: _,
                is_ref,
            } => {
                assert!(!is_ref, "C++ reference-likes not handled");
                if *is_const {
                    acc.push_str("const ");
                }
                render(ty, config, acc);
                acc.push_str(" *");
            }
            ir::Type::Path(p) => acc.push_str(p.export_name()),
            ir::Type::Primitive(ty) => acc.push_str(ty.to_repr_c(config)),
            ir::Type::Array(..) => todo!("array types not yet handled"),
            ir::Type::FuncPtr {
                args,
                ret,
                is_nullable,
                never_return,
            } => {
                assert!(!is_nullable, "nullability of funcptrs is not handled");
                assert!(!never_return, "diverging functions not handled");
                render(ret, config, acc);
                acc.push_str("(*)(");
                let mut args = args.iter();
                if let Some((_, first)) = args.next() {
                    render(first, config, acc);
                    for (_, arg) in args {
                        acc.push_str(", ");
                        render(arg, config, acc)
                    }
                }
                acc.push(')');
            }
        }
    }
    let mut acc = String::new();
    render(ty, config, &mut acc);
    acc
}

/// Calculate a mapping of exported function names to C casts to appropriate function-pointer types.
pub fn functions_as_funcptr_casts(bindings: &cbindgen::Bindings) -> HashMap<&str, String> {
    let to_funcptr = |func: &ir::Function| {
        let to_funcptr_arg = |arg: &ir::FunctionArgument| {
            let ir::FunctionArgument {
                name: _,
                ty,
                array_length,
            } = arg;
            assert!(array_length.is_none(), "array arguments not handled");
            (None, ty.clone())
        };
        ir::Type::FuncPtr {
            ret: Box::new(func.ret.clone()),
            args: func.args.iter().map(to_funcptr_arg).collect(),
            is_nullable: false,
            never_return: false,
        }
    };
    let config = &bindings.config;
    bindings
        .functions
        .iter()
        .map(|func| {
            let funcptr = to_funcptr(func);
            (func.path.name(), render_type(&funcptr, config))
        })
        .collect()
}
