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

//! The element type of a tensor, and various associated rules.

use std::fmt;

/// The possible data types for a Tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    C128, // complex
    C64,
    F64, // real
    F32,
    I64, // signed integer
    I32,
    I16,
    I8,
    U64, // unsigned integer
    U32,
    U16,
    U8,
    Bit, // bool
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let string_repr = match self {
            DType::C128 => "C128",
            DType::C64 => "C64",
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U64 => "U64",
            DType::U32 => "U32",
            DType::U16 => "U16",
            DType::U8 => "U8",
            DType::Bit => "Bit",
        };
        write!(f, "{string_repr}")
    }
}
/// Promote a pair of dtypes to the smallest type compatible with both.
///
/// This function implements the same promotion rules as NumPy, modulo that we don't
/// need to contend with the arbitrary precision types for each type kind, and that
/// we omit F16 entirely because it's unstable in Rust:
/// <https://numpy.org/doc/stable/reference/arrays.promotion.html#numerical-promotion>
/// In short, if you view the linked diagram as a DAG, this function hard-codes the
/// least-common-descendant algorithm.
pub fn promotion(lhs: DType, rhs: DType) -> DType {
    use DType::*;

    match lhs {
        C128 => C128,

        C64 => match rhs {
            U32 | U64 | I32 | I64 | F64 | C128 => C128,
            _ => C64,
        },

        F64 => match rhs {
            C64 | C128 => C128,
            _ => F64,
        },

        F32 => match rhs {
            C128 => C128,
            C64 => C64,
            U32 | U64 | I32 | I64 | F64 => F64,
            _ => F32,
        },

        I64 => match rhs {
            C64 | C128 => C128,
            U64 | F32 | F64 => F64,
            _ => I64,
        },

        I32 => match rhs {
            C64 | C128 => C128,
            U64 | F32 | F64 => F64,
            U32 | I64 => I64,
            _ => I32,
        },

        I16 => match rhs {
            U64 => F64,
            U32 => I64,
            U16 => I32,
            Bit | U8 | I8 => I16,
            _ => rhs,
        },

        I8 => match rhs {
            U64 => F64,
            U32 => I64,
            U16 => I32,
            U8 => I16,
            Bit => I8,
            _ => rhs,
        },

        U64 => match rhs {
            C128 | C64 => C128,
            F32 | F64 | I8 | I16 | I32 | I64 => F64,
            _ => U64,
        },

        U32 => match rhs {
            C64 | C128 => C128,
            F32 | F64 => F64,
            I8 | I16 | I32 | I64 => I64,
            U64 => U64,
            _ => U32,
        },

        U16 => match rhs {
            I8 | I16 => I32,
            Bit | U8 => U16,
            _ => rhs,
        },

        U8 => match rhs {
            I8 => I16,
            Bit => U8,
            _ => rhs,
        },

        Bit => rhs,
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const ALL_DTYPES: [DType; 13] = [
        DType::Bit,
        DType::U8,
        DType::U16,
        DType::U32,
        DType::U64,
        DType::I8,
        DType::I16,
        DType::I32,
        DType::I64,
        DType::F32,
        DType::F64,
        DType::C64,
        DType::C128,
    ];

    #[test]
    fn test_promotion_against_promotion_dag() {
        use DType::*;
        use hashbrown::{HashMap, HashSet};
        use rustworkx_core::dag_algo::lexicographical_topological_sort;
        use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
        use rustworkx_core::traversal::descendants;

        // define a DAG that implements all promotion rules; two DTypes
        // should be promoted to their least common descendant in the DAG
        let mut g: DiGraph<DType, ()> = DiGraph::new();
        let mut idx: HashMap<DType, NodeIndex> = HashMap::new();

        for &dtype in &ALL_DTYPES {
            idx.insert(dtype, g.add_node(dtype));
        }

        // within-kind promotions
        g.add_edge(idx[&U8], idx[&U16], ());
        g.add_edge(idx[&U16], idx[&U32], ());
        g.add_edge(idx[&U32], idx[&U64], ());

        g.add_edge(idx[&I8], idx[&I16], ());
        g.add_edge(idx[&I16], idx[&I32], ());
        g.add_edge(idx[&I32], idx[&I64], ());

        g.add_edge(idx[&F32], idx[&F64], ());

        g.add_edge(idx[&C64], idx[&C128], ());

        // bit promotions
        g.add_edge(idx[&Bit], idx[&U8], ());
        g.add_edge(idx[&Bit], idx[&I8], ());

        // uint promotions
        g.add_edge(idx[&U8], idx[&I16], ());
        g.add_edge(idx[&U16], idx[&I32], ());
        g.add_edge(idx[&U16], idx[&F32], ());
        g.add_edge(idx[&U32], idx[&I64], ());
        g.add_edge(idx[&U64], idx[&F64], ());

        // int promotions
        g.add_edge(idx[&I16], idx[&F32], ());
        g.add_edge(idx[&I32], idx[&F64], ());
        g.add_edge(idx[&I64], idx[&F64], ());

        // float promotions
        g.add_edge(idx[&F32], idx[&C64], ());
        g.add_edge(idx[&F64], idx[&C128], ());

        let order = lexicographical_topological_sort(
            &g,
            |n: NodeIndex| Ok::<usize, std::convert::Infallible>(n.index()),
            false,
            None,
        )
        .unwrap();

        let least_common_descendant = move |a: &DType, b: &DType| -> DType {
            let da: HashSet<_> = descendants(&g, idx[a]).collect();
            let db: HashSet<_> = descendants(&g, idx[b]).collect();
            let common: HashSet<NodeIndex> = da.intersection(&db).copied().collect();
            let least_idx = order.iter().find(|n| common.contains(*n)).unwrap();
            ALL_DTYPES[least_idx.index()]
        };

        for &a in &ALL_DTYPES {
            for &b in &ALL_DTYPES {
                assert_eq!(
                    promotion(a, b),
                    least_common_descendant(&a, &b),
                    "For promotion ({a}, {b})"
                )
            }
        }
    }

    #[test]
    fn test_promotion_idempotence() {
        for &a in &ALL_DTYPES {
            assert_eq!(promotion(a, a), a, "For promotion ({a}, {a})")
        }
    }

    #[test]
    fn test_promotion_commutativity() {
        for &a in &ALL_DTYPES {
            for &b in &ALL_DTYPES {
                assert_eq!(promotion(a, b), promotion(b, a), "For promotion ({a}, {b})")
            }
        }
    }

    #[test]
    fn test_dtype_display() {
        use DType::*;
        let cases = [
            (C128, "C128"),
            (C64, "C64"),
            (F64, "F64"),
            (F32, "F32"),
            (I64, "I64"),
            (I32, "I32"),
            (I16, "I16"),
            (I8, "I8"),
            (U64, "U64"),
            (U32, "U32"),
            (U16, "U16"),
            (U8, "U8"),
            (Bit, "Bit"),
        ];
        let mut fails = vec![];
        for (dtype, expected) in cases {
            let got = format!("{dtype}");
            if got != expected {
                fails.push((dtype, expected, got));
            }
        }
        assert_eq!(fails, [], "DType Display mismatches: {fails:?}");
    }
}
