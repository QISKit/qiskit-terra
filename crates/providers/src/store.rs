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

use crate::data_tree::DataTree;
use crate::program_node::ProgramNode;
use crate::tensor::{Tensor, TensorType};
use std::sync::LazyLock;

// An empty data tree is the input for all store nodes.
static EMPTY_DATA_TREE: LazyLock<DataTree<TensorType>> = LazyLock::new(DataTree::new);

/// A program node that owns constant data and outputs it unconditionally.
///
/// `Store` takes no inputs; its `call()` always returns the data it was constructed with.
/// In a data-flow graph, `Store` nodes play the role of constants — they are wired to
/// the input ports of computation nodes to supply fixed values.
pub struct Store {
    /// Tensors in DFS leaf order matching `output_types`.
    leaves: Vec<Tensor>,
    output_types: DataTree<TensorType>,
}

impl Store {
    /// Construct a new `Store` holding the given data.
    pub fn new(data: DataTree<Tensor>) -> Self {
        let output_types = data.map_leaves(Tensor::tensor_type);
        let leaves: Vec<Tensor> = data.into_leaves().collect();
        Self {
            leaves,
            output_types,
        }
    }
}

impl ProgramNode for Store {
    type CallError = std::convert::Infallible;

    fn name(&self) -> &str {
        "store"
    }

    fn namespace(&self) -> &str {
        "qiskit"
    }

    fn input_types(&self) -> &DataTree<TensorType> {
        &EMPTY_DATA_TREE
    }

    fn output_types(&self) -> &DataTree<TensorType> {
        &self.output_types
    }

    fn implements_call(&self) -> bool {
        true
    }

    fn call_flat(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::CallError> {
        Ok(self.leaves.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, DTypeLike, Dim, Tensor};

    #[test]
    fn test_store_leaf_call() {
        let data = DataTree::new_leaf(Tensor::from([1.0_f64, 2.0, 3.0]));
        let store = Store::new(data);
        let result = store.call_flat(&[]).unwrap();
        assert_eq!(result.len(), 1);
        let Tensor::F64(arr) = &result[0] else {
            panic!("expected f64 leaf");
        };
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_store_output_types_2d() {
        use ndarray::arr2;
        let data = DataTree::new_leaf(Tensor::F64(
            arr2(&[[1.0_f64, 2.0], [3.0, 4.0]]).into_dyn().into_shared(),
        ));
        let store = Store::new(data);
        let DataTree::Leaf(tt) = store.output_types() else {
            panic!("expected leaf output type");
        };
        assert_eq!(tt.shape, vec![Dim::Fixed(2), Dim::Fixed(2)]);
    }

    #[test]
    fn test_store_branched() {
        let mut data = DataTree::new();
        data.insert_leaf("a", Tensor::from([1.0_f64, 2.0]));
        data.insert_leaf("b", Tensor::from([10_i32, 20, 30]));
        let store = Store::new(data);

        assert!(store.input_types().is_empty());
        assert_eq!(store.name(), "store");
        assert_eq!(store.namespace(), "qiskit");
        assert_eq!(store.full_name(), "qiskit.store");

        let out_types = store.output_types();
        let DataTree::Leaf(tt_a) = out_types.get_by_str_key("a").unwrap() else {
            panic!("expected leaf at a");
        };
        assert!(matches!(tt_a.dtype, DTypeLike::Concrete(DType::F64)));
        assert_eq!(tt_a.shape, vec![Dim::Fixed(2)]);

        let DataTree::Leaf(tt_b) = out_types.get_by_str_key("b").unwrap() else {
            panic!("expected leaf at b");
        };
        assert!(matches!(tt_b.dtype, DTypeLike::Concrete(DType::I32)));
        assert_eq!(tt_b.shape, vec![Dim::Fixed(3)]);
    }

    #[test]
    fn test_store_branched_call_returns_flat_in_dfs_order() {
        let mut data = DataTree::new();
        data.insert_leaf("a", Tensor::from([1.0_f64]));
        data.insert_leaf("b", Tensor::from([2.0_f64]));
        let store = Store::new(data);
        let result = store.call_flat(&[]).unwrap();
        assert_eq!(result.len(), 2);
        let Tensor::F64(arr_a) = &result[0] else {
            panic!()
        };
        let Tensor::F64(arr_b) = &result[1] else {
            panic!()
        };
        assert_eq!(arr_a.as_slice().unwrap(), &[1.0]);
        assert_eq!(arr_b.as_slice().unwrap(), &[2.0]);
    }
}
