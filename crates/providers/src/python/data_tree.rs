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

//! The Python binding of the data tree.

use hashbrown::HashSet;
use numpy::{PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList, PyString, PyTuple};

use crate::data_tree::DataTree;
use crate::python::value_error;

/// A data tree whose leaves are arbitrary Python objects.
pub(super) type ObjectTree = DataTree<Py<PyAny>>;

/// A leaf holding one value, or a branch of ordered children.
///
/// Constructing a tree parses `object`: a `list` or `tuple` becomes a branch of unnamed children,
/// a `dict` becomes a branch named by its keys in insertion order, and a namedtuple becomes a
/// branch named by its fields. Anything else is a leaf, unless it defines `__datatree__()`, which
/// is called to obtain the tree to use in its place. A tree holds no record of which container a
/// name or a position came from. `leaf_of()` holds any value as a leaf, without parsing it.
///
/// A child is addressed by its name or by its position, and any value below it by a dotted path of
/// the two. Reading a child gives the value of a leaf, or a `DataTree` over a subtree::
///
///     tree = DataTree({"counts": [3, 4], "ev": 0.1})
///     tree["ev"]         # 0.1
///     tree["counts"]     # DataTree([3, 4])
///     tree["counts"][1]  # 4
///     tree["counts.1"]   # 4, the same value by one path
///     tree[-1]           # 0.1
///
/// Length, iteration and `in` cover the children of a branch, of which a leaf has none: its value
/// is read through `leaf`, with `is_leaf` to ask first. A branch names all of its children or none
/// of them, and `is_mapping` says which. One that names them has a mapping form, described by
/// `keys()`, `values()` and `items()`, so `dict(tree)` and `**tree` work. Those three raise on a
/// branch that has none.
#[pyclass(name = "DataTree", module = "qiskit.quantum_program", frozen, mapping)]
pub struct PyDataTree(pub(super) ObjectTree);

#[pymethods]
impl PyDataTree {
    // The object to parse is positional, so the parameter name is not part of the API.
    #[new]
    #[pyo3(signature = (object, /))]
    fn new(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(parse(object)?))
    }

    /// Return a leaf holding `value`, which is not parsed.
    ///
    /// Args:
    ///     value: The value to hold.
    ///
    /// Returns:
    ///     A leaf holding it.
    #[staticmethod]
    #[pyo3(signature = (value, /))]
    fn leaf_of(value: &Bound<'_, PyAny>) -> Self {
        Self(DataTree::new_leaf(value.clone().unbind()))
    }

    /// Whether this is a leaf, as opposed to a branch of children.
    #[getter]
    fn is_leaf(&self) -> bool {
        matches!(self.0, DataTree::Leaf(_))
    }

    /// The value this leaf holds. Raises `TypeError` on a branch.
    #[getter]
    fn leaf(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.0 {
            DataTree::Leaf(value) => Ok(value.clone_ref(py)),
            DataTree::Branch(_) => Err(PyTypeError::new_err(
                "a branch of a data tree has no leaf value",
            )),
        }
    }

    /// Whether this branch names its children. A branch that names them has a mapping form, and a
    /// leaf has no children to name.
    #[getter]
    fn is_mapping(&self) -> PyResult<bool> {
        if self.is_leaf() {
            return Ok(false);
        }
        branch_is_mapping(&self.children())
    }

    /// Return the number of children, of which a leaf has none.
    fn __len__(&self) -> usize {
        match &self.0 {
            DataTree::Leaf(_) => 0,
            branch @ DataTree::Branch(_) => branch.len(),
        }
    }

    /// Iterate over the children, in order.
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let children = self
            .children()
            .into_iter()
            .map(|(_, child)| child_object(py, child))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, children)?.try_iter()
    }

    /// Return the value `key` addresses.
    ///
    /// Args:
    ///     key: A dotted path if it is a string, and a position among the children if it is an
    ///         integer, counting back from the end when negative.
    ///
    /// Returns:
    ///     The value if it is a leaf, and a data tree if it is a branch.
    ///
    /// Raises:
    ///     KeyError: If no value sits at that path.
    ///     IndexError: If no child sits at that position.
    ///     TypeError: If the key is neither a string nor an integer.
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        child_object(key.py(), self.addressed(key)?)
    }

    /// Return whether `key` addresses a value, reading it as `__getitem__` does.
    ///
    /// Args:
    ///     key: A dotted path, or a position among the children.
    ///
    /// Returns:
    ///     Whether anything sits there.
    ///
    /// Raises:
    ///     TypeError: If the key is neither a string nor an integer.
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = key.py();
        match self.addressed(key) {
            Ok(_) => Ok(true),
            Err(err)
                if err.is_instance_of::<PyKeyError>(py)
                    || err.is_instance_of::<PyIndexError>(py) =>
            {
                Ok(false)
            }
            Err(err) => Err(err),
        }
    }

    /// Return the name of every child, in order.
    ///
    /// Returns:
    ///     The names, in the order the children sit in.
    ///
    /// Raises:
    ///     TypeError: If this is a leaf, or a branch naming none of its children.
    fn keys(&self) -> PyResult<Vec<String>> {
        Ok(self
            .named_children()?
            .into_iter()
            .map(|(name, _)| name.to_owned())
            .collect())
    }

    /// Return every child, in order.
    ///
    /// Returns:
    ///     The value of each child that is a leaf, and a data tree over each that is a branch.
    ///
    /// Raises:
    ///     TypeError: If this is a leaf, or a branch naming none of its children.
    fn values(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.named_children()?
            .into_iter()
            .map(|(_, child)| child_object(py, child))
            .collect()
    }

    /// Return every child with its name, in order.
    ///
    /// Returns:
    ///     A pair per child, of its name and the value `values()` gives for it.
    ///
    /// Raises:
    ///     TypeError: If this is a leaf, or a branch naming none of its children.
    fn items(&self, py: Python<'_>) -> PyResult<Vec<(String, Py<PyAny>)>> {
        self.named_children()?
            .into_iter()
            .map(|(name, child)| Ok((name.to_owned(), child_object(py, child)?)))
            .collect()
    }

    /// Return whether `other` is a data tree put together the same way, name for name, holding
    /// equal leaves. Two trees differing only in which children are named are unequal.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.cast::<Self>() else {
            return Ok(false);
        };
        trees_equal(other.py(), &self.0, &other.get().0)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("DataTree({})", render(py, &self.0)?))
    }
}

impl PyDataTree {
    /// Return the subtree `key` addresses: a dotted path if it is a string, and a position among
    /// the children if it is an integer, counting back from the end when negative.
    ///
    /// A path segment of digits addresses by position and any other segment by name, which is what
    /// makes a name with a dot in it, or one of all digits, invalid.
    fn addressed(&self, key: &Bound<'_, PyAny>) -> PyResult<&ObjectTree> {
        if let Ok(path) = key.cast::<PyString>() {
            let path = path.to_str()?;
            // An empty path resolves to the whole tree, which is not a value it holds.
            return (!path.is_empty())
                .then(|| self.0.get_by_str_key(path))
                .flatten()
                .ok_or_else(|| PyKeyError::new_err(path.to_owned()));
        }
        if let Ok(position) = key.extract::<isize>() {
            let children = self.children();
            let index = if position < 0 {
                position + children.len() as isize
            } else {
                position
            };
            return usize::try_from(index)
                .ok()
                .filter(|&index| index < children.len())
                .map(|index| children[index].1)
                .ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "position {position} addresses nothing in a data tree with {} children",
                        children.len()
                    ))
                });
        }
        Err(PyTypeError::new_err(format!(
            "a data tree addresses a value by path or by position, not by {}",
            key.get_type().name()?
        )))
    }

    /// Return the children of a branch, each with its name where it has one. A leaf has none.
    fn children(&self) -> Vec<(Option<&str>, &ObjectTree)> {
        match &self.0 {
            DataTree::Leaf(_) => Vec::new(),
            branch @ DataTree::Branch(_) => branch.iter_children().collect(),
        }
    }

    /// Return the children of a branch that has a mapping form, each with its name, or the error
    /// saying why the tree has none.
    fn named_children(&self) -> PyResult<Vec<(&str, &ObjectTree)>> {
        if self.is_leaf() {
            return Err(PyTypeError::new_err(
                "a leaf of a data tree has no mapping form",
            ));
        }
        let children = self.children();
        if !branch_is_mapping(&children)? {
            return Err(PyTypeError::new_err(
                "a data tree branch of unnamed children has no mapping form",
            ));
        }
        Ok(children
            .into_iter()
            .map(|(name, child)| (name.expect("the branch names its children"), child))
            .collect())
    }
}

/// Return whether the branch holding `children` names them. A branch that names them has a mapping
/// form.
///
/// A branch names all of its children or none of them. One mixing the two has neither a mapping
/// form nor a sequence form, and is rejected. The Rust type still permits such a branch, and
/// nothing builds one.
fn branch_is_mapping(children: &[(Option<&str>, &ObjectTree)]) -> PyResult<bool> {
    let named = children.iter().filter(|(name, _)| name.is_some()).count();
    if named == children.len() {
        Ok(true)
    } else if named == 0 {
        Ok(false)
    } else {
        Err(PyTypeError::new_err(
            "a data tree branch mixing named and unnamed children has neither a mapping form nor a \
             sequence form",
        ))
    }
}

/// Parse `object` into a data tree, as [`PyDataTree`] documents.
pub(super) fn parse(object: &Bound<'_, PyAny>) -> PyResult<ObjectTree> {
    let py = object.py();
    if let Ok(tree) = object.cast::<PyDataTree>() {
        return Ok(tree.get().0.clone());
    }
    // The hook is looked up on the type, as a special method is, so an object with a permissive
    // `__getattr__` still parses as a leaf.
    match object.get_type().getattr(intern!(py, "__datatree__")) {
        Ok(hook) => {
            let decomposed = hook.call1((object,))?;
            let Ok(tree) = decomposed.cast::<PyDataTree>() else {
                return Err(PyTypeError::new_err(format!(
                    "__datatree__ returned {}, which is not a DataTree",
                    decomposed.get_type().name()?
                )));
            };
            return Ok(tree.get().0.clone());
        }
        Err(err) if !err.is_instance_of::<PyAttributeError>(py) => return Err(err),
        Err(_) => {}
    }
    // A namedtuple is a tuple, so it has to be recognised before the tuple case takes it and
    // loses its field names.
    if let Ok(tuple) = object.cast::<PyTuple>()
        && let Ok(fields) = tuple.getattr(intern!(py, "_fields"))
    {
        let fields = fields.extract::<Vec<String>>()?;
        // A repeated name would address one child, so the branch covers the tuple only with one
        // distinct name per item.
        let distinct = fields.iter().collect::<HashSet<_>>().len();
        if fields.len() != tuple.len() || distinct != fields.len() {
            return Err(PyValueError::new_err(format!(
                "_fields does not name each of the {} items exactly once: {fields:?}",
                tuple.len()
            )));
        }
        let mut tree = DataTree::with_capacity(fields.len());
        for (field, item) in fields.iter().zip(tuple.iter()) {
            tree.insert_branch(field, parse(&item)?)
                .map_err(|err| value_error(&err))?;
        }
        return Ok(tree);
    }
    if object.is_instance_of::<PyList>() || object.is_instance_of::<PyTuple>() {
        let children = object
            .try_iter()?
            .map(|item| parse(&item?))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(DataTree::sequence(children));
    }
    if let Ok(dict) = object.cast::<PyDict>() {
        let mut tree = DataTree::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let Ok(key) = key.cast::<PyString>() else {
                return Err(PyTypeError::new_err(format!(
                    "a data tree names a child with a string, not with {}",
                    key.get_type().name()?
                )));
            };
            tree.insert_branch(key.to_str()?, parse(&value)?)
                .map_err(|err| value_error(&err))?;
        }
        return Ok(tree);
    }
    Ok(DataTree::new_leaf(object.clone().unbind()))
}

/// Return a child as Python sees it: the value of a leaf, or a data tree over a subtree.
fn child_object(py: Python<'_>, child: &ObjectTree) -> PyResult<Py<PyAny>> {
    match child {
        DataTree::Leaf(value) => Ok(value.clone_ref(py)),
        branch @ DataTree::Branch(_) => Ok(Py::new(py, PyDataTree(branch.clone()))?.into_any()),
    }
}

/// Return whether two trees are put together the same way, name for name, and hold equal leaves.
fn trees_equal(py: Python<'_>, left: &ObjectTree, right: &ObjectTree) -> PyResult<bool> {
    match (left, right) {
        (DataTree::Leaf(left), DataTree::Leaf(right)) => {
            // Identical leaves are equal without being compared, as a `list` or a `dict`
            // compares its own contents.
            let (left, right) = (left.bind(py), right.bind(py));
            Ok(left.is(right) || left.eq(right)?)
        }
        (DataTree::Branch(_), DataTree::Branch(_)) => {
            if left.len() != right.len() {
                return Ok(false);
            }
            for ((left_name, left), (right_name, right)) in
                left.iter_children().zip(right.iter_children())
            {
                if left_name != right_name || !trees_equal(py, left, right)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// The largest number of elements an array leaf can hold and still be rendered in full.
const MAX_RENDERED_ELEMENTS: usize = 10;

/// Render a tree as its skeleton, showing each leaf by its `repr`.
fn render(py: Python<'_>, tree: &ObjectTree) -> PyResult<String> {
    let DataTree::Leaf(value) = tree else {
        let mut rendered = String::from("[");
        for (position, (name, child)) in tree.iter_children().enumerate() {
            if position > 0 {
                rendered.push_str(", ");
            }
            if let Some(name) = name {
                rendered.push_str(name);
                rendered.push_str(": ");
            }
            rendered.push_str(&render(py, child)?);
        }
        rendered.push(']');
        return Ok(rendered);
    };
    render_leaf(value.bind(py))
}

/// Render a leaf by its `repr`, or by its dtype and shape if it is an array of more than
/// [`MAX_RENDERED_ELEMENTS`] elements.
fn render_leaf(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(array) = value.cast::<PyUntypedArray>()
        && array.len() > MAX_RENDERED_ELEMENTS
    {
        let dims: Vec<String> = array.shape().iter().map(usize::to_string).collect();
        return Ok(format!("{}[{}]", array.dtype(), dims.join(", ")));
    }
    Ok(value.repr()?.to_string())
}
