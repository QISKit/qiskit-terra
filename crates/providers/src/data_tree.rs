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

//! The container for structured values: a leaf, or a branch of ordered children each of which may
//! optionally have a name.

use std::fmt;

use hashbrown::HashMap;
use std::borrow::Borrow;
use thiserror::Error;

/// The validated name of one child within a branch of a [`DataTree`].
///
/// The rule a name satisfies is on [`InvalidName`]. A name taken from an existing tree is
/// re-inserted with [`DataTree::insert_name`], which skips the check.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Name(String);

impl Name {
    /// Validate `name` for use as a child's name.
    fn new(name: impl Into<String>) -> Result<Self, InvalidName> {
        let name = name.into();
        if name.is_empty() {
            Err(InvalidName::Empty)
        } else if name.contains('.') {
            Err(InvalidName::ContainsDot(name))
        } else if is_positional(&name) {
            Err(InvalidName::OnlyDigits(name))
        } else {
            Ok(Self(name))
        }
    }

    /// Return the name as a string slice.
    fn as_str(&self) -> &str {
        &self.0
    }
}

impl Borrow<str> for Name {
    fn borrow(&self) -> &str {
        &self.0
    }
}

/// Returned when a string cannot be used as the name of a child in a [`DataTree`].
///
/// A name must be non-empty, contain no `.`, and not consist only of digits. These exclusions make
/// dotted paths unambiguous as read by [`DataTree::get_by_str_key`].
#[derive(Debug, Error)]
pub enum InvalidName {
    #[error("a name cannot be empty")]
    Empty,
    #[error("name \"{0}\" contains \".\"")]
    ContainsDot(String),
    #[error("name \"{0}\" cannot consist only of digits")]
    OnlyDigits(String),
}

/// A path entry used for tracking a path through a [`DataTree`]
///
/// Each entry can either be an index or a key. A slice of `PathEntry` are used to form
/// a traversal path through the [`DataTree`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathEntry<'a> {
    Index(usize),
    Key(&'a str),
}

/// Returned by [`DataTree::unflatten`] when the supplied value count doesn't
/// match the template's leaf count.
#[derive(Debug, Error)]
#[error("unflatten: expected {expected} values, got {actual}")]
pub struct ArityMismatch {
    pub expected: usize,
    pub actual: usize,
}

/// Errors returned by [`DataTree::flatten_against`] when `data`'s structure
/// does not match self's structure.
///
/// The `path` field is rendered as a dotted string (e.g. `"x.0.creg"`),
/// built lazily at the point of error construction.
#[derive(Debug, Error)]
pub enum TreeMatchError {
    /// The path is missing in `data`, or descends through a leaf.
    #[error("missing path {path}")]
    MissingPath { path: String },
    /// A leaf was expected at this path but `data` had a branch.
    #[error("expected a leaf at {path}, found a branch")]
    ExpectedLeaf { path: String },
}

/// A struct representing a branch in a [`DataTree`].
///
/// Each branch contains a vec of [`DataTree`] that can also be assigned a
/// string key for accessing it. Typically you will not create these directly
/// but instead create them via the [`DataTree`] API.
#[derive(Debug, Clone)]
pub struct DataTreeBranch<T> {
    data: Vec<DataTree<T>>,
    keys: HashMap<Name, usize>,
}

impl<T> Default for DataTreeBranch<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DataTreeBranch<T> {
    /// Construct a new empty [`DataTreeBranch`]
    pub fn new() -> Self {
        DataTreeBranch {
            data: Vec::new(),
            keys: HashMap::new(),
        }
    }

    /// Construct a new empty [`DataTreeBranch`] with a set capacity
    pub fn with_capacity(capacity: usize) -> Self {
        DataTreeBranch {
            data: Vec::with_capacity(capacity),
            keys: HashMap::with_capacity(capacity),
        }
    }

    /// Take a path slice and return the entry at the given path
    ///
    /// This will return `None` if a path can not be found. This includes an
    /// invalid path, such as a path a leaf node in the middle.
    fn get_by_path(&self, path: &[PathEntry]) -> Option<&DataTree<T>> {
        let start = match path[0] {
            PathEntry::Index(idx) => self.data.get(idx),
            PathEntry::Key(key) => self.keys.get(key).map(|idx| &self.data[*idx]),
        }?;
        match start {
            DataTree::Leaf(_) => {
                if path.len() > 1 {
                    // If there are more components in the path this is an invalid path
                    None
                } else {
                    Some(start)
                }
            }
            DataTree::Branch(inner_tree) => {
                if path.len() > 1 {
                    inner_tree.get_by_path(&path[1..])
                } else {
                    Some(start)
                }
            }
        }
    }

    /// Return an iterator over the leaves in the `DataTree`
    ///
    /// This method will return an iterator over all leaf nodes in the tree by traversing the tree
    /// in a DFS order.
    fn iter_path(&self) -> IterDataTree<'_, T> {
        IterDataTree {
            tree: None,
            branch: Some(self),
            index: 0,
            inner: None,
            inner_next: None,
            path: vec![],
            names: self.child_names(),
        }
    }

    /// Return the number of items in this `DataTree`. This length is just the number of items in
    /// this local tree object and will not recurse through the tree to compute the total number of
    /// leaves. If you want to do that you should use [`DataTree::iter_leaves`].
    fn iter_leaves(&self) -> IterLeaves<'_, T> {
        IterLeaves {
            tree: None,
            branch: Some(self),
            index: 0,
            inner: None,
            inner_next: None,
        }
    }

    /// Return the number of [`DataTree`] in this branch.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if there are any [`DataTree`] in this branch.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return the number of string keys set on this branch.
    pub fn num_keys(&self) -> usize {
        self.keys.len()
    }

    /// Check if the branch has any string keys set.
    pub fn has_keys(&self) -> bool {
        !self.keys.is_empty()
    }

    /// Return the positional index of each string key.
    fn child_names(&self) -> HashMap<usize, &Name> {
        self.keys.iter().map(|(k, &v)| (v, k)).collect()
    }
}

impl<T> From<DataTree<T>> for DataTreeBranch<T> {
    fn from(input: DataTree<T>) -> Self {
        DataTreeBranch {
            data: vec![input],
            keys: HashMap::new(),
        }
    }
}

/// A generic tree that is addressable either by either indices or string keys
#[derive(Debug, Clone)]
pub enum DataTree<T> {
    Leaf(T),
    Branch(DataTreeBranch<T>),
}

impl<T> Default for DataTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DataTree<T> {
    /// Consume the entry and return the leaf value otherwise panic
    pub fn unwrap_leaf(self) -> T {
        match self {
            Self::Leaf(data) => data,
            Self::Branch(_) => panic!("called TreeEntry::unwrap_leaf() on a `Tree` value"),
        }
    }

    /// Create a new empty data tree
    pub fn new() -> Self {
        DataTree::Branch(DataTreeBranch::new())
    }

    /// Create a new leaf data tree
    pub fn new_leaf(value: T) -> Self {
        DataTree::Leaf(value)
    }

    /// Create a new empty data tree with an underlying allocation of a given size.
    ///
    /// The specified capacity is the number of items of type T stored in the `DataTree`
    /// along with an associated `String` key for each element in the tree. This does not
    /// account for nesting in the allocation as each layer in the tree is a separate
    /// `DataTree` object.
    pub fn with_capacity(capacity: usize) -> Self {
        DataTree::Branch(DataTreeBranch::with_capacity(capacity))
    }

    /// Return the number of items in this `DataTree`. This length is just the number of items in
    /// this local tree object and will not recurse through the tree to compute the total number of
    /// leaves. If you want to do that you should use [`DataTree::iter_leaves`].
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut inner_tree = DataTree::with_capacity(5);
    /// inner_tree.insert_leaf("y", 10)?;
    /// inner_tree.insert_leaf("z", 11)?;
    /// inner_tree.insert_leaf("a", 12)?;
    /// inner_tree.insert_leaf("b", 13)?;
    /// inner_tree.push_leaf(15);
    ///
    /// let mut tree = DataTree::new();
    /// tree.insert_branch("x", inner_tree)?;
    /// assert_eq!(tree.len(), 1);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn len(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Branch(branch) => branch.data.len(),
        }
    }

    /// Return whether this `DataTree` has an items in it.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Take a string key and return the entry at the given key.
    ///
    /// The "." character is reserved in keys and used to indicate a path
    /// through the graph. Since names cannot be all numeric, numbers between
    /// dots are converted into integers and treated as positional indices.
    ///
    /// Returns `None` if the path can not be found. Returns `self`for an empty path.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut inner_tree = DataTree::new();
    /// inner_tree.insert_leaf("y", 10)?;
    /// let mut tree = DataTree::new();
    /// tree.insert_branch("x", inner_tree)?;
    /// let result = tree.get_by_str_key("x.y").unwrap().clone().unwrap_leaf();
    /// assert_eq!(result, 10);
    /// assert_eq!(tree.get_by_str_key("0.0"), tree.get_by_str_key("x.y"));
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn get_by_str_key(&self, path: &str) -> Option<&Self> {
        if path.is_empty() {
            return Some(self);
        }
        let mut entries = Vec::new();
        for segment in path.split(".") {
            entries.push(if is_positional(segment) {
                PathEntry::Index(segment.parse().ok()?)
            } else {
                PathEntry::Key(segment)
            });
        }
        self.get_by_path(&entries)
    }

    /// Take a path slice and return the entry at the given path
    ///
    /// This will return `None` if a path can not be found. This includes an
    /// invalid path, such as a path a leaf node in the middle. An empty path
    /// will also return `self`.
    pub fn get_by_path(&self, path: &[PathEntry]) -> Option<&Self> {
        if path.is_empty() {
            return Some(self);
        }
        let Self::Branch(branch) = self else {
            return None;
        };
        let start = match path[0] {
            PathEntry::Index(idx) => branch.data.get(idx),
            PathEntry::Key(key) => branch.keys.get(key).map(|idx| &branch.data[*idx]),
        }?;
        match start {
            DataTree::Leaf(_) => {
                if path.len() > 1 {
                    // If there are more components in the path this is an invalid path
                    None
                } else {
                    Some(start)
                }
            }
            DataTree::Branch(inner_tree) => {
                if path.len() > 1 {
                    inner_tree.get_by_path(&path[1..])
                } else {
                    Some(start)
                }
            }
        }
    }

    /// Get an item from the `DataTree` by index, panic if `self` is a leaf.
    ///
    /// This will return `None` if the index is not valid.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut inner_tree = DataTree::new();
    /// inner_tree.insert_leaf("y", 10)?;
    /// let mut tree = DataTree::new();
    /// tree.insert_branch("x", inner_tree)?;
    /// tree.push_leaf(124);
    /// let result = tree.get(1).unwrap().clone().unwrap_leaf();
    /// assert_eq!(result, 124);
    /// let subtree = tree.get(0).unwrap();
    /// let subtree_result = subtree.get(0).unwrap().clone().unwrap_leaf();
    /// assert_eq!(subtree_result, 10);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn get(&self, index: usize) -> Option<&DataTree<T>> {
        match self {
            Self::Leaf(_) => panic!("Called get() on a leaf node"),
            Self::Branch(branch) => branch.data.get(index),
        }
    }

    /// Iterate over direct children, yielding `(optional_name, child)` pairs in index order, panic if `self` is a leaf.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut tree = DataTree::new();
    /// tree.push_leaf(10);            // unnamed
    /// tree.insert_leaf("b", 20)?;    // named
    /// tree.push_leaf(30);            // unnamed
    /// let children: Vec<_> = tree.iter_children().collect();
    /// assert_eq!(children[0], (None, &DataTree::Leaf(10)));
    /// assert_eq!(children[1], (Some("b"), &DataTree::Leaf(20)));
    /// assert_eq!(children[2], (None, &DataTree::Leaf(30)));
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn iter_children(&self) -> impl Iterator<Item = (Option<&str>, &DataTree<T>)> + '_ {
        self.iter_named_children()
            .map(|(name, child)| (name.map(Name::as_str), child))
    }

    /// Iterate over direct children, yielding each named child's validated name.
    fn iter_named_children(&self) -> impl Iterator<Item = (Option<&Name>, &DataTree<T>)> + '_ {
        let branch = match self {
            Self::Branch(branch) => branch,
            Self::Leaf(_) => panic!("called iter_children() on a leaf node"),
        };
        let names = branch.child_names();
        branch
            .data
            .iter()
            .enumerate()
            .map(move |(i, child)| (names.get(&i).copied(), child))
    }

    /// Insert a new leaf node with an associated name, panic if `self` is a leaf.
    ///
    /// If the name is already in the tree the new value replaces the old one in place, keeping
    /// its position among the children. Returns [`InvalidName`] if `name` cannot be used as a name.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("y", 10)?;
    /// tree.insert_leaf("y", 1000)?;
    /// assert_eq!(tree.len(), 1);
    /// let result = tree.get_by_str_key("y").unwrap().clone().unwrap_leaf();
    /// assert_eq!(result, 1000);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn insert_leaf(&mut self, name: impl Into<String>, value: T) -> Result<(), InvalidName> {
        self.insert_branch(name, Self::Leaf(value))
    }

    /// Add a new leaf to the tree, panic if `self` is a leaf.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.push_leaf(10);
    /// tree.push_leaf(1000);
    /// assert_eq!(vec![10, 1000], tree.iter_leaves().copied().collect::<Vec<_>>());
    /// ```
    pub fn push_leaf(&mut self, value: T) {
        self.push_child(DataTree::Leaf(value));
    }

    /// Add a subtree to the tree with an associated name, panic if `self` is a leaf.
    ///
    /// If the name is already in the tree the new subtree replaces the old child in place, keeping
    /// its position among the children. Returns [`InvalidName`] if `name` cannot be used as a name.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("y", 10)?;
    /// let mut subtree = DataTree::with_capacity(2);
    /// subtree.push_leaf(123);
    /// subtree.push_leaf(456);
    /// tree.insert_branch("y", subtree)?;
    /// let result = tree.get_by_str_key("y").unwrap();
    /// let leaves: Vec<_> = result.iter_leaves().copied().collect();
    /// assert_eq!(leaves, vec![123, 456]);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn insert_branch(
        &mut self,
        name: impl Into<String>,
        value: DataTree<T>,
    ) -> Result<(), InvalidName> {
        self.insert_name(Name::new(name)?, value);
        Ok(())
    }

    /// Add a subtree under a name that is already validated, panic if `self` is a leaf.
    fn insert_name(&mut self, name: Name, value: DataTree<T>) {
        match self {
            Self::Leaf(_) => panic!("Called insert_branch() on a leaf node"),
            Self::Branch(branch) => match branch.keys.get(name.as_str()) {
                Some(&index) => branch.data[index] = value,
                None => {
                    branch.data.push(value);
                    branch.keys.insert(name, branch.data.len() - 1);
                }
            },
        }
    }

    /// Append an unnamed child, panic if `self` is a leaf.
    fn push_child(&mut self, child: DataTree<T>) {
        match self {
            Self::Leaf(_) => panic!("Called push_child() on a leaf node"),
            Self::Branch(branch) => branch.data.push(child),
        }
    }

    /// Add a new branch to the tree
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let mut tree = DataTree::new();
    /// tree.push_leaf(10);
    /// let mut subtree = DataTree::with_capacity(2);
    /// subtree.push_leaf(123);
    /// subtree.push_leaf(456);
    /// tree.push_branch(subtree);
    /// assert_eq!(vec![10, 123, 456], tree.iter_leaves().copied().collect::<Vec<_>>());
    /// ```
    pub fn push_branch(&mut self, value: DataTree<T>) {
        self.push_child(value);
    }

    /// Build a branch whose children are all unnamed.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::DataTree;
    /// let tree = DataTree::sequence([DataTree::Leaf(10), DataTree::Leaf(20)]);
    /// assert_eq!(tree.get(1), Some(&DataTree::Leaf(20)));
    /// ```
    pub fn sequence(children: impl IntoIterator<Item = DataTree<T>>) -> Self {
        let mut tree = Self::new();
        for child in children {
            tree.push_child(child);
        }
        tree
    }

    /// Build a branch whose children are all named.
    ///
    /// A name given twice addresses one child, which the later value replaces in place.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let tree = DataTree::mapping([
    ///     ("x", DataTree::Leaf(10)),
    ///     ("y", DataTree::sequence([DataTree::Leaf(20)])),
    /// ])?;
    /// assert_eq!(tree.get_by_str_key("y.0"), Some(&DataTree::Leaf(20)));
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn mapping<N: Into<String>>(
        children: impl IntoIterator<Item = (N, DataTree<T>)>,
    ) -> Result<Self, InvalidName> {
        let mut tree = Self::new();
        for (name, child) in children {
            tree.insert_branch(name, child)?;
        }
        Ok(tree)
    }

    /// Return an iterator over the leaves in the `DataTree`
    ///
    /// This method will return an iterator over all leaf nodes in the tree by traversing the tree
    /// in a DFS order. A branch with no leaves beneath it yields nothing, so it is passed over.
    ///
    /// # Example
    ///
    /// Traversing this tree:
    ///
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut subsubsubtree = DataTree::new();
    /// subsubsubtree.push_leaf(3);
    /// subsubsubtree.push_leaf(4);
    /// let mut subsubtree = DataTree::new();
    /// subsubtree.push_branch(subsubsubtree);
    /// subsubtree.insert_leaf("b", 5)?;
    /// let mut subsubtree_prime = DataTree::new();
    /// subsubtree_prime.push_leaf(7);
    /// let mut subtree = DataTree::new();
    /// subtree.insert_branch("c", subsubtree)?;
    /// subtree.insert_leaf("d", 6)?;
    /// subtree.push_branch(subsubtree_prime);
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("a", 0)?;
    /// tree.insert_branch("root", subtree)?;
    /// tree.insert_leaf("z", 26)?;
    /// let leaves: Vec<_> = tree.iter_leaves().copied().collect();
    /// let expected = vec![0, 3, 4, 5, 6, 7, 26];
    /// assert_eq!(leaves, expected);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn iter_leaves(&self) -> impl Iterator<Item = &T> {
        IterLeaves {
            tree: Some(self),
            branch: None,
            index: 0,
            inner: None,
            inner_next: None,
        }
    }

    /// Return an iterator over the leaves in the `DataTree` that returns the path and leaf value.
    ///
    /// This method will return an iterator over all the leaf nodes in the tree in a DFS order.
    /// Unlike [`iter_leaves`](Self::iter_leaves) which just returns the value this will return an
    /// owned `Vec` of the path through the data tree to get to that value. This has allocation
    /// overhead for each leaf node in the tree and should only be used if you need the path along
    /// with the value.
    ///
    /// A named child contributes [`PathEntry::Key`] to the path and an unnamed one contributes
    /// [`PathEntry::Index`]. A branch with no leaves beneath it yields nothing, so it has no path.
    ///
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName, PathEntry};
    /// let mut subsubsubtree = DataTree::new();
    /// subsubsubtree.push_leaf(3);
    /// subsubsubtree.push_leaf(4);
    /// let mut subsubtree = DataTree::new();
    /// subsubtree.push_branch(subsubsubtree);
    /// subsubtree.insert_leaf("b", 5)?;
    /// let mut subsubtree_prime = DataTree::new();
    /// subsubtree_prime.push_leaf(7);
    /// let mut subtree = DataTree::new();
    /// subtree.insert_branch("c", subsubtree)?;
    /// subtree.insert_leaf("d", 6)?;
    /// subtree.push_branch(subsubtree_prime);
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("a", 0)?;
    /// tree.insert_branch("root", subtree)?;
    /// tree.insert_leaf("z", 26)?;
    /// let result: Vec<_> = tree.iter_path().map(|(a, b)| (a, *b)).collect();
    /// let expected: Vec<(Vec<PathEntry>, i32)> = vec![
    ///     (vec![PathEntry::Key("a")], 0),
    ///     (vec![PathEntry::Key("root"), PathEntry::Key("c"), PathEntry::Index(0), PathEntry::Index(0)], 3),
    ///     (vec![PathEntry::Key("root"), PathEntry::Key("c"), PathEntry::Index(0), PathEntry::Index(1)], 4),
    ///     (vec![PathEntry::Key("root"), PathEntry::Key("c"), PathEntry::Key("b")], 5),
    ///     (vec![PathEntry::Key("root"), PathEntry::Key("d")], 6),
    ///     (vec![PathEntry::Key("root"), PathEntry::Index(2), PathEntry::Index(0)], 7),
    ///     (vec![PathEntry::Key("z")], 26),
    /// ];
    /// assert_eq!(result, expected);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn iter_path(&self) -> IterDataTree<'_, T> {
        IterDataTree {
            tree: Some(self),
            branch: None,
            index: 0,
            inner: None,
            inner_next: None,
            path: Vec::new(),
            names: match self {
                Self::Leaf(_) => HashMap::new(),
                Self::Branch(branch) => branch.child_names(),
            },
        }
    }

    /// Return the number of leaves in this tree.
    pub fn leaf_count(&self) -> usize {
        self.iter_leaves().count()
    }

    /// Return this tree with its leaf values erased.
    pub fn structure(&self) -> DataTree<()> {
        self.map_leaves(|_| ())
    }

    /// Return a dotted path addressing each leaf, in the order of
    /// [`iter_leaves`](Self::iter_leaves).
    ///
    /// A named child contributes its name and an unnamed one contributes its position. A tree that
    /// is itself a leaf returns an empty path.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let tree = DataTree::mapping([
    ///     ("counts", DataTree::sequence([DataTree::Leaf(1), DataTree::Leaf(2)])),
    ///     ("shots", DataTree::Leaf(3)),
    /// ])?;
    /// assert_eq!(tree.dotted_paths(), ["counts.0", "counts.1", "shots"]);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn dotted_paths(&self) -> Vec<String> {
        self.iter_path()
            .map(|(path, _)| dotted_path(&path))
            .collect()
    }

    /// Build a tree with the same structure as `self`, replacing each leaf value
    /// by `f(&leaf)`.
    ///
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let mut tree = DataTree::new();
    /// tree.insert_leaf("a", 1)?;
    /// tree.insert_leaf("b", 2)?;
    /// let doubled = tree.map_leaves(|v| v * 2);
    /// assert_eq!(doubled.iter_leaves().copied().collect::<Vec<_>>(), vec![2, 4]);
    /// # Ok::<(), InvalidName>(())
    /// ```
    pub fn map_leaves<U>(&self, mut f: impl FnMut(&T) -> U) -> DataTree<U> {
        fn inner<T, U>(tree: &DataTree<T>, f: &mut impl FnMut(&T) -> U) -> DataTree<U> {
            match tree {
                DataTree::Leaf(value) => DataTree::new_leaf(f(value)),
                DataTree::Branch(_) => {
                    let mut result = DataTree::with_capacity(tree.len());
                    for (name, child) in tree.iter_named_children() {
                        let new_child = inner(child, f);
                        match name {
                            Some(name) => result.insert_name(name.clone(), new_child),
                            None => result.push_child(new_child),
                        }
                    }
                    result
                }
            }
        }
        inner(self, &mut f)
    }

    /// Consume this tree, yielding owned leaf values in DFS order. Mirrors
    /// [`iter_leaves`](Self::iter_leaves) for the consuming case.
    pub fn into_leaves(self) -> impl Iterator<Item = T> {
        IntoLeaves {
            stack: vec![vec![self].into_iter()],
        }
    }

    /// Build a tree with the same structure as `self`, taking leaf values from
    /// `values` in DFS order.
    ///
    /// Returns [`ArityMismatch`] if `values.len()` does not equal the leaf
    /// count of `self`.
    pub fn unflatten<U>(&self, values: Vec<U>) -> Result<DataTree<U>, ArityMismatch> {
        fn inner<T, U>(
            template: &DataTree<T>,
            iter: &mut std::vec::IntoIter<U>,
        ) -> Result<DataTree<U>, ()> {
            match template {
                DataTree::Leaf(_) => Ok(DataTree::new_leaf(iter.next().ok_or(())?)),
                DataTree::Branch(_) => {
                    let mut result = DataTree::with_capacity(template.len());
                    for (name, child) in template.iter_named_children() {
                        let subtree = inner(child, iter)?;
                        match name {
                            Some(name) => result.insert_name(name.clone(), subtree),
                            None => result.push_child(subtree),
                        }
                    }
                    Ok(result)
                }
            }
        }

        let actual = values.len();
        let mut iter = values.into_iter();
        match inner(self, &mut iter) {
            Ok(result) => {
                let left_over = iter.len();
                if left_over == 0 {
                    Ok(result)
                } else {
                    Err(ArityMismatch {
                        expected: actual - left_over,
                        actual,
                    })
                }
            }
            Err(()) => {
                // The Iterator was exhausted mid-walk, rewalk the tree to find
                // out how many we should have expected.
                Err(ArityMismatch {
                    expected: self.iter_leaves().count(),
                    actual,
                })
            }
        }
    }

    /// Walk `data` lockstep with `self`'s structure, returning `data`'s leaves
    /// in DFS order. Errors structurally if `data`'s structure doesn't match.
    pub fn flatten_against<U: Clone>(&self, data: &DataTree<U>) -> Result<Vec<U>, TreeMatchError> {
        fn inner<'a, T, U: Clone>(
            template: &'a DataTree<T>,
            data: &'a DataTree<U>,
            path: &mut Vec<PathEntry<'a>>,
            out: &mut Vec<U>,
        ) -> Result<(), TreeMatchError> {
            match (template, data) {
                (DataTree::Leaf(_), DataTree::Leaf(value)) => {
                    out.push(value.clone());
                    Ok(())
                }
                (DataTree::Leaf(_), DataTree::Branch(_)) => Err(TreeMatchError::ExpectedLeaf {
                    path: dotted_path(path),
                }),
                (DataTree::Branch(_), _) => {
                    for (i, (name, child_template)) in template.iter_children().enumerate() {
                        let entry = match name {
                            Some(name) => PathEntry::Key(name),
                            None => PathEntry::Index(i),
                        };
                        let data_child = match entry {
                            PathEntry::Key(k) => data.get_by_str_key(k),
                            PathEntry::Index(idx) => data.get(idx),
                        };
                        path.push(entry);
                        let data_child = data_child.ok_or_else(|| TreeMatchError::MissingPath {
                            path: dotted_path(path),
                        })?;
                        inner(child_template, data_child, path, out)?;
                        path.pop();
                    }
                    Ok(())
                }
            }
        }

        let mut out = Vec::new();
        inner(self, data, &mut Vec::new(), &mut out)?;
        Ok(out)
    }
}

pub struct IterDataTree<'a, T> {
    tree: Option<&'a DataTree<T>>,
    branch: Option<&'a DataTreeBranch<T>>,
    index: usize,
    inner: Option<Box<IterDataTree<'a, T>>>,
    inner_next: Option<(Vec<PathEntry<'a>>, &'a T)>,
    path: Vec<PathEntry<'a>>,
    /// The name of each named child of the branch being walked, by position.
    names: HashMap<usize, &'a Name>,
}

impl<'a, T> IterDataTree<'a, T> {
    /// Return the path entry addressing the child at `index`: its name if it has one, else its
    /// position.
    fn entry(&self, index: usize) -> PathEntry<'a> {
        match self.names.get(&index) {
            Some(name) => PathEntry::Key(name.as_str()),
            None => PathEntry::Index(index),
        }
    }
}

impl<'a, T> Iterator for IterDataTree<'a, T> {
    type Item = (Vec<PathEntry<'a>>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(tree) = self.tree {
            if let DataTree::Leaf(val) = tree {
                if self.index == 0 {
                    self.index += 1;
                    return Some((vec![], val));
                } else {
                    return None;
                }
            }
            let DataTree::Branch(branch) = tree else {
                unreachable!("Must be a branch variant");
            };
            if self.index >= branch.data.len() {
                return None;
            }
            let entry = &branch.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    let mut out_path = self.path.clone();
                    out_path.push(self.entry(self.index - 1));
                    Some((out_path, val))
                }
                DataTree::Branch(sub_branch) => {
                    if let Some(ref mut inner) = self.inner {
                        if let Some(val) = inner.next() {
                            let (return_path, return_val) = self.inner_next.replace(val).unwrap();
                            Some((return_path, return_val))
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let (return_path, return_val) = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some((return_path, return_val))
                        }
                    } else {
                        let mut inner = sub_branch.iter_path();
                        let mut inner_path = self.path.clone();
                        inner_path.push(self.entry(self.index));
                        inner.path = inner_path;
                        let Some((leaf_path, val)) = inner.next() else {
                            // A branch with no leaves under it contributes none here either.
                            self.index += 1;
                            return self.next();
                        };
                        self.inner_next = inner.next();
                        self.inner = Some(Box::new(inner));
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                        }
                        Some((leaf_path, val))
                    }
                }
            }
        } else if let Some(subtree) = self.branch {
            if self.index >= subtree.data.len() {
                return None;
            }
            let entry = &subtree.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    let mut out_path = self.path.clone();
                    out_path.push(self.entry(self.index - 1));
                    Some((out_path, val))
                }
                DataTree::Branch(subtree) => match self.inner {
                    Some(ref mut inner) => {
                        if let Some(val) = inner.next() {
                            let (return_path, return_val) = self.inner_next.replace(val).unwrap();
                            Some((return_path, return_val))
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let (return_path, return_val) = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some((return_path, return_val))
                        }
                    }
                    None => {
                        let mut inner = subtree.iter_path();
                        let mut inner_path = self.path.clone();
                        inner_path.push(self.entry(self.index));
                        inner.path = inner_path;
                        let Some((leaf_path, val)) = inner.next() else {
                            // A branch with no leaves under it contributes none here either.
                            self.index += 1;
                            return self.next();
                        };
                        self.inner_next = inner.next();
                        self.inner = Some(Box::new(inner));
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                        }
                        Some((leaf_path, val))
                    }
                },
            }
        } else {
            None
        }
    }
}

struct IntoLeaves<T> {
    /// DFS stack of child iterators, one frame per nesting level.
    /// Each frame is the remaining children of a branch node.
    stack: Vec<std::vec::IntoIter<DataTree<T>>>,
}

impl<T> Iterator for IntoLeaves<T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        loop {
            let top = self.stack.last_mut()?;
            match top.next() {
                None => {
                    self.stack.pop();
                }
                Some(DataTree::Leaf(v)) => return Some(v),
                Some(DataTree::Branch(b)) => self.stack.push(b.data.into_iter()),
            }
        }
    }
}

struct IterLeaves<'a, T> {
    tree: Option<&'a DataTree<T>>,
    branch: Option<&'a DataTreeBranch<T>>,
    index: usize,
    inner: Option<Box<IterLeaves<'a, T>>>,
    inner_next: Option<&'a T>,
}

impl<'a, T> Iterator for IterLeaves<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(tree) = self.tree {
            if let DataTree::Leaf(val) = tree {
                if self.index == 0 {
                    self.index += 1;
                    return Some(val);
                } else {
                    return None;
                }
            }
            let DataTree::Branch(branch) = tree else {
                unreachable!("Must be a branch variant");
            };
            if self.index >= branch.data.len() {
                return None;
            }
            let entry = &branch.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    Some(val)
                }
                DataTree::Branch(sub_branch) => {
                    if let Some(ref mut inner) = self.inner {
                        if let Some(val) = inner.next() {
                            let return_val = self.inner_next.replace(val).unwrap();
                            Some(return_val)
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let return_val = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some(return_val)
                        }
                    } else {
                        let mut inner = sub_branch.iter_leaves();
                        let Some(val) = inner.next() else {
                            // A branch with no leaves under it contributes none here either.
                            self.index += 1;
                            return self.next();
                        };
                        self.inner_next = inner.next();
                        self.inner = Some(Box::new(inner));
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                        }
                        Some(val)
                    }
                }
            }
        } else if let Some(subtree) = self.branch {
            if self.index >= subtree.data.len() {
                return None;
            }
            let entry = &subtree.data[self.index];
            match entry {
                DataTree::Leaf(val) => {
                    self.index += 1;
                    Some(val)
                }
                DataTree::Branch(subtree) => match self.inner {
                    Some(ref mut inner) => {
                        if let Some(val) = inner.next() {
                            let return_val = self.inner_next.replace(val).unwrap();
                            Some(return_val)
                        } else {
                            self.inner = None;
                            self.index += 1;
                            let return_val = self.inner_next.take().unwrap();
                            self.inner_next = None;
                            Some(return_val)
                        }
                    }
                    None => {
                        let mut inner = subtree.iter_leaves();
                        let Some(val) = inner.next() else {
                            // A branch with no leaves under it contributes none here either.
                            self.index += 1;
                            return self.next();
                        };
                        self.inner_next = inner.next();
                        self.inner = Some(Box::new(inner));
                        if self.inner_next.is_none() {
                            self.index += 1;
                            self.inner = None;
                        }
                        Some(val)
                    }
                },
            }
        } else {
            None
        }
    }
}

impl<T: PartialEq> PartialEq for DataTree<T> {
    fn eq(&self, other: &DataTree<T>) -> bool {
        match self {
            Self::Leaf(val) => {
                let Self::Leaf(other_val) = other else {
                    return false;
                };
                val == other_val
            }
            Self::Branch(branch) => {
                let Self::Branch(other) = other else {
                    return false;
                };
                branch.data == other.data && branch.keys == other.keys
            }
        }
    }
}

impl fmt::Display for DataTree<()> {
    /// Render a [structure](DataTree::structure) as a skeleton: a leaf as `_`, a branch as its
    /// children in brackets, each prefixed by its name where it has one.
    ///
    /// # Example
    /// ```rust
    /// use qiskit_providers::{DataTree, InvalidName};
    /// let tree = DataTree::mapping([
    ///     ("counts", DataTree::sequence([DataTree::Leaf(1), DataTree::Leaf(2)])),
    ///     ("shots", DataTree::Leaf(3)),
    /// ])?;
    /// assert_eq!(tree.structure().to_string(), "[counts: [_, _], shots: _]");
    /// # Ok::<(), InvalidName>(())
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Branch(_) = self else {
            return f.write_str("_");
        };
        // A branch is bracketed whether or not its children are named. It may mix the two.
        f.write_str("[")?;
        for (position, (name, child)) in self.iter_children().enumerate() {
            if position > 0 {
                f.write_str(", ")?;
            }
            if let Some(name) = name {
                write!(f, "{}: ", name)?;
            }
            write!(f, "{child}")?;
        }
        f.write_str("]")
    }
}

/// Return whether a path segment addresses a child by position rather than by name.
fn is_positional(segment: &str) -> bool {
    !segment.is_empty() && segment.bytes().all(|byte| byte.is_ascii_digit())
}

/// Render a `[PathEntry]` slice as a dotted path string. Empty path renders
/// as `""`. Used to format error messages and [`DataTree::dotted_paths`].
fn dotted_path(path: &[PathEntry<'_>]) -> String {
    path.iter()
        .map(|e| match e {
            PathEntry::Index(i) => i.to_string(),
            PathEntry::Key(k) => k.to_string(),
        })
        .collect::<Vec<_>>()
        .join(".")
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_data_leaf() {
        let mut tree = DataTree::new();
        tree.push_leaf(42);
        let result = tree.get(0).unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 42);
    }

    #[test]
    fn test_flat_dict() {
        let mut tree = DataTree::with_capacity(3);
        tree.insert_leaf("a", 1).unwrap();
        tree.insert_leaf("b", 2).unwrap();
        let result = tree.get_by_str_key("b").unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 2);
        let result = tree.get_by_str_key("a").unwrap().clone();
        assert_eq!(result.unwrap_leaf(), 1);
    }

    #[test]
    fn test_nested_dict() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10).unwrap();
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone()).unwrap();
        tree.insert_leaf("z", 100).unwrap();
        assert_eq!(None, tree.get_by_str_key("z.y"));
        assert_eq!(Some(&inner_tree), tree.get_by_str_key("x"));
    }

    #[test]
    fn test_nested_dict_iter() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10).unwrap();
        inner_tree.insert_leaf("yy", 1).unwrap();
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.push_leaf(4);
        inner_inner_tree.push_leaf(5);
        inner_tree.push_branch(inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone()).unwrap();
        tree.insert_leaf("z", 100).unwrap();
        assert_eq!(
            vec![10, 1, 2, 3, 4, 5, 100],
            tree.iter_leaves().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_nested_dict_iter_path() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10).unwrap();
        inner_tree.insert_leaf("yy", 1).unwrap();
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.push_leaf(4);
        inner_inner_tree.push_leaf(5);
        inner_tree.push_branch(inner_inner_tree);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone()).unwrap();
        tree.insert_leaf("z", 100).unwrap();
        let expected_paths = vec![
            vec![PathEntry::Key("x"), PathEntry::Key("y")],
            vec![PathEntry::Key("x"), PathEntry::Key("yy")],
            vec![
                PathEntry::Key("x"),
                PathEntry::Index(2),
                PathEntry::Index(0),
            ],
            vec![
                PathEntry::Key("x"),
                PathEntry::Index(2),
                PathEntry::Index(1),
            ],
            vec![
                PathEntry::Key("x"),
                PathEntry::Index(2),
                PathEntry::Index(2),
            ],
            vec![
                PathEntry::Key("x"),
                PathEntry::Index(2),
                PathEntry::Index(3),
            ],
            vec![PathEntry::Key("z")],
        ];
        let expected_vals = vec![10, 1, 2, 3, 4, 5, 100];
        let expected = expected_paths
            .into_iter()
            .zip(expected_vals)
            .collect::<Vec<_>>();
        assert_eq!(
            expected,
            tree.iter_path().map(|(a, b)| (a, *b)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_get_by_str() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10).unwrap();
        inner_tree.insert_leaf("yy", 1).unwrap();
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.insert_leaf("a", 4).unwrap();
        inner_inner_tree.push_leaf(5);
        let inner_inner_tree_expected = inner_inner_tree.clone();
        inner_tree.insert_branch("yyy", inner_inner_tree).unwrap();
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone()).unwrap();
        tree.insert_leaf("z", 100).unwrap();
        let result = tree.get_by_str_key("x.yyy.a");
        assert_eq!(result, Some(&DataTree::Leaf(4)));
        assert_eq!(tree.get_by_str_key("z"), Some(&DataTree::Leaf(100)));
        assert_eq!(
            tree.get_by_str_key("x.yyy"),
            Some(&inner_inner_tree_expected),
        );
        assert_eq!(tree.get_by_str_key("x.yy"), Some(&DataTree::Leaf(1)));
    }

    #[test]
    fn test_get_by_str_no_match() {
        let mut inner_tree = DataTree::new();
        inner_tree.insert_leaf("y", 10).unwrap();
        inner_tree.insert_leaf("yy", 1).unwrap();
        let mut inner_inner_tree = DataTree::new();
        inner_inner_tree.push_leaf(2);
        inner_inner_tree.push_leaf(3);
        inner_inner_tree.insert_leaf("a", 4).unwrap();
        inner_inner_tree.push_leaf(5);
        inner_tree.insert_branch("yyy", inner_inner_tree).unwrap();
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner_tree.clone()).unwrap();
        tree.insert_leaf("z", 100).unwrap();
        assert_eq!(None, tree.get_by_str_key("a"));
        assert_eq!(None, tree.get_by_str_key("x.yyyy"));
        assert_eq!(None, tree.get_by_str_key("x.yy.a"));
        assert_eq!(None, tree.get_by_str_key("🎩"));
        assert_eq!(None, tree.get_by_str_key("z.yyy.a"));
    }

    #[test]
    fn test_map_leaves() {
        let mut sub = DataTree::new();
        sub.insert_leaf("a", 2).unwrap();
        sub.push_leaf(3);
        let mut tree = DataTree::new();
        tree.insert_branch("x", sub).unwrap();
        tree.insert_leaf("y", 5).unwrap();

        let doubled = tree.map_leaves(|v| v * 2);
        assert_eq!(
            doubled.iter_leaves().copied().collect::<Vec<_>>(),
            vec![4, 6, 10]
        );
        // Structure is preserved: keyed children are still keyed.
        assert!(doubled.get_by_str_key("x").is_some());
        assert!(doubled.get_by_str_key("y").is_some());
    }

    #[test]
    fn test_into_leaves() {
        let mut sub = DataTree::new();
        sub.insert_leaf("a", 1).unwrap();
        sub.push_leaf(2);
        let mut tree = DataTree::new();
        tree.insert_branch("x", sub).unwrap();
        tree.insert_leaf("y", 3).unwrap();
        assert_eq!(tree.into_leaves().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn test_unflatten_preserves_named_vs_anonymous() {
        let mut sub = DataTree::new();
        sub.insert_leaf("a", 0).unwrap();
        sub.push_leaf(0);
        let mut template = DataTree::new();
        template.insert_branch("x", sub).unwrap();
        template.insert_leaf("y", 0).unwrap();

        let result = template.unflatten(vec![1, 2, 3]).unwrap();
        assert_eq!(result.get_by_str_key("x.a"), Some(&DataTree::Leaf(1)));
        assert!(result.get_by_str_key("x").unwrap().get(1).is_some()); // anonymous
        assert_eq!(result.get_by_str_key("y"), Some(&DataTree::Leaf(3)));
    }

    #[test]
    fn test_unflatten_arity_mismatch_errors() {
        let mut template = DataTree::new();
        template.insert_leaf("x", 0).unwrap();
        template.insert_leaf("y", 0).unwrap();
        // 2 leaves; passing 1 value
        let err = template.unflatten(vec![42]).unwrap_err();
        assert!(matches!(
            err,
            ArityMismatch {
                expected: 2,
                actual: 1
            }
        ));
        // 2 leaves; passing 3 values
        let err = template.unflatten(vec![1, 2, 3]).unwrap_err();
        assert!(matches!(
            err,
            ArityMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[test]
    fn test_flatten_against_branch_with_keys() {
        let mut template = DataTree::new();
        template.insert_leaf("x", 0).unwrap();
        template.insert_leaf("y", 0).unwrap();
        let mut data = DataTree::new();
        data.insert_leaf("x", 1).unwrap();
        data.insert_leaf("y", 2).unwrap();
        assert_eq!(template.flatten_against(&data).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_flatten_against_missing_key_errors() {
        let mut template = DataTree::new();
        template.insert_leaf("x", 0).unwrap();
        template.insert_leaf("y", 0).unwrap();
        let mut data = DataTree::new();
        data.insert_leaf("x", 1).unwrap();
        let err = template.flatten_against(&data).unwrap_err();
        assert!(matches!(
            err,
            TreeMatchError::MissingPath { path } if path == "y"
        ));
    }

    #[test]
    fn test_flatten_against_branch_where_leaf_expected_errors() {
        let mut template = DataTree::new();
        template.insert_leaf("x", 0).unwrap();
        template.insert_leaf("y", 0).unwrap();
        let mut data = DataTree::new();
        data.insert_leaf("x", 1).unwrap();
        data.insert_branch("y", DataTree::<i32>::new()).unwrap();
        let err = template.flatten_against(&data).unwrap_err();
        assert!(matches!(
            err,
            TreeMatchError::ExpectedLeaf { path } if path == "y"
        ));
    }

    #[test]
    fn test_flatten_against_then_unflatten_roundtrip() {
        let mut sub = DataTree::new();
        sub.insert_leaf("a", 0).unwrap();
        sub.push_leaf(0);
        let mut template = DataTree::new();
        template.insert_leaf("x", 0).unwrap();
        template.insert_branch("y", sub).unwrap();

        let mut data_sub = DataTree::new();
        data_sub.insert_leaf("a", 20).unwrap();
        data_sub.push_leaf(30);
        let mut data = DataTree::new();
        data.insert_leaf("x", 10).unwrap();
        data.insert_branch("y", data_sub).unwrap();

        let flat = template.flatten_against(&data).unwrap();
        assert_eq!(flat, vec![10, 20, 30]);
        let back = template.unflatten(flat).unwrap();
        assert_eq!(back, data);
    }

    /// Return `{"x": {"y": 10, "yy": 1, [2, 3, 4, 5]}, "z": 100}`.
    fn mixed_tree() -> DataTree<i32> {
        let mut sub = DataTree::new();
        sub.push_leaf(2);
        sub.push_leaf(3);
        sub.push_leaf(4);
        sub.push_leaf(5);
        let mut inner = DataTree::new();
        inner.insert_leaf("y", 10).unwrap();
        inner.insert_leaf("yy", 1).unwrap();
        inner.push_branch(sub);
        let mut tree = DataTree::new();
        tree.insert_branch("x", inner).unwrap();
        tree.insert_leaf("z", 100).unwrap();
        tree
    }

    #[test]
    fn test_name_rejects_dots_digits_and_emptiness() {
        assert!(matches!(Name::new(""), Err(InvalidName::Empty)));
        assert!(matches!(
            Name::new("a.b"),
            Err(InvalidName::ContainsDot(name)) if name == "a.b"
        ));
        assert!(matches!(
            Name::new("0"),
            Err(InvalidName::OnlyDigits(name)) if name == "0"
        ));
        assert!(matches!(
            Name::new("007"),
            Err(InvalidName::OnlyDigits(name)) if name == "007"
        ));
    }

    #[test]
    fn test_name_accepts_anything_else() {
        for candidate in ["counts", "creg0", "0a", "-1", "1_5", "🎩"] {
            assert!(Name::new(candidate).is_ok(), "rejected {candidate:?}");
        }
    }

    #[test]
    fn test_branch_mixes_named_and_unnamed_children() {
        let mut tree = DataTree::new();
        tree.push_leaf(1);
        tree.insert_leaf("b", 2).unwrap();
        tree.push_leaf(3);
        let names: Vec<_> = tree.iter_children().map(|(key, _)| key).collect();
        assert_eq!(names, [None, Some("b"), None]);
    }

    #[test]
    fn test_insert_replaces_a_named_child_in_place() {
        let mut tree = DataTree::new();
        tree.insert_leaf("a", 1).unwrap();
        tree.push_leaf(2);
        tree.insert_leaf("a", 3).unwrap();
        // The replaced child is not left behind as an unnamed sibling.
        assert_eq!(tree.len(), 2);
        let children: Vec<_> = tree.iter_children().collect();
        assert_eq!(
            children,
            [(Some("a"), &DataTree::Leaf(3)), (None, &DataTree::Leaf(2))]
        );
    }

    #[test]
    fn test_insert_branch_replaces_a_named_child_in_place() {
        let mut tree = DataTree::new();
        tree.insert_leaf("a", 1).unwrap();
        tree.insert_leaf("b", 2).unwrap();
        tree.insert_branch("a", DataTree::sequence([DataTree::Leaf(3)]))
            .unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.get_by_str_key("a.0"), Some(&DataTree::Leaf(3)));
        assert_eq!(tree.get_by_str_key("b"), Some(&DataTree::Leaf(2)));
    }

    #[test]
    fn test_insert_rejects_an_invalid_name() {
        let mut tree = DataTree::new();
        assert!(matches!(
            tree.insert_leaf("a.b", 1),
            Err(InvalidName::ContainsDot(name)) if name == "a.b"
        ));
        assert!(matches!(
            tree.insert_branch("0", DataTree::Leaf(1)),
            Err(InvalidName::OnlyDigits(name)) if name == "0"
        ));
        // A rejected name leaves the tree alone.
        assert!(tree.is_empty());
    }

    #[test]
    fn test_sequence_and_mapping_construct_immutably() {
        let tree = DataTree::mapping([
            (
                "x",
                DataTree::sequence([DataTree::Leaf(1), DataTree::Leaf(2)]),
            ),
            ("y", DataTree::Leaf(3)),
        ])
        .unwrap();
        assert_eq!(
            tree.iter_leaves().copied().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(tree.get_by_str_key("x.1"), Some(&DataTree::Leaf(2)));
        assert_eq!(tree.get_by_str_key("y"), Some(&DataTree::Leaf(3)));
    }

    #[test]
    fn test_mapping_repeats_a_name_by_replacing_it() {
        let tree = DataTree::mapping([
            ("a", DataTree::Leaf(1)),
            ("b", DataTree::Leaf(2)),
            ("a", DataTree::Leaf(3)),
        ])
        .unwrap();
        assert_eq!(
            tree,
            DataTree::mapping([("a", DataTree::Leaf(3)), ("b", DataTree::Leaf(2))]).unwrap()
        );
    }

    #[test]
    fn test_mapping_rejects_an_invalid_name() {
        let result = DataTree::mapping([("a.b", DataTree::Leaf(1))]);
        assert!(matches!(
            result,
            Err(InvalidName::ContainsDot(name)) if name == "a.b"
        ));
    }

    #[test]
    fn test_structure_erases_values_and_keeps_naming() {
        let tree = mixed_tree();
        assert_eq!(
            tree.structure(),
            tree.map_leaves(|value| value * 2).structure()
        );
        assert_eq!(
            tree.structure(),
            tree.map_leaves(|value| *value as f64).structure()
        );
    }

    #[test]
    fn test_structure_renders_names_positions_and_nesting() {
        assert_eq!(
            mixed_tree().structure().to_string(),
            "[x: [y: _, yy: _, [_, _, _, _]], z: _]"
        );
        assert_eq!(DataTree::Leaf(1).structure().to_string(), "_");
        assert_eq!(DataTree::<i32>::new().structure().to_string(), "[]");
    }

    #[test]
    fn test_structures_differ_when_trees_are_put_together_differently() {
        let sequence = DataTree::sequence([DataTree::Leaf(1)]);
        let mapping = DataTree::mapping([("a", DataTree::Leaf(1))]).unwrap();
        assert_ne!(sequence.structure(), mapping.structure());
        assert_ne!(DataTree::Leaf(1).structure(), sequence.structure());
    }

    #[test]
    fn test_leaf_count_counts_leaves_not_children() {
        assert_eq!(mixed_tree().leaf_count(), 7);
        assert_eq!(mixed_tree().len(), 2);
        assert_eq!(DataTree::Leaf(1).leaf_count(), 1);
        assert_eq!(DataTree::<i32>::new().leaf_count(), 0);
    }

    #[test]
    fn test_a_branch_with_no_leaves_under_it_contributes_none() {
        let empty = DataTree::new;
        let tree = DataTree::mapping([
            ("first", empty()),
            ("a", DataTree::Leaf(1)),
            ("middle", DataTree::sequence([empty(), empty()])),
            ("b", DataTree::Leaf(2)),
            ("last", empty()),
        ])
        .unwrap();

        assert_eq!(tree.iter_leaves().copied().collect::<Vec<_>>(), [1, 2]);
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.dotted_paths(), ["a", "b"]);
        assert_eq!(
            tree.unflatten(vec![10, 20]).unwrap(),
            tree.map_leaves(|leaf| leaf * 10)
        );
    }

    #[test]
    fn test_dotted_paths_name_named_slots_and_number_the_rest() {
        assert_eq!(
            mixed_tree().dotted_paths(),
            ["x.y", "x.yy", "x.2.0", "x.2.1", "x.2.2", "x.2.3", "z"]
        );
    }

    #[test]
    fn test_root_leaf_has_no_address() {
        let tree = DataTree::Leaf(42);
        assert_eq!(tree.dotted_paths(), [""]);
        assert_eq!(tree.get_by_str_key(""), Some(&tree));
    }

    #[test]
    fn test_every_dotted_path_resolves_to_its_leaf() {
        let tree = mixed_tree();
        let paths = tree.dotted_paths();
        let leaves: Vec<_> = tree.iter_leaves().copied().collect();
        assert_eq!(paths.len(), leaves.len());
        for (path, leaf) in paths.iter().zip(leaves) {
            assert_eq!(
                tree.get_by_str_key(path),
                Some(&DataTree::Leaf(leaf)),
                "at {path}"
            );
        }
    }

    #[test]
    fn test_dotted_paths_resolve_against_the_structure_they_came_from() {
        let tree = mixed_tree();
        let structure = tree.structure();
        for path in tree.dotted_paths() {
            assert_eq!(
                structure.get_by_str_key(&path),
                Some(&DataTree::Leaf(())),
                "at {path}"
            );
        }
    }

    #[test]
    fn test_resolve_addresses_a_named_child_by_position_too() {
        let tree = DataTree::mapping([("a", DataTree::Leaf(1))]).unwrap();
        assert_eq!(tree.get_by_str_key("a"), Some(&DataTree::Leaf(1)));
        assert_eq!(tree.get_by_str_key("0"), Some(&DataTree::Leaf(1)));
        // The generated path is the name.
        assert_eq!(tree.dotted_paths(), ["a"]);
    }

    #[test]
    fn test_resolve_reports_a_path_that_addresses_nothing() {
        let tree = mixed_tree();
        assert_eq!(tree.get_by_str_key("2"), None);
        assert_eq!(tree.get_by_str_key("x.4"), None);
        assert_eq!(tree.get_by_str_key("x.2.4"), None);
        assert_eq!(tree.get_by_str_key("99999999999999999999"), None);
    }
}
