// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::borrow::{Borrow, Cow};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

use qiskit_util::IndexSet;
use smallvec::SmallVec;

/// A key to retrieve a value (by reference) from an interner of the same type.  This is narrower
/// than a true reference, at the cost that it is explicitly not lifetime bound to the interner it
/// came from; it is up to the user to ensure that they never attempt to query an interner with a
/// key from a different interner.
#[derive(Debug, Eq, PartialEq, Hash)]
pub struct Interned<T: ?Sized> {
    index: u32,
    // Storing the type of the interned value adds a small amount more type safety to the interner
    // keys when there's several interners in play close to each other.  We use `*const T` because
    // the `Interned value` is like a non-lifetime-bound reference to data stored in the interner;
    // `Interned` doesn't own the data (which would be implied by `T`), and it's not using the
    // static lifetime system (which would be implied by `&'_ T`, and require us to propagate the
    // lifetime bound).
    _type: PhantomData<*const T>,
}
// The `PhantomData` marker prevents various useful things from being derived (for `Clone` and
// `Copy` it's an awkward effect of the derivation system), so we have manual implementations.
impl<T: ?Sized> Clone for Interned<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: ?Sized> Copy for Interned<T> {}
impl<T: ?Sized> Default for Interned<T>
where
    T: ToOwned<Owned: Default>,
{
    fn default() -> Self {
        Self {
            index: 0,
            _type: PhantomData,
        }
    }
}
unsafe impl<T: ?Sized> Send for Interned<T> {}
unsafe impl<T: ?Sized> Sync for Interned<T> {}

/// A map of the indices from one interner to another.
///
/// This is created by the interner-merging functions like [Interner::merge_map] and
/// [Interner::merge_slice_map].
///
/// This map can be indexed by the [Interned] keys of the smaller [Interner], and returns [Interned]
/// keys that work on the larger [Interner] (the one that expanded itself).
///
/// The indexing implementation panics if asked for the new key for an object that was filtered out
/// during the merge.
#[derive(Clone, Debug, Default)]
pub struct InternedMap<S: ?Sized, T: ?Sized = S> {
    // We can use [Vec] here, because [Interner] keys are guaranteed to be consecutive integers
    // counting from zero; it's effectively how an [Interner] does lookups from [Interned] already.
    // The [Option] is to support filtering in the map; we don't use a hash-map because we expect
    // filtering to only infrequently remove values.
    map: Vec<Option<Interned<T>>>,
    // We're pretending that we're a mapping type from [Interned<S>] to [Interned<T>].
    _other: PhantomData<Interned<S>>,
}
impl<S: ?Sized, T: ?Sized> InternedMap<S, T> {
    /// Create a new empty [InternedMap].
    ///
    /// You can use this as a persistent allocation for repeated calls to [Interner::merge_map] or
    /// related functions.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    /// Create a new empty [InternedMap] with pre-allocated capacity.
    ///
    /// You can use this as a persistent allocation for repeated calls to [Interner::merge_map] or
    /// related functions.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: Vec::with_capacity(cap),
            _other: PhantomData,
        }
    }

    /// An iterator over the pairs of values in the map.
    ///
    /// The first item of the tuple is the keys that can be used to index the map, the second is the
    /// result from mapping that key.
    pub fn iter(&self) -> impl Iterator<Item = (Interned<S>, Interned<T>)> + '_ {
        self.map.iter().enumerate().filter_map(|(key, value)| {
            value.map(|value| {
                (
                    Interned {
                        index: key as u32,
                        _type: PhantomData,
                    },
                    value,
                )
            })
        })
    }
}
impl<S: ?Sized, T: ?Sized> ::std::ops::Index<Interned<S>> for InternedMap<S, T> {
    type Output = Interned<T>;

    fn index(&self, index: Interned<S>) -> &Self::Output {
        // We could write a fallable [Interner::get] for handling filtered keys safely, but I
        // couldn't imagine a use-case for that.
        self.map[index.index as usize]
            .as_ref()
            .expect("lookup keys should not have been filtered out")
    }
}
impl<S: ?Sized, T: ?Sized> IntoIterator for InternedMap<S, T> {
    type Item = <interned_map::IntoIter<S, T> as Iterator>::Item;
    type IntoIter = interned_map::IntoIter<S, T>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter::from(self)
    }
}
// Private namespace to hide the types of the iterator in.
mod interned_map {
    use super::*;
    use std::{iter, vec};

    pub struct IntoIter<S: ?Sized, T: ?Sized = S> {
        // This ugly type is to try and re-use as much of the built-in [Iterator]-adaptor structure
        // as possible.  We have to stop when we encounter what would be a [FilterMap] because we
        // can't name the type of the mapping function.
        iter: iter::Enumerate<vec::IntoIter<Option<Interned<T>>>>,
        _type: PhantomData<S>,
    }
    impl<S: ?Sized, T: ?Sized> From<InternedMap<S, T>> for IntoIter<S, T> {
        fn from(val: InternedMap<S, T>) -> Self {
            Self {
                iter: val.map.into_iter().enumerate(),
                _type: PhantomData,
            }
        }
    }
    impl<S: ?Sized, T: ?Sized> Iterator for IntoIter<S, T> {
        type Item = (Interned<S>, Interned<T>);
        fn next(&mut self) -> Option<Self::Item> {
            for (key, value) in self.iter.by_ref() {
                let Some(value) = value else {
                    continue;
                };
                return Some((
                    Interned {
                        index: key as u32,
                        _type: PhantomData,
                    },
                    value,
                ));
            }
            None
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            self.iter.size_hint()
        }
    }
    impl<S: ?Sized, T: ?Sized> ExactSizeIterator for IntoIter<S, T> {}
    impl<S: ?Sized, T: ?Sized> iter::FusedIterator for IntoIter<S, T> {}
}

/// An append-only data structure for interning generic Rust types.
///
/// The interner can lookup keys using a reference type, and will create the corresponding owned
/// allocation on demand, if a matching entry is not already stored.  It returns manual keys into
/// itself (the `Interned` type), rather than raw references; the `Interned` type is narrower than a
/// true reference.
///
/// This is only implemented for owned types that implement `Default`, so that the convenience
/// method `Interner::get_default` can work reliably and correctly; the "default" index needs to be
/// guaranteed to be reserved and present for safety.
///
/// # Examples
///
/// ```rust
/// let mut interner = Interner::<[usize]>::new();
///
/// // These are of type `Interned<[usize]>`.
/// let default_empty = interner.get_default();
/// let empty = interner.insert(&[]);
/// let other_empty = interner.insert(&[]);
/// let key = interner.insert(&[0, 1, 2, 3, 4]);
///
/// assert_eq!(empty, other_empty);
/// assert_eq!(empty, default_empty);
/// assert_ne!(empty, key);
///
/// assert_eq!(interner.get(empty), &[]);
/// assert_eq!(interner.get(key), &[0, 1, 2, 3, 4]);
/// ```
#[derive(Default)]
pub struct Interner<T: ?Sized + ToOwned>(IndexSet<<T as ToOwned>::Owned>);

// `Clone` and `Debug` can't use the derivation mechanism because the values that are actually
// stored are of type `<T as ToOwned>::Owned`, which the derive system doesn't reason about.
impl<T> Clone for Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T> fmt::Debug for Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_tuple("Interner").field(&self.0).finish()
    }
}
// We can choose either [FromIterator<T>] or `FromIterator<<T as ToOwned>::Owned>` as the
// implementation for [Interner<T>], but we can't have both, because the blanket implementation of
// [ToOwned] for `T: Clone` would cause overlap.  If somebody's constructing a new [Interner] from
// an iterator, chances are that they've either already got owned values, or there aren't going to
// be too many duplicates.
impl<T> ::std::iter::FromIterator<<T as ToOwned>::Owned> for Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: Hash + Eq + Default,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = <T as ToOwned>::Owned>,
    {
        let iter = iter.into_iter();
        let (min, _) = iter.size_hint();
        let mut out = Self::with_capacity(min + 1);
        for x in iter {
            out.insert_owned(x);
        }
        out
    }
}

impl<T> Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: Hash + Eq + Default,
{
    /// Construct a new interner.  The stored type must have a default value, in order for
    /// `Interner::get_default` to reliably work correctly without a hash lookup (though ideally
    /// we'd just use specialisation to do that).
    pub fn new() -> Self {
        Self::with_capacity(1)
    }

    /// Retrieve the key corresponding to the default store, without any hash or equality lookup.
    /// For example, if the interned type is `[Clbit]`, the default key corresponds to the empty
    /// slice `&[]`.  This is a common operation with the cargs interner, for things like pushing
    /// gates.
    ///
    /// In an ideal world, we wouldn't have the `Default` trait bound on `new`, but would use
    /// specialisation to insert the default key only if the stored value implemented `Default`
    /// (we'd still trait-bound this method).
    #[inline(always)]
    pub fn get_default(&self) -> Interned<T> {
        Interned {
            index: 0,
            _type: PhantomData,
        }
    }

    /// Create an interner with enough space to hold `capacity` entries.
    ///
    /// Note that the default item of the interner is always allocated and given a key immediately,
    /// which will use one slot of the capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        let mut set =
            IndexSet::with_capacity_and_hasher(capacity, ::foldhash::fast::RandomState::default());
        set.insert(Default::default());
        Self(set)
    }
}

impl<T> Interner<T>
where
    T: ?Sized + ToOwned,
{
    /// Retrieve a reference to the stored value for this key.
    pub fn get(&self, index: Interned<T>) -> &T {
        self.0
            .get_index(index.index as usize)
            .expect(
                "the caller is responsible for only using interner keys from the correct interner",
            )
            .borrow()
    }

    /// The number of entries stored in the interner.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether there are zero stored keys.
    ///
    /// This is always false, because we always contain a default key, but clippy complains if we
    /// don't have it.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// An iterator over the [Interned] keys.
    pub fn keys(&self) -> impl ExactSizeIterator<Item = Interned<T>> + '_ {
        (0..self.len() as u32).map(|index| Interned {
            index,
            _type: PhantomData,
        })
    }

    /// An iterator over the stored values.
    pub fn values(&self) -> impl ExactSizeIterator<Item = &'_ T> + '_ {
        self.0.iter().map(|x| x.borrow())
    }

    /// An iterator over pairs of the [Interned] keys and their associated values.
    pub fn items(&self) -> impl ExactSizeIterator<Item = (Interned<T>, &'_ T)> + '_ {
        self.0.iter().enumerate().map(|(i, v)| {
            (
                Interned {
                    index: i as u32,
                    _type: PhantomData,
                },
                v.borrow(),
            )
        })
    }
}

impl<T> Interner<T>
where
    T: ?Sized + ToOwned + Hash + Eq,
{
    /// Get the [Interned] key corresponding to the given borrowed example, if it has already been
    /// stored.
    ///
    /// This method does not store `value` if it is not present.
    pub fn try_key(&self, value: &T) -> Option<Interned<T>> {
        self.0.get_index_of(value).map(|index| Interned {
            index: index as u32,
            _type: PhantomData,
        })
    }

    /// Return whether this value is already in the [Interner].
    ///
    /// Typically you want to use [try_key], which returns the key if present, or [insert], which
    /// stores the value if it wasn't already present.
    pub fn contains(&self, value: &T) -> bool {
        self.try_key(value).is_some()
    }
}

impl<T> Interner<T>
where
    T: ?Sized + ToOwned,
    <T as ToOwned>::Owned: Hash + Eq,
{
    /// Internal worker function that inserts an owned value assuming that the value didn't
    /// previously exist in the map.
    fn insert_new(&mut self, value: <T as ToOwned>::Owned) -> u32 {
        let index = self.0.len();
        if index == u32::MAX as usize {
            panic!("interner is out of space");
        }
        let _inserted = self.0.insert(value);
        debug_assert!(_inserted);
        index as u32
    }

    /// Get an interner key corresponding to the given owned type.  If not already stored, the value
    /// will be used as the key, otherwise it will be dropped.
    ///
    /// If you don't already have the owned value, use `insert`; this will only allocate if the
    /// lookup fails.
    pub fn insert_owned(&mut self, value: <T as ToOwned>::Owned) -> Interned<T> {
        let index = match self.0.get_index_of(&value) {
            Some(index) => index as u32,
            None => self.insert_new(value),
        };
        Interned {
            index,
            _type: PhantomData,
        }
    }
}

impl<T> Interner<T>
where
    T: ?Sized + ToOwned + Hash + Eq,
    <T as ToOwned>::Owned: Hash + Eq,
{
    /// Get an interner key corresponding to the given referenced type.  If not already stored, this
    /// function will allocate a new owned value to use as the storage.
    ///
    /// If you already have an owned value, use `insert_owned`, but in general this function will be
    /// more efficient *unless* you already had the value for other reasons.
    pub fn insert(&mut self, value: &T) -> Interned<T> {
        let index = match self.0.get_index_of(value) {
            Some(index) => index as u32,
            None => self.insert_new(value.to_owned()),
        };
        Interned {
            index,
            _type: PhantomData,
        }
    }

    /// Get an interner key corresponding to the given [Cow].
    ///
    /// If not already stored, the value will be used as the key, cloning if required.  If it is
    /// stored, the value is dropped.
    #[inline]
    pub fn insert_cow(&mut self, value: Cow<T>) -> Interned<T> {
        match value {
            Cow::Borrowed(value) => self.insert(value),
            Cow::Owned(value) => self.insert_owned(value),
        }
    }

    /// Merge another interner into this one, re-using the output storage for the key mapping.
    ///
    /// The output mapping converts [Interned] indices from `other` to their new representations in
    /// `self`.  Strictly, the interners can be for different types, but in practice it likely makes
    /// most sense for them to be the same.
    pub fn merge_map_using<S>(
        &mut self,
        other: &Interner<S>,
        mut map_fn: impl FnMut(&S) -> Option<Cow<T>>,
        target: &mut InternedMap<S, T>,
    ) where
        S: ?Sized + ToOwned + Hash + Eq,
    {
        target.map.clear();
        target.map.reserve(other.0.len());
        for key in other.0.iter() {
            target
                .map
                .push(map_fn(key.borrow()).map(|cow| self.insert_cow(cow)));
        }
    }

    /// Merge another interner into this one.
    ///
    /// The output mapping converts [Interned] indices from `other` to their new representations in
    /// `self`.  Strictly, the interners can be for different types, but in practice it likely makes
    /// most sense for them to be the same.
    pub fn merge_map<S>(
        &mut self,
        other: &Interner<S>,
        map_fn: impl FnMut(&S) -> Option<Cow<T>>,
    ) -> InternedMap<S, T>
    where
        S: ?Sized + ToOwned + Hash + Eq,
    {
        let mut out = InternedMap::new();
        self.merge_map_using(other, map_fn, &mut out);
        out
    }

    /// Rewrite all the interned values in `self` inplace, maintaining validity of all existing
    /// [Interned] keys (just with mapped values).
    ///
    /// The `map_fn` will be called exactly once for each of the non-default values; the default
    /// value will not be passed to the `map_fn`, as it is a requirement of the type that the
    /// default key matches the default value.
    ///
    /// # Panics
    ///
    /// If the `map_fn` returns two equal values for non-equal inputs; if this happened, we wouldn't
    /// have the same number of keys in the output.
    pub fn map_inplace(
        &mut self,
        mut map_fn: impl FnMut(<T as ToOwned>::Owned) -> <T as ToOwned>::Owned,
    ) {
        // This isn't a one-at-a-time operation because we might have a collision, where a new
        // value with an early key is equal to an old value with a later key.  That will be resolved
        // by the end, but if we try to go one-at-a-time, we'll mess up the key order.  We can still
        // re-use the allocation of the interner, though.
        let mut values = self.0.drain(..).collect::<Vec<_>>().into_iter();
        self.0
            .insert(values.next().expect("must have at least the default key"));
        for (i, value) in values.enumerate() {
            let expected = i as u32 + 1;
            let actual = self.insert_owned(map_fn(value)).index;
            assert!(actual == expected, "mapping returned a duplicate key");
        }
    }
}

impl<T> Interner<[T]>
where
    T: Hash + Eq + Clone,
{
    /// Merge another interner into this one, re-using the output storage for the key mapping.
    ///
    /// The mapping function is for scalar elements of the slice, as opposed to in [merge_map] where
    /// it is for the entire key at once.  This function makes it easier to avoid allocations when
    /// mapping slice-based conversions (though if `T` is not [Copy] and you're expecting there to
    /// be a lot of true insertions during the merge, there is a potential clone inefficiency).
    ///
    /// If the `scalar_map_fn` returns `None` for any element of a slice, that entire slice is
    /// filtered out from the merge.  The subsequent [InternedMap] will panic if the corresponding
    /// [Interned] key is used as a lookup.
    pub fn merge_map_slice_using<const N: usize>(
        &mut self,
        // Actually, `other` could be [Interner<[S]>], but then you'd need to specify `S` whenever
        // you want to set `N`, which is just an API annoyance since we'll probably never need the
        // two interners to be different types.
        other: &Self,
        mut scalar_map_fn: impl FnMut(&T) -> Option<T>,
        target: &mut InternedMap<[T]>,
    ) {
        // Workspace for the mapping function. The aim here is that we're working on the stack, so
        // the mapping doesn't need to make heap allocations.  We could either guess (which the
        // higher-level `merge_slice_map` does), or force the caller to tell us how much stack space
        // to allocate.  This method is lower level, so in this case we ask them to tell us; if
        // they're optimizing to the point of re-using the return allocations, they probably have a
        // good idea about the maximum slice size of the interner they'll be merging in.
        let mut work = SmallVec::<[T; N]>::with_capacity(N);
        target.map.clear();
        target.map.reserve(other.0.len());
        for slice in other.0.iter() {
            let new_slice = 'slice: {
                work.clear();
                work.reserve(slice.len());
                for value in slice {
                    let Some(scalar) = scalar_map_fn(value) else {
                        break 'slice None;
                    };
                    work.push(scalar);
                }
                Some(work.as_slice())
            };
            target.map.push(new_slice.map(|slice| self.insert(slice)));
        }
    }

    /// Merge another interner into this one.
    ///
    /// If you need to call this many times in a row, see [merge_map_slice_using] for a version that
    /// can re-use the allocations of the output mapping.
    ///
    /// The mapping function is for scalar elements of the slice, as opposed to in [merge_map] where
    /// it is for the entire key at once.  This function makes it easier to avoid allocations when
    /// mapping slice-based conversions (though if `T` is not [Copy] and you're expecting there to
    /// be a lot of true insertions during the merge, there is a potential clone inefficiency).
    ///
    /// If the `scalar_map_fn` returns `None` for any element of a slice, that entire slice is
    /// filtered out from the merge.  The subsequent [InternedMap] will panic if the corresponding
    /// [Interned] key is used as a lookup.
    pub fn merge_map_slice(
        &mut self,
        other: &Self,
        scalar_map_fn: impl FnMut(&T) -> Option<T>,
    ) -> InternedMap<[T]> {
        let mut out = InternedMap::new();
        // We're specifying the stack space here.  This is just a guess, but it's not hugely
        // important; we'll safely spill from the stack to the heap if needed, and this function is
        // an API convenience at the cost of optimal allocation performance anyway.
        self.merge_map_slice_using::<4>(other, scalar_map_fn, &mut out);
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use hashbrown::{HashMap, HashSet};

    #[test]
    fn default_key_exists() {
        let mut interner = Interner::<[u32]>::new();
        assert_eq!(interner.get_default(), interner.get_default());
        let res: &[u32] = &[];
        assert_eq!(interner.get(interner.get_default()), res);
        assert_eq!(interner.insert_owned(Vec::new()), interner.get_default());
        assert_eq!(interner.insert(&[]), interner.get_default());

        let capacity = Interner::<str>::with_capacity(4);
        assert_eq!(capacity.get_default(), capacity.get_default());
        assert_eq!(capacity.get(capacity.get_default()), "");
    }

    #[test]
    fn can_merge_two_interners() {
        let mut base = Interner::<str>::from_iter(["hello", "world"].map(String::from));
        let other = Interner::<str>::from_iter(["a", "world", "b"].map(String::from));

        fn to_hashmap<T: ?Sized + Hash + Eq + ToOwned>(
            interner: &Interner<T>,
        ) -> HashMap<Interned<T>, <T as ToOwned>::Owned> {
            interner
                .items()
                .map(|(key, value)| (key, value.to_owned()))
                .collect()
        }

        let initial = to_hashmap(&base);
        // Sanity check that we start off with the values we expect.
        let expected = ["", "hello", "world"]
            .map(String::from)
            .into_iter()
            .collect::<HashSet<_>>();
        assert_eq!(
            expected,
            HashSet::from_iter(base.values().map(String::from))
        );

        let other_items = to_hashmap(&other);
        let other_map = base.merge_map(&other, |x| Some(x.into()));
        // All of the keys from the previously stored values must be the same.
        assert_eq!(
            initial,
            initial
                .iter()
                .map(|(key, value)| (base.try_key(value).unwrap(), base.get(*key).to_owned()))
                .collect::<HashMap<_, _>>(),
        );
        // All of the keys from the merged-in map should now be present.
        assert_eq!(
            other_items,
            other
                .keys()
                .map(|key| (key, base.get(other_map[key]).to_owned()))
                .collect::<HashMap<_, _>>(),
        );

        // This interner is of a different type and will produce duplicate keys during the mapping.
        let nums = Interner::<[u8]>::from_iter([vec![4], vec![1, 5], vec![2, 4], vec![3]]);
        let map_fn = |x: &[u8]| x.iter().sum::<u8>().to_string();
        let num_map = base.merge_map(&nums, |x| Some(map_fn(x).into()));
        // All of the keys from the previously stored values must be the same.
        assert_eq!(
            initial,
            initial
                .iter()
                .map(|(key, value)| (base.try_key(value).unwrap(), base.get(*key).to_owned()))
                .collect::<HashMap<_, _>>(),
        );
        // All of the keys from the merged-in map should now be present.
        assert_eq!(
            nums.items()
                .map(|(key, value)| (key, map_fn(value)))
                .collect::<HashMap<_, _>>(),
            nums.keys()
                .map(|key| (key, base.get(num_map[key]).to_owned()))
                .collect(),
        );
    }

    #[test]
    fn can_merge_two_sliced_interners() {
        let mut map = InternedMap::<[u8]>::new();
        let mut base = Interner::<[u8]>::from_iter([
            vec![],
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![1, 2],
        ]);
        let only_2q = Interner::<[u8]>::from_iter([vec![0], vec![1], vec![0, 1]]);

        // This is the identity map, so all the values should come out the same.
        base.merge_map_slice_using::<2>(&only_2q, |x| Some(*x), &mut map);
        let expected = [vec![], vec![0], vec![1], vec![0, 1]];
        let (small, big): (Vec<_>, Vec<_>) = expected
            .iter()
            .map(|x| {
                let key = only_2q.try_key(x).unwrap();
                (only_2q.get(key).to_owned(), base.get(map[key]).to_owned())
            })
            .unzip();
        assert_eq!(small, big);

        // Map qubits [0, 1] to [2, 1].  This involves an insertion.
        base.merge_map_slice_using::<2>(&only_2q, |x| [2u8, 1].get(*x as usize).copied(), &mut map);
        let expected = HashSet::<(Vec<u8>, Vec<u8>)>::from([
            (vec![], vec![]),
            (vec![0], vec![2]),
            (vec![1], vec![1]),
            (vec![0, 1], vec![2, 1]),
        ]);
        let actual = map
            .iter()
            .map(|(small, big)| (only_2q.get(small).to_owned(), base.get(big).to_owned()))
            .collect::<HashSet<_>>();
        assert_eq!(expected, actual);
        assert_eq!(&[2, 1], base.get(map[only_2q.try_key(&[0, 1]).unwrap()]));

        // Map qubit [0] to [3], and drop things involving 1.
        base.merge_map_slice_using::<2>(&only_2q, |x| [3u8].get(*x as usize).copied(), &mut map);
        let expected = HashSet::<(Vec<u8>, Vec<u8>)>::from([(vec![], vec![]), (vec![0], vec![3])]);
        // For the last test, we'll also use the `into_iter` method.
        let actual = map
            .into_iter()
            .map(|(small, big)| (only_2q.get(small).to_owned(), base.get(big).to_owned()))
            .collect::<HashSet<_>>();
        assert_eq!(expected, actual);
    }
}
