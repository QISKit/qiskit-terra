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

//! The "quantum identifier" (qid) system.

use std::{cmp, fmt, num};

pub mod table;

pub use table::ResourceTable;

/// A single quantum identifier, representing some sort of "quantum resource" which follows the
/// [quantum semantics](#quantum-semantics) defined below.
///
/// A qid has two components:
///
/// * a "memory space" identifier ([`space`](Self::space));
/// * an integer that marks unique qids within a given memory space ([`value`](Self::value)).
///
/// The memory-space component takes one of the values enumerated by [`QidSpace`].  The semantics of
/// the particular quantum resource in MIR differs based on the memory space its qid is part of, but
/// the [quantum semantics](#quantum-semantics) are always the same.
///
///
/// # Quantum semantics
///
/// The main purpose of [`Qid`] is to define restrictions on the use of quantum resources.  You can
/// use [`Qid`] to model any resource you like that follows these semantics.  The two rules are:
///
/// - qids are neither defined nor consumed by MIR instructions, they are mutated;
/// - no MIR instruction can refer to two qids that "overlap".
///
/// The first rule is enforced in the type system by the structure of MIR instructions.  The second
/// rule is checked by the verification logic; it is an error for a complete MIR transformation to
/// leave reachable instructions in place that violate the overlap rules.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Qid(u32);
impl Qid {
    /// The bitwidth allowed for use by values.
    pub const WIDTH: u32 = 30;
    /// A bitmask that selects only the high bits used to indicate the "memory space" of the
    /// [`Qid`].
    pub const SPACE_MASK: u32 = u32::MAX << Self::WIDTH;
    /// The maximum allowed [`value`](Self::value) of a [`Qid`].
    pub const MAX_VALUE: u32 = !Self::SPACE_MASK;
    /// The maximum raw integer value that is a valid [`Qid`].  This includes the largest memory
    /// space, and is not valid as a [`Qid::value`].
    pub const MAX_RAW: u32 =
        ((<QidSpace as bytemuck::Contiguous>::MAX_VALUE as u32) << Self::WIDTH) | Self::MAX_VALUE;

    /// Create a new [`Qid`] from an explicit space and value.
    ///
    /// In debug mode, this panics if the value is too large.  In release mode, it will silently
    /// drop the disallowed high bits.  Use [`from_parts_checked`](Self::from_parts_checked) if you
    /// are unsure of the value size.
    #[inline]
    pub const fn from_parts(space: QidSpace, val: u32) -> Self {
        debug_assert!(val <= Self::MAX_VALUE);
        Self(space.as_mask() | (val & !Self::SPACE_MASK))
    }

    /// Create a new [`Qid`] from an explicit space and value, if it is in range.
    #[inline]
    pub const fn from_parts_checked(space: QidSpace, val: u32) -> Option<Self> {
        if val <= Self::MAX_VALUE {
            Some(Self::from_parts(space, val))
        } else {
            None
        }
    }

    /// Create a virtual [`Qid`] with the given in-range value.
    ///
    /// In debug mode, this panics if the value is too large.  In release mode, it will silently
    /// drop the disallowed high bits.  Use [`virt_checked`](Self::virt_checked) if you are unsure
    /// on the value size.
    #[inline]
    pub const fn virt(val: u32) -> Self {
        Self::from_parts(QidSpace::Virtual, val)
    }
    /// Create a virtual [`Qid`] with the given value, if it is in range.
    #[inline]
    pub const fn virt_checked(val: u32) -> Option<Self> {
        Self::from_parts_checked(QidSpace::Virtual, val)
    }

    /// Create a physical [`Qid`] with the given in-range value.
    ///
    /// In debug mode, this panics if the value is too large.  In release mode, it will silently
    /// drop the disallowed high bits.  Use [`phys_checked`](Self::phys_checked) if you are unsure
    /// on the value size.
    #[inline]
    pub const fn phys(val: u32) -> Self {
        Self::from_parts(QidSpace::Physical, val)
    }
    /// Create a physical [`Qid`] with the given value, if it is in range.
    #[inline]
    pub const fn phys_checked(val: u32) -> Option<Self> {
        Self::from_parts_checked(QidSpace::Physical, val)
    }

    /// Get the integer "value" of the [`Qid`], stripping off the [`space`](Self::space) component.
    #[inline]
    pub const fn value(self) -> u32 {
        self.0 & !Self::SPACE_MASK
    }

    /// The integer "value" component of the [`Qid`], with the [`space`](Self::space) component
    /// stripped, widened to be an indexing integer.
    #[inline]
    pub const fn index(self) -> usize {
        self.value() as usize
    }

    /// Get the "space" this [`Qid`] is part of.
    #[inline]
    pub const fn space(self) -> QidSpace {
        match self.0 >> Self::WIDTH {
            0 => QidSpace::Virtual,
            1 => QidSpace::Physical,
            // This should represent an internal-logic error only; it shouldn't be possible to
            // trigger this unless the construction API has been side-stepped.
            _ => panic!("qids should not store invalid spaces"),
        }
    }
}

/// SAFETY: `Qid` is a transparent wrapper of `u32` in the exact range `0..=Self::MAX_RAW`; the
/// range is inhabited; all distinct bit patterns compare unequal.
unsafe impl bytemuck::Contiguous for Qid {
    type Int = u32;
    const MAX_VALUE: Self::Int = Self::MAX_RAW;
    const MIN_VALUE: Self::Int = 0;
}
// SAFETY: `Qid` is a transparent wrapper of `u32`, and the range `0..=Self::MAX_RAW` is valid.
unsafe impl bytemuck::checked::CheckedBitPattern for Qid {
    type Bits = u32;
    #[inline]
    fn is_valid_bit_pattern(bits: &Self::Bits) -> bool {
        *bits <= Self::MAX_RAW
    }
}
impl fmt::Display for Qid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let key = match self.space() {
            QidSpace::Virtual => "v",
            QidSpace::Physical => "p",
        };
        write!(f, "Qid({}{})", key, self.value())
    }
}

/// The "memory spaces" that a [`Qid`] can exist in.
///
/// The semantics of the [`Qid`] in a complete program varies based on what space it is in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, bytemuck::Zeroable, bytemuck::Contiguous)]
#[repr(u8)]
pub enum QidSpace {
    /// Qids that can be freely created and manipulated within transformations on MIR itself.
    /// Typically, these are used at the higher abstraction levels, and steadily lowered into
    /// [`Physical`](Self::Physical) identifiers, as the complete program is laid out onto hardware.
    ///
    /// It is expected that higher-level abstractions built on top of MIR will freely define new
    /// virtual qids to represent quantum-data groupings that a particular higher-level
    /// abstraction cares about, and define the aliasing rules in that manner.  For example, virtual
    /// qids could be used to model "quantum integers" abstracting over the top of individually
    /// addressable qubits, while the same MIR program might use some different qids to represent
    /// some "unsized ancilla use" that has not yet been sized nor lowered to hardware.
    Virtual = 0,
    /// Qids that represent some fixed physical resource.  These might include (non-exhaustively)
    /// hardware qubits, hardware modules, or factories.
    ///
    /// The resource tables for physical qids are typically *not* eligible for mutation by MIR
    /// transformation logic; they are typically fixed by the backend target.
    Physical = 1,
}
impl QidSpace {
    /// Get a mask that can be bitwise or'd (`|`) with a [`Qid::value`] to produce a complete qid.
    #[inline]
    pub const fn as_mask(self) -> u32 {
        (self as u32) << Qid::WIDTH
    }
}

/// A reference to a single bit in a [`Qid`].
///
/// Logically this is just a two-tuple of `(Qid, u32)`, but with an internal trick to make
/// `Option<QidBit>` the same size as `QidBit`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct QidBit {
    qid: Qid,
    /// The bitwise negation of the index.
    ///
    /// The maximum width of a `Qid` is `u32::MAX`, which means the greatest valid index is
    /// `u32::MAX - 1`.  Therefore, `u32::MAX` is not a valid index, and we can use its bit pattern
    /// to mean "impossible".  Unfortunately, the only type-system trick Rust gives us is to use
    /// `NonZero`.
    neg_bit: num::NonZero<u32>,
}
impl QidBit {
    /// Create a new bit reference.
    ///
    /// # Panics
    ///
    /// If `bit` is [`u32::MAX`].  This is never a valid bit index.
    #[inline]
    #[track_caller]
    pub const fn new(qid: Qid, bit: u32) -> Self {
        Self {
            qid,
            neg_bit: num::NonZero::new(!bit).expect("caller ensures `bit` is not `u32::MAX`"),
        }
    }

    /// The stored [`Qid`].
    #[inline]
    pub const fn qid(self) -> Qid {
        self.qid
    }

    /// The stored bit index of the [`Qid`].
    #[inline]
    pub const fn bit(self) -> u32 {
        !self.neg_bit.get()
    }

    #[inline]
    const fn cmp_key(self) -> impl Ord {
        (self.qid(), self.bit())
    }
}
impl PartialOrd for QidBit {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for QidBit {
    // This is implemented rather than derived because the relevant `bit` is (numerically) stored
    // reversed, so the order would be wrong.
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.cmp_key().cmp(&other.cmp_key())
    }
}

/// A layout for the internal bits of a [`Qid`].
///
/// # Validity
///
/// In general, the validity of this object is dependent on the rest of the context for the [`Qid`]s
/// it refers to.  You can construct a [`BitLayout`] that does not follow the rules, but
/// [`ResourceTable`] methods will reject them.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum BitLayout {
    /// A [`Qid`] that has undefined size.
    ///
    /// These qids can have "overlap" with other qids, but the containing relationships must be
    /// explicitly added to a [`ResourceTable`] using
    /// [`mark_contains`](ResourceTable::mark_contains).
    Abstract,
    /// A [`Qid`] that can be the base of other [`Qid`]s' bit layouts.
    Base(u32),
    /// A [`Qid`] whose bits alias other [`Qid`]s' bits.
    ///
    /// The bitwidth of this layout is equal to the length of the slice.  Any [`None`] entries in
    /// the slice mean that these bits do not alias others.  The `Some` entries should all refer to
    /// qids whose layouts are the [`Base`](Self::Base) variant of this object, and all bits in that
    /// base [`Qid`] should be used by the layout.
    ///
    /// The contained [`Qid`]s are automatically tracked in a [`ResourceTable`].  Additional
    /// "abstract" mutual-exclusion relationships can be added on top of this, similarly to the
    /// [`Abstract`](Self::Abstract) variant.
    ///
    /// The length of this is assumed to not exceed `u32::MAX`.
    Alias(Box<[Option<QidBit>]>),
}
impl BitLayout {
    /// The bitwidth, if any, of this layout.
    pub const fn width(&self) -> Option<u32> {
        match self {
            Self::Abstract => None,
            Self::Base(width) => Some(*width),
            Self::Alias(layout) => Some(layout.len() as u32),
        }
    }
}

/// Generic MIR information about what a [`Qid`] represents.
///
/// This is likely to be extended as MIR develops, and we learn more about what abstractions are
/// suitable; the API split between "abstractly described in MIR" and "concretely described by
/// custom backend target descriptors" is not clear yet.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct QidInfo {
    layout: BitLayout,
}
impl QidInfo {
    /// Create a new info object for a qid with a specified layout.
    #[inline]
    pub const fn new(layout: BitLayout) -> Self {
        Self { layout }
    }

    /// Get the bit width (if defined) of this qid.
    #[inline]
    pub const fn width(&self) -> Option<u32> {
        self.layout.width()
    }

    #[inline]
    pub const fn layout(&self) -> &BitLayout {
        &self.layout
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct QidList(Vec<Qid>);

impl QidList {
    pub fn new(qids: Vec<Qid>) -> Self {
        QidList(qids)
    }
    pub fn inner(&self) -> &Vec<Qid> {
        &self.0
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qid_bit_order() {
        // We use a non-standard storage form and custom `Ord`.
        let mut weird = [
            QidBit::new(Qid::phys(0), 2),
            QidBit::new(Qid::phys(1), 1),
            QidBit::new(Qid::phys(0), 0),
            QidBit::new(Qid::phys(0), 1),
            QidBit::new(Qid::phys(1), 0),
        ];
        weird.sort();
        let expected = [
            QidBit::new(Qid::phys(0), 0),
            QidBit::new(Qid::phys(0), 1),
            QidBit::new(Qid::phys(0), 2),
            QidBit::new(Qid::phys(1), 0),
            QidBit::new(Qid::phys(1), 1),
        ];
        assert_eq!(weird, expected);
    }
}
