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

//! Internals of the decomposer creation and caching for unitary synthesis.
//!
//! The decomposer cache logic makes very specific assumptions about how the cache objects and keys
//! are constructed, so we use this module to localise where all these assumptions are being made,
//! and to enforce a safe API within the rest of the unitary synthesis logic.

use std::cmp::Ordering;
use std::f64::consts::FRAC_PI_4;
use std::hash;
use std::sync::LazyLock;

use approx::relative_eq;
use hashbrown::{HashMap, HashSet, hash_map};
use ndarray::{ArrayView2, CowArray, Ix2};
use num_complex::Complex64;
use qiskit_util::{IndexMap, IndexSet};
use smallvec::{SmallVec, smallvec};

use numpy::convert::ToPyArray;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

use super::{
    Approximation, DecompositionDirection2q, NormalizedFidelity, QpuConstraint, QpuConstraintKind,
    UnitarySynthesisConfig, UsePulseOptimizer,
};
use crate::QiskitError;
use crate::passes::optimize_clifford_t::CLIFFORD_T_GATE_NAMES;
use crate::target::{NormalOperation, Target, TargetOperation};
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::instruction::Instruction;
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyInstruction, PyOpKind, StandardGate,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{PhysicalQubit, imports};
use qiskit_synthesis::discrete_basis::solovay_kitaev::SolovayKitaevSynthesis;
use qiskit_synthesis::euler_one_qubit_decomposer::{EulerBasis, EulerBasisSet};
use qiskit_synthesis::two_qubit_decompose::{
    RXXEquivalent, TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer, TwoQubitGateSequence,
    TwoQubitWeylDecomposition,
};

/// Constructor for a 2q Ising-like decomposer.  This corresponds to
/// [TwoQubitControlledUDecomposer], and requires gates that are locally equivalent to RXX (and
/// construct like it, if they're Python-space objects).
///
/// This constructor considers itself equal to another instance of itself by the standard methods
/// for Rust-native standard gates, and by type referential equality in the case of custom Python
/// objects.
#[derive(Clone, Debug)]
struct ContinuousPauliConstructor {
    source: RXXEquivalent,
    euler: EulerBasis,
}
impl PartialEq for ContinuousPauliConstructor {
    fn eq(&self, other: &Self) -> bool {
        if self.euler != other.euler {
            return false;
        }
        match (&self.source, &other.source) {
            (RXXEquivalent::Standard(left), RXXEquivalent::Standard(right)) => left == right,
            (RXXEquivalent::CustomPython(left), RXXEquivalent::CustomPython(right)) => {
                left.is(right)
            }
            _ => false,
        }
    }
}
impl Eq for ContinuousPauliConstructor {}
impl hash::Hash for ContinuousPauliConstructor {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.euler.hash(state);
        // We don't want to apply these hashing semantics to _all_ `RXXEquivalent` instances, just
        // our own context here.
        ::std::mem::discriminant(&self.source).hash(state);
        match &self.source {
            RXXEquivalent::Standard(standard) => standard.hash(state),
            RXXEquivalent::CustomPython(ob) => ob.as_ptr().hash(state),
        }
    }
}
impl ContinuousPauliConstructor {
    fn try_build(&self) -> PyResult<TwoQubitControlledUDecomposer> {
        match self.source {
            RXXEquivalent::Standard(gate) => TwoQubitControlledUDecomposer::new_inner(
                RXXEquivalent::Standard(gate),
                self.euler.as_str(),
            ),
            RXXEquivalent::CustomPython(ref gate) => Python::attach(|py| {
                TwoQubitControlledUDecomposer::new_inner(
                    RXXEquivalent::CustomPython(gate.clone_ref(py)),
                    self.euler.as_str(),
                )
            }),
        }
    }
}

// TODO: for a first pass, while `XXDecomposer` is still completely in Python space, we just
// have the "get" method directly create the decomposer, and eschew the sharing of decomposers
// for homogeneous qargs.  This can be revisited, even without _completely_ porting XXDecomposer
// to Rust; it would be sufficient, as a first step, to make the `XXEmbodiments`
// static-dictionary logic Rust-native in terms of `StandardGate`, so the constructor can then
// be extracted matrix-free from a `Target`.
#[derive(Clone, Debug)]
struct DiscretePauliConstructor {
    decomposer: Py<PyAny>,
    is_approximate: bool,
}
impl PartialEq for DiscretePauliConstructor {
    fn eq(&self, other: &Self) -> bool {
        self.is_approximate == other.is_approximate && self.decomposer.is(&other.decomposer)
    }
}
impl Eq for DiscretePauliConstructor {}
impl hash::Hash for DiscretePauliConstructor {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.decomposer.as_ptr().hash(state);
        self.is_approximate.hash(state);
    }
}

/// A key for the source of a KAK gate for the decomposer.
///
/// # Warning
///
/// This object is intended to be used as a key for hash-based lookups.  The gate and parameters
/// alone are _not_ suitable as a hashable object (there's all sorts of trouble with the parameters
/// and Python objects in the general case), but by constructing this object, you are asserting that
/// the [key] and [constraint] fields together are sufficiently to uniquely identify the [gate].
/// This is true as long as you derive the [key] directly from a [Target]'s key, and that you never
/// mix more than one [Target] at the same time.
#[derive(Clone, Debug)]
struct StaticKakSource {
    /// The operation 'name' we retrieve this from.  We use this for equality and hashing purposes;
    /// the total constructor, including the parameters has to be derived from either a `Target` or
    /// a set of basis-gate strings.  With this assumption, the 2-tuple of constraint source and
    /// stored name is unique (unless you change out the [Target] mid compilation).
    key: (String, QpuConstraintKind),
    gate: PackedOperation,
    params: SmallVec<[f64; 3]>,
}
impl PartialEq for StaticKakSource {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for StaticKakSource {}
impl hash::Hash for StaticKakSource {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.key.hash(state);
    }
}

impl StaticKakSource {
    pub fn try_cow_array(&self) -> Option<CowArray<'_, Complex64, Ix2>> {
        match self.gate.view() {
            OperationRef::StandardGate(g) => {
                let params = self
                    .params
                    .iter()
                    .copied()
                    .map(Param::Float)
                    .collect::<Vec<_>>();
                g.matrix(&params).map(CowArray::from)
            }
            OperationRef::PyCustom(i) => i.matrix().map(CowArray::from),
            OperationRef::Unitary(u) => Some(CowArray::from(u.matrix_view())),
            _ => None,
        }
    }
}

/// Constructor for a 2q decomposer with a static KAK gate.  This corresponds to
/// [TwoQubitBasisDecomposer].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct StaticKakConstructor {
    source: StaticKakSource,
    euler: EulerBasis,
    fidelity: NormalizedFidelity,
    use_pulse_optimizer: UsePulseOptimizer,
}
impl StaticKakConstructor {
    // TODO: this is wildly overcoupled to the internals of the `TwoQubitBasisDecomposer`, but the
    // real problem is that that class is largely unpredictable for the caller.
    /// Is it possible that the [TwoQubitBasisDecomposer] will apply pulse optimisation and flip the
    /// qubit order of its 2q output?
    fn may_return_backwards_sequence(&self) -> bool {
        self.use_pulse_optimizer != UsePulseOptimizer::Forbid
            && self.source.gate.try_standard_gate() == Some(StandardGate::CX)
            && (self.euler == EulerBasis::ZSX || self.euler == EulerBasis::ZSXX)
    }

    fn try_build(&self) -> PyResult<TwoQubitBasisDecomposer> {
        let matrix = self
            .source
            .try_cow_array()
            .ok_or_else(|| PyValueError::new_err("no matrix can be found for KAK gate"))?;
        TwoQubitBasisDecomposer::new_inner(
            self.source.gate.clone(),
            self.source.params.clone(),
            matrix.view(),
            self.fidelity.get(),
            self.euler,
            self.use_pulse_optimizer.to_py_pulse_optimize(),
        )
    }
}

/// Encapsulated view on the constructor arguments for a 2q decomposer.
///
/// It's quite expensive to _construct_ a decomposer; they have to do fairly significant set-up work
/// to prepare their checking matrices, so we want to minimise how often we do it.  It's rather
/// cheaper to work out what decomposer we _would_ construct, and all of our decomposers are pure
/// with respect to these input arguments, so we can use these constructor arguments as hash keys to
/// a cache of constructed objects.
///
/// # Warning
///
/// Two of these constructors **cannot** be safely compared if they are derived from two different
/// [Target]s; the implementations of [hash::Hash][Hash] and [PartialEq] depend on the uniqueness
/// properties enforced from the base [Target].
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum Decomposer2qConstructor {
    ContinuousPauli(ContinuousPauliConstructor),
    DiscretePauli(DiscretePauliConstructor),
    StaticKak(StaticKakConstructor),
}
impl Decomposer2qConstructor {
    fn may_return_backwards_sequence(&self) -> bool {
        match self {
            Self::ContinuousPauli(_) => false,
            Self::DiscretePauli(_) => false,
            Self::StaticKak(kak) => kak.may_return_backwards_sequence(),
        }
    }
    fn try_build(&self) -> PyResult<Decomposer2q> {
        match self {
            Self::ContinuousPauli(constructor) => {
                constructor.try_build().map(Decomposer2q::ContinuousPauli)
            }
            Self::DiscretePauli(constructor) => Python::attach(|py| {
                let kwargs = [
                    ("approximate", constructor.is_approximate),
                    ("use_dag", false),
                ]
                .into_py_dict(py)?
                .unbind();
                Ok(Decomposer2q::DiscretePauli {
                    decomposer: constructor.decomposer.clone(),
                    kwargs,
                })
            }),
            Self::StaticKak(constructor) => {
                constructor
                    .try_build()
                    .map(|decomposer| Decomposer2q::StaticKak {
                        decomposer: Box::new(decomposer),
                        fidelity: constructor.fidelity,
                    })
            }
        }
    }
}

/// A constructed 2q decomposer, ready to be called on gates ([decompose]).
///
/// These are not intended to be constructed directly by callers in general; we instead expect them
/// to come from a corresponding [Decomposer2qConstructor::try_build] call.
#[derive(Clone, Debug)]
pub enum Decomposer2q {
    ContinuousPauli(TwoQubitControlledUDecomposer),
    DiscretePauli {
        decomposer: Py<PyAny>,
        kwargs: Py<PyDict>,
    },
    StaticKak {
        decomposer: Box<TwoQubitBasisDecomposer>,
        fidelity: NormalizedFidelity,
    },
}
impl Decomposer2q {
    pub fn decompose(&self, matrix: ArrayView2<Complex64>) -> PyResult<TwoQubitGateSequence> {
        match self {
            Self::ContinuousPauli(decomposer) => decomposer.call_inner(matrix, None),
            Self::DiscretePauli { decomposer, kwargs } => Python::attach(|py| {
                let circuit = decomposer
                    .bind(py)
                    .call((matrix.to_pyarray(py),), Some(kwargs.bind(py)))?
                    .getattr(intern!(py, "_data"))?
                    .cast_into::<PyCircuitData>()?;
                circuit_to_2q_sequence(&circuit.borrow())
            }),
            Self::StaticKak {
                decomposer,
                fidelity,
            } => decomposer.call_inner(matrix, None, fidelity.get() < 1.0, None),
        }
    }
}

/// Cache of constructed 2q decomposers.
///
/// A stored `None` variant indicates that a given constructor is known to fail to construct a valid
/// decomposer (such as a custom Python object that isn't supercontrolled for
/// [TwoQubitBasisConstructor]).
///
/// This exists mostly just to attach methods to and to shorten a bunch of type signatures.
#[derive(Clone, Debug, Default)]
struct Decomposer2qCacheInner(IndexMap<Decomposer2qConstructor, Option<Decomposer2q>>);
impl Decomposer2qCacheInner {
    /// Get a decomposer by known-good index.
    ///
    /// The [Option] denotes whether the construction succeeded ([Some]) or not ([None}).
    ///
    /// # Panics
    ///
    /// If the index is out of bounds; only indices returned by [cache] should be used.
    fn get(&self, index: usize) -> Option<&Decomposer2q> {
        self.0
            .get_index(index)
            .expect("caller is responsible for only passing in-bounds indices")
            .1
            .as_ref()
    }

    /// Get the index of a decomposer in the cache, constructing it only if required.
    ///
    /// This suppresses errors during the construction, and just marks that construction failed.
    fn cache(&mut self, constructor: Decomposer2qConstructor) -> Option<usize> {
        let entry = self.0.entry(constructor);
        let index = entry.index();
        entry
            .or_insert_with_key(|constructor| constructor.try_build().ok())
            .as_ref()
            .map(|_| index)
    }
}

/// Cache of suitable 1q and 2q decomposers for any seen `qargs` on a given hardware site.
///
/// The caching within this object is largely straightforwards for 1q gates (effectively, we're just
/// memoising checks of the `EulerBasisSet`), and a more complex two-level structure for 2q gates.
///
/// For 2q decomposers, there is (typically) a significant amount of preparatory work to be done
/// before starting decomposition on any given matrix, so we very much want to minimize how many
/// times we build a decomposer at all.  For loose constraints (basis gates + coupling map), this is
/// not much of a concern, because all we need to do is instantiate the right decomposers for the
/// basis gates, and make those same decomposers available whenever a valid link is queried.  For
/// [Target]s, though, we have to take a lot more care around heterogeneity;
///
/// A [Target] might have different available 2q operations on different qubits, but in the vast
/// majority of cases, it will be the same basis-gates set on every valid link of the effective
/// coupling graph.  A naive implementation of caching would cache based purely on the observed
/// `qargs`.  This would prevent decomposers from being reconstructed when the same qargs are used
/// more than once (which _is_ rather likely in a deep circuit), but would still involve duplication
/// when two links shared the same basis set.  Instead, we have a two-level structure:
///
/// 1. Before constructing a decomposer, we cache its construction arguments
///    ([Decomposer2qConstructor]), and only construct the decomposer if it has not been seen
///    before.
/// 2. For each 2q link, we cache the list of (step-1 cached) decomposers that are valid on that
///    link.  We store whether we're using the decomposer in "forwards" mode or "backwards" mode
///    (relative to the `qargs` order) separately to the decomposer itself, since we can do cheap
///    matrix tricks to swap the qubit order of a 2q matrix.
///
/// Taken together, this means that that default settings of `UnitarySynthesis` with a homogeneous
/// [Target] will construct the exact same number of decomposers as the corresponding
/// loose-constraints form (which is explicitly homogenous) would.
#[derive(Clone, Debug, Default)]
pub struct DecomposerCache {
    /// Mapping from physical qubits to its allowed Euler basis decomposers.
    ///
    /// The loose-coupling version is stored in `PhysicalQubit::MAX` if seen.
    euler_bases_1q: HashMap<PhysicalQubit, EulerBasisSet>,
    permits_solovay_kitaev: HashMap<PhysicalQubit, bool>,
    solovay_kitaev: Option<SolovayKitaevSynthesis>,
    /// Mapping from each two-qubit link to a list of (decomposer index into `decompose_2q_cache`,
    /// whether the direction needs flipping) pairs.
    decomposers_2q: HashMap<[PhysicalQubit; 2], Vec<(usize, FlipDirection)>>,
    decompose_2q_cache: Decomposer2qCacheInner,
}
impl DecomposerCache {
    /// Get which Euler decomposers are available on a given qubit.
    pub fn get_euler_1q(
        &mut self,
        qubit: PhysicalQubit,
        constraint: QpuConstraint,
    ) -> EulerBasisSet {
        match constraint {
            QpuConstraint::Target(target) => *self
                .euler_bases_1q
                .entry(qubit)
                .or_insert_with(|| euler_bases_from_target(target, qubit)),
            QpuConstraint::Loose { basis_gates, .. } => *self
                .euler_bases_1q
                .entry(PhysicalQubit::MAX)
                .or_insert_with(|| EulerBasisSet::from_support(|gate| basis_gates.contains(gate))),
        }
    }

    /// Get (or initialize) the "standard" Solovay Kitaev decomposer, if the given qubit only
    /// permits a discrete basis set.
    pub fn try_solovay_kitaev(
        &mut self,
        qubit: PhysicalQubit,
        constraint: QpuConstraint,
    ) -> Option<&SolovayKitaevSynthesis> {
        let valid_clifford_t = |name: &str| {
            static SKIP_NAMES: LazyLock<HashSet<&str>> = LazyLock::new(|| {
                HashSet::from([
                    "for_loop",
                    "while_loop",
                    "if_else",
                    "switch_case",
                    "continue_loop",
                    "break_loop",
                    "box",
                    "delay",
                    "measure",
                    "reset",
                    "barrier",
                ])
            });
            CLIFFORD_T_GATE_NAMES.contains(&name) || SKIP_NAMES.contains(name)
        };
        // TODO: this logic isn't really correct; we probably actually need to test if the basis set
        // permits _complete_ coverage of SU2/SO3 via SK.
        let permits_solovay_kitaev = || match constraint {
            QpuConstraint::Loose { basis_gates, .. } => {
                basis_gates.iter().copied().all(valid_clifford_t)
            }
            QpuConstraint::Target(target) => target
                .operation_names_for_qargs(&[qubit])
                .map(|gates| gates.into_iter().all(valid_clifford_t))
                .unwrap_or(false),
        };
        let init_solovay_kitaev = || {
            SolovayKitaevSynthesis::new(
                &[StandardGate::T, StandardGate::Tdg, StandardGate::H],
                12,
                None,
                false,
            )
            .expect("hardcoded basis should be valid for SK decomposition")
        };
        let qubit = match constraint {
            // We always use the same (dummy) qubit for loose constraints, since loose constraints
            // are the same for all qubits.
            QpuConstraint::Loose { .. } => PhysicalQubit::MAX,
            QpuConstraint::Target(_) => qubit,
        };
        self.permits_solovay_kitaev
            .entry(qubit)
            .or_insert_with(permits_solovay_kitaev)
            .then(|| &*self.solovay_kitaev.get_or_insert_with(init_solovay_kitaev))
    }

    /// An iterator over the available 2q decomposers on the given link.
    ///
    /// This populates the cache if it is not already set.
    pub fn get_2q(
        &mut self,
        qubits: [PhysicalQubit; 2],
        config: &UnitarySynthesisConfig,
        constraint: QpuConstraint,
    ) -> PyResult<impl ExactSizeIterator<Item = (&Decomposer2q, FlipDirection)>> {
        // We can't use `Entry::or_insert_with` because our creator function is fallible and we
        // might have to propagate its error.
        let entry = match self.decomposers_2q.entry(qubits) {
            hash_map::Entry::Occupied(entry) => entry,
            hash_map::Entry::Vacant(entry) => entry.insert_entry(get_2q_decomposers(
                &mut self.decompose_2q_cache,
                qubits,
                config,
                constraint,
            )?),
        };
        Ok(entry.into_mut().iter().map(|(index, flip)| {
            (
                self.decompose_2q_cache
                    .get(*index)
                    .expect("indices should only be stored if construction succeeded"),
                *flip,
            )
        }))
    }
}

/// Get the [EulerBasisSet] denoting valid 1q decompositions for a given qubit in the target.
fn euler_bases_from_target(target: &Target, qubit: PhysicalQubit) -> EulerBasisSet {
    match target.operation_names_for_qargs(&[qubit]) {
        Ok(gates) => EulerBasisSet::from_support(|gate| gates.contains(gate)),
        Err(_) => EulerBasisSet::from_support(|_| true),
    }
}

/// Calculate the 2q decomposers available on a given link, as references into the cache.
///
/// This updates the cache with any newly discovered decomposers.
fn get_2q_decomposers(
    cache: &mut Decomposer2qCacheInner,
    qubits: [PhysicalQubit; 2],
    config: &UnitarySynthesisConfig,
    constraint: QpuConstraint,
) -> PyResult<Vec<(usize, FlipDirection)>> {
    let choose_flip =
        |direction: AllowedDirection2q, constructor: &Decomposer2qConstructor| -> FlipDirection {
            match direction {
                AllowedDirection2q::Forwards => {
                    if constructor.may_return_backwards_sequence() {
                        FlipDirection::Ensure(Direction2q::Forwards)
                    } else {
                        FlipDirection::No
                    }
                }
                AllowedDirection2q::Backwards => {
                    if constructor.may_return_backwards_sequence() {
                        FlipDirection::Ensure(Direction2q::Backwards)
                    } else {
                        FlipDirection::Yes
                    }
                }
                AllowedDirection2q::Both => FlipDirection::No,
            }
        };
    match constraint {
        QpuConstraint::Loose {
            basis_gates,
            coupling,
        } => {
            let direction = match (
                coupling.contains(&qubits),
                coupling.contains(&[qubits[1], qubits[0]]),
            ) {
                (true, false) => AllowedDirection2q::Forwards,
                (false, true) => AllowedDirection2q::Backwards,
                // We allow both in the case that we're being called for a gate that's not on a
                // supported link at all.
                _ => AllowedDirection2q::Both,
            };
            let direction = match config.decomposition_direction_2q {
                DecompositionDirection2q::Any => AllowedDirection2q::Both,
                DecompositionDirection2q::UniquelyBestValid
                    if direction == AllowedDirection2q::Both =>
                {
                    return Err(QiskitError::new_err(format!(
                        concat!(
                            "No preferred direction of gate on qubits {:?} ",
                            "could be determined from coupling map or gate lengths / gate errors."
                        ),
                        qubits
                    )));
                }
                DecompositionDirection2q::UniquelyBestValid
                | DecompositionDirection2q::BestValid => direction,
            };
            // With loose constraints, we've historically only ever used the first decomposer,
            // chosen from the priority list:
            //
            // 1. Controlled-U decomposer (continuous Ising gates)
            // 2. 2q-basis decomposer (static gates)
            //
            // We choose the first supported 1q and 2q basis from within those categories.

            let euler_bases = EulerBasisSet::from_support(|gate| basis_gates.contains(gate));
            let Some(euler) = euler_bases.get_bases().next() else {
                return Ok(vec![]);
            };
            if let Some(&ising_gate) = super::PARAM_SET_BASIS_GATES
                .iter()
                .find(|gate| basis_gates.contains(gate.name()))
            {
                let constructor =
                    Decomposer2qConstructor::ContinuousPauli(ContinuousPauliConstructor {
                        source: RXXEquivalent::Standard(ising_gate),
                        euler,
                    });
                let flip = choose_flip(direction, &constructor);
                if let Some(index) = cache.cache(constructor) {
                    return Ok(vec![(index, flip)]);
                }
            }
            if let Some(&kak_gate) = super::TWO_QUBIT_BASIS_SET_GATES
                .iter()
                .find(|gate| basis_gates.contains(gate.name()))
            {
                let source = StaticKakSource {
                    key: (kak_gate.name().to_owned(), constraint.kind()),
                    gate: kak_gate.into(),
                    params: smallvec![],
                };
                let fidelity = config.approximation.synthesis_fidelity(0.0).map_err(|e| {
                    PyValueError::new_err(format!(
                        "requested synthesis fidelity is out of range: {e}"
                    ))
                })?;
                let constructor = Decomposer2qConstructor::StaticKak(StaticKakConstructor {
                    source,
                    euler,
                    fidelity,
                    use_pulse_optimizer: config.use_pulse_optimizer,
                });
                let flip = choose_flip(direction, &constructor);
                if let Some(index) = cache.cache(constructor) {
                    return Ok(vec![(index, flip)]);
                }
            }
            Ok(Vec::new())
        }
        QpuConstraint::Target(target) => {
            // For the target, we prioritise in the order:
            //
            // 1. Controlled-U decomposer (continuous Ising gates)
            // 2. 2q-basis decomposer (static gates)
            // 3. XX decomposer (Ising, but discretised)
            //
            // and return as soon as we have a match.  However, within each category, we also try
            // every valid combination of decomposers that we can construct.

            // TODO: this doesn't handle the case of heterogeneous single-qubit operations, but at
            // the moment, the 2q decomposers don't know how to do this either, so we pick
            // arbitrarily and trust basis translation / 1q optimisation to fix it up later.
            let euler_bases = euler_bases_from_target(target, qubits[0]);
            let candidates_2q =
                get_candidate_2q_operations(target, qubits, config.decomposition_direction_2q);
            let mut decomposers = IndexSet::default();

            // Ising-like gates.
            for candidate in candidates_2q.iter() {
                // Only singly parametric gates can work here, and they have to be valid for a
                // continuous set of parameters.
                if !matches!(candidate.op.params_view(), &[Param::ParameterExpression(_)]) {
                    continue;
                }
                let rxx_equivalent = match candidate.op.operation.view() {
                    OperationRef::StandardGate(standard @ super::PARAM_SET!()) => {
                        RXXEquivalent::Standard(standard)
                    }
                    OperationRef::PyCustom(PyInstruction {
                        kind: PyOpKind::Gate,
                        ob,
                        ..
                    }) => Python::attach(|py| {
                        RXXEquivalent::CustomPython(ob.bind(py).get_type().unbind())
                    }),
                    _ => continue,
                };
                // TODO: the 2q decomposers internally already do everything that's needed to handle
                // _all_ of the 1q bases simultaneously without further decompositions, but don't
                // expose that functionality.  This wastes huge amounts of time and needs a fix.
                for euler in euler_bases.get_bases() {
                    let rxx_copy = match rxx_equivalent {
                        RXXEquivalent::Standard(gate) => RXXEquivalent::Standard(gate),
                        RXXEquivalent::CustomPython(ref gate) => {
                            RXXEquivalent::CustomPython(Python::attach(|py| gate.clone_ref(py)))
                        }
                    };
                    let constructor =
                        Decomposer2qConstructor::ContinuousPauli(ContinuousPauliConstructor {
                            source: rxx_copy,
                            euler,
                        });
                    let flip = choose_flip(candidate.direction, &constructor);
                    if let Some(index) = cache.cache(constructor) {
                        decomposers.insert((index, flip));
                    }
                }
            }
            if !decomposers.is_empty() {
                return Ok(decomposers.drain(..).collect());
            }

            // Static KAK gates
            for candidate in candidates_2q.iter() {
                if !is_supercontrolled(candidate.op) {
                    continue;
                }
                let params = candidate
                    .op
                    .params_view()
                    .iter()
                    .map(Param::try_float)
                    .collect::<Option<SmallVec<[f64; 3]>>>();
                let Some(params) = params else {
                    continue;
                };
                let fidelity = config
                    .approximation
                    .synthesis_fidelity(candidate.error)
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "requested synthesis fidelity is out of range: {e}"
                        ))
                    })?;
                // TODO: the 2q decomposers internally already do everything that's needed to handle
                // _all_ of the 1q bases simultaneously without further decompositions, but don't
                // expose that functionality.  This wastes huge amounts of time and needs a fix.
                for euler in euler_bases.get_bases() {
                    // This source is the same each time, but we'd be cloning it anyway.
                    let source = StaticKakSource {
                        key: (candidate.key.to_owned(), constraint.kind()),
                        gate: candidate.op.operation.clone(),
                        params: params.clone(),
                    };
                    let constructor = Decomposer2qConstructor::StaticKak(StaticKakConstructor {
                        source,
                        euler,
                        fidelity,
                        use_pulse_optimizer: config.use_pulse_optimizer,
                    });
                    let flip = choose_flip(candidate.direction, &constructor);
                    if let Some(index) = cache.cache(constructor) {
                        decomposers.insert((index, flip));
                    }
                }
            }
            if !decomposers.is_empty() || !config.run_python_decomposers {
                return Ok(decomposers.drain(..).collect());
            }

            // TODO: port XXDecomposer to Rust, so this catch can be removed.
            if config.run_python_decomposers {
                Python::attach(|py| {
                    get_xx_decomposers(py, cache, &euler_bases, &candidates_2q, config)
                })
                .map(|maybe| maybe.into_iter().collect())
            } else {
                Ok(Default::default())
            }
        }
    }
}

fn get_xx_decomposers(
    py: Python,
    cache: &mut Decomposer2qCacheInner,
    euler_bases: &EulerBasisSet,
    candidates: &[Candidate2q],
    config: &UnitarySynthesisConfig,
) -> PyResult<Vec<(usize, FlipDirection)>> {
    // This just involves collecting all gates that the XXDecomposer can use as an embodiment (i.e.
    // anything locally equivalent to a particular angle of some 2q Pauli rotation) and storing
    // those.
    //
    // TODO: when XXDecomposer is in Rust, we'll be able to do all the work of producing the
    // decomposer matrix-free.  The Python version UnitarySynthesis using XXDecomposer already
    // implicit assumes (via lookups into the `XXEmbodiments` static dictionary) that the only valid
    // embodiments are formed of standard gates.  In that case, we can simply match the
    // `StandardGate` into a a known strength (or function of strength).
    let embodiments_lookup = imports::XX_EMBODIMENTS.get_bound(py).cast::<PyDict>()?;
    let xx_decomposer_class = imports::XX_DECOMPOSER.get_bound(py);
    let approximation = config.approximation;

    let mut extend_with_flip = |out: &mut Vec<(usize, FlipDirection)>,
                                candidates: &[Candidate2q],
                                flip: FlipDirection|
     -> PyResult<()> {
        let fidelities = PyDict::new(py);
        let embodiments = PyDict::new(py);
        for candidate in candidates {
            let Some(strength) = rxx_equivalent_strength(candidate.op) else {
                continue;
            };
            let op_type = candidate
                .op
                .into_pyobject(py)?
                .getattr(intern!(py, "base_class"))?;
            let Some(embodiment) = embodiments_lookup.get_item(op_type)? else {
                continue;
            };
            let embodiment = if embodiment
                .getattr(intern!(py, "num_parameters"))?
                .is_truthy()?
            {
                embodiment.call_method1(intern!(py, "assign_parameters"), (vec![strength],))?
            } else {
                embodiment
            };
            embodiments.set_item(strength, embodiment)?;
            fidelities.set_item(
                strength,
                approximation
                    .synthesis_fidelity(candidate.error)
                    .map_err(|e| {
                        PyValueError::new_err(format!(
                            "requested synthesis fidelity is out of range: {e}"
                        ))
                    })?
                    .get(),
            )?;
        }
        if !fidelities.is_truthy()? {
            return Ok(());
        }
        for basis in euler_bases.get_bases() {
            let ob = xx_decomposer_class.call1((
                fidelities.clone(),
                basis.as_str(),
                embodiments.clone(),
            ))?;
            let constructor = Decomposer2qConstructor::DiscretePauli(DiscretePauliConstructor {
                decomposer: ob.unbind(),
                is_approximate: approximation != Approximation::Exact,
            });
            if let Some(index) = cache.cache(constructor) {
                out.push((index, flip));
            }
        }
        Ok(())
    };

    let mut out = Vec::new();
    if candidates
        .iter()
        .any(|candidate| candidate.direction == AllowedDirection2q::Backwards)
    {
        let forwards = candidates
            .iter()
            .filter(|candidate| candidate.direction != AllowedDirection2q::Backwards)
            .copied()
            .collect::<Vec<_>>();
        let backwards = candidates
            .iter()
            .filter(|candidate| candidate.direction != AllowedDirection2q::Forwards)
            .copied()
            .collect::<Vec<_>>();
        extend_with_flip(&mut out, &forwards, FlipDirection::No)?;
        extend_with_flip(&mut out, &backwards, FlipDirection::Yes)?;
    } else {
        extend_with_flip(&mut out, candidates, FlipDirection::No)?;
    }
    Ok(out)
}

/// The direction(s) that a 2q link is allowed to be used in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum AllowedDirection2q {
    Forwards,
    Backwards,
    Both,
}
/// A concrete 2q direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Direction2q {
    Forwards,
    Backwards,
}
impl Direction2q {
    pub fn as_indices(&self) -> [u8; 2] {
        match self {
            Self::Forwards => [0, 1],
            Self::Backwards => [1, 0],
        }
    }
}

/// How a 2q gate's direction should be handled during decomposition.
///
/// Most of the decomposers will always decompose a matrix into a series of 1q operations with the
/// 2q gate always coming out in the `[qargs[0], qargs[1]]` order.  This order might not be the
/// optimal direction for hardware, however; there might be a lower error rate on the `[qargs[1],
/// qargs[0]]` link.  Worse, currently some decomposers (the only known one in
/// `TwoQubitBasisDecomposer` when `pulse_optimize` is available) will synthesise in a direction
/// that is not knowable ahead of time.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FlipDirection {
    /// The gate direction is guaranteed to come out the best way round.
    No,
    /// The gate direction is guaranteed to come out the worse way (or an invalid way) round, so the
    /// gate should be synthesised conjugated by swaps and the qargs reversed.
    Yes,
    // TODO: really, the decomposer should not be inconsistent, and it should be rewritten so the
    // output doesn't need inspecting.
    /// The decomposer is inconsistent and might return its 2q gate in (e.g.) the `[1, 0]` order
    /// even though `[0, 1]` is intended (or vice versa).  The synthesis should be retried
    /// conjugated by swaps.
    Ensure(Direction2q),
}

/// A candidate 2q gate stored in a [Target].
///
/// This is just a simple record type for use in [get_2q_decomposers].
#[derive(Clone, Copy, Debug)]
struct Candidate2q<'a> {
    /// The lookup key for the operation.  This might not be the same as `op.name()`.
    key: &'a str,
    op: &'a NormalOperation,
    // TODO: there's a chance for quality improvements in the algorithm here. If the link has
    // asymmetric errors, we might still want to use the higher-error direction if the
    // swap-conjugated matrix decomposes to fewer 2q operations.
    /// Which direction the gate is (preferentially) valid in.  It can only be `Both` if the error
    /// rate is the same in both directions.
    direction: AllowedDirection2q,
    error: f64,
}

/// Get the 2q candidate operations for a given pair of qubits.
///
/// There is no filtering relative to decomposers here (e.g. we don't check if the operation is
/// usable by any known decomposers), we only evaluate which 2q operations are available, and which
/// directions they're valid for use in.
fn get_candidate_2q_operations(
    target: &Target,
    qubits: [PhysicalQubit; 2],
    decomposition_direction_2q: DecompositionDirection2q,
) -> Vec<Candidate2q<'_>> {
    let rev_qubits = [qubits[1], qubits[0]];
    let mut candidates = Vec::new();
    let forwards_names = target
        .operation_names_for_qargs(&qubits)
        .unwrap_or_default();
    let mut reverse_names = target
        .operation_names_for_qargs(&rev_qubits)
        .unwrap_or_default();
    for name in forwards_names {
        let Some(TargetOperation::Normal(op)) = target.operation_from_name(name) else {
            continue;
        };
        if !op.operation.is_gate() || op.operation.num_qubits() != 2 {
            continue;
        }
        let error_fwd = target.get_error(name, &qubits).unwrap_or(0.);
        let (directions, error) = if reverse_names.remove(name) {
            let error_rev = target.get_error(name, &rev_qubits).unwrap_or(0.);
            // TODO: the historical behaviour of this path is to choose a direction based on the
            // gate _duration_ instead of the error.  This probably wants revisiting.
            let durations = (decomposition_direction_2q != DecompositionDirection2q::Any)
                .then(|| {
                    target
                        .get_duration(name, &qubits)
                        .zip(target.get_duration(name, &rev_qubits))
                })
                .flatten();
            let cmp = if let Some((forwards, backwards)) = durations {
                forwards.partial_cmp(&backwards)
            } else {
                error_fwd.partial_cmp(&error_rev)
            };
            match cmp {
                None => (AllowedDirection2q::Forwards, error_fwd),
                Some(Ordering::Less) => (AllowedDirection2q::Forwards, error_fwd),
                Some(Ordering::Equal) => (AllowedDirection2q::Both, error_fwd),
                Some(Ordering::Greater) => (AllowedDirection2q::Backwards, error_rev),
            }
        } else {
            (AllowedDirection2q::Forwards, error_fwd)
        };
        candidates.push(Candidate2q {
            key: name,
            op,
            direction: directions,
            error,
        });
    }
    for name in reverse_names {
        let Some(TargetOperation::Normal(op)) = target.operation_from_name(name) else {
            continue;
        };
        if !op.operation.is_gate() || op.operation.num_qubits() != 2 {
            continue;
        }
        let rev_error = target.get_error(name, &rev_qubits).unwrap_or(0.);
        if rev_error.is_nan() {
            continue;
        }
        candidates.push(Candidate2q {
            key: name,
            op,
            direction: AllowedDirection2q::Backwards,
            error: rev_error,
        });
    }
    candidates
}

#[inline]
fn is_supercontrolled(op: &NormalOperation) -> bool {
    if let Some(gate) = op.operation.try_standard_gate() {
        // If it's a zero-param standard gate, we already know whether it's valid without
        // needing to check the matrix.
        if gate.num_params() == 0 {
            return super::TWO_QUBIT_BASIS_SET_GATES.contains(&gate);
        }
        // ... if it has fixed parameters, it might still be valid.
    }
    op.try_matrix().is_some_and(|unitary| {
        let kak = TwoQubitWeylDecomposition::new_inner(unitary.view(), None, None).unwrap();
        relative_eq!(kak.a(), FRAC_PI_4) && relative_eq!(kak.c(), 0.0)
    })
}

/// The Rxx angle, if any, that the given static operation is locally equivalent to.
#[inline]
fn rxx_equivalent_strength(op: &NormalOperation) -> Option<f64> {
    op.try_matrix().and_then(|unitary| {
        let kak = TwoQubitWeylDecomposition::new_inner(unitary.view(), None, None).unwrap();
        (relative_eq!(kak.b(), 0.0) && relative_eq!(kak.c(), 0.0)).then(|| 2.0 * kak.a())
    })
}

/// Helper for the Python-space `XXDecomposer` to convert its output format into the native one used
/// by the rest of unitary synthesis.
fn circuit_to_2q_sequence(circuit: &CircuitData) -> PyResult<TwoQubitGateSequence> {
    assert!(circuit.num_qubits() <= 2);
    let global_phase = circuit
        .global_phase()
        .try_float()
        .ok_or_else(|| PyTypeError::new_err("unexpected global phase"))?;
    let gates = circuit
        .data()
        .iter()
        .map(|inst| {
            let params = inst
                .params_view()
                .iter()
                .map(|p| p.try_float())
                .collect::<Option<_>>()
                .ok_or_else(|| PyTypeError::new_err("unexpected parameter"))?;
            let qubits = circuit
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| q.index() as u8)
                .collect();
            Ok((inst.op.clone(), params, qubits))
        })
        .collect::<PyResult<_>>()?;
    Ok(TwoQubitGateSequence::from_sequence(gates, global_phase))
}
