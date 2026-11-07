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

//! Annotations for randomization.

use std::sync::Arc;

use thiserror::Error;

use qiskit_circuit::annotation::Annotation;

use super::{DistributionType, SynthesizerType};

/// The namespace for randomization annotations.
pub(crate) const NAMESPACE: &str = "randomization";

pub type Result<T> = std::result::Result<T, InvalidAnnotationError>;

/// Errors from invalid annotation combinations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum InvalidAnnotationError {
    #[error("duplicate randomization annotation on a single box")]
    DuplicateAnnotation,
    #[error("InjectNoise requires a Twirl on the same box")]
    InjectNoiseWithoutTwirl,
}

/// Which side of a box its easy gates get composed into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dressing {
    Left,
    Right,
}

/// Whether noise is injected before or after the hard content of a box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionSite {
    Before,
    After,
}

/// Which randomization annotation this is, or `None` if it is not.
pub fn annotation_kind(annotation: &dyn Annotation) -> Option<RandomizationAnnotation<'_>> {
    if annotation.namespace() != NAMESPACE {
        return None;
    }
    if let Some(twirl) = annotation.downcast_ref::<Twirl>() {
        Some(RandomizationAnnotation::Twirl(twirl))
    } else {
        annotation
            .downcast_ref::<InjectNoise>()
            .map(RandomizationAnnotation::InjectNoise)
    }
}

/// A randomization annotation borrowed from the box carrying it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomizationAnnotation<'a> {
    Twirl(&'a Twirl),
    InjectNoise(&'a InjectNoise),
}

/// Randomize this box by conjugating it with gates sampled from a distribution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Twirl {
    pub distribution: DistributionType,
    pub dressing: Dressing,
    pub decomposition: SynthesizerType,
}

impl Annotation for Twirl {
    fn namespace(&self) -> &str {
        NAMESPACE
    }
}

/// Inject Pauli-Lindblad noise into this box.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectNoise {
    pub reference: String,
    pub modifier: Option<String>,
    pub site: InjectionSite,
}

impl Annotation for InjectNoise {
    fn namespace(&self) -> &str {
        NAMESPACE
    }
}

/// Randomization annotations on a box.
///
/// Borrowed from the box carrying them: these say what a box declares, which a pass reads while
/// lowering it and nothing holds afterwards.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RandomizationAnnotations<'a> {
    pub twirl: Option<&'a Twirl>,
    pub inject_noise: Option<&'a InjectNoise>,
}

impl<'a> RandomizationAnnotations<'a> {
    /// Return validated annotations.
    pub fn new(annotations: &'a [Arc<dyn Annotation>]) -> Result<Self> {
        let mut resolved = Self::default();
        for annotation in annotations {
            let Some(kind) = annotation_kind(annotation.as_ref()) else {
                continue;
            };
            let already_set = match kind {
                RandomizationAnnotation::Twirl(twirl) => resolved.twirl.replace(twirl).is_some(),
                RandomizationAnnotation::InjectNoise(injection) => {
                    resolved.inject_noise.replace(injection).is_some()
                }
            };
            if already_set {
                return Err(InvalidAnnotationError::DuplicateAnnotation);
            }
        }

        if resolved.inject_noise.is_some() && resolved.twirl.is_none() {
            return Err(InvalidAnnotationError::InjectNoiseWithoutTwirl);
        }
        Ok(resolved)
    }

    /// Whether this box produces virtual gates.
    pub fn emits(&self) -> bool {
        self.twirl.is_some()
    }

    /// Which side this box's easy gates compose into.
    pub fn dressing(&self) -> Option<Dressing> {
        self.twirl.map(|twirl| twirl.dressing)
    }

    /// How this box synthesizes virtual gates.
    pub fn synthesizer(&self) -> Option<SynthesizerType> {
        self.twirl.map(|twirl| twirl.decomposition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn erased(annotation: impl Annotation + 'static) -> Arc<dyn Annotation> {
        Arc::new(annotation)
    }

    fn twirl(distribution: DistributionType) -> Arc<dyn Annotation> {
        erased(Twirl {
            distribution,
            dressing: Dressing::Left,
            decomposition: SynthesizerType::RzSx,
        })
    }

    fn noise(site: InjectionSite) -> Arc<dyn Annotation> {
        erased(InjectNoise {
            reference: "layer_1".to_string(),
            modifier: None,
            site,
        })
    }

    /// An annotation from outside this vocabulary.
    #[derive(Debug, Clone, PartialEq)]
    struct Foreign;

    impl Annotation for Foreign {
        fn namespace(&self) -> &str {
            "someone.else"
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Impostor;

    impl Annotation for Impostor {
        fn namespace(&self) -> &str {
            NAMESPACE
        }
    }

    #[test]
    fn test_a_twirl_resolves_to_a_dressing_and_a_synthesizer() {
        let annotations = [twirl(DistributionType::UniformPauli)];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert!(resolved.emits());
        assert_eq!(resolved.dressing(), Some(Dressing::Left));
        assert_eq!(resolved.synthesizer(), Some(SynthesizerType::RzSx));
        assert!(resolved.inject_noise.is_none());
    }

    #[test]
    fn test_a_box_with_foreign_annotations_on_it_resolves_to_nothing() {
        let annotations = [erased(Foreign)];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert!(!resolved.emits());
        assert_eq!(resolved.dressing(), None);
        assert_eq!(resolved.synthesizer(), None);
        assert_eq!(resolved, RandomizationAnnotations::default());
    }

    #[test]
    fn test_an_unannotated_box_resolves_to_nothing() {
        assert_eq!(
            RandomizationAnnotations::new(&[]).unwrap(),
            RandomizationAnnotations::default()
        );
    }

    #[test]
    fn test_noise_resolves_beside_a_twirl() {
        let annotations = [
            twirl(DistributionType::UniformPauli),
            noise(InjectionSite::Before),
        ];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert!(resolved.twirl.is_some());
        let injection = resolved.inject_noise.expect("the injection resolves");
        assert_eq!(injection.reference, "layer_1");
        assert_eq!(injection.site, InjectionSite::Before);
        assert_eq!(injection.modifier, None);
    }

    #[test]
    fn test_noise_without_a_twirl_is_refused() {
        assert_eq!(
            RandomizationAnnotations::new(&[noise(InjectionSite::After)]).unwrap_err(),
            InvalidAnnotationError::InjectNoiseWithoutTwirl
        );
    }

    #[test]
    fn test_two_of_one_kind_on_a_box_is_refused() {
        assert_eq!(
            RandomizationAnnotations::new(&[
                twirl(DistributionType::UniformPauli),
                twirl(DistributionType::HaarU2),
            ])
            .unwrap_err(),
            InvalidAnnotationError::DuplicateAnnotation
        );
    }

    #[test]
    fn test_claiming_the_namespace_is_not_enough() {
        let annotations = [erased(Impostor), twirl(DistributionType::UniformPauli)];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert_eq!(
            resolved.twirl.as_ref().map(|t| t.distribution),
            Some(DistributionType::UniformPauli)
        );
        assert!(
            RandomizationAnnotations::new(&[erased(Impostor), erased(Impostor)])
                .unwrap()
                .twirl
                .is_none()
        );
    }

    #[test]
    fn test_a_foreign_annotation_rides_through() {
        let annotations = [
            erased(Foreign),
            twirl(DistributionType::UniformPauli),
            erased(Foreign),
        ];
        let resolved = RandomizationAnnotations::new(&annotations).unwrap();
        assert!(resolved.emits());
    }

    #[test]
    fn test_the_annotations_share_one_flat_namespace() {
        for annotation in [
            twirl(DistributionType::UniformPauli),
            noise(InjectionSite::Before),
        ] {
            assert_eq!(annotation.namespace(), NAMESPACE);
        }
    }
}
