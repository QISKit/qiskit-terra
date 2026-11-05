// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::sync::Arc;

use qiskit_circuit::annotation::{
    Annotation, AnnotationFromPython, PythonAnnotation, iter_namespaces,
};

use hashbrown::HashMap;

use crate::bytes::Bytes;
use crate::error::QpyError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

/// A deserializer for an [Annotation].
///
/// This function takes a namespace and a payload, and returns an annotation.
pub type NativeDeserializer = fn(&str, &str) -> Option<Arc<dyn Annotation>>;

/// Deserializers for annotations.
///
/// This structure contains a bank of deserializers keyed by namespace.
#[derive(Debug, Default, Clone)]
pub struct NativeDeserializers(HashMap<String, NativeDeserializer>);

impl NativeDeserializers {
    /// Insert a native deserializer for the given namespace.
    ///
    /// This is the main entrypoint for consumers.
    #[allow(dead_code)]
    pub fn insert(&mut self, namespace: &str, deserializer: NativeDeserializer) {
        self.0.insert(namespace.to_string(), deserializer);
    }

    /// Load an annotation from a payload.
    ///
    /// This method uses [iter_namespaces] to find the narrowest namespace contained in this deserializer
    /// that matches the namespace of a given payload, then uses the corresponding [NativeDeserializer].
    pub fn load(&self, namespace: &str, payload: &str) -> Option<Arc<dyn Annotation>> {
        if let Some(deserializer) = iter_namespaces(namespace).find_map(|ns| self.0.get(ns)) {
            deserializer(namespace, payload)
        } else {
            None
        }
    }
}

/// A serializer for an [Annotation].
///
/// This function takes an annotation and returns a payload.
pub type NativeSerializer = fn(annotation: &Arc<dyn Annotation>) -> Option<String>;

/// Serializers for annotations.
///
/// This structure contains a bank of serializers keyed by namespace.
#[derive(Debug, Default, Clone)]
pub struct NativeSerializers(HashMap<String, NativeSerializer>);

impl NativeSerializers {
    /// Insert a native serializer for the given namespace.
    ///
    /// This is the main entrypoint for consumers.
    #[allow(dead_code)]
    pub fn insert(&mut self, namespace: &str, serializer: NativeSerializer) {
        self.0.insert(namespace.to_string(), serializer);
    }

    /// Dump an annotation into a payload.
    ///
    /// This method uses [iter_namespaces] to find the narrowest namespace contained in this serializer
    /// that matches the namespace of a given payload, then uses the corresponding [NativeSerializer].
    pub fn dump(&self, namespace: &str, annotation: &Arc<dyn Annotation>) -> Option<String> {
        if let Some(serializer) = iter_namespaces(namespace).find_map(|ns| self.0.get(ns)) {
            serializer(annotation)
        } else {
            None
        }
    }
}

/// Handles QPY annotations at the boundary appropriate to the caller.
///
/// Python QPY entry points delegate annotation handling to the existing Python state objects.
/// Native callers do not initialize any Python annotation machinery and fail only if the payload
/// actually requires annotation handling.
#[derive(Debug)]
pub enum AnnotationHandler {
    Python {
        factories: Py<PyDict>,
        serialization_state: Py<PyAny>,
        deserialization_state: Py<PyAny>,
    },
    Native {
        namespaces: Vec<String>,
        serializers: NativeSerializers,
        deserializers: NativeDeserializers,
    },
}

impl AnnotationHandler {
    pub fn python(factories: &Py<PyDict>) -> Result<Self, QpyError> {
        Python::attach(|py| {
            let module = py.import("qiskit.qpy.binary_io.circuits")?;
            let serialization_state = module
                .getattr("_AnnotationSerializationState")?
                .call1((factories.bind(py),))?
                .unbind();
            let deserialization_state = module
                .getattr("_AnnotationDeserializationState")?
                .call1((factories.bind(py),))?
                .unbind();
            Ok(Self::Python {
                factories: factories.clone_ref(py),
                serialization_state,
                deserialization_state,
            })
        })
    }

    pub fn native(
        namespaces: Vec<String>,
        serializers: NativeSerializers,
        deserializers: NativeDeserializers,
    ) -> Self {
        Self::Native {
            namespaces,
            serializers,
            deserializers,
        }
    }

    /// Create independent annotation state for a nested circuit while preserving the caller mode.
    pub fn child(&self) -> Result<Self, QpyError> {
        match self {
            Self::Python { factories, .. } => Self::python(factories),
            Self::Native {
                serializers,
                deserializers,
                ..
            } => Ok(Self::native(
                Vec::new(),
                serializers.clone(),
                deserializers.clone(),
            )),
        }
    }

    pub fn serialize(
        &mut self,
        annotation: &Arc<dyn Annotation>,
    ) -> Result<(u32, Bytes), QpyError> {
        match self {
            Self::Python {
                serialization_state,
                ..
            } => Python::attach(|py| {
                let Some(ob) = annotation.downcast_ref::<PythonAnnotation>() else {
                    return Err(QpyError::AnnotationError(
                        "Rust native annotations cannot be serialized by the Python QPY path"
                            .to_owned(),
                    ));
                };
                Ok(serialization_state
                    .call_method1(py, "serialize", (ob.annotation_obj().clone_ref(py),))?
                    .extract(py)?)
            }),
            Self::Native {
                namespaces,
                serializers,
                ..
            } => {
                let ns = annotation.namespace();
                let index = if let Some(i) = namespaces.iter().position(|a| a == ns) {
                    i
                } else {
                    namespaces.push(annotation.namespace().to_string());
                    namespaces.len() - 1
                };
                let Some(payload) = serializers.dump(ns, annotation) else {
                    return Err(QpyError::AnnotationError(format!(
                        "Could not find an appropriate serializer for namespace {ns}."
                    )));
                };
                Ok((index as u32, format!("{ns}\x00{payload}").into()))
            }
        }
    }

    pub fn load_py(&self, py: Python, index: u32, payload: Bytes) -> Result<Py<PyAny>, QpyError> {
        match self {
            Self::Python {
                deserialization_state,
                ..
            } => Ok(deserialization_state.call_method1(py, "load", (index, payload))?),
            Self::Native { .. } => Err(Self::native_error("deserialize")),
        }
    }

    pub fn load(&self, index: u32, payload: Bytes) -> Result<Arc<dyn Annotation>, QpyError> {
        match self {
            Self::Python { .. } => Python::attach(|py| {
                Ok(self
                    .load_py(py, index, payload)?
                    .bind(py)
                    .extract::<AnnotationFromPython>()?
                    .into_annotation())
            }),
            Self::Native { deserializers, .. } => {
                let text: &str = (&payload).try_into()?;
                let (ns, payload) = text.split_once("\x00").ok_or_else(|| {
                    QpyError::AnnotationError("Incorrectly formatted payload.".to_owned())
                })?;
                deserializers.load(ns, payload).ok_or_else(|| {
                    QpyError::AnnotationError(format!("Could not find a deserializer for {ns}."))
                })
            }
        }
    }

    pub fn dump_serializers(&self) -> Result<Vec<(String, Bytes)>, QpyError> {
        match self {
            Self::Python {
                serialization_state,
                ..
            } => Python::attach(|py| {
                Ok(serialization_state
                    .call_method0(py, "dump_states")?
                    .extract(py)?)
            }),
            Self::Native { namespaces, .. } => Ok(namespaces
                .iter()
                .map(|ns| (ns.clone(), Bytes::new()))
                .collect()),
        }
    }

    pub fn load_deserializers(&mut self, data: Vec<(String, Bytes)>) -> Result<(), QpyError> {
        if data.is_empty() {
            return Ok(());
        }
        match self {
            Self::Python {
                deserialization_state,
                ..
            } => Python::attach(|py| {
                for (ns, state) in data {
                    deserialization_state.call_method1(py, "initialize", (ns, state))?;
                }
                Ok(())
            }),
            Self::Native { namespaces, .. } => {
                *namespaces = data.into_iter().map(|(ns, _)| ns).collect();
                Ok(())
            }
        }
    }

    fn native_error(action: &str) -> QpyError {
        QpyError::AnnotationError(format!(
            "native QPY cannot {action} circuits containing annotations"
        ))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod test_annotation_loading {
    use crate::annotations::{NativeDeserializers, NativeSerializers};
    use qiskit_circuit::annotation::Annotation;
    use std::sync::Arc;

    #[derive(Debug, PartialEq)]
    struct Twirl {
        twirl: String,
    }

    impl Twirl {
        pub fn from_payload(payload: &str) -> Option<Self> {
            let (_, twirl) = payload.rsplit_once("twirl:")?;
            Some(Twirl {
                twirl: twirl.to_string(),
            })
        }
    }

    impl Annotation for Twirl {
        fn namespace(&self) -> &str {
            "randomization.twirl"
        }
    }

    #[derive(Debug, PartialEq)]
    struct InjectNoise(String);

    impl InjectNoise {
        pub fn from_payload(payload: &str) -> Option<Self> {
            let (_, reference) = payload.rsplit_once("ref:")?;
            Some(InjectNoise(reference.to_string()))
        }
    }

    impl Annotation for InjectNoise {
        fn namespace(&self) -> &str {
            "randomization.inject_noise"
        }
    }

    #[test]
    fn test_native_deserializers() {
        let mut deserializers = NativeDeserializers::default();

        // A deserializer than handles the randomization namespace and returns new instances with their corresponding payloads.
        deserializers.insert("randomization", |ns, payload| match ns.rsplit_once(".") {
            Some((_, ns)) => match ns {
                "twirl" => Some(Arc::new(Twirl::from_payload(payload)?)),
                "inject_noise" => Some(Arc::new(InjectNoise::from_payload(payload)?)),
                _ => None,
            },
            None => None,
        });

        // A deserializer with a narrower namespace that returns an inject noise with a fixed payload.
        deserializers.insert("randomization.inject_noise", |_, _| {
            Some(Arc::new(InjectNoise("different".to_string())))
        });

        let mut serializers = NativeSerializers::default();
        serializers.insert("randomization", |ann: &Arc<dyn Annotation>| {
            match ann.namespace().rsplit_once(".") {
                Some((_, annotation_type)) => match annotation_type {
                    "twirl" => {
                        let twirl = ann.downcast_ref::<Twirl>()?;
                        Some(format!("twirl:{0}", twirl.twirl))
                    }
                    "inject_noise" => {
                        let inject_noise = ann.downcast_ref::<InjectNoise>()?;
                        Some(format!("ref:{0}", inject_noise.0))
                    }
                    _ => None,
                },
                None => None,
            }
        });

        let annotation: Arc<dyn Annotation> = Arc::new(Twirl {
            twirl: "pauli".to_string(),
        });
        let twirl_payload = serializers
            .dump("randomization.twirl", &annotation)
            .unwrap();
        assert_eq!(twirl_payload, "twirl:pauli");

        let roundtrip = deserializers
            .load("randomization.twirl", &twirl_payload)
            .unwrap();
        assert_eq!(roundtrip.as_ref(), annotation.as_ref());

        let annotation: Arc<dyn Annotation> = Arc::new(InjectNoise("ok".to_string()));
        let inject_noise_payload = serializers
            .dump("randomization.inject_noise", &annotation)
            .unwrap();
        assert_eq!(inject_noise_payload, "ref:ok");

        let roundtrip = deserializers
            .load("randomization.inject_noise", &inject_noise_payload)
            .unwrap();
        let expected: Arc<dyn Annotation> = Arc::new(InjectNoise("different".to_string()));
        assert_ne!(roundtrip.as_ref(), annotation.as_ref());
        assert_eq!(roundtrip.as_ref(), expected.as_ref());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{assert_eq, assert_ne};

    #[derive(Debug, Clone, PartialEq)]
    struct Tag(&'static str);

    impl Annotation for Tag {
        fn namespace(&self) -> &str {
            "tag"
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct Mark;

    impl Annotation for Mark {
        fn namespace(&self) -> &str {
            "mark"
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct NonSerializable;

    impl Annotation for NonSerializable {
        fn namespace(&self) -> &str {
            "no_payload"
        }
    }

    #[test]
    fn native_handler_is_inert_without_annotations() {
        assert!(
            AnnotationHandler::native(
                Vec::new(),
                NativeSerializers::default(),
                NativeDeserializers::default()
            )
            .dump_serializers()
            .is_ok_and(|states| states.is_empty())
        );
        assert!(
            AnnotationHandler::native(
                Vec::new(),
                NativeSerializers::default(),
                NativeDeserializers::default()
            )
            .load_deserializers(Vec::new())
            .is_ok()
        );
    }

    #[test]
    fn test_native_serialize() -> Result<(), Box<dyn std::error::Error>> {
        let annotation: Arc<dyn Annotation> = Arc::new(Tag("my_tag"));
        let other_annotation: Arc<dyn Annotation> = Arc::new(Tag("my_other_tag"));
        let mark: Arc<dyn Annotation> = Arc::new(Mark);

        let mut serializers = NativeSerializers::default();
        serializers.insert("mark", |ann: &Arc<dyn Annotation>| match ann.namespace() {
            "mark" => Some("mark".to_string()),
            _ => None,
        });
        serializers.insert("tag", |ann: &Arc<dyn Annotation>| match ann.namespace() {
            "tag" => {
                let tag = ann.downcast_ref::<Tag>()?;
                Some(tag.0.to_string())
            }
            _ => None,
        });
        let mut handler =
            AnnotationHandler::native(Vec::new(), serializers, NativeDeserializers::default());

        let (idx, payload) = handler.serialize(&annotation)?;
        let (other_idx, other_payload) = handler.serialize(&other_annotation)?;
        let (mark_idx, mark_payload) = handler.serialize(&mark)?;

        assert_eq!(TryInto::<&str>::try_into(&payload)?, "tag\x00my_tag");
        assert_eq!(
            TryInto::<&str>::try_into(&other_payload)?,
            "tag\x00my_other_tag"
        );
        assert_eq!(TryInto::<&str>::try_into(&mark_payload)?, "mark\x00mark");

        assert_eq!(idx, other_idx);
        assert_ne!(idx, mark_idx);

        Ok(())
    }

    #[test]
    fn test_native_serialize_error() {
        let annotation: Arc<dyn Annotation> = Arc::new(NonSerializable);
        let mut handler = AnnotationHandler::native(
            Vec::new(),
            NativeSerializers::default(),
            NativeDeserializers::default(),
        );

        assert!(matches!(
            handler.serialize(&annotation),
            Err(QpyError::AnnotationError(_))
        ));
    }

    #[test]
    fn test_native_dump_serializers() -> Result<(), Box<dyn std::error::Error>> {
        let handler = AnnotationHandler::native(
            vec![
                "randomization".to_string(),
                "randomization.twirl".to_string(),
            ],
            NativeSerializers::default(),
            NativeDeserializers::default(),
        );
        let deserializers = handler
            .dump_serializers()?
            .into_iter()
            .map(|(s, _)| s)
            .collect::<Vec<_>>();
        assert_eq!(
            deserializers,
            vec![
                "randomization".to_string(),
                "randomization.twirl".to_string()
            ]
        );
        Ok(())
    }

    #[test]
    fn test_native_load_deserializers() {
        let mut handler = AnnotationHandler::native(
            Vec::new(),
            NativeSerializers::default(),
            NativeDeserializers::default(),
        );
        assert!(
            handler
                .load_deserializers(vec![("a.namespace".to_string(), Bytes::new())])
                .is_ok()
        );
    }

    #[test]
    fn test_native_load_error() {
        let handler = AnnotationHandler::native(
            Vec::new(),
            NativeSerializers::default(),
            NativeDeserializers::default(),
        );
        assert!(matches!(
            handler.load(0, "bad_payload".into()),
            Err(QpyError::AnnotationError(_))
        ));
    }

    #[test]
    fn test_native_child() -> Result<(), Box<dyn std::error::Error>> {
        let handler = AnnotationHandler::native(
            vec!["some.namespace".to_string()],
            NativeSerializers::default(),
            NativeDeserializers::default(),
        )
        .child()?;
        assert!(
            handler
                .dump_serializers()
                .is_ok_and(|states| states.is_empty())
        );
        Ok(())
    }
}
