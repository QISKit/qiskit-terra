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

use binrw::{BinRead, BinResult, BinWrite, Endian, VecArgs};

use crate::error::QpyError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBytes};

use num_complex::Complex64;
use std::fmt::{Debug, Write as WriteFmt};
use std::io::{Cursor, Read, Seek, Write};
use std::ops::{Deref, DerefMut};

// Bytes are the format used to store serialized data which is not automatically handled by binrw
// It's a wrapper around Vec<u8> with extended serialization/deserialization capabilities
#[derive(Debug, Clone)]
pub struct Bytes(pub Vec<u8>);

impl Bytes {
    /// This method is used for debugging; it displays the data as a string of hexadecimal digits
    pub fn to_hex_string(&self) -> String {
        self.0
            .iter()
            .fold(String::with_capacity(self.0.len() * 2), |mut acc, b| {
                let _ = write!(&mut acc, "{:02x}", b);
                acc
            })
    }
    pub fn try_to_f64(&self, endian: Endian) -> Result<f64, QpyError> {
        let byte_array: [u8; 8] = self
            .0
            .as_slice()
            .try_into()
            .map_err(|_| QpyError::ConversionError("Expected exactly 8 bytes".to_string()))?;
        match endian {
            Endian::Big => Ok(f64::from_be_bytes(byte_array)),
            Endian::Little => Ok(f64::from_le_bytes(byte_array)),
        }
    }
    // pads with 0's in the front
    pub fn try_to_16_byte_slice(&self) -> Result<[u8; 16], QpyError> {
        if self.0.len() > 16 {
            Err(QpyError::ConversionError(
                "Cannot convert sequence of more than 16 bytes to slice of 16 bytes".to_string(),
            ))
        } else {
            let mut result = [0u8; 16];
            let start = 16 - self.0.len();
            result[start..].copy_from_slice(&self.0);
            Ok(result)
        }
    }
    pub fn new() -> Self {
        Bytes(Vec::new())
    }
}

impl Default for Bytes {
    fn default() -> Self {
        Self::new()
    }
}

// we also allow the case where Bytes has 16 bytes, the first 8 being 0
// since this is how f64 are encoded inside ParameterExpressionStandardOpPack
impl TryFrom<&Bytes> for f64 {
    type Error = QpyError;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let data = &bytes.0;
        match data.len() {
            8 => {
                let byte_array: [u8; 8] = data
                    .as_slice()
                    .try_into()
                    .map_err(|_| QpyError::ConversionError("Expected exactly 8 bytes".to_string()))?;
                Ok(f64::from_be_bytes(byte_array))
            }
            16 if data[..8].iter().all(|&byte| byte == 0) => {
                let byte_array: [u8; 8] = data[8..].try_into().map_err(|_| {
                    QpyError::ConversionError("Expected exactly 16 bytes with the first 8 being 0".to_string())
                })?;
                Ok(f64::from_be_bytes(byte_array))
            }
            _ => Err(QpyError::ConversionError(
                "Decoding Bytes to f64: Expected exactly 8 bytes or 16 bytes with the first 8 being 0".to_string(),
            )),
        }
    }
}

impl TryFrom<&Bytes> for (f64, f64) {
    type Error = QpyError;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let byte_array: [u8; 16] = bytes
            .0
            .as_slice()
            .try_into()
            .map_err(|_| QpyError::ConversionError("Expected exactly 8 bytes".to_string()))?;
        Ok((
            f64::from_be_bytes(byte_array[0..8].try_into().map_err(|_| {
                QpyError::ConversionError("Failed to convert bytes to f64".to_string())
            })?),
            f64::from_be_bytes(byte_array[8..16].try_into().map_err(|_| {
                QpyError::ConversionError("Failed to convert bytes to f64".to_string())
            })?),
        ))
    }
}

// we also allow the case where Bytes has 16 bytes, the first 8 being 0
// since this is how i64 are encoded inside ParameterExpressionStandardOpPack
impl TryFrom<&Bytes> for i64 {
    type Error = QpyError;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let data = &bytes.0;
        match data.len() {
            8 => {
                let byte_array: [u8; 8] = data
                    .as_slice()
                    .try_into()
                    .map_err(|_| QpyError::ConversionError("Expected exactly 8 bytes".to_string()))?;
                Ok(i64::from_be_bytes(byte_array))
            }
            16 if data[..8].iter().all(|&byte| byte == 0) => {
                let byte_array: [u8; 8] = data[8..].try_into().map_err(|_| {
                    QpyError::ConversionError("Expected exactly 16 bytes with the first 8 being 0".to_string())
                })?;
                Ok(i64::from_be_bytes(byte_array))
            }
            _ => Err(QpyError::ConversionError(
                "Decoding Bytes to i64: Expected exactly 8 bytes or 16 bytes with the first 8 being 0".to_string(),
            )),
        }
    }
}

impl TryFrom<&Bytes> for [u8; 16] {
    type Error = QpyError;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let byte_array: [u8; 16] = bytes
            .0
            .as_slice()
            .try_into()
            .map_err(|_| QpyError::ConversionError("Expected exactly 16 bytes".to_string()))?;
        Ok(byte_array)
    }
}

impl TryFrom<&Bytes> for Complex64 {
    type Error = QpyError;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        let byte_array: [u8; 16] = bytes
            .0
            .as_slice()
            .try_into()
            .map_err(|_| QpyError::ConversionError("Expected exactly 16 bytes".to_string()))?;
        let real = f64::from_be_bytes(byte_array[0..8].try_into().map_err(|_| {
            QpyError::ConversionError("Failed to convert bytes to f64".to_string())
        })?);
        let imag = f64::from_be_bytes(byte_array[8..16].try_into().map_err(|_| {
            QpyError::ConversionError("Failed to convert bytes to f64".to_string())
        })?);
        Ok(Complex64::new(real, imag))
    }
}

impl TryFrom<Bytes> for String {
    type Error = QpyError;
    fn try_from(bytes: Bytes) -> Result<Self, Self::Error> {
        String::from_utf8(bytes.0.clone())
            .map_err(|_| QpyError::ConversionError("Not a valid UTF-8 string".to_string()))
    }
}

impl<'a> TryFrom<&'a Bytes> for &'a str {
    type Error = QpyError;
    fn try_from(bytes: &'a Bytes) -> Result<Self, Self::Error> {
        std::str::from_utf8(&bytes.0)
            .map_err(|_| QpyError::ConversionError("Not a valid UTF-8 string".to_string()))
    }
}

impl TryFrom<&Bytes> for bool {
    type Error = QpyError;
    fn try_from(bytes: &Bytes) -> Result<Self, Self::Error> {
        match bytes.0.as_slice() {
            [0] => Ok(false),
            [1] => Ok(true),
            _ => Err(QpyError::ConversionError(
                "Not a boolean representation".to_string(),
            )),
        }
    }
}

impl From<Vec<u8>> for Bytes {
    fn from(v: Vec<u8>) -> Self {
        Bytes(v)
    }
}
impl From<&[u8]> for Bytes {
    fn from(s: &[u8]) -> Self {
        Bytes(s.to_vec())
    }
}
impl From<[u8; 8]> for Bytes {
    fn from(s: [u8; 8]) -> Self {
        Bytes(s.to_vec())
    }
}

impl From<[u8; 16]> for Bytes {
    fn from(s: [u8; 16]) -> Self {
        Bytes(s.to_vec())
    }
}

impl From<String> for Bytes {
    fn from(s: String) -> Self {
        Bytes(s.into_bytes())
    }
}

impl From<&String> for Bytes {
    fn from(s: &String) -> Self {
        Bytes(s.clone().into_bytes())
    }
}

impl From<&str> for Bytes {
    fn from(s: &str) -> Self {
        Bytes(s.to_string().into_bytes())
    }
}

impl From<Cursor<Vec<u8>>> for Bytes {
    fn from(cursor: Cursor<Vec<u8>>) -> Self {
        Bytes(cursor.into_inner())
    }
}

impl From<&f64> for Bytes {
    fn from(value: &f64) -> Self {
        Bytes(value.to_be_bytes().into())
    }
}

impl From<&i64> for Bytes {
    fn from(value: &i64) -> Self {
        Bytes(value.to_be_bytes().into())
    }
}

impl From<&bool> for Bytes {
    fn from(value: &bool) -> Self {
        Bytes(vec![*value as u8])
    }
}

impl From<&Complex64> for Bytes {
    fn from(value: &Complex64) -> Self {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&value.re.to_be_bytes());
        bytes[8..16].copy_from_slice(&value.im.to_be_bytes());
        bytes.into()
    }
}

impl Deref for Bytes {
    type Target = Vec<u8>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Bytes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl IntoIterator for Bytes {
    type Item = u8;
    type IntoIter = std::vec::IntoIter<u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<u8> for Bytes {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        Bytes(Vec::from_iter(iter))
    }
}

impl BinRead for Bytes {
    type Args<'a> = VecArgs<Vec<u8>>;

    fn read_options<R: Read + Seek>(
        reader: &mut R,
        _endian: Endian,
        args: Self::Args<'_>,
    ) -> BinResult<Self> {
        let mut list: Vec<u8> = Vec::new();
        list.try_reserve_exact(args.count).map_err(|e| {
            binrw::Error::Io(binrw::io::Error::new(
                binrw::io::ErrorKind::OutOfMemory,
                e.to_string(),
            ))
        })?;
        if reader.take(args.count as u64).read_to_end(&mut list)? == args.count {
            Ok(Bytes(list))
        } else {
            Err(binrw::Error::Io(binrw::io::Error::new(
                binrw::io::ErrorKind::UnexpectedEof,
                "Insufficient bytes in QPY for specified size",
            )))
        }
    }
}

impl BinWrite for Bytes {
    type Args<'a> = ();

    fn write_options<W: Write + Seek>(
        &self,
        writer: &mut W,
        _endian: Endian,
        _args: Self::Args<'_>,
    ) -> BinResult<()> {
        writer.write_all(&self.0)?;
        Ok(())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Bytes {
    type Error = PyErr;
    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(Self(obj.extract::<Vec<u8>>()?))
    }
}

impl<'py> IntoPyObject<'py> for Bytes {
    type Target = PyBytes;
    type Output = Bound<'py, PyBytes>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(Self::Target::new(py, &self.0))
    }
}
