// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! The lexing logic for OpenQASM 2, responsible for turning a sequence of bytes into a
//! lexed [TokenStream] for consumption by the parsing machinery.  The general strategy here is
//! quite simple; for all the symbol-like tokens, the lexer can use a very simple single-byte
//! lookahead to determine what token it needs to emit.  For keywords and identifiers, we just read
//! the identifier in completely, then produce the right token once we see the end of the
//! identifier characters.
//!
//! We effectively use a custom lexing mode to handle the version information after the `OPENQASM`
//! keyword; the spec technically says that any real number is valid, but in reality that leads to
//! weirdness like `200.0e-2` being a valid version specifier.  We do things with a custom
//! context-dependent match after seeing an `OPENQASM` token, to avoid clashes with the general
//! real-number tokenization.

use hashbrown::HashMap;
use num_bigint::BigUint;

use std::path::Path;

use crate::error::{ParseError, Position, message_generic};

/// Tokenized version information data.  This is more structured than the real number suggested by
/// the specification.
#[derive(Clone, Debug)]
pub struct Version {
    pub major: usize,
    pub minor: Option<usize>,
}

/// The context that is necessary to fully extract the information from a [Token].  This owns, for
/// example, the text of each token (where a token does not have a static text representation),
/// from which the other properties can later be derived.  This struct is effectively entirely
/// opaque outside this module; the associated functions on [Token] take this context object,
/// however, and extract the information from it.
#[derive(Clone, Debug)]
pub struct TokenContext {
    text: Vec<String>,
    lookup: HashMap<Vec<u8>, usize>,
}

impl TokenContext {
    /// Create a new context for tokens.  Nothing is heap-allocated until required.
    pub fn new() -> Self {
        TokenContext {
            text: vec![],
            lookup: HashMap::new(),
        }
    }

    /// Intern the given `ascii_text` of a [Token], and return an index into the [TokenContext].
    /// This will not store strings that are already present in the context; instead, the previous
    /// index is transparently returned.
    fn index(&mut self, ascii_text: &[u8]) -> usize {
        match self.lookup.get(ascii_text) {
            Some(index) => *index,
            None => {
                let index = self.text.len();
                self.lookup.insert(ascii_text.to_vec(), index);
                self.text
                    .push(std::str::from_utf8(ascii_text).unwrap().to_owned());
                index
            }
        }
    }
}

// Clippy complains without this.
impl Default for TokenContext {
    fn default() -> Self {
        Self::new()
    }
}

/// An enumeration of the different types of [Token] that can be created during lexing.  This is
/// deliberately not a data enum, to make various abstract `expect` (and so on) methods more
/// ergonomic to use; one does not need to completely define the pattern match each time, but can
/// simply pass the type identifier.  This also saves memory, since the static variants do not need
/// to be aligned to include the space necessary for text pointers that would be in the non-static
/// forms, and allows strings to be shared between many tokens (using the [TokenContext] store).
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TokenType {
    // Keywords
    OpenQASM,
    Barrier,
    Cos,
    Creg,
    Exp,
    Gate,
    If,
    Include,
    Ln,
    Measure,
    Opaque,
    Qreg,
    Reset,
    Sin,
    Sqrt,
    Tan,
    Pi,
    // Symbols
    Plus,
    Minus,
    Arrow,
    Asterisk,
    Equals,
    Slash,
    Caret,
    Semicolon,
    Comma,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    // Content
    Id,
    Real,
    Integer,
    Filename,
    Version,
}

impl TokenType {
    pub fn variable_text(&self) -> bool {
        match self {
            TokenType::OpenQASM
            | TokenType::Barrier
            | TokenType::Cos
            | TokenType::Creg
            | TokenType::Exp
            | TokenType::Gate
            | TokenType::If
            | TokenType::Include
            | TokenType::Ln
            | TokenType::Measure
            | TokenType::Opaque
            | TokenType::Qreg
            | TokenType::Reset
            | TokenType::Sin
            | TokenType::Sqrt
            | TokenType::Tan
            | TokenType::Pi
            | TokenType::Plus
            | TokenType::Minus
            | TokenType::Arrow
            | TokenType::Asterisk
            | TokenType::Equals
            | TokenType::Slash
            | TokenType::Caret
            | TokenType::Semicolon
            | TokenType::Comma
            | TokenType::LParen
            | TokenType::RParen
            | TokenType::LBracket
            | TokenType::RBracket
            | TokenType::LBrace
            | TokenType::RBrace => false,
            TokenType::Id
            | TokenType::Real
            | TokenType::Integer
            | TokenType::Filename
            | TokenType::Version => true,
        }
    }

    /// Get a static description of the token type.  This is useful for producing messages when the
    /// full token context isn't available, or isn't important.
    pub fn describe(&self) -> &'static str {
        match self {
            TokenType::OpenQASM => "OPENQASM",
            TokenType::Barrier => "barrier",
            TokenType::Cos => "cos",
            TokenType::Creg => "creg",
            TokenType::Exp => "exp",
            TokenType::Gate => "gate",
            TokenType::If => "if",
            TokenType::Include => "include",
            TokenType::Ln => "ln",
            TokenType::Measure => "measure",
            TokenType::Opaque => "opaque",
            TokenType::Qreg => "qreg",
            TokenType::Reset => "reset",
            TokenType::Sin => "sin",
            TokenType::Sqrt => "sqrt",
            TokenType::Tan => "tan",
            TokenType::Pi => "pi",
            TokenType::Plus => "+",
            TokenType::Minus => "-",
            TokenType::Arrow => "->",
            TokenType::Asterisk => "*",
            TokenType::Equals => "==",
            TokenType::Slash => "/",
            TokenType::Caret => "^",
            TokenType::Semicolon => ";",
            TokenType::Comma => ",",
            TokenType::LParen => "(",
            TokenType::RParen => ")",
            TokenType::LBracket => "[",
            TokenType::RBracket => "]",
            TokenType::LBrace => "{",
            TokenType::RBrace => "}",
            TokenType::Id => "an identifier",
            TokenType::Real => "a real number",
            TokenType::Integer => "an integer",
            TokenType::Filename => "a filename string",
            TokenType::Version => "a '<major>.<minor>' version",
        }
    }
}

/// A representation of a token, including its type, span information and pointer to where its text
/// is stored in the context object.  These are relatively lightweight objects (though of course
/// not as light as the single type information).
#[derive(Clone, Copy, Debug)]
pub struct Token {
    pub ttype: TokenType,
    // The `line` and `col` refer only to the start of the token.  There are no tokens that span
    // more than one line (we don't tokenise comments), but the ending column offset can be
    // calculated by asking the associated `TokenContext` for the text associated with this token,
    // and inspecting the length of the returned value.
    pub line: usize,
    pub col: usize,
    // Index into the TokenContext object, to retrieve the text that makes up the token.  We don't
    // resolve this into a value during lexing; that comes with annoying typing issues or storage
    // wastage.  Instead, we only convert the text into a value type when asked to by calling a
    // relevant method on the token.
    index: usize,
}

impl Token {
    /// Get a reference to the string that was seen to generate this token.
    pub fn text<'a>(&self, context: &'a TokenContext) -> &'a str {
        match self.ttype {
            TokenType::Id
            | TokenType::Real
            | TokenType::Integer
            | TokenType::Filename
            | TokenType::Version => &context.text[self.index],
            _ => self.ttype.describe(),
        }
    }

    /// If the token is an identifier, this method can be called to get an owned string containing
    /// the text of the identifier.  Panics if the token is not an identifier.
    pub fn id(&self, context: &TokenContext) -> String {
        if self.ttype != TokenType::Id {
            panic!()
        }
        (&context.text[self.index]).into()
    }

    /// If the token is a real number, this method can be called to evaluate its value.  Panics if
    /// the token is not a float or an integer.
    pub fn real(&self, context: &TokenContext) -> f64 {
        if !(self.ttype == TokenType::Real || self.ttype == TokenType::Integer) {
            panic!()
        }
        context.text[self.index].parse().unwrap()
    }

    /// If the token is an integer (by type, not just by value), this method can be called to
    /// evaluate its value.  Panics if the token is not an integer type.
    pub fn int(&self, context: &TokenContext) -> usize {
        if self.ttype != TokenType::Integer {
            panic!()
        }
        context.text[self.index].parse().unwrap()
    }

    /// If the token is an integer (by type, not just by value), this method can be called to
    /// evaluate its value as a big integer.  Panics if the token is not an integer type.
    pub fn bigint(&self, context: &TokenContext) -> BigUint {
        if self.ttype != TokenType::Integer {
            panic!()
        }
        context.text[self.index].parse().unwrap()
    }

    /// If the token is a filename path, this method can be called to get a (regular) string
    /// representing it.  Panics if the token type was not a filename.
    pub fn filename(&self, context: &TokenContext) -> String {
        if self.ttype != TokenType::Filename {
            panic!()
        }
        let out = &context.text[self.index];
        // String slicing is fine to assume bytes here, because the characters we're slicing out
        // must both be the ASCII '"', which is a single-byte UTF-8 character.
        out[1..out.len() - 1].into()
    }

    /// If the token is a version-information token, this method can be called to evaluate the
    /// version information.  Panics if the token was not of the correct type.
    pub fn version(&self, context: &TokenContext) -> Version {
        if self.ttype != TokenType::Version {
            panic!()
        }
        // Everything in the version token is a valid ASCII character, so must be a one-byte token.
        let text = &context.text[self.index];
        match text.chars().position(|c| c == '.') {
            Some(pos) => Version {
                major: text[0..pos].parse().unwrap(),
                minor: Some(text[pos + 1..text.len()].parse().unwrap()),
            },
            None => Version {
                major: text.parse().unwrap(),
                minor: None,
            },
        }
    }
}

/// The workhouse struct of the lexer.  This represents a peekable iterable object that is abstract
/// over some buffered reader.  The struct itself essentially represents the mutable state of the
/// lexer, with its main public associated functions being the iterable method [Self::next()] and
/// the [std::iter::Peekable]-like function [Self::peek()].
///
/// The stream exposes one public attributes directly: the [filename] that this stream comes from
/// (set to some placeholder value for streams that do not have a backing file).  The associated
/// `TokenContext` object is managed separately to the stream and is passed in each call to `next`;
/// this allows for multiple streams to operate on the same context, such as when a new stream
/// begins in order to handle an `include` statement.
pub struct TokenStream {
    /// The filename from which this stream is derived.  May be a placeholder if there is no
    /// backing file or other named resource.
    pub filename: std::ffi::OsString,
    strict: bool,
    source: Box<dyn std::io::BufRead + Send + Sync>,
    line_buffer: Vec<u8>,
    done: bool,
    line: usize,
    col: usize,
    try_version: bool,
    // This is a manual peekable structure (rather than using the `peekable` method of `Iterator`)
    // because we still want to be able to access the other members of the struct at the same time.
    peeked: Option<Option<Token>>,
}

impl TokenStream {
    /// Create and initialise a generic [TokenStream], given a source that implements
    /// [std::io::BufRead] and a filename (or resource path) that describes its source.
    fn new(
        source: Box<dyn std::io::BufRead + Send + Sync>,
        filename: std::ffi::OsString,
        strict: bool,
    ) -> Self {
        TokenStream {
            filename,
            strict,
            source,
            line_buffer: Vec::with_capacity(80),
            done: false,
            // The first line is numbered "1", and the first column is "0".  The counts are
            // initialized like this so the first call to `next_byte` can easily detect that it
            // needs to extract the next line.
            line: 0,
            col: 0,
            try_version: false,
            peeked: None,
        }
    }

    /// Create a [TokenStream] from a string containing the OpenQASM 2 program.
    pub fn from_string(string: String, strict: bool) -> Self {
        TokenStream::new(
            Box::new(std::io::Cursor::new(string)),
            "<input>".into(),
            strict,
        )
    }

    /// Create a [TokenStream] from a path containing the OpenQASM 2 program.
    pub fn from_path<P: AsRef<Path>>(path: P, strict: bool) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path.as_ref())?;
        Ok(TokenStream::new(
            Box::new(std::io::BufReader::new(file)),
            Path::file_name(path.as_ref()).unwrap().into(),
            strict,
        ))
    }

    /// Read the next line into the managed buffer in the struct, updating the tracking information
    /// of the position, and the `done` state of the iterator.
    fn advance_line(&mut self) -> Result<usize, ParseError> {
        if self.done {
            Ok(0)
        } else {
            self.line += 1;
            self.col = 0;
            self.line_buffer.clear();
            // We can assume that nobody's running this on ancient Mac software that uses only '\r'
            // as its linebreak character.
            match self.source.read_until(b'\n', &mut self.line_buffer) {
                Ok(count) => {
                    if count == 0 || self.line_buffer[count - 1] != b'\n' {
                        self.done = true;
                    }
                    Ok(count)
                }
                Err(err) => {
                    self.done = true;
                    Err(ParseError::new(message_generic(
                        Some(&Position::new(&self.filename, self.line, self.col)),
                        &format!("lexer failed to read stream: {err}"),
                    )))
                }
            }
        }
    }

    /// Get the next character in the stream.  This updates the line and column information for the
    /// current byte as well.
    fn next_byte(&mut self) -> Result<Option<u8>, ParseError> {
        if self.col >= self.line_buffer.len() && self.advance_line()? == 0 {
            return Ok(None);
        }
        let out = self.line_buffer[self.col];
        self.col += 1;
        match out {
            b @ 0x80..=0xff => {
                self.done = true;
                Err(ParseError::new(message_generic(
                    Some(&Position::new(&self.filename, self.line, self.col)),
                    &format!("encountered a non-ASCII byte: {b:02X?}"),
                )))
            }
            b => Ok(Some(b)),
        }
    }

    /// Peek at the next byte in the stream without consuming it.  This still returns an error if
    /// the next byte isn't in the valid range for OpenQASM 2, or if the file/stream has failed to
    /// read into the buffer for some reason.
    fn peek_byte(&mut self) -> Result<Option<u8>, ParseError> {
        if self.col >= self.line_buffer.len() && self.advance_line()? == 0 {
            return Ok(None);
        }
        match self.line_buffer[self.col] {
            b @ 0x80..=0xff => {
                self.done = true;
                Err(ParseError::new(message_generic(
                    Some(&Position::new(&self.filename, self.line, self.col)),
                    &format!("encountered a non-ASCII byte: {b:02X?}"),
                )))
            }
            b => Ok(Some(b)),
        }
    }

    /// Expect that the next byte is not a word continuation, providing a suitable error message if
    /// it is.
    fn expect_word_boundary(&mut self, after: &str, start_col: usize) -> Result<(), ParseError> {
        match self.peek_byte()? {
            Some(c @ (b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_')) => {
                Err(ParseError::new(message_generic(
                    Some(&Position::new(&self.filename, self.line, start_col)),
                    &format!(
                        "expected a word boundary after {}, but saw '{}'",
                        after, c as char
                    ),
                )))
            }
            _ => Ok(()),
        }
    }

    /// Complete the lexing of a floating-point value from the position of maybe accepting an
    /// exponent.  The previous part of the token must be a valid stand-alone float, or the next
    /// byte must already have been peeked and known to be `b'e' | b'E'`.
    fn lex_float_exponent(&mut self, start_col: usize) -> Result<TokenType, ParseError> {
        if !matches!(self.peek_byte()?, Some(b'e' | b'E')) {
            self.expect_word_boundary("a float", start_col)?;
            return Ok(TokenType::Real);
        }
        // Consume the rest of the exponent.
        self.next_byte()?;
        if let Some(b'+' | b'-') = self.peek_byte()? {
            self.next_byte()?;
        }
        // Exponents must have at least one digit in them.
        if !matches!(self.peek_byte()?, Some(b'0'..=b'9')) {
            return Err(ParseError::new(message_generic(
                Some(&Position::new(&self.filename, self.line, start_col)),
                "needed to see an integer exponent for this float",
            )));
        }
        while let Some(b'0'..=b'9') = self.peek_byte()? {
            self.next_byte()?;
        }
        self.expect_word_boundary("a float", start_col)?;
        Ok(TokenType::Real)
    }

    /// Lex a numeric token completely.  This can return a successful integer or a real number; the
    /// function distinguishes based on what it sees.  If `self.try_version`, this can also be a
    /// version identifier (will take precedence over either other type, if possible).
    fn lex_numeric(&mut self, start_col: usize) -> Result<TokenType, ParseError> {
        let first = self.line_buffer[start_col];
        if first == b'.' {
            return match self.next_byte()? {
                // In the case of a float that begins with '.', we require at least one digit, so
                // just force consume it and then loop over the rest.
                Some(b'0'..=b'9') => {
                    while let Some(b'0'..=b'9') = self.peek_byte()? {
                        self.next_byte()?;
                    }
                    self.lex_float_exponent(start_col)
                }
                _ => Err(ParseError::new(message_generic(
                    Some(&Position::new(&self.filename, self.line, start_col)),
                    "expected a numeric fractional part after the bare decimal point",
                ))),
            };
        }
        while let Some(b'0'..=b'9') = self.peek_byte()? {
            self.next_byte()?;
        }
        match self.peek_byte()? {
            Some(b'.') => {
                self.next_byte()?;
                let mut has_fractional = false;
                while let Some(b'0'..=b'9') = self.peek_byte()? {
                    has_fractional = true;
                    self.next_byte()?;
                }
                if self.try_version
                    && has_fractional
                    && !matches!(self.peek_byte()?, Some(b'e' | b'E'))
                {
                    self.expect_word_boundary("a version identifier", start_col)?;
                    return Ok(TokenType::Version);
                }
                return self.lex_float_exponent(start_col);
            }
            // In this situation, what we've lexed so far is an integer (maybe with leading
            // zeroes), but it can still be a float if it's followed by an exponent.  This
            // particular path is not technically within the spec (so should be subject to `strict`
            // mode), but pragmatically that's more just a nuisance for OQ2 generators, since many
            // languages will happily spit out something like `5e-5` when formatting floats.
            Some(b'e' | b'E') => {
                return if self.strict {
                    Err(ParseError::new(message_generic(
                        Some(&Position::new(&self.filename, self.line, start_col)),
                        "[strict] all floats must include a decimal point",
                    )))
                } else {
                    self.lex_float_exponent(start_col)
                };
            }
            _ => (),
        }
        if first == b'0' && self.col - start_col > 1 {
            // Integers can't start with a leading zero unless they are only the single '0', but we
            // didn't see a decimal point.
            Err(ParseError::new(message_generic(
                Some(&Position::new(&self.filename, self.line, start_col)),
                "integers cannot have leading zeroes",
            )))
        } else if self.try_version {
            self.expect_word_boundary("a version identifier", start_col)?;
            Ok(TokenType::Version)
        } else {
            self.expect_word_boundary("an integer", start_col)?;
            Ok(TokenType::Integer)
        }
    }

    /// Lex a text-like token into a complete token.  This can return any of the keyword-like
    /// tokens (e.g. [TokenType::Pi]), or a [TokenType::Id] if the token is not a built-in keyword.
    fn lex_textlike(&mut self, start_col: usize) -> Result<TokenType, ParseError> {
        let first = self.line_buffer[start_col];
        while let Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_') = self.peek_byte()? {
            self.next_byte()?;
        }
        // No need to expect the word boundary after this, because it's the same check as above.
        let text = &self.line_buffer[start_col..self.col];
        if let b'A'..=b'Z' = first {
            match text {
                b"OPENQASM" => Ok(TokenType::OpenQASM),
                b"U" | b"CX" => Ok(TokenType::Id),
                _ => Err(ParseError::new(message_generic(
                    Some(&Position::new(&self.filename, self.line, start_col)),
                    "identifiers cannot start with capital letters except for the builtins 'U' and 'CX'",
                ))),
            }
        } else {
            match text {
                b"barrier" => Ok(TokenType::Barrier),
                b"cos" => Ok(TokenType::Cos),
                b"creg" => Ok(TokenType::Creg),
                b"exp" => Ok(TokenType::Exp),
                b"gate" => Ok(TokenType::Gate),
                b"if" => Ok(TokenType::If),
                b"include" => Ok(TokenType::Include),
                b"ln" => Ok(TokenType::Ln),
                b"measure" => Ok(TokenType::Measure),
                b"opaque" => Ok(TokenType::Opaque),
                b"qreg" => Ok(TokenType::Qreg),
                b"reset" => Ok(TokenType::Reset),
                b"sin" => Ok(TokenType::Sin),
                b"sqrt" => Ok(TokenType::Sqrt),
                b"tan" => Ok(TokenType::Tan),
                b"pi" => Ok(TokenType::Pi),
                _ => Ok(TokenType::Id),
            }
        }
    }

    /// Lex a filename token completely.  This is always triggered by seeing a `b'"'` byte in the
    /// input stream.
    fn lex_filename(&mut self, terminator: u8, start_col: usize) -> Result<TokenType, ParseError> {
        loop {
            match self.next_byte()? {
                None => {
                    return Err(ParseError::new(message_generic(
                        Some(&Position::new(&self.filename, self.line, start_col)),
                        "unexpected end-of-file while lexing string literal",
                    )));
                }
                Some(b'\n' | b'\r') => {
                    return Err(ParseError::new(message_generic(
                        Some(&Position::new(&self.filename, self.line, start_col)),
                        "unexpected line break while lexing string literal",
                    )));
                }
                Some(c) if c == terminator => {
                    return Ok(TokenType::Filename);
                }
                Some(_) => (),
            }
        }
    }

    /// The actual core of the iterator.  Read from the stream (ignoring preceding whitespace)
    /// until a complete [Token] has been constructed, or the end of the iterator is reached.  This
    /// returns `Some` for all tokens, including the error token, and only returns `None` if there
    /// are no more tokens left to take.
    fn next_inner(&mut self, context: &mut TokenContext) -> Result<Option<Token>, ParseError> {
        // Consume preceding whitespace.  Beware that this can still exhaust the underlying stream,
        // or scan through an invalid token in the encoding.
        loop {
            match self.peek_byte()? {
                Some(b' ' | b'\t' | b'\r' | b'\n') => {
                    self.next_byte()?;
                }
                None => return Ok(None),
                _ => break,
            }
        }
        let start_col = self.col;
        // The whitespace loop (or [Self::try_lex_version]) has already peeked the next token, so
        // we know it's going to be the `Some` variant.
        let ttype = match self.next_byte()?.unwrap() {
            b'+' => TokenType::Plus,
            b'*' => TokenType::Asterisk,
            b'^' => TokenType::Caret,
            b';' => TokenType::Semicolon,
            b',' => TokenType::Comma,
            b'(' => TokenType::LParen,
            b')' => TokenType::RParen,
            b'[' => TokenType::LBracket,
            b']' => TokenType::RBracket,
            b'{' => TokenType::LBrace,
            b'}' => TokenType::RBrace,
            b'/' => {
                if let Some(b'/') = self.peek_byte()? {
                    return if self.advance_line()? == 0 {
                        Ok(None)
                    } else {
                        self.next(context)
                    };
                } else {
                    TokenType::Slash
                }
            }
            b'-' => {
                if let Ok(Some(b'>')) = self.peek_byte() {
                    self.col += 1;
                    TokenType::Arrow
                } else {
                    TokenType::Minus
                }
            }
            b'=' => {
                if let Ok(Some(b'=')) = self.peek_byte() {
                    self.col += 1;
                    TokenType::Equals
                } else {
                    return Err(ParseError::new(
                        "single equals '=' is never valid".to_owned(),
                    ));
                }
            }
            b'0'..=b'9' | b'.' => self.lex_numeric(start_col)?,
            b'a'..=b'z' | b'A'..=b'Z' => self.lex_textlike(start_col)?,
            c @ (b'"' | b'\'') => {
                if self.strict && c != b'"' {
                    return Err(ParseError::new(message_generic(
                        Some(&Position::new(&self.filename, self.line, start_col)),
                        "[strict] paths must be in double quotes (\"\")",
                    )));
                } else {
                    self.lex_filename(c, start_col)?
                }
            }
            c => {
                return Err(ParseError::new(message_generic(
                    Some(&Position::new(&self.filename, self.line, start_col)),
                    &format!(
                        "encountered '{}', which doesn't match any valid tokens",
                        // Non-ASCII bytes should already have been rejected by `next_byte()`.
                        c as char,
                    ),
                )));
            }
        };
        self.try_version = ttype == TokenType::OpenQASM;
        Ok(Some(Token {
            ttype,
            line: self.line,
            col: start_col,
            index: if ttype.variable_text() {
                context.index(&self.line_buffer[start_col..self.col])
            } else {
                usize::MAX
            },
        }))
    }

    /// Get an optional reference to the next token in the iterator stream without consuming it.
    /// This is a direct analogue of the same method on the [std::iter::Peekable] struct, except it
    /// is manually defined here to avoid hiding the rest of the public fields of the [TokenStream]
    /// struct itself.
    pub fn peek(&mut self, context: &mut TokenContext) -> Result<Option<&Token>, ParseError> {
        if self.peeked.is_none() {
            self.peeked = Some(self.next_inner(context)?);
        }
        Ok(self.peeked.as_ref().unwrap().as_ref())
    }

    pub fn next(&mut self, context: &mut TokenContext) -> Result<Option<Token>, ParseError> {
        match self.peeked.take() {
            Some(token) => Ok(token),
            None => self.next_inner(context),
        }
    }
}
