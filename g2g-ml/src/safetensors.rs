//! A small, dependency-free reader (and writer) for the `safetensors` weight
//! format, so trained weights can be imported at runtime into the hand-rolled
//! inference elements (`WgpuInference::conv2d_from_safetensors`) without pulling
//! `serde` / the `safetensors` crate, and without ONNX-graph machinery.
//!
//! The format is a `u64` little-endian header length, then that many bytes of a
//! JSON header object `{ "name": {"dtype", "shape", "data_offsets":[begin,end]},
//! ..., "__metadata__": {..} }`, then the raw tensor bytes (`data_offsets` are
//! relative to the start of that data section). Only the JSON *subset*
//! safetensors emits is parsed (objects, arrays of non-negative integers,
//! strings), by a focused parser, not a general JSON library.
//!
//! Architecture is still defined in Rust (our `Module` is the `WgpuInference`
//! element); this loads only the *weights*, so picking a different trained
//! checkpoint at runtime is "open a different file", while the layer topology
//! stays compiled. Truly dynamic *architectures* are the `ort` backend's job.

use std::collections::BTreeMap;
use std::fmt;

/// Tensor element type as named in a safetensors header. Only [`Dtype::F32`] is
/// convertible here (the inference elements are f32); the rest are recognised so
/// a mixed file parses and a wrong-dtype lookup fails loud instead of silently
/// misreading bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F64,
    F16,
    Bf16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "F16" => Dtype::F16,
            "BF16" => Dtype::Bf16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            _ => return None,
        })
    }

    fn as_str(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::F16 => "F16",
            Dtype::Bf16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
        }
    }
}

/// Why a safetensors buffer could not be read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SafeTensorsError {
    /// Buffer shorter than the 8-byte length prefix, or header overruns it.
    Truncated,
    /// The JSON header was malformed (with a short reason).
    BadHeader(&'static str),
    /// A tensor named an unknown dtype string.
    UnknownDtype,
    /// A tensor's `data_offsets` fall outside the data section, or begin > end.
    BadOffsets,
    /// The requested tensor name is not in the file.
    Missing,
    /// The tensor's dtype is not F32, so `to_f32` cannot read it.
    NotF32,
    /// The tensor byte length is not a whole number of f32s, or disagrees with
    /// the shape's element count.
    LenMismatch,
}

impl fmt::Display for SafeTensorsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SafeTensorsError {}

struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// A parsed safetensors buffer: the tensor index plus a borrow of the data
/// section. Lookups return a [`TensorRef`] borrowing the original bytes (no copy
/// until `to_f32`).
#[derive(Debug)]
pub struct SafeTensors<'a> {
    data: &'a [u8],
    tensors: BTreeMap<String, Entry>,
}

impl fmt::Debug for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("bytes", &(self.end - self.begin))
            .finish()
    }
}

/// A view of one tensor: its dtype, shape, and the raw bytes (still in the parent
/// buffer). [`to_f32`](Self::to_f32) decodes an F32 tensor to a `Vec<f32>`.
#[derive(Debug)]
pub struct TensorRef<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    bytes: &'a [u8],
}

impl TensorRef<'_> {
    /// The tensor's raw little-endian bytes.
    pub fn bytes(&self) -> &[u8] {
        self.bytes
    }

    /// Number of elements implied by the shape (product of dims; 1 for a scalar).
    pub fn numel(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    /// Decode an F32 tensor to row-major `Vec<f32>`. Fails loud on a non-F32
    /// dtype or a byte length that is not the shape's element count of f32s.
    pub fn to_f32(&self) -> Result<Vec<f32>, SafeTensorsError> {
        if self.dtype != Dtype::F32 {
            return Err(SafeTensorsError::NotF32);
        }
        if self.bytes.len() % 4 != 0 || self.bytes.len() / 4 != self.numel() {
            return Err(SafeTensorsError::LenMismatch);
        }
        Ok(self
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }
}

impl<'a> SafeTensors<'a> {
    /// Parse a safetensors buffer, borrowing it. Validates the header length, the
    /// JSON header, and every tensor's offsets against the data section.
    pub fn parse(buf: &'a [u8]) -> Result<Self, SafeTensorsError> {
        if buf.len() < 8 {
            return Err(SafeTensorsError::Truncated);
        }
        let header_len = u64::from_le_bytes(buf[0..8].try_into().unwrap()) as usize;
        let header_end = 8usize.checked_add(header_len).ok_or(SafeTensorsError::Truncated)?;
        if header_end > buf.len() {
            return Err(SafeTensorsError::Truncated);
        }
        let header = core::str::from_utf8(&buf[8..header_end])
            .map_err(|_| SafeTensorsError::BadHeader("header is not utf-8"))?;
        let data = &buf[header_end..];

        let root = Json::parse(header).map_err(SafeTensorsError::BadHeader)?;
        let Json::Object(fields) = root else {
            return Err(SafeTensorsError::BadHeader("header root is not an object"));
        };

        let mut tensors = BTreeMap::new();
        for (name, value) in fields {
            // The `__metadata__` entry is free-form string->string; not a tensor.
            if name == "__metadata__" {
                continue;
            }
            let Json::Object(t) = value else {
                return Err(SafeTensorsError::BadHeader("tensor entry is not an object"));
            };
            let mut dtype = None;
            let mut shape = None;
            let mut offsets = None;
            for (k, v) in t {
                match k.as_str() {
                    "dtype" => {
                        let Json::String(s) = v else {
                            return Err(SafeTensorsError::BadHeader("dtype is not a string"));
                        };
                        dtype = Some(Dtype::from_str(&s).ok_or(SafeTensorsError::UnknownDtype)?);
                    }
                    "shape" => shape = Some(json_usize_array(v)?),
                    "data_offsets" => offsets = Some(json_usize_array(v)?),
                    _ => {}
                }
            }
            let dtype = dtype.ok_or(SafeTensorsError::BadHeader("tensor missing dtype"))?;
            let shape = shape.ok_or(SafeTensorsError::BadHeader("tensor missing shape"))?;
            let offsets = offsets.ok_or(SafeTensorsError::BadHeader("tensor missing data_offsets"))?;
            if offsets.len() != 2 {
                return Err(SafeTensorsError::BadHeader("data_offsets is not [begin, end]"));
            }
            let (begin, end) = (offsets[0], offsets[1]);
            if begin > end || end > data.len() {
                return Err(SafeTensorsError::BadOffsets);
            }
            tensors.insert(name, Entry { dtype, shape, begin, end });
        }
        Ok(SafeTensors { data, tensors })
    }

    /// The tensor names present, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Look up a tensor by name.
    pub fn get(&self, name: &str) -> Result<TensorRef<'_>, SafeTensorsError> {
        let e = self.tensors.get(name).ok_or(SafeTensorsError::Missing)?;
        Ok(TensorRef { dtype: e.dtype, shape: &e.shape, bytes: &self.data[e.begin..e.end] })
    }
}

/// Serialize F32 tensors to a safetensors buffer: `(name, shape, data)` each, in
/// the given order. Tensors are laid out back-to-back after the header. Useful to
/// export weights and to build test fixtures without a second dependency.
pub fn serialize(tensors: &[(&str, &[usize], &[f32])]) -> Vec<u8> {
    // Build the JSON header and the data section together so offsets line up.
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (i, (name, shape, values)) in tensors.iter().enumerate() {
        let begin = data.len();
        for v in *values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let end = data.len();
        if i > 0 {
            header.push(',');
        }
        let shape_str = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json_string(name),
            Dtype::F32.as_str(),
            shape_str,
            begin,
            end,
        ));
    }
    header.push('}');

    let header_bytes = header.into_bytes();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data);
    out
}

/// A JSON string literal for a (simple) key, with the escapes safetensors needs.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Interpret a JSON value as an array of non-negative integers (shape /
/// data_offsets).
fn json_usize_array(v: Json) -> Result<Vec<usize>, SafeTensorsError> {
    let Json::Array(items) = v else {
        return Err(SafeTensorsError::BadHeader("expected an integer array"));
    };
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        let Json::Number(n) = it else {
            return Err(SafeTensorsError::BadHeader("array element is not an integer"));
        };
        out.push(n as usize);
    }
    Ok(out)
}

/// The JSON subset a safetensors header uses: objects, arrays, strings, and
/// non-negative integers. Numbers are `u64` (shapes / offsets are never negative
/// or fractional). Parsed by a focused recursive-descent reader, not a general
/// JSON library, to keep the crate dependency-free.
enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    String(String),
    Number(u64),
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

/// Header nesting cap. A real header reaches depth ~3 (root object, tensor
/// object, shape array); this only fences off a crafted header from
/// overflowing the stack via unbounded recursion.
const MAX_JSON_DEPTH: u32 = 64;

impl Json {
    fn parse(s: &str) -> Result<Json, &'static str> {
        let mut p = Parser { bytes: s.as_bytes(), pos: 0 };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.pos != p.bytes.len() {
            return Err("trailing data after the JSON value");
        }
        Ok(v)
    }
}

impl Parser<'_> {
    fn ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn value(&mut self, depth: u32) -> Result<Json, &'static str> {
        if depth > MAX_JSON_DEPTH {
            return Err("json nesting too deep");
        }
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b) if b == b'-' || b.is_ascii_digit() => self.number(),
            _ => Err("unexpected token"),
        }
    }

    fn object(&mut self, depth: u32) -> Result<Json, &'static str> {
        self.pos += 1; // '{'
        let mut fields = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Object(fields));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return Err("expected a string key");
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return Err("expected ':' after key");
            }
            self.pos += 1;
            self.ws();
            let val = self.value(depth + 1)?;
            fields.push((key, val));
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Object(fields));
                }
                _ => return Err("expected ',' or '}' in object"),
            }
        }
    }

    fn array(&mut self, depth: u32) -> Result<Json, &'static str> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Array(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Array(items));
                }
                _ => return Err("expected ',' or ']' in array"),
            }
        }
    }

    fn string(&mut self) -> Result<String, &'static str> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let b = self.peek().ok_or("unterminated string")?;
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = self.peek().ok_or("unterminated escape")?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b't' => out.push('\t'),
                        b'r' => out.push('\r'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or("truncated \\u escape")?;
                            let code = core::str::from_utf8(hex)
                                .ok()
                                .and_then(|h| u32::from_str_radix(h, 16).ok())
                                .ok_or("bad \\u escape")?;
                            out.push(char::from_u32(code).ok_or("bad unicode scalar")?);
                            self.pos += 4;
                        }
                        _ => return Err("unknown escape"),
                    }
                }
                // A raw UTF-8 continuation/lead byte: copy it through. Tensor names
                // are ASCII in practice, but stay correct for multibyte names.
                _ => {
                    // Re-decode this byte plus any continuation bytes as one char.
                    let start = self.pos - 1;
                    let mut len = 1;
                    while self.bytes.get(start + len).is_some_and(|&c| c & 0xC0 == 0x80) {
                        len += 1;
                    }
                    let s = core::str::from_utf8(&self.bytes[start..start + len])
                        .map_err(|_| "invalid utf-8 in string")?;
                    out.push_str(s);
                    self.pos = start + len;
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, &'static str> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            return Err("negative number in header");
        }
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == start {
            return Err("empty number");
        }
        // Reject fractional / exponent forms: header ints only.
        if matches!(self.peek(), Some(b'.') | Some(b'e') | Some(b'E')) {
            return Err("non-integer number in header");
        }
        let s = core::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        s.parse::<u64>().map(Json::Number).map_err(|_| "integer overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_serialize_then_parse() {
        let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 1.0).collect();
        let b = vec![0.25f32, -0.5];
        let blob = serialize(&[("conv.weight", &[2, 2, 1, 3], &w), ("conv.bias", &[2], &b)]);

        let st = SafeTensors::parse(&blob).expect("parses");
        assert_eq!(st.names().collect::<Vec<_>>(), vec!["conv.bias", "conv.weight"]);

        let wt = st.get("conv.weight").expect("weight present");
        assert_eq!(wt.dtype, Dtype::F32);
        assert_eq!(wt.shape, &[2, 2, 1, 3]);
        assert_eq!(wt.to_f32().unwrap(), w);

        let bt = st.get("conv.bias").unwrap();
        assert_eq!(bt.shape, &[2]);
        assert_eq!(bt.to_f32().unwrap(), b);
    }

    #[test]
    fn parses_metadata_and_whitespace() {
        // Hand-built header with __metadata__, extra whitespace, and key order
        // unlike serialize's output, to exercise the parser not just the writer.
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = "{ \"__metadata__\": {\"framework\": \"pt\"} ,\n  \
            \"t\" : { \"shape\" : [2, 2] , \"dtype\":\"F32\", \"data_offsets\":[0,16] } }";
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&data);

        let st = SafeTensors::parse(&blob).expect("parses with metadata");
        assert_eq!(st.names().collect::<Vec<_>>(), vec!["t"], "__metadata__ is not a tensor");
        assert_eq!(st.get("t").unwrap().to_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_truncated_and_out_of_range() {
        assert_eq!(SafeTensors::parse(&[0u8; 4]).unwrap_err(), SafeTensorsError::Truncated);
        // Header claims more bytes than present.
        let mut blob = 1000u64.to_le_bytes().to_vec();
        blob.extend_from_slice(b"{}");
        assert_eq!(SafeTensors::parse(&blob).unwrap_err(), SafeTensorsError::Truncated);
    }

    #[test]
    fn offsets_past_data_fail_loud() {
        let header = "{\"t\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&[0u8; 8]); // only 8 data bytes, offsets want 16
        assert_eq!(SafeTensors::parse(&blob).unwrap_err(), SafeTensorsError::BadOffsets);
    }

    #[test]
    fn deeply_nested_header_is_rejected_not_overflowed() {
        // A crafted header nested past the depth cap must fail loud rather than
        // recurse into a stack overflow.
        let depth = MAX_JSON_DEPTH as usize + 50;
        let bomb = format!("{}{}", "[".repeat(depth), "]".repeat(depth));
        assert_eq!(Json::parse(&bomb).err(), Some("json nesting too deep"));
    }

    #[test]
    fn wrong_dtype_to_f32_fails() {
        let header = "{\"t\":{\"dtype\":\"I32\",\"shape\":[2],\"data_offsets\":[0,8]}}";
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(header.as_bytes());
        blob.extend_from_slice(&[0u8; 8]);
        let st = SafeTensors::parse(&blob).unwrap();
        assert_eq!(st.get("t").unwrap().dtype, Dtype::I32);
        assert_eq!(st.get("t").unwrap().to_f32(), Err(SafeTensorsError::NotF32));
    }
}
