// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

pub const EDL_URI_SCHEME: &str = "edl://";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EdlSegment {
    pub file: String,
    pub start_ns: u64,
    pub length_ns: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EdlStream {
    pub segments: Vec<EdlSegment>,
    pub no_clip: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edl {
    pub streams: Vec<EdlStream>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EdlError {
    NotAnEdlUri,
    LengthPrefix {
        at: usize,
    },
    ReservedCharacter {
        character: char,
        at: usize,
    },
    UnknownHeader(String),
    UnknownParameter(String),
    RepeatedParameter(String),
    TooManyValues,
    NoFile,
    Time {
        parameter: &'static str,
        value: String,
    },
    NoSegments,
}

impl fmt::Display for EdlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAnEdlUri => write!(f, "the URI does not start with {EDL_URI_SCHEME}"),
            Self::LengthPrefix { at } => write!(
                f,
                "the %<length>% value at byte {at} is malformed or runs past the end"
            ),
            Self::ReservedCharacter { character, at } => write!(
                f,
                "byte {at} holds '{character}', which only a %<length>% value may contain"
            ),
            Self::UnknownHeader(name) => write!(f, "the header !{name} is not supported"),
            Self::UnknownParameter(name) => write!(f, "the parameter {name} is not supported"),
            Self::RepeatedParameter(name) => write!(f, "the parameter {name} is given twice"),
            Self::TooManyValues => {
                write!(
                    f,
                    "a segment takes at most three unnamed values: file, start and length"
                )
            }
            Self::NoFile => write!(f, "a segment names no file"),
            Self::Time { parameter, value } => {
                write!(f, "{parameter}={value} is not a time in seconds")
            }
            Self::NoSegments => write!(f, "a stream has no segments"),
        }
    }
}

const SEGMENT_PARAMETERS: [&str; 3] = ["file", "start", "length"];

#[derive(Debug)]
struct Parameter {
    name: Option<String>,
    value: String,
}

#[derive(Debug)]
struct Scanner<'a> {
    text: &'a str,
    at: usize,
}

impl Scanner<'_> {
    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.at).copied()
    }

    fn at_entry_end(&self) -> bool {
        matches!(self.peek(), None | Some(b';' | b'\n'))
    }

    fn skip_entry(&mut self) {
        while !self.at_entry_end() {
            self.at += 1;
        }
    }

    fn token(&mut self, stops: &[u8]) -> Result<&str, EdlError> {
        let start = self.at;
        while let Some(byte) = self.peek() {
            if stops.contains(&byte) {
                break;
            }
            // mpv reserves `!` outside a %<length>% value
            if byte == b'!' {
                return Err(EdlError::ReservedCharacter {
                    character: '!',
                    at: self.at,
                });
            }
            self.at += 1;
        }
        Ok(&self.text[start..self.at])
    }

    fn length_prefixed(&mut self) -> Result<&str, EdlError> {
        let prefix_at = self.at;
        let malformed = EdlError::LengthPrefix { at: prefix_at };
        self.at += 1;
        let digits_start = self.at;
        while self.peek().is_some_and(|byte| byte.is_ascii_digit()) {
            self.at += 1;
        }
        let length: usize = self.text[digits_start..self.at]
            .parse()
            .map_err(|_| malformed.clone())?;
        if self.peek() != Some(b'%') {
            return Err(malformed);
        }
        self.at += 1;
        let end = self.at.checked_add(length).ok_or(malformed.clone())?;
        let value = self.text.get(self.at..end).ok_or(malformed)?;
        self.at = end;
        Ok(value)
    }

    fn value(&mut self) -> Result<String, EdlError> {
        if self.peek() == Some(b'%') {
            return self.length_prefixed().map(ToString::to_string);
        }
        self.token(b",;\n").map(ToString::to_string)
    }

    fn parameter(&mut self) -> Result<Parameter, EdlError> {
        if self.peek() == Some(b'%') {
            return Ok(Parameter {
                name: None,
                value: self.value()?,
            });
        }
        let start = self.at;
        let head = self.token(b"=,;\n")?.to_string();
        if self.peek() != Some(b'=') {
            return Ok(Parameter {
                name: None,
                value: head,
            });
        }
        if head.contains('%') {
            return Err(EdlError::ReservedCharacter {
                character: '%',
                at: start,
            });
        }
        self.at += 1;
        Ok(Parameter {
            name: Some(head),
            value: self.value()?,
        })
    }

    fn parameters(&mut self) -> Result<Vec<Parameter>, EdlError> {
        let mut parameters = alloc::vec![self.parameter()?];
        while self.peek() == Some(b',') {
            self.at += 1;
            parameters.push(self.parameter()?);
        }
        if !self.at_entry_end() {
            return Err(EdlError::ReservedCharacter {
                character: char::from(self.peek().unwrap_or_default()),
                at: self.at,
            });
        }
        Ok(parameters)
    }
}

fn nanoseconds(parameter: &'static str, value: &str) -> Result<u64, EdlError> {
    const NANOSECONDS_PER_SECOND: f64 = 1_000_000_000.0;
    let invalid = || EdlError::Time {
        parameter,
        value: value.to_string(),
    };
    let seconds: f64 = value.parse().map_err(|_| invalid())?;
    let nanoseconds = (seconds * NANOSECONDS_PER_SECOND).round();
    if !nanoseconds.is_finite() || nanoseconds < 0.0 || nanoseconds > u64::MAX as f64 {
        return Err(invalid());
    }
    Ok(nanoseconds as u64)
}

fn segment(parameters: Vec<Parameter>) -> Result<EdlSegment, EdlError> {
    let mut values: [Option<String>; 3] = Default::default();
    let mut next_position = 0;
    for parameter in parameters {
        let slot = match parameter.name {
            Some(name) => SEGMENT_PARAMETERS
                .iter()
                .position(|known| *known == name)
                .ok_or(EdlError::UnknownParameter(name))?,
            None => {
                let slot = next_position;
                next_position += 1;
                if slot >= SEGMENT_PARAMETERS.len() {
                    return Err(EdlError::TooManyValues);
                }
                slot
            }
        };
        if values[slot].replace(parameter.value).is_some() {
            return Err(EdlError::RepeatedParameter(
                SEGMENT_PARAMETERS[slot].to_string(),
            ));
        }
    }
    let [file, start, length] = values;
    let file = file
        .filter(|file| !file.is_empty())
        .ok_or(EdlError::NoFile)?;
    Ok(EdlSegment {
        file,
        start_ns: start.map_or(Ok(0), |start| nanoseconds("start", &start))?,
        length_ns: length
            .map(|length| nanoseconds("length", &length))
            .transpose()?,
    })
}

pub fn parse_edl_uri(uri: &str) -> Result<Edl, EdlError> {
    let text = uri
        .strip_prefix(EDL_URI_SCHEME)
        .ok_or(EdlError::NotAnEdlUri)?;
    let mut scanner = Scanner { text, at: 0 };
    let mut streams = alloc::vec![EdlStream::default()];
    while scanner.at < text.len() {
        match scanner.peek() {
            Some(b';' | b'\n') => {}
            Some(b'#') => scanner.skip_entry(),
            Some(b'!') => {
                scanner.at += 1;
                let mut parameters = scanner.parameters()?.into_iter();
                let header = parameters
                    .next()
                    .map(|header| header.value)
                    .unwrap_or_default();
                if !["new_stream", "no_clip"].contains(&header.as_str()) {
                    return Err(EdlError::UnknownHeader(header));
                }
                if let Some(extra) = parameters.next() {
                    return Err(EdlError::UnknownParameter(
                        extra.name.unwrap_or(extra.value),
                    ));
                }
                let current = streams.last_mut().expect("streams starts with one");
                match header.as_str() {
                    // mpv ignores a new_stream before any segment
                    "new_stream" if current.segments.is_empty() => {}
                    "new_stream" => streams.push(EdlStream::default()),
                    _ => current.no_clip = true,
                }
            }
            Some(_) => {
                let parameters = scanner.parameters()?;
                streams
                    .last_mut()
                    .expect("streams starts with one")
                    .segments
                    .push(segment(parameters)?);
            }
            None => unreachable!("the loop runs while bytes remain"),
        }
        scanner.at += 1;
    }
    if streams.iter().any(|stream| stream.segments.is_empty()) {
        return Err(EdlError::NoSegments);
    }
    Ok(Edl { streams })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seconds(count: u64) -> u64 {
        count * 1_000_000_000
    }

    fn segment(file: &str, start_ns: u64, length_ns: Option<u64>) -> EdlSegment {
        EdlSegment {
            file: file.to_string(),
            start_ns,
            length_ns,
        }
    }

    fn only_stream(uri: &str) -> Vec<EdlSegment> {
        let mut edl = parse_edl_uri(uri).unwrap();
        assert_eq!(edl.streams.len(), 1);
        edl.streams.remove(0).segments
    }

    #[test]
    fn reads_mpv_uri_example() {
        assert_eq!(
            only_stream("edl://f1.mkv,length=5,start=10;f2.mkv,30,20;f3.mkv"),
            [
                segment("f1.mkv", seconds(10), Some(seconds(5))),
                segment("f2.mkv", seconds(30), Some(seconds(20))),
                segment("f3.mkv", 0, None),
            ]
        );
    }

    #[test]
    fn length_prefixed_values_hold_reserved_characters() {
        // the example in mpv's edl-mpv.rst, with the parameters it does not support left out
        assert_eq!(
            only_stream("edl://%18%filename,with,.mkv,10,length=20"),
            [segment(
                "filename,with,.mkv",
                seconds(10),
                Some(seconds(20))
            )]
        );
        let path = "/dcp/reel two!;a.mxf";
        assert_eq!(
            only_stream(&alloc::format!("edl://%{}%{path}", path.len())),
            [segment(path, 0, None)]
        );
    }

    #[test]
    fn length_prefixes_count_bytes() {
        let path = "/dcp/café.mxf";
        assert_eq!(
            only_stream(&alloc::format!("edl://%{}%{path},0.5,1.5", path.len())),
            [segment(path, 500_000_000, Some(1_500_000_000))]
        );
    }

    #[test]
    fn new_stream_starts_a_separate_track_set() {
        let edl = parse_edl_uri(
            "edl://!new_stream;video.mkv;b.mkv;!new_stream;!no_clip;%9%audio.wav,0,1.5",
        )
        .unwrap();
        assert_eq!(
            edl.streams,
            [
                EdlStream {
                    segments: alloc::vec![segment("video.mkv", 0, None), segment("b.mkv", 0, None)],
                    no_clip: false,
                },
                EdlStream {
                    segments: alloc::vec![segment("audio.wav", 0, Some(1_500_000_000))],
                    no_clip: true,
                },
            ]
        );
    }

    #[test]
    fn line_breaks_and_comments_separate_entries() {
        assert_eq!(
            only_stream("edl://# mpv EDL v0\nf1.mkv,10,20\n\nf2.mkv"),
            [
                segment("f1.mkv", seconds(10), Some(seconds(20))),
                segment("f2.mkv", 0, None),
            ]
        );
    }

    #[test]
    fn frame_times_round_to_the_nearest_nanosecond() {
        // postkit writes the start of frame 25 at 24 fps as 25/24 s
        assert_eq!(
            only_stream("edl://a.mov,1.0416666666666667,0.5")[0].start_ns,
            1_041_666_667
        );
    }

    #[test]
    fn refuses_what_it_cannot_play() {
        for (uri, error) in [
            ("file.mkv", EdlError::NotAnEdlUri),
            ("edl://f.mkv,1,2,3", EdlError::TooManyValues),
            (
                "edl://f.mkv,title=x",
                EdlError::UnknownParameter("title".to_string()),
            ),
            (
                "edl://f.mkv,start=1,start=2",
                EdlError::RepeatedParameter("start".to_string()),
            ),
            (
                "edl://!global_tags,title=x;f.mkv",
                EdlError::UnknownHeader("global_tags".to_string()),
            ),
            ("edl://,1,2", EdlError::NoFile),
            (
                "edl://f.mkv,-1",
                EdlError::Time {
                    parameter: "start",
                    value: "-1".to_string(),
                },
            ),
            (
                "edl://f.mkv,0,nan",
                EdlError::Time {
                    parameter: "length",
                    value: "nan".to_string(),
                },
            ),
            ("edl://%40%short", EdlError::LengthPrefix { at: 0 }),
            (
                "edl://a!b.mkv",
                EdlError::ReservedCharacter {
                    character: '!',
                    at: 1,
                },
            ),
            ("edl://f.mkv;!new_stream", EdlError::NoSegments),
            ("edl://", EdlError::NoSegments),
        ] {
            assert_eq!(parse_edl_uri(uri), Err(error), "{uri}");
        }
    }
}
