//! The MoQ catalog document (draft-ietf-moq-catalogformat-01), written by the
//! publisher and read by the subscriber.
//!
//! The shape is `moq-rs/moq-catalog`'s `Root`: a version, common track fields,
//! and a `tracks` array whose entries name a track and the init track carrying
//! its `ftyp`+`moov`. Both halves are here so they cannot drift, and so neither
//! side needs a JSON dependency: the document is small, its keys are fixed, and
//! the parser only pulls the two string fields a player has to have.
//!
//! The reader takes bytes from a relay, so it never trusts the document: the
//! whole thing is size-bounded before parsing, and a truncated or nested-string
//! oddity yields no tracks rather than an error the element cannot act on.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Largest catalog document the subscriber will parse. The reference catalog
/// for a two-track broadcast is well under a kilobyte; this leaves room for a
/// many-track one and bounds a relay that sends a huge object on the catalog
/// track.
pub const MAX_CATALOG_BYTES: usize = 256 * 1024;

/// One entry of the catalog's `tracks` array, reduced to what a subscriber
/// selects on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogTrack {
    pub name: String,
    /// The `initTrack` for this track, empty when the entry omits it (the
    /// subscriber then falls back to its `init-track-name` property).
    pub init_track: String,
}

/// Build the catalog for a broadcast. `tracks` pairs each track name with its
/// `selectionParams` fragment (a JSON object body, empty when the codec is not
/// one we can describe).
pub fn build(namespace: &str, init_track: &str, tracks: &[(String, String)]) -> String {
    let mut entries = String::new();
    for (i, (name, selection_params)) in tracks.iter().enumerate() {
        if i > 0 {
            entries.push(',');
        }
        entries.push_str(&format!(
            "{{\"name\":\"{name}\",\"initTrack\":\"{init_track}\"{selection_params}}}"
        ));
    }
    format!(
        concat!(
            "{{\"version\":1,\"streamingFormat\":1,\"streamingFormatVersion\":\"0.2\",",
            "\"supportsDeltaUpdates\":true,",
            "\"commonTrackFields\":{{\"namespace\":\"{}\",\"packaging\":\"cmaf\",\"renderGroup\":1}},",
            "\"tracks\":[{}]}}"
        ),
        namespace, entries
    )
}

/// Read the `tracks` array: each entry's `name` and `initTrack`, in document
/// order. Returns nothing when the document is over the size bound, is not
/// UTF-8, or has no usable entry.
pub fn parse(bytes: &[u8]) -> Vec<CatalogTrack> {
    if bytes.len() > MAX_CATALOG_BYTES {
        return Vec::new();
    }
    let Ok(text) = core::str::from_utf8(bytes) else {
        return Vec::new();
    };
    let Some(array) = tracks_array(text) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in split_objects(array) {
        let Some(name) = string_field(entry, "name") else {
            continue;
        };
        out.push(CatalogTrack {
            name,
            init_track: string_field(entry, "initTrack").unwrap_or_default(),
        });
    }
    out
}

/// The body of the top-level `"tracks": [ ... ]` array, brace-balanced so a
/// nested array inside a track entry does not truncate it.
fn tracks_array(text: &str) -> Option<&str> {
    let at = text.find("\"tracks\"")? + "\"tracks\"".len();
    let rest = text.get(at..)?;
    let open = rest.find('[')?;
    let body = rest.get(open + 1..)?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in body.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '[' | '{' => depth += 1,
            '}' => depth -= 1,
            ']' if depth == 0 => return body.get(..i),
            ']' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Split an array body into its top-level `{...}` objects.
fn split_objects(body: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in body.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => {
                if depth == 0 {
                    start = i;
                }
                depth += 1;
            }
            '}' => {
                depth -= 1;
                if depth == 0 {
                    if let Some(entry) = body.get(start..=i) {
                        out.push(entry);
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// The string value of `"key": "value"` at this object's top level. Only the
/// two escapes a track name can plausibly carry are decoded; anything else is
/// left as written, which is enough to match a name against a track list.
fn string_field(object: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let mut at = 0usize;
    while let Some(found) = object.get(at..)?.find(&needle) {
        let after = at + found + needle.len();
        at = after;
        let rest = object.get(after..)?.trim_start();
        let Some(rest) = rest.strip_prefix(':') else {
            continue; // the key appeared as a value, not a key
        };
        let rest = rest.trim_start();
        let Some(rest) = rest.strip_prefix('"') else {
            continue; // not a string value
        };
        let mut value = String::new();
        let mut escaped = false;
        for c in rest.chars() {
            if escaped {
                value.push(match c {
                    'n' => '\n',
                    't' => '\t',
                    other => other,
                });
                escaped = false;
                continue;
            }
            match c {
                '\\' => escaped = true,
                '"' => return Some(value),
                _ => value.push(c),
            }
        }
        // An unterminated string: the document is truncated.
        return None;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    fn track(name: &str, params: &str) -> (String, String) {
        (name.to_string(), params.to_string())
    }

    #[test]
    fn a_catalog_we_wrote_reads_back_as_the_tracks_we_named() {
        let doc = build(
            "/live/cam",
            "0.mp4",
            &[
                track(
                    "1.m4s",
                    ",\"selectionParams\":{\"codec\":\"avc1.64000D\",\"width\":320,\"height\":240}",
                ),
                track("2.m4s", ""),
            ],
        );
        assert!(doc.contains("\"namespace\":\"/live/cam\""), "{doc}");
        assert_eq!(
            parse(doc.as_bytes()),
            vec![
                CatalogTrack {
                    name: "1.m4s".to_string(),
                    init_track: "0.mp4".to_string(),
                },
                CatalogTrack {
                    name: "2.m4s".to_string(),
                    init_track: "0.mp4".to_string(),
                },
            ]
        );
    }

    /// The layout `moq-pub` writes: `serde_json::to_string_pretty` of
    /// `moq_catalog::Root`, so whitespace everywhere and the common fields
    /// hoisted out of the track entries.
    #[test]
    fn the_reference_publishers_pretty_printed_catalog_parses() {
        let doc = r#"{
  "version": 1,
  "streamingFormat": 1,
  "streamingFormatVersion": "0.2",
  "supportsDeltaUpdates": true,
  "commonTrackFields": {
    "namespace": "/g2gtest",
    "packaging": "cmaf",
    "renderGroup": 1
  },
  "tracks": [
    {
      "name": "1.m4s",
      "initTrack": "0.mp4",
      "selectionParams": {
        "codec": "avc1.64000d",
        "framerate": 24,
        "bitrate": 1500000,
        "width": 64,
        "height": 48
      }
    },
    {
      "name": "2.m4s",
      "initTrack": "0.mp4",
      "selectionParams": {
        "codec": "mp4a.40.2",
        "samplerate": 48000,
        "channelConfig": "2"
      }
    }
  ]
}"#;
        let tracks = parse(doc.as_bytes());
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].name, "1.m4s");
        assert_eq!(tracks[0].init_track, "0.mp4");
        assert_eq!(tracks[1].name, "2.m4s");
    }

    #[test]
    fn a_hostile_or_truncated_catalog_yields_no_tracks_instead_of_panicking() {
        assert!(parse(b"").is_empty());
        assert!(parse(b"not json at all").is_empty());
        assert!(parse(&[0xff, 0xfe, 0xfd]).is_empty(), "not utf-8");
        // The array opens and never closes.
        assert!(parse(br#"{"tracks":[{"name":"a.m4s""#).is_empty());
        // A name string that never terminates.
        assert!(parse(br#"{"tracks":[{"name":"a.m4s}]}"#).is_empty());
        // A document past the size bound is refused without scanning it.
        let huge = vec![b'{'; MAX_CATALOG_BYTES + 1];
        assert!(parse(&huge).is_empty());
        // An entry with no name is skipped, the rest still read.
        assert_eq!(
            parse(br#"{"tracks":[{"initTrack":"0.mp4"},{"name":"b.m4s"}]}"#),
            vec![CatalogTrack {
                name: "b.m4s".to_string(),
                init_track: String::new(),
            }]
        );
        // A nested array inside an entry does not end the track list early.
        assert_eq!(
            parse(br#"{"tracks":[{"name":"a","depends":["x","]"]},{"name":"b"}]}"#).len(),
            2
        );
    }
}
