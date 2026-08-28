//! The `tags` element property: gst's taglist syntax parsed into a [`TagList`].
//!
//! A [`TagList`] travels out of band on the bus
//! ([`BusMessage::Tag`](g2g_core::BusMessage::Tag)) and an element cannot read
//! the bus, so a tag writer ([`crate::id3v2mux`], [`crate::apev2mux`],
//! [`crate::vorbistag`], [`crate::flactag`]) has no way to pick up the tags gst
//! delivers to it as a tag event. It takes them from this property instead, in
//! the same syntax [`crate::taginject`] accepts.
//!
//! The syntax is `key=value` pairs separated by commas, with a value quoted when
//! it holds a space or a comma:
//! `tags="title=\"A Title\",artist=Someone"`. Keys map through
//! [`Tag::from_key_value`], so a key this crate has no typed variant for
//! survives as [`Tag::Other`].

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use g2g_core::{PropError, PropKind, PropValue, PropertySpec, Tag, TagList};

/// The `tags` property of an element that reads one: the parsed list plus the
/// text it was set from, so `get_property` round-trips what was written.
#[derive(Debug, Default)]
pub struct TagsProperty {
    tags: TagList,
    text: String,
}

impl TagsProperty {
    /// The spec an element puts in its `properties()` table.
    pub const SPEC: PropertySpec = PropertySpec::new(
        "tags",
        PropKind::Str,
        "tags to write, gst taglist syntax: e.g. title=A Title,artist=Someone",
    );

    /// The parsed tags, empty until the property is set.
    pub fn tags(&self) -> &TagList {
        &self.tags
    }

    /// Apply a `tags=` value, rejecting a pair with no `=`.
    pub fn set(&mut self, text: &str) -> Result<(), PropError> {
        self.tags = parse_taglist(text).ok_or(PropError::Value)?;
        self.text = text.to_string();
        Ok(())
    }

    /// The property's value as `get_property` reports it, `None` while unset.
    pub fn value(&self) -> Option<PropValue> {
        (!self.text.is_empty()).then(|| PropValue::Str(self.text.clone()))
    }
}

/// Parse gst's taglist syntax into a [`TagList`], or `None` when a pair has no
/// `=`. A value may be quoted, which is the only way to write one holding a
/// comma.
pub fn parse_taglist(text: &str) -> Option<TagList> {
    let mut tags = TagList::new();
    for pair in split_unquoted_commas(text) {
        let (key, value) = pair.split_once('=')?;
        tags.push(Tag::from_key_value(key.trim(), unquote(value.trim())));
    }
    Some(tags)
}

/// Split on the commas outside a quoted region, so a quoted value keeps its own.
fn split_unquoted_commas(text: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut quote: Option<char> = None;
    for (at, c) in text.char_indices() {
        match c {
            '"' | '\'' if quote.is_none() => quote = Some(c),
            _ if Some(c) == quote => quote = None,
            ',' if quote.is_none() => {
                parts.push(&text[start..at]);
                start = at + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

/// Drop one surrounding pair of quotes from a value.
fn unquote(value: &str) -> &str {
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|v| v.strip_suffix(quote))
        {
            return inner;
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_quoted_and_bare_pairs() {
        let tags = parse_taglist("title=\"A, Title\",artist=Someone").expect("parses");
        assert_eq!(
            tags.tags(),
            [Tag::Title("A, Title".into()), Tag::Artist("Someone".into())]
        );
        assert!(parse_taglist("title").is_none());
    }

    #[test]
    fn property_round_trips_its_text() {
        let mut prop = TagsProperty::default();
        assert!(prop.value().is_none());
        prop.set("album=Set").expect("a valid taglist");
        assert_eq!(prop.value(), Some(PropValue::Str("album=Set".into())));
        assert_eq!(prop.tags().tags(), [Tag::Album("Set".into())]);
        assert_eq!(prop.set("nope"), Err(PropError::Value));
    }
}
