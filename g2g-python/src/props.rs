//! The property face shared by every hosted gst-python-ml element.
//!
//! A `pyelement` and a `pyaggregator` host the same family of Python classes, so
//! they accept the same tunables and forward them the same way. One list here so
//! adding a property to one host cannot leave the other behind, plus the parsing
//! for the caps-valued properties both hosts read themselves rather than forward.

use g2g_core::{Caps, CapsSet, PropError, PropValue};

/// What a hosted element's undeclared properties are, for a `gst-inspect` dump.
pub(crate) const FORWARDED_BLURB: &str =
    "any other property is forwarded to the hosted Python class, which declares \
     the real set (model-name, engine-name, device, ...); a name it does not \
     declare fails when the pipeline starts";

/// The one concrete caps a `input-caps=` / `output-caps=` description names.
/// A description that names a set (a `{a,b}` list, an unbounded geometry) has no
/// single answer, so it is rejected rather than silently resolved.
pub(crate) fn fixed_caps(desc: &str) -> Result<Caps, PropError> {
    CapsSet::from_gst_string(desc)
        .and_then(|set| set.fixate())
        .ok_or(PropError::Value)
}

/// Record a property the host does not read itself, to forward to the hosted
/// Python instance. Re-setting one replaces it, and the order they were set in
/// is the order they are applied.
pub(crate) fn forward(params: &mut Vec<(String, PropValue)>, name: &str, value: PropValue) {
    match params.iter_mut().find(|(key, _)| key == name) {
        Some(slot) => slot.1 = value,
        None => params.push((name.to_string(), value)),
    }
}

/// Build a host element's property list: its own entries, then the marker saying
/// every other name goes to the hosted class. `PropertySpec` resolves at the
/// call site, which already imports it.
macro_rules! hosted_element_props {
    ($($own:expr),+ $(,)?) => {
        &[
            $($own,)+
            PropertySpec::undeclared(crate::props::FORWARDED_BLURB),
        ]
    };
}

pub(crate) use hosted_element_props;
