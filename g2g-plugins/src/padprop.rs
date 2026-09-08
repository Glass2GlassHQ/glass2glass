//! The `sinkN-<knob>` naming a fan-in flattens its per-pad properties into.
//! [`properties`](g2g_core::MultiInputElement::properties) is a `&'static`
//! table, so a request pad's own property (gst's `sink_1::priority`) has no
//! place to live; each pad's knobs are spelled into the element's table
//! instead, which means the pad indices are fixed at compile time. The tables
//! that use this cover pads 0..=7.

/// Split a `sinkN-<knob>` property name into the pad index and the knob.
pub(crate) fn split_pad_name(name: &str) -> Option<(usize, &str)> {
    let (index, knob) = name.strip_prefix("sink")?.split_once('-')?;
    Some((index.parse().ok()?, knob))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pad_name_splits_into_index_and_knob() {
        assert_eq!(split_pad_name("sink3-xpos"), Some((3, "xpos")));
        assert_eq!(split_pad_name("sink0-priority"), Some((0, "priority")));
    }

    #[test]
    fn an_element_level_name_is_not_a_pad_name() {
        assert_eq!(split_pad_name("timeout"), None);
        assert_eq!(split_pad_name("sink-priority"), None);
        assert_eq!(split_pad_name("sinkx-priority"), None);
    }
}
