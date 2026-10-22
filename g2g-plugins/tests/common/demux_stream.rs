// the `stream` property of the demuxer a launch line plugs, the track a primary-stream hook selected

use g2g_core::property::PropValue;
use g2g_core::runtime::{parse_launch, GraphNodeRef};
use g2g_plugins::registry::default_registry;

pub(crate) fn demux_stream(line: &str, demux_category: &str) -> Option<PropValue> {
    let reg = default_registry();
    let graph = parse_launch(&reg, line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let vg = graph.finish().expect("valid graph");
    vg.topo().iter().find_map(|&n| match vg.element(n) {
        Some(GraphNodeRef::Element(e)) if e.log_category() == demux_category => {
            e.get_property("stream")
        }
        _ => None,
    })
}
