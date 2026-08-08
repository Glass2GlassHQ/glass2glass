# g2g-bridge

C-FFI bridge that embeds a
[glass2glass](https://github.com/boxerab/glass2glass) sub-graph inside a legacy
GStreamer pipeline. The `gstreamer` feature builds `libgstglass2glass.so`, a
GStreamer-loadable element wrapping the graph.
