# g2g-bridge

C-FFI bridge that embeds a
[glass2glass](https://github.com/boxerab/glass2glass) sub-graph inside a legacy
GStreamer pipeline. The `gstreamer` feature builds `libgstglass2glass.so`, a
GStreamer-loadable element wrapping the graph.
The `wgpu` feature adds `wgpucompositor`, `wgpuupload` and `wgpudownload` to the
elements a `fragment` can use, and on Linux `dmabuftowgpu` and `wgputodmabuf`. Input caps
with the `memory:DMABuf` feature feed the fragment dma-buf frames only, so it
must start with `dmabuftowgpu`. Linear `drm-format` values only.

The `gst-analytics` feature carries analytics metadata across the element both
ways: a buffer's `GstAnalyticsRelationMeta` reaches the fragment as an
`AnalyticsMeta`, and a frame's `AnalyticsMeta` leaves as a
`GstAnalyticsRelationMeta`. It needs gstreamer-analytics-1.0 1.26 or newer, from
gst-plugins-bad. Detections, classifications, tracking ids and segmentation masks
are mapped. Region-of-interest nodes and keypoints are not.
