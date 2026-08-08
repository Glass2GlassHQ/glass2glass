# g2g-plugins

Standard source, sink, and transform elements for
[glass2glass](https://github.com/boxerab/glass2glass): capture, network
transports, container mux/demux, parsers, codecs, display sinks, and the
`gst-launch` text DSL. `no_std + alloc` baseline; every OS-coupled element sits
behind a cargo feature that implies `std`.
