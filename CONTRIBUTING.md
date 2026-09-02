# Contributing

Issues and pull requests are welcome at
<https://github.com/Glass2GlassHQ/glass2glass/issues>.

## Build and test

Stable Rust. MSRV is 1.92, except `g2g-core`, `g2g-mcu`, `g2g-mcugen` and
`g2g-plugin`, which build on 1.86 (see `STABILITY.md`).

```sh
cargo check --workspace                 # default (no_std) build
cargo test  --workspace                 # default test suite
cargo clippy --workspace --all-targets  # lints, CI treats warnings as errors
cargo fmt --all --check                 # CI enforces formatting
```

Feature-gated elements need their features passed explicitly:

```sh
cargo test -p g2g-plugins --features rtsp
```

A feature-gated test file compiles to zero tests without its feature, so check
the reported test count is nonzero.

The stream parsers have libFuzzer targets (run weekly in CI):

```sh
cd g2g-plugins/fuzz && cargo +nightly fuzz list
cargo +nightly fuzz run h264parse
```

## Conventions

- `g2g-core` and `g2g-plugins` are `no_std` + `alloc` at the baseline. Anything
  needing the OS goes behind a cargo feature that implies `std`.
- Elements implement `AsyncElement` or `SourceLoop`. [AUTHORING.md](AUTHORING.md)
  walks the traits, the negotiation lifecycle and registration. Study
  `g2g-plugins/src/h264parse.rs` and `rtspsrc.rs` as references.
- Every `with_*` builder knob is also a runtime property (`properties()` +
  `set_property` / `get_property`), named after the matching GStreamer property
  where one exists.
- Parsers and demuxers must not trust the stream: validate counts, lengths and
  offsets with checked or saturating arithmetic so malformed input fails the
  parse instead of panicking.
- Tests exercise the real unit and mock only external boundaries (network, COM,
  GPU). Every test needs an assertion that fails if the feature breaks.

## License

MPL-2.0 for the whole repository. By contributing you agree your work is
licensed the same way.
