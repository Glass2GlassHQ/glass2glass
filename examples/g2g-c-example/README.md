# g2g-c-example

A minimal C program that drives a glass2glass pipeline through the
[`g2g-capi`](../../g2g-capi) C ABI: it builds an `appsrc ! appsink` pipeline,
pushes a few synthetic RGBA frames, pulls them back zero-copy, polls the bus,
and waits for the final frame counters.

This is the C counterpart of the Python example in
[`../../g2g-pyapi/examples/appsink_numpy.py`](../../g2g-pyapi/examples/appsink_numpy.py):
both embed g2g through the same language-neutral waist.

## Build & run

```sh
make run
```

`make` first builds the `g2g-capi` static library at the workspace root
(`cargo build --release -p g2g-capi`), then compiles [`main.c`](main.c) against
[`g2g-capi/include/g2g.h`](../../g2g-capi/include/g2g.h) and links
`libg2g_capi.a` plus the system libraries a Rust staticlib needs on Linux
(`-lpthread -ldl -lm`). Expected output:

```
frame 0: 16 bytes, first=0x01, pts=0 ns
frame 1: 16 bytes, first=0x02, pts=33333333 ns
frame 2: 16 bytes, first=0x03, pts=66666666 ns
bus: kind=0
...
done rc=0: emitted=3 consumed=3 dropped=0 (pulled 3)
```

This directory is not a Cargo workspace member; it is a standalone C consumer of
the published C ABI, built with `make`, not `cargo`.
