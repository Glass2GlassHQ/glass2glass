# g2g-pyapi

Python bindings to **drive** glass2glass pipelines: parse a `gst-launch`-style
string, run it on a background thread, watch the bus, and push/pull buffers via
`appsrc` / `appsink`. This is the inverse of `g2g-python`, which *hosts* Python
elements inside a pipeline; both sit on the same language-neutral waist as
`g2g-capi`.

The compiled extension imports as `g2g` (the `#[pymodule] fn g2g`) and exposes
`Pipeline`, `AppSrc`, `AppSink`, and `FrameView`.

## Building the wheel (maturin)

The crate ships a `pyproject.toml` with the [maturin](https://www.maturin.rs)
backend. From this directory, in a virtualenv:

```sh
pip install maturin numpy

# Build + install into the active venv (fastest dev loop):
maturin develop --release

# Or build a redistributable wheel:
maturin build --release         # -> ../target/wheels/g2g-*.whl
pip install ../target/wheels/g2g-*.whl
```

`pyproject.toml` enables the `python` feature plus `pyo3/extension-module`, so
the wheel is loaded by an existing interpreter (it does not embed/link
libpython, the inverse of the auto-initialize path the in-crate `cargo test`
suite uses).

## Example

```python
import g2g
import numpy as np

src  = g2g.AppSrc("in")
sink = g2g.AppSink("out")
p = g2g.Pipeline(
    "appsrc channel=in caps=video/x-raw,format=RGBA,"
    "width=2,height=2,framerate=30/1 ! appsink channel=out"
)

src.push(np.full(16, 1, np.uint8).tobytes(), 0)
src.end_of_stream()

view = sink.pull()                      # a zero-copy FrameView, None at EOS
arr  = np.frombuffer(view, np.uint8)    # shares the buffer (OWNDATA False)
print(arr, view.pts_ns)
p.wait()
```

`AppSink.pull()` lends the frame through the buffer protocol: the `FrameView`
owns the `Frame`, so the bytes (and any `memoryview` / numpy array over them)
stay valid past later pulls and pipeline teardown. Pass `timeout_ms` for a
bounded blocking pull; pair it with `Pipeline.is_done()` to tell a timeout from
a real end of stream.

A runnable version is in [`examples/appsink_numpy.py`](examples/appsink_numpy.py):

```sh
maturin develop --release
python examples/appsink_numpy.py
```

The C equivalent (same waist, via `g2g-capi`) is in
[`../examples/g2g-c-example`](../examples/g2g-c-example).

## Testing

The in-crate tests drive the bindings through an embedded interpreter, so they
need libpython at build time:

```sh
cargo test -p g2g-pyapi --features python
```
