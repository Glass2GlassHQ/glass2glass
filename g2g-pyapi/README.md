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

## Changing a running pipeline

A `Pipeline` accepts edits while it runs. Elements are addressed by their
instance name: the `name=` from the launch line, or the name an insert returned.
A new element is written as one launch-line element, `"videoflip method=rotate-180"`,
without `name=` or `log-category=`.

```python
p = g2g.Pipeline("videotestsrc name=src ! clocksync ! fakesink name=sink")
gate = p.insert_after("src", "valve drop=false")
p.set_property(gate, "drop", "true")
print(p.get_property(gate, "drop"))   # "true"
p.remove(gate)
```

| Method | Does |
| :--- | :--- |
| `insert_after(node, description)` | Splices the element onto the edge below `node`, returns its name. |
| `insert_before(node, description)` | Splices the element onto the edge above `node`, returns its name. |
| `insert_chain_after(node, description)` | Splices a chain such as `"identity ! valve"` below `node`, all of it or none, returns the names. |
| `insert_chain_before(node, description)` | The same, above `node`. |
| `remove(node)` | Lifts a transform out of the pipeline and drops it. |
| `add_branch(tee, description)` | Attaches a chain under the tee, its last element the branch's sink, returns the names. |
| `remove_branch(sink)` | Drops the branch ending at `sink` from under its tee. |
| `add_input(muxer, description)` | Attaches a source to the muxer's next free pad, returns its name. |
| `replace_sink(node, description)` | Swaps the sink `node` for a new one, returns the new name. |
| `replace_source(node, description)` | Swaps the source `node` for a new one, returns the new name. |
| `set_property(node, name, value)` | Sets a property from its launch-line text. |
| `get_property(node, name)` | The property's launch-line text, `None` if the element has no such property. |

A refused edit raises `ValueError` carrying the reason (`UnknownNode`,
`NotMutable`, `PropertyRejected`, ...) and leaves the pipeline running as it
was. Each call blocks, with the GIL released, until the element upstream of the
edit pushes its next packet, and raises `TimedOut` after 5 s. An element whose
output format the pipeline below refuses gets a `videoconvert` or `audioconvert`
spliced in after it. A launch line gives a muxer one pad per linked input, so
`add_input` only finds a free pad on a muxer with fixed pads, such as
`cccombiner` fed only video.

## Testing

The in-crate tests drive the bindings through an embedded interpreter, so they
need libpython at build time:

```sh
cargo test -p g2g-pyapi --features python
```
