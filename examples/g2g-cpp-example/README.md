# g2g-cpp-example

A C++ program that drives glass2glass through [`g2g.hpp`](../../g2g-capi/include/g2g.hpp), the header-only C++17 wrapper over the [`g2g-capi`](../../g2g-capi) C ABI. It round-trips three frames through `appsrc ! appsink`, then edits a running pipeline. It throws, and exits non-zero, on any frame or edit it does not get back.

```sh
make run
```

`make` builds the `g2g-capi` static library with cargo, then links `main.cpp` against it.

In `g2g.hpp`, handles free themselves and a refused call throws `g2g::Error` with the C ABI's reason, such as `UnknownNode`. `Pipeline::wait` throws when the run ends in an error, and the error itself is on the bus. Destroying a `Pipeline` blocks until it reaches end of stream. Create an `AppSrc` or `AppSink` before the pipeline that names its channel. The zero-copy `appsrc` lend and the per-frame `appsink` callback have no C++ wrapper yet.
