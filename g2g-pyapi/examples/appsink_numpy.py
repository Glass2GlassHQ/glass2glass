#!/usr/bin/env python3
"""glass2glass Python example: drive a pipeline and pull frames zero-copy.

Synthesizes a few solid-color RGBA frames with numpy, feeds them through an
``appsrc ! appsink`` pipeline, pulls them back as a zero-copy ``FrameView``, and
wraps each in a numpy array that *shares* the pipeline's buffer (no copy). This
is the smallest realistic shape of a Python application embedding g2g through
the same language-neutral waist as the C ABI.

Build the wheel and run it::

    cd g2g-pyapi
    pip install maturin numpy
    maturin develop --release       # builds + installs `g2g` into the venv
    python examples/appsink_numpy.py
"""

import g2g
import numpy as np

WIDTH, HEIGHT = 2, 2
NUM_FRAMES = 3
FRAME_BYTES = WIDTH * HEIGHT * 4  # RGBA


def main() -> int:
    # Register the endpoints before launching: the pipeline binds its
    # appsrc/appsink to these channels by name at parse time.
    src = g2g.AppSrc("in")
    sink = g2g.AppSink("out")
    p = g2g.Pipeline(
        "appsrc channel=in caps=video/x-raw,format=RGBA,"
        "width=2,height=2,framerate=30/1 ! appsink channel=out"
    )

    # Push NUM_FRAMES solid-color frames built with numpy, 30 fps PTS step.
    for i in range(NUM_FRAMES):
        frame = np.full(FRAME_BYTES, i + 1, dtype=np.uint8)
        pts_ns = i * 33_333_333
        while not src.push(frame.tobytes(), pts_ns):
            pass  # feed full: a real app would yield
    src.end_of_stream()

    # Pull every frame back. `pull()` returns a FrameView lent through the
    # buffer protocol; np.frombuffer over it shares the bytes (OWNDATA False),
    # so there is no copy at the language boundary.
    pulled = 0
    while True:
        view = sink.pull(timeout_ms=1000)
        if view is None:
            if p.is_done():
                break  # real end of stream
            continue   # just a timeout, keep polling
        arr = np.frombuffer(view, dtype=np.uint8)
        assert arr.flags["OWNDATA"] is False, "expected a zero-copy view, not a copy"
        print(f"frame {pulled}: {view.nbytes} bytes, first={arr[0]:#04x}, "
              f"pts={view.pts_ns} ns, shares_buffer={not arr.flags['OWNDATA']}")
        pulled += 1

    # Drain the bus, then wait for the final frame counters.
    while (msg := p.bus_poll()) is not None:
        kind, text, a, b = msg
        print(f"bus: {kind}{' ' + text if text else ''}")
    emitted, consumed, dropped = p.wait()
    print(f"done: emitted={emitted} consumed={consumed} dropped={dropped} "
          f"(pulled {pulled})")

    return 0 if consumed == NUM_FRAMES and pulled == NUM_FRAMES else 1


if __name__ == "__main__":
    raise SystemExit(main())
