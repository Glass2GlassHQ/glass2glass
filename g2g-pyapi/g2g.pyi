"""Type stubs for the `g2g` extension (g2g-pyapi).

Drive glass2glass pipelines from Python: parse a gst-launch-style string, run it
on a background thread, watch the bus, and push/pull buffers via appsrc/appsink.
"""

from typing import Optional, Tuple

class FrameView:
    """A pulled sample, lent zero-copy through the buffer protocol.

    Owns the underlying frame, so the bytes stay valid for the life of the view
    and of any ``memoryview`` / numpy array over it. Wrap with
    ``memoryview(view)`` or ``numpy.frombuffer(view, numpy.uint8)``; the bytes
    are read-only.
    """

    pts_ns: int
    """Presentation timestamp, in nanoseconds."""
    nbytes: int
    """Number of host-visible bytes (0 for a non-host / GPU domain)."""

    def __buffer__(self, flags: int) -> memoryview: ...

class AppSrc:
    """Application push source feeding ``appsrc channel=<name>``."""

    def __init__(self, channel: str = ...) -> None: ...
    def push(self, data: bytes, pts_ns: int = ...) -> bool:
        """Push a buffer (copied) with timestamp ``pts_ns``. False if the feed
        is full (retry) or the pipeline is gone."""
        ...
    def end_of_stream(self) -> bool:
        """Signal end-of-stream."""
        ...

class AppSink:
    """Application pull sink draining ``appsink channel=<name>``."""

    def __init__(self, channel: str = ...) -> None: ...
    def pull(self, timeout_ms: Optional[int] = ...) -> Optional[FrameView]:
        """Block for the next sample, returning a zero-copy ``FrameView`` or
        ``None`` once the stream ends. With ``timeout_ms``, returns ``None`` on
        timeout; pair with ``Pipeline.is_done()`` to tell a timeout from EOS."""
        ...
    def try_pull(self) -> Optional[FrameView]:
        """Non-blocking: a ``FrameView`` if a sample is ready, else ``None``."""
        ...

class Pipeline:
    """A running pipeline parsed from a gst-launch-style string."""

    def __init__(self, description: str) -> None: ...
    def bus_poll(self) -> Optional[Tuple[str, Optional[str], int, int]]:
        """Pop one bus message as ``(kind, text_or_None, a, b)``, or ``None`` if
        the bus is empty."""
        ...
    def is_done(self) -> bool:
        """True once the run thread has finished (EOS or error)."""
        ...
    def wait(self) -> Tuple[int, int, int]:
        """Block until the pipeline ends; returns ``(emitted, consumed,
        dropped)``. Raises on a pipeline error."""
        ...
