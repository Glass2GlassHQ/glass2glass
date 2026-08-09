# Hosted element for the M988 GIL-offload measurement: a deliberately
# compute-bound *pure Python* frame callback, so the work cannot leave the
# interpreter (numpy or torch would release the GIL and prove nothing).
#
# It also records, from inside the call, whether the GIL is enabled after the
# native `g2g` module has been imported: a module that does not declare
# `gil_used = false` makes CPython re-enable the GIL process-wide on import, which
# is the failure this measurement would otherwise silently absorb.

import sys

import g2g  # noqa: F401 - imported for its effect on the free-threaded interpreter

# Tuned so one call is long enough to time (~80 ms) and short enough that a
# handful of them stay quick.
SPIN_ITERATIONS = 2_000_000

# Filled by the first call: whether the interpreter still has the GIL off.
STATE = {}


def gil_enabled():
    """True on a stock interpreter; False on a free-threaded one that has stayed
    free-threaded. Pre-3.13 interpreters have no such query and always have it."""
    query = getattr(sys, "_is_gil_enabled", None)
    return True if query is None else query()


class SpinTransform:
    """Burns CPU inside the interpreter, then writes the result into the frame so
    the work cannot be optimized away."""

    def g2g_process(self, buf, width, height, fmt, meta):
        STATE["gil_enabled"] = gil_enabled()
        STATE["version"] = sys.version
        total = 0
        for i in range(SPIN_ITERATIONS):
            total += i * i
        memoryview(buf)[0] = total % 256
