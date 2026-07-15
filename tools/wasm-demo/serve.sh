#!/usr/bin/env bash
# Serve tools/wasm-demo over HTTP so the browser can load the ES module + .wasm.
# No special headers needed: the g2g pipeline is single-threaded, so unlike
# thread-based wasm (gst.wasm) this needs NO SharedArrayBuffer / COOP+COEP.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
port="${1:-8000}"
echo "serving $here at http://127.0.0.1:$port/  (open index.html there)"
exec python3 -m http.server "$port" --bind 127.0.0.1
