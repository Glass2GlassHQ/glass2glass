// Dedicated module worker that runs a whole g2g graph off the main thread
// (M1054). One wasm instance per worker, the same single-threaded executor, no
// SharedArrayBuffer and no cross-origin isolation.
//
// The page posts {mode, url, canvas} once, with the OffscreenCanvas in the
// transfer list; everything after that runs here. console output is mirrored
// back to the page so the demo log shows the pipeline's own lines.
import init, { run_worker_graph } from "./pkg/g2g_web.js";

for (const level of ["log", "error", "warn"]) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    original(...args);
    self.postMessage({ log: args.map(String).join(" ") });
  };
}

let target = null;

self.onmessage = async (e) => {
  // Read back what the 2D sink drew, the only way to see the transferred
  // canvas from outside the worker (headless/run-worker.mjs asserts on it).
  if (e.data.probe) {
    const d = target.getContext("2d").getImageData(0, 0, target.width, target.height).data;
    const colors = new Set();
    for (let i = 0; i < d.length; i += 4) colors.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    self.postMessage({ probe: { colors: [...colors] } });
    return;
  }
  if (target) return;
  const { mode, url, canvas } = e.data;
  target = canvas;
  await init();
  console.log("g2g[worker]: module initialized, mode=" + mode + " url=" + url);
  run_worker_graph(mode, url, canvas);
};
