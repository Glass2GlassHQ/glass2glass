// ONNX Runtime Web shim for the g2g browser detection pipeline (Stage 2).
//
// Rust's `WebOrtDetect` element imports `ort_init` / `ort_run` from this module
// via wasm-bindgen. onnxruntime-web is loaded LAZILY from the CDN on the first
// `ort_init` (so merely loading the wasm module, which statically imports this
// file, costs nothing until detection actually starts). Single-threaded
// (`numThreads = 1`) so it needs NO SharedArrayBuffer / cross-origin-isolation
// (COOP/COEP) headers, matching the rest of the single-threaded g2g wasm
// pipeline. The g2g side owns preprocessing (RGBA -> NCHW f32) and postprocessing
// (YOLOv8 decode + NMS via g2g-ml's DetectionPostprocess); this shim owns only
// the ONNX session and `session.run`.
const ORT_VER = '1.20.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

let ort = null;
let session = null;

// Load onnxruntime-web (once) and create an inference session for `model_url`.
export async function ort_init(model_url) {
  if (!ort) {
    ort = await import(ORT_BASE + 'ort.mjs');
    ort.env.wasm.wasmPaths = ORT_BASE; // fetch ort-wasm-*.wasm from the CDN too
    ort.env.wasm.numThreads = 1;       // no SAB -> no cross-origin isolation
    ort.env.wasm.proxy = false;
  }
  session = await ort.InferenceSession.create(model_url, { executionProviders: ['wasm'] });
  console.log('g2g[ort]: session ready in=[' + session.inputNames + '] out=[' + session.outputNames + ']');
}

// Run one inference over an NCHW f32 input, returning the first output's flat
// data (Float32Array) and its dims (so g2g configures the decoder from the real
// output shape). `data` arrives from Rust as a Float32Array (a Vec<f32> copy).
export async function ort_run(data, n, c, h, w) {
  if (!session) throw new Error('ort_run before ort_init');
  const input = new ort.Tensor('float32', data, [n, c, h, w]);
  const feeds = {};
  feeds[session.inputNames[0]] = input;
  const results = await session.run(feeds);
  const out = results[session.outputNames[0]];
  return { data: out.data, dims: out.dims };
}
