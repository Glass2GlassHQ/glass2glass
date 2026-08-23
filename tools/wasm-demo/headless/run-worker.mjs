// Headless validation for the Web Worker executor (M1054).
//
// Drives WebSocketSrc -> WebCodecsDecode -> WebDetect -> AnalyticsOverlay ->
// CanvasSink entirely inside a dedicated module worker, presenting to an
// OffscreenCanvas the page transferred in, against the committed H.264 fixture
// streamed by ws-fixture-server (one AU/message).
//
// Asserts: the worker's wasm instance initializes off the main thread, the
// graph consumes frames and finishes ok, and the transferred canvas is neither
// blank nor missing the overlay's box color (read back inside the worker, the
// only place the canvas is reachable). SyntheticDetect plants one class-0
// detection per frame, so exactly that one box color must be on the canvas.
//
// Prereqs: `npm i -D playwright` (or G2G_PLAYWRIGHT pointing at an install) and
// a full (WebCodecs-capable) Chromium. Run from tools/wasm-demo:
//   node headless/run-worker.mjs
// Env overrides: G2G_CHROME, G2G_WS_SERVER_BIN, G2G_FIXTURE, G2G_MODE
// (worker graph: detect / canvas / webgpu), G2G_HEADFUL=1.
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join, extname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, ".."); // tools/wasm-demo, served as the web root
const HTTP_PORT = 8197;
const WS_PORT = 8196;
const FIXTURE = process.env.G2G_FIXTURE ||
  resolve(ROOT, "../../g2g-plugins/tests/fixtures/h264_640x480.h264");
const MODE = process.env.G2G_MODE || "detect";
const NEED_FRAMES = 5; // frames the worker's graph must consume
// AnalyticsOverlay's class-0 box color: the deterministic pixel SyntheticDetect's
// single planted detection puts on the canvas.
const CLASS_0_BOX_COLOR = 0xff3b30;
const TIMEOUT_MS = 60000;

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".onnx": "application/octet-stream",
  ".json": "application/json", ".css": "text/css",
};

function log(...a) { console.log("[harness]", ...a); }
function fail(msg) { console.error("[harness] FAIL:", msg); shutdown(1); }

let http, wsProc, browser;
function shutdown(code) {
  try { browser?.close(); } catch {}
  try { wsProc?.kill("SIGKILL"); } catch {}
  try { http?.close(); } catch {}
  process.exit(code);
}

// Static file server for tools/wasm-demo (pkg/, worker.js, headless/).
function startHttp() {
  return new Promise((res) => {
    http = createServer(async (req, resp) => {
      const path = decodeURIComponent(req.url.split("?")[0]);
      const file = join(ROOT, path === "/" ? "/index.html" : path);
      if (!file.startsWith(ROOT)) { resp.writeHead(403).end(); return; }
      try {
        const body = await readFile(file);
        resp.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream" });
        resp.end(body);
      } catch { resp.writeHead(404).end("not found"); }
    }).listen(HTTP_PORT, "127.0.0.1", () => res());
  });
}

function startWsServer() {
  const bin = process.env.G2G_WS_SERVER_BIN;
  const addr = `127.0.0.1:${WS_PORT}`;
  const [cmd, args] = bin && existsSync(bin)
    ? [bin, [addr, FIXTURE, "10"]]
    : ["cargo", ["run", "--release", "--manifest-path",
        resolve(ROOT, "ws-fixture-server/Cargo.toml"), "--", addr, FIXTURE, "10"]];
  log("ws server:", cmd, args.join(" "));
  wsProc = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
  wsProc.stdout.on("data", (d) => process.stdout.write("[ws] " + d));
  wsProc.stderr.on("data", (d) => process.stderr.write("[ws] " + d));
  return new Promise((res) => {
    const ready = (d) => { if (d.toString().includes("serving ws://")) { wsProc.stdout.off("data", ready); res(); } };
    wsProc.stdout.on("data", ready);
    setTimeout(res, bin ? 800 : 30000); // fallback: assume up
  });
}

async function main() {
  if (!existsSync(join(ROOT, "pkg/g2g_web.js"))) fail("pkg/g2g_web.js missing (run build.sh)");
  await startHttp();
  await startWsServer();
  log("http on", HTTP_PORT, "ws on", WS_PORT, "mode", MODE);

  const pw = process.env.G2G_PLAYWRIGHT
    ? await import(pathToFileURL(process.env.G2G_PLAYWRIGHT).href)
    : await import("playwright");
  const { chromium } = pw.default || pw;
  browser = await chromium.launch({
    headless: !process.env.G2G_HEADFUL,
    executablePath: process.env.G2G_CHROME || undefined,
    args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"],
  });
  const page = await browser.newPage();
  let workerInit = false, finishedOk = false, pipelineError = null;
  let framesConsumed = 0, probeColors = null;
  page.on("console", (m) => {
    const t = m.text();
    if (t.startsWith("g2g[")) log("page:", t);
    if (t.includes("module initialized")) workerInit = true;
    const fm = t.match(/frames_consumed: (\d+)/);
    if (fm) framesConsumed = Number(fm[1]);
    if (t.includes("finished ok")) finishedOk = true;
    if (t.includes("pipeline error") || t.includes("worker-error")) pipelineError = t;
    const pm = t.match(/g2g\[probe\]: (\{.*\})/);
    if (pm) probeColors = JSON.parse(pm[1]).colors;
  });
  page.on("pageerror", (e) => { pipelineError = String(e); });

  const url = `http://127.0.0.1:${HTTP_PORT}/headless/worker.html`
    + `?ws=${encodeURIComponent(`ws://127.0.0.1:${WS_PORT}`)}&mode=${MODE}`;
  log("navigating", url);
  await page.goto(url);

  const hasWebCodecs = await page.evaluate(() => typeof VideoDecoder !== "undefined");
  if (!hasWebCodecs) fail("browser lacks WebCodecs (use a full Chromium, not headless_shell)");
  // A headless Chromium on swiftshader exposes navigator.gpu but hands back no
  // adapter, which the sink can only report as a hardware error.
  const gpuAdapter = MODE !== "webgpu" || await page.evaluate(async () => {
    if (!navigator.gpu) return false;
    try { return !!(await navigator.gpu.requestAdapter({ powerPreference: "high-performance" })); }
    catch { return false; }
  });
  if (!gpuAdapter) {
    log("SKIP: this browser hands back no WebGPU adapter, so the webgpu graph cannot run");
    shutdown(0);
  }

  // Finite source: stop feeding after one fixture pass so the graph reaches EOS
  // and the runner reports its stats.
  setTimeout(() => { try { wsProc?.kill("SIGKILL"); } catch {} log("finite source: stopped ws feed"); }, 1100);

  const t0 = Date.now();
  while (Date.now() - t0 < TIMEOUT_MS) {
    if (pipelineError) fail("worker pipeline error: " + pipelineError);
    if (finishedOk) break;
    await page.waitForTimeout(200);
  }
  if (!workerInit) fail("the worker never initialized its wasm module");
  if (!finishedOk) fail(`worker graph did not finish (frames=${framesConsumed})`);
  if (framesConsumed < NEED_FRAMES) fail(`only ${framesConsumed}/${NEED_FRAMES} frames consumed in the worker`);

  if (MODE === "webgpu") {
    log(`PASS: worker ran the webgpu graph off-thread, ${framesConsumed} frames consumed, finished ok`);
    shutdown(0);
  }

  // Read the transferred canvas back inside the worker.
  await page.evaluate(() => self.probeWorker());
  const t1 = Date.now();
  while (!probeColors && Date.now() - t1 < 5000) await page.waitForTimeout(100);
  if (!probeColors) fail("worker never answered the canvas probe");

  if (probeColors.length < 2) fail("the worker's canvas is blank (one color)");
  // Only the detect graph draws a box; the plain canvas graph presents the
  // decoded video alone.
  if (MODE === "detect" && !probeColors.includes(CLASS_0_BOX_COLOR)) {
    fail(`overlay box not on the worker's canvas (${probeColors.length} colors)`);
  }

  log(`PASS: worker ran the ${MODE} graph off-thread, ${framesConsumed} frames consumed,`
    + ` finished ok, ${probeColors.length} colors on the transferred canvas`);
  shutdown(0);
}

setTimeout(() => fail("overall timeout"), TIMEOUT_MS + 15000);
main().catch((e) => fail(String(e)));
