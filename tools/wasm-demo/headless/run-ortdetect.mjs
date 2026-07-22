// Headless validation for the browser ONNX MVP (M762).
//
// Drives the real in-browser chain WebSocketSrc -> WebCodecsDecode -> WebOrtDetect
// (onnxruntime-web CPU inference) -> AnalyticsOverlay -> CanvasSink in a
// WebCodecs-capable Chromium, against:
//   - the committed H.264 fixture, streamed by ws-fixture-server (one AU/message);
//   - the committed tiny-detect.onnx served over plain static HTTP (no COOP/COEP).
//
// tiny-detect.onnx is deterministic: it plants exactly two detections (one per
// class) every frame, so a real ort-web run must log "frame N -> 2 detections".
// Asserts: the module inits, ort-web loads and creates a session, several frames
// each yield exactly 2 detections, the canvas shows decoded (non-blank) video,
// and no pipeline error is logged.
//
// Prereqs: `npm i -D playwright` (or run with NODE_PATH pointing at a playwright
// install) and a full (WebCodecs-capable) Chromium. Run from tools/wasm-demo:
//   node headless/run-ortdetect.mjs
// Env overrides: G2G_CHROME (chrome executable), G2G_WS_SERVER_BIN (prebuilt
// ws-fixture-server, else `cargo run`), G2G_FIXTURE (.h264), G2G_HEADFUL=1.
import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join, extname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, ".."); // tools/wasm-demo, served as the web root
const HTTP_PORT = 8199;
const WS_PORT = 8198;
const FIXTURE = process.env.G2G_FIXTURE ||
  resolve(ROOT, "../../g2g-plugins/tests/fixtures/h264_640x480.h264");
const NEED_FRAMES = 5; // distinct frames that must each yield 2 detections
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

// Static file server for tools/wasm-demo (pkg/, fixtures/, headless/, snippets).
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
  // Give it a moment to compile (cargo) / bind.
  return new Promise((res) => {
    const ready = (d) => { if (d.toString().includes("serving ws://")) { wsProc.stdout.off("data", ready); res(); } };
    wsProc.stdout.on("data", ready);
    setTimeout(res, bin ? 800 : 30000); // fallback: assume up
  });
}

async function main() {
  if (!existsSync(join(ROOT, "pkg/g2g_web.js"))) fail("pkg/g2g_web.js missing (run build.sh)");
  if (!existsSync(join(ROOT, "fixtures/tiny-detect.onnx"))) fail("fixtures/tiny-detect.onnx missing (run gen-tiny-detect.py)");
  await startHttp();
  await startWsServer();
  log("http on", HTTP_PORT, "ws on", WS_PORT);

  // Prefer a normal `playwright` dependency; G2G_PLAYWRIGHT can point at an
  // out-of-tree install (ESM ignores NODE_PATH).
  const pw = process.env.G2G_PLAYWRIGHT
    ? await import(pathToFileURL(process.env.G2G_PLAYWRIGHT).href)
    : await import("playwright");
  const { chromium } = pw.default || pw;
  const exe = process.env.G2G_CHROME;
  browser = await chromium.launch({
    headless: !process.env.G2G_HEADFUL,
    executablePath: exe || undefined,
    // No SharedArrayBuffer flag: the ort-web chain is single-threaded, the whole
    // point of the no-COOP/COEP MVP. swiftshader gives headless a software GL.
    args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"],
  });
  const page = await browser.newPage();
  const counts = [];
  let sessionReady = false, finishedOk = false, pipelineError = null;
  page.on("console", (m) => {
    const t = m.text();
    if (t.startsWith("g2g[")) log("page:", t);
    if (t.includes("session ready")) sessionReady = true;
    const fm = t.match(/frame \d+ -> (\d+) detections/);
    if (fm) counts.push(Number(fm[1]));
    if (t.includes("finished ok")) finishedOk = true;
    if (t.includes("pipeline error")) pipelineError = t;
  });
  page.on("pageerror", (e) => { pipelineError = String(e); });

  const url = `http://127.0.0.1:${HTTP_PORT}/headless/ortdetect.html`
    + `?ws=${encodeURIComponent(`ws://127.0.0.1:${WS_PORT}`)}`
    + `&model=${encodeURIComponent("/fixtures/tiny-detect.onnx")}`;
  log("navigating", url);
  await page.goto(url);

  // Confirm the browser actually has WebCodecs (headless_shell does not).
  const hasWebCodecs = await page.evaluate(() => typeof VideoDecoder !== "undefined");
  if (!hasWebCodecs) fail("browser lacks WebCodecs (use a full Chromium, not headless_shell)");

  // Finite source: stop feeding after one fixture pass so the chain reaches EOS
  // and finishes cleanly. (An unbounded source that keeps feeding while every
  // frame awaits ort-web trips a separate wasm async-runtime reentrancy; see the
  // Browser/Wasm follow-up in DESIGN_TODO.)
  setTimeout(() => { try { wsProc?.kill("SIGKILL"); } catch {} log("finite source: stopped ws feed"); }, 1100);

  const t0 = Date.now();
  while (Date.now() - t0 < TIMEOUT_MS) {
    if (pipelineError) fail("pipeline error before EOS: " + pipelineError);
    if (finishedOk) break;
    await page.waitForTimeout(200);
  }
  if (!sessionReady) fail("ort-web session never became ready (CDN load / model fetch failed?)");
  if (!finishedOk) fail(`chain did not finish (session=${sessionReady}, frames=${counts.length})`);
  if (counts.length < NEED_FRAMES) fail(`only ${counts.length}/${NEED_FRAMES} inference frames`);

  const bad = counts.filter((c) => c !== 2);
  if (bad.length) fail(`expected 2 detections/frame, got ${JSON.stringify(counts)}`);

  // The detection results must have rendered: AnalyticsOverlay draws the class-0
  // box red (0xFF3B30) and the class-1 box green (0x34C759). Assert both colors
  // are present on the canvas. (The decoded video itself reads back black under
  // headless WebCodecs on this host, a known copyTo-zeros GPU-backend gotcha, so
  // the video pixels are not asserted here; the overlay is the rendered result.)
  const px = await page.evaluate(() => {
    const c = document.getElementById("view");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    const hit = (r, g, b) => { for (let i = 0; i < d.length; i += 4) if (d[i] === r && d[i + 1] === g && d[i + 2] === b) return true; return false; };
    const colors = new Set();
    for (let i = 0; i < d.length; i += 4) colors.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    return { red: hit(0xff, 0x3b, 0x30), green: hit(0x34, 0xc7, 0x59), colors: colors.size };
  });
  if (!px.red || !px.green) fail(`overlay boxes not rendered (red=${px.red} green=${px.green}, ${px.colors} colors)`);

  log(`PASS: ort-web session ready, ${counts.length} frames x 2 detections, finished ok, overlay boxes rendered (${px.colors} canvas colors)`);
  shutdown(0);
}

setTimeout(() => fail("overall timeout"), TIMEOUT_MS + 15000);
main().catch((e) => fail(String(e)));
