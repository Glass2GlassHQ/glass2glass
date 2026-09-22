// Headless validation for the receive direction of the distributed-graph
// primitive (M1190).
//
// A browser can only dial out, so the native side listens: wire-serve-server
// runs `VideoTestSrc -> RemoteWsSink listen=true` and serves the whole
// PipelinePacket stream to whoever connects. The page runs the browser half,
// `WsWireSrc -> CanvasSink`, so this exercises the one path that only a browser
// can run: caps discovered off the wire, then decoded frames painted.
//
// Asserts: the module inits, the chain reaches EOS cleanly (no pipeline error),
// it reports the frames the server sent, and the canvas holds the test pattern
// rather than a blank surface.
//
// Prereqs: `npm i playwright-core` here (or a `playwright` install) and a full
// Chromium. Run from tools/wasm-demo:
//   node headless/run-wireingest.mjs
// Env overrides: G2G_CHROME (chrome executable), G2G_WIRE_SERVER_BIN (prebuilt
// wire-serve-server, else `cargo run`), G2G_HEADFUL=1.
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
/// Frames the server sends; the browser must report the same count.
const FRAMES = 30;
const TIMEOUT_MS = 60000;

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".wasm": "application/wasm", ".json": "application/json", ".css": "text/css",
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

// Static file server for tools/wasm-demo (pkg/, headless/).
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

function startWireServer() {
  const bin = process.env.G2G_WIRE_SERVER_BIN;
  const addr = `127.0.0.1:${WS_PORT}`;
  const [cmd, args] = bin && existsSync(bin)
    ? [bin, [addr, String(FRAMES)]]
    : ["cargo", ["run", "--release", "--manifest-path",
        resolve(ROOT, "wire-serve-server/Cargo.toml"), "--", addr, String(FRAMES)]];
  log("wire server:", cmd, args.join(" "));
  wsProc = spawn(cmd, args, { stdio: ["ignore", "pipe", "pipe"] });
  wsProc.stdout.on("data", (d) => process.stdout.write("[wire] " + d));
  wsProc.stderr.on("data", (d) => process.stderr.write("[wire] " + d));
  // The sink binds before it accepts, and the browser's connect does not retry.
  return new Promise((res) => {
    const ready = (d) => { if (d.toString().includes("serving ws://")) { wsProc.stdout.off("data", ready); setTimeout(res, 200); } };
    wsProc.stdout.on("data", ready);
    setTimeout(res, bin ? 1500 : 60000); // fallback: assume up
  });
}

async function main() {
  if (!existsSync(join(ROOT, "pkg/g2g_web.js"))) fail("pkg/g2g_web.js missing (run build.sh)");
  await startHttp();
  await startWireServer();
  log("http on", HTTP_PORT, "wire on", WS_PORT);

  // Prefer a normal `playwright` dependency; playwright-core (no bundled
  // browsers) works too when G2G_CHROME names one.
  const pw = process.env.G2G_PLAYWRIGHT
    ? await import(pathToFileURL(process.env.G2G_PLAYWRIGHT).href)
    : await import("playwright-core").catch(() => import("playwright"));
  const { chromium } = pw.default || pw;
  browser = await chromium.launch({
    headless: !process.env.G2G_HEADFUL,
    executablePath: process.env.G2G_CHROME || undefined,
    args: ["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader"],
  });
  const page = await browser.newPage();
  let finished = null, pipelineError = null;
  page.on("console", (m) => {
    const t = m.text();
    if (t.startsWith("g2g[")) log("page:", t);
    if (t.includes("finished ok")) finished = t;
    if (t.includes("pipeline error")) pipelineError = t;
  });
  page.on("pageerror", (e) => { pipelineError = String(e); });

  const url = `http://127.0.0.1:${HTTP_PORT}/headless/wireingest.html`
    + `?ws=${encodeURIComponent(`ws://127.0.0.1:${WS_PORT}`)}`;
  log("navigating", url);
  await page.goto(url);

  const t0 = Date.now();
  while (Date.now() - t0 < TIMEOUT_MS) {
    if (pipelineError) fail("pipeline error: " + pipelineError);
    if (finished) break;
    await page.waitForTimeout(200);
  }
  if (!finished) fail("the browser chain never finished");

  // The stats line names the frames the browser consumed: every frame the
  // server sent has to have crossed.
  const consumed = Number((finished.match(/frames_consumed: (\d+)/) || [])[1]);
  if (consumed !== FRAMES) fail(`browser consumed ${consumed} of ${FRAMES} frames: ${finished}`);

  // The canvas has to hold the test pattern, not a blank surface: several
  // distinct colors, and not one flat fill.
  const px = await page.evaluate(() => {
    const c = document.getElementById("view");
    const d = c.getContext("2d").getImageData(0, 0, c.width, c.height).data;
    const colors = new Set();
    for (let i = 0; i < d.length; i += 4) colors.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    return { colors: colors.size, first: (d[0] << 16) | (d[1] << 8) | d[2] };
  });
  if (px.colors < 3) fail(`canvas shows ${px.colors} colors, so no pattern was painted`);

  log(`PASS: ${consumed} frames read off the wire and painted (${px.colors} canvas colors)`);
  shutdown(0);
}

setTimeout(() => fail("overall timeout"), TIMEOUT_MS + 15000);
main().catch((e) => fail(String(e)));
