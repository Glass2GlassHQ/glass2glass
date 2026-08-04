// Headless validation that a browser plays a g2g MoQ Transport broadcast (M904).
//
// Starts a local `moq-relay-ietf`, publishes into it with
// `videotestsrc pattern=smpte ! videoconvert ! x264enc ! mp4mux ! moqtsink`,
// and drives a real Chromium through tools/moqt-demo/index.html, whose MoQT
// client is the third-party MOQtail draft-16 implementation. Nothing in the
// browser shares code with g2g's Rust wire layer, so a successful play is an
// independent decode of our bytes.
//
// Asserts, on the frames the browser's own H.264 decoder produced:
//   - the decoded size is 320x240 (what the publisher encoded);
//   - the video pipeline reports at least NEED_FRAMES decoded frames;
//   - the seven SMPTE bars are on screen in the right order, sampled off the
//     canvas the <video> was drawn into.
//
// Prereqs: `npm install` in tools/moqt-demo (playwright + moqtail), a full
// Chromium (headless_shell has no H.264), a `moq-relay-ietf` build, and
// `cargo build --release -p g2g-plugins --features moqt,ffmpeg --bin g2g-launch`.
// Prints SKIP and exits 0 when any of those is missing.
//
// Run from tools/moqt-demo:  node headless/run-moqt-play.mjs
// Env: MOQ_RS_BIN, G2G_LAUNCH, G2G_CHROME, G2G_PLAYWRIGHT, G2G_HEADFUL=1.
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
import {
  ROOT, chromeBinary, freeUdpPort, launchBinary, mintCertificate, pageUrl,
  publishPipeline, relayBinary, spawnPublisher, spawnRelay, startHttp, whenPublishing,
  SMPTE_PIPELINE,
} from "../local-relay.mjs";

const HTTP_PORT = 8197;
const NAMESPACE = "g2gdemo";
const NEED_FRAMES = 10;
const TIMEOUT_MS = 60000;
// 75% SMPTE bars, in order, as which channels are lit. `videotestsrc
// pattern=smpte` draws rgb(192,...) / rgb(0,...); asserting per-channel high
// or low rather than the exact value keeps this independent of the YUV matrix
// the browser's decoder happens to use.
const BARS = [
  ["gray", 1, 1, 1], ["yellow", 1, 1, 0], ["cyan", 0, 1, 1], ["green", 0, 1, 0],
  ["magenta", 1, 0, 1], ["red", 1, 0, 0], ["blue", 0, 0, 1],
];
const LIT = 128;
const DARK = 96;

function log(...a) { console.log("[harness]", ...a); }

let http, browser, tls;
const children = [];

async function shutdown(code) {
  for (const c of children) c.kill();
  try { await browser?.close(); } catch {}
  try { http?.close(); } catch {}
  try { await tls?.cleanup(); } catch {}
  process.exit(code);
}

function fail(msg) {
  console.error("[harness] FAIL:", msg);
  shutdown(1);
}

function skip(msg) {
  console.log("[harness] SKIP:", msg);
  shutdown(0);
}

async function main() {
  const relayBin = relayBinary();
  if (!relayBin) {
    skip("moq-relay-ietf not found. Build it with `cargo build --release -p moq-relay-ietf` " +
      "in a cloudflare/moq-rs checkout, or point $MOQ_RS_BIN at its directory.");
  }
  const launchBin = launchBinary();
  if (!launchBin) {
    skip("g2g-launch not built. Run `cargo build --release -p g2g-plugins " +
      "--features moqt,ffmpeg --bin g2g-launch`, or set $G2G_LAUNCH.");
  }
  const chrome = chromeBinary();
  if (!chrome) {
    skip("no full Chromium found (headless_shell cannot decode H.264). " +
      "Run `npx playwright install chromium`, or set $G2G_CHROME.");
  }
  if (!existsSync(join(ROOT, "node_modules/moqtail/dist/index.js"))) {
    skip("moqtail not installed. Run `npm install` in tools/moqt-demo.");
  }
  const pwPath = process.env.G2G_PLAYWRIGHT || join(ROOT, "node_modules/playwright/index.js");
  if (!existsSync(pwPath)) {
    skip("playwright not installed. Run `npm install` in tools/moqt-demo, or set $G2G_PLAYWRIGHT.");
  }

  tls = await mintCertificate();
  const relayPort = await freeUdpPort();
  log(`relay on ${relayPort}, leaf sha-256 ${tls.hashHex}`);
  children.push(spawnRelay(relayBin, tls, relayPort, log));
  // The relay binds asynchronously; give it a moment rather than racing it.
  await sleep(1000);

  const pipeline = publishPipeline(SMPTE_PIPELINE, relayPort, NAMESPACE, tls.hashHex);
  log("publishing:", pipeline);
  const publisher = spawnPublisher(launchBin, pipeline, (line) => {
    // libx264's end-of-run statistics and the progress line say nothing about
    // the broadcast.
    if (!line.includes("libx264") && !line.includes("running...")) log(line);
  });
  children.push(publisher);
  // A subscriber that attaches before the first frame goes unacknowledged
  // (`moqtsink` applies control messages as frames arrive) and the relay
  // refuses the subscribe.
  if (!(await whenPublishing(publisher))) fail("the publisher produced no frames");

  http = await startHttp(HTTP_PORT);
  const pw = await import(pathToFileURL(pwPath).href);
  const { chromium } = pw.default || pw;
  browser = await chromium.launch({
    headless: !process.env.G2G_HEADFUL,
    executablePath: chrome,
    args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required",
      "--use-gl=angle", "--use-angle=swiftshader"],
  });
  const page = await browser.newPage();
  page.on("console", (m) => { if (m.text().startsWith("g2g[")) log("page:", m.text()); });
  page.on("pageerror", (e) => log("page error:", String(e)));

  const extra = { autostart: "1" };
  if (process.env.G2G_MOQT_DEBUG) extra.debug = "1";
  const url = pageUrl(HTTP_PORT, relayPort, NAMESPACE, tls.hashHex, extra);
  log("navigating", url);
  await page.goto(url);
  if (!(await page.evaluate(() => typeof WebTransport !== "undefined"))) {
    fail("browser has no WebTransport");
  }

  // Wait for enough decoded frames, or the page reporting it could not start.
  const t0 = Date.now();
  let report = null;
  while (Date.now() - t0 < TIMEOUT_MS) {
    const state = await page.evaluate(() => window.g2gState);
    if (state.error) fail(`player failed: ${state.error}`);
    if (state.started) {
      report = await page.evaluate(() => window.g2gReport());
      if (report.totalVideoFrames >= NEED_FRAMES && report.bars) break;
    }
    await page.waitForTimeout(250);
  }
  if (!report) fail("the player never reached playback");
  log("report:", JSON.stringify(report));

  if (report.totalVideoFrames < NEED_FRAMES) {
    fail(`only ${report.totalVideoFrames}/${NEED_FRAMES} frames decoded ` +
      `(${(await page.evaluate(() => window.g2gState)).fragments} fragments received)`);
  }
  if (report.width !== 320 || report.height !== 240) {
    fail(`decoded ${report.width}x${report.height}, expected 320x240`);
  }
  const wrong = BARS.map(([name, ...want], i) => {
    const px = report.bars[i];
    const ok = want.every((lit, c) => (lit ? px[c] >= LIT : px[c] <= DARK));
    return ok ? null : `${name} bar reads rgb(${px})`;
  }).filter(Boolean);
  if (wrong.length) fail(`SMPTE bars wrong: ${wrong.join("; ")}`);

  log(`PASS: ${report.totalVideoFrames} frames decoded at ${report.width}x${report.height}, ` +
    `all seven SMPTE bars correct, ${report.droppedVideoFrames} dropped`);
  shutdown(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

setTimeout(() => fail("overall timeout"), TIMEOUT_MS + 20000);
main().catch((e) => fail(String(e)));
