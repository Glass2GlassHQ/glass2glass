// Headless comparison of the two decode paths on one live broadcast (M942).
//
// Publishes the same recorded SMPTE clip twice, paced to its captured 30 fps by
// `replaysrc sync=true` and muxed one MOQT object per access unit with a `prft`
// per fragment, then plays it once through MSE and once through WebCodecs and
// reports what each measured.
//
// Asserts, per mode, on what the browser's decoders produced:
//   - at least NEED_FRAMES frames decoded, at the 320x240 the publisher encoded;
//   - the seven SMPTE bars on screen in the right order (off the <video> for
//     MSE, straight off the canvas the VideoFrames were drawn to for WebCodecs);
//   - a latency measurement exists in both modes, and WebCodecs' median is the
//     lower of the two, which is the point of the mode.
//
// Latency here is end to end: the wall clock the muxer wrote into the fragment's
// `prft` against the moment the frame reached the screen, both read on this one
// machine, so the two clocks are one clock.
//
// A live-paced publisher is the whole reason this is a separate run. The plain
// `videotestsrc ! x264enc` pipeline the other headless run uses encodes hundreds
// of frames a second, so a player is handed minutes of media per minute of wall
// clock and whatever latency it measures describes that, not the player.
//
// Prereqs: the same as run-moqt-play.mjs. Prints SKIP and exits 0 without them.
//
// Run from tools/moqt-demo:  node headless/run-moqt-latency.mjs
// Env: MOQ_RS_BIN, G2G_LAUNCH, G2G_CHROME, G2G_PLAYWRIGHT, G2G_HEADFUL=1.
import {
  freeUdpPort, mintCertificate, pacedPublishPipeline, pageUrl, recordPacedClip,
  spawnPublisher, spawnRelay, startHttp, whenPublishing,
} from "../local-relay.mjs";
import { launchBrowser, openPlayer, prereqs, waitForReport, wrongBars } from "./common.mjs";

const HTTP_PORT = 8199;
const NEED_FRAMES = 10;
// Frames each mode must have measured the latency of before its median is read,
// so the number describes steady state and not the first fragment out of the
// join. The player keeps a 120-sample window, i.e. four seconds at 30 fps.
const NEED_LATENCY_SAMPLES = 90;
// Media in the clip: enough for both passes to connect, join at the next
// keyframe, and then run for the samples above.
const CLIP_SECONDS = 25;
const PASS_TIMEOUT_MS = 60000;

function log(...a) { console.log("[latency]", ...a); }

let http, browser, tls, clipDir;
const children = [];

async function shutdown(code) {
  for (const c of children) c.kill();
  try { await browser?.close(); } catch {}
  try { http?.close(); } catch {}
  try { await tls?.cleanup(); } catch {}
  try { await clipDir?.cleanup(); } catch {}
  process.exit(code);
}

function fail(msg) {
  console.error("[latency] FAIL:", msg);
  shutdown(1);
}

function skip(msg) {
  console.log("[latency] SKIP:", msg);
  shutdown(0);
}

// One decode mode against its own fresh broadcast of the same clip, so both
// modes see the same media from its first frame.
async function measure(mode, { launchBin, clip, relayPort }) {
  const namespace = `g2g${mode}`;
  const pipeline = pacedPublishPipeline(clip, relayPort, namespace, tls.hashHex);
  log(`${mode}: publishing ${pipeline}`);
  const publisher = spawnPublisher(launchBin, pipeline, (line) => {
    if (!line.includes("libx264") && !line.includes("running...")) log(line);
  });
  children.push(publisher);
  if (!(await whenPublishing(publisher))) fail(`${mode}: the publisher produced no frames`);

  const extra = { autostart: "1" };
  if (mode === "webcodecs") extra.decoder = "webcodecs";
  if (process.env.G2G_MOQT_DEBUG) extra.debug = "1";
  const page = await openPlayer(
    browser, pageUrl(HTTP_PORT, relayPort, namespace, tls.hashHex, extra), log);

  const report = await waitForReport(page, {
    timeoutMs: PASS_TIMEOUT_MS,
    ready: (r) =>
      r.totalVideoFrames >= NEED_FRAMES && r.bars && r.latencySamples >= NEED_LATENCY_SAMPLES,
  });
  await page.close();
  publisher.kill();
  log(`${mode} report:`, JSON.stringify(report));
  return report;
}

function check(mode, report) {
  if (report.totalVideoFrames < NEED_FRAMES) {
    fail(`${mode}: only ${report.totalVideoFrames}/${NEED_FRAMES} frames decoded ` +
      `(${report.fragments} fragments received)`);
  }
  if (report.width !== 320 || report.height !== 240) {
    fail(`${mode}: decoded ${report.width}x${report.height}, expected 320x240`);
  }
  const wrong = wrongBars(report.bars);
  if (wrong.length) fail(`${mode}: SMPTE bars wrong: ${wrong.join("; ")}`);
  if (report.latencyMedianMs === null || report.latencySamples < NEED_LATENCY_SAMPLES) {
    fail(`${mode}: only ${report.latencySamples} latency samples, no prft reached the player`);
  }
}

async function main() {
  const { relayBin, launchBin, chrome, pwPath, skip: missing } = prereqs();
  if (missing) skip(missing);

  tls = await mintCertificate();
  const relayPort = await freeUdpPort();
  log(`relay on ${relayPort}, leaf sha-256 ${tls.hashHex}`);
  children.push(spawnRelay(relayBin, tls, relayPort, log));
  // The relay binds asynchronously; give it a moment rather than racing it.
  await sleep(1000);

  clipDir = await mintClipDir();
  const clip = recordPacedClip(launchBin, clipDir.dir, CLIP_SECONDS);
  log(`recorded ${CLIP_SECONDS}s of SMPTE bars to ${clip}`);

  http = await startHttp(HTTP_PORT);
  browser = await launchBrowser(pwPath, chrome);

  const opts = { launchBin, clip, relayPort };
  const mse = await measure("mse", opts);
  const webcodecs = await measure("webcodecs", opts);
  check("mse", mse);
  check("webcodecs", webcodecs);

  const ms = (v) => `${Math.round(v)} ms`;
  log(`median end-to-end latency: MSE ${ms(mse.latencyMedianMs)} ` +
    `(${mse.latencySamples} frames), WebCodecs ${ms(webcodecs.latencyMedianMs)} ` +
    `(${webcodecs.latencySamples} frames)`);
  if (webcodecs.latencyMedianMs >= mse.latencyMedianMs) {
    fail(`WebCodecs is not the lower-latency path here: ${ms(webcodecs.latencyMedianMs)} ` +
      `against MSE's ${ms(mse.latencyMedianMs)}`);
  }
  log(`PASS: both modes decoded the bars; WebCodecs cut the median latency from ` +
    `${ms(mse.latencyMedianMs)} to ${ms(webcodecs.latencyMedianMs)}, ` +
    `${ms(mse.latencyMedianMs - webcodecs.latencyMedianMs)} lower`);
  shutdown(0);
}

// A temp directory for the recorded clip, cleaned up like the certificate is.
async function mintClipDir() {
  const { mkdtemp, rm } = await import("node:fs/promises");
  const { tmpdir } = await import("node:os");
  const { join } = await import("node:path");
  const dir = await mkdtemp(join(tmpdir(), "g2g-moqt-clip-"));
  return { dir, cleanup: () => rm(dir, { recursive: true, force: true }) };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

setTimeout(() => fail("overall timeout"), PASS_TIMEOUT_MS * 2 + 60000);
main().catch((e) => fail(String(e)));
