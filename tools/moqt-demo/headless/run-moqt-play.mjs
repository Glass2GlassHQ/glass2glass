// Headless validation that a browser plays a g2g MoQ Transport broadcast (M904).
//
// Starts a local `moq-relay-ietf`, publishes video and audio into it with
// `videotestsrc ! clocksync ! x264enc ! mux.  audiotestsrc ! clocksync !
// avenc_aac ! mux.  mp4mux name=mux ! moqtsink`, and drives a real Chromium through
// tools/moqt-demo/index.html, whose MoQT client is the third-party MOQtail
// draft-16 implementation. Nothing in the browser shares code with g2g's Rust
// wire layer, so a successful play is an independent decode of our bytes.
//
// Asserts, on what the browser's own decoders produced:
//   - the decoded size is 320x240 (what the publisher encoded);
//   - the video pipeline reports at least NEED_FRAMES decoded frames;
//   - the seven SMPTE bars are on screen in the right order, sampled off the
//     canvas the <video> was drawn into;
//   - the audio track was subscribed, appended and decoded (Chromium's
//     `webkitAudioDecodedByteCount` past zero).
//
// The publisher is paced to the wall clock by `clocksync`, so it is a live 30
// fps broadcast and each access unit is its own MOQT object. This is the MSE
// path only; the WebCodecs path and the latency both decode modes report are run
// by run-moqt-latency.mjs.
//
// Prereqs: `pnpm install` in tools/moqt-demo (playwright + moqtail), a full
// Chromium (headless_shell has no H.264 or AAC), a `moq-relay-ietf` build, and
// `cargo build --release -p g2g-plugins --features moqt,ffmpeg --bin g2g-launch`.
// Prints SKIP and exits 0 when any of those is missing.
//
// Run from tools/moqt-demo:  node headless/run-moqt-play.mjs
// Env: MOQ_RS_BIN, G2G_LAUNCH, G2G_CHROME, G2G_PLAYWRIGHT, G2G_HEADFUL=1.
import {
  freeUdpPort, mintCertificate, pageUrl, publishPipeline, spawnPublisher, spawnRelay,
  startHttp, whenPublishing, SMPTE_PIPELINE,
} from "../local-relay.mjs";
import { launchBrowser, openPlayer, prereqs, waitForReport, wrongBars } from "./common.mjs";

const HTTP_PORT = 8197;
const NAMESPACE = "g2gdemo";
const NEED_FRAMES = 10;
const TIMEOUT_MS = 60000;

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
  const { relayBin, launchBin, chrome, pwPath, skip: missing } = prereqs();
  if (missing) skip(missing);

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
  browser = await launchBrowser(pwPath, chrome);

  const extra = { autostart: "1" };
  if (process.env.G2G_MOQT_DEBUG) extra.debug = "1";
  const url = pageUrl(HTTP_PORT, relayPort, NAMESPACE, tls.hashHex, extra);
  const page = await openPlayer(browser, url, log);

  // Wait for enough decoded frames, or the page reporting it could not start.
  const report = await waitForReport(page, {
    timeoutMs: TIMEOUT_MS,
    ready: (r) =>
      r.totalVideoFrames >= NEED_FRAMES && r.bars &&
      r.audioFragments > 0 && (r.audioDecodedBytes ?? 1) > 0,
  });
  log("report:", JSON.stringify(report));

  if (report.totalVideoFrames < NEED_FRAMES) {
    fail(`only ${report.totalVideoFrames}/${NEED_FRAMES} frames decoded ` +
      `(${report.fragments} fragments received)`);
  }
  if (report.width !== 320 || report.height !== 240) {
    fail(`decoded ${report.width}x${report.height}, expected 320x240`);
  }
  if (!report.audioFragments) {
    fail("no audio fragments reached the page: the broadcast published no audio track");
  }
  if (report.audioDecodedBytes === 0) {
    fail("the audio track was appended but the browser decoded none of it");
  }

  const wrong = wrongBars(report.bars);
  if (wrong.length) fail(`SMPTE bars wrong: ${wrong.join("; ")}`);

  log(`PASS: ${report.totalVideoFrames} frames decoded at ${report.width}x${report.height}, ` +
    `all seven SMPTE bars correct, ${report.droppedVideoFrames} dropped, ` +
    `${report.audioFragments} audio fragments and ${report.audioDecodedBytes} audio bytes decoded`);
  shutdown(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

setTimeout(() => fail("overall timeout"), TIMEOUT_MS + 20000);
main().catch((e) => fail(String(e)));
