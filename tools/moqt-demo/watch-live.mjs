// Watch a live g2g MoQ Transport broadcast in a browser, from one command.
//
// Starts a local `moq-relay-ietf`, publishes the camera into it through
// `moqtsink`, serves the player page and opens a browser on it. Ctrl-C stops
// everything. The headless sibling (headless/run-moqt-play.mjs) shares the
// certificate minting and the relay/publisher startup with this.
//
// Prereqs: a `moq-relay-ietf` build, `npm install` here, and
// `cargo build --release -p g2g-plugins --features libcamera,moqt,ffmpeg --bin g2g-launch`.
// Without a camera it falls back to the SMPTE test pattern and says so.
//
// The broadcast carries a 440 Hz AAC tone alongside the video, so the page has
// two tracks to play; `G2G_MOQT_NO_AUDIO=1` publishes video only. The player
// starts muted (browsers block unmuted autoplay), so unmute it to hear the tone.
//
// `G2G_MOQT_WEBCODECS=1` decodes with WebCodecs instead of MSE: video only, and
// the page's latency HUD then reads the frame it just drew rather than the
// playhead of a buffered <video>. The camera is live-paced so that number means
// something; the SMPTE fallback encodes as fast as the CPU allows, and against
// a source running ahead of real time the HUD reads whatever that implies.
//
// Run from tools/moqt-demo:  node watch-live.mjs
// Env: MOQ_RS_BIN, G2G_LAUNCH, G2G_CHROME, G2G_MOQT_PATTERN=1 (force the
// test pattern), G2G_MOQT_NO_AUDIO=1, G2G_MOQT_WEBCODECS=1,
// G2G_CAMERA_SIZE=1280x720.
import { spawn } from "node:child_process";
import {
  CAMERA_PIPELINE, SMPTE_PIPELINE, chromeBinary, freeUdpPort, hasCamera,
  launchBinary, mintCertificate, pageUrl, playerParams, publishPipeline,
  relayBinary, spawnPublisher, spawnRelay, startHttp, whenPublishing,
} from "./local-relay.mjs";

const HTTP_PORT = 8196;
const NAMESPACE = "g2glive";

function log(...a) { console.log("[watch]", ...a); }
function die(msg) { console.error("[watch]", msg); process.exit(1); }

let http, tls, viewer;
const children = [];

async function stop() {
  for (const c of children) c.kill();
  try { viewer?.kill(); } catch {}
  try { http?.close(); } catch {}
  try { await tls?.cleanup(); } catch {}
  process.exit(0);
}

function cameraPipeline() {
  const size = process.env.G2G_CAMERA_SIZE;
  if (!size) return CAMERA_PIPELINE;
  const [w, h] = size.split("x");
  return CAMERA_PIPELINE.replace("width=640", `width=${w}`).replace("height=480", `height=${h}`);
}

async function main() {
  // Installed before anything is started, so a Ctrl-C during startup still
  // reaps the relay and removes the certificate.
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);

  const relayBin = relayBinary();
  if (!relayBin) {
    die("moq-relay-ietf not found. Build it with `cargo build --release -p moq-relay-ietf` " +
      "in a cloudflare/moq-rs checkout, or point $MOQ_RS_BIN at its directory.");
  }
  const launchBin = launchBinary();
  if (!launchBin) {
    die("g2g-launch not built. Run `cargo build --release -p g2g-plugins " +
      "--features libcamera,moqt,ffmpeg --bin g2g-launch`.");
  }

  let prefix = cameraPipeline();
  if (process.env.G2G_MOQT_PATTERN || !hasCamera()) {
    log(process.env.G2G_MOQT_PATTERN
      ? "G2G_MOQT_PATTERN set: publishing the SMPTE test pattern."
      : "no /dev/video0 on this host: publishing the SMPTE test pattern instead of the camera.");
    prefix = SMPTE_PIPELINE;
  }

  tls = await mintCertificate();
  const relayPort = await freeUdpPort();
  children.push(spawnRelay(relayBin, tls, relayPort, log));
  await sleep(1000);

  const audio = !process.env.G2G_MOQT_NO_AUDIO;
  const pipeline = publishPipeline(prefix, relayPort, NAMESPACE, tls.hashHex, { audio });
  log("publishing:", pipeline);
  const publisher = spawnPublisher(launchBin, pipeline, (line) => {
    if (!line.includes("libx264") && !line.includes("running...")) log(line);
  });
  children.push(publisher);

  http = await startHttp(HTTP_PORT);
  const url = pageUrl(HTTP_PORT, relayPort, NAMESPACE, tls.hashHex, playerParams());
  log("player:", url);

  log("waiting for the first frames (a camera takes a moment to start)...");
  if (!(await whenPublishing(publisher))) {
    log("no frames yet; opening the player anyway, reload it if it does not connect.");
  }

  const chrome = chromeBinary({ preferSystem: true });
  if (chrome) {
    viewer = spawn(chrome, [url], { stdio: "ignore" });
    log("opened a browser. Ctrl-C to stop.");
  } else {
    log("no browser found: open the URL above yourself. Ctrl-C to stop.");
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

main().catch((e) => die(String(e)));
