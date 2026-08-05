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
// Run from tools/moqt-demo:  node watch-live.mjs
// Env: MOQ_RS_BIN, G2G_LAUNCH, G2G_CHROME, G2G_MOQT_PATTERN=1 (force the
// test pattern), G2G_CAMERA_SIZE=1280x720.
import { spawn } from "node:child_process";
import {
  CAMERA_PIPELINE, SMPTE_PIPELINE, chromeBinary, freeUdpPort, hasCamera,
  launchBinary, mintCertificate, pageUrl, publishPipeline, relayBinary,
  spawnPublisher, spawnRelay, startHttp,
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

  const pipeline = publishPipeline(prefix, relayPort, NAMESPACE, tls.hashHex);
  log("publishing:", pipeline);
  children.push(spawnPublisher(launchBin, pipeline, (line) => {
    if (!line.includes("libx264")) log(line);
  }));

  http = await startHttp(HTTP_PORT);
  const url = pageUrl(HTTP_PORT, relayPort, NAMESPACE, tls.hashHex, { autostart: "1" });
  log("player:", url);

  const chrome = chromeBinary();
  if (chrome) {
    viewer = spawn(chrome, [url], { stdio: "ignore" });
    log("opened a browser. Ctrl-C to stop.");
  } else {
    log("no browser found: open the URL above yourself. Ctrl-C to stop.");
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

main().catch((e) => die(String(e)));
