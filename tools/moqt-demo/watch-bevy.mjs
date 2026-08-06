// Watch a Bevy scene rendered and published over MoQ Transport, from one
// command.
//
// Same shape as watch-live.mjs, but the publisher is the bevy-g2g `stream`
// example instead of a g2g-launch pipeline: Bevy renders headless, g2g encodes
// H.264 and `mp4mux ! moqtsink` publishes it into a locally spawned
// `moq-relay-ietf`. Video only, no audio track. Ctrl-C stops everything.
//
// The muxer runs at its default fragment duration, so every frame is one MOQT
// object (a group per keyframe): the low-latency shape, rather than the
// half-second CMAF fragments watch-live.mjs publishes.
//
// Prereqs: a `moq-relay-ietf` build, `npm install` here, and a cargo toolchain
// (the example is built on first run, which takes a few minutes).
//
// The stream is live-paced at the render framerate and carries a `prft` per
// fragment, so the page's latency HUD reads a real end-to-end number here.
// `G2G_MOQT_WEBCODECS=1` decodes with WebCodecs instead of MSE, which is the
// interesting comparison: same broadcast, a few hundred ms less latency.
//
// Run from tools/moqt-demo:  node watch-bevy.mjs
// Env: MOQ_RS_BIN, G2G_CHROME, G2G_MOQT_DEBUG=1 (player debug output),
// G2G_MOQT_WEBCODECS=1, G2G_FRAMES (default 0, run until Ctrl-C), plus
// `--features nvenc` through G2G_BEVY_FEATURES for the zero-copy encode path.
import { spawn } from "node:child_process";
import { resolve } from "node:path";
import {
  Reaped, chromeBinary, freeUdpPort, mintCertificate, pageUrl, playerParams,
  relayBinary, spawnRelay, startHttp, ROOT,
} from "./local-relay.mjs";

const HTTP_PORT = 8198;
const NAMESPACE = "bevy";
const BEVY = resolve(ROOT, "../../bevy-g2g");
// The example logs this once the first rendered frame reaches g2g, which is
// when the broadcast starts existing: `moqtsink` applies control messages as
// frames arrive, so a page that attaches before then is refused by the relay.
const FIRST_FRAME = "first frame handed to g2g";
const READY_TIMEOUT_MS = 300000;

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

// Run the example through cargo, so a first run builds it. The G2G_MOQT_* vars
// are the from_env convention StreamSettings reads.
function spawnBevy(relayPort, hashHex) {
  const args = ["run", "--release"];
  if (process.env.G2G_BEVY_FEATURES) args.push("--features", process.env.G2G_BEVY_FEATURES);
  args.push("--example", "stream");
  const child = spawn("cargo", args, {
    cwd: BEVY,
    env: {
      ...process.env,
      G2G_MOQT_URL: `https://127.0.0.1:${relayPort}/`,
      G2G_MOQT_NAMESPACE: NAMESPACE,
      G2G_MOQT_CERT_HASHES: hashHex,
      G2G_FRAMES: process.env.G2G_FRAMES ?? "0",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  return new Reaped(child);
}

// Resolve once the example reports its first frame. Resolves false on timeout,
// so the caller can open the page anyway and let the failure surface there.
function whenRendering(reaped, onLine) {
  return new Promise((res) => {
    let done = false;
    const timer = setTimeout(() => { done = true; res(false); }, READY_TIMEOUT_MS);
    for (const s of [reaped.child.stdout, reaped.child.stderr]) {
      s?.setEncoding("utf8");
      s?.on("data", (d) => {
        for (const line of d.split(/[\r\n]/)) {
          if (!line.trim()) continue;
          onLine(line);
          if (!done && line.includes(FIRST_FRAME)) {
            done = true;
            clearTimeout(timer);
            res(true);
          }
        }
      });
    }
  });
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

  tls = await mintCertificate();
  const relayPort = await freeUdpPort();
  children.push(spawnRelay(relayBin, tls, relayPort, log));
  await sleep(1000);

  log(`rendering and publishing to https://127.0.0.1:${relayPort}/ as ${NAMESPACE}`);
  const publisher = spawnBevy(relayPort, tls.hashHex);
  children.push(publisher);

  http = await startHttp(HTTP_PORT);
  const url = pageUrl(HTTP_PORT, relayPort, NAMESPACE, tls.hashHex, playerParams());
  log("player:", url);

  log("waiting for the first rendered frame (a cold cargo build takes minutes)...");
  if (!(await whenRendering(publisher, (line) => {
    if (!line.includes("libx264")) log("[bevy]", line);
  }))) {
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
