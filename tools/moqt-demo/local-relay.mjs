// Shared plumbing for the two ways to drive the MoQ demo: the headless
// assertion run (headless/run-moqt-play.mjs) and the live watch script
// (watch-live.mjs). Both need the same self-signed certificate, the same
// relay, the same g2g publisher and the same static file server.
import { createServer } from "node:http";
import { spawn, spawnSync } from "node:child_process";
import { createSocket } from "node:dgram";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { existsSync, readdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { dirname, join, extname, resolve } from "node:path";

export const ROOT = resolve(dirname(fileURLToPath(import.meta.url)));
const REPO = resolve(ROOT, "../..");

const MIME = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".css": "text/css",
};

// A child that is killed when the harness exits, so a failure never leaves a
// relay or a publisher holding a port.
export class Reaped {
  constructor(child) {
    this.child = child;
  }
  kill() {
    try {
      this.child.kill("SIGKILL");
    } catch {}
  }
}

export function freeUdpPort() {
  return new Promise((res, rej) => {
    const s = createSocket("udp4");
    s.on("error", rej);
    s.bind(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => res(port));
    });
  });
}

// Where the reference relay lives: `$MOQ_RS_BIN`, else the release build of a
// moq-rs checkout under $HOME, else `PATH`. Mirrors `reference_binary` in the
// M903 interop test so both skip for the same reason.
export function relayBinary() {
  const name = "moq-relay-ietf";
  if (process.env.MOQ_RS_BIN) {
    const p = join(process.env.MOQ_RS_BIN, name);
    return existsSync(p) ? p : null;
  }
  if (process.env.HOME) {
    const p = join(process.env.HOME, "src/moq-rs/target/release", name);
    if (existsSync(p)) return p;
  }
  const probe = spawnSync(name, ["--help"], { stdio: "ignore" });
  return probe.status === 0 ? name : null;
}

// The g2g launcher, built with whichever features the caller needs.
export function launchBinary() {
  const p = process.env.G2G_LAUNCH || join(REPO, "target/release/g2g-launch");
  return existsSync(p) ? p : null;
}

// A full (WebCodecs / MSE capable) Chromium. `headless_shell` will not do:
// it ships no proprietary codecs and cannot decode H.264.
//
// `preferSystem` picks the browser a person already uses, profile and sign-in
// intact, which is what the live demo wants; the headless run wants the
// playwright build instead, because that is the one its driver was built for.
export function chromeBinary({ preferSystem = false } = {}) {
  if (process.env.G2G_CHROME) {
    return existsSync(process.env.G2G_CHROME) ? process.env.G2G_CHROME : null;
  }
  const system = ["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"]
    .find((p) => existsSync(p));
  if (preferSystem && system) return system;
  const cache = join(process.env.HOME || "", ".cache/ms-playwright");
  for (const entry of listDirs(cache).sort().reverse()) {
    // `chromium_headless_shell-*` sorts under the same prefix check, so match
    // the full-browser directory only.
    if (!entry.startsWith("chromium-")) continue;
    for (const sub of ["chrome-linux64/chrome", "chrome-linux/chrome"]) {
      const p = join(cache, entry, sub);
      if (existsSync(p)) return p;
    }
  }
  return system || null;
}

function listDirs(dir) {
  try {
    return readdirSync(dir);
  } catch {
    return [];
  }
}

// Mint a CA and a leaf signed by it, and return the leaf's SHA-256 as the hex
// `serverCertificateHashes` wants. A *self-signed* certificate used directly
// as the leaf is rejected by the QUIC stack (`CaUsedAsEndEntity`), so the two
// have to be separate. Browsers additionally require ECDSA P-256 and a
// validity window no longer than 14 days.
export async function mintCertificate() {
  const dir = await mkdtemp(join(tmpdir(), "g2g-moqt-"));
  const run = (args) => {
    const r = spawnSync("openssl", args, { cwd: dir, encoding: "buffer" });
    if (r.status !== 0) {
      throw new Error(`openssl ${args.join(" ")} failed: ${r.stderr}`);
    }
    return r;
  };
  run(["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
    "-keyout", "ca.key", "-out", "ca.pem", "-days", "13", "-nodes", "-subj", "/CN=g2g-demo-ca"]);
  run(["req", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1",
    "-keyout", "leaf.key", "-out", "leaf.csr", "-nodes", "-subj", "/CN=localhost"]);
  await writeFile(join(dir, "leaf.ext"),
    "subjectAltName=DNS:localhost,IP:127.0.0.1\nbasicConstraints=critical,CA:FALSE\nextendedKeyUsage=serverAuth\n");
  run(["x509", "-req", "-in", "leaf.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
    "-out", "leaf.pem", "-days", "13", "-extfile", "leaf.ext"]);
  const der = run(["x509", "-in", "leaf.pem", "-outform", "der"]).stdout;
  // The relay wants the leaf first and its issuer after it, or a peer that
  // validates the chain rejects the leaf as its own CA.
  const chain = Buffer.concat([await readFile(join(dir, "leaf.pem")), await readFile(join(dir, "ca.pem"))]);
  await writeFile(join(dir, "chain.pem"), chain);
  return {
    dir,
    certPath: join(dir, "chain.pem"),
    keyPath: join(dir, "leaf.key"),
    hashHex: createHash("sha256").update(der).digest("hex"),
    async cleanup() {
      await rm(dir, { recursive: true, force: true });
    },
  };
}

export function spawnRelay(bin, tls, port, onLine) {
  const child = spawn(bin, [
    "--bind", `127.0.0.1:${port}`,
    "--tls-cert", tls.certPath,
    "--tls-key", tls.keyPath,
  ], { stdio: ["ignore", "pipe", "pipe"] });
  pipeLines(child, "relay", onLine);
  return new Reaped(child);
}

// Run one `g2g-launch` pipeline. `pipeline` is the launch line as a single
// string, exactly what a person would type.
export function spawnPublisher(bin, pipeline, onLine) {
  const child = spawn(bin, [pipeline], { stdio: ["ignore", "pipe", "pipe"] });
  pipeLines(child, "publish", onLine);
  return new Reaped(child);
}

function pipeLines(child, tag, onLine) {
  for (const s of [child.stdout, child.stderr]) {
    s?.setEncoding("utf8");
    s?.on("data", (d) => {
      // The launcher rewrites its progress line with a bare \r, so splitting on
      // newlines alone would hold it back until something else printed.
      for (const line of d.split(/[\r\n]/)) {
        if (line.trim()) onLine?.(`[${tag}] ${line}`);
      }
    });
  }
}

// Resolve once the publisher has pushed its first frames, which is the point
// the broadcast exists: `moqtsink` applies control messages as frames arrive,
// so a subscriber that attaches before then goes unacknowledged and the relay
// gives up establishing the upstream subscription. Resolves false on timeout,
// so a caller can carry on and let the failure surface where it happens.
export function whenPublishing(reaped, timeoutMs = 20000) {
  return new Promise((res) => {
    const timer = setTimeout(() => res(false), timeoutMs);
    const watch = (d) => {
      if (!d.includes("running...")) return;
      clearTimeout(timer);
      for (const s of [reaped.child.stdout, reaped.child.stderr]) s?.off("data", watch);
      res(true);
    };
    for (const s of [reaped.child.stdout, reaped.child.stderr]) s?.on("data", watch);
  });
}

// Static file server rooted at tools/moqt-demo, so the page can import the
// moqtail ESM straight out of node_modules.
export function startHttp(port) {
  return new Promise((res) => {
    const http = createServer(async (req, resp) => {
      const path = decodeURIComponent(req.url.split("?")[0]);
      const file = join(ROOT, path === "/" ? "/index.html" : path);
      if (!file.startsWith(ROOT)) {
        resp.writeHead(403).end();
        return;
      }
      try {
        const body = await readFile(file);
        resp.writeHead(200, { "content-type": MIME[extname(file)] || "application/octet-stream" });
        resp.end(body);
      } catch {
        resp.writeHead(404).end("not found");
      }
    });
    http.listen(port, "127.0.0.1", () => res(http));
  });
}

// The page URL with everything the player needs to connect.
export function pageUrl(httpPort, relayPort, namespace, hashHex, extra = {}) {
  const q = new URLSearchParams({
    url: `https://127.0.0.1:${relayPort}/`,
    namespace,
    cert: hashHex,
    ...extra,
  });
  return `http://127.0.0.1:${httpPort}/?${q}`;
}

// The query parameters the watch scripts open the page with: autostart, plus
// whatever the environment asks for.
export function playerParams() {
  const extra = { autostart: "1" };
  if (process.env.G2G_MOQT_WEBCODECS) extra.decoder = "webcodecs";
  if (process.env.G2G_MOQT_DEBUG) extra.debug = "1";
  return extra;
}

// True when this host has a camera libcamerasrc could open.
export function hasCamera() {
  return existsSync("/dev/video0");
}

export const SMPTE_PIPELINE =
  "videotestsrc width=320 height=240 pattern=smpte ! videoconvert ! x264enc";
export const CAMERA_PIPELINE =
  "libcamerasrc width=640 height=480 framerate=30 ! videoconvert ! x264enc";

// One MOQT object per this many milliseconds of each track.
const FRAGMENT_MS = 500;

// A producer reference time box ahead of each fragment, mapping its decode time
// to the muxer's wall clock. That is what the page's latency HUD measures
// against; one 32-byte box per fragment, so it is always on here.
const PRFT = "write-prft=true";

// The audio half of the broadcast: a 440 Hz tone encoded as AAC-LC, which is
// what MSE plays as `mp4a.40.2`.
export const AUDIO_PIPELINE = "audiotestsrc ! avenc_aac";

// Complete a source+encoder prefix into a publishing pipeline. With audio the
// two branches meet in the fan-in `mp4mux`, so one `moov` names both tracks and
// `moqtsink` publishes a track each.
export function publishPipeline(prefix, relayPort, namespace, hashHex, { audio = true } = {}) {
  const sink = moqtSink(relayPort, namespace, hashHex);
  // Half-second fragments: one MOQT object per half second of each track
  // rather than one per access unit, which is what a CMAF broadcast looks like
  // and what keeps a browser's append queue ahead of an unpaced publisher.
  const mux = `mp4mux name=mux fragment-duration=${FRAGMENT_MS} ${PRFT}`;
  if (!audio) return `${prefix} ! mp4mux fragment-duration=${FRAGMENT_MS} ${PRFT} ! ${sink}`;
  return `${prefix} ! mux.   ${AUDIO_PIPELINE} ! mux.   ${mux} ! ${sink}`;
}

function moqtSink(relayPort, namespace, hashHex) {
  return (
    `moqtsink location=https://127.0.0.1:${relayPort}/ ` +
    `namespace=${namespace} server-certificate-hashes=${hashHex}`
  );
}

// --- the paced low-latency broadcast --------------------------------------
//
// `videotestsrc ! x264enc` runs as fast as the CPU allows, hundreds of frames a
// second, so a player joining it is fed hours of media per minute and any
// latency it measures is a statement about that, not about the player. A
// recording replayed with `replaysrc sync=true` is paced to the recorded PTS
// instead, which is a live 30 fps source, and it costs a fraction of a second
// to make.

const PACED_FPS = 30;
export const PACED_WIDTH = 320;
export const PACED_HEIGHT = 240;

// Record `seconds` of encoded SMPTE bars into `dir`, returning the clip path.
// Encoding happens here, once and unpaced, so the live run only replays.
export function recordPacedClip(launchBin, dir, seconds) {
  const clip = join(dir, "smpte.g2g");
  const pipeline =
    `videotestsrc width=${PACED_WIDTH} height=${PACED_HEIGHT} pattern=smpte ` +
    `framerate=${PACED_FPS}/1 num-buffers=${Math.round(seconds * PACED_FPS)} ` +
    `! videoconvert ! x264enc ! recordsink location=${clip}`;
  const r = spawnSync(launchBin, [pipeline], { encoding: "utf8" });
  if (r.status !== 0) throw new Error(`recording the paced clip failed: ${r.stderr || r.stdout}`);
  return clip;
}

// Publish a recorded clip at its captured pacing, one MOQT object per access
// unit: the low-latency shape, and the one a WebCodecs player can actually
// exploit (a fragment holding half a second of media cannot be decoded before
// all of it has arrived, whatever the decoder).
export function pacedPublishPipeline(clip, relayPort, namespace, hashHex) {
  return (
    `replaysrc location=${clip} sync=true ` +
    `! mp4mux ${PRFT} ` +
    `! ${moqtSink(relayPort, namespace, hashHex)}`
  );
}
