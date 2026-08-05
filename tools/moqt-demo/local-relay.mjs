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
export function chromeBinary() {
  if (process.env.G2G_CHROME) {
    return existsSync(process.env.G2G_CHROME) ? process.env.G2G_CHROME : null;
  }
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
  for (const p of ["/usr/bin/google-chrome", "/usr/bin/chromium-browser", "/usr/bin/chromium"]) {
    if (existsSync(p)) return p;
  }
  return null;
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
      for (const line of d.split("\n")) {
        if (line.trim()) onLine?.(`[${tag}] ${line}`);
      }
    });
  }
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

// True when this host has a camera libcamerasrc could open.
export function hasCamera() {
  return existsSync("/dev/video0");
}

export const SMPTE_PIPELINE =
  "videotestsrc width=320 height=240 pattern=smpte ! videoconvert ! x264enc";
export const CAMERA_PIPELINE =
  "libcamerasrc width=640 height=480 framerate=30 ! videoconvert ! x264enc";

// Complete a source+encoder prefix into a publishing pipeline.
export function publishPipeline(prefix, relayPort, namespace, hashHex) {
  return `${prefix} ! mp4mux ! moqtsink location=https://127.0.0.1:${relayPort}/ ` +
    `namespace=${namespace} server-certificate-hashes=${hashHex}`;
}
