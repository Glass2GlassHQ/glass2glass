// What the two headless runs (run-moqt-play.mjs, run-moqt-latency.mjs) have in
// common: finding the binaries, driving a real Chromium at the player page, and
// reading the SMPTE bars back off the canvas.
import { existsSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";
import { ROOT, chromeBinary, launchBinary, relayBinary } from "../local-relay.mjs";

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

// The bars that read wrong, as messages; empty when all seven are right.
export function wrongBars(bars) {
  if (!bars) return ["no pixels were sampled"];
  return BARS.map(([name, ...want], i) => {
    const px = bars[i];
    const ok = want.every((lit, c) => (lit ? px[c] >= LIT : px[c] <= DARK));
    return ok ? null : `${name} bar reads rgb(${px})`;
  }).filter(Boolean);
}

// Everything a run needs, or the reason it has to skip.
export function prereqs() {
  const relayBin = relayBinary();
  if (!relayBin) {
    return { skip: "moq-relay-ietf not found. Build it with `cargo build --release -p moq-relay-ietf` " +
      "in a cloudflare/moq-rs checkout, or point $MOQ_RS_BIN at its directory." };
  }
  const launchBin = launchBinary();
  if (!launchBin) {
    return { skip: "g2g-launch not built. Run `cargo build --release -p g2g-plugins " +
      "--features moqt,ffmpeg --bin g2g-launch`, or set $G2G_LAUNCH." };
  }
  const chrome = chromeBinary();
  if (!chrome) {
    return { skip: "no full Chromium found (headless_shell cannot decode H.264). " +
      "Run `npx playwright install chromium`, or set $G2G_CHROME." };
  }
  if (!existsSync(join(ROOT, "node_modules/moqtail/dist/index.js"))) {
    return { skip: "moqtail not installed. Run `pnpm install` in tools/moqt-demo." };
  }
  const pwPath = process.env.G2G_PLAYWRIGHT || join(ROOT, "node_modules/playwright/index.js");
  if (!existsSync(pwPath)) {
    return { skip: "playwright not installed. Run `pnpm install` in tools/moqt-demo, or set $G2G_PLAYWRIGHT." };
  }
  return { relayBin, launchBin, chrome, pwPath };
}

// SwiftShader rather than a real GPU, so the run is the same on a headless CI
// box as here.
export async function launchBrowser(pwPath, chrome) {
  const pw = await import(pathToFileURL(pwPath).href);
  const { chromium } = pw.default || pw;
  return chromium.launch({
    headless: !process.env.G2G_HEADFUL,
    executablePath: chrome,
    args: ["--no-sandbox", "--autoplay-policy=no-user-gesture-required",
      "--use-gl=angle", "--use-angle=swiftshader"],
  });
}

export async function openPlayer(browser, url, log) {
  const page = await browser.newPage();
  page.on("console", (m) => { if (m.text().startsWith("g2g[")) log("page:", m.text()); });
  page.on("pageerror", (e) => log("page error:", String(e)));
  log("navigating", url);
  await page.goto(url);
  if (!(await page.evaluate(() => typeof WebTransport !== "undefined"))) {
    throw new Error("browser has no WebTransport");
  }
  return page;
}

// Poll the page until `ready(report, state)` says the run has what it came for,
// or the timeout runs out. Returns the last report read (with the page's own
// counters folded in), or throws if the player never reached playback.
export async function waitForReport(page, { timeoutMs, ready }) {
  const t0 = Date.now();
  let report = null;
  while (Date.now() - t0 < timeoutMs) {
    const state = await page.evaluate(() => window.g2gState);
    if (state.error) throw new Error(`player failed: ${state.error}`);
    if (state.started) {
      report = await page.evaluate(() => window.g2gReport());
      report.fragments = state.fragments;
      report.audioFragments = state.audioFragments;
      if (ready(report)) break;
    }
    await page.waitForTimeout(250);
  }
  if (!report) throw new Error("the player never reached playback");
  return report;
}
