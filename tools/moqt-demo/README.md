# g2g MoQ Transport browser demo

A browser subscribes to a broadcast g2g is publishing over IETF MoQ Transport
(`... ! mp4mux ! moqtsink` into a local `moq-relay-ietf`) and plays it.

The MoQT client in the page is [MOQtail](https://github.com/moqtail/moqtail)
(`moqtail` on npm), a third-party draft-16 implementation. Nothing in the browser
shares code with g2g's Rust wire layer, so a successful play is an independent
decode of our bytes: the page reads our `.catalog`, fetches the init track, and
hands the `moof`+`mdat` objects to Media Source Extensions unchanged.

## Prerequisites

- `npm install` here (pulls `moqtail` and `playwright`).
- A `moq-relay-ietf` build: `cargo build --release -p moq-relay-ietf` in a
  [cloudflare/moq-rs](https://github.com/cloudflare/moq-rs) checkout. Found under
  `$HOME/src/moq-rs/target/release`, on `PATH`, or at `$MOQ_RS_BIN`.
- The launcher: `cargo build --release -p g2g-plugins --features libcamera,moqt,ffmpeg
  --bin g2g-launch` (drop `libcamera` if you only want the test pattern).
- A full Chromium. `headless_shell` ships no H.264 and cannot play this.

## Watch it live

```sh
node watch-live.mjs
```

One command: mints a certificate, starts the relay, publishes the camera
(`libcamerasrc ! videoconvert ! x264enc ! mp4mux ! moqtsink`), serves the page and
opens your browser on it once frames are flowing. `moqtsink` applies control
messages as frames arrive, so a subscriber that attaches before the first frame
goes unacknowledged and the relay refuses the subscribe; the wait avoids that,
and a camera takes a second or two to start. Ctrl-C stops everything. With no
`/dev/video0` it
publishes the SMPTE test pattern instead and says so; `G2G_MOQT_PATTERN=1` forces
that, `G2G_CAMERA_SIZE=1280x720` changes the capture size.

## Headless check

```sh
node headless/run-moqt-play.mjs
```

Publishes the SMPTE pattern, drives the page in headless Chromium, and asserts on
what the browser's own decoder produced: at least 10 decoded frames, a decoded
size of 320x240, and the seven SMPTE bars in order sampled off a canvas the
`<video>` was drawn into. Prints `SKIP` and exits 0 when the relay, the launcher,
Chromium or the npm deps are missing. `G2G_MOQT_DEBUG=1` logs every MoQT control
message the page sends and receives.

## Certificates

QUIC is always TLS, so the page needs `serverCertificateHashes`: browsers accept
that only for an ECDSA P-256 certificate valid at most 14 days. A self-signed
certificate used directly as the leaf is rejected (`CaUsedAsEndEntity`), so
`local-relay.mjs` mints a CA and a leaf signed by it and feeds the relay the
`leaf, CA` chain. Both scripts do this per run; nothing is left behind.

## Files

| Path | What |
| :--- | :--- |
| `index.html` | the player page; reads `?url=&namespace=&cert=&autostart=&debug=` |
| `moqt-player.js` | subscribe via MOQtail, catalog + init + media into one MSE SourceBuffer |
| `local-relay.mjs` | certificate minting, relay and publisher startup, static server |
| `watch-live.mjs` | the live camera demo, one command |
| `headless/run-moqt-play.mjs` | the headless run and its assertions |

## Known interop note

MOQtail 0.12.1 fails the session when a subgroup stream arrives before the
SUBSCRIBE_OK naming its track alias. Those are two different QUIC streams and
either can land first, so against a relay that already holds the object (a
catalog or an init segment) it happens most of the time. `moqt-player.js` binds
such an alias to the single SUBSCRIBE still in flight instead; g2g's own
subscriber holds the objects until SUBSCRIBE_OK arrives, which is what draft-16
asks for.
