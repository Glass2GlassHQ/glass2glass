# g2g MoQ Transport browser demo

A browser subscribes to a broadcast g2g is publishing over IETF MoQ Transport
(`... ! mp4mux ! moqtsink` into a local `moq-relay-ietf`) and plays it, video and
audio.

The MoQT client in the page is [MOQtail](https://github.com/moqtail/moqtail)
(`moqtail` on npm), a third-party draft-16 implementation. Nothing in the browser
shares code with g2g's Rust wire layer, so a successful play is an independent
decode of our bytes: the page reads our `.catalog`, fetches the init track, and
hands the `moof`+`mdat` objects to Media Source Extensions unchanged.

## Decode modes

`?decoder=webcodecs` swaps MSE for WebCodecs. The page then demuxes itself
(`mp4-parse.js`): it pulls the `avcC` out of the init segment to configure a
`VideoDecoder`, cuts each fragment's `trun` into individual access units, and
draws each `VideoFrame` on a canvas as it comes out. Nothing buffers, so latency
is a frame or two rather than MSE's fill level. **Video only**: pairing an
`AudioDecoder` with it would mean building the A/V sync MSE gives for free, so
the audio track is left unsubscribed. The default stays MSE, audio included.

Both modes show a latency HUD in the top right: the wall clock the muxer wrote
into the fragment's `prft` against the moment the frame reached the screen, as
the current value and a running median over the last 120 frames. Producer and
player are the same machine here, so the two clocks are one clock and no offset
estimation is involved.

The number only means something against a live-paced source. Every publisher
here is one: a camera and the bevy render loop are paced by their own capture,
the test pattern by `clocksync`, which holds each frame until its PTS is due on
the clock, and the recorded clip the latency check replays by `replaysrc
sync=true`. Drop the `clocksync` out of the test-pattern line and it encodes
hundreds of frames a second, the media timeline runs ahead of the wall clock,
and the HUD reads whatever that implies, including a negative figure.

## Prerequisites

- `pnpm install` here (pulls `moqtail` and `playwright`).
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

One command: mints a certificate, starts the relay, publishes the camera and a
440 Hz test tone (`libcamerasrc ! videoconvert ! x264enc ! mux.  audiotestsrc !
avenc_aac ! mux.  mp4mux name=mux ! moqtsink`), serves the page and opens your
browser on it once frames are flowing. The page starts muted, because browsers
block unmuted autoplay: unmute it to hear the tone, or publish video only with
`G2G_MOQT_NO_AUDIO=1`. `moqtsink` applies control
messages as frames arrive, so a subscriber that attaches before the first frame
goes unacknowledged and the relay refuses the subscribe; the wait avoids that,
and a camera takes a second or two to start. Ctrl-C stops everything. With no
`/dev/video0` it
publishes the SMPTE test pattern instead and says so; `G2G_MOQT_PATTERN=1` forces
that, `G2G_CAMERA_SIZE=1280x720` changes the capture size.
`G2G_MOQT_WEBCODECS=1` opens the page in WebCodecs mode (same for
`watch-bevy.mjs`).

## Headless checks

```sh
node headless/run-moqt-play.mjs      # MSE, video + audio
node headless/run-moqt-latency.mjs   # MSE against WebCodecs, two decode modes
```

The first publishes the SMPTE pattern and the tone, both paced by `clocksync`,
drives the page in headless Chromium, and asserts on what the browser's own
decoders produced: at least 10 decoded frames, a decoded size of 320x240, the
seven SMPTE bars in order sampled off a canvas the `<video>` was drawn into, and
audio fragments both appended and decoded (`webkitAudioDecodedByteCount` past
zero).

The second plays one broadcast twice, once per decode mode, asserting the frames
and the bars for each (WebCodecs samples them straight off the canvas it drew
to) and reporting the median end-to-end latency of both. It replays a recorded
clip (`replaysrc sync=true`) rather than encoding live, so both passes measure
the same bytes with no encoder competing for the CPU. Measured here, MSE sits
around 45 ms and WebCodecs around 1 ms.

Both print `SKIP` and exit 0 when the relay, the launcher, Chromium or the npm
deps are missing. `G2G_MOQT_DEBUG=1` logs every MoQT control message the page
sends and receives.

## Certificates

QUIC is always TLS, so the page needs `serverCertificateHashes`: browsers accept
that only for an ECDSA P-256 certificate valid at most 14 days. A self-signed
certificate used directly as the leaf is rejected (`CaUsedAsEndEntity`), so
`local-relay.mjs` mints a CA and a leaf signed by it and feeds the relay the
`leaf, CA` chain. Both scripts do this per run; nothing is left behind.

## Files

| Path | What |
| :--- | :--- |
| `index.html` | the player page; reads `?url=&namespace=&cert=&decoder=&autostart=&debug=` |
| `moqt-player.js` | subscribe via MOQtail; MSE or WebCodecs decode, and the latency tracker |
| `mp4-parse.js` | the ISO-BMFF the WebCodecs path demuxes itself: `avcC`, `trun` samples, `prft` |
| `local-relay.mjs` | certificate minting, relay and publisher startup, static server |
| `watch-live.mjs` | the live camera demo, one command |
| `watch-bevy.mjs` | the bevy-g2g remote-render demo, one command |
| `headless/common.mjs` | binaries, browser and assertions shared by the two headless runs |
| `headless/run-moqt-play.mjs` | the MSE run and its assertions |
| `headless/run-moqt-latency.mjs` | the two-mode latency comparison |

## Why one SourceBuffer

The broadcast has a single init segment whose `moov` names every track, which is
what `mp4mux` writes and what `moqtsink` publishes on the init track. A
SourceBuffer opened with both codecs (`video/mp4; codecs="avc1..., mp4a.40.2"`)
takes that init segment and then both tracks' fragments. Two SourceBuffers would
each need an init segment describing only their own track, and the broadcast
carries no such thing.

## Fragment size

One MOQT object per access unit, everywhere. That is the low-latency shape and
the only one a WebCodecs player can exploit: a fragment holding half a second of
media cannot be decoded before all of it has arrived, whatever the decoder. It
is viable because every publisher is paced to the wall clock, so the objects
arrive at the rate the browser consumes them.

## Known interop note

MOQtail 0.12.1 fails the session when a subgroup stream arrives before the
SUBSCRIBE_OK naming its track alias. Those are two different QUIC streams and
either can land first, so against a relay that already holds the object (a
catalog or an init segment) it happens most of the time. `moqt-player.js` binds
such an alias to the single SUBSCRIBE still in flight instead; g2g's own
subscriber holds the objects until SUBSCRIBE_OK arrives, which is what draft-16
asks for.
