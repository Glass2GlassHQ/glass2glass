// Subscribe to a g2g MoQ Transport broadcast and play it.
//
// The MoQT client is MOQtail (`moqtail` on npm), a third-party draft-16
// implementation: nothing here shares code with g2g's Rust wire layer, so a
// successful play is an independent decode of our bytes.
//
// The shape it expects is what `moqtsink` publishes: a `.catalog` object naming
// the tracks, an init track holding `ftyp`+`moov`, and one media track per
// track of that `moov`, whose objects are `moof`+`mdat` pairs.
//
// Two decode paths (M942), chosen by the caller:
//
//   `mse`        everything goes to one MSE SourceBuffer unchanged, so the
//                browser's own demuxer and decoders do the work, audio
//                included. One buffer rather than one per track, because the
//                broadcast has a single init segment: its `moov` names every
//                track, which is what a SourceBuffer opened with both codecs
//                expects. Two SourceBuffers would each need an init segment
//                describing only their own track, and no such thing is
//                published. MSE holds a few hundred ms of buffer of its own.
//   `webcodecs`  the page demuxes (see mp4-parse.js) and feeds access units to
//                a `VideoDecoder` one at a time, drawing each `VideoFrame` on a
//                canvas as it comes out. No buffer beyond the decoder's own, so
//                latency is a frame or two rather than a fill level. Video
//                only: there is no `AudioDecoder` here, and pairing one with
//                the video would mean building the A/V sync MSE gives for free.
//
// Both paths measure their own end-to-end latency from the `prft` the muxer
// writes ahead of each fragment; see `LatencyTracker`.
import { FilterType, FullTrackName, GroupOrder, Location, MOQtailClient, RequestError }
  from "./node_modules/moqtail/dist/index.js";
import { parseFragment, parseInit } from "./mp4-parse.js";

// The catalog and init tracks each hold exactly one object, published in group
// 0 before any subscriber exists. A "latest object" filter therefore delivers
// nothing (the relay only forwards what arrives *after* the subscribe), so
// they have to be requested from the start of the track.
const FROM_START = { filterType: FilterType.AbsoluteStart, startLocation: new Location(0, 0) };
// A group is a GOP, so the next group start is the earliest point a decoder
// can begin: the first fragment delivered opens with a keyframe.
const FROM_NEXT_GROUP = { filterType: FilterType.NextGroupStart };

const log = (msg) => console.log(`g2g[moqt] ${msg}`);
const bigints = (_k, v) => (typeof v === "bigint" ? String(v) : v);

// Which half of the broadcast a catalog entry is. `moqtsink` writes the MP4
// sample entry's codec into `selectionParams`, so the codec string says it.
function isAudioTrack(track) {
  return /^(mp4a|opus|ac-3|ec-3|flac)/i.test(track.selectionParams?.codec || "");
}

function hexToBytes(hex) {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
  return out;
}

// A subgroup stream and the SUBSCRIBE_OK that names its track alias are two
// different QUIC streams, so either can arrive first. MOQtail 0.12.1 treats an
// alias it has not seen a SUBSCRIBE_OK for as a protocol violation and tears
// the session down; against a relay holding the object already (a catalog or
// an init segment) that is most of the time.
//
// This player subscribes to one track at a time, so an unclaimed alias can
// only belong to the single SUBSCRIBE still in flight: bind it there and let
// the SUBSCRIBE_OK confirm the same mapping when it lands.
function holdUnclaimedAlias(client) {
  const known = client.subscriptions.get.bind(client.subscriptions);
  client.subscriptions.get = (alias) => {
    const found = known(alias);
    if (found) return found;
    const pending = [...client.requests.values()]
      .filter((r) => r.stream && !client.subscriptionAliasMap.has(r.requestId));
    if (pending.length !== 1) return undefined;
    const request = pending[0];
    client.subscriptions.set(alias, request);
    client.subscriptionAliasMap.set(request.requestId, alias);
    client.aliasFullTrackNameMap.set(alias, request.fullTrackName);
    log(`alias ${alias} arrived before its SUBSCRIBE_OK, bound to request ${request.requestId}`);
    return request;
  };
}

// Subscribe to one track and hand each object's payload to `onObject`. The
// returned promise resolves once the reader has started, not when the track
// ends.
async function subscribeTrack(client, namespace, name, onObject, filter) {
  const sub = await client.subscribe({
    fullTrackName: FullTrackName.tryNew(namespace, name),
    priority: 0,
    groupOrder: GroupOrder.Original,
    forward: true,
    ...filter,
  });
  if (sub instanceof RequestError) {
    throw new Error(`SUBSCRIBE ${name} refused: ${sub.errorCode} ${sub.reasonPhrase}`);
  }
  log(`subscribed ${name} (request ${sub.requestId})`);
  (async () => {
    const reader = sub.stream.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value?.payload?.length) onObject(value.payload);
    }
  })().catch((e) => log(`track ${name} ended: ${e}`));
  return sub;
}

// Wait for the first object on a track: the catalog and the init segment are
// each exactly one object in group 0.
function firstObject(client, namespace, name, timeoutMs) {
  return new Promise((res, rej) => {
    const timer = setTimeout(() => rej(new Error(`no object on ${name} within ${timeoutMs} ms`)), timeoutMs);
    subscribeTrack(client, namespace, name, (payload) => {
      clearTimeout(timer);
      res(payload);
    }, FROM_START).catch((e) => {
      clearTimeout(timer);
      rej(e);
    });
  });
}

// Append to a SourceBuffer one segment at a time. MSE rejects a second append
// while `updating`, so segments queue here.
//
// A live subscriber joins at whatever media time the broadcast has reached,
// so the buffered range does not start at zero and the element would sit at
// `currentTime = 0` with nothing to play. Each append therefore pulls the
// playhead into the buffered range.
class AppendQueue {
  constructor(sourceBuffer, video) {
    this.sb = sourceBuffer;
    this.video = video;
    this.queue = [];
    this.busy = false;
    this.sb.addEventListener("updateend", () => {
      this.busy = false;
      this.seekIntoBuffer();
      this.pump();
    });
  }
  seekIntoBuffer() {
    const b = this.sb.buffered;
    if (!b.length) return;
    const t = this.video.currentTime;
    if (t < b.start(0) || t > b.end(b.length - 1)) {
      this.video.currentTime = b.start(0);
      log(`playhead moved to ${b.start(0).toFixed(3)}s`);
    }
  }
  push(bytes) {
    this.queue.push(bytes);
    this.pump();
  }
  pump() {
    if (this.busy || !this.queue.length || this.sb.updating) return;
    this.busy = true;
    try {
      this.sb.appendBuffer(this.queue.shift());
    } catch (e) {
      this.busy = false;
      log(`append failed: ${e}`);
    }
  }
}

// End-to-end latency, measured the way DASH-IF low-latency players do it.
//
// `mp4mux write-prft=true` writes a producer reference time box ahead of each
// fragment, pinning one decode time to the producer's wall clock. Those pairs
// are the anchors here: the time a frame at any decode time was produced is the
// bracketing anchor's wall clock plus how far past it that frame sits in the
// media timeline. Latency is then the difference between now and that, sampled
// when the frame is put on screen.
//
// The anchor has to be the last one *at or before* the frame, not the newest
// one: extrapolating backwards from a later anchor would assume the producer
// generated media at exactly wall-clock rate, which an unpaced publisher (the
// SMPTE test pipeline outruns real time by a wide margin) does not.
//
// Producer and player are the same machine in this demo, so their clocks are
// the same clock and no offset estimation is needed.
export class LatencyTracker {
  constructor(timescale, windowSize = 120) {
    this.timescale = timescale;
    this.windowSize = windowSize;
    this.anchors = [];
    this.samples = [];
    this.last = null;
    this.count = 0;
  }
  // A `prft`: media time `mediaTime` (in the track timescale) was produced at
  // `epochMs`.
  anchor(mediaTime, epochMs) {
    const top = this.anchors[this.anchors.length - 1];
    if (top && mediaTime <= top.mediaTime) return;
    this.anchors.push({ mediaTime, epochMs });
    // A long run must not grow this without bound; an hour of per-GOP anchors
    // fits well inside this.
    if (this.anchors.length > 4096) this.anchors.splice(0, 2048);
  }
  producedAtMs(mediaTime) {
    if (!this.anchors.length) return null;
    let lo = 0;
    let hi = this.anchors.length - 1;
    if (mediaTime < this.anchors[0].mediaTime) return null;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.anchors[mid].mediaTime <= mediaTime) lo = mid;
      else hi = mid - 1;
    }
    const a = this.anchors[lo];
    return a.epochMs + ((mediaTime - a.mediaTime) / this.timescale) * 1000;
  }
  // Record that the frame at `mediaTime` reached the screen at `nowMs`.
  observe(mediaTime, nowMs) {
    const produced = this.producedAtMs(mediaTime);
    if (produced === null) return;
    this.last = nowMs - produced;
    this.count += 1;
    this.samples.push(this.last);
    if (this.samples.length > this.windowSize) this.samples.shift();
  }
  medianMs() {
    if (!this.samples.length) return null;
    const sorted = [...this.samples].sort((a, b) => a - b);
    return sorted[sorted.length >> 1];
  }
}

// Decode each fragment's access units straight into a `VideoDecoder`, drawing
// every frame as it comes out. Resolves once the subscription is running.
async function playWebCodecs(client, namespace, track, initSegment, canvas, state) {
  if (typeof VideoDecoder === "undefined") throw new Error("browser has no WebCodecs VideoDecoder");
  const init = parseInit(initSegment);
  const codec = track.selectionParams?.codec || "avc1.64001e";
  state.width = init.width;
  state.height = init.height;
  state.latency = new LatencyTracker(init.timescale);
  canvas.width = init.width;
  canvas.height = init.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  log(`webcodecs: ${codec} ${init.width}x${init.height}, ${init.description.length}-byte avcC, ` +
    `timescale ${init.timescale}`);

  const decoder = new VideoDecoder({
    output: (frame) => {
      try {
        ctx.drawImage(frame, 0, 0);
        state.decodedFrames += 1;
        if (state.decodedFrames === 1) log("first frame decoded and drawn");
        // The frame's own timestamp back in media units. Measured at the draw
        // rather than after the compositor picks the canvas up, which is under
        // a frame early and the same order of error as the MSE path's
        // post-presentation callback.
        const mediaTime = Math.round((frame.timestamp * init.timescale) / 1e6);
        state.latency.observe(mediaTime, Date.now());
      } finally {
        frame.close();
      }
    },
    error: (e) => log(`decoder error: ${e}`),
  });
  // `optimizeForLatency` tells the decoder to emit each frame as soon as it can
  // rather than filling a reorder buffer first, which is the whole point here.
  decoder.configure({ codec, description: init.description, optimizeForLatency: true });

  // A decoder handed a delta frame before any keyframe errors out. The
  // subscription starts at a group boundary, so this only bites if the relay
  // hands us a partial group after a reconnect.
  let started = false;
  await subscribeTrack(client, namespace, track.name, (payload) => {
    state.fragments += 1;
    const frag = parseFragment(payload);
    if (frag.prft) state.latency.anchor(frag.prft.mediaTime, frag.prft.epochMs);
    for (const s of frag.samples) {
      if (!started && !s.isSync) continue;
      started = true;
      decoder.decode(new EncodedVideoChunk({
        type: s.isSync ? "key" : "delta",
        timestamp: Math.round((s.pts / init.timescale) * 1e6),
        duration: Math.round((s.duration / init.timescale) * 1e6),
        // AVCC length-prefixed, exactly as it sits in the mdat: what a decoder
        // configured with an avcC `description` expects.
        data: s.data,
      }));
    }
  }, FROM_NEXT_GROUP);
}

// Sample the video track's `prft` for the MSE path, which otherwise never looks
// inside a fragment, and report latency once per presented frame.
function trackMseLatency(video, state) {
  if (!video.requestVideoFrameCallback) {
    log("no requestVideoFrameCallback: MSE latency not measured");
    return;
  }
  const tick = (_now, meta) => {
    state.latency.observe(Math.round(meta.mediaTime * state.latency.timescale), Date.now());
    video.requestVideoFrameCallback(tick);
  };
  video.requestVideoFrameCallback(tick);
}

// Connect, discover the tracks, and play the first video track: into `video`
// through MSE, or into `canvas` through WebCodecs when `decoder` is
// `"webcodecs"`. Resolves with a handle carrying the live counters and the
// latency tracker the caller reports on.
export async function play({
  url, namespace, certHash, video, canvas, decoder = "mse", timeoutMs = 15000, debug = false,
}) {
  const transportOptions = {};
  if (certHash) {
    transportOptions.serverCertificateHashes = [
      { algorithm: "sha-256", value: hexToBytes(certHash) },
    ];
  }
  log(`connecting ${url} namespace=${namespace}`);
  const callbacks = debug ? {
    onMessageSent: (m) => log(`-> ${m.constructor.name} ${JSON.stringify(m, bigints)}`),
    onMessageReceived: (m) => log(`<- ${m.constructor.name} ${JSON.stringify(m, bigints)}`),
    onSessionTerminated: (r) => log(`session terminated: ${r}`),
  } : undefined;
  const client = await MOQtailClient.new({ url, transportOptions, callbacks });
  log("session up, moqt-16 negotiated");
  holdUnclaimedAlias(client);

  const catalogBytes = await firstObject(client, namespace, ".catalog", timeoutMs);
  const catalog = JSON.parse(new TextDecoder().decode(catalogBytes));
  log(`catalog: ${JSON.stringify(catalog.tracks)}`);
  const tracks = catalog.tracks || [];
  const track = tracks.find((t) => !isAudioTrack(t));
  if (!track) throw new Error("catalog names no video track");
  const audio = tracks.find(isAudioTrack);
  const initTrack = track.initTrack || "0.mp4";
  const codecs = [track.selectionParams?.codec || "avc1.64001e"];
  if (audio) codecs.push(audio.selectionParams?.codec || "mp4a.40.2");
  const mime = `video/mp4; codecs="${codecs.join(", ").toLowerCase()}"`;

  const state = {
    mode: decoder, fragments: 0, audioFragments: 0, decodedFrames: 0,
    width: 0, height: 0, latency: new LatencyTracker(90000),
  };

  if (decoder === "webcodecs") {
    const initSegment = await firstObject(client, namespace, initTrack, timeoutMs);
    log(`init segment ${initSegment.length} bytes, decoding with WebCodecs (video only)`);
    await playWebCodecs(client, namespace, track, initSegment, canvas, state);
    return state;
  }

  if (!MediaSource.isTypeSupported(mime)) throw new Error(`browser cannot play ${mime}`);
  const initSegment = await firstObject(client, namespace, initTrack, timeoutMs);
  log(`init segment ${initSegment.length} bytes, mime ${mime}`);

  const mediaSource = new MediaSource();
  video.src = URL.createObjectURL(mediaSource);
  const sourceBuffer = await new Promise((res) => {
    mediaSource.addEventListener("sourceopen", () => {
      const sb = mediaSource.addSourceBuffer(mime);
      // The publisher's fragments carry their own decode times, so the buffer
      // must not restamp them.
      sb.mode = "segments";
      res(sb);
    }, { once: true });
  });

  state.latency = new LatencyTracker(parseInit(initSegment).timescale);
  const queue = new AppendQueue(sourceBuffer, video);
  queue.push(initSegment);
  trackMseLatency(video, state);
  await subscribeTrack(client, namespace, track.name, (payload) => {
    state.fragments += 1;
    const prft = parseFragment(payload).prft;
    if (prft) state.latency.anchor(prft.mediaTime, prft.epochMs);
    queue.push(payload);
    if (state.fragments === 1) log("first video fragment appended");
  }, FROM_NEXT_GROUP);
  if (audio) {
    await subscribeTrack(client, namespace, audio.name, (payload) => {
      state.audioFragments += 1;
      queue.push(payload);
      if (state.audioFragments === 1) log("first audio fragment appended");
    }, FROM_NEXT_GROUP);
  } else {
    log("catalog names no audio track: video only");
  }

  // Not awaited: play() stays pending until the element has data, which is
  // after the first fragments land.
  video.play().then(() => log("playing")).catch((e) => log(`play() rejected: ${e}`));
  return state;
}

// Seven columns across the top of whatever is on the canvas: the SMPTE bar
// centres.
function sampleBars(canvas) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const y = Math.floor(canvas.height * 0.15);
  const bars = [];
  for (let i = 0; i < 7; i++) {
    const x = Math.floor((canvas.width * (i + 0.5)) / 7);
    const [r, g, b] = ctx.getImageData(x, y, 1, 1).data;
    bars.push([r, g, b]);
  }
  return bars;
}

// Read back what the decoder actually produced: dimensions, the frame count,
// the measured latency, and the pixels on screen. In WebCodecs mode the canvas
// *is* the display, so the bars come straight off it; in MSE mode the video
// element is drawn into it first.
export function playbackReport(state, video, canvas) {
  if (state.mode === "webcodecs") {
    return {
      mode: state.mode,
      width: canvas.width,
      height: canvas.height,
      currentTime: null,
      totalVideoFrames: state.decodedFrames,
      droppedVideoFrames: 0,
      audioDecodedBytes: null,
      latencyMedianMs: state.latency.medianMs(),
      latencyLastMs: state.latency.last,
      latencySamples: state.latency.count,
      bars: state.decodedFrames ? sampleBars(canvas) : null,
    };
  }
  const q = video.getVideoPlaybackQuality?.() || { totalVideoFrames: 0, droppedVideoFrames: 0 };
  const report = {
    mode: state.mode,
    width: video.videoWidth,
    height: video.videoHeight,
    currentTime: video.currentTime,
    totalVideoFrames: q.totalVideoFrames,
    droppedVideoFrames: q.droppedVideoFrames,
    // Chromium's decoder-side byte counters: audio bytes past zero mean the
    // browser decoded the audio track, not just buffered it. `null` on an
    // engine that does not expose them.
    audioDecodedBytes:
      typeof video.webkitAudioDecodedByteCount === "number"
        ? video.webkitAudioDecodedByteCount
        : null,
    latencyMedianMs: state.latency.medianMs(),
    latencyLastMs: state.latency.last,
    latencySamples: state.latency.count,
    bars: null,
  };
  if (canvas && video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d", { willReadFrequently: true }).drawImage(video, 0, 0);
    report.bars = sampleBars(canvas);
  }
  return report;
}
