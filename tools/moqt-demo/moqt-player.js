// Subscribe to a g2g MoQ Transport broadcast and play it.
//
// The MoQT client is MOQtail (`moqtail` on npm), a third-party draft-16
// implementation: nothing here shares code with g2g's Rust wire layer, so a
// successful play is an independent decode of our bytes.
//
// The shape it expects is what `moqtsink` publishes: a `.catalog` object naming
// the tracks, an init track holding `ftyp`+`moov`, and one media track per
// track of that `moov`, whose objects are `moof`+`mdat` pairs. Everything goes
// to one MSE SourceBuffer unchanged, so the browser's own demuxer and decoders
// do the work.
//
// One buffer rather than one per track, because the broadcast has a single init
// segment: its `moov` names every track, which is exactly what a SourceBuffer
// opened with both codecs expects. Two SourceBuffers would each need an init
// segment describing only their own track, and no such thing is published.
import { FilterType, FullTrackName, GroupOrder, Location, MOQtailClient, RequestError }
  from "./node_modules/moqtail/dist/index.js";

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

// Connect, discover the tracks, and play the first video track into `video`.
// Resolves with a handle carrying the live counters the caller asserts on.
export async function play({ url, namespace, certHash, video, timeoutMs = 15000, debug = false }) {
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

  const state = { fragments: 0, audioFragments: 0 };
  const queue = new AppendQueue(sourceBuffer, video);
  queue.push(initSegment);
  await subscribeTrack(client, namespace, track.name, (payload) => {
    state.fragments += 1;
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

// Read back what the decoder actually produced: dimensions, the frame count
// the pipeline reports, and the pixels on screen.
export function playbackReport(video, canvas) {
  const q = video.getVideoPlaybackQuality?.() || { totalVideoFrames: 0, droppedVideoFrames: 0 };
  const report = {
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
    bars: null,
  };
  if (canvas && video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(video, 0, 0);
    // Seven columns across the top of the frame: the SMPTE bar centres.
    const y = Math.floor(canvas.height * 0.15);
    report.bars = [];
    for (let i = 0; i < 7; i++) {
      const x = Math.floor((canvas.width * (i + 0.5)) / 7);
      const [r, g, b] = ctx.getImageData(x, y, 1, 1).data;
      report.bars.push([r, g, b]);
    }
  }
  return report;
}
