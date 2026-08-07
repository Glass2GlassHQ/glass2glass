// The little bit of ISO-BMFF the WebCodecs path has to read itself.
//
// Media Source Extensions takes `moof`+`mdat` whole; WebCodecs does not, so a
// player that skips MSE has to do the demuxing MSE was doing: pull the
// `avcC` out of the init segment to configure the decoder, and cut each
// fragment into the individual access units an `EncodedVideoChunk` wants.
// Sample data stays AVCC (length-prefixed) exactly as it sits in the `mdat`,
// which is what a decoder configured with a `description` expects.
//
// The `prft` reader serves the latency HUD and is used by both decode paths:
// it is what maps a fragment's decode time to the producer's wall clock.

// Walk the top-level boxes of `buf` between `start` and `end`.
export function* boxes(buf, start = 0, end = buf.length) {
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let at = start;
  while (at + 8 <= end) {
    let size = view.getUint32(at);
    const type = String.fromCharCode(buf[at + 4], buf[at + 5], buf[at + 6], buf[at + 7]);
    let head = 8;
    if (size === 1) {
      if (at + 16 > end) return;
      // 64-bit largesize. Past 2^53 the arithmetic is wrong anyway, and no
      // fragment is that big.
      size = view.getUint32(at + 8) * 2 ** 32 + view.getUint32(at + 12);
      head = 16;
    }
    if (size < head || at + size > end) return;
    yield { type, start: at, body: at + head, end: at + size };
    at += size;
  }
}

// The first child box of `type`, or null.
export function findBox(buf, type, start, end) {
  for (const b of boxes(buf, start, end)) if (b.type === type) return b;
  return null;
}

// Follow a chain of box types down from `start`, e.g. ["moov", "trak", "mdia"].
function descend(buf, path, start = 0, end = buf.length) {
  let box = { body: start, end };
  for (const type of path) {
    box = findBox(buf, type, box.body, box.end);
    if (!box) return null;
  }
  return box;
}

// An NTP 64-bit timestamp as milliseconds since the Unix epoch.
function ntpToEpochMs(view, at) {
  const NTP_UNIX_OFFSET = 2_208_988_800;
  const secs = view.getUint32(at) - NTP_UNIX_OFFSET;
  const frac = view.getUint32(at + 4) / 2 ** 32;
  return secs * 1000 + frac * 1000;
}

// What the WebCodecs decoder needs out of the init segment: the `avcC` for its
// `description`, the coded size, and the media timescale every decode time in
// the fragments is expressed in. Throws when the moov names no H.264 track,
// which is the only shape this player handles.
export function parseInit(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const moov = findBox(buf, "moov", 0, buf.length);
  if (!moov) throw new Error("init segment has no moov");

  let trackId = 0;
  for (const trak of boxes(buf, moov.body, moov.end)) {
    if (trak.type !== "trak") continue;
    trackId += 1;
    const stsd = descend(buf, ["mdia", "minf", "stbl", "stsd"], trak.body, trak.end);
    const mdhd = descend(buf, ["mdia", "mdhd"], trak.body, trak.end);
    if (!stsd || !mdhd) continue;
    // stsd is a full box with a u32 entry count before the sample entries.
    for (const entry of boxes(buf, stsd.body + 8, stsd.end)) {
      if (entry.type !== "avc1" && entry.type !== "avc3") continue;
      // VisualSampleEntry: 8 bytes of SampleEntry, then 16 reserved, then the
      // coded size; its child boxes (avcC among them) start 78 bytes in.
      const width = view.getUint32(entry.body + 24) >>> 16;
      const height = view.getUint32(entry.body + 24) & 0xffff;
      const avcC = findBox(buf, "avcC", entry.body + 78, entry.end);
      if (!avcC) throw new Error("avc1 sample entry has no avcC");
      const version = buf[mdhd.body];
      const timescale = view.getUint32(mdhd.body + (version === 1 ? 20 : 12));
      return {
        trackId,
        width,
        height,
        timescale,
        description: buf.slice(avcC.body, avcC.end),
      };
    }
  }
  throw new Error("init segment names no H.264 track");
}

// tfhd flags.
const TFHD_BASE_DATA_OFFSET = 0x000001;
const TFHD_SAMPLE_DESCRIPTION = 0x000002;
const TFHD_DEFAULT_DURATION = 0x000008;
const TFHD_DEFAULT_SIZE = 0x000010;
const TFHD_DEFAULT_FLAGS = 0x000020;
// trun flags.
const TRUN_DATA_OFFSET = 0x000001;
const TRUN_FIRST_SAMPLE_FLAGS = 0x000004;
const TRUN_SAMPLE_DURATION = 0x000100;
const TRUN_SAMPLE_SIZE = 0x000200;
const TRUN_SAMPLE_FLAGS = 0x000400;
const TRUN_SAMPLE_CTS = 0x000800;
// Set in a sample's flags when it is *not* a sync sample.
const SAMPLE_NON_SYNC = 0x00010000;

// One MOQT media object: whatever `styp` / `prft` opened the fragment, its
// `moof`, and its `mdat`. Returns the producer reference time (when the
// fragment carries one) and the fragment's samples, each a slice of `bytes`.
//
// Sample data is addressed the way CMAF requires: `default-base-is-moof`, so a
// trun's data offset is relative to the start of its own `moof`.
export function parseFragment(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const out = { prft: null, trackId: 0, samples: [] };

  for (const top of boxes(buf)) {
    if (top.type === "prft") {
      const version = buf[top.body];
      const at = top.body + 4;
      out.prft = {
        trackId: view.getUint32(at),
        epochMs: ntpToEpochMs(view, at + 4),
        // version 1 stores a 64-bit media time; only the low half can matter
        // at any plausible timescale.
        mediaTime:
          version === 1
            ? view.getUint32(at + 12) * 2 ** 32 + view.getUint32(at + 16)
            : view.getUint32(at + 12),
      };
      continue;
    }
    if (top.type !== "moof") continue;
    const traf = findBox(buf, "traf", top.body, top.end);
    if (!traf) continue;
    const tfhd = findBox(buf, "tfhd", traf.body, traf.end);
    const trun = findBox(buf, "trun", traf.body, traf.end);
    const tfdt = findBox(buf, "tfdt", traf.body, traf.end);
    if (!tfhd || !trun) continue;

    let at = tfhd.body + 4;
    const tfhdFlags = view.getUint32(tfhd.body) & 0xffffff;
    out.trackId = view.getUint32(at);
    at += 4;
    let baseDataOffset = null;
    if (tfhdFlags & TFHD_BASE_DATA_OFFSET) {
      baseDataOffset = view.getUint32(at) * 2 ** 32 + view.getUint32(at + 4);
      at += 8;
    }
    if (tfhdFlags & TFHD_SAMPLE_DESCRIPTION) at += 4;
    let defaultDuration = 0;
    if (tfhdFlags & TFHD_DEFAULT_DURATION) {
      defaultDuration = view.getUint32(at);
      at += 4;
    }
    let defaultSize = 0;
    if (tfhdFlags & TFHD_DEFAULT_SIZE) {
      defaultSize = view.getUint32(at);
      at += 4;
    }
    let defaultFlags = 0;
    if (tfhdFlags & TFHD_DEFAULT_FLAGS) {
      defaultFlags = view.getUint32(at);
      at += 4;
    }

    let dts = 0;
    if (tfdt) {
      dts =
        buf[tfdt.body] === 1
          ? view.getUint32(tfdt.body + 4) * 2 ** 32 + view.getUint32(tfdt.body + 8)
          : view.getUint32(tfdt.body + 4);
    }

    const trunFlags = view.getUint32(trun.body) & 0xffffff;
    const count = view.getUint32(trun.body + 4);
    let t = trun.body + 8;
    let dataOffset = 0;
    if (trunFlags & TRUN_DATA_OFFSET) {
      dataOffset = view.getInt32(t);
      t += 4;
    }
    let firstSampleFlags = null;
    if (trunFlags & TRUN_FIRST_SAMPLE_FLAGS) {
      firstSampleFlags = view.getUint32(t);
      t += 4;
    }
    // Absent a base_data_offset the samples follow the moof, which is what
    // default-base-is-moof means.
    let offset = (baseDataOffset ?? top.start) + dataOffset;

    for (let i = 0; i < count; i++) {
      let duration = defaultDuration;
      let size = defaultSize;
      let flags = i === 0 && firstSampleFlags !== null ? firstSampleFlags : defaultFlags;
      let cts = 0;
      if (trunFlags & TRUN_SAMPLE_DURATION) {
        duration = view.getUint32(t);
        t += 4;
      }
      if (trunFlags & TRUN_SAMPLE_SIZE) {
        size = view.getUint32(t);
        t += 4;
      }
      if (trunFlags & TRUN_SAMPLE_FLAGS) {
        flags = view.getUint32(t);
        t += 4;
      }
      if (trunFlags & TRUN_SAMPLE_CTS) {
        cts = view.getInt32(t);
        t += 4;
      }
      if (offset + size > buf.length) break;
      out.samples.push({
        dts,
        pts: dts + cts,
        duration,
        isSync: !(flags & SAMPLE_NON_SYNC),
        data: buf.subarray(offset, offset + size),
      });
      offset += size;
      dts += duration;
    }
  }
  return out;
}
