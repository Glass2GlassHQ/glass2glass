//! Conformance batteries (M614): the cases that produce the [`MaturityRecord`]s the
//! maturity report is derived from.
//!
//! Each battery here exercises a *real* element (never a mock) with cheap,
//! self-contained checks and adds a piece of [`Evidence`] only when a check actually
//! passes. So the maturity a battery reports is computed from behavior observed in
//! this process, not asserted by hand: a regression that breaks a round-trip drops
//! the derived level, and the honest ceiling of a loopback-only check is
//! `UnitTested` (no external-peer `Oracle` evidence is emitted, so the ST 2110 cores
//! surface as "unit-tested, interop pending" rather than claiming more).
//!
//! Only in-process, dependency-free checks run here, so `g2g-inspect --maturity` can
//! run the whole battery live. `Oracle` (ffmpeg / reference-gear), `Hardware`
//! (GPU / device), and `Quality` (codec-feature-gated) evidence is produced by the
//! feature-gated / host-gated integration tests that own those resources, not by this
//! always-on battery.
//!
//! The measurement helpers those `Quality` batteries share live here too:
//! [`fnv1a_64`] for a committed golden digest, and [`psnr_db`] / [`pooled_psnr_db`]
//! for a fidelity floor against a reference image.

use alloc::vec::Vec;

use g2g_core::conformance::{
    ConformanceDimension as D, ConformanceReport, Evidence, MaturityRecord,
};
use g2g_core::RawVideoFormat;

use crate::st2110dup::SeamlessDedup;
use crate::st2110video::{Sampling, St2110VideoDepacketizer, St2110VideoPacketizer};

/// FNV-1a 64: a dependency-free stable digest for a committed golden. Used by the
/// decoder-regression batteries to pin decoded pixels / samples to a value in the
/// test, so a silent change in decode output fails rather than passing unnoticed.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The three planes of an I420 buffer of `width` x `height`, or `None` if the buffer
/// is not exactly that size (odd geometry included, which I420 cannot represent).
pub fn i420_planes(bytes: &[u8], width: usize, height: usize) -> Option<[&[u8]; 3]> {
    if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return None;
    }
    let luma = width.checked_mul(height)?;
    let chroma = (width / 2).checked_mul(height / 2)?;
    if bytes.len() != luma.checked_add(chroma.checked_mul(2)?)? {
        return None;
    }
    let (y, rest) = bytes.split_at(luma);
    let (u, v) = rest.split_at(chroma);
    Some([y, u, v])
}

/// Peak signal-to-noise ratio in dB of `measured` against `reference`, both 8-bit
/// samples of one plane. `None` on a length mismatch or an empty plane; infinite
/// when the two are identical (zero error has no finite dB value, and a threshold
/// comparison against it still reads correctly).
///
/// Needs `std` for `log10`; the `no_std` baseline has no float math.
#[cfg(feature = "std")]
pub fn psnr_db(reference: &[u8], measured: &[u8]) -> Option<f64> {
    if reference.is_empty() || reference.len() != measured.len() {
        return None;
    }
    let sum: f64 = reference
        .iter()
        .zip(measured)
        .map(|(&a, &b)| {
            let d = a as f64 - b as f64;
            d * d
        })
        .sum();
    Some(psnr_from_mse(sum / reference.len() as f64))
}

/// PSNR in dB over several planes at once, with the mean squared error pooled by
/// sample count so a large luma plane weighs more than the chroma planes. This is
/// the aggregate figure the encode / decode batteries assert against.
#[cfg(feature = "std")]
pub fn pooled_psnr_db(planes: &[(&[u8], &[u8])]) -> Option<f64> {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (reference, measured) in planes {
        if reference.is_empty() || reference.len() != measured.len() {
            return None;
        }
        for (&a, &b) in reference.iter().zip(*measured) {
            let d = a as f64 - b as f64;
            sum += d * d;
        }
        count += reference.len();
    }
    if count == 0 {
        return None;
    }
    Some(psnr_from_mse(sum / count as f64))
}

/// dB for a mean squared error over 8-bit samples (peak 255).
#[cfg(feature = "std")]
fn psnr_from_mse(mse: f64) -> f64 {
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Conformance of the ST 2110-20 (RFC 4175) video packetizer / depacketizer core.
///
/// Verifies it constructs (`Instantiate`), round-trips a frame byte-exact through
/// packetize -> depacketize (`RoundTrip`), and reconstructs a frame under ST 2110-7
/// redundant-path loss via the sequence-number merge (`LossResilience`). It emits no
/// `Oracle` evidence: it has not been validated against reference -20 gear, so it
/// tops out at `UnitTested`.
pub fn st2110_video() -> MaturityRecord {
    let mut rec = MaturityRecord::new("st2110video");
    let (w, h) = (8usize, 8usize);

    // Instantiate: the packetizer / depacketizer construct for a real sampling.
    if St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h).is_some() {
        rec.add(Evidence::new(D::Instantiate));
    }

    // RoundTrip: an RGBA frame survives packetize -> depacketize byte-exact.
    let frame: Vec<u8> = (0..w * 4 * h).map(|i| (i * 7 + 1) as u8).collect();
    let mut tx = St2110VideoPacketizer::new(96, 0xABCD, Sampling::Rgba8, 60);
    if let Some(packets) = tx.packetize(&frame, w, h, 1_000_000_000) {
        if let Some(mut rx) = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h) {
            let mut out = None;
            for p in &packets {
                if let Some(f) = rx.depacketize(p) {
                    out = Some(f.bytes);
                }
            }
            if out.as_deref() == Some(frame.as_slice()) {
                rec.add(
                    Evidence::new(D::RoundTrip)
                        .codec("rgba8")
                        .detail("packetize/depacketize loopback"),
                );
            }
        }
    }

    // LossResilience: the same frame reconstructs when each of two redundant paths
    // drops a different subset of packets (never the marker, none lost on both),
    // merged by the -7 SeamlessDedup. This is the M610 receive path without sockets.
    if reconstructs_through_redundant_loss(&frame, w, h) {
        rec.add(
            Evidence::new(D::LossResilience)
                .detail("ST 2110-7 seamless merge through per-path drops"),
        );
    }

    rec
}

/// Reconstruct a frame through two lossy redundant paths merged by [`SeamlessDedup`],
/// returning whether the result is byte-exact.
fn reconstructs_through_redundant_loss(frame: &[u8], w: usize, h: usize) -> bool {
    let mut tx = St2110VideoPacketizer::new(96, 0xBEEF, Sampling::Rgba8, 60);
    let Some(packets) = tx.packetize(frame, w, h, 1_000_000_000) else {
        return false;
    };
    if packets.len() < 6 {
        return false; // too few packets to model a meaningful loss split
    }
    let last = packets.len() - 1;
    let mut dedup = SeamlessDedup::new();
    let Some(mut rx) = St2110VideoDepacketizer::new(RawVideoFormat::Rgba8, w, h) else {
        return false;
    };
    let mut done = None;
    for (i, p) in packets.iter().enumerate() {
        // Path A drops packet 2, path B drops 3 and 5; the marker (last) is on both.
        let on_a = i == last || i != 2;
        let on_b = i == last || (i != 3 && i != 5);
        for present in [on_a, on_b] {
            if present && dedup.accept(p) {
                if let Some(f) = rx.depacketize(p) {
                    done = Some(f.bytes);
                }
            }
        }
    }
    done.as_deref() == Some(frame)
}

/// Conformance of the ST 2110-30 (AES67) PCM audio packetizer / depacketizer core.
///
/// Verifies it constructs (`Instantiate`) and round-trips interleaved PCM byte-exact
/// through packetize -> depacketize (`RoundTrip`). Like the video core it emits no
/// `Oracle` evidence, so it tops out at `UnitTested`.
pub fn st2110_audio() -> MaturityRecord {
    use crate::st2110audio::{SampleDepth, St2110AudioDepacketizer, St2110AudioPacketizer};

    let mut rec = MaturityRecord::new("st2110audio");
    rec.add(Evidence::new(D::Instantiate));

    // Stereo L16 samples across the signed range so the round-trip is exact.
    let samples: Vec<i32> = (0..96i32).map(|i| ((i * 331) % 30_000) - 15_000).collect();
    let mut tx = St2110AudioPacketizer::new(96, 0x1234, 48_000, 2, SampleDepth::L16, 48);
    let packets = tx.packetize(&samples, 0);
    let rx = St2110AudioDepacketizer::new(2, SampleDepth::L16);
    let mut got: Vec<i32> = Vec::new();
    for p in &packets {
        if let Some(pkt) = rx.depacketize(p) {
            got.extend(pkt.samples);
        }
    }
    if got == samples {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("l16")
                .detail("packetize/depacketize loopback"),
        );
    }

    rec
}

/// An Annex-B access unit: a 4-byte start code, `header` as the NAL header byte,
/// then `body` bytes of a repeating pattern (so a truncated reassembly shows up).
fn annexb_nal(header: u8, body: usize) -> Vec<u8> {
    let mut au = alloc::vec![0, 0, 0, 1, header];
    au.extend((0..body).map(|i| (i * 31 + 7) as u8));
    au
}

/// Conformance of the H.264 RTP payload core (RFC 6184): the `rtppay`
/// packetizer and the `rtpdepay` depayloader that `udpsink` / `udpsrc`, the RTSP
/// server, and the WebRTC path all share.
///
/// Verifies it constructs (`Instantiate`), round-trips an access unit whose slice
/// NAL exceeds the MTU through FU-A fragmentation and reassembly byte-exact
/// (`RoundTrip`), and that a dropped fragment costs only its own access unit: the
/// sequence gap discards the damaged one instead of welding it to the next, which
/// still depayloads intact (`LossResilience`). No `Oracle` evidence: the
/// ffmpeg-peer check for this path is `udpsrc`'s, not this core's.
pub fn rtp_h264() -> MaturityRecord {
    use crate::rtpdepay::RtpH264Depayloader;
    use crate::rtppay::RtpH264Packetizer;

    let mut rec = MaturityRecord::new("rtph264");
    rec.add(Evidence::new(D::Instantiate));

    // A parameter-set NAL plus a slice far past the 200-byte MTU, so the access
    // unit spans a single-NAL packet and a run of FU-A fragments.
    let mut au = annexb_nal(0x67, 8);
    au.extend(annexb_nal(0x65, 900));
    let mut tx = RtpH264Packetizer::new(96, 0x5EED).with_max_payload(200);
    let packets = tx.packetize(&au, 9000);
    let mut rx = RtpH264Depayloader::new();
    let mut out = None;
    for p in &packets {
        if let Some(unit) = rx.depacketize(p) {
            out = Some(unit);
        }
    }
    if out.as_ref().map(|u| u.data.as_slice()) == Some(au.as_slice()) {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("h264")
                .detail("FU-A fragmented access unit reassembled byte-exact"),
        );
    }

    if survives_a_dropped_fragment(&au) {
        rec.add(
            Evidence::new(D::LossResilience)
                .codec("h264")
                .detail("dropped FU-A fragment discards only its own access unit"),
        );
    }

    rec
}

/// Payloadize `au` twice (two access units on one sequence run), drop a fragment
/// of the first, and report whether exactly the second comes out intact.
fn survives_a_dropped_fragment(au: &[u8]) -> bool {
    use crate::rtpdepay::RtpH264Depayloader;
    use crate::rtppay::RtpH264Packetizer;

    let mut tx = RtpH264Packetizer::new(96, 0x5EED).with_max_payload(200);
    let first = tx.packetize(au, 9000);
    let second = tx.packetize(au, 12000);
    if first.len() < 4 {
        return false; // no fragment run to damage
    }
    let mut rx = RtpH264Depayloader::new();
    let mut units = Vec::new();
    for (i, p) in first.iter().enumerate() {
        if i == 2 {
            continue; // the lost fragment
        }
        if let Some(u) = rx.depacketize(p) {
            units.push(u.data);
        }
    }
    for p in &second {
        if let Some(u) = rx.depacketize(p) {
            units.push(u.data);
        }
    }
    units.len() == 1 && units[0] == au
}

/// Conformance of the RTP jitter buffer (`rtpjitter`), the receive-side reorder
/// stage shared by `udpsrc`, `rtspsrc`, and the RTSP server.
///
/// Verifies it constructs (`Instantiate`) and that a wire which reorders packets
/// and loses one still yields intact access units (`LossResilience`): the
/// reordered packets are released in sequence order, the hole is reported for NACK
/// and then declared lost once it is overdue rather than stalling the stream.
pub fn rtp_jitter() -> MaturityRecord {
    use crate::rtpdepay::RtpH264Depayloader;
    use crate::rtpjitter::{JitterConfig, RtpJitterBuffer};
    use crate::rtppay::RtpH264Packetizer;

    let mut rec = MaturityRecord::new("rtpjitter");
    let config = JitterConfig::new(50, 64);
    rec.add(Evidence::new(D::Instantiate));

    let mut au = annexb_nal(0x67, 8);
    au.extend(annexb_nal(0x65, 900));
    let mut tx = RtpH264Packetizer::new(96, 0x1EAF).with_max_payload(200);
    let mut wire = tx.packetize(&au, 9000);
    let first_len = wire.len();
    wire.extend(tx.packetize(&au, 12000));
    if first_len < 4 {
        return rec;
    }

    // Arrival order: swap two adjacent pairs (reorder) and drop one packet of the
    // first access unit (loss).
    let lost_seq = 1u16;
    let mut jb = RtpJitterBuffer::new(config);
    let mut arrival: Vec<&Vec<u8>> = wire.iter().collect();
    arrival.swap(2, 3);
    arrival.swap(first_len, first_len + 1);
    for (i, p) in arrival.into_iter().enumerate() {
        if u16::from_be_bytes([p[2], p[3]]) == lost_seq {
            continue;
        }
        jb.push(p, i as u64 * 1_000_000);
    }
    let hole_seen = jb.missing_seqs().contains(&lost_seq);

    // Drain past the hold bound: the missing head is declared lost and the rest
    // releases in sequence order.
    let mut rx = RtpH264Depayloader::new();
    let mut units = Vec::new();
    let now = 10 * config.max_hold_ns;
    while let Some(p) = jb.pop(now) {
        if let Some(u) = rx.depacketize(&p) {
            units.push(u.data);
        }
    }
    let stats = jb.stats();
    if hole_seen && stats.reordered > 0 && stats.lost > 0 && units == alloc::vec![au] {
        rec.add(
            Evidence::new(D::LossResilience)
                .codec("h264")
                .detail("reordered arrival released in sequence order, lost packet skipped"),
        );
    }

    rec
}

// ================================================================
// Container and codec-core batteries (M1063).

/// A NAL body whose first RBSP bit is 1: `first_mb_in_slice = 0` coded as ue(0),
/// which opens a new coded picture under both the H.264 and the H.265
/// access-unit boundary rules.
const FIRST_SLICE_RBSP: [u8; 3] = [0x88, 0x84, 0x21];

/// An Annex-B NAL: 4-byte start code, the `header` bytes (one for H.264, two for
/// H.265), then `body`.
fn annexb(header: &[u8], body: &[u8]) -> Vec<u8> {
    let mut nal = alloc::vec![0, 0, 0, 1];
    nal.extend_from_slice(header);
    nal.extend_from_slice(body);
    nal
}

/// `len` bytes of a repeating pattern, so a truncated or spliced reassembly does
/// not compare equal to what went in.
fn pattern(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i * 31 + 7) as u8).collect()
}

/// The two H.264 access units the container batteries mux: an IDR picture led by
/// its parameter sets, then a non-IDR picture. The IDR slice is long enough that
/// its PES spans several transport-stream packets, which the MPEG-TS loss check
/// needs.
fn h264_access_units() -> [Vec<u8>; 2] {
    const SPS: u8 = 0x67;
    const PPS: u8 = 0x68;
    const IDR: u8 = 0x65;
    const NON_IDR: u8 = 0x41;
    const IDR_SLICE_BYTES: usize = 900;

    let mut idr = annexb(&[SPS], &pattern(12));
    idr.extend(annexb(&[PPS], &pattern(6)));
    let mut slice = Vec::from(FIRST_SLICE_RBSP);
    slice.extend(pattern(IDR_SLICE_BYTES));
    idr.extend(annexb(&[IDR], &slice));

    let mut inter = Vec::from(FIRST_SLICE_RBSP);
    inter.extend(pattern(64));
    [idr, annexb(&[NON_IDR], &inter)]
}

/// The AAC AudioSpecificConfig of a stereo 48 kHz LC stream, built from the named
/// fields: audio object type in the top 5 bits, then the sampling-frequency index
/// and the channel configuration.
fn aac_config() -> [u8; 2] {
    use crate::aacparse::SAMPLE_RATES;
    /// AAC Low Complexity (ISO/IEC 14496-3 audio object type 2).
    const AAC_LC: u8 = 2;
    const CHANNELS: u8 = 2;
    const SAMPLE_RATE_HZ: u32 = 48_000;

    let index = SAMPLE_RATES
        .iter()
        .position(|&r| r == SAMPLE_RATE_HZ)
        .expect("48 kHz has an ADTS sampling-frequency index") as u8;
    [
        (AAC_LC << 3) | (index >> 1),
        ((index & 1) << 7) | (CHANNELS << 3),
    ]
}

/// One ADTS-framed AAC access unit of the stereo 48 kHz stream above.
fn aac_adts_frame(payload: usize) -> Vec<u8> {
    crate::aacparse::adts_from_asc(&aac_config(), &pattern(payload))
        .expect("stereo 48 kHz frames within the 13-bit ADTS length")
}

/// The evidence a container round trip earns, recorded against both the muxer and
/// the demuxer: one wrote the bytes the other read back, so a passing check covers
/// the pair. Both cores construct infallibly, so `Instantiate` is unconditional;
/// every other piece of evidence comes from a check that passed.
fn container_records(mux: &str, demux: &str, earned: &[Evidence]) -> [MaturityRecord; 2] {
    let mut records = [MaturityRecord::new(mux), MaturityRecord::new(demux)];
    for rec in &mut records {
        rec.add(Evidence::new(D::Instantiate));
        for ev in earned {
            rec.add(ev.clone());
        }
    }
    records
}

/// The video PTS the MPEG-TS battery stamps, on the 90 kHz PES clock.
const TS_VIDEO_PTS_90KHZ: u64 = 900_000;
/// The audio PTS, one frame later on the same clock.
const TS_AUDIO_PTS_90KHZ: u64 = 903_000;

/// Conformance of the MPEG-TS muxer / demuxer cores (`mpegtsmux`, `tsdemux`).
///
/// Muxes one H.264 access unit and one AAC ADTS frame and demuxes both back
/// byte-exact with the timestamps they were stamped with (`RoundTrip`), then drops
/// a continuation packet of the video PES and checks the damage stays inside it:
/// the following PES still reassembles intact (`LossResilience`).
pub fn mpegts() -> [MaturityRecord; 2] {
    let mut earned = Vec::new();
    if mpegts_round_trips() {
        earned.push(
            Evidence::new(D::RoundTrip)
                .codec("h264+aac")
                .detail("PES mux/demux loopback, timestamps intact"),
        );
    }
    if mpegts_contains_a_dropped_packet() {
        earned.push(
            Evidence::new(D::LossResilience)
                .codec("h264+aac")
                .detail("a dropped TS packet costs only its own PES"),
        );
    }
    container_records("mpegtsmux", "tsdemux", &earned)
}

/// Mux a video access unit and an AAC frame as two separate runs of TS packets
/// (video first, tables included), with the two access units that went in.
fn mpegts_stream() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    use crate::mpegts::{TsMuxer, STREAM_TYPE_AAC, STREAM_TYPE_H264};

    const AAC_PAYLOAD_BYTES: usize = 200;

    let [video, _] = h264_access_units();
    let audio = aac_adts_frame(AAC_PAYLOAD_BYTES);
    let mut mux = TsMuxer::with_streams(&[STREAM_TYPE_H264, STREAM_TYPE_AAC]);
    let video_ts = mux.push_au_on(0, &video, Some(TS_VIDEO_PTS_90KHZ), None);
    let audio_ts = mux.push_au_on(1, &audio, Some(TS_AUDIO_PTS_90KHZ), None);
    (video_ts, audio_ts, video, audio)
}

/// Demux whole TS packets into their access units.
fn mpegts_units(bytes: &[u8]) -> Vec<crate::mpegts::EsUnit> {
    use crate::mpegts::{TsDemuxer, TS_PACKET_LEN};

    let mut demux = TsDemuxer::new();
    for pkt in bytes.chunks(TS_PACKET_LEN) {
        demux.push_packet(pkt);
    }
    demux.flush();
    demux.take_units()
}

/// Whether both elementary streams survive the mux / demux byte-exact, with their
/// timestamps.
fn mpegts_round_trips() -> bool {
    let (video_ts, audio_ts, video, audio) = mpegts_stream();
    let mut bytes = video_ts;
    bytes.extend(audio_ts);
    let units = mpegts_units(&bytes);
    units.len() == 2
        && units[0].data == video
        && units[0].pts_90khz == Some(TS_VIDEO_PTS_90KHZ)
        && units[1].data == audio
        && units[1].pts_90khz == Some(TS_AUDIO_PTS_90KHZ)
}

/// Drop the last TS packet of the video PES (a continuation packet: the tables and
/// the PES header lead the run) and report whether the loss stayed inside that
/// access unit: it no longer matches what went in, and the audio PES that follows
/// still comes out byte-exact.
fn mpegts_contains_a_dropped_packet() -> bool {
    use crate::mpegts::TS_PACKET_LEN;

    let (video_ts, audio_ts, video, audio) = mpegts_stream();
    if video_ts.len() < 4 * TS_PACKET_LEN {
        return false; // too short for the PES to have a continuation packet
    }
    let mut kept = Vec::from(&video_ts[..video_ts.len() - TS_PACKET_LEN]);
    kept.extend(audio_ts);
    let units = mpegts_units(&kept);
    let damaged = units.iter().any(|u| u.data != video && u.data != audio);
    let next_intact = units.iter().any(|u| u.data == audio);
    damaged && next_intact
}

/// Conformance of the fragmented-MP4 writer and reader (`mp4mux`, `fmp4demux`).
///
/// Muxes two H.264 access units into an `ftyp` + `moov` + `moof`/`mdat` stream and
/// reads the geometry, the sync-sample flag, and both access units back byte-exact
/// (`RoundTrip`). Video only: the A/V interleave lives in the `Mp4MuxN` element,
/// which has no sans-IO core to drive here.
#[cfg(feature = "std")]
pub fn mp4() -> [MaturityRecord; 2] {
    use crate::fmp4::{parse_fragments, parse_header};
    use crate::fmp4mux::Fmp4Muxer;
    use g2g_core::{TagList, VideoCodec};

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const FRAME_DURATION_NS: u64 = 33_000_000;

    let [idr, inter] = h264_access_units();
    let mut mux = Fmp4Muxer::new(VideoCodec::H264, WIDTH, HEIGHT, TagList::new());
    let mut bytes = mux.push_au(&idr, 0, FRAME_DURATION_NS).unwrap_or_default();
    if let Ok(more) = mux.push_au(&inter, FRAME_DURATION_NS, FRAME_DURATION_NS) {
        bytes.extend(more);
    }
    bytes.extend(mux.flush());

    let mut earned = Vec::new();
    if let Ok(header) = parse_header(&bytes) {
        let samples = parse_fragments(&bytes, header.timescale, VideoCodec::H264, None, 0, None);
        if let Ok(samples) = samples {
            let framed: Vec<&[u8]> = samples.iter().map(|s| s.annexb.as_slice()).collect();
            let geometry = header.width == WIDTH && header.height == HEIGHT;
            if geometry && framed == [idr.as_slice(), inter.as_slice()] && samples[0].keyframe {
                earned.push(
                    Evidence::new(D::RoundTrip)
                        .codec("h264")
                        .detail("moov geometry, sync sample, and access units intact"),
                );
            }
        }
    }
    container_records("mp4mux", "fmp4demux", &earned)
}

/// Conformance of the Matroska muxer / demuxer cores (`matroskamux`,
/// `matroskademux`): a two-track A/V file whose H.264 sample, AAC frame,
/// timestamps, geometry, and keyframe flag all survive the round trip.
pub fn matroska() -> [MaturityRecord; 2] {
    use crate::aacparse::strip_adts;
    use crate::annexb::{avcc_record, avcc_sample, split_annexb};
    use crate::matroska::{MatroskaDemuxer, MatroskaMuxer, MkvCodec, MkvTrackConfig, MkvTrackSpec};

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 240;
    const CHANNELS: u8 = 2;
    const SAMPLE_RATE_HZ: u32 = 48_000;
    const AUDIO_PTS_NS: u64 = 21_000_000;
    const AAC_PAYLOAD_BYTES: usize = 200;

    let [idr, _] = h264_access_units();
    let nalus = split_annexb(&idr);
    let video_sample = avcc_sample(&nalus);
    let adts = aac_adts_frame(AAC_PAYLOAD_BYTES);
    let audio_sample = strip_adts(&adts).to_vec();

    let mut mux = MatroskaMuxer::new_multi(alloc::vec![
        MkvTrackConfig {
            spec: MkvTrackSpec {
                codec: MkvCodec::H264,
                width: WIDTH,
                height: HEIGHT,
                channels: 0,
                sample_rate: 0,
            },
            codec_private: avcc_record(&nalus[..2]),
        },
        MkvTrackConfig {
            spec: MkvTrackSpec {
                codec: MkvCodec::Aac,
                width: 0,
                height: 0,
                channels: CHANNELS,
                sample_rate: SAMPLE_RATE_HZ,
            },
            codec_private: Vec::from(aac_config()),
        },
    ]);
    let mut bytes = mux.push_frame_on(0, &video_sample, 0, true, 0);
    bytes.extend(mux.push_frame_on(1, &audio_sample, AUDIO_PTS_NS, true, 0));

    let mut demux = MatroskaDemuxer::new();
    demux.push_data(&bytes);
    let tracks_match = demux.tracks().len() == 2
        && demux.tracks()[0].codec == MkvCodec::H264
        && demux.tracks()[0].width == WIDTH
        && demux.tracks()[0].height == HEIGHT
        && demux.tracks()[1].codec == MkvCodec::Aac
        && demux.tracks()[1].sample_rate == SAMPLE_RATE_HZ;
    let frames = demux.take_frames();
    let frames_match = frames.len() == 2
        && frames[0].data == video_sample
        && frames[0].pts_ns == 0
        && frames[0].keyframe
        && frames[1].data == audio_sample
        && frames[1].pts_ns == AUDIO_PTS_NS;

    let mut earned = Vec::new();
    if tracks_match && frames_match {
        earned.push(
            Evidence::new(D::RoundTrip)
                .codec("h264+aac")
                .detail("Tracks geometry and both tracks' blocks intact"),
        );
    }
    container_records("matroskamux", "matroskademux", &earned)
}

/// Conformance of the FLV muxer / demuxer cores (`flvmux`, `flvdemux`): an A/V
/// stream whose H.264 sample, AAC frame, decoder configs, and timestamps survive
/// the round trip.
pub fn flv() -> [MaturityRecord; 2] {
    use crate::aacparse::strip_adts;
    use crate::annexb::{avcc_record, avcc_sample, split_annexb};
    use crate::flv::{FlvCodec, FlvDemuxer, FlvMuxer, FlvTrack};

    const VIDEO_DTS_MS: u32 = 0;
    const AUDIO_PTS_MS: u32 = 21;
    const NO_COMPOSITION_OFFSET: i32 = 0;
    const AAC_PAYLOAD_BYTES: usize = 200;

    let [idr, _] = h264_access_units();
    let nalus = split_annexb(&idr);
    let video_sample = avcc_sample(&nalus);
    let video_config = avcc_record(&nalus[..2]);
    let adts = aac_adts_frame(AAC_PAYLOAD_BYTES);
    let audio_sample = strip_adts(&adts).to_vec();
    let audio_config = Vec::from(aac_config());

    let mut mux = FlvMuxer::new_av(
        FlvCodec::H264,
        video_config.clone(),
        FlvCodec::Aac,
        audio_config.clone(),
    );
    let mut bytes = mux.push_video(&video_sample, VIDEO_DTS_MS, NO_COMPOSITION_OFFSET, true);
    bytes.extend(mux.push_audio(&audio_sample, AUDIO_PTS_MS));

    let mut demux = FlvDemuxer::new();
    demux.push_data(&bytes);
    let units = demux.take_units();
    let configs_match = demux.video_config() == Some(video_config.as_slice())
        && demux.audio_config() == Some(audio_config.as_slice());
    let video_unit = units.iter().find(|u| u.track() == FlvTrack::Video);
    let audio_unit = units.iter().find(|u| u.track() == FlvTrack::Audio);
    let units_match = units.len() == 2
        && video_unit.is_some_and(|u| u.data == video_sample && u.dts_ms == VIDEO_DTS_MS)
        && audio_unit.is_some_and(|u| u.data == audio_sample && u.pts_ms == AUDIO_PTS_MS);

    let mut earned = Vec::new();
    if configs_match && units_match {
        earned.push(
            Evidence::new(D::RoundTrip)
                .codec("h264+aac")
                .detail("sequence headers and both tracks' tags intact"),
        );
    }
    container_records("flvmux", "flvdemux", &earned)
}

/// Conformance of the Ogg page writer / demuxer cores (`oggmux`, `oggdemux`).
///
/// Frames an Opus stream (`OpusHead`, `OpusTags`, then audio packets, one page
/// each) and reads the audio packets back byte-exact with the setup headers
/// skipped (`RoundTrip`); then drops a whole page and checks the demuxer resyncs
/// on the next one instead of stalling or splicing (`LossResilience`).
pub fn ogg() -> [MaturityRecord; 2] {
    let mut earned = Vec::new();
    let pages = ogg_pages();
    let packets = opus_packets();
    let whole: Vec<u8> = pages.concat();
    if ogg_demux(&whole) == packets {
        earned.push(
            Evidence::new(D::RoundTrip)
                .codec("opus")
                .detail("page lacing loopback, setup headers skipped"),
        );
    }
    // Drop the page carrying the middle audio packet: the two headers lead, so
    // that is page index 3.
    const LOST_PAGE: usize = 3;
    let mut lossy = Vec::new();
    for (i, page) in pages.iter().enumerate() {
        if i != LOST_PAGE {
            lossy.extend_from_slice(page);
        }
    }
    let survivors = alloc::vec![packets[0].clone(), packets[2].clone()];
    if pages.len() > LOST_PAGE && ogg_demux(&lossy) == survivors {
        earned.push(
            Evidence::new(D::LossResilience)
                .codec("opus")
                .detail("a dropped page resyncs on the next one"),
        );
    }
    container_records("oggmux", "oggdemux", &earned)
}

/// Three Opus audio packets: a fullband 20 ms TOC byte and an opaque body each.
fn opus_packets() -> Vec<Vec<u8>> {
    /// TOC config 31: CELT-only, fullband, 20 ms frames.
    const TOC_CELT_FULLBAND_20MS: u8 = 31;
    /// Frame-count code 0: one frame in the packet.
    const TOC_ONE_FRAME: u8 = 0;
    /// The TOC stereo bit.
    const TOC_STEREO: u8 = 1 << 2;
    const PACKET_BODIES: [usize; 3] = [40, 55, 48];

    let toc = (TOC_CELT_FULLBAND_20MS << 3) | TOC_STEREO | TOC_ONE_FRAME;
    PACKET_BODIES
        .iter()
        .map(|&len| {
            let mut p = alloc::vec![toc];
            p.extend(pattern(len));
            p
        })
        .collect()
}

/// The Ogg stream carrying [`opus_packets`], one page per packet so a page can be
/// dropped whole.
fn ogg_pages() -> Vec<Vec<u8>> {
    use crate::ogg::OggPageWriter;
    use crate::opusparse::{packet_samples, synth_opus_head, OPUS_RATE_HZ};

    const SERIAL: u32 = 0xC0FFEE;
    const CHANNELS: u8 = 2;
    /// The RFC 7845 comment header, whose contents this check does not read.
    const OPUS_TAGS: &[u8] = b"OpusTags\0\0\0\0\0\0\0\0";

    let mut writer = OggPageWriter::new(SERIAL);
    let mut pages = Vec::new();
    writer.push_packet(synth_opus_head(CHANNELS, OPUS_RATE_HZ), 0);
    pages.push(writer.flush(false));
    writer.push_packet(Vec::from(OPUS_TAGS), 0);
    pages.push(writer.flush(false));

    let packets = opus_packets();
    let last = packets.len() - 1;
    let mut granule = 0u64;
    for (i, packet) in packets.into_iter().enumerate() {
        granule += u64::from(packet_samples(&packet));
        writer.push_packet(packet, granule);
        pages.push(writer.flush(i == last));
    }
    pages
}

/// The audio packets an Ogg byte stream demuxes to.
fn ogg_demux(bytes: &[u8]) -> Vec<Vec<u8>> {
    use crate::ogg::OggDemuxer;

    let mut demux = OggDemuxer::new();
    demux.push_data(bytes);
    demux.take_packets()
}

/// Conformance of the ST 2110-40 (RFC 8331) ancillary-data core.
///
/// Round-trips a CEA-708 caption distribution packet through packetize ->
/// depacketize: the ANC packet comes back field-for-field, the CDP still parses to
/// the caption triples that went in, and a single flipped bit in the 10-bit word
/// stream is rejected, which is the parity / checksum actually being checked
/// rather than carried (`RoundTrip`). No `Oracle` evidence: no reference -40 gear
/// has read this, so it tops out at `UnitTested`.
pub fn st2110_anc() -> MaturityRecord {
    use crate::cea::{build_cdp, parse_cdp, CcTriple, CDP_FRAME_RATE_29_97};
    use crate::st2110anc::{AncField, AncPacket, St2110AncDepacketizer, St2110AncPacketizer};

    /// DID / SDID of a CEA-708 caption ANC packet (SMPTE ST 334).
    const ANC_DID_CAPTIONS: u8 = 0x61;
    const ANC_SDID_CEA708: u8 = 0x01;
    const CDP_SEQUENCE: u16 = 0x1234;
    const PAYLOAD_TYPE: u8 = 96;
    const SSRC: u32 = 0x0ACC_0ACC;
    const TAI_NS: u64 = 1_000_000_000;

    let mut rec = MaturityRecord::new("st2110anc");
    rec.add(Evidence::new(D::Instantiate));

    let triples = alloc::vec![
        CcTriple {
            cc_type: 0,
            b0: 0x14,
            b1: 0x20,
        },
        CcTriple {
            cc_type: 3,
            b0: 0x01,
            b1: 0x02,
        },
    ];
    let cdp = build_cdp(&triples, CDP_FRAME_RATE_29_97, CDP_SEQUENCE);
    let anc = AncPacket::generic(ANC_DID_CAPTIONS, ANC_SDID_CEA708, cdp);
    let mut tx = St2110AncPacketizer::new(PAYLOAD_TYPE, SSRC);
    let packet = tx.packetize(core::slice::from_ref(&anc), TAI_NS, AncField::Progressive);
    let rx = St2110AncDepacketizer::new();

    let recovered = rx.depacketize(&packet);
    let intact = recovered.as_ref().is_some_and(|frame| {
        frame.field == AncField::Progressive
            && frame.packets == alloc::vec![anc.clone()]
            && parse_cdp(&frame.packets[0].user_data) == Some(triples.clone())
    });
    // A single flipped bit in the ANC data section must fail parity / checksum.
    let mut corrupt = packet.clone();
    let middle = corrupt.len() / 2;
    corrupt[middle] ^= 1;
    if intact && rx.depacketize(&corrupt).is_none() {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("cea708")
                .detail("CDP through a 10-bit ANC packet, parity/checksum enforced"),
        );
    }
    rec
}

/// Conformance of the ST 2110-22 (RFC 9134) JPEG XS core: a codestream several
/// MTUs long is sliced into codestream-mode packets and reassembled byte-exact on
/// the marker bit (`RoundTrip`). The codestream is opaque bytes, so this needs no
/// JPEG XS codec. No `Oracle` evidence: no reference -22 gear has read it.
pub fn st2110_jxs() -> MaturityRecord {
    use crate::st2110jxs::{St2110JxsDepacketizer, St2110JxsPacketizer};

    const PAYLOAD_TYPE: u8 = 97;
    const SSRC: u32 = 0x0A5C_0000;
    const MAX_PACKET_BYTES: usize = 1460;
    const CODESTREAM_BYTES: usize = 4 * MAX_PACKET_BYTES;
    const MAX_FRAME_BYTES: usize = 1 << 20;
    const TAI_NS: u64 = 1_000_000_000;

    let mut rec = MaturityRecord::new("st2110jxs");
    rec.add(Evidence::new(D::Instantiate));

    let codestream = pattern(CODESTREAM_BYTES);
    let mut tx = St2110JxsPacketizer::new(PAYLOAD_TYPE, SSRC, MAX_PACKET_BYTES);
    let packets = tx.packetize(&codestream, TAI_NS);
    let mut rx = St2110JxsDepacketizer::new(MAX_FRAME_BYTES);
    let mut frame = None;
    for p in &packets {
        if let Some(f) = rx.depacketize(p) {
            frame = Some(f);
        }
    }
    let spans_packets = packets.len() > 1;
    if spans_packets && frame.map(|f| f.codestream) == Some(codestream) {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("jpegxs")
                .detail("codestream reassembled across MTU-sized packets"),
        );
    }
    rec
}

/// Conformance of the H.264 access-unit parser (`h264parse`): the boundary walk
/// that re-frames a byte stream into one access unit per buffer.
pub fn h264_parse() -> MaturityRecord {
    use crate::h264parse::{H264Codec, H264Parse};
    use g2g_core::AsyncElement;

    let mut rec = MaturityRecord::new("h264parse");
    let parser = H264Parse::new();
    if !parser.metadata().long_name.is_empty() && !parser.properties().is_empty() {
        rec.add(Evidence::new(D::Instantiate));
    }
    let [idr, inter] = h264_access_units();
    if reframes_into::<H264Codec>(&[&idr, &inter]) {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("h264")
                .detail("SPS+PPS+IDR then a non-IDR picture re-framed to one AU each"),
        );
    }
    rec
}

/// Conformance of the H.265 access-unit parser (`h265parse`), the HEVC sibling of
/// the check above: the NAL header is two bytes and the first-slice flag sits at
/// the top of the slice RBSP, so the boundary rules differ from H.264's.
pub fn h265_parse() -> MaturityRecord {
    use crate::h265parse::{H265Codec, H265Parse};
    use g2g_core::AsyncElement;

    /// H.265 NAL types: VPS, SPS, PPS, an IDR picture, and a trailing picture.
    const VPS_NUT: u8 = 32;
    const SPS_NUT: u8 = 33;
    const PPS_NUT: u8 = 34;
    const IDR_W_RADL_NUT: u8 = 19;
    const TRAIL_R_NUT: u8 = 1;
    /// `nuh_layer_id` 0 with `nuh_temporal_id_plus1` 1: the base layer.
    const BASE_LAYER: u8 = 1;

    let header = |nut: u8| [nut << 1, BASE_LAYER];
    let mut irap = annexb(&header(VPS_NUT), &pattern(8));
    irap.extend(annexb(&header(SPS_NUT), &pattern(12)));
    irap.extend(annexb(&header(PPS_NUT), &pattern(6)));
    let mut slice = Vec::from(FIRST_SLICE_RBSP);
    slice.extend(pattern(120));
    irap.extend(annexb(&header(IDR_W_RADL_NUT), &slice));
    let trail = annexb(&header(TRAIL_R_NUT), &slice);

    let mut rec = MaturityRecord::new("h265parse");
    let parser = H265Parse::new();
    if !parser.metadata().long_name.is_empty() && !parser.properties().is_empty() {
        rec.add(Evidence::new(D::Instantiate));
    }
    if reframes_into::<H265Codec>(&[&irap, &trail]) {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("h265")
                .detail("VPS+SPS+PPS+IDR then a trailing picture re-framed to one AU each"),
        );
    }
    rec
}

/// Whether the codec's access-unit boundary walk cuts the concatenation of `aus`
/// back into exactly those access units, with the keyframe flag set on the first
/// and clear on the rest.
fn reframes_into<C: crate::nalparse::NalCodec>(aus: &[&[u8]]) -> bool {
    let stream: Vec<u8> = aus.concat();
    let starts = C::au_starts(&stream);
    if starts.len() != aus.len() {
        return false;
    }
    let mut at = 0;
    for (i, au) in aus.iter().enumerate() {
        if starts[i] != at || &&stream[at..at + au.len()] != au {
            return false;
        }
        at += au.len();
    }
    at == stream.len() && C::au_is_keyframe(aus[0]) && !C::au_is_keyframe(aus[aus.len() - 1])
}

/// Conformance of the AV1 temporal-unit parser (`av1parse`): the OBU walk that
/// finds the sequence header, classifies a key frame, and strips the temporal
/// delimiters the ISOBMFF / Matroska mappings do not store.
#[cfg(feature = "std")]
pub fn av1_parse() -> MaturityRecord {
    use crate::av1parse::{av1_keyframe, has_sequence_header, strip_temporal_delimiters, Av1Parse};
    use g2g_core::AsyncElement;

    /// OBU types (AV1 §5.3.1).
    const OBU_SEQUENCE_HEADER: u8 = 1;
    const OBU_TEMPORAL_DELIMITER: u8 = 2;
    const OBU_FRAME: u8 = 6;
    /// `obu_has_size_field`, so each OBU carries its own leb128 length.
    const OBU_HAS_SIZE: u8 = 0x02;
    /// A frame OBU payload opening `show_existing_frame = 0`, `frame_type = KEY`.
    const KEY_FRAME_HEADER: u8 = 0x00;
    /// The same with `frame_type = INTER`.
    const INTER_FRAME_HEADER: u8 = 0x20;

    let obu = |obu_type: u8, payload: &[u8]| {
        let mut out = alloc::vec![(obu_type << 3) | OBU_HAS_SIZE];
        // leb128 length; every payload here is under 128 bytes, so one byte.
        out.push(payload.len() as u8);
        out.extend_from_slice(payload);
        out
    };
    let frame_payload = |first: u8| {
        let mut p = alloc::vec![first];
        p.extend(pattern(32));
        p
    };

    let delimiter = obu(OBU_TEMPORAL_DELIMITER, &[]);
    let sequence = obu(OBU_SEQUENCE_HEADER, &pattern(16));
    let key_frame = obu(OBU_FRAME, &frame_payload(KEY_FRAME_HEADER));
    let mut key_unit = delimiter.clone();
    key_unit.extend_from_slice(&sequence);
    key_unit.extend_from_slice(&key_frame);
    let mut inter_unit = delimiter;
    inter_unit.extend(obu(OBU_FRAME, &frame_payload(INTER_FRAME_HEADER)));

    let mut rec = MaturityRecord::new("av1parse");
    let parser = Av1Parse::new();
    if !parser.metadata().long_name.is_empty() {
        rec.add(Evidence::new(D::Instantiate));
    }
    let stored: Vec<u8> = [sequence.as_slice(), key_frame.as_slice()].concat();
    let framing = strip_temporal_delimiters(&key_unit) == stored
        && has_sequence_header(&key_unit)
        && av1_keyframe(&key_unit)
        && !has_sequence_header(&inter_unit)
        && !av1_keyframe(&inter_unit);
    if framing {
        rec.add(
            Evidence::new(D::RoundTrip).codec("av1").detail(
                "OBU walk: sequence header found, key frame classified, delimiter stripped",
            ),
        );
    }
    rec
}

/// Conformance of the AAC ADTS parser (`aacparse`): the header the muxers strip
/// and the demuxers re-synthesize, plus the declared frame length a frame splitter
/// walks a byte stream with.
pub fn aac_parse() -> MaturityRecord {
    use crate::aacparse::{adts_from_asc, asc_from_adts, strip_adts, AacParse};
    use g2g_core::AsyncElement;

    const PAYLOADS: [usize; 2] = [200, 260];

    let mut rec = MaturityRecord::new("aacparse");
    if !AacParse::new().metadata().long_name.is_empty() {
        rec.add(Evidence::new(D::Instantiate));
    }

    let config = aac_config();
    let framed = PAYLOADS.iter().all(|&len| {
        let raw = pattern(len);
        let Some(frame) = adts_from_asc(&config, &raw) else {
            return false;
        };
        asc_from_adts(&frame) == Some(config)
            && strip_adts(&frame) == raw.as_slice()
            && adts_frame_length(&frame) == Some(frame.len())
    });
    if framed {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("aac")
                .detail("ADTS header round trips through the AudioSpecificConfig"),
        );
    }
    rec
}

/// The `aac_frame_length` an ADTS header declares (13 bits across bytes 3..6), or
/// `None` if the buffer is not an ADTS frame. This is the field a frame splitter
/// advances by, so it must equal the frame's own length.
fn adts_frame_length(frame: &[u8]) -> Option<usize> {
    if frame.len() < 7 || frame[0] != 0xFF || (frame[1] & 0xF0) != 0xF0 {
        return None;
    }
    Some(
        (usize::from(frame[3] & 0x03) << 11)
            | (usize::from(frame[4]) << 3)
            | (usize::from(frame[5]) >> 5),
    )
}

/// Conformance of the Opus parser (`opusparse`): the `OpusHead` identification
/// header and the TOC-byte packet duration the Ogg / RTP timing depends on.
pub fn opus_parse() -> MaturityRecord {
    use crate::opusparse::{
        is_opus_config, packet_samples, parse_opus_head, synth_opus_head, OpusParse,
        OPUS_ENCODER_PRE_SKIP, OPUS_RATE_HZ,
    };
    use g2g_core::AsyncElement;

    const CHANNELS: u8 = 2;
    /// The frame duration coded by the TOC config [`opus_packets`] builds.
    const FRAME_DURATION_MS: u32 = 20;

    let mut rec = MaturityRecord::new("opusparse");
    if !OpusParse::new().metadata().long_name.is_empty() {
        rec.add(Evidence::new(D::Instantiate));
    }

    let head = synth_opus_head(CHANNELS, OPUS_RATE_HZ);
    let packets = opus_packets();
    let expected_samples = OPUS_RATE_HZ * FRAME_DURATION_MS / 1000;
    let parsed = parse_opus_head(&head) == Some((CHANNELS, OPUS_ENCODER_PRE_SKIP))
        && is_opus_config(&head)
        && packets
            .iter()
            .all(|p| !is_opus_config(p) && packet_samples(p) == expected_samples);
    if parsed {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("opus")
                .detail("OpusHead round trips and the TOC gives the packet duration"),
        );
    }
    rec
}

/// Conformance of the subtitle parser (`subparse`): a minimal document in each
/// text format parses to the cue times and text it declares.
pub fn sub_parse() -> MaturityRecord {
    use crate::subparse::{parse_srt, parse_ssa, parse_ttml, parse_webvtt, Cue, SubParse};
    use g2g_core::AsyncElement;

    const START_NS: u64 = 1_000_000_000;
    const END_NS: u64 = 4_000_000_000;
    const TEXT: &str = "Hello world";

    let mut rec = MaturityRecord::new("subparse");
    if !SubParse::new().metadata().long_name.is_empty() {
        rec.add(Evidence::new(D::Instantiate));
    }

    let srt = "1\n00:00:01,000 --> 00:00:04,000\nHello world\n";
    let webvtt = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nHello world\n";
    let ssa = "[Events]\n\
        Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n\
        Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Hello world\n";
    let ttml = "<tt xmlns=\"http://www.w3.org/ns/ttml\" xml:lang=\"en\"><body><div>\
        <p begin=\"00:00:01.000\" end=\"00:00:04.000\">Hello world</p>\
        </div></body></tt>";

    let cases: [(&str, Vec<Cue>); 4] = [
        ("srt", parse_srt(srt)),
        ("webvtt", parse_webvtt(webvtt)),
        ("ssa", parse_ssa(ssa)),
        ("ttml", parse_ttml(ttml)),
    ];
    for (format, cues) in cases {
        let matches = cues.len() == 1
            && cues[0].start_ns == START_NS
            && cues[0].end_ns == END_NS
            && cues[0].text == TEXT;
        if matches {
            rec.add(
                Evidence::new(D::RoundTrip)
                    .codec(format)
                    .detail("one cue with the declared times and text"),
            );
        }
    }
    rec
}

/// Conformance of the CEA-608 / 708 closed-caption core (`cea`): the encoders are
/// the inverse of the decoders, and the two carriages the pipeline uses (an
/// in-band SEI, a CDP) preserve the caption triples.
pub fn cea_captions() -> MaturityRecord {
    use crate::cea::{
        build_cc_sei, build_cdp, extract_cc_data, parse_cdp, Cc608Enc, Cc708Enc, CcTriple, Cea608,
        Cea708, CDP_FRAME_RATE_29_97,
    };
    use crate::subparse::{Cue, CueSettings};
    use g2g_core::VideoCodec;

    const CAPTION_TEXT: &str = "HELLO";
    const FIRST_FRAME_NS: u64 = 1_000;
    const FRAME_PERIOD_NS: u64 = 33_000;
    const IDLE_FRAMES: usize = 3;
    const CDP_SEQUENCE: u16 = 0x1234;

    let mut rec = MaturityRecord::new("cea");
    rec.add(Evidence::new(D::Instantiate));

    let cue = Cue {
        start_ns: 0,
        end_ns: 0,
        text: CAPTION_TEXT.into(),
        settings: CueSettings::default(),
    };

    // CEA-608: encode the cue, drain the byte pairs one per frame, erase, and read
    // the same text back out of the decoder.
    let mut enc608 = Cc608Enc::new();
    enc608.push_cue(&cue);
    let mut dec608 = Cea608::new();
    let mut t = FIRST_FRAME_NS;
    while enc608.pending() {
        let (b0, b1) = enc608.next_pair();
        dec608.push_pair(b0, b1, t);
        t += FRAME_PERIOD_NS;
    }
    for _ in 0..IDLE_FRAMES {
        let (b0, b1) = enc608.next_pair();
        dec608.push_pair(b0, b1, t);
        t += FRAME_PERIOD_NS;
    }
    enc608.erase();
    let erased_at = t;
    while enc608.pending() {
        let (b0, b1) = enc608.next_pair();
        dec608.push_pair(b0, b1, t);
        t += FRAME_PERIOD_NS;
    }
    let cues = dec608.take_cues();
    if cues.len() == 1 && cues[0].text == CAPTION_TEXT && cues[0].end_ns == erased_at {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("cea608")
                .detail("encoded caption decodes to the same text"),
        );
    }

    // CEA-708: the same shape over DTVCC triples.
    let mut enc708 = Cc708Enc::new();
    enc708.push_cue(&cue);
    let mut dec708 = Cea708::new();
    let mut t = FIRST_FRAME_NS;
    while enc708.pending() {
        let (cc_type, b0, b1) = enc708.next_triple();
        dec708.push_triple(cc_type, b0, b1, t);
        t += FRAME_PERIOD_NS;
    }
    for _ in 0..IDLE_FRAMES {
        let (cc_type, b0, b1) = enc708.next_triple();
        dec708.push_triple(cc_type, b0, b1, t);
        t += FRAME_PERIOD_NS;
    }
    enc708.erase();
    while enc708.pending() {
        let (cc_type, b0, b1) = enc708.next_triple();
        dec708.push_triple(cc_type, b0, b1, t);
        t += FRAME_PERIOD_NS;
    }
    dec708.flush(t + FRAME_PERIOD_NS);
    let cues = dec708.take_cues();
    if cues.len() == 1 && cues[0].text == CAPTION_TEXT {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("cea708")
                .detail("encoded caption decodes to the same text"),
        );
    }

    // The two carriages: an in-band SEI (H.264 and H.265) and a CDP.
    let triples = alloc::vec![
        CcTriple {
            cc_type: 0,
            b0: 0x14,
            b1: 0x20,
        },
        CcTriple {
            cc_type: 3,
            b0: 0x01,
            b1: 0x02,
        },
    ];
    let sei_intact = [VideoCodec::H264, VideoCodec::H265]
        .into_iter()
        .all(|codec| extract_cc_data(&build_cc_sei(&triples, codec), codec) == triples);
    let cdp_intact = parse_cdp(&build_cdp(&triples, CDP_FRAME_RATE_29_97, CDP_SEQUENCE))
        == Some(triples.clone());
    if sei_intact && cdp_intact {
        rec.add(
            Evidence::new(D::RoundTrip)
                .codec("cea608+cea708")
                .detail("caption triples survive the SEI and CDP carriages"),
        );
    }
    rec
}

/// The in-process conformance report: run every always-on battery and collect its
/// derived [`MaturityRecord`]. These are the checks that run anywhere (no ffmpeg, no
/// GPU), so they top out at `UnitTested`.
pub fn report() -> ConformanceReport {
    let mut report = ConformanceReport::new();
    report.push(st2110_video());
    report.push(st2110_audio());
    report.push(rtp_h264());
    report.push(rtp_jitter());
    for record in mpegts() {
        report.push(record);
    }
    for record in matroska() {
        report.push(record);
    }
    for record in flv() {
        report.push(record);
    }
    for record in ogg() {
        report.push(record);
    }
    #[cfg(feature = "std")]
    for record in mp4() {
        report.push(record);
    }
    report.push(st2110_anc());
    report.push(st2110_jxs());
    report.push(h264_parse());
    report.push(h265_parse());
    #[cfg(feature = "std")]
    report.push(av1_parse());
    report.push(aac_parse());
    report.push(opus_parse());
    report.push(sub_parse());
    report.push(cea_captions());
    report
}

/// Persisted conformance evidence (M615): the `Oracle` (reference-implementation) and
/// `Hardware` (device) checks that cannot run in-process are produced by the
/// integration tests that own those resources (ffmpeg, a GPU), which append their
/// evidence to a shared log. [`full_report`] folds that log into the in-process
/// [`report`] so `g2g-inspect --maturity` shows the `InteropTested` /
/// `HardwareValidated` rows those tests earned, without inspect itself needing the
/// resources.
///
/// The log is a simple tab-separated append-only file (one evidence line each), so
/// concurrent tests can append without coordination and it stays greppable. Its path
/// is `$G2G_CONFORMANCE_LOG`, or a default under the temp dir.
#[cfg(feature = "std")]
pub mod persist {
    use super::*;
    use alloc::format;
    use alloc::string::{String, ToString};
    use g2g_core::conformance::ConformanceDimension;
    use std::io::Write;
    use std::path::PathBuf;

    /// The evidence log path: `$G2G_CONFORMANCE_LOG` or `<tempdir>/g2g-conformance.tsv`.
    pub fn evidence_log_path() -> PathBuf {
        match std::env::var_os("G2G_CONFORMANCE_LOG") {
            Some(p) => PathBuf::from(p),
            None => std::env::temp_dir().join("g2g-conformance.tsv"),
        }
    }

    /// A field for the TSV line: `-` for absent, tabs / newlines flattened to spaces.
    fn field(v: Option<&str>) -> String {
        match v {
            None => "-".into(),
            Some(s) => s.replace(['\t', '\n', '\r'], " "),
        }
    }

    /// Parse a `-`/value TSV field back to an `Option`.
    fn unfield(s: &str) -> Option<String> {
        if s == "-" {
            None
        } else {
            Some(s.to_string())
        }
    }

    /// Append one piece of evidence for `element` to the log (creating it if needed).
    /// Called by a resource-owning conformance test when a check passes.
    pub fn record_evidence(element: &str, ev: &Evidence) -> std::io::Result<()> {
        let line = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\n",
            field(Some(element)),
            ev.dimension.label(),
            field(ev.platform.as_deref()),
            field(ev.codec.as_deref()),
            field(ev.peer.as_deref()),
            field(ev.detail.as_deref()),
        );
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(evidence_log_path())?;
        f.write_all(line.as_bytes())
    }

    /// Load the persisted evidence into a report (empty if the log is absent). A line
    /// with an unknown dimension or too few fields is skipped, never trusted.
    pub fn load_persisted() -> ConformanceReport {
        let mut report = ConformanceReport::new();
        let Ok(text) = std::fs::read_to_string(evidence_log_path()) else {
            return report;
        };
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 6 {
                continue;
            }
            let Some(dimension) = ConformanceDimension::from_label(f[1]) else {
                continue;
            };
            let mut ev = Evidence::new(dimension);
            ev.platform = unfield(f[2]);
            ev.codec = unfield(f[3]);
            ev.peer = unfield(f[4]);
            ev.detail = unfield(f[5]);
            report.record_mut(f[0]).add(ev);
        }
        report
    }

    /// The in-process [`report`](super::report) with the persisted `Oracle` /
    /// `Hardware` evidence folded in. This is what `g2g-inspect --maturity` renders.
    pub fn full_report() -> ConformanceReport {
        let mut report = super::report();
        report.absorb(load_persisted());
        report
    }

    /// The platform tag for a `Hardware` evidence row: `$G2G_CONFORMANCE_PLATFORM`
    /// when the runner names itself, else the device name `name_device` finds,
    /// else the host os / arch (which names no device, so it cannot pass for a
    /// device claim).
    #[cfg(all(
        target_os = "linux",
        any(feature = "cuda", feature = "v4l2", feature = "libcamera")
    ))]
    fn platform_tag(name_device: impl FnOnce() -> Option<String>) -> String {
        if let Some(name) = std::env::var_os("G2G_CONFORMANCE_PLATFORM") {
            return name.to_string_lossy().into_owned();
        }
        name_device()
            .unwrap_or_else(|| format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH))
    }

    /// The platform tag a CUDA-bound test should put on its `Hardware` evidence:
    /// the name the driver gives CUDA device 0 (the device the CUDA elements
    /// bind). Named per family rather than "the first GPU": a box with an
    /// integrated and a discrete GPU would otherwise tag a CUDA run with
    /// whichever adapter enumerated first.
    ///
    /// A test that already holds its own device name (Vulkan Video, a wgpu
    /// adapter it opened itself) passes that instead.
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    pub fn cuda_platform_tag() -> String {
        use g2g_core::runtime::DeviceProvider;
        platform_tag(|| {
            crate::gpudevice::GpuDeviceProvider::new()
                .probe()
                .ok()?
                .into_iter()
                .find(|d| d.persistent_id.starts_with("cuda:0:"))
                .map(|d| d.display_name)
        })
    }

    /// The platform tag a V4L2 capture test should put on its `Hardware`
    /// evidence: the card name the driver reports for the node it captured from
    /// (the camera model), which says which camera the evidence came from where
    /// `/dev/videoN` alone would not survive a replug.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    pub fn v4l2_platform_tag(device: &str) -> String {
        use g2g_core::runtime::DeviceProvider;
        platform_tag(|| {
            crate::v4l2device::V4l2DeviceProvider::new()
                .probe()
                .ok()?
                .into_iter()
                .find(|d| d.props.iter().any(|(k, v)| k == "device" && v == device))
                .map(|d| d.display_name)
        })
    }

    /// The platform tag a libcamera capture test should put on its `Hardware`
    /// evidence: the id libcamera gives the camera at `index`, which identifies
    /// the physical device (bus path plus USB vendor / product).
    #[cfg(all(target_os = "linux", feature = "libcamera"))]
    pub fn libcamera_platform_tag(index: usize) -> String {
        platform_tag(|| {
            let manager = libcamera::camera_manager::CameraManager::new().ok()?;
            let cameras = manager.cameras();
            cameras.get(index).map(|camera| camera.id().to_string())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::conformance::MaturityLevel;

    #[test]
    fn video_battery_is_unit_tested_not_interop() {
        let rec = st2110_video();
        assert!(rec.has(D::Instantiate));
        assert!(rec.has(D::RoundTrip), "loopback round-trip passed");
        assert!(rec.has(D::LossResilience), "-7 reconstruction passed");
        // The honesty contract: loopback validation is NOT interop validation.
        assert!(!rec.has(D::Oracle), "no reference-gear oracle evidence");
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
    }

    #[test]
    fn audio_battery_round_trips() {
        let rec = st2110_audio();
        assert!(rec.has(D::RoundTrip), "PCM loopback passed");
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
        assert!(!rec.has(D::Oracle));
    }

    #[test]
    fn rtp_h264_battery_round_trips_and_survives_loss() {
        let rec = rtp_h264();
        assert!(rec.has(D::RoundTrip), "FU-A reassembly is byte-exact");
        assert!(
            rec.has(D::LossResilience),
            "a dropped fragment costs only its own access unit"
        );
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
        assert!(!rec.has(D::Oracle), "no peer validated this core");
    }

    #[test]
    fn rtp_jitter_battery_reorders_and_skips_a_hole() {
        let rec = rtp_jitter();
        assert!(
            rec.has(D::LossResilience),
            "reordered arrival reconstructs, the hole is skipped not stalled"
        );
        assert_eq!(rec.level(), MaturityLevel::UnitTested);
    }

    #[test]
    fn fnv1a_is_stable_and_order_sensitive() {
        // Pinned against the reference FNV-1a 64 vector for "a", so a rewrite of the
        // digest cannot silently invalidate every committed golden.
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_ne!(fnv1a_64(b"ab"), fnv1a_64(b"ba"));
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    }

    #[test]
    fn i420_planes_splits_and_rejects_a_bad_size() {
        let buf = alloc::vec![0u8; 4 * 4 + 2 * 2 * 2];
        let [y, u, v] = i420_planes(&buf, 4, 4).expect("exact I420 size");
        assert_eq!((y.len(), u.len(), v.len()), (16, 4, 4));
        assert!(i420_planes(&buf, 4, 6).is_none(), "wrong height rejected");
        assert!(i420_planes(&buf, 3, 4).is_none(), "odd width rejected");
    }

    #[cfg(feature = "std")]
    #[test]
    fn psnr_scales_with_error_and_is_infinite_when_identical() {
        let reference = [10u8, 20, 30, 40];
        assert_eq!(psnr_db(&reference, &reference), Some(f64::INFINITY));
        // A uniform error of 1 LSB is 20*log10(255) = 48.13 dB.
        let off_by_one = [11u8, 21, 31, 41];
        let db = psnr_db(&reference, &off_by_one).expect("same length");
        assert!(
            (db - 48.13).abs() < 0.01,
            "1 LSB of error is ~48.13 dB: {db}"
        );
        // A larger error must score lower.
        let off_by_four = [14u8, 24, 34, 44];
        assert!(psnr_db(&reference, &off_by_four).expect("same length") < db);
        assert!(psnr_db(&reference, &[0u8; 3]).is_none(), "length mismatch");
    }

    #[cfg(feature = "std")]
    #[test]
    fn pooled_psnr_weighs_planes_by_sample_count() {
        // A big clean plane and a small dirty one pool to more than the dirty plane
        // alone, because the error is averaged over every sample.
        let clean = [128u8; 16];
        let dirty_reference = [128u8; 4];
        let dirty = [132u8; 4];
        let dirty_only = psnr_db(&dirty_reference, &dirty).expect("same length");
        let pooled =
            pooled_psnr_db(&[(&clean[..], &clean[..]), (&dirty_reference[..], &dirty[..])])
                .expect("same lengths");
        assert!(pooled > dirty_only, "{pooled} > {dirty_only}");
        assert!(pooled.is_finite());
        assert!(pooled_psnr_db(&[]).is_none(), "nothing measured");
    }

    #[test]
    fn report_renders_every_battery() {
        let report = report();
        let table = report.to_table();
        assert!(table.contains("st2110video"), "video row:\n{table}");
        assert!(table.contains("st2110audio"), "audio row:\n{table}");
        assert!(table.contains("rtph264"), "rtp payload row:\n{table}");
        assert!(table.contains("rtpjitter"), "jitter row:\n{table}");
        assert!(
            table.contains("unit-tested"),
            "derived levels shown:\n{table}"
        );
        // Every element is at least unit-tested (nothing regressed to instantiated).
        assert_eq!(report.min_level(), MaturityLevel::UnitTested);
    }

    #[cfg(feature = "std")]
    #[test]
    fn persisted_evidence_round_trips_and_merges_into_full_report() {
        // Record an Oracle datapoint (no ffmpeg needed for the persistence path
        // itself), reload it, and confirm it derives InteropTested and folds into the
        // in-process batteries via full_report.
        use g2g_core::conformance::ConformanceDimension;
        let log = std::env::temp_dir().join("g2g-conformance-unit-roundtrip.tsv");
        std::env::set_var("G2G_CONFORMANCE_LOG", &log);
        let _ = std::fs::remove_file(&log);

        persist::record_evidence(
            "x264enc",
            &Evidence::new(ConformanceDimension::Oracle)
                .peer("ffmpeg")
                .codec("h264")
                .detail("decoded by ffmpeg"),
        )
        .unwrap();

        let loaded = persist::load_persisted();
        let rec = loaded
            .records
            .iter()
            .find(|r| r.element == "x264enc")
            .expect("persisted");
        assert_eq!(rec.level(), MaturityLevel::InteropTested);
        assert_eq!(rec.peers(), alloc::vec!["ffmpeg"]);
        assert_eq!(
            rec.evidence[0].detail.as_deref(),
            Some("decoded by ffmpeg"),
            "detail survives"
        );

        // full_report carries both the in-process batteries and the persisted row.
        let full = persist::full_report();
        assert!(
            full.records.iter().any(|r| r.element == "st2110video"),
            "battery present"
        );
        assert!(
            full.records.iter().any(|r| r.element == "x264enc"),
            "persisted present"
        );

        let _ = std::fs::remove_file(&log);
    }
}
