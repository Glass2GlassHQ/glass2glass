#![cfg(feature = "std")]

mod demux_latency_common;

use demux_latency_common::{data_frame, run_demux};
use g2g_core::runtime::block_on;
use g2g_core::{
    ByteStreamEncoding, Caps, Dim, FrameTiming, G2gError, MultiInputElement, OutputSink,
    PipelinePacket, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::ogg::{OggCodec, OggPageWriter};
use g2g_plugins::oggdemux::{OggDemuxN, OggPort};

const NANOS_PER_SECOND: u64 = 1_000_000_000;
const NANOS_PER_MILLISECOND: u64 = 1_000_000;

const OPUS_RATE_HZ: u64 = 48_000;
// TOC config 1 (SILK narrowband, 20 ms), mono, one frame per packet
const OPUS_20_MS_TOC: u8 = 0x08;
const OPUS_PACKET_SAMPLES: u64 = 960;
const OPUS_PACKET_NS: u64 = OPUS_PACKET_SAMPLES * NANOS_PER_SECOND / OPUS_RATE_HZ;
const OPUS_CHANNELS: u8 = 1;
const OPUS_HEAD_VERSION: u8 = 1;
const OGG_SERIAL: u32 = 0x1206;
const PACKETS_PER_PAGE: u64 = 5;
const AUDIO_PAGE_COUNT: u64 = 4;
const OGG_PAGE_NS: u64 = PACKETS_PER_PAGE * OPUS_PACKET_NS;
// the OpusHead the demuxer forwards in-band ahead of the audio
const OGG_CONFIG_FRAMES: u64 = 1;

const VIDEO_WIDTH: u32 = 320;
const VIDEO_HEIGHT: u32 = 240;
const VIDEO_FRAMERATE_Q16: u32 = 25 << 16;
const VIDEO_FRAME_NS: u64 = 40_000_000;
const VIDEO_FRAME_COUNT: u64 = 12;
const FRAMES_PER_FRAGMENT: u64 = 4;
const MP4_FRAGMENT_NS: u64 = FRAMES_PER_FRAGMENT * VIDEO_FRAME_NS;
const SPS: [u8; 5] = [0x67, 0x42, 0x00, 0x1e, 0x88];
const PPS: [u8; 4] = [0x68, 0xce, 0x3c, 0x80];
const IDR: [u8; 4] = [0x65, 0x88, 0x84, 0x00];
const ANNEX_B_START_CODE: [u8; 4] = [0, 0, 0, 1];

fn opus_head() -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(OPUS_HEAD_VERSION);
    head.push(OPUS_CHANNELS);
    head.extend_from_slice(&0u16.to_le_bytes()); // pre-skip
    head.extend_from_slice(&(OPUS_RATE_HZ as u32).to_le_bytes());
    head.extend_from_slice(&0i16.to_le_bytes()); // output gain
    head.push(0); // channel mapping family
    head
}

fn opus_tags() -> Vec<u8> {
    let vendor = b"g2g";
    let mut tags = b"OpusTags".to_vec();
    tags.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    tags.extend_from_slice(vendor);
    tags.extend_from_slice(&0u32.to_le_bytes()); // comment count
    tags
}

// One Ogg page per chunk: the two header pages, then audio pages of PACKETS_PER_PAGE packets.
fn ogg_opus_pages() -> Vec<Vec<u8>> {
    let mut writer = OggPageWriter::new(OGG_SERIAL);
    let mut pages = Vec::new();
    for header in [opus_head(), opus_tags()] {
        let mut page = writer.push_packet(header, 0);
        page.extend(writer.flush(false));
        pages.push(page);
    }
    let mut granule = 0u64;
    for page_index in 0..AUDIO_PAGE_COUNT {
        let mut page = Vec::new();
        for _ in 0..PACKETS_PER_PAGE {
            granule += OPUS_PACKET_SAMPLES;
            page.extend(writer.push_packet(vec![OPUS_20_MS_TOC, 0], granule));
        }
        page.extend(writer.flush(page_index + 1 == AUDIO_PAGE_COUNT));
        pages.push(page);
    }
    pages
}

#[derive(Default)]
struct ChunkCapture {
    chunks: Vec<Vec<u8>>,
}

impl OutputSink for ChunkCapture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = packet {
            if let Some(bytes) = frame.domain.as_system_slice() {
                self.chunks.push(bytes.to_vec());
            }
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn annex_b(nal_units: &[&[u8]]) -> Vec<u8> {
    nal_units
        .iter()
        .flat_map(|nal_unit| [&ANNEX_B_START_CODE[..], nal_unit].concat())
        .collect()
}

// Every frame is an IDR, so a fragment closes as soon as it reaches FRAMES_PER_FRAGMENT frames.
fn h264_mp4_chunks(mux: Mp4MuxN) -> Vec<Vec<u8>> {
    let mut mux = mux;
    let caps = Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(VIDEO_WIDTH),
        height: Dim::Fixed(VIDEO_HEIGHT),
        framerate: Rate::Fixed(VIDEO_FRAMERATE_Q16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    };
    mux.configure_pipeline(0, &caps).expect("configure mp4 mux");
    let mut capture = ChunkCapture::default();
    block_on(async {
        for index in 0..VIDEO_FRAME_COUNT {
            let access_unit = if index == 0 {
                annex_b(&[&SPS, &PPS, &IDR])
            } else {
                annex_b(&[&IDR])
            };
            let pts_ns = index * VIDEO_FRAME_NS;
            let timing = FrameTiming {
                pts_ns,
                dts_ns: pts_ns,
                duration_ns: VIDEO_FRAME_NS,
                keyframe: true,
                ..FrameTiming::default()
            };
            mux.process(0, data_frame(access_unit, timing, index), &mut capture)
                .await
                .expect("mux access unit");
        }
        mux.process(0, PipelinePacket::Eos, &mut capture)
            .await
            .expect("mux eos");
    });
    capture.chunks
}

fn mp4_demux_for(chunks: &[Vec<u8>]) -> Mp4DemuxN {
    let file = chunks.concat();
    let ports = forwardable_streams(&file)
        .into_iter()
        .map(|stream| Mp4Port {
            track_id: stream.track_id,
            caps: stream.caps,
        })
        .collect();
    Mp4DemuxN::new(ports)
}

fn iso_bmff_caps() -> Caps {
    Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    }
}

#[test]
fn oggdemux_paces_its_sinks_with_the_page_it_holds() {
    let demux = OggDemuxN::new(vec![OggPort::new(0, OggCodec::Opus)]);
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    };
    let (log, stats) = run_demux(demux, caps, ogg_opus_pages());

    assert_eq!(
        log.per_frame_path_latency_ns.len() as u64,
        OGG_CONFIG_FRAMES + AUDIO_PAGE_COUNT * PACKETS_PER_PAGE,
        "the OpusHead and every audio packet reach the sink"
    );
    assert_eq!(
        log.startup_path_latency_ns,
        Some(0),
        "nothing is parsed before the run starts"
    );
    assert_eq!(
        log.per_frame_path_latency_ns.last(),
        Some(&OGG_PAGE_NS),
        "the sink ends paced with one held page: {:?}",
        log.per_frame_path_latency_ns
    );
    assert_eq!(stats.latency.min_ns, OGG_PAGE_NS);
}

#[test]
fn oggdemux_measures_one_page_when_a_chunk_carries_several() {
    let demux = OggDemuxN::new(vec![OggPort::new(0, OggCodec::Opus)]);
    let caps = Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    };
    let (_, stats) = run_demux(demux, caps, vec![ogg_opus_pages().concat()]);

    assert_eq!(stats.latency.min_ns, OGG_PAGE_NS);
}

#[test]
fn mp4demux_paces_its_sinks_with_the_fragment_it_holds() {
    let mux = Mp4MuxN::new(1).with_fragment_duration_ms(MP4_FRAGMENT_NS / NANOS_PER_MILLISECOND);
    let chunks = h264_mp4_chunks(mux);
    let demux = mp4_demux_for(&chunks);
    let (log, stats) = run_demux(demux, iso_bmff_caps(), chunks);

    assert_eq!(
        log.per_frame_path_latency_ns.len() as u64,
        VIDEO_FRAME_COUNT,
        "every access unit reaches the sink"
    );
    assert_eq!(
        log.startup_path_latency_ns,
        Some(0),
        "nothing is parsed before the run starts"
    );
    assert_eq!(
        log.per_frame_path_latency_ns.last(),
        Some(&MP4_FRAGMENT_NS),
        "the sink ends paced with one held fragment: {:?}",
        log.per_frame_path_latency_ns
    );
    assert_eq!(stats.latency.min_ns, MP4_FRAGMENT_NS);
}

#[test]
fn progressive_mp4_reports_no_held_latency() {
    let chunks = h264_mp4_chunks(Mp4MuxN::new(1).with_fragmented(false));
    let demux = mp4_demux_for(&chunks);
    let (log, stats) = run_demux(demux, iso_bmff_caps(), chunks);

    assert_eq!(
        log.per_frame_path_latency_ns.len() as u64,
        VIDEO_FRAME_COUNT,
        "every access unit reaches the sink"
    );
    assert!(
        log.per_frame_path_latency_ns.iter().all(|&ns| ns == 0),
        "the whole-file parse adds nothing to the deadlines: {:?}",
        log.per_frame_path_latency_ns
    );
    assert_eq!(stats.latency.min_ns, 0);
}
