//! M1549: an MP4 AAC track whose object type ADTS cannot signal (ER AAC, ELD,
//! USAC) leaves the demuxer as raw access units led by its AudioSpecificConfig,
//! and `aacparse` forwards both unchanged.
#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, MultiOutputElement,
    MultiOutputSink, PropValue, PushOutcome,
};
use g2g_plugins::aacparse::AacParse;
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};

use collect_sink::CollectSink;

// AAC-ELD, 48 kHz, stereo, as ffmpeg's FATE sample er_eld2000np_48_ep0 carries it
const ELD_CONFIG: [u8; 4] = [0xF8, 0xE6, 0x40, 0x00];
const CHANNELS: u16 = 2;
const SAMPLE_RATE: u32 = 48_000;
const SAMPLES_PER_ACCESS_UNIT: u32 = 480;
const TRACK_ID: u32 = 1;

// the second unit holds a valid ADTS header for 16 kHz mono, which a parser scanning raw bytes would announce
fn access_units() -> Vec<Vec<u8>> {
    vec![
        vec![0x7E, 0x89, 0xB4],
        vec![0xFF, 0xF1, 0x60, 0x40, 0x00, 0xFF, 0xFC],
        vec![0x56, 0x78, 0x9A, 0xBC],
    ]
}

fn mp4_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

fn full_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    mp4_box(kind, &[&[0u8; 4][..], payload].concat())
}

fn descriptor(tag: u8, body: &[u8]) -> Vec<u8> {
    [&[tag, body.len() as u8][..], body].concat()
}

fn table(entries: &[u32]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

// a progressive file with one AAC-ELD track: ftyp, mdat, then the moov indexing the mdat as one chunk
fn progressive_eld_mp4() -> Vec<u8> {
    let units = access_units();
    let ftyp = mp4_box(b"ftyp", b"isom\0\0\0\0isom");
    let mdat = mp4_box(b"mdat", &units.concat());
    let mdat_payload_offset = (ftyp.len() + 8) as u32;
    let count = units.len() as u32;

    let mut tkhd = vec![0u8; 80];
    tkhd[8..12].copy_from_slice(&TRACK_ID.to_be_bytes());
    let mut mdhd = vec![0u8; 16];
    mdhd[8..12].copy_from_slice(&SAMPLE_RATE.to_be_bytes());
    mdhd[12..16].copy_from_slice(&(count * SAMPLES_PER_ACCESS_UNIT).to_be_bytes());
    let mut hdlr = vec![0u8; 20];
    hdlr[4..8].copy_from_slice(b"soun");

    let decoder_config = [
        &[0x40, 0x15][..],
        &[0u8; 11],
        &descriptor(0x05, &ELD_CONFIG),
    ]
    .concat();
    let es = descriptor(
        0x03,
        &[&[0u8; 3][..], &descriptor(0x04, &decoder_config)].concat(),
    );
    let mut mp4a = vec![0u8; 28];
    mp4a[16..18].copy_from_slice(&CHANNELS.to_be_bytes());
    mp4a.extend_from_slice(&full_box(b"esds", &es));

    let sizes: Vec<u32> = units.iter().map(|unit| unit.len() as u32).collect();
    let stbl = [
        full_box(b"stsd", &[table(&[1]), mp4_box(b"mp4a", &mp4a)].concat()),
        full_box(b"stts", &table(&[1, count, SAMPLES_PER_ACCESS_UNIT])),
        full_box(b"stsc", &table(&[1, 1, count, 1])),
        full_box(b"stsz", &[table(&[0, count]), table(&sizes)].concat()),
        full_box(b"stco", &table(&[1, mdat_payload_offset])),
    ]
    .concat();
    let minf = mp4_box(b"minf", &mp4_box(b"stbl", &stbl));
    let mdia = mp4_box(
        b"mdia",
        &[full_box(b"mdhd", &mdhd), full_box(b"hdlr", &hdlr), minf].concat(),
    );
    let trak = mp4_box(b"trak", &[full_box(b"tkhd", &tkhd), mdia].concat());
    [ftyp, mdat, mp4_box(b"moov", &trak)].concat()
}

fn data_frame(data: Vec<u8>) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming::default(),
        0,
    ))
}

fn frame_bytes(packets: &[PipelinePacket]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => Some(
                frame
                    .domain
                    .require_system_slice("test")
                    .expect("system frame")
                    .to_vec(),
            ),
            _ => None,
        })
        .collect()
}

fn caps_changes(packets: &[PipelinePacket]) -> Vec<Caps> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::CapsChanged(caps) => Some(caps.clone()),
            _ => None,
        })
        .collect()
}

fn assert_config_leads_raw_units(frames: &[Vec<u8>]) {
    let expected: Vec<Vec<u8>> = std::iter::once(ELD_CONFIG.to_vec())
        .chain(access_units())
        .collect();
    assert_eq!(
        frames, expected,
        "the AudioSpecificConfig leads, then each access unit unframed"
    );
}

async fn single_output_demux(file: Vec<u8>) -> Vec<PipelinePacket> {
    let mut demux = Mp4Demux::new();
    demux
        .set_property("stream", PropValue::Str("aac".into()))
        .expect("stream=aac");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    demux
        .process(data_frame(file), &mut sink)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.packets
}

#[derive(Default)]
struct PortCapture {
    frames: Vec<Vec<u8>>,
}

impl MultiOutputSink for PortCapture {
    fn poll_push_to(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        _port: usize,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        self.frames.extend(frame_bytes(&[packet]));
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }

    fn port_count(&self) -> usize {
        1
    }
}

#[tokio::test]
async fn mp4demux_sends_eld_config_ahead_of_raw_units() {
    let packets = single_output_demux(progressive_eld_mp4()).await;
    assert!(
        packets.iter().any(|packet| matches!(
            packet,
            PipelinePacket::CapsChanged(Caps::Audio {
                format: AudioFormat::Aac,
                ..
            })
        )),
        "AAC caps announced"
    );
    assert_config_leads_raw_units(&frame_bytes(&packets));
}

#[tokio::test]
async fn multi_output_demux_sends_eld_config_ahead_of_raw_units() {
    let file = progressive_eld_mp4();
    let streams = forwardable_streams(&file);
    assert_eq!(streams.len(), 1, "one AAC track discovered");
    let mut demux = Mp4DemuxN::new(vec![Mp4Port {
        track_id: streams[0].track_id,
        caps: streams[0].caps.clone(),
    }]);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure");
    let mut capture = PortCapture::default();
    demux
        .process(data_frame(file), &mut capture)
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut capture)
        .await
        .expect("eos");
    assert_config_leads_raw_units(&capture.frames);
}

#[tokio::test]
async fn aacparse_forwards_config_and_raw_units_unchanged() {
    let demuxed = single_output_demux(progressive_eld_mp4()).await;
    let mut parse = AacParse::new();
    parse
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Aac,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("configure");
    let demuxed_caps = caps_changes(&demuxed);
    let mut sink = CollectSink::default();
    for packet in demuxed {
        parse.process(packet, &mut sink).await.expect("parse");
    }
    parse
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    assert_config_leads_raw_units(&frame_bytes(&sink.packets));
    assert_eq!(
        caps_changes(&sink.packets),
        demuxed_caps,
        "only the demuxer's caps pass, nothing read out of the raw bytes"
    );
}
