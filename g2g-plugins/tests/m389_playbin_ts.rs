//! M389 - `playbin uri=*.ts` multi-stream fan-out + multi-hook dispatch. A lone
//! `playbin uri=file://x.ts` probes the transport stream's PMT and auto-builds
//! `FileSrc -> TsDemuxN -> {decode -> auto sink}`, one branch per stream, the
//! MPEG-TS sibling of the MKV fan-out (M382). The registry now holds several
//! playbin hooks (M389): a TS file is handled by `ts_playbin`, an MKV file by
//! `mkv_playbin`, each declining the other's container.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;
use std::path::PathBuf;

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::{parse_launch, ElementFactory, LaunchFactory, Registry};
use g2g_core::{
    AudioFormat, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, G2gError, MultiInputElement,
    OutputSink, PadTemplate, PadTemplates, PushOutcome, RawVideoFormat, Rate, VideoCodec,
};

use g2g_plugins::mkvmuxn::MkvMuxN;
use g2g_plugins::mpegts::{STREAM_TYPE_AAC, STREAM_TYPE_H264};

const TS_SYNC: u8 = 0x47;
const TS_PACKET_LEN: usize = 188;

// --- caps + stub-element helpers ---
fn h264_any() -> Caps {
    Caps::CompressedVideo { codec: VideoCodec::H264, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn aac_any() -> Caps {
    Caps::Audio { format: AudioFormat::Aac, channels: 0, sample_rate: 0 }
}
fn raw_video() -> Caps {
    Caps::RawVideo { format: RawVideoFormat::Nv12, width: Dim::Any, height: Dim::Any, framerate: Rate::Any }
}
fn raw_audio() -> Caps {
    Caps::Audio { format: AudioFormat::PcmS16Le, channels: 0, sample_rate: 0 }
}

#[derive(Default)]
struct NullSink;
impl PadTemplates for NullSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::new()
    }
}
impl AsyncElement for NullSink {
    type ProcessFuture<'a> = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>> where Self: 'a;
    fn intercept_caps(&self, c: &Caps) -> Result<Caps, G2gError> {
        Ok(c.clone())
    }
    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }
    fn configure_pipeline(&mut self, _c: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }
    fn process<'a>(
        &'a mut self,
        _packet: g2g_core::PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async { Ok(()) })
    }
}

/// A registry with both playbin hooks (MKV + TS), stub H.264 / AAC decoders, and
/// the auto-sink names, so a probed container's branches plug to raw.
fn registry() -> Registry {
    let mut reg = Registry::new();
    reg.register_playbin(g2g_plugins::uridecodebin::mkv_playbin);
    reg.register_playbin(g2g_plugins::uridecodebin::ts_playbin);
    reg.register(ElementFactory::new(
        "h264stub",
        Vec::from([PadTemplate::sink(CapsSet::one(h264_any())), PadTemplate::source(CapsSet::one(raw_video()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register(ElementFactory::new(
        "aacstub",
        Vec::from([PadTemplate::sink(CapsSet::one(aac_any())), PadTemplate::source(CapsSet::one(raw_audio()))]),
        |_| Box::new(g2g_plugins::identity::IdentityTransform::new()),
    ));
    reg.register_launch(LaunchFactory::of::<NullSink>("autovideosink", || Box::new(NullSink)));
    reg.register_launch(LaunchFactory::of::<NullSink>("autoaudiosink", || Box::new(NullSink)));
    reg
}

fn temp_uri(tag: &str, bytes: &[u8]) -> (PathBuf, String) {
    let path = std::env::temp_dir().join(format!("g2g_m389_{}_{}.bin", std::process::id(), tag));
    std::fs::write(&path, bytes).expect("write fixture");
    let uri = format!("file://{}", path.display());
    (path, uri)
}

// --- MPEG-TS A/V builder (PAT + 2-stream PMT + video/audio PES) ---
fn ts_packet(pid: u16, pusi: bool, payload: &[u8]) -> Vec<u8> {
    const ROOM: usize = TS_PACKET_LEN - 4;
    let mut p = vec![0u8; TS_PACKET_LEN];
    p[0] = TS_SYNC;
    p[1] = if pusi { 0x40 } else { 0x00 } | ((pid >> 8) as u8 & 0x1F);
    p[2] = (pid & 0xFF) as u8;
    let l = payload.len();
    if l == ROOM {
        p[3] = 0x10;
        p[4..].copy_from_slice(payload);
    } else {
        p[3] = 0x30;
        let af_len = ROOM - 1 - l;
        p[4] = af_len as u8;
        if af_len >= 1 {
            p[5] = 0x00;
            for b in p.iter_mut().take(6 + (af_len - 1)).skip(6) {
                *b = 0xFF;
            }
        }
        p[5 + af_len..].copy_from_slice(payload);
    }
    p
}
fn psi(pid: u16, table_id: u8, body: &[u8]) -> Vec<u8> {
    let section_length = body.len() + 4;
    let mut s = vec![table_id, 0xB0 | ((section_length >> 8) as u8 & 0x0F), (section_length & 0xFF) as u8];
    s.extend_from_slice(body);
    s.extend_from_slice(&[0, 0, 0, 0]);
    let mut payload = vec![0u8];
    payload.extend_from_slice(&s);
    ts_packet(pid, true, &payload)
}
fn pat(pmt_pid: u16) -> Vec<u8> {
    psi(0x0000, 0x00, &[0, 1, 0xC1, 0, 0, 0, 1, 0xE0 | (pmt_pid >> 8) as u8 & 0x1F, pmt_pid as u8])
}
fn pmt2(v_pid: u16, v_type: u8, a_pid: u16, a_type: u8) -> Vec<u8> {
    psi(
        0x1000,
        0x02,
        &[
            0x00, 0x01, 0xC1, 0x00, 0x00,
            0xE0 | (v_pid >> 8) as u8 & 0x1F, v_pid as u8, 0xF0, 0x00,
            v_type, 0xE0 | (v_pid >> 8) as u8 & 0x1F, v_pid as u8, 0xF0, 0x00,
            a_type, 0xE0 | (a_pid >> 8) as u8 & 0x1F, a_pid as u8, 0xF0, 0x00,
        ],
    )
}
fn pes_id(stream_id: u8, es: &[u8]) -> Vec<u8> {
    let mut p = vec![0x00, 0x00, 0x01, stream_id];
    let header = [0x80u8, 0x00, 0x00];
    let len = header.len() + es.len();
    p.push((len >> 8) as u8);
    p.push((len & 0xFF) as u8);
    p.extend_from_slice(&header);
    p.extend_from_slice(es);
    p
}
fn av_ts() -> Vec<u8> {
    let mut ts = pat(0x1000);
    ts.extend_from_slice(&pmt2(0x0100, STREAM_TYPE_H264, 0x0101, STREAM_TYPE_AAC));
    ts.extend_from_slice(&ts_packet(0x0100, true, &pes_id(0xE0, &[0, 0, 0, 1, 0x67])));
    ts.extend_from_slice(&ts_packet(0x0101, true, &pes_id(0xC0, &[0xFF, 0xF1, 0x50])));
    ts
}

// --- Matroska A/V builder (for the dispatch cross-check) ---
fn frame(data: Vec<u8>, pts_ns: u64) -> g2g_core::PipelinePacket {
    g2g_core::PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming { pts_ns, dts_ns: pts_ns, ..FrameTiming::default() },
        0,
    ))
}
fn annexb(nals: &[&[u8]]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in nals {
        v.extend_from_slice(&[0, 0, 0, 1]);
        v.extend_from_slice(n);
    }
    v
}
fn adts_au(payload: &[u8]) -> Vec<u8> {
    let frame_len = payload.len() + 7;
    let mut au = vec![
        0xFF, 0xF1, (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F, 0xFC,
    ];
    au.extend_from_slice(payload);
    au
}
#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: g2g_core::PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let g2g_core::PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}
async fn av_mkv() -> Vec<u8> {
    let mut mux = MkvMuxN::new(2);
    mux.configure_pipeline(0, &h264_any()).unwrap();
    mux.configure_pipeline(1, &Caps::Audio { format: AudioFormat::Aac, channels: 2, sample_rate: 48_000 }).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&[0x67u8, 0x42, 0, 0x1e, 0x88], &[0x68u8, 0xce, 0x3c, 0x80], &[0x65u8, 0x88]]), 0), &mut sink).await.unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2]), 0), &mut sink).await.unwrap();
    mux.process(0, g2g_core::PipelinePacket::Eos, &mut sink).await.unwrap();
    mux.process(1, g2g_core::PipelinePacket::Eos, &mut sink).await.unwrap();
    sink.bytes
}

#[tokio::test]
async fn playbin_fans_out_a_transport_stream() {
    let (path, uri) = temp_uri("av_ts", &av_ts());
    let reg = registry();
    let graph = parse_launch(&reg, &format!("playbin uri={uri}")).expect("ts playbin fans out");
    std::fs::remove_file(&path).ok();

    // FileSrc -> TsDemuxN(2); each port: demux.out(i) -> stub decoder -> auto sink.
    assert_eq!(graph.node_count(), 6, "source, demux, two decoders, two auto sinks");
    assert_eq!(graph.edges().len(), 5, "one decode branch per forwardable stream");
}

#[tokio::test]
async fn the_right_hook_handles_each_container() {
    let reg = registry();

    // A TS file is handled by ts_playbin (mkv_playbin declines it).
    let (ts_path, ts_uri) = temp_uri("disp_ts", &av_ts());
    let ts_graph = parse_launch(&reg, &format!("playbin uri={ts_uri}")).expect("ts handled");
    std::fs::remove_file(&ts_path).ok();
    assert_eq!(ts_graph.node_count(), 6, "TS fans out via ts_playbin");

    // An MKV file is handled by mkv_playbin (ts_playbin declines it).
    let (mkv_path, mkv_uri) = temp_uri("disp_mkv", &av_mkv().await);
    let mkv_graph = parse_launch(&reg, &format!("playbin uri={mkv_uri}")).expect("mkv handled");
    std::fs::remove_file(&mkv_path).ok();
    assert_eq!(mkv_graph.node_count(), 6, "MKV fans out via mkv_playbin");
}
