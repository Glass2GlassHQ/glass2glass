//! M475 - app-driven stream selection on the MPEG-TS and MP4 multi-output
//! demuxers, bringing them to parity with `MkvDemuxN` (M381). A
//! `StreamSelectController` re-maps which stream each port carries: the app names
//! a stream id per port (from the M386/M376 collection), and the demuxer re-routes
//! accordingly, re-arming each changed port's `CapsChanged` and confirming the
//! active ids on the bus.
//!
//! Each test starts a two-port demux as [video, audio], discovers the announced
//! ids, re-selects to swap the ports to [audio, video], then feeds the payload:
//! port 0 must end up carrying the audio track and port 1 the video track.

#![cfg(feature = "std")]

use core::future::Future;
use core::pin::Pin;

use g2g_core::fanout::{MultiOutputElement, MultiOutputSink};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::StreamSelectController;
use g2g_core::{
    AudioFormat, Bus, BusHandle, BusMessage, ByteStreamEncoding, Caps, Dim, G2gError,
    MultiInputElement, OutputSink, PushOutcome, Rate, StreamType, VideoCodec,
};

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
    }
}
fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 2,
        sample_rate: 48_000,
    }
}

/// Records, per port, the DataFrame payloads and CapsChanged it receives.
#[derive(Default)]
struct PortTap {
    frames: Vec<Vec<Vec<u8>>>,
    caps: Vec<Vec<Caps>>,
}
impl PortTap {
    fn new(ports: usize) -> Self {
        Self {
            frames: vec![Vec::new(); ports],
            caps: vec![Vec::new(); ports],
        }
    }
}
impl MultiOutputSink for PortTap {
    fn push_to<'a>(
        &'a mut self,
        port: usize,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(Frame {
                    domain: MemoryDomain::System(s),
                    ..
                }) => {
                    self.frames[port].push(s.as_slice().to_vec());
                }
                PipelinePacket::CapsChanged(c) => self.caps[port].push(c),
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
    fn port_count(&self) -> usize {
        self.frames.len()
    }
}

/// A byte sink for capturing a muxer's output stream.
#[derive(Default)]
struct Collect {
    bytes: Vec<u8>,
}
impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            if let PipelinePacket::DataFrame(f) = packet {
                if let MemoryDomain::System(s) = &f.domain {
                    self.bytes.extend_from_slice(s.as_slice());
                }
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

fn frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}
fn data_frame(bytes: &[u8]) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(bytes.to_vec().into_boxed_slice())),
        FrameTiming::default(),
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
        0xFF,
        0xF1,
        (1 << 6) | (3 << 2),
        ((2 & 3) << 6) | ((frame_len >> 11) & 3) as u8,
        ((frame_len >> 3) & 0xFF) as u8,
        (((frame_len & 7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    au.extend_from_slice(payload);
    au
}

/// The two ids (video, audio) from the first StreamCollection drained off the bus.
fn discover_av_ids(bus: &Bus) -> Option<(String, String)> {
    while let Some(msg) = bus.try_recv() {
        if let BusMessage::StreamCollection(c) = msg {
            let v = c
                .streams_of_type(StreamType::Video)
                .next()
                .map(|s| s.id.clone());
            let a = c
                .streams_of_type(StreamType::Audio)
                .next()
                .map(|s| s.id.clone());
            if let (Some(v), Some(a)) = (v, a) {
                return Some((v, a));
            }
        }
    }
    None
}

// ---- MPEG-TS -------------------------------------------------------------

async fn mux_ts_av() -> Vec<u8> {
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = g2g_plugins::tsmuxn::TsMux::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 40_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 20_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

#[tokio::test]
async fn tsdemuxn_selection_remaps_which_stream_each_port_carries() {
    use g2g_plugins::tsdemux::{TsDemuxN, TsStream};

    let file = mux_ts_av().await;
    let (bus, handle): (Bus, BusHandle) = Bus::new(16);
    let select = StreamSelectController::new();
    // Ports start [video, audio]; the selection will swap them.
    let mut demux = TsDemuxN::new(vec![TsStream::H264, TsStream::Aac])
        .with_bus(handle)
        .with_stream_select(select.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("configure");

    let mut tap = PortTap::new(2);
    // Feed one 188-byte TS packet at a time; the moment the PMT is parsed and the
    // collection is announced, select the swap [audio, video], before any PES
    // access unit has been routed.
    let mut selected = false;
    for pkt in file.chunks(188) {
        demux.process(data_frame(pkt), &mut tap).await.unwrap();
        if !selected {
            if let Some((v, a)) = discover_av_ids(&bus) {
                select.select(vec![a, v]);
                selected = true;
            }
        }
    }
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();
    assert!(selected, "the PMT was parsed and the collection announced");

    // Port 0 re-mapped to audio, port 1 to video.
    assert!(
        tap.caps[0]
            .last()
            .map(|c| matches!(
                c,
                Caps::Audio {
                    format: AudioFormat::Aac,
                    ..
                }
            ))
            .unwrap_or(false),
        "port 0 re-mapped to the audio stream: {:?}",
        tap.caps[0]
    );
    // The demuxed AAC is ADTS-framed (7-byte header + the payload).
    assert_eq!(
        tap.frames[0].len(),
        2,
        "port 0 carries the two audio access units after the swap"
    );
    assert!(
        tap.frames[0]
            .iter()
            .all(|au| au[0] == 0xFF && (au[1] & 0xF0) == 0xF0),
        "ADTS-framed AAC"
    );
    assert!(
        tap.frames[0][0].ends_with(&[0xA1, 0xA2, 0xA3])
            && tap.frames[0][1].ends_with(&[0xB4, 0xB5]),
        "audio payloads preserved"
    );
    assert!(
        tap.caps[1]
            .last()
            .map(|c| matches!(
                c,
                Caps::CompressedVideo {
                    codec: VideoCodec::H264,
                    ..
                }
            ))
            .unwrap_or(false),
        "port 1 re-mapped to the video stream: {:?}",
        tap.caps[1]
    );
    assert_eq!(
        tap.frames[1].len(),
        2,
        "port 1 carries the two H.264 access units after the swap"
    );

    // The active selection is confirmed on the bus.
    let selected_ids: Vec<_> = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| match m {
            BusMessage::StreamsSelected { ids } => Some(ids),
            _ => None,
        })
        .collect();
    assert!(
        !selected_ids.is_empty(),
        "StreamsSelected confirms the per-port ids"
    );
}

// ---- fragmented MP4 ------------------------------------------------------

async fn mux_mp4_av() -> Vec<u8> {
    use g2g_plugins::mp4muxn::Mp4MuxN;
    let sps = [0x67u8, 0x42, 0x00, 0x1e, 0x88];
    let pps = [0x68u8, 0xce, 0x3c, 0x80];
    let idr = [0x65u8, 0x88, 0x84, 0x00];
    let mut mux = Mp4MuxN::new(2);
    mux.configure_pipeline(0, &h264_caps()).unwrap();
    mux.configure_pipeline(1, &aac_caps()).unwrap();
    let mut sink = Collect::default();
    mux.process(0, frame(annexb(&[&sps, &pps, &idr]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(1, frame(adts_au(&[0xA1, 0xA2, 0xA3]), 0), &mut sink)
        .await
        .unwrap();
    mux.process(
        0,
        frame(annexb(&[&[0x41u8, 0x9a, 0x00]]), 33_000_000),
        &mut sink,
    )
    .await
    .unwrap();
    mux.process(1, frame(adts_au(&[0xB4, 0xB5]), 21_000_000), &mut sink)
        .await
        .unwrap();
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    mux.process(1, PipelinePacket::Eos, &mut sink)
        .await
        .unwrap();
    sink.bytes
}

#[tokio::test]
async fn mp4demuxn_selection_remaps_which_track_each_port_carries() {
    use g2g_plugins::mp4demuxn::{forwardable_streams, Mp4DemuxN, Mp4Port};

    let file = mux_mp4_av().await;
    let streams = forwardable_streams(&file);
    assert_eq!(streams.len(), 2, "video + audio tracks discovered");
    // Ports start [track for stream 0 (video), track for stream 1 (audio)].
    let ports: Vec<Mp4Port> = streams
        .iter()
        .map(|s| Mp4Port {
            track_id: s.track_id,
            caps: s.caps.clone(),
        })
        .collect();

    let (bus, handle): (Bus, BusHandle) = Bus::new(16);
    let select = StreamSelectController::new();
    let mut demux = Mp4DemuxN::new(ports)
        .with_bus(handle)
        .with_stream_select(select.clone());
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::IsoBmff,
        })
        .expect("configure");

    // Split the init segment (ftyp+moov) from the fragments, so the collection is
    // announced (and the swap selected) before any fragment is emitted.
    let split = file
        .windows(4)
        .position(|w| w == b"moof")
        .expect("fragmented file has a moof");
    let (init, fragments) = file.split_at(split);

    let mut tap = PortTap::new(2);
    demux.process(data_frame(init), &mut tap).await.unwrap();
    assert!(
        tap.frames.iter().all(|p| p.is_empty()),
        "no frames before any fragment"
    );

    let (v, a) = discover_av_ids(&bus).expect("collection announced from the moov");
    select.select(vec![a.clone(), v.clone()]); // swap: port 0 <- audio, port 1 <- video

    demux
        .process(data_frame(fragments), &mut tap)
        .await
        .unwrap();
    demux.process(PipelinePacket::Eos, &mut tap).await.unwrap();

    // Port 0 re-mapped to the audio track, port 1 to the video track.
    assert!(
        tap.caps[0]
            .last()
            .map(|c| matches!(
                c,
                Caps::Audio {
                    format: AudioFormat::Aac,
                    ..
                }
            ))
            .unwrap_or(false),
        "port 0 re-mapped to the audio track: {:?}",
        tap.caps[0]
    );
    assert_eq!(
        tap.frames[0].len(),
        2,
        "port 0 carries the two audio access units after the swap"
    );
    assert!(
        tap.caps[1]
            .last()
            .map(|c| matches!(
                c,
                Caps::CompressedVideo {
                    codec: VideoCodec::H264,
                    ..
                }
            ))
            .unwrap_or(false),
        "port 1 re-mapped to the video track: {:?}",
        tap.caps[1]
    );
    assert_eq!(
        tap.frames[1].len(),
        2,
        "port 1 carries the two video access units after the swap"
    );

    let selected_ids: Vec<_> = core::iter::from_fn(|| bus.try_recv())
        .filter_map(|m| match m {
            BusMessage::StreamsSelected { ids } => Some(ids),
            _ => None,
        })
        .collect();
    assert_eq!(
        selected_ids,
        vec![vec![a, v]],
        "StreamsSelected confirms the per-port track ids"
    );
}
