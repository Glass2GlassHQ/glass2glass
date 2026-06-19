//! M28: `Mp4Src` reads back what `Mp4Sink` writes. Round trip is
//! byte-exact (Annex-B in, fMP4, Annex-B out), the caps probe recovers the
//! recorded geometry during negotiation, and on Windows the full circle
//! runs encode -> container -> demux -> decode through both real MFTs.

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{Caps, Dim, G2gError, Rate, VideoCodec};
use g2g_plugins::mp4sink::Mp4Sink;
use g2g_plugins::mp4src::Mp4Src;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m28_{}_{}.mp4", std::process::id(), name))
}

fn h264_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

#[derive(Default)]
struct Collect {
    packets: Vec<PipelinePacket>,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            self.packets.push(packet);
            Ok(PushOutcome::Accepted)
        })
    }
}

impl Collect {
    fn frames(&self) -> Vec<&Frame> {
        self.packets
            .iter()
            .filter_map(|p| match p {
                PipelinePacket::DataFrame(f) => Some(f),
                _ => None,
            })
            .collect()
    }
}

fn frame_bytes(f: &Frame) -> &[u8] {
    let MemoryDomain::System(slice) = &f.domain else {
        panic!("System frames expected");
    };
    slice.as_slice()
}

fn au_frame(bytes: Vec<u8>, pts_ns: u64, sequence: u64) -> Frame {
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns,
            dts_ns: pts_ns,
            duration_ns: 33_333_333,
            ..FrameTiming::default()
        },
        sequence,
        meta: Default::default(),
    }
}

/// Record `aus` through `Mp4Sink` into `path`.
async fn record(path: &PathBuf, aus: &[Vec<u8>], w: u32, h: u32) {
    let mut sink = Mp4Sink::new(path);
    sink.configure_pipeline(&h264_caps(w, h)).expect("configure sink");
    let mut null = Collect::default();
    for (i, au) in aus.iter().enumerate() {
        sink.process(
            PipelinePacket::DataFrame(au_frame(au.clone(), i as u64 * 33_333_333, i as u64)),
            &mut null,
        )
        .await
        .expect("mux AU");
    }
    sink.process(PipelinePacket::Eos, &mut null).await.expect("eos");
}

#[tokio::test]
async fn round_trip_recovers_access_units_and_timing() {
    let path = temp_path("roundtrip");
    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    let idr_au: Vec<u8> = [
        &[0, 0, 0, 1][..],
        &sps,
        &[0, 0, 0, 1],
        &pps,
        &[0, 0, 0, 1],
        &[0x65, 0xAA, 0xBB],
    ]
    .concat();
    let p_au = |fill: u8| [&[0, 0, 0, 1][..], &[0x41, fill, fill, fill]].concat();
    let aus = vec![idr_au, p_au(1), p_au(2)];

    record(&path, &aus, 64, 48).await;

    // probe before negotiation: dims recovered from the moov.
    let mut src = Mp4Src::new(&path);
    let caps = src.intercept_caps().await.expect("probe header");
    assert_eq!(
        caps,
        Caps::CompressedVideo {
            codec: VideoCodec::H264,
            width: Dim::Fixed(64),
            height: Dim::Fixed(48),
            framerate: Rate::Any,
        }
    );

    src.configure_pipeline(&caps).expect("configure");
    let mut out = Collect::default();
    let produced = src.run(&mut out).await.expect("demux to EOS");
    assert_eq!(produced, 3);

    let frames = out.frames();
    assert_eq!(frames.len(), 3);
    for (i, original) in aus.iter().enumerate() {
        assert_eq!(
            frame_bytes(frames[i]),
            &original[..],
            "AU {i} must round trip byte-exactly"
        );
    }
    // timing recovered from tfdt/trun at 90 kHz granularity.
    assert_eq!(frames[0].timing.pts_ns, 0);
    let pts1 = frames[1].timing.pts_ns;
    assert!(
        (pts1 as i64 - 33_333_333).abs() < 20_000,
        "second AU pts {pts1} should be ~33.33 ms (90 kHz rounding)"
    );
    assert!(frames[0].timing.duration_ns > 33_000_000);
    assert!(
        matches!(out.packets.last(), Some(PipelinePacket::Eos)),
        "EOS terminates the stream"
    );
    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn missing_or_invalid_file_fails_loud() {
    let mut missing = Mp4Src::new(temp_path("missing"));
    assert!(missing.intercept_caps().await.is_err());

    // a non-MP4 file is rejected at probe, not silently emitted.
    let path = temp_path("garbage");
    std::fs::write(&path, b"not an mp4 at all").expect("write");
    let mut garbage = Mp4Src::new(&path);
    assert_eq!(
        garbage.intercept_caps().await.err(),
        Some(G2gError::CapsMismatch)
    );
    let _ = std::fs::remove_file(&path);
}

/// Full circle on Windows: real encode -> container -> demux -> real decode.
#[cfg(all(target_os = "windows", feature = "mf-encode", feature = "mf-decode"))]
#[tokio::test(flavor = "current_thread")]
async fn encode_mux_demux_decode_full_circle() {
    use g2g_plugins::mfdecode::MfDecode;
    use g2g_plugins::mfencode::MfEncode;

    const W: u32 = 320;
    const H: u32 = 240;
    const FRAMES: usize = 10;

    // encode synthetic NV12.
    let mut enc = MfEncode::new();
    let nv12 = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Nv12,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    };
    enc.configure_pipeline(&nv12).expect("encoder init");
    let mut encoded = Collect::default();
    for i in 0..FRAMES {
        let mut data = vec![128u8; (W * H * 3 / 2) as usize];
        for (j, b) in data.iter_mut().take((W * H) as usize).enumerate() {
            *b = ((j + i * 16) % 256) as u8;
        }
        enc.process(
            PipelinePacket::DataFrame(au_frame(data, i as u64 * 33_333_333, i as u64)),
            &mut encoded,
        )
        .await
        .expect("encode");
    }
    enc.process(PipelinePacket::Eos, &mut encoded).await.expect("drain");

    // mux to fMP4.
    let path = temp_path("full_circle");
    let mut mux = Mp4Sink::new(&path);
    mux.configure_pipeline(&h264_caps(W, H)).expect("configure mux");
    let mut null = Collect::default();
    for f in encoded.frames() {
        mux.process(
            PipelinePacket::DataFrame(au_frame(
                frame_bytes(f).to_vec(),
                f.timing.pts_ns,
                f.sequence,
            )),
            &mut null,
        )
        .await
        .expect("mux");
    }
    mux.process(PipelinePacket::Eos, &mut null).await.expect("eos");

    // demux and decode.
    let mut src = Mp4Src::new(&path);
    let caps = src.intercept_caps().await.expect("probe");
    src.configure_pipeline(&caps).expect("configure src");
    let mut demuxed = Collect::default();
    src.run(&mut demuxed).await.expect("demux");

    let mut dec = MfDecode::new();
    dec.configure_pipeline(&caps).expect("decoder init");
    let mut decoded = Collect::default();
    for f in demuxed.frames() {
        dec.process(
            PipelinePacket::DataFrame(au_frame(
                frame_bytes(f).to_vec(),
                f.timing.pts_ns,
                f.sequence,
            )),
            &mut decoded,
        )
        .await
        .expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut decoded).await.expect("drain");

    let frames = decoded.frames();
    assert_eq!(frames.len(), FRAMES, "every frame survives the full circle");
    let expected_len = (W * H * 3 / 2) as usize;
    for f in frames {
        assert_eq!(frame_bytes(f).len(), expected_len, "packed NV12 out");
    }
    let _ = std::fs::remove_file(&path);
}
