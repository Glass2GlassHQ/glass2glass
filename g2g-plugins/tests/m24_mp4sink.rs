//! M24: `Mp4Sink` writes a structurally valid fragmented MP4. The
//! platform-agnostic test feeds synthetic access units and walks the box
//! tree back; the Windows test records a real `MfEncode` stream.

use g2g_core::element::AsyncElement;
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{Caps, Dim, G2gError, Rate, VideoCodec};
use g2g_plugins::mp4sink::Mp4Sink;

use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m24_{}_{}.mp4", std::process::id(), name))
}

fn h264_caps(w: u32, h: u32) -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(w),
        height: Dim::Fixed(h),
        framerate: Rate::Fixed(30 << 16),
    }
}

struct NullOut;
impl g2g_core::OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _packet: PipelinePacket,
    ) -> g2g_core::element::BoxFuture<'a, Result<g2g_core::element::PushOutcome, G2gError>> {
        Box::pin(async { Ok(g2g_core::element::PushOutcome::Accepted) })
    }
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

/// Walk the top-level box sequence of an MP4 buffer: (fourcc, payload range).
fn walk_boxes(data: &[u8]) -> Vec<(String, usize, usize)> {
    let mut boxes = Vec::new();
    let mut i = 0;
    while i + 8 <= data.len() {
        let size = u32::from_be_bytes(data[i..i + 4].try_into().unwrap()) as usize;
        let kind = String::from_utf8_lossy(&data[i + 4..i + 8]).into_owned();
        assert!(size >= 8 && i + size <= data.len(), "box {kind} overruns file");
        boxes.push((kind, i + 8, i + size));
        i += size;
    }
    assert_eq!(i, data.len(), "trailing bytes after the last box");
    boxes
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

#[tokio::test]
async fn writes_ftyp_moov_then_one_fragment_per_access_unit() {
    let path = temp_path("synthetic");
    let mut sink = Mp4Sink::new(&path);
    sink.configure_pipeline(&h264_caps(64, 48)).expect("configure");

    let sps = [0x67u8, 0x42, 0xC0, 0x1E, 0x11, 0x22];
    let pps = [0x68u8, 0xCE, 0x3C, 0x80];
    // first AU: SPS + PPS + IDR; later AUs: non-IDR slices.
    let idr_au: Vec<u8> = [
        &[0, 0, 0, 1][..],
        &sps,
        &[0, 0, 0, 1],
        &pps,
        &[0, 0, 0, 1],
        &[0x65, 0xAA, 0xBB],
    ]
    .concat();
    let p_au = |fill: u8| [&[0, 0, 0, 1][..], &[0x41, fill, fill]].concat();

    let mut out = NullOut;
    sink.process(PipelinePacket::DataFrame(au_frame(idr_au, 0, 0)), &mut out)
        .await
        .expect("IDR AU");
    for i in 1..4u64 {
        sink.process(
            PipelinePacket::DataFrame(au_frame(p_au(i as u8), i * 33_333_333, i)),
            &mut out,
        )
        .await
        .expect("P AU");
    }
    sink.process(PipelinePacket::Eos, &mut out).await.expect("eos");
    assert!(sink.eos_seen());
    assert_eq!(sink.fragments_written(), 4);

    let data = std::fs::read(&path).expect("mp4 exists");
    let boxes = walk_boxes(&data);
    let kinds: Vec<&str> = boxes.iter().map(|(k, _, _)| k.as_str()).collect();
    assert_eq!(
        kinds,
        vec!["ftyp", "moov", "moof", "mdat", "moof", "mdat", "moof", "mdat", "moof", "mdat"],
        "header then one moof+mdat per access unit"
    );

    // moov carries the avcC with the exact in-band SPS/PPS bytes.
    let (_, moov_start, moov_end) = boxes.iter().find(|(k, _, _)| k == "moov").unwrap();
    let moov = &data[*moov_start..*moov_end];
    assert!(find_subslice(moov, b"avcC").is_some(), "moov holds an avcC");
    assert!(find_subslice(moov, &sps).is_some(), "avcC holds the SPS");
    assert!(find_subslice(moov, &pps).is_some(), "avcC holds the PPS");

    // the first mdat holds the AVCC (length-prefixed) IDR sample.
    let (_, mdat_start, mdat_end) = boxes.iter().find(|(k, _, _)| k == "mdat").unwrap();
    let mdat = &data[*mdat_start..*mdat_end];
    assert!(
        find_subslice(mdat, &[0, 0, 0, 3, 0x65, 0xAA, 0xBB]).is_some(),
        "IDR NALU is 4-byte length prefixed in the mdat"
    );

    // fragment sequence numbers increment from 1 (mfhd payload).
    let moofs: Vec<_> = boxes.iter().filter(|(k, _, _)| k == "moof").collect();
    for (i, (_, start, end)) in moofs.iter().enumerate() {
        let moof = &data[*start..*end];
        let p = find_subslice(moof, b"mfhd").expect("moof holds mfhd");
        let seq = u32::from_be_bytes(moof[p + 8..p + 12].try_into().unwrap());
        assert_eq!(seq as usize, i + 1);
    }

    let _ = std::fs::remove_file(&path);
}

#[tokio::test]
async fn rejects_non_h264_caps() {
    let mut sink = Mp4Sink::new(temp_path("reject"));
    let raw = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Nv12,
        width: Dim::Fixed(64),
        height: Dim::Fixed(48),
        framerate: Rate::Any,
    };
    let err = sink.configure_pipeline(&raw).expect_err("raw video rejected");
    assert_eq!(err, G2gError::CapsMismatch);
    assert_eq!(sink.intercept_caps(&raw), Err(G2gError::CapsMismatch));
}

/// Record a real software-encoder stream into the container on Windows.
#[cfg(all(target_os = "windows", feature = "mf-encode"))]
#[tokio::test(flavor = "current_thread")]
async fn records_a_real_mfencode_stream() {
    use g2g_plugins::mfencode::MfEncode;

    const W: u32 = 320;
    const H: u32 = 240;
    const FRAMES: usize = 10;

    // collect encoded AUs from the real encoder MFT.
    struct Collect(Vec<PipelinePacket>);
    impl g2g_core::OutputSink for Collect {
        fn push<'a>(
            &'a mut self,
            packet: PipelinePacket,
        ) -> g2g_core::element::BoxFuture<'a, Result<g2g_core::element::PushOutcome, G2gError>>
        {
            self.0.push(packet);
            Box::pin(async { Ok(g2g_core::element::PushOutcome::Accepted) })
        }
    }

    let mut enc = MfEncode::new();
    let nv12 = Caps::RawVideo {
        format: g2g_core::RawVideoFormat::Nv12,
        width: Dim::Fixed(W),
        height: Dim::Fixed(H),
        framerate: Rate::Fixed(30 << 16),
    };
    enc.configure_pipeline(&nv12).expect("encoder init");
    let mut encoded = Collect(Vec::new());
    for i in 0..FRAMES {
        let mut data = vec![128u8; (W * H * 3 / 2) as usize];
        for (j, b) in data.iter_mut().take((W * H) as usize).enumerate() {
            *b = ((j + i * 16) % 256) as u8;
        }
        let frame = au_frame(data, i as u64 * 33_333_333, i as u64);
        enc.process(PipelinePacket::DataFrame(frame), &mut encoded)
            .await
            .expect("encode");
    }
    enc.process(PipelinePacket::Eos, &mut encoded).await.expect("drain");

    let path = temp_path("mfencode");
    let mut sink = Mp4Sink::new(&path);
    sink.configure_pipeline(&h264_caps(W, H)).expect("configure");
    let mut null = NullOut;
    for p in encoded.0 {
        if let PipelinePacket::DataFrame(f) = p {
            sink.process(PipelinePacket::DataFrame(f), &mut null)
                .await
                .expect("mux AU");
        }
    }
    sink.process(PipelinePacket::Eos, &mut null).await.expect("eos");

    assert_eq!(sink.fragments_written(), FRAMES as u64);
    let data = std::fs::read(&path).expect("mp4 exists");
    let kinds: Vec<String> = walk_boxes(&data).into_iter().map(|(k, _, _)| k).collect();
    assert_eq!(&kinds[..2], &["ftyp", "moov"]);
    assert_eq!(kinds.len(), 2 + 2 * FRAMES, "one moof+mdat per encoded frame");
    let _ = std::fs::remove_file(&path);
}
