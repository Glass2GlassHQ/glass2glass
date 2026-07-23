//! M761: `UdpSink`'s FEC modes as launch properties. Proves a `gst-launch`-style
//! property assignment (the same `set_property` path `parse_launch` drives) both
//! selects the FEC scheme and applies its block geometry: the repair packets that
//! reach the wire match the configured rows x columns, not just a stored field.

#![cfg(feature = "udp-egress")]

use core::future::Future;
use core::pin::Pin;
use std::net::UdpSocket as StdUdpSocket;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::parse_launch;
use g2g_core::{
    AsyncElement, Caps, Dim, FrameTiming, G2gError, MemoryDomain, OutputSink, PipelinePacket,
    PropValue, PushOutcome, Rate, VideoCodec,
};
use g2g_plugins::registry::default_registry;
use g2g_plugins::udpsink::UdpSink;

struct NullOut;
impl OutputSink for NullOut {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async { Ok(PushOutcome::Accepted) })
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(1280),
        height: Dim::Fixed(720),
        framerate: Rate::Fixed(30 << 16),
    }
}

fn alloc_au(index: u8) -> Vec<u8> {
    let mut au = vec![0u8, 0, 0, 1, 0x65, index];
    au.extend_from_slice(&[0xAA; 8]);
    au
}

const MEDIA_PT: u8 = 96;
const FEC_PT: u8 = 98;

/// Drive `frames` single-NAL access units into a sink configured by `set_props`
/// (a closure applying `set_property` as `parse_launch` would), collecting every
/// datagram it emits. Returns `(media packet count, FEC repair count)`, split by
/// RTP payload type.
async fn run_and_split(frames: u8, set_props: impl FnOnce(&mut UdpSink)) -> (usize, usize) {
    // A bound receiver socket is the sink's destination; drain it after sending.
    let recv = StdUdpSocket::bind("127.0.0.1:0").expect("bind receiver");
    recv.set_nonblocking(true).unwrap();
    let dest = recv.local_addr().unwrap();

    let mut sink = UdpSink::new(dest);
    set_props(&mut sink);
    sink.configure_pipeline(&h264_caps())
        .expect("configure sink");

    let mut null = NullOut;
    for i in 0u8..frames {
        let au = alloc_au(i);
        let frame = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.into_boxed_slice())),
            timing: FrameTiming {
                pts_ns: i as u64 * 33_000_000,
                ..FrameTiming::default()
            },
            sequence: i as u64,
            meta: Default::default(),
        };
        sink.process(PipelinePacket::DataFrame(frame), &mut null)
            .await
            .expect("send");
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }

    let mut media = 0usize;
    let mut fec = 0usize;
    let mut buf = [0u8; 2048];
    // Give the loopback datagrams a moment to land, then drain non-blocking.
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    while let Ok(n) = recv.recv(&mut buf) {
        if n < 12 {
            continue;
        }
        match buf[1] & 0x7F {
            p if p == MEDIA_PT => media += 1,
            p if p == FEC_PT => fec += 1,
            _ => {}
        }
    }
    (media, fec)
}

#[tokio::test]
async fn flexfec_2d_property_emits_row_and_column_repairs() {
    // fec-mode=flexfec with a 4-column x 3-row block: one full block of 12 media
    // packets yields 3 row repairs + 4 column repairs = 7 FEC packets.
    let (media, fec) = run_and_split(12, |s| {
        s.set_property("fec-mode", PropValue::Str("flexfec".into()))
            .unwrap();
        s.set_property("fec-columns", PropValue::Uint(4)).unwrap();
        s.set_property("fec-rows", PropValue::Uint(3)).unwrap();
        s.set_property("fec-payload-type", PropValue::Uint(FEC_PT as u64))
            .unwrap();
        s.set_property("fec-ssrc", PropValue::Uint(0xFEC0_0004))
            .unwrap();
    })
    .await;
    assert_eq!(media, 12, "one RTP packet per access unit");
    assert_eq!(
        fec, 7,
        "3 row + 4 column FlexFEC repairs over the 4x3 block"
    );
}

#[tokio::test]
async fn flexfec_1d_property_emits_only_row_repairs() {
    // Same scheme, rows=1: the block is 1-D, so only the per-column-group row
    // repairs appear (12 media / group 4 = 3 repairs, no column repairs). Proves
    // fec-rows actually changes the geometry on the wire.
    let (media, fec) = run_and_split(12, |s| {
        s.set_property("fec-mode", PropValue::Str("flexfec".into()))
            .unwrap();
        s.set_property("fec-columns", PropValue::Uint(4)).unwrap();
        s.set_property("fec-rows", PropValue::Uint(1)).unwrap();
        s.set_property("fec-payload-type", PropValue::Uint(FEC_PT as u64))
            .unwrap();
        s.set_property("fec-ssrc", PropValue::Uint(0xFEC0_0005))
            .unwrap();
    })
    .await;
    assert_eq!(media, 12);
    assert_eq!(fec, 3, "one row repair per group of 4, no column repairs");
}

#[tokio::test]
async fn fec_mode_none_emits_no_repairs() {
    // The default (no fec-mode set) sends media only, so the property genuinely
    // gates the repair stream.
    let (media, fec) = run_and_split(12, |_| {}).await;
    assert_eq!(media, 12);
    assert_eq!(fec, 0, "no FEC packets without a fec-mode");
}

#[test]
fn parse_launch_accepts_fec_properties_and_rejects_bogus() {
    let reg = default_registry();
    // parse_launch looks the fec-* names up in properties() and calls
    // set_property; a line setting them must build.
    assert!(
        parse_launch(
            &reg,
            "videotestsrc num-buffers=2 ! udpsink host=127.0.0.1 port=5004 \
             fec-mode=flexfec fec-columns=4 fec-rows=3 fec-payload-type=98 fec-ssrc=1"
        )
        .is_ok(),
        "a launch line setting the FEC knobs parses"
    );
    assert!(
        parse_launch(
            &reg,
            "videotestsrc num-buffers=2 ! udpsink host=127.0.0.1 fec-mode=nonsense"
        )
        .is_err(),
        "an invalid FEC scheme is rejected"
    );
}
