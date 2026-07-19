//! M643: the RTP egress seam. The mock-sender tests assert the datasheet-
//! level contract (header fields, PTS -> media-tick timestamping, sequence
//! wrap, first-packet marker, one-frame-one-packet sizing); the ignored
//! ffmpeg test is the real-peer validation: the reference audio chain's
//! tail (G.711 encode -> RtpSink) sends PCMU/RTP over a local UDP socket
//! into ffmpeg's RTP demuxer, and ffmpeg's decoded byte stream must equal
//! our payloads exactly (the loopback a g2g<->g2g test cannot provide).

mod util;

use g2g_core::error::G2gError;
use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_core::staticpool::StaticLendRing;
use g2g_core::{MediaClock, StaticSink};
use g2g_mcu::rtp::PacketSender;
use g2g_mcu::RtpSink;
use util::{block_on, frame_of, le_bytes};

/// Records every packet as one contiguous datagram, as a UDP stack would.
#[derive(Default)]
struct Collect {
    packets: Vec<Vec<u8>>,
}

impl PacketSender for Collect {
    async fn send(
        &mut self,
        header: &[u8; RTP_HEADER_LEN],
        payload: &[u8],
    ) -> Result<(), G2gError> {
        let mut pkt = header.to_vec();
        pkt.extend_from_slice(payload);
        self.packets.push(pkt);
        Ok(())
    }
}

/// 20 ms of G.711 at 8 kHz: the canonical telephony packet.
const PTIME_NS: u64 = 20_000_000;
const PTIME_BYTES: usize = 160;

#[test]
fn packets_carry_the_contracted_header() {
    let ring: StaticLendRing<1, 256> = StaticLendRing::new();
    let mut sink = RtpSink::new(
        Collect::default(),
        MediaClock::audio(8000),
        0,
        0xDEAD_BEEF,
        100,
    );
    for i in 0..3u64 {
        let payload = [i as u8; PTIME_BYTES];
        let frame = frame_of(&ring, &payload, i * PTIME_NS, i);
        block_on(sink.consume(frame)).expect("send");
    }
    let packets = sink.free().packets;
    assert_eq!(packets.len(), 3, "one packet per frame");
    for (i, pkt) in packets.iter().enumerate() {
        assert_eq!(pkt.len(), RTP_HEADER_LEN + PTIME_BYTES);
        assert_eq!(pkt[0], 0x80, "V=2, no padding/extension/CSRCs");
        let marker = pkt[1] & 0x80 != 0;
        assert_eq!(
            marker,
            i == 0,
            "marker only on the first packet (talkspurt start)"
        );
        assert_eq!(pkt[1] & 0x7F, 0, "static PT 0 = PCMU");
        let seq = u16::from_be_bytes([pkt[2], pkt[3]]);
        assert_eq!(
            seq,
            100 + i as u16,
            "sequence starts at the seed and increments"
        );
        let ts = u32::from_be_bytes([pkt[4], pkt[5], pkt[6], pkt[7]]);
        assert_eq!(
            ts,
            i as u32 * PTIME_BYTES as u32,
            "PTS in 8 kHz media ticks: 160 per 20 ms"
        );
        let ssrc = u32::from_be_bytes([pkt[8], pkt[9], pkt[10], pkt[11]]);
        assert_eq!(ssrc, 0xDEAD_BEEF);
        assert_eq!(
            &pkt[RTP_HEADER_LEN..],
            &[i as u8; PTIME_BYTES],
            "payload verbatim"
        );
    }
}

#[test]
fn sequence_wraps() {
    let ring: StaticLendRing<1, 16> = StaticLendRing::new();
    let mut sink = RtpSink::new(Collect::default(), MediaClock::audio(8000), 0, 1, u16::MAX);
    for i in 0..2u64 {
        let frame = frame_of(&ring, &[0u8; 8], i * PTIME_NS, i);
        block_on(sink.consume(frame)).expect("send");
    }
    assert_eq!(sink.next_sequence(), 1, "u16 wrap, no panic path");
    let packets = sink.free().packets;
    assert_eq!(u16::from_be_bytes([packets[0][2], packets[0][3]]), u16::MAX);
    assert_eq!(u16::from_be_bytes([packets[1][2], packets[1][3]]), 0);
}

#[test]
fn oversized_and_empty_payloads_are_rejected() {
    let ring: StaticLendRing<2, 64> = StaticLendRing::new();
    let mut sink =
        RtpSink::new(Collect::default(), MediaClock::audio(8000), 0, 1, 0).with_max_payload(16);
    let over = frame_of(&ring, &[0u8; 17], 0, 0);
    assert!(
        matches!(block_on(sink.consume(over)), Err(G2gError::CapsMismatch)),
        "an over-MTU frame is a configuration bug, never fragmented"
    );
    let empty = frame_of(&ring, &[], 0, 0);
    assert!(
        matches!(block_on(sink.consume(empty)), Err(G2gError::CapsMismatch)),
        "an empty payload makes no packet"
    );
    assert!(sink.free().packets.is_empty(), "nothing reached the wire");
}

// ---- real-peer validation (the conformance job runs this with --ignored) ----

use std::io::Write as _;
use std::net::UdpSocket;
use std::process::Command;
use std::time::Duration;

/// A [`PacketSender`] over a real UDP socket: the host stand-in for the
/// board's network stack, concatenating the scatter-gather pair like a
/// flat-buffer stack would.
struct UdpSender {
    socket: UdpSocket,
    dest: String,
}

impl PacketSender for UdpSender {
    async fn send(
        &mut self,
        header: &[u8; RTP_HEADER_LEN],
        payload: &[u8],
    ) -> Result<(), G2gError> {
        let mut pkt = Vec::with_capacity(RTP_HEADER_LEN + payload.len());
        pkt.extend_from_slice(header);
        pkt.extend_from_slice(payload);
        self.socket
            .send_to(&pkt, &self.dest)
            .map_err(|_| G2gError::Hardware(g2g_core::error::HardwareError::Peripheral))?;
        Ok(())
    }
}

#[test]
#[ignore = "needs ffmpeg with the rtp demuxer; opens a local UDP socket"]
fn ffmpeg_depacketizes_the_rtp_stream() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("skipping: ffmpeg not available");
        return;
    }

    // A free local port: bind, read it back, release it to ffmpeg.
    let port = UdpSocket::bind("127.0.0.1:0")
        .expect("probe port")
        .local_addr()
        .unwrap()
        .port();

    // ffmpeg is the receiving peer: it reads the session description, binds
    // the port, depacketizes PCMU/RTP, and writes the raw mu-law bytes out.
    let dir = std::env::temp_dir().join(format!("g2g-m643-{port}"));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let sdp_path = dir.join("g2g.sdp");
    let out_path = dir.join("out.ul");
    let mut sdp = std::fs::File::create(&sdp_path).expect("sdp file");
    write!(
        sdp,
        "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=g2g\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\nm=audio {port} RTP/AVP 0\r\n"
    )
    .expect("write sdp");
    drop(sdp);
    let mut ffmpeg = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file,udp,rtp"])
        .args(["-i", sdp_path.to_str().unwrap()])
        // Two seconds of media, then ffmpeg closes the output and exits.
        .args(["-t", "2", "-f", "mulaw", "-y", out_path.to_str().unwrap()])
        .spawn()
        .expect("spawn ffmpeg");
    // Let ffmpeg bind before the first datagram; RTP has no handshake.
    std::thread::sleep(Duration::from_millis(800));

    // The reference chain's tail: S16LE PCM -> G711Enc -> RtpSink, 20 ms
    // frames. 2.5 s of signal so ffmpeg's 2 s window closes with margin.
    let samples: Vec<i16> = (0..20_000)
        .map(|i| (((i * 331) % 24001) - 12000) as i16)
        .collect();
    let pcm_ring: StaticLendRing<1, 512> = StaticLendRing::new();
    let ulaw_ring: StaticLendRing<1, 256> = StaticLendRing::new();
    // SAFETY: the ring outlives every frame in this test.
    let mut enc = unsafe { g2g_mcu::G711Enc::with_ring(g2g_mcu::Law::Mulaw, &ulaw_ring) };
    let socket = UdpSocket::bind("127.0.0.1:0").expect("bind send socket");
    let sender = UdpSender {
        socket,
        dest: format!("127.0.0.1:{port}"),
    };
    let mut sink = RtpSink::new(sender, MediaClock::audio(8000), 0, 0x6767_6767, 0);

    let mut expected = Vec::new();
    for (i, chunk) in samples.chunks(PTIME_BYTES).enumerate() {
        let frame = frame_of(&pcm_ring, &le_bytes(chunk), i as u64 * PTIME_NS, i as u64);
        let coded = block_on(g2g_core::StaticTransform::process(&mut enc, frame))
            .expect("encode")
            .expect("frame");
        expected.extend_from_slice(util::payload(&coded));
        block_on(sink.consume(coded)).expect("send");
        // Light pacing: enough to keep loopback reordering/drops away without
        // making the test minutes long (real time would be 20 ms per frame).
        std::thread::sleep(Duration::from_millis(2));
    }

    // ffmpeg exits on its own at -t; give it a bounded wait.
    let mut waited = Duration::ZERO;
    let status = loop {
        if let Some(s) = ffmpeg.try_wait().expect("poll ffmpeg") {
            break s;
        }
        if waited > Duration::from_secs(20) {
            let _ = ffmpeg.kill();
            panic!("ffmpeg did not finish its 2s capture window");
        }
        std::thread::sleep(Duration::from_millis(100));
        waited += Duration::from_millis(100);
    };
    assert!(status.success(), "ffmpeg rtp capture");

    // ffmpeg's depacketized stream must be our payload bytes, verbatim, from
    // the first packet (2 s = 16000 of the ~20000 sent).
    let theirs = std::fs::read(&out_path).expect("ffmpeg output");
    assert!(
        theirs.len() >= 16_000,
        "ffmpeg captured its 2s window, got {}",
        theirs.len()
    );
    assert_eq!(
        &theirs[..],
        &expected[..theirs.len()],
        "ffmpeg's depacketized mu-law equals our RTP payloads byte-for-byte"
    );
    let _ = std::fs::remove_dir_all(&dir);

    // A real reference peer consumed the MCU RTP egress on the wire: persist
    // peer-tagged Oracle evidence for the maturity table.
    use g2g_core::conformance::{ConformanceDimension, Evidence};
    g2g_plugins::conformance::persist::record_evidence(
        "rtpsink",
        &Evidence::new(ConformanceDimension::Oracle)
            .peer("ffmpeg")
            .codec("pcmu")
            .detail("PCMU/RTP (RFC 3551 PT 0) depacketized by ffmpeg byte-for-byte"),
    )
    .expect("record oracle evidence");
}
