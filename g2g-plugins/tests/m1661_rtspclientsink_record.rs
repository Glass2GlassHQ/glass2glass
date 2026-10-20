#![cfg(all(feature = "rtsp-client-sink", feature = "rtsp-server"))]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::net::TcpListener as StdTcpListener;
use std::rc::Rc;
use std::time::Duration;

use md5::{Digest, Md5};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::rtp::RTP_HEADER_LEN;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    AudioFormat, Caps, ChannelLayout, Dim, FrameTiming, G2gError, MemoryDomain, MultiInputElement,
    OutputSink, PipelinePacket, PushOutcome, Rate, VideoCodec,
};

use g2g_core::mediaclock::MediaClock;
use g2g_core::{unix_nanos_from_ntp, PropValue};
use g2g_plugins::rtcp::{self, RtcpPacket};
use g2g_plugins::rtpdepay::RtpH264Depayloader;
use g2g_plugins::rtspclient::LowerTransport;
use g2g_plugins::rtspclientsink::RtspClientSink;
use g2g_plugins::rtspserver::{
    interleaved_frame, RtspEvent, RtspRequest, RtspResponder, INTERLEAVED_MARKER,
};
use g2g_plugins::rtspserversrc::RtspServerSrc;
use g2g_plugins::sdp::{scan_sections, SdpMedia};
use g2g_plugins::textencoding::hex_encode;

// High profile SPS header (profile_idc 100, level 40) and a PPS, as the RTSP server's SDP tests use
const SPS: &[u8] = &[0x67, 0x64, 0x00, 0x28, 0xAC, 0xD9];
const PPS: &[u8] = &[0x68, 0xEB, 0xEC, 0xB2];
const IDR_NAL_HEADER: u8 = 0x65;
const START_CODE: &[u8] = &[0, 0, 0, 1];
const ACCESS_UNITS: usize = 8;
// past the 1400-byte packet payload, so every access unit goes out as FU-A fragments
const IDR_BODY_LEN: usize = 3000;
const FRAME_PERIOD_NS: u64 = 33_333_333;
const SERVER_DEADLINE: Duration = Duration::from_secs(10);
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const NS_PER_SECOND: u64 = 1_000_000_000;
// the smallest rtcp-sr-interval the element accepts
const MIN_REPORT_INTERVAL_MS: u64 = 1;

const AUDIO_RATE_HZ: u32 = 48_000;
const AUDIO_CHANNELS: u8 = 2;
const AAC_LC_OBJECT_TYPE: u8 = 2;
const AAC_48K_RATE_INDEX: u8 = 3;
const AAC_FRAME_SAMPLES: u64 = 1024;
const AAC_AU_HEADER_SECTION_LEN: usize = 4;
const ADTS_HEADER_LEN: usize = 7;
const OPUS_FRAME_SAMPLES: u64 = 960;
// TOC config 31 (CELT fullband 20 ms) with the stereo flag
const OPUS_STEREO_TOC: u8 = 0xFC;

const VIDEO_PAD: usize = 0;
const AAC_PAD: usize = 1;
const OPUS_PAD: usize = 2;

const DIGEST_REALM: &str = "g2g-test";
const DIGEST_NONCE: &str = "6e6f6e6365";
const USER: &str = "publisher";
const PASSWORD: &str = "s3cret";

#[derive(Default)]
struct Capture {
    access_units: Rc<RefCell<Vec<Vec<u8>>>>,
}

impl OutputSink for Capture {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        let packet = packet_slot.take().expect("poll_push without a packet");
        if let PipelinePacket::DataFrame(frame) = &packet {
            let data = frame.domain.as_system_slice().expect("system frame");
            self.access_units.borrow_mut().push(data.to_vec());
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn h264_caps() -> Caps {
    Caps::CompressedVideo {
        codec: VideoCodec::H264,
        width: Dim::Fixed(320),
        height: Dim::Fixed(240),
        framerate: Rate::Fixed(30 << 16),
        colorimetry: g2g_core::Colorimetry::UNKNOWN,
    }
}

fn audio_caps(format: AudioFormat) -> Caps {
    Caps::Audio {
        format,
        channels: AUDIO_CHANNELS,
        sample_rate: AUDIO_RATE_HZ,
        channel_layout: ChannelLayout::UNSPECIFIED,
    }
}

fn annex_b(nals: &[&[u8]]) -> Vec<u8> {
    nals.iter()
        .flat_map(|nal| START_CODE.iter().chain(nal.iter()).copied())
        .collect()
}

// the first access unit carries the parameter sets the ANNOUNCE needs, every one is tagged at the IDR's first body byte
fn access_unit(index: usize) -> Vec<u8> {
    let mut idr = vec![IDR_NAL_HEADER, index as u8];
    idr.extend((0..IDR_BODY_LEN).map(|i| (i % 251) as u8 | 1));
    if index == 0 {
        annex_b(&[SPS, PPS, &idr])
    } else {
        annex_b(&[&idr])
    }
}

fn video_frames() -> Vec<Vec<u8>> {
    (0..ACCESS_UNITS).map(access_unit).collect()
}

// ISO/IEC 13818-7 ADTS header without CRC, for AAC LC at 48 kHz stereo
fn adts_frame(payload: &[u8]) -> Vec<u8> {
    let len = ADTS_HEADER_LEN + payload.len();
    let profile = AAC_LC_OBJECT_TYPE - 1;
    let mut frame = vec![
        0xFF,
        0xF1,
        (profile << 6) | (AAC_48K_RATE_INDEX << 2) | (AUDIO_CHANNELS >> 2),
        ((AUDIO_CHANNELS & 0x3) << 6) | (len >> 11) as u8,
        (len >> 3) as u8,
        (((len & 0x7) << 5) as u8) | 0x1F,
        0xFC,
    ];
    frame.extend_from_slice(payload);
    frame
}

// the two-byte AudioSpecificConfig the same object type, rate index and channel count spell
fn expected_audio_specific_config() -> Vec<u8> {
    let bits = (u16::from(AAC_LC_OBJECT_TYPE) << 11)
        | (u16::from(AAC_48K_RATE_INDEX) << 7)
        | (u16::from(AUDIO_CHANNELS) << 3);
    bits.to_be_bytes().to_vec()
}

fn aac_frames() -> Vec<Vec<u8>> {
    (0..ACCESS_UNITS)
        .map(|i| adts_frame(&[0x21, i as u8, 0xA5, 0x5A]))
        .collect()
}

fn opus_frames() -> Vec<Vec<u8>> {
    (0..ACCESS_UNITS)
        .map(|i| vec![OPUS_STEREO_TOC, i as u8, 0x77])
        .collect()
}

fn pts_at(index: usize, samples_per_frame: u64) -> u64 {
    index as u64 * samples_per_frame * NS_PER_SECOND / u64::from(AUDIO_RATE_HZ)
}

fn data_frame(data: Vec<u8>, pts_ns: u64) -> PipelinePacket {
    PipelinePacket::DataFrame(Frame::new(
        MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
        FrameTiming {
            pts_ns,
            ..FrameTiming::default()
        },
        0,
    ))
}

struct PadFrames {
    caps: Caps,
    frames: Vec<(Vec<u8>, u64)>,
}

fn video_pad() -> PadFrames {
    PadFrames {
        caps: h264_caps(),
        frames: video_frames()
            .into_iter()
            .enumerate()
            .map(|(i, f)| (f, i as u64 * FRAME_PERIOD_NS))
            .collect(),
    }
}

// pushes the pads' frames interleaved in turn, waiting `pace` after each round
async fn publish_frames(
    sink: &mut RtspClientSink,
    pads: &[PadFrames],
    pace: Duration,
) -> Result<(), G2gError> {
    let mut discard = Capture::default();
    for (input, pad) in pads.iter().enumerate() {
        sink.configure_pipeline(input, &pad.caps)?;
    }
    for i in 0..ACCESS_UNITS {
        for (input, pad) in pads.iter().enumerate() {
            let (data, pts_ns) = &pad.frames[i];
            sink.process(input, data_frame(data.clone(), *pts_ns), &mut discard)
                .await?;
        }
        tokio::time::sleep(pace).await;
    }
    Ok(())
}

async fn end_streams(sink: &mut RtspClientSink, pads: usize) -> Result<(), G2gError> {
    let mut discard = Capture::default();
    for input in 0..pads {
        sink.process(input, PipelinePacket::Eos, &mut discard)
            .await?;
    }
    Ok(())
}

async fn ingest_one_publisher(
    sink: &mut RtspClientSink,
    mut server: RtspServerSrc,
) -> (Vec<Vec<u8>>, Result<(), G2gError>) {
    server.configure_pipeline(&h264_caps()).expect("configure");
    let received = Rc::new(RefCell::new(Vec::new()));
    let mut capture = Capture {
        access_units: received.clone(),
    };
    let receive = async {
        tokio::time::timeout(SERVER_DEADLINE, server.run(&mut capture))
            .await
            .expect("server finishes")
            .expect("server runs")
    };
    let pads = [video_pad()];
    let send = async {
        publish_frames(sink, &pads, Duration::ZERO).await?;
        // UDP media can arrive after the TEARDOWN on the control connection, so EOS waits for the server
        while received.borrow().len() < ACCESS_UNITS {
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        end_streams(sink, pads.len()).await
    };
    let (_, published) = tokio::join!(receive, send);
    let access_units = received.borrow().clone();
    (access_units, published)
}

fn one_shot_server() -> (RtspServerSrc, u16) {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let port = listener.local_addr().unwrap().port();
    let server = RtspServerSrc::from_listener(listener)
        .unwrap()
        .with_max_sessions(1);
    (server, port)
}

#[tokio::test]
async fn h264_published_over_tcp_interleaved_reaches_the_record_server_intact() {
    let (server, port) = one_shot_server();
    let mut sink = RtspClientSink::new(1, format!("rtsp://127.0.0.1:{port}/stream"));
    let (received, published) = ingest_one_publisher(&mut sink, server).await;
    published.expect("publish and teardown");
    assert_eq!(
        received,
        video_frames(),
        "every access unit depayloads back to the bytes published"
    );
}

#[tokio::test]
async fn h264_published_over_udp_reaches_the_record_server_intact() {
    let (server, port) = one_shot_server();
    let mut sink = RtspClientSink::new(1, format!("rtsp://127.0.0.1:{port}/stream"))
        .with_protocols(vec![LowerTransport::Udp]);
    let (received, published) = ingest_one_publisher(&mut sink, server).await;
    published.expect("publish and teardown");
    assert_eq!(received, video_frames());
}

#[derive(Default)]
struct RecordedPublish {
    announced_sdp: String,
    // RTP packets by interleaved channel, in arrival order
    channels: BTreeMap<u8, Vec<Vec<u8>>>,
    requests: Vec<String>,
    torn_down: bool,
}

struct DigestGate {
    user: &'static str,
    password: &'static str,
}

fn md5_hex(text: &str) -> String {
    hex_encode(&Md5::digest(text.as_bytes()))
}

impl DigestGate {
    fn challenge(&self) -> String {
        format!("Digest realm=\"{DIGEST_REALM}\", nonce=\"{DIGEST_NONCE}\"")
    }

    // RFC 2617 §3.2.2 as the server checks it
    fn authorizes(&self, method: &str, uri: &str, request_head: &str) -> bool {
        let Some(header) = request_head
            .lines()
            .find_map(|line| line.strip_prefix("Authorization: Digest "))
        else {
            return false;
        };
        let user_hash = md5_hex(&format!("{}:{DIGEST_REALM}:{}", self.user, self.password));
        let request_hash = md5_hex(&format!("{method}:{uri}"));
        let expected = md5_hex(&format!("{user_hash}:{DIGEST_NONCE}:{request_hash}"));
        header.contains(&format!("response=\"{expected}\""))
            && header.contains(&format!("uri=\"{uri}\""))
    }
}

// one publisher against g2g's RTSP responder, recording what it announces and sends
// with `drop_from_timestamp`, the connection closes at the first RTP packet stamped at or after it
async fn record_one_publisher(
    listener: &tokio::net::TcpListener,
    gate: Option<DigestGate>,
    mut responder: RtspResponder,
    drop_from_timestamp: Option<u32>,
) -> RecordedPublish {
    let (mut conn, _) = listener.accept().await.expect("publisher connects");
    let mut recorded = RecordedPublish::default();
    let mut pending = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        loop {
            if pending.first() == Some(&INTERLEAVED_MARKER) {
                let Some(frame) = interleaved_frame(&pending) else {
                    break;
                };
                let (channel, packet, consumed) =
                    (frame.channel, frame.packet.to_vec(), frame.consumed);
                let is_rtp = channel % 2 == 0;
                if is_rtp && drop_from_timestamp.is_some_and(|ts| rtp_timestamp(&packet) >= ts) {
                    return recorded;
                }
                recorded.channels.entry(channel).or_default().push(packet);
                pending.drain(..consumed);
                continue;
            }
            let Some((request, consumed)) = RtspRequest::parse(&pending) else {
                break;
            };
            let head = String::from_utf8_lossy(&pending[..consumed]).to_string();
            pending.drain(..consumed);
            recorded.requests.push(head.clone());
            if let Some(gate) = &gate {
                if !gate.authorizes(&request.method, &request.uri, &head) {
                    let refusal = format!(
                        "RTSP/1.0 401 Unauthorized\r\nCSeq: {}\r\nWWW-Authenticate: {}\r\n\r\n",
                        request.cseq,
                        gate.challenge()
                    );
                    conn.write_all(refusal.as_bytes()).await.unwrap();
                    continue;
                }
            }
            if request.method == "ANNOUNCE" {
                recorded.announced_sdp = String::from_utf8(request.body.clone()).unwrap();
            }
            let (response, event) = responder.handle_request(&request);
            conn.write_all(&response).await.unwrap();
            if event == RtspEvent::Teardown {
                recorded.torn_down = true;
                return recorded;
            }
        }
        match conn.read(&mut buf).await {
            Ok(0) | Err(_) => return recorded,
            Ok(n) => pending.extend_from_slice(&buf[..n]),
        }
    }
}

async fn recorder_listener() -> (tokio::net::TcpListener, String) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, format!("rtsp://127.0.0.1:{port}/av"))
}

fn recorder_responder() -> RtspResponder {
    RtspResponder::new("", 0, 0)
}

async fn publish_and_record(
    listener: tokio::net::TcpListener,
    gate: Option<DigestGate>,
    responder: RtspResponder,
    sink: &mut RtspClientSink,
    pads: &[PadFrames],
    pace: Duration,
) -> (RecordedPublish, Result<(), G2gError>) {
    let recorder = async {
        tokio::time::timeout(
            SERVER_DEADLINE,
            record_one_publisher(&listener, gate, responder, None),
        )
        .await
        .expect("recorder finishes")
    };
    let send = async {
        publish_frames(sink, pads, pace).await?;
        end_streams(sink, pads.len()).await
    };
    tokio::join!(recorder, send)
}

async fn publish_to_recorder(
    pads: &[PadFrames],
    gate: Option<DigestGate>,
    credentials: Option<(&str, &str)>,
) -> (RecordedPublish, Result<(), G2gError>) {
    let (listener, location) = recorder_listener().await;
    let mut sink = RtspClientSink::new(pads.len(), location);
    if let Some((user, password)) = credentials {
        sink = sink.with_credentials(user, password);
    }
    publish_and_record(
        listener,
        gate,
        recorder_responder(),
        &mut sink,
        pads,
        Duration::ZERO,
    )
    .await
}

fn rtp_timestamp(packet: &[u8]) -> u32 {
    u32::from_be_bytes([packet[4], packet[5], packet[6], packet[7]])
}

fn interleaved_rtp_channel(pad: usize) -> u8 {
    (pad * 2) as u8
}

fn interleaved_rtcp_channel(pad: usize) -> u8 {
    interleaved_rtp_channel(pad) + 1
}

fn rtp_ssrc(packet: &[u8]) -> u32 {
    u32::from_be_bytes([packet[8], packet[9], packet[10], packet[11]])
}

// RFC 3550 §6.4.1: the sender's packet and octet counts follow the SSRC, NTP and RTP timestamps
const SR_PACKET_COUNT_OFFSET: usize = 20;
const SR_OCTET_COUNT_OFFSET: usize = 24;

fn word_at(packet: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(packet[offset..offset + 4].try_into().unwrap())
}

#[derive(Debug)]
struct SenderReport {
    ssrc: u32,
    ntp: u64,
    rtp_ts: u32,
    packet_count: u32,
    octet_count: u32,
}

// a compound report as RFC 3550 §6.1 requires it: an SR first, then the SDES carrying the CNAME
fn sender_report(compound: &[u8]) -> SenderReport {
    let packets = rtcp::parse_compound(compound);
    let [RtcpPacket::SenderReport {
        ssrc, ntp, rtp_ts, ..
    }, RtcpPacket::Other { pt: rtcp::PT_SDES }] = packets.as_slice()
    else {
        panic!("SR then SDES: {packets:?}");
    };
    SenderReport {
        ssrc: *ssrc,
        ntp: *ntp,
        rtp_ts: *rtp_ts,
        packet_count: word_at(compound, SR_PACKET_COUNT_OFFSET),
        octet_count: word_at(compound, SR_OCTET_COUNT_OFFSET),
    }
}

// the wall-clock time a report maps RTP timestamp zero to
fn rtp_zero_unix_ns(report: &SenderReport, clock_rate: u32) -> i64 {
    let unix_ns = unix_nanos_from_ntp(report.ntp).expect("NTP in range");
    unix_ns - (u64::from(report.rtp_ts) * NS_PER_SECOND / u64::from(clock_rate)) as i64
}

#[tokio::test]
async fn audio_and_video_pads_publish_as_one_announced_session() {
    let aac = aac_frames();
    let opus = opus_frames();
    let pads = [
        video_pad(),
        PadFrames {
            caps: audio_caps(AudioFormat::Aac),
            frames: aac
                .iter()
                .enumerate()
                .map(|(i, f)| (f.clone(), pts_at(i, AAC_FRAME_SAMPLES)))
                .collect(),
        },
        PadFrames {
            caps: audio_caps(AudioFormat::Opus),
            frames: opus
                .iter()
                .enumerate()
                .map(|(i, f)| (f.clone(), pts_at(i, OPUS_FRAME_SAMPLES)))
                .collect(),
        },
    ];
    let (recorded, published) = publish_to_recorder(&pads, None, None).await;
    published.expect("publish and teardown");
    assert!(recorded.torn_down, "EOS on every pad sends TEARDOWN");

    let media = SdpMedia::parse_all(&recorded.announced_sdp);
    let caps: Vec<&Caps> = media.iter().map(|m| &m.caps).collect();
    assert!(
        matches!(
            caps.as_slice(),
            [
                Caps::CompressedVideo {
                    codec: VideoCodec::H264,
                    ..
                },
                Caps::Audio {
                    format: AudioFormat::Aac,
                    channels: AUDIO_CHANNELS,
                    sample_rate: AUDIO_RATE_HZ,
                    ..
                },
                Caps::Audio {
                    format: AudioFormat::Opus,
                    channels: AUDIO_CHANNELS,
                    sample_rate: AUDIO_RATE_HZ,
                    ..
                },
            ]
        ),
        "one media section per pad in pad order: {}",
        recorded.announced_sdp
    );
    assert_eq!(
        media[VIDEO_PAD].parameter_sets,
        annex_b(&[SPS, PPS]),
        "the H.264 parameter sets of the first frame ride the fmtp"
    );
    let aac_fmtp = scan_sections(&recorded.announced_sdp)[AAC_PAD]
        .fmtp
        .clone()
        .expect("AAC fmtp");
    assert!(
        aac_fmtp.contains(&format!(
            "config={}",
            hex_encode(&expected_audio_specific_config())
        )),
        "the AudioSpecificConfig the ADTS header implies: {aac_fmtp}"
    );

    let mut depayloader = RtpH264Depayloader::new();
    let video: Vec<Vec<u8>> = recorded.channels[&interleaved_rtp_channel(VIDEO_PAD)]
        .iter()
        .filter_map(|packet| depayloader.depacketize(packet))
        .map(|access_unit| access_unit.data)
        .collect();
    assert_eq!(video, video_frames());

    let aac_packets = &recorded.channels[&interleaved_rtp_channel(AAC_PAD)];
    let aac_payloads: Vec<&[u8]> = aac_packets
        .iter()
        .map(|p| &p[RTP_HEADER_LEN + AAC_AU_HEADER_SECTION_LEN..])
        .collect();
    let raw_aac: Vec<&[u8]> = aac.iter().map(|f| &f[ADTS_HEADER_LEN..]).collect();
    assert_eq!(
        aac_payloads, raw_aac,
        "AAC goes out without its ADTS header"
    );
    assert!(
        aac_packets
            .windows(2)
            .all(|w| rtp_timestamp(&w[1]).wrapping_sub(rtp_timestamp(&w[0]))
                == AAC_FRAME_SAMPLES as u32),
        "AAC RTP timestamps tick at the sample rate"
    );

    let opus_packets = &recorded.channels[&interleaved_rtp_channel(OPUS_PAD)];
    let opus_payloads: Vec<&[u8]> = opus_packets.iter().map(|p| &p[RTP_HEADER_LEN..]).collect();
    assert_eq!(
        opus_payloads,
        opus.iter().map(Vec::as_slice).collect::<Vec<_>>()
    );
    assert!(
        opus_packets
            .windows(2)
            .all(|w| rtp_timestamp(&w[1]).wrapping_sub(rtp_timestamp(&w[0]))
                == OPUS_FRAME_SAMPLES as u32),
        "Opus RTP timestamps tick at 48 kHz"
    );
}

#[tokio::test]
async fn sender_reports_count_the_media_sent_before_them() {
    let (listener, location) = recorder_listener().await;
    let mut sink = RtspClientSink::new(1, location);
    sink.set_property("rtcp-sr-interval", PropValue::Uint(MIN_REPORT_INTERVAL_MS))
        .expect("interval");
    let pads = [video_pad()];
    let pace = Duration::from_millis(MIN_REPORT_INTERVAL_MS * 2);
    let (recorded, published) =
        publish_and_record(listener, None, recorder_responder(), &mut sink, &pads, pace).await;
    published.expect("publish and teardown");

    let media = &recorded.channels[&interleaved_rtp_channel(VIDEO_PAD)];
    let reports: Vec<SenderReport> = recorded.channels[&interleaved_rtcp_channel(VIDEO_PAD)]
        .iter()
        .map(|compound| sender_report(compound))
        .collect();
    assert!(
        reports.len() > 1,
        "a report every interval while frames flow: {reports:?}"
    );
    for report in &reports {
        assert_eq!(report.ssrc, rtp_ssrc(&media[0]), "the media SSRC reports");
        let counted: Vec<&Vec<u8>> = media
            .iter()
            .filter(|p| rtp_timestamp(p) <= report.rtp_ts)
            .collect();
        assert_eq!(report.packet_count as usize, counted.len());
        assert_eq!(
            report.octet_count as usize,
            counted
                .iter()
                .map(|p| p.len() - RTP_HEADER_LEN)
                .sum::<usize>(),
            "payload octets, without the RTP header"
        );
    }
}

#[tokio::test]
async fn audio_and_video_sender_reports_share_one_wall_clock() {
    let aac = aac_frames();
    let pads = [
        video_pad(),
        PadFrames {
            caps: audio_caps(AudioFormat::Aac),
            frames: aac
                .iter()
                .enumerate()
                .map(|(i, f)| (f.clone(), pts_at(i, AAC_FRAME_SAMPLES)))
                .collect(),
        },
    ];
    let (listener, location) = recorder_listener().await;
    let mut sink = RtspClientSink::new(pads.len(), location);
    sink.set_property("rtcp-sr-interval", PropValue::Uint(MIN_REPORT_INTERVAL_MS))
        .expect("interval");
    let pace = Duration::from_millis(MIN_REPORT_INTERVAL_MS * 2);
    let (recorded, published) =
        publish_and_record(listener, None, recorder_responder(), &mut sink, &pads, pace).await;
    published.expect("publish and teardown");
    let rtp_zeros = |pad: usize, clock_rate: u32| -> Vec<i64> {
        recorded.channels[&interleaved_rtcp_channel(pad)]
            .iter()
            .map(|compound| rtp_zero_unix_ns(&sender_report(compound), clock_rate))
            .collect()
    };
    let mut zeros = rtp_zeros(VIDEO_PAD, MediaClock::VIDEO_RATE_HZ);
    zeros.extend(rtp_zeros(AAC_PAD, AUDIO_RATE_HZ));
    assert!(zeros.len() > pads.len(), "several reports per stream");
    let one_audio_tick_ns = (NS_PER_SECOND / u64::from(AUDIO_RATE_HZ)) as i64;
    assert!(
        zeros
            .iter()
            .all(|zero| (zero - zeros[0]).abs() <= one_audio_tick_ns),
        "every report of both streams puts PTS zero at one wall-clock instant: {zeros:?}"
    );
}

// the recorder's session timeout gives a one-second keepalive interval
const SHORT_SESSION_TIMEOUT_S: u32 = 2;
// two keepalive intervals pass within the eight frames, with a margin either side
const KEEPALIVE_TEST_PACE: Duration = Duration::from_millis(400);
const KEEPALIVES_WITHIN_TEST: usize = 2;

async fn keepalive_requests(keep_alive: bool) -> Vec<String> {
    let (listener, location) = recorder_listener().await;
    let mut sink = RtspClientSink::new(1, location);
    sink.set_property("do-rtsp-keep-alive", PropValue::Bool(keep_alive))
        .expect("keep-alive");
    let responder = recorder_responder().with_session_timeout_secs(SHORT_SESSION_TIMEOUT_S);
    let pads = [video_pad()];
    let (recorded, published) = publish_and_record(
        listener,
        None,
        responder,
        &mut sink,
        &pads,
        KEEPALIVE_TEST_PACE,
    )
    .await;
    published.expect("publish and teardown");
    recorded
        .requests
        .into_iter()
        .filter(|head| head.starts_with("GET_PARAMETER "))
        .collect()
}

#[tokio::test]
async fn keepalives_go_out_ahead_of_the_session_timeout() {
    let keepalives = keepalive_requests(true).await;
    assert!(
        keepalives.len() >= KEEPALIVES_WITHIN_TEST,
        "a GET_PARAMETER every keepalive interval: {keepalives:?}"
    );
    assert!(
        keepalives.iter().all(|head| head.contains("Session: ")),
        "each keepalive names the session"
    );
}

// upstream sends one frame and then nothing for this long, several keepalive and report intervals
const STALL: Duration = Duration::from_secs(5);
const STALL_REPORT_INTERVAL_MS: u64 = 1_000;
// a randomized interval is at most one and a half times nominal, so the stall holds at least this many more
const REPORTS_WITHIN_STALL: usize = 3;

#[tokio::test]
async fn a_stalled_upstream_still_gets_keepalives_and_sender_reports() {
    let (listener, location) = recorder_listener().await;
    let mut sink = RtspClientSink::new(1, location);
    sink.set_property(
        "rtcp-sr-interval",
        PropValue::Uint(STALL_REPORT_INTERVAL_MS),
    )
    .expect("interval");
    let responder = recorder_responder().with_session_timeout_secs(SHORT_SESSION_TIMEOUT_S);
    let recorder = async {
        tokio::time::timeout(
            SERVER_DEADLINE,
            record_one_publisher(&listener, None, responder, None),
        )
        .await
        .expect("recorder finishes")
    };
    let send = async {
        let mut discard = Capture::default();
        sink.configure_pipeline(VIDEO_PAD, &h264_caps())?;
        sink.process(VIDEO_PAD, data_frame(access_unit(0), 0), &mut discard)
            .await?;
        tokio::time::sleep(STALL).await;
        end_streams(&mut sink, 1).await
    };
    let (recorded, published) = tokio::join!(recorder, send);
    published.expect("publish and teardown");

    let keepalives = recorded
        .requests
        .iter()
        .filter(|head| head.starts_with("GET_PARAMETER "))
        .count();
    assert!(
        keepalives >= KEEPALIVES_WITHIN_TEST,
        "keepalives go out with no frames arriving: {keepalives}"
    );
    let reports: Vec<SenderReport> = recorded.channels[&interleaved_rtcp_channel(VIDEO_PAD)]
        .iter()
        .map(|compound| sender_report(compound))
        .collect();
    assert!(
        reports.len() > REPORTS_WITHIN_STALL,
        "reports keep going with no frames arriving: {reports:?}"
    );
    let media_packets = recorded.channels[&interleaved_rtp_channel(VIDEO_PAD)].len();
    for report in &reports {
        assert_eq!(report.packet_count as usize, media_packets);
        assert_eq!(
            report.rtp_ts, reports[0].rtp_ts,
            "the last frame's timestamp"
        );
        assert_eq!(
            report.ntp, reports[0].ntp,
            "the same point on the NTP mapping"
        );
    }
}

#[tokio::test]
async fn do_rtsp_keep_alive_false_sends_none() {
    assert!(keepalive_requests(false).await.is_empty());
}

const P_SLICE_NAL_HEADER: u8 = 0x41;
// the frame the server drops the connection at, and the next keyframe after it
const DROPPED_FRAME: usize = 2;
const KEYFRAME_AFTER_DROP: usize = 5;
const RECONNECT_TEST_PACE: Duration = Duration::from_millis(20);

// an IDR opens the stream and the frame after the drop, P slices fill the rest
fn gop_frames() -> Vec<Vec<u8>> {
    (0..ACCESS_UNITS)
        .map(|i| match i {
            0 | KEYFRAME_AFTER_DROP => access_unit(i),
            _ => annex_b(&[&[P_SLICE_NAL_HEADER, i as u8, 0x9A]]),
        })
        .collect()
}

fn depayloaded_video(recorded: &RecordedPublish) -> Vec<Vec<u8>> {
    let mut depayloader = RtpH264Depayloader::new();
    recorded
        .channels
        .get(&interleaved_rtp_channel(VIDEO_PAD))
        .into_iter()
        .flatten()
        .filter_map(|packet| depayloader.depacketize(packet))
        .map(|access_unit| access_unit.data)
        .collect()
}

struct DroppedConnectionPublish {
    listener: tokio::net::TcpListener,
    sink: RtspClientSink,
    pads: [PadFrames; 1],
}

async fn dropped_connection_publish(reconnect_attempts: u64) -> DroppedConnectionPublish {
    let (listener, location) = recorder_listener().await;
    let mut sink = RtspClientSink::new(1, location);
    sink.set_property("reconnect", PropValue::Uint(reconnect_attempts))
        .expect("reconnect");
    sink.set_property("reconnect-backoff", PropValue::Uint(1))
        .expect("backoff");
    let pads = [PadFrames {
        caps: h264_caps(),
        frames: gop_frames()
            .into_iter()
            .enumerate()
            .map(|(i, f)| (f, i as u64 * FRAME_PERIOD_NS))
            .collect(),
    }];
    DroppedConnectionPublish {
        listener,
        sink,
        pads,
    }
}

async fn record_until_the_drop(listener: &tokio::net::TcpListener) -> RecordedPublish {
    let drop_at = rtp_timestamp_of_frame(DROPPED_FRAME);
    record_one_publisher(listener, None, recorder_responder(), Some(drop_at)).await
}

fn rtp_timestamp_of_frame(index: usize) -> u32 {
    (index as u64 * FRAME_PERIOD_NS * u64::from(MediaClock::VIDEO_RATE_HZ) / NS_PER_SECOND) as u32
}

#[tokio::test]
async fn a_dropped_connection_republishes_from_the_next_keyframe() {
    let DroppedConnectionPublish {
        listener,
        mut sink,
        pads,
    } = dropped_connection_publish(1).await;
    let recorders = async {
        let first = record_until_the_drop(&listener).await;
        let second = record_one_publisher(&listener, None, recorder_responder(), None).await;
        (first, second)
    };
    let send = async {
        publish_frames(&mut sink, &pads, RECONNECT_TEST_PACE).await?;
        end_streams(&mut sink, pads.len()).await
    };
    let ((first, second), published) =
        tokio::time::timeout(SERVER_DEADLINE, async { tokio::join!(recorders, send) })
            .await
            .expect("both sessions finish");
    published.expect("the publish survives the drop");
    let frames = gop_frames();
    assert_eq!(depayloaded_video(&first), frames[..DROPPED_FRAME]);
    assert!(
        second
            .requests
            .iter()
            .any(|head| head.starts_with("ANNOUNCE ")),
        "a new session is announced"
    );
    assert_eq!(
        depayloaded_video(&second),
        frames[KEYFRAME_AFTER_DROP..],
        "the new session starts at a keyframe"
    );
    assert!(second.torn_down, "EOS tears the new session down");
}

#[tokio::test]
async fn without_reconnect_a_dropped_connection_fails_the_publish() {
    let DroppedConnectionPublish {
        listener,
        mut sink,
        pads,
    } = dropped_connection_publish(0).await;
    let send = async {
        publish_frames(&mut sink, &pads, RECONNECT_TEST_PACE).await?;
        end_streams(&mut sink, pads.len()).await
    };
    let (_, published) = tokio::join!(record_until_the_drop(&listener), send);
    assert!(published.is_err());
}

#[tokio::test]
async fn digest_challenge_is_answered_on_every_request() {
    let gate = DigestGate {
        user: USER,
        password: PASSWORD,
    };
    let (recorded, published) =
        publish_to_recorder(&[video_pad()], Some(gate), Some((USER, PASSWORD))).await;
    published.expect("publish and teardown");
    assert!(recorded.torn_down);
    assert!(
        !recorded.channels[&interleaved_rtp_channel(VIDEO_PAD)].is_empty(),
        "media flowed after the authorized RECORD"
    );
    let unanswered = recorded
        .requests
        .iter()
        .filter(|head| !head.contains("Authorization: Digest "))
        .count();
    assert_eq!(
        unanswered, 1,
        "only the first request goes out before the challenge"
    );
}

#[tokio::test]
async fn refused_credentials_fail_the_publish() {
    let gate = DigestGate {
        user: USER,
        password: PASSWORD,
    };
    let (recorded, published) =
        publish_to_recorder(&[video_pad()], Some(gate), Some((USER, "wrong"))).await;
    assert!(published.is_err(), "a second 401 fails the element");
    assert!(recorded.channels.is_empty(), "no media without RECORD");
}

#[test]
fn launch_line_properties_reach_the_element() {
    use g2g_core::runtime::parse_launch;
    use g2g_plugins::registry::default_registry;

    let registry = default_registry();
    let settings = [
        ("location", "rtsp://127.0.0.1:8554/live"),
        ("protocols", "udp+tcp"),
        ("user-id", USER),
        ("user-pw", PASSWORD),
        ("user-agent", "canary"),
        ("tcp-timeout", "5000000"),
        ("rtcp-sr-interval", "2500"),
        ("do-rtsp-keep-alive", "false"),
        ("reconnect", "3"),
        ("reconnect-backoff", "100"),
        ("reconnect-backoff-max", "2000"),
    ];
    let assignments: Vec<String> = settings.iter().map(|(k, v)| format!("{k}={v}")).collect();
    parse_launch(
        &registry,
        &format!(
            "rtspclientsink name=s {} fakesrc ! s.sink_0 fakesrc ! s.sink_1",
            assignments.join(" ")
        ),
    )
    .expect("two request pads and every property parse");

    let mut sink = registry
        .make_muxer("rtspclientsink", 2)
        .expect("registered as a fan-in");
    for (name, text) in settings {
        let spec = *sink
            .properties()
            .iter()
            .find(|s| s.name == name)
            .expect("declared");
        let value = spec.parse_value(text).expect("parses");
        sink.set_property(name, value.clone()).expect("applies");
        assert_eq!(sink.get_property(name), Some(value), "{name} round-trips");
    }
    assert_eq!(
        sink.get_property("protocols"),
        Some(PropValue::Flags(vec!["udp".to_string(), "tcp".to_string()])),
        "transports keep the order written"
    );
}
