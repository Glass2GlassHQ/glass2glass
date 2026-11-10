// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "icecast")]

#[path = "common/hls_mux_fixture.rs"]
mod hls_mux_fixture;
use hls_mux_fixture::{frame, NullSink};

mod ffprobe;
use ffprobe::{fixture_path, Probe};

use std::process::Command;
use std::time::Duration;

use g2g_core::bus::{Bus, BusMessage};
use g2g_core::clock::MonotonicClock;
use g2g_core::element::AsyncElement;
use g2g_core::frame::{FrameTiming, PipelinePacket};
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, HardwareError, PipelineClock};
use g2g_plugins::icecastsink::IcecastSink;
use g2g_plugins::ogg::OggPageWriter;
use g2g_plugins::registry::default_registry;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const LOOPBACK: &str = "127.0.0.1";
const HEAD_END: &[u8] = b"\r\n\r\n";
const CONTINUE: &[u8] = b"HTTP/1.1 100 Continue\r\n\r\n";
const MP3_FIXTURE: &str = "mp3_stereo_44100_id3v2.mp3";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

async fn read_head(stream: &mut TcpStream) -> String {
    let mut head = Vec::new();
    while !head.ends_with(HEAD_END) {
        let mut byte = [0u8; 1];
        stream.read_exact(&mut byte).await.expect("a request head");
        head.push(byte[0]);
    }
    String::from_utf8(head).expect("an ascii head")
}

async fn accept_source(listener: &TcpListener) -> (TcpStream, String) {
    let (mut stream, _) = listener.accept().await.expect("the sink connects");
    let head = read_head(&mut stream).await;
    stream.write_all(CONTINUE).await.expect("answer 100");
    (stream, head)
}

fn head_lines(head: &str) -> Vec<&str> {
    head.split("\r\n").filter(|line| !line.is_empty()).collect()
}

#[tokio::test]
async fn bare_uri_streams_mp3_with_the_source_headers() {
    let probe = Probe::read(MP3_FIXTURE);
    let listener = TcpListener::bind((LOOPBACK, 0)).await.expect("bind");
    let port = listener.local_addr().expect("address").port();
    let server = tokio::spawn(async move {
        let (mut stream, head) = accept_source(&listener).await;
        let mut body = Vec::new();
        stream.read_to_end(&mut body).await.expect("the body");
        (head, body)
    });
    let line = format!(
        "filesrc location={} ! typefind ! mpegaudioparse ! \
         ice+http://source:hackme@{LOOPBACK}:{port}/radio.mp3 public=false stream-name=g2g",
        fixture_path(MP3_FIXTURE).display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");
    run_graph(graph, &ZeroClock, 4)
        .await
        .expect("the pipeline runs");
    let (head, body) = server.await.expect("the server task");

    let audio_info = format!(
        "Ice-audio-info: channels={};samplerate={}",
        probe.channels, probe.sample_rate
    );
    let lines = head_lines(&head);
    assert_eq!(lines[0], "PUT /radio.mp3 HTTP/1.1");
    for expected in [
        format!("Host: {LOOPBACK}:{port}").as_str(),
        // base64 of "source:hackme"
        "Authorization: Basic c291cmNlOmhhY2ttZQ==",
        "Content-Type: audio/mpeg",
        audio_info.as_str(),
        "Ice-public: 0",
        "Ice-name: g2g",
        "Expect: 100-continue",
    ] {
        assert!(lines.contains(&expected), "{expected:?} in {lines:?}");
    }
    let fixture = std::fs::read(fixture_path(MP3_FIXTURE)).expect("the fixture");
    assert!(!body.is_empty(), "the sink sends audio");
    assert!(
        fixture.windows(body.len()).any(|window| window == body),
        "the body is the fixture's audio frames, {} bytes",
        body.len()
    );
}

#[tokio::test]
async fn refused_credentials_fail_with_the_http_status() {
    const UNAUTHORIZED: u16 = 401;
    let listener = TcpListener::bind((LOOPBACK, 0)).await.expect("bind");
    let port = listener.local_addr().expect("address").port();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("the sink connects");
        read_head(&mut stream).await;
        let refusal = format!("HTTP/1.1 {UNAUTHORIZED} Unauthorized\r\n\r\n");
        stream.write_all(refusal.as_bytes()).await.expect("refuse");
    });
    let line = format!(
        "filesrc location={} ! typefind ! mpegaudioparse ! \
         icecastsink location=ice+http://source:wrong@{LOOPBACK}:{port}/radio.mp3",
        fixture_path(MP3_FIXTURE).display()
    );
    let graph = parse_launch(&default_registry(), &line).expect("the line parses");
    let error = run_graph(graph, &ZeroClock, 4)
        .await
        .expect_err("the server refuses the source");
    server.await.expect("the server task");
    assert_eq!(error, G2gError::Hardware(HardwareError::Http(UNAUTHORIZED)));
}

const OPUS_CHANNELS: u8 = 2;
const OPUS_INPUT_RATE: u32 = 48_000;
const OPUS_FRAME_SAMPLES: u64 = 960;
const OGG_SERIAL: u32 = 7;

fn opus_head() -> Vec<u8> {
    let mut head = b"OpusHead".to_vec();
    head.push(1);
    head.push(OPUS_CHANNELS);
    head.extend_from_slice(&0u16.to_le_bytes());
    head.extend_from_slice(&OPUS_INPUT_RATE.to_le_bytes());
    head.extend_from_slice(&[0, 0, 0]);
    head
}

#[tokio::test]
async fn reconnect_resends_the_ogg_header_pages() {
    let mut writer = OggPageWriter::new(OGG_SERIAL);
    writer.push_packet(opus_head(), 0);
    let mut headers = writer.flush(false);
    writer.push_packet(b"OpusTags\0\0\0\0\0\0\0\0".to_vec(), 0);
    headers.extend(writer.flush(false));
    let mut audio_page = |index: u64| {
        writer.push_packet(audio_packet(index), index * OPUS_FRAME_SAMPLES);
        writer.flush(false)
    };

    let listener = TcpListener::bind((LOOPBACK, 0)).await.expect("bind");
    let port = listener.local_addr().expect("address").port();
    let (reconnected_tx, mut reconnected) = tokio::sync::oneshot::channel();
    let first_len = headers.len();
    let server = tokio::spawn(async move {
        let (mut first, first_head) = accept_source(&listener).await;
        let mut first_body = vec![0u8; first_len];
        first
            .read_exact(&mut first_body)
            .await
            .expect("the header pages");
        drop(first);
        let (mut second, second_head) = accept_source(&listener).await;
        reconnected_tx.send(()).expect("the test waits");
        let mut second_body = Vec::new();
        second
            .read_to_end(&mut second_body)
            .await
            .expect("the body");
        (first_head, first_body, second_head, second_body)
    });

    let (bus, handle) = Bus::new(16);
    let mut sink = IcecastSink::new(&format!("ice+http://{LOOPBACK}:{port}/live.ogg"))
        .expect("a valid location");
    sink.set_bus(handle);
    sink.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::Ogg,
    })
    .expect("Ogg is accepted");
    let mut out = NullSink;
    let send = |bytes: Vec<u8>| frame(bytes, FrameTiming::default());
    sink.process(send(headers.clone()), &mut out)
        .await
        .expect("the header pages are held");
    let mut index = 1;
    sink.process(send(audio_page(index)), &mut out)
        .await
        .expect("the first audio page sends");
    const MAX_PAGES: u64 = 200;
    while reconnected.try_recv().is_err() {
        index += 1;
        assert!(index < MAX_PAGES, "the sink never reconnected");
        sink.process(send(audio_page(index)), &mut out)
            .await
            .expect("a broken connection is reconnected, not fatal");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let last_page = audio_page(index + 1);
    sink.process(send(last_page.clone()), &mut out)
        .await
        .expect("the page after the reconnect sends");
    sink.process(PipelinePacket::Eos, &mut out)
        .await
        .expect("eos closes the connection");
    let (first_head, first_body, second_head, second_body) = server.await.expect("the server task");

    let audio_info =
        format!("Ice-audio-info: channels={OPUS_CHANNELS};samplerate={OPUS_INPUT_RATE}");
    for head in [&first_head, &second_head] {
        assert!(
            head_lines(head).contains(&"Content-Type: audio/ogg"),
            "{head}"
        );
        assert!(head_lines(head).contains(&audio_info.as_str()), "{head}");
    }
    assert_eq!(first_body, headers);
    assert!(
        second_body.starts_with(&headers),
        "the headers lead the new connection"
    );
    assert!(second_body.ends_with(&last_page));
    assert!(
        matches!(
            bus.try_recv(),
            Some(BusMessage::Warning(G2gError::Network(_)))
        ),
        "the broken connection is posted as a warning"
    );
}

fn audio_packet(index: u64) -> Vec<u8> {
    const PACKET_BYTES: usize = 160;
    vec![index as u8; PACKET_BYTES]
}

const REAL_SERVER_VARIABLE: &str = "G2G_ICECAST_SOURCE_URL";
const STREAM_SECONDS: &str = "2";
// the mount exists once the pipeline has connected, well before its stream ends
const LISTENER_DELAY: Duration = Duration::from_millis(500);

struct RealFormat {
    extension: &'static str,
    ffmpeg_encoder: &'static str,
    ffmpeg_codec: &'static str,
    chain: &'static str,
}

const REAL_FORMATS: &[RealFormat] = &[
    RealFormat {
        extension: "mp3",
        ffmpeg_encoder: "libmp3lame",
        ffmpeg_codec: "mp3",
        chain: "typefind ! mpegaudioparse ! clocksync",
    },
    RealFormat {
        extension: "aac",
        ffmpeg_encoder: "aac",
        ffmpeg_codec: "aac",
        chain: "typefind ! aacparse ! clocksync",
    },
    RealFormat {
        extension: "flac",
        ffmpeg_encoder: "flac",
        ffmpeg_codec: "flac",
        chain: "typefind ! flacparse ! clocksync",
    },
    RealFormat {
        extension: "ogg",
        ffmpeg_encoder: "libopus",
        ffmpeg_codec: "opus",
        chain: "oggdemux ! clocksync ! oggmux",
    },
];

fn encode_tone(format: &RealFormat, dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join(format!("tone.{}", format.extension));
    let status = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
        .arg(format!("sine=frequency=440:duration={STREAM_SECONDS}"))
        .args(["-ac", "2", "-c:a", format.ffmpeg_encoder])
        .arg(&path)
        .status()
        .expect("ffmpeg runs");
    assert!(status.success(), "ffmpeg encodes {}", format.extension);
    path
}

#[tokio::test]
#[ignore]
async fn real_icecast_relays_every_format() {
    let source_url = std::env::var(REAL_SERVER_VARIABLE)
        .unwrap_or_else(|_| panic!("{REAL_SERVER_VARIABLE} names the server"));
    let listener_base = {
        let without_scheme = source_url.split_once("://").expect("a URL").1;
        let authority = without_scheme
            .rsplit_once('@')
            .map_or(without_scheme, |(_, a)| a);
        format!("http://{}", authority.trim_end_matches('/'))
    };
    let dir = std::env::temp_dir().join(format!("g2g-m2136-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("work dir");
    for format in REAL_FORMATS {
        let input = encode_tone(format, &dir);
        let mount = format!("/g2g-{}", format.extension);
        let line = format!(
            "filesrc location={} ! {} ! {}{mount}",
            input.display(),
            format.chain,
            source_url.trim_end_matches('/')
        );
        let graph = parse_launch(&default_registry(), &line).expect("the line parses");
        let listener_url = format!("{listener_base}{mount}");
        let listener = std::thread::spawn(move || {
            std::thread::sleep(LISTENER_DELAY);
            Command::new("ffprobe")
                .args(["-v", "error", "-count_packets", "-show_entries"])
                .arg("stream=codec_name,nb_read_packets")
                .args(["-of", "csv=p=0"])
                .arg(listener_url)
                .output()
                .expect("ffprobe runs")
        });
        run_graph(graph, &MonotonicClock, 4)
            .await
            .unwrap_or_else(|error| panic!("{} streams: {error:?}", format.extension));
        let probe = listener.join().expect("the listener thread");
        let report = String::from_utf8_lossy(&probe.stdout);
        let (codec, packets) = report
            .trim()
            .split_once(',')
            .unwrap_or_else(|| panic!("ffprobe reads {mount}: {report:?} {:?}", probe.stderr));
        assert_eq!(codec, format.ffmpeg_codec, "{mount}");
        let packets: u64 = packets.parse().expect("a packet count");
        assert!(packets > 0, "{mount} relays audio");
    }
}
