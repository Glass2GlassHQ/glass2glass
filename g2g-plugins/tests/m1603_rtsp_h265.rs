#![cfg(all(
    feature = "std",
    feature = "rtsp",
    feature = "ffmpeg",
    target_os = "linux"
))]

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::process::{Child, Command, Stdio};

use g2g_core::runtime::{parse_launch, run_graph, LatencyProfile};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
use ffmpeg_tool::have_ffmpeg;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h265_640x480.h265"
);
const DECODED_FRAMES: u64 = 10;
const INTERLEAVED_RTP_CHANNEL: u8 = 0;
const MAX_DATAGRAM: usize = 65536;

struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

// ffmpeg packetizes the RTP, this only answers RTSP and relays each datagram interleaved
fn serve_one_client(listener: TcpListener, sdp: String, rtp: UdpSocket) {
    let (stream, _) = listener.accept().expect("accept the client");
    let mut write_half = stream.try_clone().expect("clone the control connection");
    let mut reader = BufReader::new(stream);
    loop {
        let (method, cseq) = read_request(&mut reader);
        let response = match method.as_str() {
            "OPTIONS" => format!(
                "RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nPublic: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n\r\n"
            ),
            "DESCRIBE" => format!(
                "RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{sdp}",
                sdp.len()
            ),
            "SETUP" => format!(
                "RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 1\r\nTransport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n"
            ),
            "PLAY" => format!("RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 1\r\n\r\n"),
            other => panic!("unexpected RTSP request {other}"),
        };
        write_half.write_all(response.as_bytes()).expect("respond");
        if method == "PLAY" {
            break;
        }
    }
    let mut datagram = vec![0u8; MAX_DATAGRAM];
    loop {
        let len = rtp.recv(&mut datagram).expect("receive ffmpeg's rtp");
        let mut framed = vec![b'$', INTERLEAVED_RTP_CHANNEL];
        framed.extend_from_slice(&(len as u16).to_be_bytes());
        framed.extend_from_slice(&datagram[..len]);
        if write_half.write_all(&framed).is_err() {
            return;
        }
    }
}

fn read_request(reader: &mut BufReader<TcpStream>) -> (String, String) {
    let mut method = String::new();
    let mut cseq = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).expect("read a request line");
        let line = line.trim_end();
        if line.is_empty() && !method.is_empty() {
            return (method, cseq);
        }
        if method.is_empty() {
            method = line.split(' ').next().unwrap_or_default().to_string();
        } else if let Some(value) = line.strip_prefix("CSeq:") {
            cseq = value.trim().to_string();
        }
    }
}

#[tokio::test]
async fn rtspsrc_decodes_an_h265_track_packetized_by_ffmpeg() {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the reference-peer RTSP H.265 decode");
        return;
    }
    let rtp = UdpSocket::bind("127.0.0.1:0").expect("bind rtp");
    let rtp_port = rtp.local_addr().unwrap().port();
    let sdp_path = std::env::temp_dir().join(format!("m1603-{rtp_port}.sdp"));
    let _ffmpeg = KillOnDrop(
        Command::new("ffmpeg")
            .args([
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-stream_loop",
                "-1",
                "-i",
                FIXTURE,
            ])
            .args(["-c", "copy", "-f", "rtp", "-sdp_file"])
            .arg(&sdp_path)
            .arg(format!("rtp://127.0.0.1:{rtp_port}"))
            .stdin(Stdio::null())
            .spawn()
            .expect("spawn ffmpeg"),
    );
    let sdp = loop {
        match std::fs::read_to_string(&sdp_path) {
            Ok(text) if text.contains("H265") => break text,
            _ => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    };
    let _ = std::fs::remove_file(&sdp_path);

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || serve_one_client(listener, sdp, rtp));

    let line = format!(
        "rtspsrc location=rtsp://127.0.0.1:{port}/stream num-buffers={DECODED_FRAMES} \
         ! h265parse ! avdec_h265 ! fakesink"
    );
    let graph =
        parse_launch(&default_registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    let stats = run_graph(graph, &WallClock::new(), LatencyProfile::Throughput)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    assert_eq!(
        stats.frames_emitted, DECODED_FRAMES,
        "rtspsrc delivers every frame"
    );
    assert!(
        stats.frames_consumed > 0,
        "decoded H.265 frames reach the sink"
    );
}
