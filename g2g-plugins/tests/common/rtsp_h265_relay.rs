// ffmpeg packetizes an H.265 fixture as RTP, this answers RTSP and relays each datagram
// interleaved, one client session at a time

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h265_640x480.h265"
);
const INTERLEAVED_RTP_CHANNEL: u8 = 0;
const MAX_DATAGRAM: usize = 65536;
// shorter than the fixed RTP header, so the client fails the session on it
const CORRUPT_RTP: [u8; 4] = [0; 4];

pub(crate) struct H265Relay {
    pub(crate) url: String,
    sessions: Arc<AtomicUsize>,
    ffmpeg: Child,
}

impl H265Relay {
    // RTSP connections the relay has accepted
    pub(crate) fn sessions(&self) -> usize {
        self.sessions.load(Ordering::SeqCst)
    }
}

impl Drop for H265Relay {
    fn drop(&mut self) {
        let _ = self.ffmpeg.kill();
        let _ = self.ffmpeg.wait();
    }
}

// the first `broken_sessions` sessions get a corrupt RTP packet right after PLAY instead of media
pub(crate) fn serve_h265_fixture(broken_sessions: usize) -> H265Relay {
    let rtp = UdpSocket::bind("127.0.0.1:0").expect("bind rtp");
    let rtp_port = rtp.local_addr().unwrap().port();
    let sdp_path = std::env::temp_dir().join(format!("h265-relay-{rtp_port}.sdp"));
    let ffmpeg = Command::new("ffmpeg")
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
        .expect("spawn ffmpeg");
    let sdp = loop {
        match std::fs::read_to_string(&sdp_path) {
            Ok(text) if text.contains("H265") => break text,
            _ => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    };
    let _ = std::fs::remove_file(&sdp_path);

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let port = listener.local_addr().unwrap().port();
    let sessions = Arc::new(AtomicUsize::new(0));
    let accepted = sessions.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { return };
            let index = accepted.fetch_add(1, Ordering::SeqCst);
            serve_client(stream, &sdp, &rtp, index < broken_sessions);
        }
    });
    H265Relay {
        url: format!("rtsp://127.0.0.1:{port}/stream"),
        sessions,
        ffmpeg,
    }
}

fn serve_client(stream: TcpStream, sdp: &str, rtp: &UdpSocket, broken: bool) {
    let mut write_half = stream.try_clone().expect("clone the control connection");
    let mut reader = BufReader::new(stream);
    loop {
        let Some((method, cseq)) = read_request(&mut reader) else {
            return;
        };
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
    if broken {
        let _ = write_half.write_all(&interleaved(&CORRUPT_RTP));
        return;
    }
    let mut datagram = vec![0u8; MAX_DATAGRAM];
    loop {
        let len = rtp.recv(&mut datagram).expect("receive ffmpeg's rtp");
        if write_half
            .write_all(&interleaved(&datagram[..len]))
            .is_err()
        {
            return;
        }
    }
}

fn interleaved(packet: &[u8]) -> Vec<u8> {
    let mut framed = vec![b'$', INTERLEAVED_RTP_CHANNEL];
    framed.extend_from_slice(&(packet.len() as u16).to_be_bytes());
    framed.extend_from_slice(packet);
    framed
}

// `None` once the client closes the connection
fn read_request(reader: &mut BufReader<TcpStream>) -> Option<(String, String)> {
    let mut method = String::new();
    let mut cseq = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end();
        if line.is_empty() && !method.is_empty() {
            return Some((method, cseq));
        }
        if method.is_empty() {
            method = line.split(' ').next().unwrap_or_default().to_string();
        } else if let Some(value) = line.strip_prefix("CSeq:") {
            cseq = value.trim().to_string();
        }
    }
}
