// ffmpeg packetizes a fixture's video and audio as RTP, this answers RTSP and relays each
// track interleaved on its own channel pair, so a test plays an A/V stream from a real packetizer

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

const MAX_DATAGRAM: usize = 65536;
const SDP_DEADLINE: Duration = Duration::from_secs(15);
// ffmpeg prints each SDP it has so far under this header once a muxer starts
const SDP_HEADER: &str = "SDP:\n";
// ffmpeg writes the media sections in the order of its outputs: video, then audio
const TRACK_MEDIA: [&str; 2] = ["m=video", "m=audio"];
const TRACK_CONTROL: &str = "trackID=";

pub(crate) struct AvRelay {
    pub(crate) url: String,
    ffmpeg: Child,
}

impl Drop for AvRelay {
    fn drop(&mut self) {
        let _ = self.ffmpeg.kill();
        let _ = self.ffmpeg.wait();
    }
}

pub(crate) fn serve_av_fixture(fixture: &str) -> AvRelay {
    let sockets: Vec<Arc<UdpSocket>> = TRACK_MEDIA
        .iter()
        .map(|_| Arc::new(UdpSocket::bind("127.0.0.1:0").expect("bind rtp")))
        .collect();
    let ports: Vec<u16> = sockets
        .iter()
        .map(|socket| socket.local_addr().unwrap().port())
        .collect();
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-stream_loop",
            "-1",
        ])
        .arg("-i")
        .arg(fixture);
    for (map, port) in ["0:v", "0:a"].iter().zip(&ports) {
        command
            .args(["-map", map, "-c", "copy", "-f", "rtp"])
            .arg(format!("rtp://127.0.0.1:{port}"));
    }
    let mut ffmpeg = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn ffmpeg");
    let stdout = ffmpeg.stdout.take().expect("ffmpeg's stdout is piped");
    let sdp = with_track_controls(&read_full_sdp(stdout));

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind rtsp");
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { return };
            let (sdp, sockets) = (sdp.clone(), sockets.clone());
            std::thread::spawn(move || serve_client(stream, &sdp, &sockets));
        }
    });
    AvRelay {
        url: format!("rtsp://127.0.0.1:{port}/stream"),
        ffmpeg,
    }
}

// ffmpeg 6 prints a video-only SDP as the first muxer starts, the last print lists both tracks
fn read_full_sdp(mut stdout: ChildStdout) -> String {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let mut text = String::new();
        let mut chunk = [0u8; 4096];
        let mut sent = false;
        // the pipe stays open until ffmpeg exits so a later print cannot kill it with SIGPIPE
        while let Ok(len @ 1..) = stdout.read(&mut chunk) {
            if sent {
                continue;
            }
            text.push_str(&String::from_utf8_lossy(&chunk[..len]));
            let Some(start) = text.rfind(SDP_HEADER) else {
                continue;
            };
            let sdp = text[start + SDP_HEADER.len()..].trim_end();
            if TRACK_MEDIA.iter().all(|media| sdp.contains(media)) {
                let _ = sender.send(sdp.to_string());
                sent = true;
            }
        }
    });
    receiver
        .recv_timeout(SDP_DEADLINE)
        .expect("ffmpeg prints an SDP with both tracks")
}

// ffmpeg's SDP names no control URL, and two tracks need one each to be set up apart
fn with_track_controls(sdp: &str) -> String {
    let mut out = String::new();
    let mut track = 0;
    for line in sdp.lines() {
        out.push_str(line);
        out.push_str("\r\n");
        if line.starts_with("m=") {
            out.push_str(&format!("a=control:{TRACK_CONTROL}{track}\r\n"));
            track += 1;
        }
    }
    out
}

fn serve_client(stream: TcpStream, sdp: &str, sockets: &[Arc<UdpSocket>]) {
    let writer = Arc::new(Mutex::new(
        stream.try_clone().expect("clone the connection"),
    ));
    let mut reader = BufReader::new(stream);
    while let Some(request) = read_request(&mut reader) {
        let cseq = &request.cseq;
        let response = match request.method.as_str() {
            "DESCRIBE" => format!(
                "RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Type: application/sdp\r\nContent-Length: {}\r\n\r\n{sdp}",
                sdp.len()
            ),
            "SETUP" => {
                let channel = 2 * track_of(&request.url);
                format!(
                    "RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 1\r\nTransport: RTP/AVP/TCP;unicast;interleaved={channel}-{}\r\n\r\n",
                    channel + 1
                )
            }
            "OPTIONS" => format!(
                "RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nPublic: OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN\r\n\r\n"
            ),
            _ => format!("RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 1\r\n\r\n"),
        };
        if writer
            .lock()
            .unwrap()
            .write_all(response.as_bytes())
            .is_err()
        {
            return;
        }
        if request.method == "PLAY" {
            break;
        }
    }
    for (track, socket) in sockets.iter().enumerate() {
        let (socket, writer) = (socket.clone(), writer.clone());
        std::thread::spawn(move || relay(&socket, &writer, (2 * track) as u8));
    }
}

fn relay(socket: &UdpSocket, writer: &Mutex<TcpStream>, channel: u8) {
    let mut datagram = vec![0u8; MAX_DATAGRAM];
    loop {
        let len = socket.recv(&mut datagram).expect("receive ffmpeg's rtp");
        let mut framed = vec![b'$', channel];
        framed.extend_from_slice(&(len as u16).to_be_bytes());
        framed.extend_from_slice(&datagram[..len]);
        if writer.lock().unwrap().write_all(&framed).is_err() {
            return;
        }
    }
}

fn track_of(url: &str) -> usize {
    let (_, track) = url
        .rsplit_once(TRACK_CONTROL)
        .unwrap_or_else(|| panic!("SETUP of {url} names no track"));
    track.parse().expect("track number")
}

struct Request {
    method: String,
    url: String,
    cseq: String,
}

fn read_request(reader: &mut BufReader<TcpStream>) -> Option<Request> {
    let mut request_line = String::new();
    let mut cseq = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).ok()? == 0 {
            return None;
        }
        let line = line.trim_end();
        if line.is_empty() && !request_line.is_empty() {
            break;
        }
        if request_line.is_empty() {
            request_line = line.to_string();
        } else if let Some(value) = line.strip_prefix("CSeq:") {
            cseq = value.trim().to_string();
        }
    }
    let mut words = request_line.split(' ');
    Some(Request {
        method: words.next()?.to_string(),
        url: words.next()?.to_string(),
        cseq,
    })
}
