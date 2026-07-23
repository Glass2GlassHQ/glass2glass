//! M163 HLS SAMPLE-AES key auto-wiring. A `METHOD=SAMPLE-AES` playlist makes
//! `HlsSrc` fetch the `#EXT-X-KEY` key, publish it (with the resolved IV) into a
//! shared handle for a downstream `SampleAesDecrypt`, and forward the segment
//! bytes *undecrypted* (sample encryption is handled after the demuxer). Without
//! a handle the playlist is rejected. The decrypt half is covered by the
//! `sampleaesdecrypt` unit tests; this proves the publish + passthrough path.

#![cfg(feature = "hls")]

use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use g2g_core::runtime::SourceLoop;
use g2g_core::{ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PushOutcome};
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::sampleaesdecrypt::{new_key_handle, SampleAesKey};

const KEY: [u8; 16] = *b"0123456789abcdef";
const IV: [u8; 16] = [0x22; 16];

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    eos: bool,
}

impl OutputSink for CaptureSink {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> Pin<Box<dyn Future<Output = Result<PushOutcome, G2gError>> + 'a>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    if let Some(s) = f.domain.as_system_slice() {
                        self.body.extend_from_slice(s);
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

const PLAYLIST: &str = "#EXTM3U\n\
    #EXT-X-TARGETDURATION:4\n\
    #EXT-X-MEDIA-SEQUENCE:0\n\
    #EXT-X-KEY:METHOD=SAMPLE-AES,URI=\"enc.key\",IV=0x22222222222222222222222222222222\n\
    #EXTINF:4.0,\n\
    seg0.ts\n\
    #EXT-X-ENDLIST\n";

fn serve(segment: Vec<u8>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    thread::spawn(move || {
        for conn in listener.incoming() {
            let mut stream = match conn {
                Ok(s) => s,
                Err(_) => break,
            };
            let mut req = Vec::new();
            let mut byte = [0u8; 1];
            while stream.read(&mut byte).unwrap_or(0) == 1 {
                req.push(byte[0]);
                if req.ends_with(b"\r\n\r\n") {
                    break;
                }
            }
            let line = String::from_utf8_lossy(&req);
            let path = line.split_whitespace().nth(1).unwrap_or("");
            let body: Vec<u8> = match path {
                "/sa.m3u8" => PLAYLIST.as_bytes().to_vec(),
                "/enc.key" => KEY.to_vec(),
                "/seg0.ts" => segment.clone(),
                _ => {
                    let _ = stream.write_all(
                        b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                    continue;
                }
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    format!("http://127.0.0.1:{port}/sa.m3u8")
}

#[tokio::test]
async fn publishes_key_and_forwards_sample_aes_segment_undecrypted() {
    // The "segment" stands in for the TS bytes; the publish + passthrough path
    // does not inspect them.
    let segment: Vec<u8> = (0..6_000u32).map(|i| (i % 251) as u8 ^ 0x3c).collect();
    let url = serve(segment.clone());

    let handle = new_key_handle();
    let mut src = HlsSrc::new(url).with_sample_aes_key_handle(handle.clone());
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();

    assert!(sink.eos);
    assert_eq!(
        sink.body, segment,
        "SAMPLE-AES bytes forwarded undecrypted to the demuxer"
    );
    assert_eq!(
        *handle.lock().unwrap(),
        Some(SampleAesKey { key: KEY, iv: IV }),
        "HlsSrc published the fetched key and explicit IV for the downstream decryptor",
    );
}

#[tokio::test]
async fn sample_aes_without_a_handle_is_rejected() {
    let url = serve((0..100u32).map(|i| i as u8).collect());
    let mut src = HlsSrc::new(url);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    assert_eq!(src.run(&mut sink).await, Err(G2gError::CapsMismatch));
}
