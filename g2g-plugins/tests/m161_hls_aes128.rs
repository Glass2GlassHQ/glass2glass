//! M161 HLS AES-128 segment decryption (`#EXT-X-KEY:METHOD=AES-128`). A local
//! HTTP server serves a media playlist, a 16-byte key resource, and segments
//! encrypted with AES-128-CBC. `HlsSrc` fetches the key and decrypts each
//! segment back to its plaintext, both with an explicit `IV` and with the
//! media-sequence-derived IV. The ciphertext is produced independently here (the
//! `aes`/`cbc` encryptor side) so the test exercises the real decrypt path.

#![cfg(feature = "hls")]

use core::future::Future;
use core::pin::Pin;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

use aes::cipher::{block_padding::Pkcs7, BlockEncryptMut, KeyIvInit};
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    ByteStreamEncoding, Caps, G2gError, MemoryDomain, OutputSink, PipelinePacket, PushOutcome,
};
use g2g_plugins::hlssrc::HlsSrc;

type Aes128CbcEnc = cbc::Encryptor<aes::Aes128>;

const KEY: [u8; 16] = *b"0123456789abcdef";

fn encrypt(iv: &[u8; 16], plaintext: &[u8]) -> Vec<u8> {
    let mut buf = plaintext.to_vec();
    let msg_len = buf.len();
    buf.resize(msg_len + 16, 0);
    let ct = Aes128CbcEnc::new(&KEY.into(), &(*iv).into())
        .encrypt_padded_mut::<Pkcs7>(&mut buf, msg_len)
        .unwrap();
    ct.to_vec()
}

fn iv_from_sequence(seq: u64) -> [u8; 16] {
    let mut iv = [0u8; 16];
    iv[8..].copy_from_slice(&seq.to_be_bytes());
    iv
}

#[derive(Default)]
struct CaptureSink {
    body: Vec<u8>,
    frames: usize,
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
                    if let MemoryDomain::System(s) = &f.domain {
                        self.body.extend_from_slice(s.as_slice());
                        self.frames += 1;
                    }
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

/// Serve `/enc.m3u8`, the key at `/enc.key`, and `/seg{n}.ts` ciphertext blobs.
fn serve(playlist: String, segs: Vec<Vec<u8>>) -> String {
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
            let body: Vec<u8> = if path == "/enc.m3u8" {
                playlist.clone().into_bytes()
            } else if path == "/enc.key" {
                KEY.to_vec()
            } else if let Some(idx) = path
                .strip_prefix("/seg")
                .and_then(|s| s.strip_suffix(".ts"))
                .and_then(|s| s.parse::<usize>().ok())
            {
                segs.get(idx).cloned().unwrap_or_default()
            } else {
                let _ = stream.write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                );
                continue;
            };
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(header.as_bytes());
            let _ = stream.write_all(&body);
        }
    });
    format!("http://127.0.0.1:{port}/enc.m3u8")
}

async fn run_and_capture(url: String) -> CaptureSink {
    let mut src = HlsSrc::new(url);
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut sink = CaptureSink::default();
    src.run(&mut sink).await.unwrap();
    sink
}

#[tokio::test]
async fn aes128_explicit_iv_decrypts_segments() {
    let pt0: Vec<u8> = (0..5_000u32).map(|i| (i % 251) as u8).collect();
    let pt1: Vec<u8> = (0..4_001u32).map(|i| (i % 239) as u8 ^ 0x5a).collect();
    let iv = [0x11u8; 16];
    let segs = vec![encrypt(&iv, &pt0), encrypt(&iv, &pt1)];

    let hex_iv = "0x11111111111111111111111111111111";
    let playlist = format!(
        "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:0\n\
         #EXT-X-KEY:METHOD=AES-128,URI=\"enc.key\",IV={hex_iv}\n\
         #EXTINF:4.0,\nseg0.ts\n#EXTINF:4.0,\nseg1.ts\n#EXT-X-ENDLIST\n"
    );
    let url = serve(playlist, segs);

    let sink = run_and_capture(url).await;
    assert!(sink.eos);
    assert_eq!(sink.frames, 2);
    let expected = [pt0, pt1].concat();
    assert_eq!(
        sink.body, expected,
        "explicit-IV AES-128 segments decrypt to plaintext"
    );
}

#[tokio::test]
async fn aes128_sequence_derived_iv_decrypts_segments() {
    // No IV in the tag: each segment uses its media-sequence number as the IV.
    let media_sequence = 7u64;
    let pt0: Vec<u8> = (0..3_000u32).map(|i| (i % 200) as u8).collect();
    let pt1: Vec<u8> = (0..3_100u32).map(|i| (i % 100) as u8 ^ 0x33).collect();
    let segs = vec![
        encrypt(&iv_from_sequence(media_sequence), &pt0),
        encrypt(&iv_from_sequence(media_sequence + 1), &pt1),
    ];

    let playlist = format!(
        "#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:{media_sequence}\n\
         #EXT-X-KEY:METHOD=AES-128,URI=\"enc.key\"\n\
         #EXTINF:4.0,\nseg0.ts\n#EXTINF:4.0,\nseg1.ts\n#EXT-X-ENDLIST\n"
    );
    let url = serve(playlist, segs);

    let sink = run_and_capture(url).await;
    assert!(sink.eos);
    assert_eq!(sink.frames, 2);
    let expected = [pt0, pt1].concat();
    assert_eq!(
        sink.body, expected,
        "sequence-derived-IV AES-128 segments decrypt to plaintext"
    );
}
