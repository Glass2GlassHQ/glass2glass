// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// the test builds file URLs from unix paths
#![cfg(all(unix, feature = "hls", feature = "dash"))]

use std::path::{Path, PathBuf};

use g2g_core::runtime::{Registry, SourceLoop};
use g2g_core::{ByteStreamEncoding, Caps, G2gError, OutputSink, PipelinePacket, PushOutcome};
use g2g_plugins::dashsrc::DashSrc;
use g2g_plugins::fakesink::FakeSink;
use g2g_plugins::hlssrc::HlsSrc;
use g2g_plugins::uridecodebin;

const SEGMENT_0: &[u8] = b"first segment";
const RANGED_FILE: &[u8] = b"skip this, then the ranged part, then skip this";
const RANGE_OFFSET: usize = 11;
const RANGE_LENGTH: usize = 15;

// the space in the directory name reaches the fetch layer as %20
fn package_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("g2g m2219 {} {name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn file_url(path: &Path) -> String {
    format!("file://{}", path.display()).replace(' ', "%20")
}

#[derive(Default)]
struct Body(Vec<u8>);

impl OutputSink for Body {
    fn poll_push(
        &mut self,
        _cx: &mut core::task::Context<'_>,
        packet_slot: &mut Option<PipelinePacket>,
    ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
        if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
            self.0
                .extend_from_slice(frame.domain.as_system_slice().expect("system memory"));
        }
        core::task::Poll::Ready(Ok(PushOutcome::Accepted))
    }
}

fn write_hls_package(dir: &Path) -> PathBuf {
    std::fs::write(dir.join("seg0.ts"), SEGMENT_0).unwrap();
    std::fs::write(dir.join("ranged.ts"), RANGED_FILE).unwrap();
    let playlist = format!(
        "#EXTM3U\n\
         #EXT-X-VERSION:4\n\
         #EXT-X-TARGETDURATION:4\n\
         #EXTINF:4.0,\n\
         seg0.ts\n\
         #EXTINF:4.0,\n\
         #EXT-X-BYTERANGE:{RANGE_LENGTH}@{RANGE_OFFSET}\n\
         ranged.ts\n\
         #EXT-X-ENDLIST\n"
    );
    let playlist_path = dir.join("index.m3u8");
    std::fs::write(&playlist_path, playlist).unwrap();
    playlist_path
}

#[tokio::test]
async fn hlssrc_plays_a_package_on_disk() {
    let dir = package_dir("hls");
    let playlist_path = write_hls_package(&dir);
    let mut src = HlsSrc::new(file_url(&playlist_path));
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let mut body = Body::default();
    src.run(&mut body).await.expect("the package plays");

    let mut expected = SEGMENT_0.to_vec();
    expected.extend_from_slice(&RANGED_FILE[RANGE_OFFSET..RANGE_OFFSET + RANGE_LENGTH]);
    assert_eq!(body.0, expected);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn hlssrc_reports_a_missing_segment_file() {
    let dir = package_dir("missing");
    let playlist_path = write_hls_package(&dir);
    std::fs::remove_file(dir.join("seg0.ts")).unwrap();
    let mut src = HlsSrc::new(file_url(&playlist_path));
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::MpegTs,
    })
    .unwrap();
    let result = src.run(&mut Body::default()).await;
    assert_eq!(result.err(), Some(G2gError::FileNotFound));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn uridecodebin_sends_a_local_playlist_to_hlssrc() {
    let dir = package_dir("uri");
    let playlist_path = write_hls_package(&dir);
    let mut registry = Registry::new();
    registry.register_uri(uridecodebin::file_handler());
    let is_transport_stream = |caps: &Caps| {
        *caps
            == Caps::ByteStream {
                encoding: ByteStreamEncoding::MpegTs,
            }
    };
    let graph = registry
        .build_uridecodebin(
            &file_url(&playlist_path),
            FakeSink::new(),
            &is_transport_stream,
            4,
        )
        .expect("the playlist resolves to an HLS source");
    assert_eq!(graph.finish().expect("graph validates").node_count(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn dashsrc_plays_a_package_on_disk() {
    const SEGMENTS: usize = 2;
    let dir = package_dir("dash");
    let init = b"init".to_vec();
    let segments: Vec<Vec<u8>> = (0..SEGMENTS)
        .map(|index| format!("segment {index}").into_bytes())
        .collect();
    std::fs::write(dir.join("init.mp4"), &init).unwrap();
    for (index, segment) in segments.iter().enumerate() {
        std::fs::write(dir.join(format!("seg{index}.m4s")), segment).unwrap();
    }
    let mpd = format!(
        "<?xml version=\"1.0\"?>\n\
         <MPD mediaPresentationDuration=\"PT{SEGMENTS}S\" type=\"static\">\n\
           <Period>\n\
             <AdaptationSet mimeType=\"video/mp4\" codecs=\"avc1.4d401f\">\n\
               <SegmentTemplate initialization=\"init.mp4\" media=\"seg$Number$.m4s\" \
                  startNumber=\"0\" duration=\"1000\" timescale=\"1000\"/>\n\
               <Representation id=\"v0\" bandwidth=\"1000000\" width=\"64\" height=\"48\"/>\n\
             </AdaptationSet>\n\
           </Period>\n\
         </MPD>"
    );
    let mpd_path = dir.join("manifest.mpd");
    std::fs::write(&mpd_path, mpd).unwrap();

    let mut src = DashSrc::new(file_url(&mpd_path));
    src.configure_pipeline(&Caps::ByteStream {
        encoding: ByteStreamEncoding::IsoBmff,
    })
    .unwrap();
    let mut body = Body::default();
    src.run(&mut body).await.expect("the package plays");

    let expected: Vec<u8> = std::iter::once(init).chain(segments).flatten().collect();
    assert_eq!(body.0, expected);
    let _ = std::fs::remove_dir_all(&dir);
}
