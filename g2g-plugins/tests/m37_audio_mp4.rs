//! M37: AAC audio through the fMP4 container. `Mp4AudioSink` writes an
//! `mp4a`/`esds` track from synthetic AAC access units; `Mp4AudioSrc` reads
//! them back byte-exactly with the codec/channels/rate and AudioSpecificConfig
//! recovered during the caps probe. No encoder needed: the elements only frame
//! the bitstream, so hand-built access units exercise the full path on any
//! platform.

use g2g_core::element::{AsyncElement, BoxFuture, OutputSink, PushOutcome};
use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::runtime::SourceLoop;
use g2g_core::{AudioFormat, Caps, G2gError};
use g2g_plugins::mp4audiosink::Mp4AudioSink;
use g2g_plugins::mp4audiosrc::Mp4AudioSrc;

use std::path::PathBuf;

const RATE: u32 = 48_000;
const CHANNELS: u8 = 2;
const FRAMES: usize = 8;
// AAC-LC 48 kHz stereo AudioSpecificConfig.
const ASC: [u8; 2] = [0x11, 0x90];

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("g2g_m37_{}_{}.m4a", std::process::id(), name))
}

/// A synthetic AAC access unit: distinct per index, not a real bitstream (the
/// container only frames bytes).
fn aac_au(index: usize) -> Vec<u8> {
    let len = 24 + index * 3;
    (0..len).map(|b| (b + index * 7) as u8).collect()
}

fn frame(bytes: Vec<u8>, index: usize) -> Frame {
    let pts = (index as u64) * 1024 * 1_000_000_000 / RATE as u64;
    Frame {
        domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
        timing: FrameTiming {
            pts_ns: pts,
            dts_ns: pts,
            duration_ns: 1024 * 1_000_000_000 / RATE as u64,
            capture_ns: pts,
            ..FrameTiming::default()
        },
        sequence: index as u64,
    }
}

#[derive(Default)]
struct Collect {
    aus: Vec<Vec<u8>>,
    eos: bool,
}

impl OutputSink for Collect {
    fn push<'a>(
        &'a mut self,
        packet: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(f) => {
                    let MemoryDomain::System(s) = &f.domain else {
                        panic!("expected system frame");
                    };
                    self.aus.push(s.as_slice().to_vec());
                }
                PipelinePacket::Eos => self.eos = true,
                _ => {}
            }
            Ok(PushOutcome::Accepted)
        })
    }
}

struct Discard;
impl OutputSink for Discard {
    fn push<'a>(
        &'a mut self,
        _p: PipelinePacket,
    ) -> BoxFuture<'a, Result<PushOutcome, G2gError>> {
        Box::pin(async move { Ok(PushOutcome::Accepted) })
    }
}

#[tokio::test]
async fn aac_round_trips_through_the_audio_mp4_container() {
    let path = temp_path("roundtrip");
    let aac_caps = Caps::Audio {
        format: AudioFormat::Aac,
        channels: CHANNELS,
        sample_rate: RATE,
    };

    // --- mux ---
    let aus: Vec<Vec<u8>> = (0..FRAMES).map(aac_au).collect();
    let mut sink = Mp4AudioSink::new(&path).with_audio_specific_config(ASC.to_vec());
    let narrowed = sink.intercept_caps(&aac_caps).expect("intercept aac");
    sink.configure_pipeline(&narrowed).expect("configure sink");
    for (i, au) in aus.iter().enumerate() {
        sink.process(PipelinePacket::DataFrame(frame(au.clone(), i)), &mut Discard)
            .await
            .expect("mux au");
    }
    sink.process(PipelinePacket::Eos, &mut Discard)
        .await
        .expect("mux eos");
    assert_eq!(sink.fragments_written(), FRAMES as u64);

    // --- probe + demux ---
    let mut src = Mp4AudioSrc::new(&path);
    let probed = src.intercept_caps().await.expect("probe");
    assert_eq!(probed, aac_caps, "probe recovers AAC codec/channels/rate");
    assert_eq!(
        src.audio_specific_config(),
        Some(&ASC[..]),
        "probe recovers the AudioSpecificConfig"
    );
    src.configure_pipeline(&probed).expect("configure src");

    let mut out = Collect::default();
    let emitted = src.run(&mut out).await.expect("demux");
    assert_eq!(emitted, FRAMES as u64);
    assert!(out.eos);
    assert_eq!(out.aus, aus, "every access unit recovered byte-exactly");

    let _ = std::fs::remove_file(&path);
}

/// Full audio file loop on Windows: real PCM -> AAC encode -> mux -> demux ->
/// AAC decode -> PCM, with the AudioSpecificConfig carried through the `esds`.
#[cfg(all(target_os = "windows", feature = "mf-aac"))]
#[tokio::test(flavor = "current_thread")]
async fn pcm_aac_mp4_demux_decode_full_circle() {
    use g2g_plugins::mfaacdecode::MfAacDecode;
    use g2g_plugins::mfaacencode::MfAacEncode;

    let path = temp_path("fullcircle");
    let pcm_caps = Caps::Audio {
        format: AudioFormat::PcmS16Le,
        channels: CHANNELS,
        sample_rate: RATE,
    };

    // encode PCM -> AAC
    let mut enc = MfAacEncode::new();
    let narrowed = enc.intercept_caps(&pcm_caps).expect("intercept pcm");
    enc.configure_pipeline(&narrowed).expect("encoder init");
    let asc = enc.audio_specific_config().expect("asc").to_vec();

    let mut encoded = Collect::default();
    for i in 0..40usize {
        let mut data = Vec::with_capacity(1024 * CHANNELS as usize * 2);
        for s in 0..1024 {
            let v = ((((s + i * 16) % 128) as i16) - 64) * 200;
            for _ in 0..CHANNELS {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        let f = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(data.into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: i as u64,
        };
        enc.process(PipelinePacket::DataFrame(f), &mut encoded)
            .await
            .expect("encode");
    }
    enc.process(PipelinePacket::Eos, &mut encoded).await.expect("enc eos");
    assert!(!encoded.aus.is_empty());

    // mux the AAC into an .m4a
    let mut sink = Mp4AudioSink::new(&path).with_audio_specific_config(asc);
    let aac_caps = Caps::Audio {
        format: AudioFormat::Aac,
        channels: CHANNELS,
        sample_rate: RATE,
    };
    sink.configure_pipeline(&aac_caps).expect("mux init");
    for (i, au) in encoded.aus.iter().enumerate() {
        sink.process(PipelinePacket::DataFrame(frame(au.clone(), i)), &mut Discard)
            .await
            .expect("mux");
    }
    sink.process(PipelinePacket::Eos, &mut Discard).await.expect("mux eos");

    // demux and decode back to PCM
    let mut src = Mp4AudioSrc::new(&path);
    let probed = src.intercept_caps().await.expect("probe");
    let recovered_asc = src.audio_specific_config().expect("asc").to_vec();
    src.configure_pipeline(&probed).expect("demux init");

    let mut demuxed = Collect::default();
    src.run(&mut demuxed).await.expect("demux");
    assert_eq!(demuxed.aus.len(), encoded.aus.len(), "all AUs survive the container");
    assert_eq!(demuxed.aus, encoded.aus, "AUs byte-exact through the container");

    let mut dec = MfAacDecode::new().with_audio_specific_config(recovered_asc);
    dec.configure_pipeline(&probed).expect("decoder init");
    let mut pcm = Collect::default();
    for au in &demuxed.aus {
        let f = Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(au.clone().into_boxed_slice())),
            timing: FrameTiming::default(),
            sequence: 0,
        };
        dec.process(PipelinePacket::DataFrame(f), &mut pcm).await.expect("decode");
    }
    dec.process(PipelinePacket::Eos, &mut pcm).await.expect("dec eos");
    assert!(
        pcm.aus.iter().map(|p| p.len()).sum::<usize>() > 0,
        "the file loop reproduces PCM audio"
    );

    let _ = std::fs::remove_file(&path);
}
