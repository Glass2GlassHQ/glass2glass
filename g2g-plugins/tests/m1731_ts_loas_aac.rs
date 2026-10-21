#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/ffmpeg_tool.rs"]
mod ffmpeg_tool;
#[path = "common/ffprobe_tool.rs"]
mod ffprobe_tool;

use std::path::{Path, PathBuf};
use std::process::Command;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{AsyncElement, AudioFormat, ByteStreamEncoding, Caps, MultiInputElement, PropValue};
use g2g_plugins::aacparse::AacParse;
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mpegts::{TsDemuxer, TsPacketSplitter, STREAM_TYPE_AAC_LATM};
use g2g_plugins::tsdemux::{TsDemux, TsStream};
use g2g_plugins::tsmux::TsMux;
use g2g_plugins::tsmuxn::TsMux as TsMuxN;

use collect_sink::CollectSink;
use ffmpeg_tool::have_ffmpeg;
use ffprobe_tool::have_ffprobe;

// LOAS AudioSyncStream (ISO/IEC 14496-3 1.7.2): an 11-bit syncword and a 13-bit length in 3 bytes
const LOAS_SYNC_WORD: u32 = 0x2B7;
const LOAS_HEADER_LEN: usize = 3;
// a PayloadLengthInfo byte of this value continues the length into the next byte
const LATM_LENGTH_CONTINUES: usize = 255;

#[derive(Debug, Clone, Copy)]
struct ObjectType {
    fixture: &'static str,
    // ffmpeg's LATM decoder leaves the sample rate unset for ER AAC (LD, ELD), so its CLI decodes no LOAS of them
    ffmpeg_decodes_loas: bool,
}

const AAC_LD: ObjectType = ObjectType {
    fixture: "aac_ld_stereo_48000.m4a",
    ffmpeg_decodes_loas: false,
};
const AAC_ELD: ObjectType = ObjectType {
    fixture: "aac_eld_stereo_48000.m4a",
    ffmpeg_decodes_loas: false,
};
const USAC: ObjectType = ObjectType {
    fixture: "aac_usac_stereo_48000.m4a",
    ffmpeg_decodes_loas: true,
};

#[tokio::test]
async fn aac_ld_goes_into_mpeg_ts_as_loas() {
    check(AAC_LD).await;
}

#[tokio::test]
async fn aac_eld_goes_into_mpeg_ts_as_loas() {
    check(AAC_ELD).await;
}

#[tokio::test]
async fn usac_goes_into_mpeg_ts_as_loas() {
    check(USAC).await;
}

async fn check(object_type: ObjectType) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(object_type.fixture);
    let Ok(file) = std::fs::read(&source) else {
        eprintln!(
            "{} is absent (tools/aac-loas-fixtures.sh makes it); skipping",
            source.display()
        );
        return;
    };
    let packets = demux(&file).await;
    let caps = packets
        .iter()
        .find_map(|packet| match packet {
            PipelinePacket::CapsChanged(caps) => Some(caps.clone()),
            _ => None,
        })
        .expect("the demuxer announces the track caps");
    let frames = data_frames(&packets);
    let (config, units) = frames.split_first().expect("a config frame leads the track");

    let muxed = [
        ("tsmux", mux_single(demux(&file).await, &caps).await),
        ("tsmuxn", mux_fan_in(demux(&file).await, &caps).await),
    ];
    for (muxer, ts) in muxed {
        let carried = read_back(&ts, muxer);
        let loas: Vec<LoasFrame> = carried.iter().map(|frame| parse_loas(frame)).collect();
        assert!(
            loas.iter().all(|frame| frame.config == *config),
            "{muxer}: every LOAS frame carries the source config"
        );
        assert_eq!(
            loas.into_iter()
                .map(|frame| frame.payload)
                .collect::<Vec<_>>(),
            units,
            "{muxer}: the source access units, in order and unchanged"
        );

        let demuxed = tsdemux_frames(ts.clone()).await;
        assert_eq!(
            demuxed, carried,
            "{muxer}: TsDemux forwards each LOAS frame as its PES carries it"
        );
        let parsed = aacparse(&demuxed).await;
        assert_eq!(
            parsed.caps,
            [caps.clone()],
            "{muxer}: AacParse reads the source caps back from the LOAS frames"
        );
        assert_eq!(
            parsed.frames, demuxed,
            "{muxer}: AacParse forwards the LOAS frames unchanged"
        );

        let ts_path = scratch_path(object_type, muxer, "ts");
        std::fs::write(&ts_path, &ts).expect("write the transport stream");
        check_ffprobe(&source, &ts_path, muxer);
        check_gstreamer_parse(&ts_path, &caps, muxer);
        check_decoded_pcm(object_type, &source, &ts_path, muxer);
    }
}

async fn demux(file: &[u8]) -> Vec<PipelinePacket> {
    let mut demux = Mp4Demux::new();
    demux
        .set_property("stream", PropValue::Str("aac".into()))
        .expect("stream=aac");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    demux
        .process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(file.to_vec().into_boxed_slice())),
                FrameTiming::default(),
                0,
            )),
            &mut sink,
        )
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.packets
        .into_iter()
        .filter(|packet| !matches!(packet, PipelinePacket::Eos))
        .collect()
}

fn data_frames(packets: &[PipelinePacket]) -> Vec<Vec<u8>> {
    packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice().map(<[u8]>::to_vec),
            _ => None,
        })
        .collect()
}

async fn mux_single(packets: Vec<PipelinePacket>, caps: &Caps) -> Vec<u8> {
    let mut mux = TsMux::new();
    mux.configure_pipeline(caps).expect("configure");
    let mut sink = CollectSink::default();
    for packet in packets {
        mux.process(packet, &mut sink).await.expect("mux");
    }
    data_frames(&sink.packets).concat()
}

async fn mux_fan_in(packets: Vec<PipelinePacket>, caps: &Caps) -> Vec<u8> {
    let mut mux = TsMuxN::new(1);
    mux.configure_pipeline(0, caps).expect("configure");
    let mut sink = CollectSink::default();
    for packet in packets {
        mux.process(0, packet, &mut sink).await.expect("mux");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    data_frames(&sink.packets).concat()
}

// the PES payloads of the one stream in `ts`, which the PMT must declare LATM
fn read_back(ts: &[u8], muxer: &str) -> Vec<Vec<u8>> {
    let mut demuxer = TsDemuxer::new();
    let mut splitter = TsPacketSplitter::new();
    splitter.push(ts, &mut demuxer);
    splitter.finish(&mut demuxer);
    demuxer.flush();
    let stream_types: Vec<u8> = demuxer
        .streams()
        .iter()
        .map(|stream| stream.stream_type)
        .collect();
    assert_eq!(
        stream_types,
        [STREAM_TYPE_AAC_LATM],
        "{muxer}: the PMT declares LATM"
    );
    demuxer
        .take_units()
        .into_iter()
        .map(|unit| unit.data)
        .collect()
}

async fn tsdemux_frames(ts: Vec<u8>) -> Vec<Vec<u8>> {
    let mut demux = TsDemux::new().with_stream(TsStream::Aac);
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::MpegTs,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    demux
        .process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(ts.into_boxed_slice())),
                FrameTiming::default(),
                0,
            )),
            &mut sink,
        )
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    data_frames(&sink.packets)
}

#[derive(Debug)]
struct Parsed {
    caps: Vec<Caps>,
    frames: Vec<Vec<u8>>,
}

async fn aacparse(loas_frames: &[Vec<u8>]) -> Parsed {
    let mut parse = AacParse::new();
    parse
        .configure_pipeline(&Caps::Audio {
            format: AudioFormat::Aac,
            channels: 0,
            sample_rate: 0,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .expect("configure");
    let mut sink = CollectSink::default();
    for frame in loas_frames {
        parse
            .process(
                PipelinePacket::DataFrame(Frame::new(
                    MemoryDomain::System(SystemSlice::from_boxed(frame.clone().into_boxed_slice())),
                    FrameTiming::default(),
                    0,
                )),
                &mut sink,
            )
            .await
            .expect("parse");
    }
    Parsed {
        caps: sink
            .packets
            .iter()
            .filter_map(|packet| match packet {
                PipelinePacket::CapsChanged(caps) => Some(caps.clone()),
                _ => None,
            })
            .collect(),
        frames: data_frames(&sink.packets),
    }
}

#[derive(Debug)]
struct LoasFrame {
    config: Vec<u8>,
    payload: Vec<u8>,
}

struct BitCursor<'a> {
    data: &'a [u8],
    position: usize,
}

impl BitCursor<'_> {
    fn read(&mut self, bits: u32) -> u32 {
        (0..bits).fold(0, |value, _| {
            let byte = self.data[self.position / 8];
            let bit = (byte >> (7 - self.position % 8)) & 1;
            self.position += 1;
            (value << 1) | u32::from(bit)
        })
    }

    // LatmGetValue: the byte count less one in two bits, then the value
    fn latm_value(&mut self) -> u32 {
        let bytes = self.read(2) + 1;
        self.read(bytes * 8)
    }

    fn bytes(&mut self, count: usize) -> Vec<u8> {
        (0..count).map(|_| self.read(8) as u8).collect()
    }
}

// a LOAS frame read by ISO/IEC 14496-3 1.7.3: one AudioMuxElement with its StreamMuxConfig, one program, layer and subframe
fn parse_loas(frame: &[u8]) -> LoasFrame {
    let mut bits = BitCursor {
        data: frame,
        position: 0,
    };
    assert_eq!(bits.read(11), LOAS_SYNC_WORD, "LOAS syncword");
    let element_len = bits.read(13) as usize;
    assert_eq!(
        element_len,
        frame.len() - LOAS_HEADER_LEN,
        "one whole LOAS frame per PES"
    );
    assert_eq!(bits.read(1), 0, "useSameStreamMux: the config rides every frame");
    assert_eq!(bits.read(1), 1, "audioMuxVersion");
    assert_eq!(bits.read(1), 0, "audioMuxVersionA");
    bits.latm_value(); // taraBufferFullness
    assert_eq!(bits.read(1), 1, "allStreamsSameTimeFraming");
    assert_eq!(bits.read(6), 0, "numSubFrames");
    assert_eq!(bits.read(4), 0, "numProgram");
    assert_eq!(bits.read(3), 0, "numLayer");
    let config_bits = bits.latm_value() as usize;
    assert_eq!(config_bits % 8, 0, "the config goes in whole bytes");
    let config = bits.bytes(config_bits / 8);
    assert_eq!(bits.read(3), 0, "frameLengthType: a byte-counted payload");
    bits.read(8); // latmBufferFullness
    assert_eq!(bits.read(1), 0, "otherDataPresent");
    assert_eq!(bits.read(1), 0, "crcCheckPresent");
    let mut payload_len = 0;
    loop {
        let step = bits.read(8) as usize;
        payload_len += step;
        if step != LATM_LENGTH_CONTINUES {
            break;
        }
    }
    let payload = bits.bytes(payload_len);
    assert!(
        frame.len() * 8 - bits.position < 8,
        "only byte alignment follows the payload"
    );
    LoasFrame { config, payload }
}

fn scratch_path(object_type: ObjectType, muxer: &str, extension: &str) -> PathBuf {
    let stem = object_type.fixture.trim_end_matches(".m4a");
    std::env::temp_dir().join(format!(
        "g2g-m1731-{}-{stem}-{muxer}.{extension}",
        std::process::id()
    ))
}

// ffprobe's key=value lines for the first audio stream of `path`
fn ffprobe_stream(path: &Path) -> Vec<String> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,codec_tag,profile",
            "-of",
            "default=nw=1",
        ])
        .arg(path)
        .output()
        .expect("run ffprobe");
    assert!(out.status.success(), "ffprobe reads {}", path.display());
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
}

fn check_ffprobe(source: &Path, ts: &Path, muxer: &str) {
    if !have_ffprobe() {
        eprintln!("ffprobe not present; skipping the ffprobe leg");
        return;
    }
    let source_profile = ffprobe_stream(source)
        .into_iter()
        .find(|line| line.starts_with("profile="))
        .expect("ffprobe names the source profile");
    let ts_stream = ffprobe_stream(ts);
    for expected in [
        String::from("codec_name=aac_latm"),
        format!("codec_tag={STREAM_TYPE_AAC_LATM:#06x}"),
        source_profile,
    ] {
        assert!(
            ts_stream.contains(&expected),
            "{muxer}: ffprobe reports {expected}, got {ts_stream:?}"
        );
    }
}

fn have_gstreamer_element(element: &str) -> bool {
    Command::new("gst-inspect-1.0")
        .args(["--exists", element])
        .status()
        .is_ok_and(|status| status.success())
}

fn check_gstreamer_parse(ts: &Path, caps: &Caps, muxer: &str) {
    if !have_gstreamer_element("tsdemux") || !have_gstreamer_element("aacparse") {
        eprintln!("GStreamer tsdemux / aacparse not present; skipping the GStreamer parse leg");
        return;
    }
    let Caps::Audio {
        channels,
        sample_rate,
        ..
    } = caps
    else {
        panic!("the demuxed track has audio caps");
    };
    let out = Command::new("gst-launch-1.0")
        .arg("-v")
        .arg("filesrc")
        .arg(format!("location={}", ts.display()))
        .args(["!", "tsdemux", "!", "aacparse", "!", "fakesink"])
        .output()
        .expect("run gst-launch-1.0");
    let log = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{muxer}: GStreamer plays the stream: {log}");
    let expected = [
        String::from("stream-format=(string)loas"),
        format!("rate=(int){sample_rate}"),
        format!("channels=(int){channels}"),
    ];
    assert!(
        log.lines().any(|line| line.contains("GstAacParse")
            && line.contains("src: caps")
            && expected.iter().all(|field| line.contains(field.as_str()))),
        "{muxer}: aacparse announces {expected:?}: {log}"
    );
}

// float samples keep the comparison free of any dither a conversion to integers adds
fn ffmpeg_f32(input_args: &[&str], input: &Path) -> Vec<u8> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error"])
        .args(input_args)
        .arg("-i")
        .arg(input)
        .args(["-f", "f32le", "-"])
        .output()
        .expect("run ffmpeg");
    assert!(
        out.status.success(),
        "ffmpeg decodes {}: {}",
        input.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn check_decoded_pcm(object_type: ObjectType, source: &Path, ts: &Path, muxer: &str) {
    if !have_ffmpeg() {
        eprintln!("ffmpeg not present; skipping the decode legs");
        return;
    }
    // the TS carries every access unit, so the reference keeps the priming an edit list or iTunSMPB trims
    let reference = ffmpeg_f32(
        &["-ignore_editlist", "1", "-flags2", "+skip_manual"],
        source,
    );
    assert!(!reference.is_empty(), "ffmpeg decodes the source");
    if object_type.ffmpeg_decodes_loas {
        assert!(
            ffmpeg_f32(&[], ts) == reference,
            "{muxer}: ffmpeg decodes the TS to the source PCM"
        );
    }
    if !have_gstreamer_element("avdec_aac_latm") {
        eprintln!("GStreamer avdec_aac_latm not present; skipping the GStreamer decode leg");
        return;
    }
    let pcm_path = scratch_path(object_type, muxer, "f32");
    let status = Command::new("gst-launch-1.0")
        .arg("-q")
        .arg("filesrc")
        .arg(format!("location={}", ts.display()))
        .args([
            "!",
            "tsdemux",
            "!",
            "aacparse",
            "!",
            "avdec_aac_latm",
            "!",
            "audioconvert",
            "!",
            "audio/x-raw,format=F32LE,layout=interleaved",
            "!",
            "filesink",
        ])
        .arg(format!("location={}", pcm_path.display()))
        .status()
        .expect("run gst-launch-1.0");
    assert!(status.success(), "{muxer}: GStreamer decodes the TS");
    let decoded = std::fs::read(&pcm_path).expect("read the GStreamer decode");
    assert!(
        decoded == reference,
        "{muxer}: GStreamer decodes the TS to the source PCM ({} bytes against {})",
        decoded.len(),
        reference.len()
    );
}
