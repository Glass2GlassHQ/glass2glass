// `decodebin current-video=N` / `current-audio=N` decode the Nth stream of that kind in
// container order, the stream `ffmpeg -map 0:v:N` / `-map 0:a:N` reads. Each fixture holds
// two H.264 and two AAC tracks, interleaved, that differ in size or in rate and channels.
#![cfg(all(feature = "std", feature = "ffmpeg", target_os = "linux"))]

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Mutex;

use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory, ParseError};
use g2g_core::stream::StreamType;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, OutputSink, PadTemplate,
    PadTemplates, PipelineClock, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};
use g2g_plugins::registry::default_registry;

const CONTAINERS: [&str; 3] = ["mp4", "mkv", "ts"];
const RECORDING_SINK: &str = "capsrecordingsink";

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

// the last caps each recording sink saw, by its label
static RECORDED_CAPS: Mutex<BTreeMap<String, Caps>> = Mutex::new(BTreeMap::new());

#[derive(Debug, Default)]
struct CapsRecordingSink {
    label: String,
}

static RECORDING_SINK_PROPS: &[PropertySpec] = &[PropertySpec::new(
    "label",
    PropKind::Str,
    "key the caps are recorded under",
)];

impl PadTemplates for CapsRecordingSink {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::sink_any()])
    }
}

impl AsyncElement for CapsRecordingSink {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn intercept_caps(&self, caps: &Caps) -> Result<Caps, G2gError> {
        Ok(caps.clone())
    }

    fn caps_constraint_as_sink(&self) -> CapsConstraint<'_> {
        CapsConstraint::AcceptsAny
    }

    fn configure_pipeline(&mut self, _caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        RECORDING_SINK_PROPS
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "label" => {
                self.label = value.as_str().ok_or(PropError::Type)?.to_string();
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "label" => Some(PropValue::Str(self.label.clone())),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        _out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if let PipelinePacket::CapsChanged(caps) = packet {
                RECORDED_CAPS
                    .lock()
                    .expect("caps record lock")
                    .insert(self.label.clone(), caps);
            }
            Ok(())
        })
    }
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn fixture_name(container: &str) -> String {
    format!("multitrack_h264_aac.{container}")
}

// the checked-in `ffprobe -show_entries stream=codec_type,width,height,sample_rate,channels
// -of flat` of a fixture, one field map per stream in ffprobe's order
fn probed_streams(container: &str) -> Vec<BTreeMap<String, String>> {
    let path = fixture(&format!("{}.probe", fixture_name(container)));
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is checked in: {e}", path.display()));
    let mut streams: Vec<BTreeMap<String, String>> = Vec::new();
    for line in text.lines() {
        let (key, value) = line.split_once('=').expect("a flat key=value line");
        let mut parts = key.split('.');
        let index: usize = parts
            .nth(2)
            .and_then(|index| index.parse().ok())
            .expect("a streams.stream.N.field key");
        let field = parts.next().expect("a field after the stream index");
        if streams.len() <= index {
            streams.resize_with(index + 1, BTreeMap::new);
        }
        streams[index].insert(field.to_string(), value.trim_matches('"').to_string());
    }
    streams
}

fn probed_of_kind(container: &str, kind: StreamType) -> Vec<BTreeMap<String, String>> {
    probed_streams(container)
        .into_iter()
        .filter(|stream| stream["codec_type"] == kind.name())
        .collect()
}

fn probed_u32(stream: &BTreeMap<String, String>, field: &str) -> u32 {
    stream[field]
        .parse()
        .unwrap_or_else(|e| panic!("probed {field} {}: {e}", stream[field]))
}

fn registry() -> g2g_core::runtime::Registry {
    let mut registry = default_registry();
    registry.register_launch(LaunchFactory::of::<CapsRecordingSink>(
        RECORDING_SINK,
        || Box::new(CapsRecordingSink::default()),
    ));
    registry
}

fn launch_line(container: &str, kind: StreamType, index: usize, label: &str) -> String {
    format!(
        "filesrc location={} ! decodebin current-{}={index} ! {RECORDING_SINK} label={label}",
        fixture(&fixture_name(container)).display(),
        kind.name()
    )
}

async fn decoded_caps(container: &str, kind: StreamType, index: usize) -> Caps {
    let label = format!("{container}-{}-{index}", kind.name());
    let line = launch_line(container, kind, index, &label);
    let graph = parse_launch(&registry(), &line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    RECORDED_CAPS
        .lock()
        .expect("caps record lock")
        .get(&label)
        .cloned()
        .unwrap_or_else(|| panic!("`{line}` delivered no caps"))
}

#[tokio::test]
async fn each_video_index_decodes_its_own_track() {
    for container in CONTAINERS {
        let probed = probed_of_kind(container, StreamType::Video);
        assert!(
            probed.len() > 1,
            "{container} fixture has several video tracks"
        );
        for (index, stream) in probed.iter().enumerate() {
            let Caps::RawVideo { width, height, .. } =
                decoded_caps(container, StreamType::Video, index).await
            else {
                panic!("{container} video {index} decoded to raw video");
            };
            assert_eq!(
                (width, height),
                (
                    Dim::Fixed(probed_u32(stream, "width")),
                    Dim::Fixed(probed_u32(stream, "height"))
                ),
                "{container} current-video={index}"
            );
        }
    }
}

#[tokio::test]
async fn each_audio_index_decodes_its_own_track() {
    for container in CONTAINERS {
        let probed = probed_of_kind(container, StreamType::Audio);
        assert!(
            probed.len() > 1,
            "{container} fixture has several audio tracks"
        );
        for (index, stream) in probed.iter().enumerate() {
            let Caps::Audio {
                sample_rate,
                channels,
                ..
            } = decoded_caps(container, StreamType::Audio, index).await
            else {
                panic!("{container} audio {index} decoded to audio");
            };
            assert_eq!(
                (sample_rate, u32::from(channels)),
                (
                    probed_u32(stream, "sample_rate"),
                    probed_u32(stream, "channels")
                ),
                "{container} current-audio={index}"
            );
        }
    }
}

#[test]
fn an_index_past_the_last_stream_fails_at_parse() {
    for container in CONTAINERS {
        for kind in [StreamType::Video, StreamType::Audio] {
            let count = probed_of_kind(container, kind).len();
            let line = launch_line(container, kind, count, "out-of-range");
            let error = parse_launch(&registry(), &line)
                .err()
                .unwrap_or_else(|| panic!("`{line}` names a stream past the last one"));
            assert!(
                matches!(
                    error,
                    ParseError::StreamIndexOutOfRange { kind: k, index, count: c, .. }
                        if k == kind && index == count && c == count
                ),
                "`{line}`: {error}"
            );
        }
    }
}
