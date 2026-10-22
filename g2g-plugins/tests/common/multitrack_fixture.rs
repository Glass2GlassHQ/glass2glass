// the multi-track fixtures (two H.264 and two AAC tracks, interleaved, that differ in size or
// in rate and channels), their checked-in ffprobe streams, and a sink that records decoded caps

use std::collections::BTreeMap;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Mutex;

use g2g_core::runtime::{parse_launch, run_graph, LaunchFactory, Registry};
use g2g_core::stream::StreamType;
use g2g_core::{
    AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, G2gError, OutputSink, PadTemplate,
    PadTemplates, PipelineClock, PipelinePacket, PropError, PropKind, PropValue, PropertySpec,
};
use g2g_plugins::registry::default_registry;

pub(crate) const RECORDING_SINK: &str = "capsrecordingsink";

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

pub(crate) fn multitrack_fixture(container: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("multitrack_h264_aac.{container}"))
}

// the checked-in `ffprobe -show_entries stream=codec_type,width,height,sample_rate,channels
// -of flat` of a fixture, one field map per stream in ffprobe's order
fn probed_streams(container: &str) -> Vec<BTreeMap<String, String>> {
    let mut path = multitrack_fixture(container).into_os_string();
    path.push(".probe");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is checked in: {e}", path.to_string_lossy()));
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

pub(crate) fn probed_of_kind(container: &str, kind: StreamType) -> Vec<BTreeMap<String, String>> {
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

pub(crate) fn registry() -> Registry {
    let mut registry = default_registry();
    registry.register_launch(LaunchFactory::of::<CapsRecordingSink>(
        RECORDING_SINK,
        || Box::new(CapsRecordingSink::default()),
    ));
    registry
}

// runs the line to its end and returns the last caps each labelled recording sink saw
pub(crate) async fn run_recording(line: &str, labels: &[&str]) -> Vec<Caps> {
    let graph = parse_launch(&registry(), line).unwrap_or_else(|e| panic!("parses `{line}`: {e}"));
    run_graph(graph, &ZeroClock, 4)
        .await
        .unwrap_or_else(|e| panic!("runs `{line}`: {e:?}"));
    // released before a missing label panics, so the other tests can still read theirs
    let recorded: Vec<Option<Caps>> = {
        let recorded = RECORDED_CAPS.lock().expect("caps record lock");
        labels
            .iter()
            .map(|label| recorded.get(*label).cloned())
            .collect()
    };
    recorded
        .into_iter()
        .zip(labels)
        .map(|(caps, label)| {
            caps.unwrap_or_else(|| panic!("`{line}` delivered no caps to {label}"))
        })
        .collect()
}

// decoded video has the probed size, decoded audio the probed rate and channel count
pub(crate) fn assert_decoded_stream(caps: &Caps, stream: &BTreeMap<String, String>, context: &str) {
    match caps {
        Caps::RawVideo { width, height, .. } => assert_eq!(
            (width.clone(), height.clone()),
            (
                Dim::Fixed(probed_u32(stream, "width")),
                Dim::Fixed(probed_u32(stream, "height"))
            ),
            "{context}"
        ),
        Caps::Audio {
            sample_rate,
            channels,
            ..
        } => assert_eq!(
            (*sample_rate, u32::from(*channels)),
            (
                probed_u32(stream, "sample_rate"),
                probed_u32(stream, "channels")
            ),
            "{context}"
        ),
        other => panic!("{context} decoded to {other:?}"),
    }
}
