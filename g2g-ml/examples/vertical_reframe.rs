use std::future::Future;
use std::pin::Pin;
use std::process;

use g2g_core::element::{AsyncElement, OutputSink};
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::{parse_launch, run_graph_threaded, LaunchFactory};
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, Caps, ConfigureOutcome, ElementMetadata, G2gError, PropError,
    PropKind, PropValue, PropertySpec, Roi,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::TokioThreadSpawner;

const ANY_LABEL: i64 = -1;
const LINK_CAPACITY: usize = 4;

static FOLLOW_TARGET_PROPERTIES: &[PropertySpec] = &[PropertySpec::new(
    "label",
    PropKind::Int,
    "class index of the detections to follow, -1 for any class",
)
.with_default("-1")];

#[derive(Debug, Default)]
struct FollowTarget {
    label: Option<u32>,
    followed_track: Option<u64>,
    subject: u32,
}

impl FollowTarget {
    fn mark_target(&mut self, meta: &mut AnalyticsMeta) {
        let candidates: Vec<(usize, Roi, f32)> = meta
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| match node {
                AnalyticsNode::Detection(detection)
                    if self.label.is_none_or(|wanted| wanted == detection.label) =>
                {
                    let roi = Roi {
                        bbox: detection.bbox,
                        id: 0,
                        label: detection.label,
                    };
                    Some((index, roi, detection.confidence))
                }
                _ => None,
            })
            .collect();
        let followed = self.followed_track.and_then(|id| {
            candidates
                .iter()
                .find(|(index, _, _)| meta.tracking_id(*index) == Some(id))
        });
        let chosen = followed.or_else(|| {
            candidates
                .iter()
                .max_by(|(_, _, a), (_, _, b)| a.total_cmp(b))
        });
        let Some(&(index, mut roi, _)) = chosen else {
            return;
        };
        let track = meta.tracking_id(index);
        if track != self.followed_track {
            self.subject = self.subject.wrapping_add(1);
            self.followed_track = track;
        }
        roi.id = self.subject;
        meta.push(AnalyticsNode::Roi(roi));
    }
}

impl AsyncElement for FollowTarget {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Follow target",
            "Filter/Analytics",
            "Marks the detection to frame with a Roi node",
            "g2g example",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn output_domain_follows_input(&self) -> bool {
        true
    }

    fn configure_pipeline(&mut self, _absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        Ok(ConfigureOutcome::Accepted)
    }

    fn properties(&self) -> &'static [PropertySpec] {
        FOLLOW_TARGET_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "label" => {
                let label = value.as_int().ok_or(PropError::Type)?;
                self.label = match label {
                    ANY_LABEL => None,
                    _ => Some(u32::try_from(label).map_err(|_| PropError::Value)?),
                };
                Ok(())
            }
            _ => Err(PropError::Unknown),
        }
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "label" => Some(PropValue::Int(self.label.map_or(ANY_LABEL, i64::from))),
            _ => None,
        }
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    if let Some(meta) = frame.meta.get_mut::<AnalyticsMeta>() {
                        self.mark_target(meta);
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // g2g's transform arm forwards EOS.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let line = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    if line.is_empty() {
        eprintln!("usage: vertical_reframe <launch line>");
        process::exit(2);
    }
    let mut registry = default_registry();
    g2g_ml::register(&mut registry);
    registry.register_launch(LaunchFactory::new("followtarget", Vec::new(), || {
        Box::new(FollowTarget::default())
    }));
    let graph = match parse_launch(&registry, &line) {
        Ok(graph) => graph,
        Err(error) => {
            eprintln!("vertical_reframe: {error:?}");
            process::exit(1);
        }
    };
    let clock = WallClock::new();
    if let Err(error) = run_graph_threaded(graph, &clock, LINK_CAPACITY, &TokioThreadSpawner).await
    {
        eprintln!("vertical_reframe: {error:?}");
        process::exit(1);
    }
}
