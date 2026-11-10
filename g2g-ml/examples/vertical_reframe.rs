// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::future::Future;
use std::pin::Pin;
use std::process;

use g2g_core::element::{AsyncElement, OutputSink};
use g2g_core::frame::PipelinePacket;
use g2g_core::runtime::{parse_launch, run_graph_threaded, LaunchFactory};
use g2g_core::{
    AnalyticsMeta, AnalyticsNode, BBox, Caps, ConfigureOutcome, ElementMetadata, G2gError,
    ObjectDetection, PropError, PropKind, PropValue, PropertySpec, Roi,
};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;
use g2g_plugins::TokioThreadSpawner;

const NO_LABEL: i64 = -1;
const LINK_CAPACITY: usize = 4;
const BALL_HOLD_NS: u64 = 1_000_000_000;
// players closer than this, in normalized frame coordinates, count as one group
const GROUP_RADIUS: f32 = 0.15;
const FOLLOWED_REGION_ID: u32 = 1;

static FOLLOW_TARGET_PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new("label", PropKind::Int, "class index of the ball").with_default("-1"),
    PropertySpec::new(
        "group-label",
        PropKind::Int,
        "class index of the players framed while the ball is lost, -1 for none",
    )
    .with_default("-1"),
];

fn centre(bbox: &BBox) -> (f32, f32) {
    (bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0)
}

fn union(boxes: &[BBox]) -> BBox {
    let left = boxes.iter().map(|b| b.x).fold(f32::MAX, f32::min);
    let top = boxes.iter().map(|b| b.y).fold(f32::MAX, f32::min);
    let right = boxes.iter().map(|b| b.x + b.w).fold(f32::MIN, f32::max);
    let bottom = boxes.iter().map(|b| b.y + b.h).fold(f32::MIN, f32::max);
    BBox {
        x: left,
        y: top,
        w: right - left,
        h: bottom - top,
    }
}

fn densest_group(players: &[BBox]) -> Option<BBox> {
    let near = |a: &BBox, b: &BBox| {
        let ((ax, ay), (bx, by)) = (centre(a), centre(b));
        (ax - bx).hypot(ay - by) <= GROUP_RADIUS
    };
    let seed = players
        .iter()
        .max_by_key(|seed| players.iter().filter(|other| near(seed, other)).count())?;
    let group: Vec<BBox> = players
        .iter()
        .filter(|other| near(seed, other))
        .copied()
        .collect();
    Some(union(&group))
}

fn label_property(value: PropValue) -> Result<Option<u32>, PropError> {
    match value.as_int().ok_or(PropError::Type)? {
        NO_LABEL => Ok(None),
        label => Ok(Some(u32::try_from(label).map_err(|_| PropError::Value)?)),
    }
}

#[derive(Debug, Default)]
struct FollowTarget {
    ball_label: Option<u32>,
    group_label: Option<u32>,
    last_ball_pts_ns: Option<u64>,
}

impl FollowTarget {
    fn detections_of(meta: &AnalyticsMeta, label: Option<u32>) -> Vec<&ObjectDetection> {
        meta.detections()
            .filter(|detection| Some(detection.label) == label)
            .collect()
    }

    fn mark_target(&mut self, meta: &mut AnalyticsMeta, pts_ns: u64) {
        let ball = Self::detections_of(meta, self.ball_label)
            .into_iter()
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            .map(|detection| detection.bbox);
        let ball_recently_seen = self
            .last_ball_pts_ns
            .is_some_and(|seen| pts_ns.saturating_sub(seen) < BALL_HOLD_NS);
        let target = match ball {
            Some(ball) => {
                self.last_ball_pts_ns = Some(pts_ns);
                Some(ball)
            }
            None if ball_recently_seen => None,
            None => {
                let players: Vec<BBox> = Self::detections_of(meta, self.group_label)
                    .into_iter()
                    .map(|detection| detection.bbox)
                    .collect();
                densest_group(&players)
            }
        };
        if let Some(bbox) = target {
            meta.push(AnalyticsNode::Roi(Roi {
                bbox,
                id: FOLLOWED_REGION_ID,
                label: 0,
            }));
        }
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
            "Marks the ball, or the densest group of players while the ball is lost, with a Roi node",
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
            "label" => self.ball_label = label_property(value)?,
            "group-label" => self.group_label = label_property(value)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        let label = match name {
            "label" => self.ball_label,
            "group-label" => self.group_label,
            _ => return None,
        };
        Some(PropValue::Int(label.map_or(NO_LABEL, i64::from)))
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    let pts_ns = frame.timing.pts_ns;
                    if let Some(meta) = frame.meta.get_mut::<AnalyticsMeta>() {
                        self.mark_target(meta, pts_ns);
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
    g2g_core::log::init_from_env();
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
