use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use g2g_core::{
    AnalyticsMeta, AnalyticsNode, AsyncElement, BBox, Caps, ConfigureOutcome, ElementMetadata,
    G2gError, ObjectDetection, OutputSink, PipelinePacket, PropError, PropKind, PropValue,
    PropertySpec, RelationKind, Tracking,
};

use crate::audiofx::double_in_range;

// The property defaults are Ultralytics' bytetrack.yaml, the tracker the hosted
// Python element runs.
const TRACK_HIGH_THRESH_DEFAULT: f64 = 0.25;
const TRACK_LOW_THRESH_DEFAULT: f64 = 0.1;
const NEW_TRACK_THRESH_DEFAULT: f64 = 0.25;
const MATCH_THRESH_DEFAULT: f64 = 0.8;
const TRACK_BUFFER_DEFAULT: u64 = 30;

// Cost limit of the second association, which matches low-confidence boxes on IoU alone.
const LOW_BAND_MATCH_THRESH: f64 = 0.5;
// Cost limit for confirming a new track against the high-confidence boxes left over.
const TENTATIVE_MATCH_THRESH: f64 = 0.7;
// A lost track overlapping a tracked one by more than this is the same object twice.
const DUPLICATE_IOU: f32 = 0.85;

// Kalman noise of the SORT / ByteTrack filter: standard deviations as a fraction
// of the box height, except the aspect ratio, which has fixed ones.
const POSITION_STD_WEIGHT: f32 = 1.0 / 20.0;
const VELOCITY_STD_WEIGHT: f32 = 1.0 / 160.0;
const INITIAL_POSITION_STD_FACTOR: f32 = 2.0;
const INITIAL_VELOCITY_STD_FACTOR: f32 = 10.0;
const ASPECT_POSITION_STD: f32 = 1e-2;
const ASPECT_VELOCITY_STD: f32 = 1e-5;
const ASPECT_MEASUREMENT_STD: f32 = 1e-1;

// Filter axes, in the order of `measurement`.
const AXES: usize = 4;
const ASPECT: usize = 2;
const HEIGHT: usize = 3;

static TRACKER_PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new(
        "track-high-thresh",
        PropKind::Double,
        "confidence a detection needs for the first association",
    )
    .with_range("0", "1")
    .with_default("0.25"),
    PropertySpec::new(
        "track-low-thresh",
        PropKind::Double,
        "confidence a detection must exceed for the second association",
    )
    .with_range("0", "1")
    .with_default("0.1"),
    PropertySpec::new(
        "new-track-thresh",
        PropKind::Double,
        "confidence an unmatched detection needs to start a track",
    )
    .with_range("0", "1")
    .with_default("0.25"),
    PropertySpec::new(
        "match-thresh",
        PropKind::Double,
        "highest 1 - IoU x confidence cost the first association accepts",
    )
    .with_range("0", "1")
    .with_default("0.8"),
    PropertySpec::new(
        "track-buffer",
        PropKind::Uint,
        "frames a lost track keeps its id before it is dropped",
    )
    .with_default("30"),
];

// A ByteTrack multi-object tracker over the `ObjectDetection` nodes of each
// frame's `AnalyticsMeta`, writing a `Tracking` node and a `Tracks` relation for
// every detection on a confirmed track.
#[derive(Debug)]
pub struct Tracker {
    track_high_thresh: f64,
    track_low_thresh: f64,
    new_track_thresh: f64,
    match_thresh: f64,
    track_buffer: u64,
    frame: u64,
    last_id: u64,
    tracks: Vec<Track>,
    tentative: Vec<BoxFilter>,
}

#[derive(Debug)]
struct Track {
    id: u64,
    filter: BoxFilter,
    first_frame: u64,
    last_matched_frame: u64,
}

impl Track {
    fn age(&self) -> u64 {
        self.last_matched_frame - self.first_frame
    }
}

// One state axis and its velocity. The constant-velocity model never couples two
// axes, so the 8x8 covariance of the full filter is four of these 2x2 blocks.
#[derive(Debug, Clone, Copy)]
struct AxisFilter {
    position: f32,
    velocity: f32,
    position_variance: f32,
    covariance: f32,
    velocity_variance: f32,
}

impl AxisFilter {
    fn predict(&mut self, position_std: f32, velocity_std: f32) {
        self.position += self.velocity;
        self.position_variance +=
            2.0 * self.covariance + self.velocity_variance + position_std * position_std;
        self.covariance += self.velocity_variance;
        self.velocity_variance += velocity_std * velocity_std;
    }

    fn update(&mut self, measured: f32, measurement_std: f32) {
        let innovation_variance = self.position_variance + measurement_std * measurement_std;
        let position_gain = self.position_variance / innovation_variance;
        let velocity_gain = self.covariance / innovation_variance;
        let innovation = measured - self.position;
        self.position += position_gain * innovation;
        self.velocity += velocity_gain * innovation;
        self.velocity_variance -= velocity_gain * self.covariance;
        self.position_variance -= position_gain * self.position_variance;
        self.covariance -= position_gain * self.covariance;
    }
}

// Constant-velocity Kalman filter over box centre, aspect ratio and height.
#[derive(Debug, Clone, Copy)]
struct BoxFilter {
    axes: [AxisFilter; AXES],
}

impl BoxFilter {
    fn new(bbox: BBox) -> Self {
        let measured = measurement(bbox);
        let height = measured[HEIGHT];
        let axes = core::array::from_fn(|axis| {
            let (position_std, velocity_std) = if axis == ASPECT {
                process_std(axis, height)
            } else {
                let (position_std, velocity_std) = process_std(axis, height);
                (
                    INITIAL_POSITION_STD_FACTOR * position_std,
                    INITIAL_VELOCITY_STD_FACTOR * velocity_std,
                )
            };
            AxisFilter {
                position: measured[axis],
                velocity: 0.0,
                position_variance: position_std * position_std,
                covariance: 0.0,
                velocity_variance: velocity_std * velocity_std,
            }
        });
        Self { axes }
    }

    // A lost track stops growing or shrinking, as in ByteTrack.
    fn predict(&mut self, tracked_last_frame: bool) {
        if !tracked_last_frame {
            self.axes[HEIGHT].velocity = 0.0;
        }
        let height = self.axes[HEIGHT].position;
        for (axis, filter) in self.axes.iter_mut().enumerate() {
            let (position_std, velocity_std) = process_std(axis, height);
            filter.predict(position_std, velocity_std);
        }
    }

    fn update(&mut self, bbox: BBox) {
        let measured = measurement(bbox);
        let height = self.axes[HEIGHT].position;
        for (axis, filter) in self.axes.iter_mut().enumerate() {
            let measurement_std = if axis == ASPECT {
                ASPECT_MEASUREMENT_STD
            } else {
                POSITION_STD_WEIGHT * height
            };
            filter.update(measured[axis], measurement_std);
        }
    }

    fn bbox(&self) -> BBox {
        let [centre_x, centre_y, aspect, height] = self.axes.map(|axis| axis.position);
        let width = aspect * height;
        BBox {
            x: centre_x - width / 2.0,
            y: centre_y - height / 2.0,
            w: width,
            h: height,
        }
    }
}

fn measurement(bbox: BBox) -> [f32; AXES] {
    [
        bbox.x + bbox.w / 2.0,
        bbox.y + bbox.h / 2.0,
        bbox.w / bbox.h,
        bbox.h,
    ]
}

fn process_std(axis: usize, height: f32) -> (f32, f32) {
    if axis == ASPECT {
        (ASPECT_POSITION_STD, ASPECT_VELOCITY_STD)
    } else {
        (POSITION_STD_WEIGHT * height, VELOCITY_STD_WEIGHT * height)
    }
}

// The filter divides by the height, so a flat or non-finite box cannot start or feed a track.
fn is_trackable(bbox: &BBox) -> bool {
    let finite = [bbox.x, bbox.y, bbox.w, bbox.h]
        .iter()
        .all(|value| value.is_finite());
    finite && bbox.w > 0.0 && bbox.h > 0.0
}

fn iou_similarity(track_box: &BBox, detection: &ObjectDetection) -> f64 {
    f64::from(track_box.iou(&detection.bbox))
}

fn score_fused_similarity(track_box: &BBox, detection: &ObjectDetection) -> f64 {
    iou_similarity(track_box, detection) * f64::from(detection.confidence)
}

// Minimum-cost pairs of `track_boxes` and `detections` on the cost `1 - similarity`,
// as (track box index, detection index) pairs.
fn associate(
    track_boxes: &[BBox],
    detections: &[ObjectDetection],
    similarity: fn(&BBox, &ObjectDetection) -> f64,
    cost_limit: f64,
) -> Vec<(usize, usize)> {
    let costs: Vec<f64> = track_boxes
        .iter()
        .flat_map(|track_box| {
            detections
                .iter()
                .map(move |detection| 1.0 - similarity(track_box, detection))
        })
        .collect();
    assign(&costs, track_boxes.len(), detections.len(), cost_limit)
}

// Kuhn-Munkres on a `rows` x `columns` row-major cost matrix, extended the way
// lapjv's `cost_limit` does: leaving a row and a column both unmatched costs
// `cost_limit`, so only pairs cheaper than that are worth making.
fn assign(costs: &[f64], rows: usize, columns: usize, cost_limit: f64) -> Vec<(usize, usize)> {
    let size = rows + columns;
    let cell = |row: usize, column: usize| match (row < rows, column < columns) {
        (true, true) => costs[row * columns + column],
        (false, false) => 0.0,
        _ => cost_limit / 2.0,
    };
    // Index 0 is a sentinel and row / column `k` of the matrix is index `k + 1`.
    let mut row_potential = vec![0.0; size + 1];
    let mut column_potential = vec![0.0; size + 1];
    let mut row_of_column = vec![0usize; size + 1];
    let mut previous_column = vec![0usize; size + 1];
    for row in 1..=size {
        row_of_column[0] = row;
        let mut column = 0;
        let mut slack = vec![f64::INFINITY; size + 1];
        let mut visited = vec![false; size + 1];
        loop {
            visited[column] = true;
            let current_row = row_of_column[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0;
            for candidate in 1..=size {
                if visited[candidate] {
                    continue;
                }
                let reduced = cell(current_row - 1, candidate - 1)
                    - row_potential[current_row]
                    - column_potential[candidate];
                if reduced < slack[candidate] {
                    slack[candidate] = reduced;
                    previous_column[candidate] = column;
                }
                if slack[candidate] < delta {
                    delta = slack[candidate];
                    next_column = candidate;
                }
            }
            for candidate in 0..=size {
                if visited[candidate] {
                    row_potential[row_of_column[candidate]] += delta;
                    column_potential[candidate] -= delta;
                } else {
                    slack[candidate] -= delta;
                }
            }
            column = next_column;
            if row_of_column[column] == 0 {
                break;
            }
        }
        while column != 0 {
            let previous = previous_column[column];
            row_of_column[column] = row_of_column[previous];
            column = previous;
        }
    }
    (1..=size)
        .map(|column| (row_of_column[column] - 1, column - 1))
        .filter(|&(row, column)| row < rows && column < columns)
        .collect()
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            track_high_thresh: TRACK_HIGH_THRESH_DEFAULT,
            track_low_thresh: TRACK_LOW_THRESH_DEFAULT,
            new_track_thresh: NEW_TRACK_THRESH_DEFAULT,
            match_thresh: MATCH_THRESH_DEFAULT,
            track_buffer: TRACK_BUFFER_DEFAULT,
            frame: 0,
            last_id: 0,
            tracks: Vec::new(),
            tentative: Vec::new(),
        }
    }

    fn next_id(&mut self) -> u64 {
        self.last_id += 1;
        self.last_id
    }

    // One ByteTrack step: the id of the confirmed track each detection belongs to.
    fn assign_ids(&mut self, detections: &[ObjectDetection]) -> Vec<Option<u64>> {
        self.frame += 1;
        let frame = self.frame;
        let mut high = Vec::new();
        let mut low = Vec::new();
        for (index, detection) in detections.iter().enumerate() {
            if !is_trackable(&detection.bbox) {
                continue;
            }
            let confidence = f64::from(detection.confidence);
            if confidence >= self.track_high_thresh {
                high.push(index);
            } else if confidence > self.track_low_thresh {
                low.push(index);
            }
        }
        let pick = |indices: &[usize]| -> Vec<ObjectDetection> {
            indices.iter().map(|&index| detections[index]).collect()
        };

        let tracked_last_frame: Vec<bool> = self
            .tracks
            .iter()
            .map(|track| track.last_matched_frame + 1 == frame)
            .collect();
        for (track, &tracked) in self.tracks.iter_mut().zip(&tracked_last_frame) {
            track.filter.predict(tracked);
        }
        // (track id, detection index) for every detection on a confirmed track.
        let mut matches = Vec::new();

        // First association: every confirmed track, lost ones included, against
        // the high band.
        let track_boxes: Vec<BBox> = self.tracks.iter().map(|t| t.filter.bbox()).collect();
        let pairs = associate(
            &track_boxes,
            &pick(&high),
            score_fused_similarity,
            self.match_thresh,
        );
        let mut track_matched = vec![false; self.tracks.len()];
        let mut high_matched = vec![false; high.len()];
        for (track_index, candidate) in pairs {
            let detection = high[candidate];
            self.match_track(track_index, detections[detection].bbox);
            matches.push((self.tracks[track_index].id, detection));
            track_matched[track_index] = true;
            high_matched[candidate] = true;
        }

        // Second association: the tracks still unmatched that were tracked last
        // frame, against the low band.
        let retry: Vec<usize> = (0..self.tracks.len())
            .filter(|&index| !track_matched[index] && tracked_last_frame[index])
            .collect();
        let retry_boxes: Vec<BBox> = retry.iter().map(|&index| track_boxes[index]).collect();
        let pairs = associate(
            &retry_boxes,
            &pick(&low),
            iou_similarity,
            LOW_BAND_MATCH_THRESH,
        );
        for (retry_index, candidate) in pairs {
            let track_index = retry[retry_index];
            let detection = low[candidate];
            self.match_track(track_index, detections[detection].bbox);
            matches.push((self.tracks[track_index].id, detection));
        }

        // Last frame's new tracks are confirmed by a second match or dropped.
        let leftover: Vec<usize> = high
            .iter()
            .zip(&high_matched)
            .filter(|(_, &matched)| !matched)
            .map(|(&index, _)| index)
            .collect();
        let tentative = core::mem::take(&mut self.tentative);
        let tentative_boxes: Vec<BBox> = tentative.iter().map(BoxFilter::bbox).collect();
        let pairs = associate(
            &tentative_boxes,
            &pick(&leftover),
            score_fused_similarity,
            TENTATIVE_MATCH_THRESH,
        );
        let mut leftover_matched = vec![false; leftover.len()];
        for (tentative_index, candidate) in pairs {
            let detection = leftover[candidate];
            let mut filter = tentative[tentative_index];
            filter.update(detections[detection].bbox);
            let id = self.next_id();
            self.tracks.push(Track {
                id,
                filter,
                first_frame: frame - 1,
                last_matched_frame: frame,
            });
            matches.push((id, detection));
            leftover_matched[candidate] = true;
        }

        // ByteTrack confirms the tracks of the first frame at once, there being
        // no earlier frame to have seen them in.
        let new_track_thresh = self.new_track_thresh;
        let new_detections = leftover
            .iter()
            .zip(&leftover_matched)
            .filter(|(_, &matched)| !matched)
            .map(|(&index, _)| index)
            .filter(|&index| f64::from(detections[index].confidence) >= new_track_thresh);
        for detection in new_detections {
            let filter = BoxFilter::new(detections[detection].bbox);
            if frame == 1 {
                let id = self.next_id();
                self.tracks.push(Track {
                    id,
                    filter,
                    first_frame: frame,
                    last_matched_frame: frame,
                });
                matches.push((id, detection));
            } else {
                self.tentative.push(filter);
            }
        }

        let track_buffer = self.track_buffer;
        self.tracks
            .retain(|track| frame - track.last_matched_frame <= track_buffer);
        self.remove_duplicates();

        let mut ids = vec![None; detections.len()];
        for (id, detection) in matches {
            if self.tracks.iter().any(|track| track.id == id) {
                ids[detection] = Some(id);
            }
        }
        ids
    }

    fn match_track(&mut self, track_index: usize, bbox: BBox) {
        let track = &mut self.tracks[track_index];
        track.filter.update(bbox);
        track.last_matched_frame = self.frame;
    }

    // Of a tracked and a lost track on one object, keep the one tracked longer.
    // On a tie the tracked one goes, and a new track always loses, as in ByteTrack.
    fn remove_duplicates(&mut self) {
        let frame = self.frame;
        let lost_boxes: Vec<BBox> = self
            .tracks
            .iter()
            .filter(|track| track.last_matched_frame != frame)
            .map(|track| track.filter.bbox())
            .collect();
        let overlaps_lost =
            |bbox: BBox| lost_boxes.iter().any(|lost| bbox.iou(lost) > DUPLICATE_IOU);
        self.tentative
            .retain(|filter| !overlaps_lost(filter.bbox()));

        let mut duplicate = vec![false; self.tracks.len()];
        for (tracked_index, tracked) in self.tracks.iter().enumerate() {
            if tracked.last_matched_frame != frame {
                continue;
            }
            for (lost_index, lost) in self.tracks.iter().enumerate() {
                if lost.last_matched_frame == frame
                    || tracked.filter.bbox().iou(&lost.filter.bbox()) <= DUPLICATE_IOU
                {
                    continue;
                }
                if tracked.age() > lost.age() {
                    duplicate[lost_index] = true;
                } else {
                    duplicate[tracked_index] = true;
                }
            }
        }
        let tracks = core::mem::take(&mut self.tracks);
        self.tracks = tracks
            .into_iter()
            .zip(duplicate)
            .filter(|(_, duplicate)| !duplicate)
            .map(|(track, _)| track)
            .collect();
    }

    fn annotate(&mut self, meta: &mut AnalyticsMeta) {
        let mut detection_nodes = Vec::new();
        let mut detections = Vec::new();
        for (node, entry) in meta.nodes.iter().enumerate() {
            if let AnalyticsNode::Detection(detection) = entry {
                detection_nodes.push(node);
                detections.push(*detection);
            }
        }
        let ids = self.assign_ids(&detections);
        for (node, id) in detection_nodes.into_iter().zip(ids) {
            let Some(object_id) = id else {
                continue;
            };
            let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id }));
            meta.relate(node, tracking, RelationKind::Tracks);
        }
    }
}

impl AsyncElement for Tracker {
    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Tracker",
            "Filter/Analytics",
            "Gives each detection a persistent track id across frames (ByteTrack)",
            "g2g",
        )
    }
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

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
        TRACKER_PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "track-high-thresh" => self.track_high_thresh = double_in_range(value, 0.0, 1.0)?,
            "track-low-thresh" => self.track_low_thresh = double_in_range(value, 0.0, 1.0)?,
            "new-track-thresh" => self.new_track_thresh = double_in_range(value, 0.0, 1.0)?,
            "match-thresh" => self.match_thresh = double_in_range(value, 0.0, 1.0)?,
            "track-buffer" => self.track_buffer = value.as_uint().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "track-high-thresh" => Some(PropValue::Double(self.track_high_thresh)),
            "track-low-thresh" => Some(PropValue::Double(self.track_low_thresh)),
            "new-track-thresh" => Some(PropValue::Double(self.new_track_thresh)),
            "match-thresh" => Some(PropValue::Double(self.match_thresh)),
            "track-buffer" => Some(PropValue::Uint(self.track_buffer)),
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
                        self.annotate(meta);
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // The runner's transform arm forwards EOS.
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use g2g_core::frame::Frame;
    use g2g_core::memory::SystemSlice;
    use g2g_core::{Classification, FrameTiming, MemoryDomain, PushOutcome};

    const SIDE: f32 = 0.1;
    const CONFIDENT: f32 = 0.9;
    // Inside the default low band, between `track-low-thresh` and `track-high-thresh`.
    const LOW_BAND: f32 = 0.15;

    fn detection(x: f32, y: f32, confidence: f32) -> ObjectDetection {
        ObjectDetection {
            bbox: BBox {
                x,
                y,
                w: SIDE,
                h: SIDE,
            },
            label: 0,
            confidence,
        }
    }

    fn tracker_with(name: &str, value: PropValue) -> Tracker {
        let mut tracker = Tracker::new();
        tracker.set_property(name, value).unwrap();
        tracker
    }

    // A steady box for `frames` frames, returning the id it got on each.
    fn hold(tracker: &mut Tracker, frames: usize, confidence: f32) -> Vec<Option<u64>> {
        (0..frames)
            .map(|_| tracker.assign_ids(&[detection(0.4, 0.4, confidence)])[0])
            .collect()
    }

    fn gap(tracker: &mut Tracker, frames: u64) {
        for _ in 0..frames {
            tracker.assign_ids(&[]);
        }
    }

    #[test]
    fn a_box_moving_in_a_line_keeps_one_id() {
        let mut tracker = Tracker::new();
        let step = 0.01;
        for frame in 0..80 {
            let x = 0.05 + step * frame as f32;
            let ids = tracker.assign_ids(&[detection(x, 0.3, CONFIDENT)]);
            assert_eq!(ids, [Some(1)], "frame {frame}");
        }
    }

    #[test]
    fn crossing_boxes_keep_their_ids() {
        // Each box moves 0.04 a frame. At frame 9 the box coming from the left
        // overlaps the other box's frame-8 position more than its own, so matching
        // on last positions swaps the ids, while the predicted positions do not.
        let mut tracker = Tracker::new();
        let speed = 0.04;
        for frame in 0..20 {
            let travelled = speed * frame as f32;
            let from_left = detection(0.1 + travelled, 0.4, CONFIDENT);
            let from_right = detection(0.8 - travelled, 0.4, CONFIDENT);
            let ids = tracker.assign_ids(&[from_left, from_right]);
            assert_eq!(ids, [Some(1), Some(2)], "frame {frame}");
        }
    }

    #[test]
    fn a_gap_within_track_buffer_resumes_the_id_and_a_longer_one_does_not() {
        let buffer = 5;
        let mut tracker = tracker_with("track-buffer", PropValue::Uint(buffer));
        assert_eq!(hold(&mut tracker, 3, CONFIDENT), [Some(1); 3]);
        gap(&mut tracker, buffer);
        assert_eq!(hold(&mut tracker, 2, CONFIDENT), [Some(1); 2], "resumed");

        gap(&mut tracker, buffer + 1);
        assert_eq!(
            hold(&mut tracker, 2, CONFIDENT),
            [None, Some(2)],
            "a new track, confirmed on its second frame"
        );
    }

    #[test]
    fn a_low_band_detection_keeps_a_track_alive_but_never_starts_one() {
        let mut tracker = Tracker::new();
        assert_eq!(hold(&mut tracker, 3, CONFIDENT), [Some(1); 3]);
        assert_eq!(hold(&mut tracker, 40, LOW_BAND), [Some(1); 40]);

        let mut tracker = Tracker::new();
        assert_eq!(hold(&mut tracker, 5, LOW_BAND), [None; 5]);
    }

    #[test]
    fn each_threshold_property_changes_the_outcome() {
        let mut default = Tracker::new();
        assert_eq!(hold(&mut default, 2, 0.5), [Some(1); 2]);

        let mut strict_start = tracker_with("new-track-thresh", PropValue::Double(0.6));
        assert_eq!(hold(&mut strict_start, 3, 0.5), [None; 3]);

        // Raising the high band puts 0.5 in the low band, which cannot start a track.
        let mut strict_high = tracker_with("track-high-thresh", PropValue::Double(0.6));
        assert_eq!(hold(&mut strict_high, 3, 0.5), [None; 3]);

        // Raising the low band leaves the low-band detection with no track to keep alive.
        let mut strict_low = tracker_with("track-low-thresh", PropValue::Double(0.2));
        assert_eq!(hold(&mut strict_low, 2, CONFIDENT), [Some(1); 2]);
        assert_eq!(hold(&mut strict_low, 1, LOW_BAND), [None]);

        // A box moving half its width a frame matches at the default cost limit only.
        let moving = |tracker: &mut Tracker| -> Vec<Option<u64>> {
            (0..4)
                .map(|frame| {
                    let x = 0.1 + SIDE / 2.0 * frame as f32;
                    tracker.assign_ids(&[detection(x, 0.4, CONFIDENT)])[0]
                })
                .collect()
        };
        assert_eq!(moving(&mut Tracker::new()), [Some(1); 4]);
        let mut strict_match = tracker_with("match-thresh", PropValue::Double(0.1));
        assert_ne!(moving(&mut strict_match), [Some(1); 4]);
    }

    #[test]
    fn assign_leaves_pairs_over_the_cost_limit_unmatched() {
        // Both rows are cheapest on column 1. Under a limit of 0.5 the second row
        // stays unmatched rather than take column 0 at 0.95.
        let costs = [0.9, 0.1, 0.95, 0.2];
        assert_eq!(assign(&costs, 2, 2, 0.5), [(0, 1)]);
        assert_eq!(assign(&costs, 2, 2, 1.0), [(1, 0), (0, 1)]);
    }

    #[derive(Default)]
    struct FrameSink {
        frames: Vec<Frame>,
    }

    impl OutputSink for FrameSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            if let Some(PipelinePacket::DataFrame(frame)) = packet_slot.take() {
                self.frames.push(frame);
            }
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn frame(meta: Option<AnalyticsMeta>) -> Frame {
        let mut frame = Frame::new(
            MemoryDomain::System(SystemSlice::from_boxed(vec![7u8; 16].into_boxed_slice())),
            FrameTiming::default(),
            0,
        );
        if let Some(meta) = meta {
            frame.meta.attach(meta);
        }
        frame
    }

    #[tokio::test]
    async fn process_passes_bare_frames_and_wires_one_tracking_node_per_track() {
        let mut tracker = Tracker::new();
        let mut sink = FrameSink::default();
        tracker
            .process(PipelinePacket::DataFrame(frame(None)), &mut sink)
            .await
            .unwrap();
        let bare = &sink.frames[0];
        assert!(bare.meta.is_empty(), "no meta added to a bare frame");
        assert_eq!(
            bare.domain.as_system_slice().unwrap().to_vec(),
            vec![7u8; 16]
        );

        let mut meta = AnalyticsMeta::new();
        meta.push(AnalyticsNode::Classification(Classification {
            label: 3,
            confidence: 0.5,
        }));
        let first = meta.add_detection(detection(0.1, 0.1, CONFIDENT));
        let second = meta.add_detection(detection(0.6, 0.6, CONFIDENT));
        meta.add_detection(detection(0.3, 0.3, LOW_BAND));
        let nodes_before = meta.nodes.len();
        tracker
            .process(PipelinePacket::DataFrame(frame(Some(meta))), &mut sink)
            .await
            .unwrap();

        let meta = sink.frames[1].meta.get::<AnalyticsMeta>().unwrap();
        let ids: Vec<(usize, u64)> = meta
            .relations
            .iter()
            .map(|relation| {
                assert_eq!(relation.kind, RelationKind::Tracks);
                let AnalyticsNode::Tracking(tracking) = &meta.nodes[relation.to] else {
                    panic!("a Tracks relation points at a Tracking node");
                };
                (relation.from, tracking.object_id)
            })
            .collect();
        assert_eq!(ids, [(first, 1), (second, 2)]);
        assert_eq!(meta.nodes.len(), nodes_before + 2, "one node per track");
    }
}
