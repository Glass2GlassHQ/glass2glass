use core::ffi::{c_char, c_int, CStr};
use core::ptr;
use core::slice;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;

use g2g_core::meta::{
    AnalyticsMeta, AnalyticsNode, BBox, Classification, Mask, ObjectDetection, RelationKind,
    Segmentation, Tracking,
};

const MTD_DETECTION: c_int = 0;
const MTD_CLASSIFICATION: c_int = 1;
const MTD_TRACKING: c_int = 2;
const MTD_SEGMENTATION: c_int = 3;

// GstAnalyticsRelTypes bits from gstanalyticsmeta.h
const REL_IS_PART_OF: u32 = 1 << 1;
const REL_CONTAIN: u32 = 1 << 2;
const REL_RELATE_TO: u32 = 1 << 3;
const REL_N_TO_N: u32 = 1 << 4;

const MAX_NODES: usize = 4096;
const MAX_RELATIONS: usize = 16384;
const MAX_MASK_BYTES: usize = 64 << 20;

const GRAY8_SAMPLE_BYTES: usize = 1;
const GRAY16_SAMPLE_BYTES: usize = 2;

// GStreamer's value for a confidence the producer left unset
const UNDEFINED_CONFIDENCE: f32 = -1.0;
const UNLABELLED: &str = "";

const COVERED: u8 = u8::MAX;
const COVERED_THRESHOLD: u8 = 128;

// 0 in a GStreamer mask means no region
const REGION_ID: u8 = 1;
static REGION_IDS: [u32; 1] = [REGION_ID as u32];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct G2gAnalyticsMtd {
    kind: c_int,
    id: u32,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    confidence: f32,
    label: *const c_char,
    tracking_id: u64,
    mask: *const u8,
    mask_len: usize,
    mask_width: u32,
    mask_height: u32,
    mask_stride: i32,
    mask_sample_bytes: u32,
    region_ids: *const u32,
    region_count: usize,
}

impl Default for G2gAnalyticsMtd {
    fn default() -> Self {
        G2gAnalyticsMtd {
            kind: MTD_DETECTION,
            id: 0,
            x: 0,
            y: 0,
            w: 0,
            h: 0,
            confidence: 0.0,
            label: ptr::null(),
            tracking_id: 0,
            mask: ptr::null(),
            mask_len: 0,
            mask_width: 0,
            mask_height: 0,
            mask_stride: 0,
            mask_sample_bytes: 0,
            region_ids: ptr::null(),
            region_count: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct G2gAnalyticsRelation {
    from: u32,
    to: u32,
    kind: u32,
}

#[repr(C)]
#[derive(Debug)]
pub struct G2gAnalyticsPlan {
    mtds: *const G2gAnalyticsMtd,
    mtd_count: usize,
    relations: *const G2gAnalyticsRelation,
    relation_count: usize,
}

impl Default for G2gAnalyticsPlan {
    fn default() -> Self {
        G2gAnalyticsPlan {
            mtds: ptr::null(),
            mtd_count: 0,
            relations: ptr::null(),
            relation_count: 0,
        }
    }
}

#[derive(Debug)]
enum ReadMtd {
    Detection(u32, ObjectDetection),
    Class(u32, Classification),
    Tracking(u32, Tracking),
    Segmentation(u32, BBox, Vec<Mask>),
}

impl ReadMtd {
    fn id(&self) -> u32 {
        match self {
            ReadMtd::Detection(id, _)
            | ReadMtd::Class(id, _)
            | ReadMtd::Tracking(id, _)
            | ReadMtd::Segmentation(id, _, _) => *id,
        }
    }

    fn kind(&self) -> c_int {
        match self {
            ReadMtd::Detection(..) => MTD_DETECTION,
            ReadMtd::Class(..) => MTD_CLASSIFICATION,
            ReadMtd::Tracking(..) => MTD_TRACKING,
            ReadMtd::Segmentation(..) => MTD_SEGMENTATION,
        }
    }
}

#[derive(Debug, Default)]
struct ClassNames {
    names: Vec<Box<str>>,
    index: BTreeMap<Box<str>, u32>,
}

impl ClassNames {
    fn intern(&mut self, name: &str) -> u32 {
        if let Some(&label) = self.index.get(name) {
            return label;
        }
        let label = self.names.len() as u32;
        self.names.push(name.into());
        self.index.insert(name.into(), label);
        label
    }

    // `name` must be null or a valid C string
    unsafe fn intern_c(&mut self, name: *const c_char) -> u32 {
        if name.is_null() {
            return self.intern(UNLABELLED);
        }
        // SAFETY: caller contract on `name`.
        let name = unsafe { CStr::from_ptr(name) };
        self.intern(&name.to_string_lossy())
    }
}

#[derive(Debug, Default)]
pub(crate) struct GstAnalyticsReader {
    mtds: Vec<ReadMtd>,
    relations: Vec<G2gAnalyticsRelation>,
    class_names: ClassNames,
    node_count: usize,
    mask_bytes: usize,
}

impl GstAnalyticsReader {
    // every pointer in `mtd` must be null or valid for its length during this call
    pub(crate) unsafe fn read(
        &mut self,
        mtd: &G2gAnalyticsMtd,
        frame_width: u32,
        frame_height: u32,
    ) {
        if self.node_count >= MAX_NODES || frame_width == 0 || frame_height == 0 {
            return;
        }
        let bbox = BBox {
            x: mtd.x as f32 / frame_width as f32,
            y: mtd.y as f32 / frame_height as f32,
            w: mtd.w as f32 / frame_width as f32,
            h: mtd.h as f32 / frame_height as f32,
        };
        let read = match mtd.kind {
            MTD_DETECTION => ReadMtd::Detection(
                mtd.id,
                ObjectDetection {
                    bbox,
                    // SAFETY: caller contract on `label`.
                    label: unsafe { self.class_names.intern_c(mtd.label) },
                    confidence: mtd.confidence,
                },
            ),
            MTD_CLASSIFICATION => ReadMtd::Class(
                mtd.id,
                Classification {
                    // SAFETY: caller contract on `label`.
                    label: unsafe { self.class_names.intern_c(mtd.label) },
                    confidence: mtd.confidence,
                },
            ),
            MTD_TRACKING => ReadMtd::Tracking(
                mtd.id,
                Tracking {
                    object_id: mtd.tracking_id,
                },
            ),
            MTD_SEGMENTATION => {
                // SAFETY: caller contract on `mask` and `region_ids`.
                let Some(masks) = (unsafe { self.region_masks(mtd) }) else {
                    return;
                };
                ReadMtd::Segmentation(mtd.id, bbox, masks)
            }
            _ => return,
        };
        self.node_count += match &read {
            ReadMtd::Segmentation(_, _, masks) => masks.len(),
            _ => 1,
        };
        self.mtds.push(read);
    }

    pub(crate) fn relate(&mut self, from: u32, to: u32, kind: u32) {
        if self.relations.len() < MAX_RELATIONS {
            self.relations.push(G2gAnalyticsRelation { from, to, kind });
        }
    }

    // same pointer contract as `read`
    unsafe fn region_masks(&mut self, mtd: &G2gAnalyticsMtd) -> Option<Vec<Mask>> {
        let sample_bytes = usize::try_from(mtd.mask_sample_bytes).ok()?;
        if !matches!(sample_bytes, GRAY8_SAMPLE_BYTES | GRAY16_SAMPLE_BYTES)
            || mtd.mask.is_null()
            || mtd.region_ids.is_null()
        {
            return None;
        }
        let width = usize::try_from(mtd.mask_width).ok()?;
        let height = usize::try_from(mtd.mask_height).ok()?;
        let stride = usize::try_from(mtd.mask_stride).ok()?;
        let row_bytes = width.checked_mul(sample_bytes)?;
        let pixels = width.checked_mul(height)?;
        let mask_bytes = pixels.checked_mul(mtd.region_count)?;
        if stride < row_bytes
            || pixels > mtd.mask_len
            || mtd.region_count > MAX_NODES - self.node_count
            || mask_bytes > MAX_MASK_BYTES - self.mask_bytes
        {
            return None;
        }
        // SAFETY: caller contract: `mask_len` readable bytes at `mask`.
        let image = unsafe { slice::from_raw_parts(mtd.mask, mtd.mask_len) };
        // SAFETY: caller contract: `region_count` values at `region_ids`.
        let region_ids = unsafe { slice::from_raw_parts(mtd.region_ids, mtd.region_count) };
        let mut samples = Vec::with_capacity(pixels);
        for row in 0..height {
            let start = row.checked_mul(stride)?;
            let row = image.get(start..start.checked_add(row_bytes)?)?;
            samples.extend(row.chunks_exact(sample_bytes).map(|sample| match sample {
                [low, high] => u32::from(u16::from_le_bytes([*low, *high])),
                [value] => u32::from(*value),
                _ => 0,
            }));
        }
        self.mask_bytes += mask_bytes;
        region_ids
            .iter()
            .map(|&region| {
                let coverage = samples
                    .iter()
                    .map(|&sample| if sample == region { COVERED } else { 0 })
                    .collect();
                Mask::new(mtd.mask_width, mtd.mask_height, mtd.mask_width, coverage)
            })
            .collect()
    }

    pub(crate) fn finish(self) -> Option<AnalyticsMeta> {
        let GstAnalyticsReader {
            mtds,
            relations,
            mut class_names,
            ..
        } = self;
        let kinds: BTreeMap<u32, c_int> = mtds.iter().map(|mtd| (mtd.id(), mtd.kind())).collect();
        let kind_of = |id: u32| kinds.get(&id).copied();

        // A segmentation names its regions through a classification related to it.
        let label_tables: BTreeMap<u32, u32> = relations
            .iter()
            .filter(|relation| relation.kind & (REL_N_TO_N | REL_RELATE_TO) != 0)
            .filter(|relation| {
                kind_of(relation.from) == Some(MTD_SEGMENTATION)
                    && kind_of(relation.to) == Some(MTD_CLASSIFICATION)
            })
            .map(|relation| (relation.from, relation.to))
            .collect();
        let table_ids: BTreeSet<u32> = label_tables.values().copied().collect();
        let mut classes: BTreeMap<u32, Vec<Classification>> = BTreeMap::new();
        for mtd in &mtds {
            if let ReadMtd::Class(id, class) = mtd {
                classes.entry(*id).or_default().push(*class);
            }
        }

        let mut meta = AnalyticsMeta::new();
        let mut nodes: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for mtd in mtds {
            let id = mtd.id();
            match mtd {
                ReadMtd::Detection(_, detection) => {
                    let node = meta.add_detection(detection);
                    nodes.entry(id).or_default().push(node);
                }
                ReadMtd::Class(_, class) if !table_ids.contains(&id) => {
                    let node = meta.push(AnalyticsNode::Classification(class));
                    nodes.entry(id).or_default().push(node);
                }
                ReadMtd::Class(..) => {}
                ReadMtd::Tracking(_, tracking) => {
                    let node = meta.push(AnalyticsNode::Tracking(tracking));
                    nodes.entry(id).or_default().push(node);
                }
                ReadMtd::Segmentation(_, bbox, masks) => {
                    let labels = label_tables.get(&id).and_then(|table| classes.get(table));
                    for (region, mask) in masks.into_iter().enumerate() {
                        let class = labels.and_then(|labels| labels.get(region)).copied();
                        let class = class.unwrap_or_else(|| Classification {
                            label: class_names.intern(UNLABELLED),
                            confidence: UNDEFINED_CONFIDENCE,
                        });
                        let node = meta.push(AnalyticsNode::Segmentation(Segmentation {
                            bbox,
                            label: class.label,
                            confidence: class.confidence,
                            mask,
                        }));
                        nodes.entry(id).or_default().push(node);
                    }
                }
            }
        }

        // IS_PART_OF is always written as the inverse of a CONTAIN, so only CONTAIN is read.
        for relation in &relations {
            let pair = (kind_of(relation.from), kind_of(relation.to));
            let kind = if relation.kind & REL_CONTAIN != 0 {
                match pair {
                    (Some(MTD_DETECTION), Some(MTD_CLASSIFICATION)) => RelationKind::Classifies,
                    _ => RelationKind::Contains,
                }
            } else if relation.kind & REL_RELATE_TO != 0
                && pair == (Some(MTD_DETECTION), Some(MTD_TRACKING))
            {
                RelationKind::Tracks
            } else {
                continue;
            };
            let (Some(from), Some(to)) = (nodes.get(&relation.from), nodes.get(&relation.to))
            else {
                continue;
            };
            for &from in from {
                for &to in to {
                    if meta.relations.len() < MAX_RELATIONS {
                        meta.relate(from, to, kind);
                    }
                }
            }
        }

        if meta.nodes.is_empty() {
            return None;
        }
        meta.set_class_names(class_names.names);
        Some(meta)
    }
}

#[derive(Debug, Default)]
pub(crate) struct GstAnalyticsWriter {
    mtds: Vec<G2gAnalyticsMtd>,
    relations: Vec<G2gAnalyticsRelation>,
    labels: Vec<CString>,
    masks: Vec<Vec<u8>>,
}

impl GstAnalyticsWriter {
    // the plan's pointers stay valid until the next call
    pub(crate) fn plan(
        &mut self,
        meta: Option<&AnalyticsMeta>,
        frame_width: u32,
        frame_height: u32,
    ) -> G2gAnalyticsPlan {
        self.mtds.clear();
        self.relations.clear();
        self.labels.clear();
        self.masks.clear();
        if let Some(meta) = meta {
            self.fill(meta, frame_width, frame_height);
        }
        G2gAnalyticsPlan {
            mtds: self.mtds.as_ptr(),
            mtd_count: self.mtds.len(),
            relations: self.relations.as_ptr(),
            relation_count: self.relations.len(),
        }
    }

    fn fill(&mut self, meta: &AnalyticsMeta, frame_width: u32, frame_height: u32) {
        let pixels = |bbox: &BBox| {
            let width = frame_width as f32;
            let height = frame_height as f32;
            (
                (bbox.x * width).round() as i32,
                (bbox.y * height).round() as i32,
                (bbox.w * width).round() as i32,
                (bbox.h * height).round() as i32,
            )
        };
        let mut mtd_of_node = Vec::with_capacity(meta.nodes.len());
        for node in &meta.nodes {
            let mtd = match node {
                AnalyticsNode::Detection(detection) => {
                    let (x, y, w, h) = pixels(&detection.bbox);
                    let label = self.label(meta, detection.label);
                    Some(self.add(G2gAnalyticsMtd {
                        kind: MTD_DETECTION,
                        x,
                        y,
                        w,
                        h,
                        confidence: detection.confidence,
                        label,
                        ..G2gAnalyticsMtd::default()
                    }))
                }
                AnalyticsNode::Classification(class) => Some(self.add_class(meta, class)),
                AnalyticsNode::Tracking(tracking) => Some(self.add(G2gAnalyticsMtd {
                    kind: MTD_TRACKING,
                    tracking_id: tracking.object_id,
                    ..G2gAnalyticsMtd::default()
                })),
                AnalyticsNode::Segmentation(segmentation) => {
                    self.add_segmentation(meta, segmentation, pixels(&segmentation.bbox))
                }
                _ => None,
            };
            mtd_of_node.push(mtd);
        }
        for relation in &meta.relations {
            let mtd = |node: usize| mtd_of_node.get(node).copied().flatten();
            let (Some(from), Some(to)) = (mtd(relation.from), mtd(relation.to)) else {
                continue;
            };
            match relation.kind {
                RelationKind::Classifies | RelationKind::Contains => {
                    self.relate(from, to, REL_CONTAIN);
                    self.relate(to, from, REL_IS_PART_OF);
                }
                RelationKind::Tracks => self.relate(from, to, REL_RELATE_TO),
                _ => {}
            }
        }
    }

    fn add(&mut self, mtd: G2gAnalyticsMtd) -> u32 {
        self.mtds.push(G2gAnalyticsMtd {
            id: self.mtds.len() as u32,
            ..mtd
        });
        self.mtds.len() as u32 - 1
    }

    fn add_class(&mut self, meta: &AnalyticsMeta, class: &Classification) -> u32 {
        let label = self.label(meta, class.label);
        self.add(G2gAnalyticsMtd {
            kind: MTD_CLASSIFICATION,
            confidence: class.confidence,
            label,
            ..G2gAnalyticsMtd::default()
        })
    }

    // A one-region GStreamer mask plus the classification that labels the region.
    fn add_segmentation(
        &mut self,
        meta: &AnalyticsMeta,
        segmentation: &Segmentation,
        (x, y, w, h): (i32, i32, i32, i32),
    ) -> Option<u32> {
        let mask = &segmentation.mask;
        let image: Vec<u8> = (0..mask.height())
            .flat_map(|row| (0..mask.width()).map(move |column| (column, row)))
            .map(|(column, row)| match mask.sample(column, row) {
                Some(coverage) if coverage >= COVERED_THRESHOLD => REGION_ID,
                _ => 0,
            })
            .collect();
        if image.is_empty() {
            return None;
        }
        let mask_stride = i32::try_from(mask.width()).ok()?;
        let segmentation_mtd = self.add(G2gAnalyticsMtd {
            kind: MTD_SEGMENTATION,
            x,
            y,
            w,
            h,
            mask: image.as_ptr(),
            mask_len: image.len(),
            mask_width: mask.width(),
            mask_height: mask.height(),
            mask_stride,
            mask_sample_bytes: GRAY8_SAMPLE_BYTES as u32,
            region_ids: REGION_IDS.as_ptr(),
            region_count: REGION_IDS.len(),
            ..G2gAnalyticsMtd::default()
        });
        self.masks.push(image);
        let class = Classification {
            label: segmentation.label,
            confidence: segmentation.confidence,
        };
        let class_mtd = self.add_class(meta, &class);
        self.relate(segmentation_mtd, class_mtd, REL_N_TO_N);
        Some(segmentation_mtd)
    }

    fn relate(&mut self, from: u32, to: u32, kind: u32) {
        self.relations.push(G2gAnalyticsRelation { from, to, kind });
    }

    // Null for an unlabelled node, else the class name, else the decimal label.
    fn label(&mut self, meta: &AnalyticsMeta, label: u32) -> *const c_char {
        let name = meta
            .class_name(label)
            .map_or_else(|| label.to_string(), str::to_owned);
        if name == UNLABELLED {
            return ptr::null();
        }
        let name = CString::new(name).unwrap_or_else(|_| {
            CString::new(label.to_string()).expect("a decimal number holds no NUL")
        });
        let pointer = name.as_ptr();
        self.labels.push(name);
        pointer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_WIDTH: u32 = 640;
    const FRAME_HEIGHT: u32 = 480;

    extern "C" {
        fn g2g_bridge_analytics_mtd_size() -> usize;
        fn g2g_bridge_analytics_relation_size() -> usize;
    }

    #[test]
    fn mtd_and_relation_match_the_c_structs() {
        // SAFETY: both return a sizeof, no arguments.
        let (mtd, relation) = unsafe {
            (
                g2g_bridge_analytics_mtd_size(),
                g2g_bridge_analytics_relation_size(),
            )
        };
        assert_eq!(mtd, core::mem::size_of::<G2gAnalyticsMtd>());
        assert_eq!(relation, core::mem::size_of::<G2gAnalyticsRelation>());
    }

    fn read_all(mtds: &[G2gAnalyticsMtd], relations: &[(u32, u32, u32)]) -> Option<AnalyticsMeta> {
        let mut reader = GstAnalyticsReader::default();
        for mtd in mtds {
            // SAFETY: every pointer in the test mtds outlives this call.
            unsafe { reader.read(mtd, FRAME_WIDTH, FRAME_HEIGHT) };
        }
        for &(from, to, kind) in relations {
            reader.relate(from, to, kind);
        }
        reader.finish()
    }

    fn label_of(meta: &AnalyticsMeta, label: u32) -> &str {
        meta.class_name(label)
            .expect("the label is in the class table")
    }

    #[test]
    fn gstreamer_mtds_become_an_analytics_graph() {
        let person = c"person";
        let adult = c"adult";
        let leaf = c"leaf";
        let (detection_id, class_id, tracking_id, segmentation_id, region_class_id) =
            (0, 1, 2, 3, 4);
        let tracked_object = 77;
        let region = 9;
        // a 3x2 GRAY8 region-id image with a padded stride
        let mask_width = 3;
        let mask_height = 2;
        let mask_stride = 4;
        let image = [region, 0, region, 0xff, 0, region, 0, 0xff];
        let region_ids = [u32::from(region)];
        let mtds = [
            G2gAnalyticsMtd {
                kind: MTD_DETECTION,
                id: detection_id,
                x: 64,
                y: 48,
                w: 320,
                h: 240,
                confidence: 0.9,
                label: person.as_ptr(),
                ..G2gAnalyticsMtd::default()
            },
            G2gAnalyticsMtd {
                kind: MTD_CLASSIFICATION,
                id: class_id,
                confidence: 0.6,
                label: adult.as_ptr(),
                ..G2gAnalyticsMtd::default()
            },
            G2gAnalyticsMtd {
                kind: MTD_TRACKING,
                id: tracking_id,
                tracking_id: tracked_object,
                ..G2gAnalyticsMtd::default()
            },
            G2gAnalyticsMtd {
                kind: MTD_SEGMENTATION,
                id: segmentation_id,
                x: 0,
                y: 0,
                w: 64,
                h: 48,
                mask: image.as_ptr(),
                mask_len: image.len(),
                mask_width,
                mask_height,
                mask_stride,
                mask_sample_bytes: GRAY8_SAMPLE_BYTES as u32,
                region_ids: region_ids.as_ptr(),
                region_count: region_ids.len(),
                ..G2gAnalyticsMtd::default()
            },
            G2gAnalyticsMtd {
                kind: MTD_CLASSIFICATION,
                id: region_class_id,
                confidence: 0.8,
                label: leaf.as_ptr(),
                ..G2gAnalyticsMtd::default()
            },
        ];
        let relations = [
            (detection_id, class_id, REL_CONTAIN),
            (class_id, detection_id, REL_IS_PART_OF),
            (detection_id, tracking_id, REL_RELATE_TO),
            (segmentation_id, region_class_id, REL_N_TO_N),
        ];
        let meta = read_all(&mtds, &relations).expect("the mtds make a graph");

        let detection = meta.detections().next().expect("one detection");
        assert_eq!(
            detection.bbox,
            BBox {
                x: 0.1,
                y: 0.1,
                w: 0.5,
                h: 0.5
            }
        );
        assert_eq!(label_of(&meta, detection.label), "person");
        assert_eq!(detection.confidence, 0.9);

        let segmentation = meta.segmentations().next().expect("one segmentation");
        assert_eq!(label_of(&meta, segmentation.label), "leaf");
        assert_eq!(segmentation.confidence, 0.8);
        assert_eq!(
            segmentation.mask.data(),
            &[COVERED, 0, COVERED, 0, COVERED, 0]
        );

        // the region's classification labels the mask instead of becoming a node
        let classes: Vec<_> = meta
            .nodes
            .iter()
            .filter_map(|node| match node {
                AnalyticsNode::Classification(class) => Some(*class),
                _ => None,
            })
            .collect();
        assert_eq!(classes.len(), 1);
        assert_eq!(label_of(&meta, classes[0].label), "adult");

        let tracking_node = meta
            .nodes
            .iter()
            .position(|node| {
                *node
                    == AnalyticsNode::Tracking(Tracking {
                        object_id: tracked_object,
                    })
            })
            .expect("a tracking node");
        let detection_node = 0;
        let class_node = 1;
        assert_eq!(meta.relations.len(), 2);
        assert!(meta.relations.contains(&g2g_core::meta::Relation {
            from: detection_node,
            to: class_node,
            kind: RelationKind::Classifies,
        }));
        assert!(meta.relations.contains(&g2g_core::meta::Relation {
            from: detection_node,
            to: tracking_node,
            kind: RelationKind::Tracks,
        }));
    }

    #[test]
    fn a_mask_past_its_buffer_is_dropped() {
        let image = [1u8; 4];
        let region_ids = [1u32];
        let mtd = G2gAnalyticsMtd {
            kind: MTD_SEGMENTATION,
            mask: image.as_ptr(),
            mask_len: image.len(),
            mask_width: 4,
            mask_height: 2,
            mask_stride: 4,
            mask_sample_bytes: GRAY8_SAMPLE_BYTES as u32,
            region_ids: region_ids.as_ptr(),
            region_count: region_ids.len(),
            ..G2gAnalyticsMtd::default()
        };
        assert!(read_all(&[mtd], &[]).is_none());
    }

    fn sample_meta() -> AnalyticsMeta {
        let mut meta = AnalyticsMeta::new();
        meta.set_class_names(["person", "car"]);
        let detection = meta.add_detection(ObjectDetection {
            bbox: BBox {
                x: 0.25,
                y: 0.5,
                w: 0.5,
                h: 0.25,
            },
            label: 1,
            confidence: 0.75,
        });
        let class = meta.push(AnalyticsNode::Classification(Classification {
            label: 0,
            confidence: 0.5,
        }));
        let tracking = meta.push(AnalyticsNode::Tracking(Tracking { object_id: 5 }));
        let mask = Mask::new(2, 2, 2, vec![COVERED, 0, 0, COVERED]).expect("a 2x2 mask");
        meta.push(AnalyticsNode::Segmentation(Segmentation {
            bbox: BBox {
                x: 0.0,
                y: 0.0,
                w: 0.5,
                h: 0.5,
            },
            label: 0,
            confidence: 0.25,
            mask,
        }));
        meta.relate(detection, class, RelationKind::Classifies);
        meta.relate(detection, tracking, RelationKind::Tracks);
        meta
    }

    fn label_str(mtd: &G2gAnalyticsMtd) -> Option<&str> {
        if mtd.label.is_null() {
            return None;
        }
        // SAFETY: the writer's labels are valid C strings until its next plan.
        unsafe { CStr::from_ptr(mtd.label) }.to_str().ok()
    }

    #[test]
    fn an_analytics_graph_becomes_gstreamer_mtds() {
        let meta = sample_meta();
        let mut writer = GstAnalyticsWriter::default();
        writer.plan(Some(&meta), FRAME_WIDTH, FRAME_HEIGHT);
        let mtds = &writer.mtds;
        let kinds: Vec<_> = mtds.iter().map(|mtd| mtd.kind).collect();
        assert_eq!(
            kinds,
            [
                MTD_DETECTION,
                MTD_CLASSIFICATION,
                MTD_TRACKING,
                MTD_SEGMENTATION,
                MTD_CLASSIFICATION
            ]
        );
        let detection = &mtds[0];
        assert_eq!(
            (detection.x, detection.y, detection.w, detection.h),
            (160, 240, 320, 120)
        );
        assert_eq!(label_str(detection), Some("car"));
        assert_eq!(label_str(&mtds[1]), Some("person"));
        assert_eq!(mtds[2].tracking_id, 5);
        let segmentation = &mtds[3];
        // SAFETY: the writer's masks hold `mask_len` bytes until its next plan.
        let image = unsafe { slice::from_raw_parts(segmentation.mask, segmentation.mask_len) };
        assert_eq!(image, &[REGION_ID, 0, 0, REGION_ID]);
        assert_eq!(label_str(&mtds[4]), Some("person"));
        let relation = |from, to, kind| G2gAnalyticsRelation { from, to, kind };
        assert_eq!(
            writer.relations,
            [
                relation(3, 4, REL_N_TO_N),
                relation(0, 1, REL_CONTAIN),
                relation(1, 0, REL_IS_PART_OF),
                relation(0, 2, REL_RELATE_TO),
            ]
        );
    }

    #[test]
    fn a_graph_survives_the_trip_through_gstreamer_mtds() {
        let meta = sample_meta();
        let mut writer = GstAnalyticsWriter::default();
        writer.plan(Some(&meta), FRAME_WIDTH, FRAME_HEIGHT);
        let relations: Vec<_> = writer
            .relations
            .iter()
            .map(|relation| (relation.from, relation.to, relation.kind))
            .collect();
        let back = read_all(&writer.mtds, &relations).expect("the plan reads back");

        assert_eq!(back.nodes.len(), meta.nodes.len());
        for (sent, received) in meta.nodes.iter().zip(&back.nodes) {
            let name = |meta: &AnalyticsMeta, node: &AnalyticsNode| match node {
                AnalyticsNode::Detection(d) => meta.class_name(d.label).map(str::to_owned),
                AnalyticsNode::Classification(c) => meta.class_name(c.label).map(str::to_owned),
                AnalyticsNode::Segmentation(s) => meta.class_name(s.label).map(str::to_owned),
                _ => None,
            };
            assert_eq!(name(&meta, sent), name(&back, received));
        }
        assert_eq!(
            back.detections().next().map(|d| d.bbox),
            meta.detections().next().map(|d| d.bbox)
        );
        assert_eq!(
            back.segmentations().next().map(|s| s.mask.clone()),
            meta.segmentations().next().map(|s| s.mask.clone())
        );
        assert_eq!(back.relations, meta.relations);
    }

    #[test]
    fn an_unnamed_label_is_written_as_its_number() {
        let mut meta = AnalyticsMeta::new();
        let label = 42;
        meta.add_detection(ObjectDetection {
            bbox: BBox {
                x: 0.0,
                y: 0.0,
                w: 1.0,
                h: 1.0,
            },
            label,
            confidence: 1.0,
        });
        let mut writer = GstAnalyticsWriter::default();
        writer.plan(Some(&meta), FRAME_WIDTH, FRAME_HEIGHT);
        assert_eq!(label_str(&writer.mtds[0]), Some("42"));
    }
}
