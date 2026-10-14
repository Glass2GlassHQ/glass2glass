#![cfg(feature = "gst-analytics")]

use core::ffi::{c_char, c_int, c_void, CStr};
use core::ptr;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct GstAnalyticsMtd {
    id: u32,
    meta: *mut c_void,
}

const EMPTY_MTD: GstAnalyticsMtd = GstAnalyticsMtd {
    id: 0,
    meta: ptr::null_mut(),
};

type Buffer = *mut c_void;
type RelationMeta = *mut c_void;

const GST_STATE_NULL: c_int = 1;
const GST_STATE_PLAYING: c_int = 4;
const GST_FLOW_OK: c_int = 0;
const GST_VIDEO_FORMAT_GRAY8: c_int = 25;
const GST_VIDEO_FRAME_FLAG_NONE: c_int = 0;
const GST_VIDEO_MAX_PLANES: usize = 4;
const GST_SEGMENTATION_TYPE_INSTANCE: c_int = 1;
const GST_CLOCK_TIME_NONE: u64 = u64::MAX;
const REL_IS_PART_OF: c_int = 1 << 1;
const REL_CONTAIN: c_int = 1 << 2;
const REL_RELATE_TO: c_int = 1 << 3;
const REL_N_TO_N: c_int = 1 << 4;

const PULL_TIMEOUT_NS: u64 = 5_000_000_000;
const FRAME_WIDTH: usize = 64;
const FRAME_HEIGHT: usize = 48;
const RGBA_BYTES: usize = 4;
const PIPELINE: &CStr = c"appsrc name=src format=time caps=video/x-raw,format=RGBA,width=64,height=48,framerate=30/1 ! glass2glass fragment=identity ! appsink name=sink sync=false";

extern "C" {
    fn gst_init(argc: *mut c_int, argv: *mut *mut *mut c_char);
    fn gst_element_factory_find(name: *const c_char) -> *mut c_void;
    fn gst_object_unref(object: *mut c_void);
    fn gst_mini_object_unref(object: *mut c_void);
    fn gst_parse_launch(description: *const c_char, error: *mut *mut c_void) -> *mut c_void;
    fn gst_bin_get_by_name(bin: *mut c_void, name: *const c_char) -> *mut c_void;
    fn gst_element_set_state(element: *mut c_void, state: c_int) -> c_int;
    fn g_signal_emit_by_name(instance: *mut c_void, signal: *const c_char, ...);
    fn gst_sample_get_buffer(sample: *mut c_void) -> Buffer;
    fn gst_buffer_new_allocate(allocator: *mut c_void, size: usize, params: *mut c_void) -> Buffer;
    fn gst_buffer_memset(buffer: Buffer, offset: usize, value: u8, size: usize) -> usize;
    fn gst_buffer_new_memdup(data: *const u8, size: usize) -> Buffer;
    fn gst_buffer_extract(buffer: Buffer, offset: usize, dest: *mut u8, size: usize) -> usize;
    fn gst_buffer_add_video_meta_full(
        buffer: Buffer,
        flags: c_int,
        format: c_int,
        width: u32,
        height: u32,
        n_planes: u32,
        offset: *const usize,
        stride: *const c_int,
    ) -> *mut c_void;
    fn g_quark_from_string(name: *const c_char) -> u32;
    fn g_quark_to_string(quark: u32) -> *const c_char;

    fn gst_buffer_add_analytics_relation_meta(buffer: Buffer) -> RelationMeta;
    fn gst_buffer_get_analytics_relation_meta(buffer: Buffer) -> RelationMeta;
    fn gst_analytics_relation_meta_set_relation(
        meta: RelationMeta,
        kind: c_int,
        first: u32,
        second: u32,
    ) -> c_int;
    fn gst_analytics_relation_meta_get_relation(
        meta: RelationMeta,
        first: u32,
        second: u32,
    ) -> c_int;
    fn gst_analytics_relation_meta_iterate(
        meta: RelationMeta,
        state: *mut *mut c_void,
        kind: usize,
        mtd: *mut GstAnalyticsMtd,
    ) -> c_int;
    fn gst_analytics_relation_meta_add_od_mtd(
        meta: RelationMeta,
        label: u32,
        x: c_int,
        y: c_int,
        w: c_int,
        h: c_int,
        confidence: f32,
        mtd: *mut GstAnalyticsMtd,
    ) -> c_int;
    fn gst_analytics_relation_meta_add_one_cls_mtd(
        meta: RelationMeta,
        confidence: f32,
        label: u32,
        mtd: *mut GstAnalyticsMtd,
    ) -> c_int;
    fn gst_analytics_relation_meta_add_tracking_mtd(
        meta: RelationMeta,
        tracking_id: u64,
        first_seen: u64,
        mtd: *mut GstAnalyticsMtd,
    ) -> c_int;
    fn gst_analytics_relation_meta_add_segmentation_mtd(
        meta: RelationMeta,
        mask: Buffer,
        kind: c_int,
        region_count: usize,
        region_ids: *const u32,
        x: c_int,
        y: c_int,
        w: u32,
        h: u32,
        mtd: *mut GstAnalyticsMtd,
    ) -> c_int;
    fn gst_analytics_od_mtd_get_mtd_type() -> usize;
    fn gst_analytics_cls_mtd_get_mtd_type() -> usize;
    fn gst_analytics_tracking_mtd_get_mtd_type() -> usize;
    fn gst_analytics_segmentation_mtd_get_mtd_type() -> usize;
    fn gst_analytics_od_mtd_get_location(
        mtd: *const GstAnalyticsMtd,
        x: *mut c_int,
        y: *mut c_int,
        w: *mut c_int,
        h: *mut c_int,
        confidence: *mut f32,
    ) -> c_int;
    fn gst_analytics_od_mtd_get_obj_type(mtd: *const GstAnalyticsMtd) -> u32;
    fn gst_analytics_cls_mtd_get_quark(mtd: *const GstAnalyticsMtd, index: usize) -> u32;
    fn gst_analytics_tracking_mtd_get_info(
        mtd: *const GstAnalyticsMtd,
        tracking_id: *mut u64,
        first_seen: *mut u64,
        last_seen: *mut u64,
        lost: *mut c_int,
    ) -> c_int;
    fn gst_analytics_segmentation_mtd_get_mask(
        mtd: *const GstAnalyticsMtd,
        x: *mut c_int,
        y: *mut c_int,
        w: *mut u32,
        h: *mut u32,
    ) -> Buffer;
    fn gst_analytics_segmentation_mtd_get_region_id(
        mtd: *const GstAnalyticsMtd,
        index: usize,
    ) -> u32;
}

fn quark(name: &CStr) -> u32 {
    // SAFETY: `name` is a valid C string.
    unsafe { g_quark_from_string(name.as_ptr()) }
}

fn quark_name(quark: u32) -> String {
    // SAFETY: GLib returns null or a static string for any quark.
    let name = unsafe { g_quark_to_string(quark) };
    assert!(!name.is_null(), "quark {quark} has a name");
    // SAFETY: non-null quark strings are valid C strings.
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

fn plugin_available(name: &CStr) -> bool {
    // SAFETY: `name` is a valid C string, gst_init has run.
    let factory = unsafe { gst_element_factory_find(name.as_ptr()) };
    if factory.is_null() {
        return false;
    }
    // SAFETY: the factory came back with a reference this call owns.
    unsafe { gst_object_unref(factory) };
    true
}

// The single mtd of `kind` in `meta`.
fn only_mtd(meta: RelationMeta, kind: usize) -> GstAnalyticsMtd {
    let mut state = ptr::null_mut();
    let mut mtd = EMPTY_MTD;
    let mut found = Vec::new();
    // SAFETY: `meta` is a live relation meta, `state` and `mtd` are writable.
    while unsafe { gst_analytics_relation_meta_iterate(meta, &mut state, kind, &mut mtd) } != 0 {
        found.push(mtd);
    }
    assert_eq!(found.len(), 1, "one mtd of the kind");
    found[0]
}

const PERSON: &CStr = c"person";
const ADULT: &CStr = c"adult";
const LEAF: &CStr = c"leaf";
const BOX: (c_int, c_int, c_int, c_int) = (8, 12, 32, 24);
const SEGMENT_BOX: (c_int, c_int, u32, u32) = (0, 0, 16, 12);
const CONFIDENCE: f32 = 0.875;
const TRACKED_OBJECT: u64 = 77;
const REGION: u8 = 9;
const MASK_WIDTH: u32 = 3;
const MASK_HEIGHT: u32 = 2;
const MASK_STRIDE: c_int = 4;
const PADDING: u8 = 0xff;
const MASK: [u8; 8] = [REGION, 0, REGION, PADDING, 0, REGION, 0, PADDING];

// SAFETY (caller): `buffer` is a live, writable GstBuffer.
unsafe fn add_analytics(buffer: Buffer) {
    // SAFETY: caller contract.
    let meta = unsafe { gst_buffer_add_analytics_relation_meta(buffer) };
    let (x, y, w, h) = BOX;
    let (mut detection, mut class, mut tracking, mut segmentation, mut region_class) =
        (EMPTY_MTD, EMPTY_MTD, EMPTY_MTD, EMPTY_MTD, EMPTY_MTD);
    let offsets = [0usize; GST_VIDEO_MAX_PLANES];
    let mut strides = [0 as c_int; GST_VIDEO_MAX_PLANES];
    strides[0] = MASK_STRIDE;
    let region_ids = [u32::from(REGION)];
    let (mask_x, mask_y, mask_w, mask_h) = SEGMENT_BOX;
    // SAFETY: `meta` is the buffer's live relation meta, every out pointer is a
    // writable local, and the mask buffer moves into the segmentation mtd.
    unsafe {
        assert_ne!(
            gst_analytics_relation_meta_add_od_mtd(
                meta,
                quark(PERSON),
                x,
                y,
                w,
                h,
                CONFIDENCE,
                &mut detection
            ),
            0
        );
        assert_ne!(
            gst_analytics_relation_meta_add_one_cls_mtd(meta, CONFIDENCE, quark(ADULT), &mut class),
            0
        );
        assert_ne!(
            gst_analytics_relation_meta_add_tracking_mtd(
                meta,
                TRACKED_OBJECT,
                GST_CLOCK_TIME_NONE,
                &mut tracking
            ),
            0
        );
        let mask = gst_buffer_new_memdup(MASK.as_ptr(), MASK.len());
        gst_buffer_add_video_meta_full(
            mask,
            GST_VIDEO_FRAME_FLAG_NONE,
            GST_VIDEO_FORMAT_GRAY8,
            MASK_WIDTH,
            MASK_HEIGHT,
            1,
            offsets.as_ptr(),
            strides.as_ptr(),
        );
        assert_ne!(
            gst_analytics_relation_meta_add_segmentation_mtd(
                meta,
                mask,
                GST_SEGMENTATION_TYPE_INSTANCE,
                region_ids.len(),
                region_ids.as_ptr(),
                mask_x,
                mask_y,
                mask_w,
                mask_h,
                &mut segmentation,
            ),
            0
        );
        assert_ne!(
            gst_analytics_relation_meta_add_one_cls_mtd(
                meta,
                CONFIDENCE,
                quark(LEAF),
                &mut region_class
            ),
            0
        );
        gst_analytics_relation_meta_set_relation(meta, REL_CONTAIN, detection.id, class.id);
        gst_analytics_relation_meta_set_relation(meta, REL_IS_PART_OF, class.id, detection.id);
        gst_analytics_relation_meta_set_relation(meta, REL_RELATE_TO, detection.id, tracking.id);
        gst_analytics_relation_meta_set_relation(
            meta,
            REL_N_TO_N,
            segmentation.id,
            region_class.id,
        );
    }
}

// SAFETY (caller): `buffer` is a live GstBuffer.
unsafe fn check_analytics(buffer: Buffer) {
    // SAFETY: caller contract.
    let meta = unsafe { gst_buffer_get_analytics_relation_meta(buffer) };
    assert!(!meta.is_null(), "the output buffer carries a relation meta");
    // SAFETY: plain getters.
    let (od_type, cls_type, tracking_type, segmentation_type) = unsafe {
        (
            gst_analytics_od_mtd_get_mtd_type(),
            gst_analytics_cls_mtd_get_mtd_type(),
            gst_analytics_tracking_mtd_get_mtd_type(),
            gst_analytics_segmentation_mtd_get_mtd_type(),
        )
    };
    let detection = only_mtd(meta, od_type);
    let tracking = only_mtd(meta, tracking_type);
    let segmentation = only_mtd(meta, segmentation_type);
    let mut classes = Vec::new();
    let mut state = ptr::null_mut();
    let mut mtd = EMPTY_MTD;
    // SAFETY: `meta` is live, `state` and `mtd` are writable.
    while unsafe { gst_analytics_relation_meta_iterate(meta, &mut state, cls_type, &mut mtd) } != 0
    {
        classes.push(mtd);
    }
    // SAFETY: every handle below came from iterating `meta`, out pointers are locals.
    unsafe {
        let (mut x, mut y, mut w, mut h, mut confidence) = (0, 0, 0, 0, 0.0);
        assert_ne!(
            gst_analytics_od_mtd_get_location(
                &detection,
                &mut x,
                &mut y,
                &mut w,
                &mut h,
                &mut confidence
            ),
            0
        );
        assert_eq!((x, y, w, h), BOX);
        assert_eq!(confidence, CONFIDENCE);
        assert_eq!(
            quark_name(gst_analytics_od_mtd_get_obj_type(&detection)),
            "person"
        );

        let class_named = |name: &str| {
            *classes
                .iter()
                .find(|class| quark_name(gst_analytics_cls_mtd_get_quark(*class, 0)) == name)
                .unwrap_or_else(|| panic!("a classification named {name}"))
        };
        let class = class_named("adult");
        let region_class = class_named("leaf");
        assert_eq!(
            gst_analytics_relation_meta_get_relation(meta, detection.id, class.id),
            REL_CONTAIN
        );
        assert_eq!(
            gst_analytics_relation_meta_get_relation(meta, class.id, detection.id),
            REL_IS_PART_OF
        );

        let (mut tracking_id, mut first_seen, mut last_seen, mut lost) = (0, 0, 0, 0);
        assert_ne!(
            gst_analytics_tracking_mtd_get_info(
                &tracking,
                &mut tracking_id,
                &mut first_seen,
                &mut last_seen,
                &mut lost
            ),
            0
        );
        assert_eq!(tracking_id, TRACKED_OBJECT);
        assert_eq!(
            gst_analytics_relation_meta_get_relation(meta, detection.id, tracking.id),
            REL_RELATE_TO
        );

        assert_eq!(
            gst_analytics_relation_meta_get_relation(meta, segmentation.id, region_class.id),
            REL_N_TO_N
        );
        let region = gst_analytics_segmentation_mtd_get_region_id(&segmentation, 0);
        let (mut mask_x, mut mask_y, mut mask_w, mut mask_h) = (0, 0, 0, 0);
        let mask = gst_analytics_segmentation_mtd_get_mask(
            &segmentation,
            &mut mask_x,
            &mut mask_y,
            &mut mask_w,
            &mut mask_h,
        );
        assert!(!mask.is_null(), "the segmentation has a mask");
        assert_eq!((mask_x, mask_y, mask_w, mask_h), SEGMENT_BOX);
        let mut pixels = [0u8; (MASK_WIDTH * MASK_HEIGHT) as usize];
        let read = gst_buffer_extract(mask, 0, pixels.as_mut_ptr(), pixels.len());
        gst_mini_object_unref(mask);
        assert_eq!(read, pixels.len());
        let inside = u8::try_from(region).expect("a GRAY8 region id");
        assert_eq!(pixels, [inside, 0, inside, 0, inside, 0]);
    }
}

#[test]
fn analytics_cross_the_bridge_element_both_ways() {
    // SAFETY: GStreamer allows null argc/argv.
    unsafe { gst_init(ptr::null_mut(), ptr::null_mut()) };
    if !plugin_available(c"appsrc") || !plugin_available(c"appsink") {
        eprintln!("skipping: GStreamer's appsrc / appsink are not installed");
        return;
    }
    g2g_bridge::ffi::gst_plugin_glass2glass_register();

    // SAFETY: a valid launch line, errors are reported as a null pipeline.
    let pipeline = unsafe { gst_parse_launch(PIPELINE.as_ptr(), ptr::null_mut()) };
    assert!(!pipeline.is_null(), "the pipeline parses");
    // SAFETY: `pipeline` is a live bin.
    let (source, sink) = unsafe {
        (
            gst_bin_get_by_name(pipeline, c"src".as_ptr()),
            gst_bin_get_by_name(pipeline, c"sink".as_ptr()),
        )
    };
    assert!(!source.is_null() && !sink.is_null());

    let size = FRAME_WIDTH * FRAME_HEIGHT * RGBA_BYTES;
    let mut flow = GST_FLOW_OK;
    let mut sample: *mut c_void = ptr::null_mut();
    // SAFETY: every object is live, the signals take these argument types, and each
    // reference this test owns is released once.
    unsafe {
        gst_element_set_state(pipeline, GST_STATE_PLAYING);
        let buffer = gst_buffer_new_allocate(ptr::null_mut(), size, ptr::null_mut());
        gst_buffer_memset(buffer, 0, 0, size);
        add_analytics(buffer);
        g_signal_emit_by_name(source, c"push-buffer".as_ptr(), buffer, &mut flow);
        gst_mini_object_unref(buffer);
        g_signal_emit_by_name(
            sink,
            c"try-pull-sample".as_ptr(),
            PULL_TIMEOUT_NS,
            &mut sample,
        );
    }
    assert_eq!(flow, GST_FLOW_OK, "appsrc took the buffer");
    assert!(!sample.is_null(), "the bridge produced a frame");
    // SAFETY: `sample` is the live sample just pulled, its buffer borrowed from it.
    unsafe {
        check_analytics(gst_sample_get_buffer(sample));
        gst_mini_object_unref(sample);
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(source);
        gst_object_unref(sink);
        gst_object_unref(pipeline);
    }
}
