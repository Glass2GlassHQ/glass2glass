//! M1060: a picture access granularity larger than 16 changes the DPB image
//! extent, not the decoded picture.
//!
//! `VkVideoCapabilitiesKHR::pictureAccessGranularity` is the unit in which a
//! device accesses video picture resources, so every image a decode session
//! writes must be the coded picture rounded up to it. `VulkanVideoDec` allocates
//! its DPB slots at that rounded extent and copies each picture out at its own
//! coded extent, so the padding never reaches an output frame.
//!
//! The RTX 3060 reports 16x16, which 640x480 already satisfies, so the test
//! forces the granularity to 32x32 and 64x64 through a hidden hook and decodes
//! the H.264 / H.265 / AV1 clips again, on the system readback path and on the
//! GPU-resident texture path. It asserts both halves: the DPB images really grew
//! (`dpb_image_extent`), and every decoded frame matches the unforced decode.
//! All four cases run in one test function, sequentially: see the note there.
//! Skips with no adapter / no decode support.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::conformance::fnv1a_64;
use g2g_plugins::vulkanvideo::{
    aligned_extent, extract_av1_sequence_header, extract_h264_parameter_sets,
    extract_h265_parameter_sets, open_av1_decode_device, open_h264_decode_device,
    open_h265_decode_device, to_std_av1_seq_header, to_std_h265_params, Nv12Frame,
    VulkanVideoError,
};

const H264_CLIP: &[u8] = include_bytes!("fixtures/h264_640x480.h264");
const H265_CLIP: &[u8] = include_bytes!("fixtures/h265_640x480.h265");
const AV1_CLIP: &[u8] = include_bytes!("fixtures/av1_640x480.obu");

const PICTURE: (u32, u32) = (640, 480);
/// Granularities to force. 32 leaves 640x480 alone; 64 pushes the height to 512,
/// so the padded case is actually exercised.
const FORCED: [(u32, u32); 2] = [(32, 32), (64, 64)];

/// What one decode run is compared on: the extent the DPB images were created at,
/// and a per-frame digest of the decoded picture.
struct Decoded {
    image_extent: (u32, u32),
    frames: Vec<(u32, u32, u64, u64)>,
}

fn digest(frames: &[Nv12Frame]) -> Vec<(u32, u32, u64, u64)> {
    frames
        .iter()
        .map(|f| (f.width, f.height, fnv1a_64(&f.luma), fnv1a_64(&f.chroma)))
        .collect()
}

/// `None` when this machine cannot decode the codec at all (the caller skips).
fn unsupported(e: &VulkanVideoError) -> bool {
    matches!(
        e,
        VulkanVideoError::NoVulkanAdapter
            | VulkanVideoError::ExtensionUnsupported
            | VulkanVideoError::NoDecodeQueue
    )
}

fn decode_h264(granularity: Option<(u32, u32)>) -> Option<Decoded> {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(e) if unsupported(&e) => return None,
        Err(e) => panic!("open H.264 decode device: {e:?}"),
    };
    let device = match granularity {
        Some(g) => device.with_picture_access_granularity(g),
        None => device,
    };
    let ps = extract_h264_parameter_sets(H264_CLIP).expect("parse SPS+PPS");
    let session = device
        .create_h264_session(&ps, PICTURE.0, PICTURE.1)
        .expect("create H.264 session");
    let mut decoder = device
        .create_h264_dpb_decoder(&session, &ps)
        .expect("create H.264 DPB decoder");
    let frames = decoder.decode_all(H264_CLIP).expect("decode whole stream");
    Some(Decoded {
        image_extent: decoder.dpb_image_extent(),
        frames: digest(&frames),
    })
}

fn decode_h265(granularity: Option<(u32, u32)>) -> Option<Decoded> {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(e) if unsupported(&e) => return None,
        Err(e) => panic!("open H.265 decode device: {e:?}"),
    };
    let device = match granularity {
        Some(g) => device.with_picture_access_granularity(g),
        None => device,
    };
    let ps = extract_h265_parameter_sets(H265_CLIP).expect("parse VPS+SPS+PPS");
    let std = to_std_h265_params(&ps);
    let session = device
        .create_h265_session(&std, PICTURE.0, PICTURE.1)
        .expect("create H.265 session");
    let mut decoder = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("create H.265 DPB decoder");
    let frames = decoder.decode_all(H265_CLIP).expect("decode whole stream");
    Some(Decoded {
        image_extent: decoder.dpb_image_extent(),
        frames: digest(&frames),
    })
}

fn decode_av1(granularity: Option<(u32, u32)>) -> Option<Decoded> {
    let device = match block_on(open_av1_decode_device()) {
        Ok(d) => d,
        Err(e) if unsupported(&e) => return None,
        Err(e) => panic!("open AV1 decode device: {e:?}"),
    };
    let device = match granularity {
        Some(g) => device.with_picture_access_granularity(g),
        None => device,
    };
    let seq = extract_av1_sequence_header(AV1_CLIP).expect("parse sequence header");
    let std = to_std_av1_seq_header(&seq);
    let session = device
        .create_av1_session(&std, PICTURE.0, PICTURE.1)
        .expect("create AV1 session");
    let mut decoder = device
        .create_av1_dpb_decoder(&session, &seq)
        .expect("create AV1 DPB decoder");
    let frames = decoder.decode_all(AV1_CLIP).expect("decode whole stream");
    Some(Decoded {
        image_extent: decoder.dpb_image_extent(),
        frames: digest(&frames),
    })
}

/// Decode once unforced, then once per forced granularity, and check the images
/// grew while the pictures did not change.
fn check(codec: &str, decode: fn(Option<(u32, u32)>) -> Option<Decoded>) {
    let Some(baseline) = decode(None) else {
        eprintln!("skip m1060: no Vulkan {codec} decode adapter");
        return;
    };
    assert!(
        !baseline.frames.is_empty(),
        "{codec}: the unforced decode produced no frames"
    );
    // The unforced run is the reference, so it must not already be padded past
    // what this machine's real granularity needs.
    assert!(
        baseline.image_extent.0 >= PICTURE.0 && baseline.image_extent.1 >= PICTURE.1,
        "{codec}: DPB images {:?} are smaller than the picture {PICTURE:?}",
        baseline.image_extent
    );

    for granularity in FORCED {
        let forced = decode(Some(granularity)).expect("device opened once already");
        assert_eq!(
            forced.image_extent,
            aligned_extent(PICTURE, granularity),
            "{codec}: DPB images were not rounded up to granularity {granularity:?}"
        );
        assert_eq!(
            forced.frames, baseline.frames,
            "{codec}: granularity {granularity:?} changed the decoded pictures"
        );
    }
}

/// The GPU-resident path samples the DPB slot through a `VkSamplerYcbcrConversion`,
/// whose normalized coordinates are relative to the whole slot image, so the
/// converter is told the padded extent. Returns each frame's RGBA readback plus
/// the DPB image extent. `None` when this machine has no GPU decode path.
/// The DPB image extent plus each frame's RGBA readback.
type TextureDecode = ((u32, u32), Vec<Vec<u8>>);

fn decode_h264_textures(granularity: Option<(u32, u32)>) -> Option<TextureDecode> {
    let device = match block_on(open_h264_decode_device()) {
        Ok(d) => d,
        Err(e) if unsupported(&e) => return None,
        Err(e) => panic!("open H.264 decode device: {e:?}"),
    };
    let device = match granularity {
        Some(g) => device.with_picture_access_granularity(g),
        None => device,
    };
    let ps = extract_h264_parameter_sets(H264_CLIP).expect("parse SPS+PPS");
    let session = device
        .create_h264_session(&ps, PICTURE.0, PICTURE.1)
        .expect("create H.264 session");
    let mut decoder = match device.create_h264_dpb_decoder_gpu(&session, &ps) {
        Ok(d) => d,
        Err(VulkanVideoError::NoComputeQueue) => return None,
        Err(e) => panic!("create GPU DPB decoder: {e:?}"),
    };
    let textures = decoder
        .decode_all_to_textures(H264_CLIP)
        .expect("decode to textures");
    let mut frames = Vec::new();
    for texture in &textures {
        assert_eq!(
            (texture.width(), texture.height()),
            PICTURE,
            "the RGBA output stays at the picture extent, not the padded one"
        );
        frames.push(device.read_rgba_texture(texture));
    }
    Some((decoder.dpb_image_extent(), frames))
}

fn check_textures() {
    let Some((baseline_extent, baseline)) = decode_h264_textures(None) else {
        eprintln!("skip m1060: no Vulkan H.264 GPU decode path");
        return;
    };
    assert!(
        !baseline.is_empty(),
        "the unforced decode produced no frames"
    );
    assert!(
        baseline_extent.0 >= PICTURE.0 && baseline_extent.1 >= PICTURE.1,
        "DPB images {baseline_extent:?} are smaller than the picture {PICTURE:?}"
    );

    for granularity in FORCED {
        let (extent, frames) = decode_h264_textures(Some(granularity)).expect("device opened once");
        let padded = aligned_extent(PICTURE, granularity);
        assert_eq!(
            extent, padded,
            "DPB images were not rounded up to granularity {granularity:?}"
        );
        assert_eq!(frames.len(), baseline.len());
        // This host's driver writes only the picture it decodes, so the rows it
        // was forced to allocate past the picture hold undefined data, and the
        // chroma filter reaches into the first of them for the bottom luma row.
        // A device that really reported this granularity would have written the
        // whole block. Compare every row the filter cannot reach. The assert on
        // `padded.0` keeps a padded width from slipping past the same argument.
        assert_eq!(
            padded.0, PICTURE.0,
            "a padded width needs a column exclusion"
        );
        let rows = if padded.1 > PICTURE.1 {
            PICTURE.1 - 1
        } else {
            PICTURE.1
        } as usize;
        let compared = rows * PICTURE.0 as usize * 4;
        for (i, (frame, reference)) in frames.iter().zip(&baseline).enumerate() {
            assert_eq!(
                frame[..compared],
                reference[..compared],
                "granularity {granularity:?} changed texture {i}"
            );
        }
    }
}

// One test for all four cases: libtest runs test functions on parallel threads,
// and two threads building a `wgpu::Instance` at once fault inside the Vulkan
// loader's `loader_icd_scan`, which is why this file keeps one device-creating
// test.
#[test]
fn decode_survives_a_larger_picture_access_granularity() {
    // Guard the fixture choice: at 64x64 the 480-row picture must actually pad,
    // or the forced runs would prove nothing.
    assert_ne!(aligned_extent(PICTURE, (64, 64)), PICTURE);
    check("H.264", decode_h264);
    check("H.265", decode_h265);
    check("AV1", decode_av1);
    check_textures();
}
