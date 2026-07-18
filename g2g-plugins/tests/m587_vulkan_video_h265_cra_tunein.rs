//! M587: H.265 mid-stream random-access tune-in at a CRA (discard RASL followers).
//!
//! M577 made full-stream open-GOP HEVC decode correctly (a CRA in continuous
//! decoding keeps the pre-CRA references its RASL leading pictures use). This is
//! the seek case: tuning in at a CRA mid-stream (a fresh decode that reset the
//! decoder), where those pre-CRA references are ABSENT. The CRA's RASL followers
//! reference them, so they cannot decode and must be discarded (H.265 8.1.3,
//! NoRaslOutputFlag == 1); the CRA's trailing pictures and the following GOPs
//! decode normally.
//!
//! The test decodes the whole open-GOP clip once as an oracle (POC-complete,
//! validated in M577), then `reset()`s the decoder and decodes starting at a
//! mid-stream CRA. It asserts: (a) tune-in decodes without error; (b) it drops
//! exactly the CRA's leading RASL pictures (fewer frames than the coded-picture
//! count of the tuned-in substream); (c) every emitted frame is bit-exact to some
//! full-decode frame (so the kept pictures decoded correctly from the CRA). A
//! decoder that did NOT skip the RASL would decode them against a flushed DPB
//! (garbage or error) and emit the wrong count.
//!
//! Runs on the RTX 3060; skips with no adapter / no decode support.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, to_std_h265_params, Nv12Frame,
    VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/h265_640x480_opengop.hevc");

/// Byte offsets of each NAL payload (just past its start code) and the offset of
/// the start code itself, so a substream can be sliced from a start-code boundary.
fn nal_units(data: &[u8]) -> Vec<(usize, usize)> {
    // (start_code_offset, payload_offset)
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if i + 4 <= data.len() && data[i + 2] == 0 && data[i + 3] == 1 {
                out.push((i, i + 4));
                i += 4;
                continue;
            }
            if data[i + 2] == 1 {
                out.push((i, i + 3));
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    out
}

fn nal_type(data: &[u8], payload: usize) -> u8 {
    (data[payload] >> 1) & 0x3F
}

/// The concatenated NV12 planes of a decoded frame (luma then interleaved CbCr).
fn planes(f: &Nv12Frame) -> Vec<u8> {
    let mut v = Vec::with_capacity(f.luma.len() + f.chroma.len());
    v.extend_from_slice(&f.luma);
    v.extend_from_slice(&f.chroma);
    v
}

#[test]
fn h265_cra_tunein_skips_rasl() {
    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m587: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    };
    let ps = extract_h265_parameter_sets(CLIP).expect("vps/sps/pps");
    let (w, h) = (
        ps.sps.pic_width_in_luma_samples,
        ps.sps.pic_height_in_luma_samples,
    );
    let std = to_std_h265_params(&ps);
    let session = device.create_h265_session(&std, w, h).expect("session");
    let mut dec = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("decoder");

    // Oracle: whole-stream decode (M577-validated). Collect every frame's bytes.
    let (fm, mut ff) = dec.decode_push_meta(CLIP).expect("full decode");
    ff.extend(dec.decode_flush().expect("full flush"));
    assert_eq!(fm.len(), ff.len(), "one meta per full-decode frame");
    let full_frames: Vec<Vec<u8>> = ff.iter().map(planes).collect();
    assert!(
        full_frames.len() >= 20,
        "open-GOP fixture decodes its whole timeline"
    );

    // Find the first mid-stream CRA (nal_unit_type 21) and slice the substream from
    // its start code (VCL only from there; the session already holds the params).
    let nals = nal_units(CLIP);
    let cra_pos = nals
        .iter()
        .find(|&&(_, p)| nal_type(CLIP, p) == 21)
        .map(|&(sc, _)| sc)
        .expect("fixture carries a CRA (open-GOP)");
    let from_cra = &CLIP[cra_pos..];

    // Count the substream's coded pictures and the tuned-in CRA's leading RASL
    // (its contiguous leading run: NAL types 6/7 RADL + 8/9 RASL, right after the
    // CRA, until the first non-leading VCL). Those RASL are what tune-in drops.
    let sub_nals = nal_units(from_cra);
    let n_pics = sub_nals
        .iter()
        .filter(|&&(_, p)| nal_type(from_cra, p) <= 31)
        .count();
    let mut dropped_rasl = 0usize;
    for (k, &(_, p)) in sub_nals.iter().enumerate() {
        let t = nal_type(from_cra, p);
        if k == 0 {
            continue; // the CRA itself
        }
        match t {
            8 | 9 => dropped_rasl += 1, // RASL leading picture
            6 | 7 => {}                 // RADL leading picture (kept)
            _ => break,                 // first non-leading VCL ends the CRA's leading run
        }
    }
    assert!(
        dropped_rasl > 0,
        "fixture's tuned-in CRA must have RASL followers to skip"
    );

    // Tune in: reset (a seek) then decode from the CRA. The CRA is now the first
    // picture (NoRaslOutputFlag == 1), so its RASL followers are discarded.
    dec.reset();
    let (tm, mut tf) = dec
        .decode_push_meta(from_cra)
        .expect("tune-in decode (RASL must be skipped)");
    tf.extend(dec.decode_flush().expect("tune-in flush"));
    assert_eq!(tm.len(), tf.len(), "one meta per tuned-in frame");
    assert!(
        !tf.is_empty(),
        "tune-in decodes the CRA and its trailing pictures"
    );

    // (b) Exactly the CRA's RASL were dropped.
    assert_eq!(
        tf.len(),
        n_pics - dropped_rasl,
        "tune-in must emit every coded picture except the CRA's {dropped_rasl} RASL followers"
    );

    // (c) Every emitted frame is a correct decode: bit-exact to some full-decode
    // frame (a RASL decoded against the flushed DPB would match nothing).
    for (i, f) in tf.iter().enumerate() {
        let bytes = planes(f);
        assert!(
            full_frames.contains(&bytes),
            "tuned-in frame {i} is not bit-exact to any full-decode frame (decoded wrong)"
        );
    }
    eprintln!(
        "m587 h265 CRA tune-in: {} coded pics, dropped {} RASL, {} frames all bit-exact",
        n_pics,
        dropped_rasl,
        tf.len()
    );
}
