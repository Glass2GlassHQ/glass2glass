//! M743: Vulkan Video H.265 long-term reference pictures.
//!
//! The decoder used to reject any SPS with `long_term_ref_pics_present_flag`
//! set: the long-term tables were not parsed, the slice header's long-term
//! block was never read, and the DPB knew only short-term references. Now the
//! SPS long-term table rides `pLongTermRefPicsSps`, the slice header's
//! long-term entries (SPS-indexed and slice-coded, with the accumulated
//! `DeltaPocMsbCycleLt`) resolve against the DPB by full POC or POC lsb, the
//! RPS prune keeps long-term-listed pictures, `RefPicSetLtCurr` carries the
//! used-by-current slots, and each reference's `Std*` info flags its
//! short/long-term marking (which changes the driver's MV scaling, so a wrong
//! marking corrupts prediction rather than erroring).
//!
//! Fixture: `LTRPSPS_A_Qualcomm_1.bit`, the JCT-VC conformance vector for
//! long-term reference pictures signalled in the SPS (416x240 8-bit Main, from
//! the ffmpeg FATE suite). It carries an 8-entry SPS long-term table and
//! slices using both SPS-indexed (`num_long_term_sps`) and slice-coded
//! (`num_long_term_pics`) entries.
//!
//! Bit-exactness: every display-order frame is SAD/px 0 against the ffmpeg
//! software decoder (the reference is dumped at test time; the test skips if
//! ffmpeg is absent). Runs on the RTX 3060; skips with no adapter / decode
//! support.
#![cfg(all(
    any(target_os = "linux", target_os = "windows"),
    feature = "vulkan-video"
))]

use g2g_core::runtime::block_on;
use g2g_plugins::vulkanvideo::{
    extract_h265_parameter_sets, open_h265_decode_device, parse_h265_slice_header,
    to_std_h265_params, Nv12Frame, VulkanVideoError,
};

const CLIP: &[u8] = include_bytes!("fixtures/LTRPSPS_A_Qualcomm_1.bit");
const W: usize = 416;
const H: usize = 240;

/// Iterate Annex-B NAL payloads (after the start code).
fn nal_units(stream: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 < stream.len() {
        let sc3 = stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1;
        let sc4 = stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 0 && stream[i + 3] == 1;
        if sc4 {
            starts.push(i + 4);
            i += 4;
        } else if sc3 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    starts
        .iter()
        .enumerate()
        .map(|(k, &s)| {
            let mut end = starts.get(k + 1).copied().unwrap_or(stream.len());
            // Trim the next start code (3 or 4 bytes) off this payload's tail.
            if end >= 4 && k + 1 < starts.len() {
                end -= if stream[end - 4] == 0 { 4 } else { 3 };
            }
            &stream[s..end]
        })
        .collect()
}

/// Per-frame luma + chroma SAD/px between decoded frames (display order) and a
/// planar `yuv420p` display-order reference.
fn assert_bit_exact(frames: &[Nv12Frame], ref_yuv: &[u8]) {
    let cw = W / 2;
    let ch = H / 2;
    let fb = W * H + 2 * cw * ch;
    assert_eq!(
        ref_yuv.len(),
        fb * frames.len(),
        "reference frame count mismatch"
    );
    let mut bad = 0usize;
    for (i, f) in frames.iter().enumerate() {
        let base = i * fb;
        let ry = &ref_yuv[base..base + W * H];
        let ru = &ref_yuv[base + W * H..base + W * H + cw * ch];
        let rv = &ref_yuv[base + W * H + cw * ch..base + fb];
        let ysad: u64 = f
            .luma
            .iter()
            .zip(ry)
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as u64)
            .sum();
        let mut usad = 0u64;
        let mut vsad = 0u64;
        for k in 0..cw * ch {
            usad += (f.chroma[2 * k] as i32 - ru[k] as i32).unsigned_abs() as u64;
            vsad += (f.chroma[2 * k + 1] as i32 - rv[k] as i32).unsigned_abs() as u64;
        }
        if ysad != 0 || usad != 0 || vsad != 0 {
            eprintln!(
                "display frame {i}: Y {:.4} U {:.4} V {:.4}",
                ysad as f64 / (W * H) as f64,
                usad as f64 / (cw * ch) as f64,
                vsad as f64 / (cw * ch) as f64
            );
            bad += 1;
        }
    }
    assert_eq!(
        bad, 0,
        "{bad} frames not bit-exact (long-term reference handling wrong)"
    );
}

#[test]
fn h265_long_term_refs_decode_bit_exact() {
    let ps = extract_h265_parameter_sets(CLIP).expect("vps/sps/pps");

    // The fixture must genuinely exercise long-term references, at both levels:
    // an SPS table, and slices that index it / code their own entries.
    assert_eq!(ps.sps.long_term_ref_pics_present_flag, 1);
    assert_eq!(
        ps.sps.num_long_term_ref_pics_sps, 8,
        "conformance vector carries an 8-entry SPS long-term table"
    );
    let (mut lt_slices, mut lt_entries) = (0usize, 0usize);
    for nal in nal_units(CLIP) {
        if nal.len() < 2 || (nal[0] >> 1) & 0x3f > 21 {
            continue;
        }
        if let Some(hdr) = parse_h265_slice_header(nal, &ps.sps, &ps.pps) {
            if hdr.first_slice_segment_in_pic_flag && !hdr.lt.is_empty() {
                lt_slices += 1;
                lt_entries += hdr.lt.len();
            }
        }
    }
    // Ground truth via ffmpeg trace_headers: 184 SPS-indexed + 171 slice-coded
    // long-term entries across the vector.
    assert_eq!(
        lt_entries, 355,
        "fixture's long-term entries misparsed (got {lt_entries} across {lt_slices} slices)"
    );

    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m743: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    };
    let std = to_std_h265_params(&ps);
    let session = device
        .create_h265_session(&std, W as u32, H as u32)
        .expect("session (driver validates the SPS long-term table)");
    let mut dec = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("decoder");

    let frames = dec.decode_all(CLIP).expect("decode");
    assert!(frames.len() > 100, "vector has hundreds of pictures");
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (W as u32, H as u32));
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(max > min, "display frame {i} luma uniform; decode failed");
    }

    // ffmpeg software decode is the oracle, dumped display-order at test time.
    let dir = std::env::temp_dir().join("g2g_m743");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let clip_path = dir.join("ltrpsps.bit");
    let ref_path = dir.join("ref.yuv");
    std::fs::write(&clip_path, CLIP).expect("write clip");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&clip_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&ref_path)
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("ffmpeg reference decode failed: {s}"),
        Err(_) => {
            eprintln!("skip m743 bit-exact half: ffmpeg not on PATH");
            return;
        }
    }
    let ref_yuv = std::fs::read(&ref_path).expect("read reference");
    assert_bit_exact(&frames, &ref_yuv);
    eprintln!(
        "m743: {} frames with long-term references bit-exact vs ffmpeg",
        frames.len()
    );
}
