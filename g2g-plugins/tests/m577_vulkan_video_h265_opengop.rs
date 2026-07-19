//! M577: Vulkan Video H.265 open-GOP (CRA / RASL) decode.
//!
//! x265 defaults to `open-gop=1`, so most HEVC in the wild does random access at a
//! CRA (Clean Random Access) picture rather than an IDR. A CRA is followed in
//! decode order by RASL (Random Access Skipped Leading) pictures that display
//! *before* it and reference pictures decoded *before* the CRA. The decoder used
//! to flush the DPB at every IRAP (IDR / CRA / BLA alike), which is only correct
//! when `NoRaslOutputFlag == 1`; a mid-stream CRA has it 0, so flushing destroyed
//! the pre-CRA references its RASL followers need and those frames decoded against
//! a cleared DPB (garbage). The M503 / M569 H.265 fixtures dodged this by forcing
//! closed GOPs. `H265DpbDecoder` now flushes only on an IRAP with
//! `NoRaslOutputFlag == 1` and otherwise applies the picture's reference-picture
//! set (which, for a CRA, retains the pre-CRA pictures), so RASL leading pictures
//! decode correctly.
//!
//! Fixture (`h265_640x480_opengop.hevc`, 30 pictures, one leading IDR then two CRA
//! GOPs with RASL leading pictures, POC monotonic across the CRAs):
//! ```text
//! ffmpeg -f lavfi -i testsrc2=size=640x480:rate=30:duration=1 -c:v libx265 \
//!   -pix_fmt yuv420p -x265-params \
//!   keyint=12:min-keyint=12:open-gop=1:bframes=3:b-pyramid=1:scenecut=0:rc-lookahead=12:aud=0:repeat-headers=1 \
//!   -f hevc h265_640x480_opengop.hevc
//! ```
//! Structural + content assertions run always (the stream must actually carry a
//! CRA with RASL followers, else the fix is untested). Bit-exactness against the
//! software decoder's DISPLAY-order output is checked when `G2G_H265_OPENGOP_REF`
//! points at a raw `yuv420p` dump (`ffmpeg -i clip -f rawvideo -pix_fmt yuv420p
//! ref.yuv`): every frame at its display index is SAD/px 0, which only holds if
//! the CRA kept the references its RASL followers reference.
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
const W: usize = 640;
const H: usize = 480;

/// Count VCL NAL units of each `nal_unit_type` in an Annex-B stream. Used to prove
/// the fixture is genuinely open-GOP (has a CRA + RASL leading pictures), so the
/// decode below actually exercises the pre-CRA reference retention.
fn count_nal_types(stream: &[u8]) -> (usize, usize, usize) {
    let (mut cra, mut rasl, mut idr) = (0, 0, 0);
    let mut i = 0;
    while i + 3 < stream.len() {
        let sc3 = stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1;
        let sc4 = stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 0 && stream[i + 3] == 1;
        if sc3 || sc4 {
            let hdr = if sc4 { i + 4 } else { i + 3 };
            if hdr < stream.len() {
                match (stream[hdr] >> 1) & 0x3f {
                    21 => cra += 1,      // CRA_NUT
                    8 | 9 => rasl += 1,  // RASL_N / RASL_R
                    19 | 20 => idr += 1, // IDR_W_RADL / IDR_N_LP
                    _ => {}
                }
            }
            i = hdr;
        } else {
            i += 1;
        }
    }
    (idr, cra, rasl)
}

/// Per-frame luma + chroma SAD/px between decoded frames (display order) and a
/// planar `yuv420p` display-order reference; panics on length mismatch.
fn assert_bit_exact(frames: &[Nv12Frame], ref_path: &str) {
    let ref_yuv = std::fs::read(ref_path).expect("read reference");
    let cw = W / 2;
    let ch = H / 2;
    let fb = W * H + 2 * cw * ch;
    assert!(ref_yuv.len() >= fb * frames.len(), "reference too short");
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
        eprintln!(
            "display frame {i}: Y {:.4} U {:.4} V {:.4}",
            ysad as f64 / (W * H) as f64,
            usad as f64 / (cw * ch) as f64,
            vsad as f64 / (cw * ch) as f64
        );
        assert_eq!(
            ysad, 0,
            "frame {i} luma not bit-exact (open-GOP RASL decode or ordering wrong)"
        );
        assert_eq!(usad, 0, "frame {i} Cb not bit-exact");
        assert_eq!(vsad, 0, "frame {i} Cr not bit-exact");
    }
}

#[test]
fn h265_opengop_cra_rasl_decodes_correctly() {
    // The fixture must actually be open-GOP: a leading IDR, CRA anchors, and RASL
    // leading pictures. Otherwise the pre-CRA reference retention is never tested.
    let (idr, cra, rasl) = count_nal_types(CLIP);
    assert_eq!(idr, 1, "fixture should have exactly one leading IDR");
    assert!(cra >= 1, "fixture has no CRA (not open-GOP)");
    assert!(
        rasl >= 1,
        "fixture has no RASL leading pictures (open-GOP retention untested)"
    );

    let device = match block_on(open_h265_decode_device()) {
        Ok(d) => d,
        Err(VulkanVideoError::NoVulkanAdapter)
        | Err(VulkanVideoError::ExtensionUnsupported)
        | Err(VulkanVideoError::NoDecodeQueue) => {
            eprintln!("skip m577: no Vulkan H.265 decode adapter");
            return;
        }
        Err(e) => panic!("open h265 device: {e:?}"),
    };
    let ps = extract_h265_parameter_sets(CLIP).expect("vps/sps/pps");
    let std = to_std_h265_params(&ps);
    let session = device
        .create_h265_session(&std, W as u32, H as u32)
        .expect("session");
    let mut dec = device
        .create_h265_dpb_decoder(&session, &ps)
        .expect("decoder");

    // POC is monotonic across the mid-stream CRAs (single coded video sequence),
    // but the RASL leading pictures make it non-monotonic in decode order.
    let metas = dec.index_pictures(CLIP).expect("index");
    let pocs: Vec<i32> = metas.iter().map(|m| m.poc).collect();
    assert!(
        pocs.windows(2).any(|w| w[1] < w[0]),
        "no reorder in fixture (RASL leading pics absent)"
    );

    let frames = dec.decode_all(CLIP).expect("decode");
    assert_eq!(frames.len(), metas.len(), "one frame per coded picture");
    assert_eq!(frames.len(), 30, "fixture has 30 coded pictures");

    // Every frame is real, non-uniform content. Before the fix the RASL frames
    // decoded against a flushed DPB and came out corrupt; this and the bit-exact
    // check below fail on that.
    for (i, f) in frames.iter().enumerate() {
        assert_eq!((f.width, f.height), (W as u32, H as u32));
        let min = *f.luma.iter().min().unwrap();
        let max = *f.luma.iter().max().unwrap();
        assert!(
            max > min,
            "display frame {i} luma uniform ({min}=={max}); decode failed"
        );
    }

    if let Ok(p) = std::env::var("G2G_H265_OPENGOP_REF") {
        assert_bit_exact(&frames, &p);
        eprintln!(
            "m577: {} open-GOP frames bit-exact in display order",
            frames.len()
        );
    }
}
