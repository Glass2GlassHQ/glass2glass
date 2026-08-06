//! M934: per-frame video PTS synthesis in the PS demuxer. A DVD stamps roughly
//! one PES packet per GOP; the pictures in between must not inherit the last
//! stamp verbatim (a pacing sink plays that as burst-and-freeze), but get
//! `gop_base + temporal_reference * frame_period`, exact across B-frame
//! reordering.

use g2g_plugins::psdemux::PsDemuxer;

/// 25 fps frame period in 90 kHz units.
const PERIOD: u64 = 3600;

/// A 33-bit timestamp in the five-byte PES field layout.
fn pts_field(pts90: u64) -> [u8; 5] {
    let p = pts90 & 0x1_ffff_ffff;
    [
        0x20 | ((p >> 29) as u8 & 0x0e) | 1,
        (p >> 22) as u8,
        (p >> 14) as u8 | 1,
        (p >> 7) as u8,
        ((p << 1) as u8) | 1,
    ]
}

/// One MPEG-2 video PES packet on stream 0xE0.
fn pes(pts90: Option<u64>, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::from([0x80u8, 0x00, 0x00]);
    if let Some(p) = pts90 {
        body[1] = 0x80;
        body[2] = 0x05;
        body.extend_from_slice(&pts_field(p));
    }
    body.extend_from_slice(payload);
    let mut out = Vec::from([0x00u8, 0x00, 0x01, 0xE0]);
    out.extend_from_slice(&(body.len() as u16).to_be_bytes());
    out.extend_from_slice(&body);
    out
}

/// A picture with the given `temporal_reference` and coding type (1 = I,
/// 2 = P, 3 = B), plus filler standing in for coded data.
fn pic(tref: u16, coding_type: u8) -> Vec<u8> {
    let mut out = Vec::from([
        0x00,
        0x00,
        0x01,
        0x00,
        (tref >> 2) as u8,
        (((tref & 0x3) as u8) << 6) | (coding_type << 3) | 0x07,
        0xFF,
    ]);
    out.extend_from_slice(&[0x42u8; 32]);
    out
}

/// A sequence header (720x480, 25 fps) plus a GOP header, opening a GOP.
fn seq_and_gop() -> Vec<u8> {
    let (w, h) = (720u32, 480u32);
    let mut out = Vec::from([
        0x00,
        0x00,
        0x01,
        0xB3,
        (w >> 4) as u8,
        (((w & 0xF) << 4) | (h >> 8)) as u8,
        h as u8,
        0x13, // aspect_ratio 1, frame_rate_code 3 (25 fps)
    ]);
    out.extend_from_slice(&[0x00, 0x00, 0x01, 0xB8, 0x00, 0x08, 0x00, 0x40]);
    out
}

/// The video units' PTS values, in emission (coded) order.
fn video_pts(demux: &mut PsDemuxer) -> Vec<Option<u64>> {
    demux.flush();
    demux
        .take_units()
        .into_iter()
        .filter(|u| u.stream_id == 0xE0)
        .map(|u| u.pts_90khz)
        .collect()
}

/// Two closed IBBP GOPs, PES-stamped only on each GOP's opening packet, the
/// shape a real DVD mux produces. Coded order I(0) P(3) B(1) B(2): every
/// picture's PTS must land on its own display slot, and the second, unstamped
/// GOP's base must advance past the four pictures of the first.
#[test]
fn unstamped_pictures_get_display_order_pts() {
    let mut demux = PsDemuxer::new();
    let base = 90_000u64;
    let mut gop1 = seq_and_gop();
    gop1.extend_from_slice(&pic(0, 1));
    demux.push_data(&pes(Some(base), &gop1));
    for (tref, ct) in [(3u16, 2u8), (1, 3), (2, 3)] {
        demux.push_data(&pes(None, &pic(tref, ct)));
    }
    let mut gop2 = seq_and_gop();
    gop2.extend_from_slice(&pic(0, 1));
    demux.push_data(&pes(None, &gop2));
    for (tref, ct) in [(3u16, 2u8), (1, 3), (2, 3)] {
        demux.push_data(&pes(None, &pic(tref, ct)));
    }

    let base2 = base + 4 * PERIOD;
    assert_eq!(
        video_pts(&mut demux),
        [0, 3, 1, 2, 4, 7, 5, 6]
            .iter()
            .map(|k| Some(base + k * PERIOD))
            .collect::<Vec<_>>(),
        "each picture presents at gop base + temporal_reference * period, \
         and the unstamped second GOP starts at {base2}"
    );
}

/// A stream that stamps every packet keeps its own timestamps untouched.
#[test]
fn real_stamps_pass_through_unchanged() {
    let mut demux = PsDemuxer::new();
    let base = 45_000u64;
    let mut first = seq_and_gop();
    first.extend_from_slice(&pic(0, 1));
    demux.push_data(&pes(Some(base), &first));
    // Deliberately off-grid stamps: synthesis must not "correct" them.
    let stamps = [(3u16, 2u8, 11_111u64), (1, 3, 22_222), (2, 3, 33_333)];
    for (tref, ct, off) in stamps {
        demux.push_data(&pes(Some(base + off), &pic(tref, ct)));
    }
    assert_eq!(
        video_pts(&mut demux),
        vec![
            Some(base),
            Some(base + 11_111),
            Some(base + 22_222),
            Some(base + 33_333)
        ],
    );
}

/// A mid-stream stamp re-anchors the GOP base, so later synthesized pictures
/// follow the real timeline (VBR drift) rather than the arithmetic one.
#[test]
fn a_mid_gop_stamp_reanchors_the_base() {
    let mut demux = PsDemuxer::new();
    let base = 90_000u64;
    let mut gop = seq_and_gop();
    gop.extend_from_slice(&pic(0, 1));
    demux.push_data(&pes(Some(base), &gop));
    // The P at display slot 3 arrives stamped 90 ticks later than arithmetic.
    let drift = 90u64;
    demux.push_data(&pes(Some(base + 3 * PERIOD + drift), &pic(3, 2)));
    demux.push_data(&pes(None, &pic(1, 3)));
    demux.push_data(&pes(None, &pic(2, 3)));
    assert_eq!(
        video_pts(&mut demux),
        vec![
            Some(base),
            Some(base + 3 * PERIOD + drift),
            Some(base + PERIOD + drift),
            Some(base + 2 * PERIOD + drift)
        ],
    );
}
