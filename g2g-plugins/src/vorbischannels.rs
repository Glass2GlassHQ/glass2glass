use g2g_core::{ChannelLayout, ChannelPosition};

// the Vorbis-order channel behind each channel of the ffmpeg default order
pub(crate) fn vorbis_layout(channels: u8) -> Option<(ChannelLayout, &'static [usize])> {
    use ChannelPosition::*;
    Some(match channels {
        3 => (ChannelLayout::of(&[Fl, Fr, Fc]), &[0, 2, 1]),
        4 => (ChannelLayout::of(&[Fl, Fr, Bl, Br]), &[0, 1, 2, 3]),
        5 => (ChannelLayout::of(&[Fl, Fr, Fc, Bl, Br]), &[0, 2, 1, 3, 4]),
        6 => (ChannelLayout::SURROUND_5_1, &[0, 2, 1, 5, 3, 4]),
        7 => (
            ChannelLayout::of(&[Fl, Fr, Fc, Lfe, Bc, Sl, Sr]),
            &[0, 2, 1, 6, 5, 3, 4],
        ),
        8 => (ChannelLayout::SURROUND_7_1, &[0, 2, 1, 7, 5, 6, 3, 4]),
        _ => return None,
    })
}
