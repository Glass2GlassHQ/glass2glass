//! Speaker channel positions and layouts for multichannel PCM. A
//! [`ChannelLayout`] is a bitmask in the WAV / ffmpeg bit order, and that bit
//! order (ascending) is also the interleave order of the samples, so a layout
//! fully describes which speaker each interleaved channel index feeds.
//!
//! [`Caps::Audio`](crate::Caps::Audio) carries only a channel count; the layout
//! for a count follows the ffmpeg default-layout convention
//! ([`ChannelLayout::default_for`]), which is what the decode path emits.

/// One speaker position. The discriminant is the WAV / ffmpeg mask bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChannelPosition {
    /// Front left.
    Fl = 0,
    /// Front right.
    Fr = 1,
    /// Front center.
    Fc = 2,
    /// Low-frequency effects (subwoofer).
    Lfe = 3,
    /// Back left.
    Bl = 4,
    /// Back right.
    Br = 5,
    /// Front left-of-center.
    Flc = 6,
    /// Front right-of-center.
    Frc = 7,
    /// Back center.
    Bc = 8,
    /// Side left.
    Sl = 9,
    /// Side right.
    Sr = 10,
}

impl ChannelPosition {
    /// Every position, in mask-bit (interleave) order.
    pub const ALL: [ChannelPosition; 11] = [
        ChannelPosition::Fl,
        ChannelPosition::Fr,
        ChannelPosition::Fc,
        ChannelPosition::Lfe,
        ChannelPosition::Bl,
        ChannelPosition::Br,
        ChannelPosition::Flc,
        ChannelPosition::Frc,
        ChannelPosition::Bc,
        ChannelPosition::Sl,
        ChannelPosition::Sr,
    ];

    const fn bit(self) -> u16 {
        1 << (self as u16)
    }
}

/// A set of speaker positions; ascending bit order is the interleave order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelLayout(u16);

impl ChannelLayout {
    /// Mono: front center only.
    pub const MONO: ChannelLayout = ChannelLayout::of(&[ChannelPosition::Fc]);
    /// Stereo: front left + right.
    pub const STEREO: ChannelLayout =
        ChannelLayout::of(&[ChannelPosition::Fl, ChannelPosition::Fr]);
    /// 5.1 (back): FL FR FC LFE BL BR.
    pub const SURROUND_5_1: ChannelLayout = ChannelLayout::of(&[
        ChannelPosition::Fl,
        ChannelPosition::Fr,
        ChannelPosition::Fc,
        ChannelPosition::Lfe,
        ChannelPosition::Bl,
        ChannelPosition::Br,
    ]);
    /// 7.1: FL FR FC LFE BL BR SL SR.
    pub const SURROUND_7_1: ChannelLayout = ChannelLayout::of(&[
        ChannelPosition::Fl,
        ChannelPosition::Fr,
        ChannelPosition::Fc,
        ChannelPosition::Lfe,
        ChannelPosition::Bl,
        ChannelPosition::Br,
        ChannelPosition::Sl,
        ChannelPosition::Sr,
    ]);

    /// Layout from a set of positions.
    pub const fn of(positions: &[ChannelPosition]) -> Self {
        let mut mask = 0u16;
        let mut i = 0;
        while i < positions.len() {
            mask |= positions[i].bit();
            i += 1;
        }
        ChannelLayout(mask)
    }

    /// The conventional layout for a channel count (the ffmpeg default-layout
    /// table, which the decode path emits): mono, stereo, 2.1, 4.0, 5.0, 5.1,
    /// 6.1, 7.1. `None` for 0 or > 8 channels.
    pub const fn default_for(channels: u8) -> Option<Self> {
        use ChannelPosition::*;
        Some(match channels {
            1 => Self::MONO,
            2 => Self::STEREO,
            3 => Self::of(&[Fl, Fr, Lfe]),
            4 => Self::of(&[Fl, Fr, Fc, Bc]),
            5 => Self::of(&[Fl, Fr, Fc, Bl, Br]),
            6 => Self::SURROUND_5_1,
            7 => Self::of(&[Fl, Fr, Fc, Lfe, Bc, Sl, Sr]),
            8 => Self::SURROUND_7_1,
            _ => return None,
        })
    }

    /// Number of channels in the layout.
    pub const fn channels(self) -> u8 {
        self.0.count_ones() as u8
    }

    pub const fn contains(self, position: ChannelPosition) -> bool {
        self.0 & position.bit() != 0
    }

    /// The interleaved channel index of `position` (its rank among the set
    /// bits), or `None` if the layout lacks it.
    pub const fn index_of(self, position: ChannelPosition) -> Option<usize> {
        if !self.contains(position) {
            return None;
        }
        Some((self.0 & (position.bit() - 1)).count_ones() as usize)
    }

    /// The positions in interleave order.
    pub fn positions(self) -> impl Iterator<Item = ChannelPosition> {
        ChannelPosition::ALL
            .into_iter()
            .filter(move |p| self.contains(*p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_layouts_match_their_counts() {
        for n in 1..=8 {
            let layout = ChannelLayout::default_for(n).expect("default exists");
            assert_eq!(layout.channels(), n, "count {n}");
        }
        assert_eq!(ChannelLayout::default_for(0), None);
        assert_eq!(ChannelLayout::default_for(9), None);
    }

    #[test]
    fn index_follows_interleave_order() {
        // 5.1 interleaves FL FR FC LFE BL BR.
        let l = ChannelLayout::SURROUND_5_1;
        assert_eq!(l.index_of(ChannelPosition::Fl), Some(0));
        assert_eq!(l.index_of(ChannelPosition::Fc), Some(2));
        assert_eq!(l.index_of(ChannelPosition::Lfe), Some(3));
        assert_eq!(l.index_of(ChannelPosition::Br), Some(5));
        assert_eq!(l.index_of(ChannelPosition::Sl), None);
        let expected = [
            ChannelPosition::Fl,
            ChannelPosition::Fr,
            ChannelPosition::Fc,
            ChannelPosition::Lfe,
            ChannelPosition::Bl,
            ChannelPosition::Br,
        ];
        assert!(l.positions().eq(expected));
    }
}
