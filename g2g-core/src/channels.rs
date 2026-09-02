//! Speaker channel positions and layouts for multichannel PCM. A
//! [`ChannelLayout`] is a bitmask in the WAV / ffmpeg bit order, and that bit
//! order (ascending) is also the interleave order of the samples, so a layout
//! fully describes which speaker each interleaved channel index feeds.
//!
//! [`Caps::Audio`](crate::Caps::Audio) carries a layout alongside its channel
//! count. [`ChannelLayout::UNSPECIFIED`] (mask 0) is the wildcard: it means the
//! producer does not know the positions, and every consumer then falls back to
//! the ffmpeg default-layout convention for the count
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

    /// The `GstAudioChannelPosition` this position is spelled as in a gst
    /// `channel-mask` bitmask. gst agrees with the WAV bit order up to
    /// `REAR_CENTER` (8) and then spends bit 9 on a second LFE g2g does not
    /// model, so the two side positions sit one bit higher.
    const fn gst_bit(self) -> u32 {
        match self {
            ChannelPosition::Sl => 10,
            ChannelPosition::Sr => 11,
            other => other as u32,
        }
    }

    const fn from_gst_bit(bit: u32) -> Option<Self> {
        use ChannelPosition::*;
        Some(match bit {
            0 => Fl,
            1 => Fr,
            2 => Fc,
            3 => Lfe,
            4 => Bl,
            5 => Br,
            6 => Flc,
            7 => Frc,
            8 => Bc,
            10 => Sl,
            11 => Sr,
            _ => return None,
        })
    }
}

/// A set of speaker positions; ascending bit order is the interleave order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelLayout(u16);

impl ChannelLayout {
    /// No positions declared: the wildcard a producer that does not know the
    /// speaker layout carries. Intersects with any layout, and every consumer
    /// substitutes [`ChannelLayout::default_for`] the channel count.
    pub const UNSPECIFIED: ChannelLayout = ChannelLayout(0);
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

    /// Number of channels in the layout. `0` for [`Self::UNSPECIFIED`].
    pub const fn channels(self) -> u8 {
        self.0.count_ones() as u8
    }

    /// Whether no positions are declared (the wildcard).
    pub const fn is_unspecified(self) -> bool {
        self.0 == 0
    }

    /// The raw WAV / ffmpeg bitmask, for a container field or the wire format.
    pub const fn mask(self) -> u16 {
        self.0
    }

    /// A layout from a raw WAV / ffmpeg bitmask (a WAV `dwChannelMask`). Bits
    /// above the positions g2g models are dropped, so an extended-layout file
    /// keeps the positions this crate can name.
    pub const fn from_mask(mask: u16) -> Self {
        let mut known = 0u16;
        let mut i = 0;
        while i < ChannelPosition::ALL.len() {
            known |= ChannelPosition::ALL[i].bit();
            i += 1;
        }
        ChannelLayout(mask & known)
    }

    /// Narrow one layout against another: [`Self::UNSPECIFIED`] is the wildcard
    /// and yields the other side, two equal layouts yield themselves, and two
    /// differing declared layouts do not overlap (`None`).
    pub const fn intersect(self, other: Self) -> Option<Self> {
        if self.is_unspecified() {
            return Some(other);
        }
        if other.is_unspecified() || self.0 == other.0 {
            return Some(self);
        }
        None
    }

    /// This layout when declared, otherwise the conventional layout for
    /// `channels`. The single place a consumer resolves the wildcard, so an
    /// unspecified layout behaves exactly as a bare channel count always did.
    pub const fn or_default_for(self, channels: u8) -> Option<Self> {
        if self.is_unspecified() {
            return Self::default_for(channels);
        }
        Some(self)
    }

    /// The GStreamer `channel-mask` bitmask for this layout.
    pub fn to_gst_mask(self) -> u64 {
        self.positions()
            .map(|p| 1u64 << p.gst_bit())
            .fold(0, |acc, b| acc | b)
    }

    /// A layout from a GStreamer `channel-mask` bitmask. `None` when the mask
    /// names a position g2g does not model (a top / bottom speaker, the second
    /// LFE), rather than silently dropping the channel.
    pub fn from_gst_mask(mask: u64) -> Option<Self> {
        let mut layout = 0u16;
        for bit in 0..u64::BITS {
            if mask & (1u64 << bit) == 0 {
                continue;
            }
            layout |= ChannelPosition::from_gst_bit(bit)?.bit();
        }
        Some(ChannelLayout(layout))
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

    #[test]
    fn unspecified_is_the_intersect_wildcard() {
        let any = ChannelLayout::UNSPECIFIED;
        assert!(any.is_unspecified());
        assert_eq!(
            any.intersect(ChannelLayout::SURROUND_5_1),
            Some(ChannelLayout::SURROUND_5_1)
        );
        assert_eq!(
            ChannelLayout::SURROUND_5_1.intersect(any),
            Some(ChannelLayout::SURROUND_5_1)
        );
        assert_eq!(any.intersect(any), Some(any));
        assert_eq!(
            ChannelLayout::SURROUND_5_1.intersect(ChannelLayout::SURROUND_5_1),
            Some(ChannelLayout::SURROUND_5_1)
        );
        // 5.0 and 5.1 are both six-ish surround shapes but different speakers.
        let five_zero = ChannelLayout::default_for(5).unwrap();
        assert_eq!(ChannelLayout::SURROUND_5_1.intersect(five_zero), None);
    }

    #[test]
    fn unspecified_falls_back_to_the_count_convention() {
        assert_eq!(
            ChannelLayout::UNSPECIFIED.or_default_for(6),
            Some(ChannelLayout::SURROUND_5_1)
        );
        // A declared layout wins even when it disagrees with the count's default.
        assert_eq!(
            ChannelLayout::SURROUND_7_1.or_default_for(2),
            Some(ChannelLayout::SURROUND_7_1)
        );
        assert_eq!(ChannelLayout::UNSPECIFIED.or_default_for(9), None);
    }

    #[test]
    fn gst_mask_round_trips_and_refuses_unknown_positions() {
        // gst spells 5.1 FL|FR|FC|LFE1|RL|RR = 0x3f, same as WAV.
        assert_eq!(ChannelLayout::SURROUND_5_1.to_gst_mask(), 0x3f);
        assert_eq!(
            ChannelLayout::from_gst_mask(0x3f),
            Some(ChannelLayout::SURROUND_5_1)
        );
        // 7.1 uses gst's SIDE_LEFT/SIDE_RIGHT at bits 10/11, not the WAV 9/10.
        assert_eq!(ChannelLayout::SURROUND_7_1.to_gst_mask(), 0xc3f);
        assert_eq!(
            ChannelLayout::from_gst_mask(0xc3f),
            Some(ChannelLayout::SURROUND_7_1)
        );
        for n in 1..=8 {
            let layout = ChannelLayout::default_for(n).unwrap();
            assert_eq!(
                ChannelLayout::from_gst_mask(layout.to_gst_mask()),
                Some(layout),
                "count {n}"
            );
        }
        // LFE2 (bit 9) and TOP_FRONT_LEFT (bit 12) have no g2g position.
        assert_eq!(ChannelLayout::from_gst_mask(1 << 9), None);
        assert_eq!(ChannelLayout::from_gst_mask(0x3f | (1 << 12)), None);
        assert_eq!(
            ChannelLayout::from_gst_mask(0),
            Some(ChannelLayout::UNSPECIFIED)
        );
    }

    #[test]
    fn raw_mask_keeps_only_modeled_positions() {
        assert_eq!(
            ChannelLayout::from_mask(ChannelLayout::SURROUND_5_1.mask()),
            ChannelLayout::SURROUND_5_1
        );
        // WAV bits 11.. are top speakers g2g does not model.
        assert_eq!(
            ChannelLayout::from_mask(ChannelLayout::STEREO.mask() | 0xf800),
            ChannelLayout::STEREO
        );
    }
}
