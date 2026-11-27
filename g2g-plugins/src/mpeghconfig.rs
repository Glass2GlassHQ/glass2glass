// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use crate::annexb::BitReader;

// USAC sampling frequency indices, 0 where the index is reserved
const USAC_SAMPLE_RATES: [u32; 31] = [
    96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350, 0, 0,
    57600, 51200, 40000, 38400, 34150, 28800, 25600, 20000, 19200, 17075, 14400, 12800, 9600, 0, 0,
    0,
];
const USAC_EXPLICIT_SAMPLE_RATE_INDEX: u32 = 31;

// the sample rate an mpegh3daConfig() declares after its profile and level byte
pub(crate) fn mpegh_config_sample_rate(config: &[u8]) -> Option<u32> {
    let mut reader = BitReader::new(config);
    reader.skip_bits(8)?;
    let index = reader.read_bits(5)?;
    let sample_rate = if index == USAC_EXPLICIT_SAMPLE_RATE_INDEX {
        reader.read_bits(24)?
    } else {
        *USAC_SAMPLE_RATES.get(usize::try_from(index).ok()?)?
    };
    (sample_rate != 0).then_some(sample_rate)
}

// ISO/IEC 14496-15 mhaC: configurationVersion, profile and level, reference channel layout, then a 16-bit length
const MHAC_CONFIG_OFFSET: usize = 5;

#[derive(Debug)]
pub(crate) struct MhaConfigRecord<'a> {
    pub(crate) reference_channel_layout: u8,
    pub(crate) config: &'a [u8],
}

pub(crate) fn mha_config_record(payload: &[u8]) -> Option<MhaConfigRecord<'_>> {
    let reference_channel_layout = *payload.get(2)?;
    let config_len = usize::from(u16::from_be_bytes([*payload.get(3)?, *payload.get(4)?]));
    let config = payload.get(MHAC_CONFIG_OFFSET..MHAC_CONFIG_OFFSET.checked_add(config_len)?)?;
    (!config.is_empty()).then_some(MhaConfigRecord {
        reference_channel_layout,
        config,
    })
}

// ISO/IEC 23091-3 ChannelConfiguration, 8 is not one
pub(crate) fn cicp_channel_count(index: u8) -> Option<u8> {
    Some(match index {
        1..=6 => index,
        7 => 8,
        9 => 3,
        10 => 4,
        11 => 7,
        12 => 8,
        13 => 24,
        14 => 8,
        15 => 12,
        16 => 10,
        17 => 12,
        18 => 14,
        19 => 12,
        20 => 14,
        _ => return None,
    })
}

// mpegh3daConfig() opening: profile and level 0x0b, sampling frequency index 3 (48 kHz), frame length index 1
#[cfg(test)]
pub(crate) const MPEGH_CONFIG_48K_PREFIX: [u8; 2] = [0x0b, 0b0001_1001];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mha_config_record_reads_the_layout_and_bounds_the_config() {
        let [profile, rate] = MPEGH_CONFIG_48K_PREFIX;
        let payload = [1, profile, 2, 0, 2, profile, rate];
        let record = mha_config_record(&payload).expect("a record");
        assert_eq!(record.reference_channel_layout, 2);
        assert_eq!(record.config, MPEGH_CONFIG_48K_PREFIX);
        assert_eq!(mpegh_config_sample_rate(record.config), Some(48_000));
        assert!(mha_config_record(&payload[..6]).is_none());
    }
}
