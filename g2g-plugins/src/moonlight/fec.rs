// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::vec::Vec;

// moonlight decodes with nanors, which caps data plus parity shards at 255
pub(crate) const MAX_SHARDS: usize = 255;

// GF(2^8) under x^8 + x^4 + x^3 + x^2 + 1, the field nanors and Moonlight use
const FIELD_POLYNOMIAL: u16 = 0x11d;

struct FieldTables {
    exponent: [u8; 255],
    logarithm: [u8; 256],
}

const FIELD: FieldTables = {
    let mut exponent = [0u8; 255];
    let mut logarithm = [0u8; 256];
    let mut value: u16 = 1;
    let mut power = 0;
    while power < 255 {
        exponent[power] = value as u8;
        logarithm[value as usize] = power as u8;
        value <<= 1;
        if value & 0x100 != 0 {
            value ^= FIELD_POLYNOMIAL;
        }
        power += 1;
    }
    FieldTables {
        exponent,
        logarithm,
    }
};

fn multiply(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let power = FIELD.logarithm[a as usize] as usize + FIELD.logarithm[b as usize] as usize;
    FIELD.exponent[power % 255]
}

fn inverse(a: u8) -> u8 {
    FIELD.exponent[(255 - FIELD.logarithm[a as usize] as usize) % 255]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShardCountError {
    pub(crate) data: usize,
    pub(crate) parity: usize,
}

impl core::fmt::Display for ShardCountError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} data and {} parity shards: Moonlight decodes 1 to {MAX_SHARDS} shards with at least one of each",
            self.data, self.parity
        )
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ParityMatrix {
    data_shards: usize,
    coefficients: Vec<u8>,
}

// moonlight's audio decoder copies these rows into nanors in place of the matrix nanors builds
const MOONLIGHT_AUDIO_ROWS: [[u8; MOONLIGHT_AUDIO_DATA_SHARDS]; 2] =
    [[0x77, 0x40, 0x38, 0x0e], [0xc7, 0xa7, 0x0d, 0x6c]];
pub(crate) const MOONLIGHT_AUDIO_DATA_SHARDS: usize = 4;

impl ParityMatrix {
    pub(crate) fn moonlight_audio() -> Self {
        Self {
            data_shards: MOONLIGHT_AUDIO_DATA_SHARDS,
            coefficients: MOONLIGHT_AUDIO_ROWS.concat(),
        }
    }

    pub(crate) fn cauchy(
        data_shards: usize,
        parity_shards: usize,
    ) -> Result<Self, ShardCountError> {
        if data_shards == 0 || parity_shards == 0 || data_shards + parity_shards > MAX_SHARDS {
            return Err(ShardCountError {
                data: data_shards,
                parity: parity_shards,
            });
        }
        let coefficients = (0..parity_shards)
            .flat_map(|row| {
                (0..data_shards).map(move |column| inverse(((parity_shards + column) ^ row) as u8))
            })
            .collect();
        Ok(Self {
            data_shards,
            coefficients,
        })
    }

    pub(crate) fn parity_shards(&self) -> usize {
        self.coefficients.len() / self.data_shards
    }

    pub(crate) fn encode<D: AsRef<[u8]>, P: AsMut<[u8]>>(&self, data: &[D], parity: &mut [P]) {
        assert_eq!(data.len(), self.data_shards, "data shard count");
        assert_eq!(parity.len(), self.parity_shards(), "parity shard count");
        let mut products = [0u8; 256];
        for (row, output) in self
            .coefficients
            .chunks_exact(self.data_shards)
            .zip(parity.iter_mut())
        {
            let output = output.as_mut();
            output.fill(0);
            for (&coefficient, input) in row.iter().zip(data) {
                let input = input.as_ref();
                assert_eq!(input.len(), output.len(), "shard length");
                if coefficient == 1 {
                    output.iter_mut().zip(input).for_each(|(o, i)| *o ^= i);
                    continue;
                }
                for (value, product) in products.iter_mut().enumerate() {
                    *product = multiply(coefficient, value as u8);
                }
                output
                    .iter_mut()
                    .zip(input)
                    .for_each(|(o, &i)| *o ^= products[i as usize]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // nanors computed these from the same layouts and data
    const NANORS_PARITY: &[u8] = include_bytes!("testdata/nanors-parity.bin");
    const LAYOUTS: [(usize, usize); 6] = [(1, 1), (4, 2), (10, 3), (32, 7), (200, 40), (253, 2)];
    const SHARD_BYTES: usize = 24;

    fn data_shard(index: usize) -> Vec<u8> {
        (0..SHARD_BYTES)
            .map(|offset| (index * 37 + offset * 11 + 5) as u8)
            .collect()
    }

    #[test]
    fn parity_matches_nanors() {
        let (shards, rest) = NANORS_PARITY.as_chunks::<SHARD_BYTES>();
        assert!(rest.is_empty(), "fixture ends in a partial shard");
        let mut expected = shards.iter().map(|shard| shard.as_slice());
        for (data_shards, parity_shards) in LAYOUTS {
            let data: Vec<Vec<u8>> = (0..data_shards).map(data_shard).collect();
            let mut parity = alloc::vec![alloc::vec![0xaau8; SHARD_BYTES]; parity_shards];
            ParityMatrix::cauchy(data_shards, parity_shards)
                .expect("layout fits")
                .encode(&data, &mut parity);
            for (index, shard) in parity.iter().enumerate() {
                assert_eq!(
                    Some(shard.as_slice()),
                    expected.next(),
                    "{data_shards}+{parity_shards}, parity shard {index}"
                );
            }
        }
        assert_eq!(
            expected.next(),
            None,
            "fixture has more shards than the layouts"
        );
    }

    #[test]
    fn audio_parity_matches_nanors_with_moonlights_matrix() {
        let expected: &[u8] = include_bytes!("testdata/nanors-audio-parity.bin");
        let data: Vec<Vec<u8>> = (0..MOONLIGHT_AUDIO_DATA_SHARDS).map(data_shard).collect();
        let mut parity = alloc::vec![alloc::vec![0xaau8; SHARD_BYTES]; 2];
        ParityMatrix::moonlight_audio().encode(&data, &mut parity);
        assert_eq!(parity.concat(), expected);
    }

    #[test]
    fn layouts_past_the_decoder_limit_are_refused() {
        assert!(ParityMatrix::cauchy(0, 1).is_err());
        assert!(ParityMatrix::cauchy(1, 0).is_err());
        assert!(ParityMatrix::cauchy(250, 6).is_err());
        assert!(ParityMatrix::cauchy(250, 5).is_ok());
    }
}
