// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::time::Duration;

use bytes::{BufMut, Bytes, BytesMut};
use tokio::time::Instant;

use g2g_core::{G2gError, NetworkFailure};

use crate::remotewtio::wt_err;

const FRAME_INDEX_LEN: usize = 8;
const FRAME_LEN_LEN: usize = 4;
const COUNT_LEN: usize = 2;
const SHARD_HEADER_LEN: usize = FRAME_INDEX_LEN + FRAME_LEN_LEN + 3 * COUNT_LEN;

// reed-solomon-simd accepts up to this many original shards with up to this many recovery shards
const MAX_SHARDS: usize = 32_768;

const MAX_FRAMES_ASSEMBLING: usize = 64;

fn malformed() -> G2gError {
    G2gError::Network(NetworkFailure::Stream)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ShardHeader {
    frame_index: u64,
    frame_len: usize,
    original_count: usize,
    recovery_count: usize,
    shard_index: usize,
}

impl ShardHeader {
    fn write(self, datagram: &mut BytesMut) {
        datagram.put_u64_le(self.frame_index);
        datagram.put_u32_le(self.frame_len as u32);
        datagram.put_u16_le(self.original_count as u16);
        datagram.put_u16_le(self.recovery_count as u16);
        datagram.put_u16_le(self.shard_index as u16);
    }

    fn read(datagram: &[u8]) -> Result<(Self, &[u8]), G2gError> {
        let (frame_index, rest) = datagram
            .split_first_chunk::<FRAME_INDEX_LEN>()
            .ok_or_else(malformed)?;
        let (frame_len, rest) = rest
            .split_first_chunk::<FRAME_LEN_LEN>()
            .ok_or_else(malformed)?;
        let (original_count, rest) = rest
            .split_first_chunk::<COUNT_LEN>()
            .ok_or_else(malformed)?;
        let (recovery_count, rest) = rest
            .split_first_chunk::<COUNT_LEN>()
            .ok_or_else(malformed)?;
        let (shard_index, shard) = rest
            .split_first_chunk::<COUNT_LEN>()
            .ok_or_else(malformed)?;
        let header = ShardHeader {
            frame_index: u64::from_le_bytes(*frame_index),
            frame_len: u32::from_le_bytes(*frame_len) as usize,
            original_count: usize::from(u16::from_le_bytes(*original_count)),
            recovery_count: usize::from(u16::from_le_bytes(*recovery_count)),
            shard_index: usize::from(u16::from_le_bytes(*shard_index)),
        };
        let counts_valid = (1..=MAX_SHARDS).contains(&header.original_count)
            && (1..=MAX_SHARDS).contains(&header.recovery_count)
            && header.shard_index < header.original_count + header.recovery_count;
        let shard_len_valid = !shard.is_empty() && shard.len() % 2 == 0;
        let frame_len_valid = header.frame_len > 0
            && header.frame_len <= header.original_count.saturating_mul(shard.len());
        if !(counts_valid && shard_len_valid && frame_len_valid) {
            return Err(malformed());
        }
        Ok((header, shard))
    }
}

pub(crate) fn shard_frame(
    frame_index: u64,
    frame: &[u8],
    max_datagram_len: usize,
    path_loss: &PathLoss,
) -> Result<Vec<Bytes>, G2gError> {
    // reed-solomon-simd needs shards of an even length
    let max_shard_len = max_datagram_len.saturating_sub(SHARD_HEADER_LEN) & !1;
    if max_shard_len == 0 {
        return Err(wt_err(max_datagram_len));
    }
    let frame_len = u32::try_from(frame.len()).map_err(wt_err)?;
    let original_count = frame.len().div_ceil(max_shard_len);
    if original_count > MAX_SHARDS {
        return Err(wt_err(original_count));
    }
    let shard_len = frame.len().div_ceil(original_count).next_multiple_of(2);
    let recovery_count = path_loss.recovery_count(original_count);
    let mut originals: Vec<Vec<u8>> = frame.chunks(shard_len).map(<[u8]>::to_vec).collect();
    if let Some(last) = originals.last_mut() {
        last.resize(shard_len, 0);
    }
    let recoveries =
        reed_solomon_simd::encode(original_count, recovery_count, &originals).map_err(wt_err)?;
    let shards = originals.iter().chain(recoveries.iter());
    Ok(shards
        .enumerate()
        .map(|(shard_index, shard)| {
            let mut datagram = BytesMut::with_capacity(SHARD_HEADER_LEN + shard_len);
            ShardHeader {
                frame_index,
                frame_len: frame_len as usize,
                original_count,
                recovery_count,
                shard_index,
            }
            .write(&mut datagram);
            datagram.extend_from_slice(shard);
            datagram.freeze()
        })
        .collect())
}

#[derive(Debug)]
struct FrameShards {
    first: ShardHeader,
    shard_len: usize,
    originals: BTreeMap<usize, Bytes>,
    recoveries: BTreeMap<usize, Bytes>,
}

impl FrameShards {
    fn rebuild(&self) -> Result<Vec<u8>, G2gError> {
        let mut frame = Vec::with_capacity(self.first.original_count * self.shard_len);
        if self.originals.len() == self.first.original_count {
            for shard in self.originals.values() {
                frame.extend_from_slice(shard);
            }
        } else {
            let recovery_offset = self.first.original_count;
            let restored = reed_solomon_simd::decode(
                self.first.original_count,
                self.first.recovery_count,
                self.originals.iter().map(|(&index, shard)| (index, shard)),
                self.recoveries
                    .iter()
                    .map(|(&index, shard)| (index - recovery_offset, shard)),
            )
            .map_err(|_| malformed())?;
            for index in 0..self.first.original_count {
                let shard = match self.originals.get(&index) {
                    Some(shard) => shard.as_ref(),
                    None => restored.get(&index).ok_or_else(malformed)?.as_slice(),
                };
                frame.extend_from_slice(shard);
            }
        }
        frame.truncate(self.first.frame_len);
        Ok(frame)
    }
}

#[derive(Debug)]
enum FrameAssembly {
    Collecting(FrameShards),
    Rebuilt,
}

#[derive(Debug, Default)]
pub(crate) struct ShardAssembly {
    frames: BTreeMap<u64, FrameAssembly>,
}

impl ShardAssembly {
    pub(crate) fn insert(
        &mut self,
        datagram: &Bytes,
        next_frame: u64,
    ) -> Result<Option<Vec<u8>>, G2gError> {
        let (header, shard) = ShardHeader::read(datagram)?;
        self.frames = self.frames.split_off(&next_frame);
        if header.frame_index < next_frame {
            return Ok(None);
        }
        if !self.frames.contains_key(&header.frame_index)
            && self.frames.len() >= MAX_FRAMES_ASSEMBLING
        {
            self.frames.pop_first();
        }
        let assembly = self.frames.entry(header.frame_index).or_insert_with(|| {
            FrameAssembly::Collecting(FrameShards {
                first: header,
                shard_len: shard.len(),
                originals: BTreeMap::new(),
                recoveries: BTreeMap::new(),
            })
        });
        let FrameAssembly::Collecting(shards) = assembly else {
            return Ok(None);
        };
        let same_frame = ShardHeader {
            shard_index: header.shard_index,
            ..shards.first
        } == header;
        if !same_frame || shard.len() != shards.shard_len {
            return Err(malformed());
        }
        let shard = datagram.slice(SHARD_HEADER_LEN..);
        let into = if header.shard_index < header.original_count {
            &mut shards.originals
        } else {
            &mut shards.recoveries
        };
        if into.insert(header.shard_index, shard).is_some() {
            return Err(malformed());
        }
        if shards.originals.len() + shards.recoveries.len() < header.original_count {
            return Ok(None);
        }
        let frame = shards.rebuild()?;
        *assembly = FrameAssembly::Rebuilt;
        Ok(Some(frame))
    }
}

const INITIAL_LOSS: f64 = 0.05;
// a loss estimate below one half keeps the repair search finite
const MAX_LOSS: f64 = 0.3;
const FRAME_LOSS_TARGET: f64 = 0.01;
const NEW_LOSS_WEIGHT: f64 = 0.25;
const MEASURE_PERIOD: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PathCounts {
    pub(crate) lost_packets: u64,
    pub(crate) sent_packets: u64,
}

fn frame_loss_probability(original_count: usize, recovery_count: usize, loss: f64) -> f64 {
    let shard_count = original_count + recovery_count;
    let lost_to_kept = (loss / (1.0 - loss)).ln();
    let mut log_probability = shard_count as f64 * (1.0 - loss).ln();
    let mut rebuilt = 0.0;
    for lost in 0..=recovery_count {
        rebuilt += log_probability.exp();
        log_probability += ((shard_count - lost) as f64 / (lost + 1) as f64).ln() + lost_to_kept;
    }
    (1.0 - rebuilt).max(0.0)
}

#[derive(Debug)]
pub(crate) struct PathLoss {
    smoothed: f64,
    last: PathCounts,
    measured_at: Instant,
}

impl PathLoss {
    pub(crate) fn new(now: Instant) -> Self {
        Self {
            smoothed: INITIAL_LOSS,
            last: PathCounts {
                lost_packets: 0,
                sent_packets: 0,
            },
            measured_at: now,
        }
    }

    pub(crate) fn observe(&mut self, now: Instant, counts: PathCounts) {
        if now < self.measured_at + MEASURE_PERIOD {
            return;
        }
        let sent = counts.sent_packets.saturating_sub(self.last.sent_packets);
        let lost = counts.lost_packets.saturating_sub(self.last.lost_packets);
        self.last = counts;
        self.measured_at = now;
        if sent == 0 {
            return;
        }
        let loss = (lost as f64 / sent as f64).min(1.0);
        self.smoothed = (self.smoothed + NEW_LOSS_WEIGHT * (loss - self.smoothed)).min(MAX_LOSS);
    }

    fn recovery_count(&self, original_count: usize) -> usize {
        let (mut fewest, mut most) = (1, MAX_SHARDS);
        while fewest < most {
            let middle = (fewest + most) / 2;
            if frame_loss_probability(original_count, middle, self.smoothed) <= FRAME_LOSS_TARGET {
                most = middle;
            } else {
                fewest = middle + 1;
            }
        }
        fewest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DATAGRAM_LEN: usize = 1200;

    fn frame(len: usize) -> Vec<u8> {
        (0..len).map(|index| (index * 7 + 3) as u8).collect()
    }

    fn reassemble(shards: &[Bytes]) -> Option<Vec<u8>> {
        let mut assembly = ShardAssembly::default();
        let mut rebuilt = None;
        for shard in shards {
            if let Some(frame) = assembly.insert(shard, 0).expect("a well-formed shard") {
                rebuilt = Some(frame);
            }
        }
        rebuilt
    }

    #[test]
    fn a_frame_rebuilds_from_any_shards_as_many_as_its_originals() {
        let path_loss = PathLoss::new(Instant::now());
        let sent = frame(10_000);
        let shards = shard_frame(0, &sent, DATAGRAM_LEN, &path_loss).unwrap();
        let originals = sent.len().div_ceil(DATAGRAM_LEN - SHARD_HEADER_LEN);
        let recoveries = shards.len() - originals;
        assert_eq!(recoveries, path_loss.recovery_count(originals));
        assert!(shards.iter().all(|shard| shard.len() <= DATAGRAM_LEN));
        // lose as many originals as there are recovery shards
        let survivors: Vec<Bytes> = shards[recoveries..].to_vec();
        assert_eq!(reassemble(&survivors), Some(sent));
    }

    #[test]
    fn a_frame_missing_more_shards_than_its_repair_stays_incomplete() {
        let path_loss = PathLoss::new(Instant::now());
        let sent = frame(10_000);
        let shards = shard_frame(0, &sent, DATAGRAM_LEN, &path_loss).unwrap();
        let originals = sent.len().div_ceil(DATAGRAM_LEN - SHARD_HEADER_LEN);
        let survivors: Vec<Bytes> = shards[shards.len() - originals + 1..].to_vec();
        assert_eq!(reassemble(&survivors), None);
    }

    #[test]
    fn a_one_shard_frame_rebuilds_from_its_repair_shard() {
        let path_loss = PathLoss::new(Instant::now());
        let sent = frame(301);
        let shards = shard_frame(5, &sent, DATAGRAM_LEN, &path_loss).unwrap();
        assert_eq!(shards.len(), 2, "one original and one repair shard");
        let mut assembly = ShardAssembly::default();
        assert_eq!(assembly.insert(&shards[1], 0).unwrap(), Some(sent));
        assert_eq!(
            assembly.insert(&shards[0], 0).unwrap(),
            None,
            "a late shard of a rebuilt frame is dropped"
        );
    }

    #[test]
    fn shards_of_a_frame_already_past_are_dropped() {
        let path_loss = PathLoss::new(Instant::now());
        let shards = shard_frame(3, &frame(100), DATAGRAM_LEN, &path_loss).unwrap();
        let mut assembly = ShardAssembly::default();
        assert_eq!(assembly.insert(&shards[0], 4).unwrap(), None);
    }

    #[test]
    fn malformed_shards_are_rejected() {
        let path_loss = PathLoss::new(Instant::now());
        let shards = shard_frame(0, &frame(3000), DATAGRAM_LEN, &path_loss).unwrap();
        let mut assembly = ShardAssembly::default();
        assert!(assembly
            .insert(&shards[0].slice(..SHARD_HEADER_LEN - 1), 0)
            .is_err());
        let mut zero_originals = BytesMut::from(shards[0].as_ref());
        zero_originals[FRAME_INDEX_LEN + FRAME_LEN_LEN..][..COUNT_LEN].fill(0);
        assert!(assembly.insert(&zero_originals.freeze(), 0).is_err());
        let mut oversized = BytesMut::from(shards[0].as_ref());
        oversized[FRAME_INDEX_LEN..][..FRAME_LEN_LEN].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(assembly.insert(&oversized.freeze(), 0).is_err());
        assembly.insert(&shards[0], 0).unwrap();
        assert!(
            assembly.insert(&shards[0], 0).is_err(),
            "a repeated shard is malformed"
        );
    }

    #[test]
    fn the_frame_loss_odds_match_the_binomial_tail() {
        // five originals and one repair shard at 5% loss, from a reference binomial sum
        const REFERENCE: f64 = 0.032_773;
        assert!((frame_loss_probability(5, 1, 0.05) - REFERENCE).abs() < 1e-5);
        assert_eq!(frame_loss_probability(5, 1, 0.0), 0.0);
    }

    #[test]
    fn repair_keeps_each_frame_under_the_loss_target_at_the_measured_loss() {
        let start = Instant::now();
        let mut path_loss = PathLoss::new(start);
        let mut counts = PathCounts {
            lost_packets: 0,
            sent_packets: 0,
        };
        let mut now = start;
        for _ in 0..40 {
            now += MEASURE_PERIOD;
            counts.sent_packets += 1000;
            counts.lost_packets += 50;
            path_loss.observe(now, counts);
        }
        assert!((path_loss.smoothed - 0.05).abs() < 0.001);
        for (original_count, recovery_count) in [(1, 1), (5, 2), (10, 3)] {
            assert_eq!(path_loss.recovery_count(original_count), recovery_count);
        }
        for _ in 0..40 {
            now += MEASURE_PERIOD;
            counts.sent_packets += 1000;
            path_loss.observe(now, counts);
        }
        assert_eq!(
            path_loss.recovery_count(30),
            1,
            "a clean path keeps one repair shard"
        );
    }
}
