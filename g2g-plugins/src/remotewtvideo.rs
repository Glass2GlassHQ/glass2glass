// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::time::Duration;

use tokio::time::Instant;

use g2g_core::frame::Frame;
use g2g_core::wire::{decode_packet, encode_packet, WireError};
use g2g_core::{G2gError, NetworkFailure, PipelinePacket, PropError, PropValue};

use crate::remotewire::map_wire;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoMode {
    Stream,
    StreamPerFrame,
    Datagrams,
}

impl VideoMode {
    pub(crate) fn name(self) -> &'static str {
        match self {
            VideoMode::Stream => "stream",
            VideoMode::StreamPerFrame => "stream-per-frame",
            VideoMode::Datagrams => "datagrams",
        }
    }

    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "stream" => Some(VideoMode::Stream),
            "stream-per-frame" => Some(VideoMode::StreamPerFrame),
            "datagrams" => Some(VideoMode::Datagrams),
            _ => None,
        }
    }
}

pub(crate) fn set_video_mode(target: &mut VideoMode, value: &PropValue) -> Result<(), PropError> {
    let name = value.as_str().ok_or(PropError::Type)?;
    *target = VideoMode::from_name(name).ok_or(PropError::Value)?;
    Ok(())
}

pub(crate) const KEYFRAME_REQUEST: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PacketOrder {
    pub(crate) frames_before: u64,
    pub(crate) controls_before: u64,
}

const ORDER_FIELD_LEN: usize = 8;
const ORDER_HEADER_LEN: usize = 2 * ORDER_FIELD_LEN;

impl PacketOrder {
    pub(crate) fn encode(self, packet: &PipelinePacket) -> Result<Vec<u8>, G2gError> {
        let body = encode_packet(packet).map_err(map_wire)?;
        let mut bytes = Vec::with_capacity(ORDER_HEADER_LEN + body.len());
        bytes.extend_from_slice(&self.frames_before.to_le_bytes());
        bytes.extend_from_slice(&self.controls_before.to_le_bytes());
        bytes.extend_from_slice(&body);
        Ok(bytes)
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<(Self, PipelinePacket), G2gError> {
        let (frames_before, rest) = bytes
            .split_first_chunk::<ORDER_FIELD_LEN>()
            .ok_or(map_wire(WireError::Truncated))?;
        let (controls_before, body) = rest
            .split_first_chunk::<ORDER_FIELD_LEN>()
            .ok_or(map_wire(WireError::Truncated))?;
        let order = PacketOrder {
            frames_before: u64::from_le_bytes(*frames_before),
            controls_before: u64::from_le_bytes(*controls_before),
        };
        Ok((order, decode_packet(body).map_err(map_wire)?))
    }
}

#[derive(Debug)]
pub(crate) enum NextPacket {
    Ready(PipelinePacket),
    Wait { deadline: Option<Instant> },
}

#[derive(Debug, Default)]
pub(crate) struct PacketReorder {
    next_frame: u64,
    next_control: u64,
    // keyed by frame index, the value carries the frame's controls_before
    frames: BTreeMap<u64, (u64, Frame)>,
    // keyed by control index, the value carries the control's frames_before
    controls: BTreeMap<u64, (u64, PipelinePacket)>,
    awaiting_keyframe: bool,
    gap_started: Option<(u64, Instant)>,
    keyframe_wanted: bool,
}

impl PacketReorder {
    pub(crate) fn insert(
        &mut self,
        order: PacketOrder,
        packet: PipelinePacket,
    ) -> Result<(), G2gError> {
        match packet {
            PipelinePacket::DataFrame(frame) => {
                // frames behind next_frame were skipped, they arrived too late to show
                if order.frames_before < self.next_frame {
                    return Ok(());
                }
                self.frames
                    .insert(order.frames_before, (order.controls_before, frame));
            }
            other => {
                if order.controls_before < self.next_control
                    || self.controls.contains_key(&order.controls_before)
                {
                    return Err(G2gError::Network(NetworkFailure::Stream));
                }
                self.controls
                    .insert(order.controls_before, (order.frames_before, other));
            }
        }
        Ok(())
    }

    pub(crate) fn next_frame(&self) -> u64 {
        self.next_frame
    }

    pub(crate) fn take_keyframe_wanted(&mut self) -> bool {
        core::mem::take(&mut self.keyframe_wanted)
    }

    pub(crate) fn pop(&mut self, now: Instant, frame_timeout: Option<Duration>) -> NextPacket {
        loop {
            if let Some(control) = self.controls.first_entry() {
                if *control.key() == self.next_control && control.get().0 <= self.next_frame {
                    self.next_control = self.next_control.saturating_add(1);
                    return NextPacket::Ready(control.remove().1);
                }
            }
            if let Some(frame) = self.frames.first_entry() {
                if *frame.key() == self.next_frame {
                    if frame.get().0 > self.next_control {
                        return NextPacket::Wait { deadline: None };
                    }
                    let (_, frame) = frame.remove();
                    self.next_frame = self.next_frame.saturating_add(1);
                    self.gap_started = None;
                    if frame.timing.keyframe {
                        self.awaiting_keyframe = false;
                    } else if self.awaiting_keyframe {
                        continue;
                    }
                    return NextPacket::Ready(PipelinePacket::DataFrame(frame));
                }
            }
            let Some(target) = self.skip_target(now, frame_timeout) else {
                return NextPacket::Wait {
                    deadline: self.gap_deadline(frame_timeout),
                };
            };
            let target_controls_before = self
                .frames
                .get(&target)
                .map_or(0, |(controls, _)| *controls);
            let jump = if target_controls_before > self.next_control {
                // controls sent before the target still come first, at the frame position they were sent at
                match self.controls.first_key_value() {
                    Some((&index, &(frames_before, _))) if index == self.next_control => {
                        frames_before.min(target)
                    }
                    _ => return NextPacket::Wait { deadline: None },
                }
            } else {
                target
            };
            self.frames = self.frames.split_off(&jump);
            self.next_frame = jump;
            self.awaiting_keyframe = true;
            self.gap_started = None;
        }
    }

    fn skip_target(&mut self, now: Instant, frame_timeout: Option<Duration>) -> Option<u64> {
        let first_ready = self.frames.first_key_value().map(|(&index, _)| index);
        let next_control_at = self
            .controls
            .get(&self.next_control)
            .map(|&(frames_before, _)| frames_before);
        let first_waiting = first_ready.into_iter().chain(next_control_at).min()?;
        // senders reset every unfinished frame before a keyframe, so a missing frame ahead of one never arrives
        let keyframe_ahead = self
            .frames
            .iter()
            .find(|(_, (_, frame))| frame.timing.keyframe)
            .map(|(&index, _)| index);
        if keyframe_ahead.is_some() {
            return keyframe_ahead;
        }
        let timeout = frame_timeout?;
        let started = match self.gap_started {
            Some((index, at)) if index == self.next_frame => at,
            _ => {
                self.gap_started = Some((self.next_frame, now));
                now
            }
        };
        if now < started + timeout {
            return None;
        }
        self.keyframe_wanted = true;
        Some(first_waiting)
    }

    fn gap_deadline(&self, frame_timeout: Option<Duration>) -> Option<Instant> {
        let (index, started) = self.gap_started?;
        if index != self.next_frame {
            return None;
        }
        Some(started + frame_timeout?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;
    use g2g_core::memory::{MemoryDomain, SystemSlice};
    use g2g_core::FrameTiming;

    const TIMEOUT: Duration = Duration::from_millis(33);

    fn frame(sequence: u64, keyframe: bool) -> PipelinePacket {
        PipelinePacket::DataFrame(Frame {
            domain: MemoryDomain::System(SystemSlice::from_boxed(vec![0u8; 4].into_boxed_slice())),
            timing: FrameTiming {
                keyframe,
                ..FrameTiming::default()
            },
            sequence,
            meta: Default::default(),
        })
    }

    fn order(frames_before: u64, controls_before: u64) -> PacketOrder {
        PacketOrder {
            frames_before,
            controls_before,
        }
    }

    fn drain(reorder: &mut PacketReorder, now: Instant, timeout: Option<Duration>) -> Vec<String> {
        let mut out = Vec::new();
        while let NextPacket::Ready(packet) = reorder.pop(now, timeout) {
            out.push(match packet {
                PipelinePacket::DataFrame(frame) => alloc::format!("f{}", frame.sequence),
                PipelinePacket::Eos => "eos".into(),
                _ => "control".into(),
            });
        }
        out
    }

    #[test]
    fn frames_and_controls_leave_in_send_order() {
        let mut reorder = PacketReorder::default();
        let now = Instant::now();
        // f0, f1, control, f2, eos as sent, received out of order
        reorder.insert(order(3, 1), PipelinePacket::Eos).unwrap();
        reorder.insert(order(2, 1), frame(2, false)).unwrap();
        reorder.insert(order(2, 0), PipelinePacket::Flush).unwrap();
        reorder.insert(order(1, 0), frame(1, false)).unwrap();
        assert!(
            drain(&mut reorder, now, None).is_empty(),
            "f0 is still missing"
        );
        reorder.insert(order(0, 0), frame(0, true)).unwrap();
        assert_eq!(
            drain(&mut reorder, now, None),
            ["f0", "f1", "control", "f2", "eos"]
        );
    }

    #[test]
    fn a_later_keyframe_skips_the_missing_frames_and_the_deltas_before_it() {
        let mut reorder = PacketReorder::default();
        let now = Instant::now();
        reorder.insert(order(0, 0), frame(0, true)).unwrap();
        // f1 was reset by the sender, f2 depends on it
        reorder.insert(order(2, 0), frame(2, false)).unwrap();
        reorder.insert(order(3, 0), frame(3, true)).unwrap();
        reorder.insert(order(4, 0), frame(4, false)).unwrap();
        assert_eq!(drain(&mut reorder, now, None), ["f0", "f3", "f4"]);
        assert!(
            !reorder.take_keyframe_wanted(),
            "the keyframe is already here"
        );
        reorder.insert(order(1, 0), frame(1, false)).unwrap();
        assert!(
            drain(&mut reorder, now, None).is_empty(),
            "late frames are dropped"
        );
    }

    #[test]
    fn a_frame_missing_past_the_timeout_is_skipped_and_a_keyframe_requested() {
        let mut reorder = PacketReorder::default();
        let start = Instant::now();
        reorder.insert(order(0, 0), frame(0, true)).unwrap();
        reorder.insert(order(2, 0), frame(2, false)).unwrap();
        assert_eq!(drain(&mut reorder, start, Some(TIMEOUT)), ["f0"]);
        assert!(matches!(
            reorder.pop(start, Some(TIMEOUT)),
            NextPacket::Wait { deadline: Some(deadline) } if deadline == start + TIMEOUT
        ));
        assert!(drain(&mut reorder, start + TIMEOUT / 2, Some(TIMEOUT)).is_empty());
        // f2 depends on the lost f1, so nothing shows until a keyframe
        assert!(drain(&mut reorder, start + TIMEOUT, Some(TIMEOUT)).is_empty());
        assert!(reorder.take_keyframe_wanted());
        reorder.insert(order(3, 0), frame(3, false)).unwrap();
        reorder.insert(order(4, 0), frame(4, true)).unwrap();
        assert_eq!(drain(&mut reorder, start + TIMEOUT, Some(TIMEOUT)), ["f4"]);
    }

    #[test]
    fn a_lost_last_frame_does_not_hold_back_the_end_of_stream() {
        let mut reorder = PacketReorder::default();
        let start = Instant::now();
        reorder.insert(order(0, 0), frame(0, true)).unwrap();
        reorder.insert(order(2, 0), PipelinePacket::Eos).unwrap();
        assert_eq!(drain(&mut reorder, start, Some(TIMEOUT)), ["f0"]);
        assert_eq!(drain(&mut reorder, start + TIMEOUT, Some(TIMEOUT)), ["eos"]);
    }

    #[test]
    fn without_a_timeout_a_missing_frame_is_waited_for() {
        let mut reorder = PacketReorder::default();
        let start = Instant::now();
        reorder.insert(order(1, 0), frame(1, false)).unwrap();
        assert!(drain(&mut reorder, start + Duration::from_secs(60), None).is_empty());
        reorder.insert(order(0, 0), frame(0, true)).unwrap();
        assert_eq!(drain(&mut reorder, start, None), ["f0", "f1"]);
    }

    #[test]
    fn a_skip_stops_at_a_control_sent_between_the_missing_frame_and_the_keyframe() {
        let mut reorder = PacketReorder::default();
        let now = Instant::now();
        reorder.insert(order(0, 0), frame(0, true)).unwrap();
        // f1 lost, then a caps change, then keyframe f2 that needs it first
        reorder.insert(order(2, 1), frame(2, true)).unwrap();
        assert_eq!(
            drain(&mut reorder, now, None),
            ["f0"],
            "the control is in flight"
        );
        reorder.insert(order(2, 0), PipelinePacket::Flush).unwrap();
        assert_eq!(drain(&mut reorder, now, None), ["control", "f2"]);
    }

    #[test]
    fn a_far_ahead_frame_index_jumps_without_walking_the_gap() {
        let mut reorder = PacketReorder::default();
        let now = Instant::now();
        reorder
            .insert(order(u64::MAX - 1, 0), frame(9, true))
            .unwrap();
        assert_eq!(drain(&mut reorder, now, None), ["f9"]);
    }

    #[test]
    fn a_repeated_control_index_is_malformed() {
        let mut reorder = PacketReorder::default();
        reorder.insert(order(0, 0), PipelinePacket::Flush).unwrap();
        assert!(reorder.insert(order(0, 0), PipelinePacket::Flush).is_err());
    }

    #[test]
    fn the_order_header_round_trips_and_rejects_a_short_packet() {
        let header = order(7, 3);
        let bytes = header.encode(&PipelinePacket::Eos).unwrap();
        let (decoded, packet) = PacketOrder::decode(&bytes).unwrap();
        assert_eq!(decoded, header);
        assert!(matches!(packet, PipelinePacket::Eos));
        assert!(PacketOrder::decode(&bytes[..ORDER_HEADER_LEN - 1]).is_err());
    }
}
