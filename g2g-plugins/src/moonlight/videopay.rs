// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use aes_gcm::aead::{array::Array, AeadInOut, KeyInit};
use aes_gcm::Aes128Gcm;
use g2g_core::log::LogSource;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim,
    ElementMetadata, Frame, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec, Rate, Reconfigure, VideoCodec,
};

use super::fec::{ParityMatrix, MAX_SHARDS};
use crate::textencoding::hex_decode;

const RTP_HEADER_BYTES: usize = 12;
// moonlight skips four bytes after the RTP header, where an RTP extension header would sit
const RTP_PADDING_BYTES: usize = 4;
const VIDEO_PACKET_HEADER_BYTES: usize = 16;
const VIDEO_PACKET_HEADER_OFFSET: usize = RTP_HEADER_BYTES + RTP_PADDING_BYTES;
const PAYLOAD_OFFSET: usize = VIDEO_PACKET_HEADER_OFFSET + VIDEO_PACKET_HEADER_BYTES;
const FRAME_HEADER_BYTES: usize = 8;
// RTP version 2 with the extension bit set
const RTP_FIRST_BYTE: u8 = 0x90;
const FRAME_HEADER_TYPE: u8 = 0x01;
const MULTI_FEC_FLAGS: u8 = 0x10;
const FLAG_CONTAINS_PICTURE_DATA: u8 = 0x1;
const FLAG_END_OF_FRAME: u8 = 0x2;
const FLAG_START_OF_FRAME: u8 = 0x4;
const FRAME_TYPE_PREDICTED: u8 = 1;
const FRAME_TYPE_KEY: u8 = 2;
// two bits of block index on the wire
const MAX_FEC_BLOCKS: usize = 4;
// ten bits of data shard count on the wire
const MAX_DATA_SHARDS_PER_BLOCK: usize = 1023;
const GCM_KEY_BYTES: usize = 16;
const GCM_NONCE_BYTES: usize = 12;
const GCM_TAG_BYTES: usize = 16;
// moonlight reads the nonce, the frame number and the tag ahead of each encrypted packet
const ENCRYPTION_PREFIX_BYTES: usize = GCM_NONCE_BYTES + 4 + GCM_TAG_BYTES;
// moonlight tells video nonces from control nonces by this last byte
const VIDEO_NONCE_MARKER: u8 = b'V';
const RTP_CLOCK_HZ: u128 = 90_000;
const NANOSECONDS_PER_SECOND: u128 = 1_000_000_000;
// moonlight reads the host's processing time in units of 100 microseconds
const LATENCY_UNIT_NANOSECONDS: u64 = 100_000;
const MIN_PACKET_SIZE: usize = 200;
// 65_535 minus the IPv6 and UDP headers and the 16 bytes around each payload
const MAX_PACKET_SIZE: usize = 65_471;
const DEFAULT_PACKET_SIZE: usize = 1392;
const DEFAULT_FEC_PERCENTAGE: u8 = 20;
const DEFAULT_MINIMUM_FEC_PACKETS: u32 = 2;
// h264 and hevc keep at most 16 reference frames, so an older loss needs a key frame anyway
const SENT_FRAMES_KEPT: usize = 16;
const NO_INVALIDATION: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameTooLarge {
    pub(crate) data_shards: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EncodedFrame<'a> {
    pub(crate) access_unit: &'a [u8],
    pub(crate) keyframe: bool,
    pub(crate) rtp_timestamp: u32,
    pub(crate) latency_units: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockParity {
    shards: usize,
    percentage: u8,
}

// moonlight derives the parity count as ceil(data * percentage / 100)
fn block_parity(data_shards: usize, percentage: u8, minimum: u32) -> BlockParity {
    let room = MAX_SHARDS.saturating_sub(data_shards);
    let minimum = (minimum as usize).min(room);
    let minimum_percentage = match minimum {
        0 => 0,
        minimum => (minimum - 1) * 100 / data_shards + 1,
    };
    let maximum_percentage = (room * 100 / data_shards).min(u8::MAX as usize);
    let percentage = (percentage as usize)
        .max(minimum_percentage)
        .min(maximum_percentage);
    BlockParity {
        shards: (data_shards * percentage).div_ceil(100),
        percentage: percentage as u8,
    }
}

struct GcmSealer {
    cipher: Aes128Gcm,
    nonce_counter: u64,
}

impl core::fmt::Debug for GcmSealer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GcmSealer")
            .field("nonce_counter", &self.nonce_counter)
            .finish_non_exhaustive()
    }
}

impl GcmSealer {
    fn seal(&mut self, datagram: &mut [u8], frame_number: u32) {
        let mut nonce = [0u8; GCM_NONCE_BYTES];
        nonce[..8].copy_from_slice(&self.nonce_counter.to_le_bytes());
        nonce[GCM_NONCE_BYTES - 1] = VIDEO_NONCE_MARKER;
        self.nonce_counter += 1;
        let (prefix, body) = datagram.split_at_mut(ENCRYPTION_PREFIX_BYTES);
        let tag = self
            .cipher
            .encrypt_inout_detached(&Array(nonce), &[], body.into())
            .expect("a video packet is far below the AES-GCM length limit");
        prefix[..GCM_NONCE_BYTES].copy_from_slice(&nonce);
        prefix[GCM_NONCE_BYTES..GCM_NONCE_BYTES + 4].copy_from_slice(&frame_number.to_le_bytes());
        prefix[GCM_NONCE_BYTES + 4..].copy_from_slice(&tag.0);
    }
}

#[derive(Debug)]
pub(crate) struct VideoPacketizer {
    packet_size: usize,
    fec_percentage: u8,
    minimum_fec_packets: u32,
    sealer: Option<GcmSealer>,
    frame_number: u32,
    sequence_number: u32,
    parity_encoders: BTreeMap<(usize, usize), ParityMatrix>,
}

impl VideoPacketizer {
    pub(crate) fn new(
        packet_size: usize,
        fec_percentage: u8,
        minimum_fec_packets: u32,
        key: Option<[u8; GCM_KEY_BYTES]>,
    ) -> Self {
        assert!(
            packet_size > VIDEO_PACKET_HEADER_BYTES,
            "packet size {packet_size}"
        );
        Self {
            packet_size,
            fec_percentage,
            minimum_fec_packets,
            sealer: key.map(|key| GcmSealer {
                cipher: Aes128Gcm::new(&Array(key)),
                nonce_counter: 0,
            }),
            frame_number: 0,
            sequence_number: 0,
            parity_encoders: BTreeMap::new(),
        }
    }

    fn payload_bytes(&self) -> usize {
        self.packet_size - VIDEO_PACKET_HEADER_BYTES
    }

    pub(crate) fn packetize(
        &mut self,
        frame: EncodedFrame<'_>,
    ) -> Result<Vec<Vec<u8>>, FrameTooLarge> {
        let payload_bytes = self.payload_bytes();
        let stream_bytes = FRAME_HEADER_BYTES + frame.access_unit.len();
        let data_shards = stream_bytes.div_ceil(payload_bytes);
        if data_shards > MAX_FEC_BLOCKS * MAX_DATA_SHARDS_PER_BLOCK {
            return Err(FrameTooLarge { data_shards });
        }
        self.frame_number = self.frame_number.wrapping_add(1);
        let last_payload_bytes = stream_bytes - (data_shards - 1) * payload_bytes;
        let mut frame_header = [0u8; FRAME_HEADER_BYTES];
        frame_header[0] = FRAME_HEADER_TYPE;
        frame_header[1..3].copy_from_slice(&frame.latency_units.to_le_bytes());
        frame_header[3] = match frame.keyframe {
            true => FRAME_TYPE_KEY,
            false => FRAME_TYPE_PREDICTED,
        };
        frame_header[4..8].copy_from_slice(&(last_payload_bytes as u32).to_le_bytes());
        let stream = Stream {
            header: &frame_header,
            access_unit: frame.access_unit,
        };

        let blocks_at_full_parity = MAX_SHARDS * 100 / (100 + self.fec_percentage as usize);
        let blocks = data_shards
            .div_ceil(blocks_at_full_parity)
            .min(MAX_FEC_BLOCKS);
        let (shards_per_block, larger_blocks) = (data_shards / blocks, data_shards % blocks);
        let mut datagrams = Vec::new();
        let mut first_shard = 0;
        for block in 0..blocks {
            let block_shards = shards_per_block + usize::from(block < larger_blocks);
            let blocks_byte = ((block as u8) << 4) | (((blocks - 1) as u8) << 6);
            let block_datagrams = self.packetize_block(
                &stream,
                first_shard..first_shard + block_shards,
                blocks_byte,
                frame.rtp_timestamp,
            );
            datagrams.extend(block_datagrams);
            first_shard += block_shards;
        }
        Ok(datagrams)
    }

    fn packetize_block(
        &mut self,
        stream: &Stream<'_>,
        shards: core::ops::Range<usize>,
        blocks_byte: u8,
        rtp_timestamp: u32,
    ) -> Vec<Vec<u8>> {
        let payload_bytes = self.payload_bytes();
        let prefix = match self.sealer {
            Some(_) => ENCRYPTION_PREFIX_BYTES,
            None => 0,
        };
        let shard_bytes = PAYLOAD_OFFSET + payload_bytes;
        let data_shards = shards.len();
        let parity = block_parity(data_shards, self.fec_percentage, self.minimum_fec_packets);
        let fec_info = |index: usize| {
            ((index << 12) | (data_shards << 22) | ((parity.percentage as usize) << 4)) as u32
        };
        let mut datagrams: Vec<Vec<u8>> = (0..data_shards + parity.shards)
            .map(|_| alloc::vec![0u8; prefix + shard_bytes])
            .collect();
        for (index, (stream_shard, datagram)) in shards.zip(&mut datagrams).enumerate() {
            let shard = &mut datagram[prefix..];
            shard[0] = RTP_FIRST_BYTE;
            shard[2..4].copy_from_slice(&(self.sequence_number as u16).to_be_bytes());
            shard[4..8].copy_from_slice(&rtp_timestamp.to_be_bytes());
            let mut flags = FLAG_CONTAINS_PICTURE_DATA;
            if index == 0 {
                flags |= FLAG_START_OF_FRAME;
            }
            if index == data_shards - 1 {
                flags |= FLAG_END_OF_FRAME;
            }
            let header = &mut shard[VIDEO_PACKET_HEADER_OFFSET..PAYLOAD_OFFSET];
            header[0..4].copy_from_slice(&(self.sequence_number << 8).to_le_bytes());
            header[4..8].copy_from_slice(&self.frame_number.to_le_bytes());
            header[8] = flags;
            header[10] = MULTI_FEC_FLAGS;
            header[11] = blocks_byte;
            header[12..16].copy_from_slice(&fec_info(index).to_le_bytes());
            stream.copy_into(stream_shard * payload_bytes, &mut shard[PAYLOAD_OFFSET..]);
            self.sequence_number = self.sequence_number.wrapping_add(1);
        }
        if parity.shards > 0 {
            let encoder = self
                .parity_encoders
                .entry((data_shards, parity.shards))
                .or_insert_with(|| {
                    ParityMatrix::cauchy(data_shards, parity.shards)
                        .expect("block_parity keeps the block within MAX_SHARDS")
                });
            let (data, parity_datagrams) = datagrams.split_at_mut(data_shards);
            let data: Vec<&[u8]> = data.iter().map(|d| &d[prefix..]).collect();
            let mut parity_shards: Vec<&mut [u8]> = parity_datagrams
                .iter_mut()
                .map(|d| &mut d[prefix..])
                .collect();
            encoder.encode(&data, &mut parity_shards);
            for (offset, shard) in parity_shards.into_iter().enumerate() {
                shard[0] = RTP_FIRST_BYTE;
                shard[1] = 0;
                shard[2..4].copy_from_slice(&(self.sequence_number as u16).to_be_bytes());
                let header = &mut shard[VIDEO_PACKET_HEADER_OFFSET..PAYLOAD_OFFSET];
                header[4..8].copy_from_slice(&self.frame_number.to_le_bytes());
                header[11] = blocks_byte;
                header[12..16].copy_from_slice(&fec_info(data_shards + offset).to_le_bytes());
                self.sequence_number = self.sequence_number.wrapping_add(1);
            }
        }
        if let Some(sealer) = &mut self.sealer {
            for datagram in &mut datagrams {
                sealer.seal(datagram, self.frame_number);
            }
        }
        datagrams
    }
}

struct Stream<'a> {
    header: &'a [u8; FRAME_HEADER_BYTES],
    access_unit: &'a [u8],
}

impl Stream<'_> {
    fn copy_into(&self, offset: usize, destination: &mut [u8]) {
        let mut written = 0;
        if offset < FRAME_HEADER_BYTES {
            let header = &self.header[offset..];
            let count = header.len().min(destination.len());
            destination[..count].copy_from_slice(&header[..count]);
            written = count;
        }
        let start = offset.saturating_sub(FRAME_HEADER_BYTES);
        let Some(rest) = self.access_unit.get(start..) else {
            return;
        };
        let count = rest.len().min(destination.len() - written);
        destination[written..written + count].copy_from_slice(&rest[..count]);
    }
}

const PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new(
        "packet-size",
        PropKind::Uint,
        "bytes of video packet header and payload per packet, the size Moonlight asks for",
    )
    .with_range("200", "65471")
    .with_default("1392"),
    PropertySpec::new(
        "fec-percentage",
        PropKind::Uint,
        "parity packets per frame block, as a percentage of its data packets",
    )
    .with_range("0", "255")
    .with_default("20"),
    PropertySpec::new(
        "min-required-fec-packets",
        PropKind::Uint,
        "fewest parity packets per frame block, the minimum Moonlight asks for",
    )
    .with_default("2"),
    PropertySpec::new(
        "key",
        PropKind::Str,
        "AES-128 key as 32 hexadecimal digits, the rikey of the Moonlight launch",
    ),
    PropertySpec::new(
        "encrypt",
        PropKind::Bool,
        "seal every packet with AES-GCM under key",
    )
    .with_default("true"),
];

fn input_caps() -> Vec<Caps> {
    [VideoCodec::H264, VideoCodec::H265]
        .into_iter()
        .map(|codec| Caps::CompressedVideo {
            codec,
            width: Dim::Any,
            height: Dim::Any,
            framerate: Rate::Any,
            colorimetry: g2g_core::Colorimetry::UNKNOWN,
            pixel_aspect: g2g_core::PixelAspect::Any,
            alignment: g2g_core::BitstreamAlignment::AccessUnit,
        })
        .collect()
}

const OUTPUT_CAPS: Caps = Caps::ByteStream {
    encoding: ByteStreamEncoding::Datagram,
};

#[derive(Debug)]
pub struct RecoveryRequests {
    keyframe: AtomicBool,
    invalidate_from_frame: AtomicU32,
}

impl RecoveryRequests {
    pub fn request_keyframe(&self) {
        self.keyframe.store(true, Ordering::Relaxed);
    }

    // frame numbers are the ones Moonlight reads from the video packets
    pub fn invalidate_from(&self, frame_number: u32) {
        self.invalidate_from_frame
            .fetch_min(frame_number, Ordering::Relaxed);
    }
}

impl Default for RecoveryRequests {
    fn default() -> Self {
        Self {
            keyframe: AtomicBool::new(false),
            invalidate_from_frame: AtomicU32::new(NO_INVALIDATION),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct SentFrame {
    frame_number: u32,
    pts_ns: u64,
}

#[derive(Debug)]
pub struct MoonlightVideoPay {
    packet_size: usize,
    fec_percentage: u8,
    minimum_fec_packets: u32,
    key: Option<[u8; GCM_KEY_BYTES]>,
    encrypt: bool,
    recovery: Arc<RecoveryRequests>,
    sent_frames: VecDeque<SentFrame>,
    packetizer: Option<VideoPacketizer>,
}

impl Default for MoonlightVideoPay {
    fn default() -> Self {
        Self {
            packet_size: DEFAULT_PACKET_SIZE,
            fec_percentage: DEFAULT_FEC_PERCENTAGE,
            minimum_fec_packets: DEFAULT_MINIMUM_FEC_PACKETS,
            key: None,
            encrypt: true,
            recovery: Arc::default(),
            sent_frames: VecDeque::new(),
            packetizer: None,
        }
    }
}

impl MoonlightVideoPay {
    pub fn recovery_requests(&self) -> Arc<RecoveryRequests> {
        Arc::clone(&self.recovery)
    }

    async fn send_frame(&mut self, frame: Frame, out: &mut dyn OutputSink) -> Result<(), G2gError> {
        let access_unit = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let pts_ns = frame.timing.pts().ok_or_else(|| {
            g2g_core::g2g_error!(self, "an encoded frame has no presentation time");
            G2gError::InputRefused
        })?;
        let rtp_timestamp = (pts_ns as u128 * RTP_CLOCK_HZ / NANOSECONDS_PER_SECOND) as u32;
        let latency_units = match frame.timing.arrival_ns {
            0 => 0,
            arrival_ns => {
                let elapsed = g2g_core::metrics::monotonic_ns().saturating_sub(arrival_ns);
                (elapsed / LATENCY_UNIT_NANOSECONDS).min(u16::MAX as u64) as u16
            }
        };
        let packetizer = self.packetizer.as_mut().ok_or(G2gError::NotConfigured)?;
        let datagrams = match packetizer.packetize(EncodedFrame {
            access_unit,
            keyframe: frame.timing.keyframe,
            rtp_timestamp,
            latency_units,
        }) {
            Ok(datagrams) => datagrams,
            Err(FrameTooLarge { data_shards }) => {
                g2g_core::g2g_warn!(
                    self,
                    "dropped a frame of {data_shards} packets, past the {} Moonlight reads",
                    MAX_FEC_BLOCKS * MAX_DATA_SHARDS_PER_BLOCK
                );
                self.recovery.request_keyframe();
                return Ok(());
            }
        };
        if self.sent_frames.len() == SENT_FRAMES_KEPT {
            self.sent_frames.pop_front();
        }
        self.sent_frames.push_back(SentFrame {
            frame_number: packetizer.frame_number,
            pts_ns,
        });
        let timing = frame.timing;
        for datagram in datagrams {
            let domain = MemoryDomain::System(SystemSlice::from_boxed(datagram.into_boxed_slice()));
            out.push(PipelinePacket::DataFrame(Frame::new(
                domain,
                timing,
                frame.sequence,
            )))
            .await?;
        }
        Ok(())
    }
}

impl LogSource for MoonlightVideoPay {
    fn log_category(&self) -> &'static str {
        "moonlightvideopay"
    }
}

impl AsyncElement for MoonlightVideoPay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Moonlight video payloader",
            "Codec/Payloader/Network",
            "Cuts H.264 or HEVC access units into the FEC-protected, AES-GCM sealed packets \
             Moonlight clients read",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "packet-size" => {
                let size = value.as_uint().ok_or(PropError::Type)?;
                self.packet_size = usize::try_from(size)
                    .ok()
                    .filter(|size| (MIN_PACKET_SIZE..=MAX_PACKET_SIZE).contains(size))
                    .ok_or(PropError::Value)?;
            }
            "fec-percentage" => {
                let percentage = value.as_uint().ok_or(PropError::Type)?;
                self.fec_percentage = u8::try_from(percentage).map_err(|_| PropError::Value)?;
            }
            "min-required-fec-packets" => {
                let packets = value.as_uint().ok_or(PropError::Type)?;
                self.minimum_fec_packets = u32::try_from(packets).map_err(|_| PropError::Value)?;
            }
            "key" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                let key = hex_decode(text).ok_or(PropError::Value)?;
                self.key = Some(key.try_into().map_err(|_| PropError::Value)?);
            }
            "encrypt" => self.encrypt = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "packet-size" => Some(PropValue::Uint(self.packet_size as u64)),
            "fec-percentage" => Some(PropValue::Uint(u64::from(self.fec_percentage))),
            "min-required-fec-packets" => {
                Some(PropValue::Uint(u64::from(self.minimum_fec_packets)))
            }
            // the key is a session secret
            "key" => Some(PropValue::Str(String::new())),
            "encrypt" => Some(PropValue::Bool(self.encrypt)),
            _ => None,
        }
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        input_caps()
            .iter()
            .find_map(|accepted| upstream_caps.intersect(accepted).ok())
            .ok_or(G2gError::CapsMismatch)
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::Mapping(Vec::from([(
            CapsSet::from_alternatives(input_caps()),
            CapsSet::one(OUTPUT_CAPS),
        )]))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.intercept_caps(absolute_caps)?;
        // moonlight drops frames numbered below the last it received, and gcm nonces must not repeat
        if self.packetizer.is_some() {
            return Ok(ConfigureOutcome::Accepted);
        }
        let key = match (self.encrypt, self.key) {
            (false, _) => None,
            (true, Some(key)) => Some(key),
            (true, None) => {
                g2g_core::g2g_error!(self, "encrypt is set but no key is: set the `key` property");
                return Err(G2gError::NotConfigured);
            }
        };
        self.packetizer = Some(VideoPacketizer::new(
            self.packet_size,
            self.fec_percentage,
            self.minimum_fec_packets,
            key,
        ));
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.send_frame(frame, out).await?,
                PipelinePacket::CapsChanged(_) => {
                    out.push(PipelinePacket::CapsChanged(OUTPUT_CAPS)).await?;
                }
                PipelinePacket::Eos => {}
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }

    fn take_reconfigure(&mut self) -> Option<Reconfigure> {
        let invalidate_from = self
            .recovery
            .invalidate_from_frame
            .swap(NO_INVALIDATION, Ordering::Relaxed);
        if self.recovery.keyframe.swap(false, Ordering::Relaxed) {
            return Some(Reconfigure::ForceKeyframe);
        }
        if invalidate_from == NO_INVALIDATION {
            return None;
        }
        let lost = self
            .sent_frames
            .iter()
            .find(|sent| sent.frame_number == invalidate_from);
        match lost {
            Some(lost) => Some(Reconfigure::InvalidateReferences {
                from_pts_ns: lost.pts_ns,
            }),
            None => {
                g2g_core::g2g_info!(
                    self,
                    "frame {invalidate_from} is older than the frames kept, sending a key frame"
                );
                Some(Reconfigure::ForceKeyframe)
            }
        }
    }
}

impl PadTemplates for MoonlightVideoPay {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([
            PadTemplate::sink(CapsSet::from_alternatives(input_caps())),
            PadTemplate::source(CapsSet::one(OUTPUT_CAPS)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textencoding::hex_encode;

    const PACKET_SIZE: usize = 80;
    const PAYLOAD_BYTES: usize = PACKET_SIZE - VIDEO_PACKET_HEADER_BYTES;
    const KEY: [u8; GCM_KEY_BYTES] = [0x42; GCM_KEY_BYTES];

    fn access_unit_filling(data_shards: usize) -> Vec<u8> {
        (0..data_shards * PAYLOAD_BYTES - FRAME_HEADER_BYTES)
            .map(|index| (index % 251) as u8)
            .collect()
    }

    fn frame(access_unit: &[u8]) -> EncodedFrame<'_> {
        EncodedFrame {
            access_unit,
            keyframe: true,
            rtp_timestamp: 0x1234_5678,
            latency_units: 7,
        }
    }

    fn fec_info(packet: &[u8]) -> (usize, usize, usize) {
        let header = &packet[VIDEO_PACKET_HEADER_OFFSET..PAYLOAD_OFFSET];
        let info = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
        ((info >> 12) & 0x3ff, info >> 22, (info >> 4) & 0xff)
    }

    fn block(packet: &[u8]) -> u8 {
        (packet[VIDEO_PACKET_HEADER_OFFSET + 11] >> 4) & 3
    }

    #[test]
    fn a_frame_reassembles_from_its_data_packets_with_its_header() {
        let unit = access_unit_filling(600);
        let packets = VideoPacketizer::new(PACKET_SIZE, 20, 2, None)
            .packetize(frame(&unit))
            .expect("fits");
        let mut layout = Vec::new();
        let mut stream = Vec::new();
        for packet in &packets {
            let (index, data, percentage) = fec_info(packet);
            let parity = (data * percentage).div_ceil(100);
            if index == 0 {
                layout.push((block(packet), data, parity));
            }
            if index < data {
                stream.extend_from_slice(&packet[PAYLOAD_OFFSET..]);
            }
        }
        assert_eq!(layout, [(0, 200, 40), (1, 200, 40), (2, 200, 40)]);
        assert_eq!(packets.len(), 3 * 240);
        assert_eq!(stream[0], FRAME_HEADER_TYPE);
        assert_eq!(u16::from_le_bytes([stream[1], stream[2]]), 7);
        assert_eq!(stream[3], FRAME_TYPE_KEY);
        assert_eq!(
            u32::from_le_bytes(stream[4..8].try_into().unwrap()) as usize,
            PAYLOAD_BYTES
        );
        assert!(stream[FRAME_HEADER_BYTES..] == unit[..]);
    }

    #[test]
    fn parity_packets_carry_the_parity_of_their_data_packets() {
        let unit = access_unit_filling(10);
        let packets = VideoPacketizer::new(PACKET_SIZE, 20, 2, None)
            .packetize(frame(&unit))
            .expect("fits");
        let (_, data, percentage) = fec_info(&packets[0]);
        let parity = (data * percentage).div_ceil(100);
        assert_eq!((data, parity), (10, 2));
        let (data_packets, parity_packets) = packets.split_at(data);
        let encoder = ParityMatrix::cauchy(data, parity).unwrap();
        let mut expected = alloc::vec![alloc::vec![0u8; data_packets[0].len()]; parity];
        encoder.encode(data_packets, &mut expected);
        for (sent, computed) in parity_packets.iter().zip(&expected) {
            assert_eq!(sent[PAYLOAD_OFFSET..], computed[PAYLOAD_OFFSET..]);
        }
    }

    #[test]
    fn a_frame_past_four_full_blocks_is_refused_without_using_sequence_numbers() {
        let mut packetizer = VideoPacketizer::new(PACKET_SIZE, 20, 2, None);
        let unit = access_unit_filling(MAX_FEC_BLOCKS * MAX_DATA_SHARDS_PER_BLOCK + 1);
        assert!(packetizer.packetize(frame(&unit)).is_err());
        let small = access_unit_filling(1);
        let packets = packetizer.packetize(frame(&small)).expect("fits");
        assert_eq!(u16::from_be_bytes([packets[0][2], packets[0][3]]), 0);
    }

    #[test]
    fn a_sealed_packet_opens_to_the_plain_packet() {
        let unit = access_unit_filling(3);
        let plain = VideoPacketizer::new(PACKET_SIZE, 20, 2, None)
            .packetize(frame(&unit))
            .unwrap();
        let sealed = VideoPacketizer::new(PACKET_SIZE, 20, 2, Some(KEY))
            .packetize(frame(&unit))
            .unwrap();
        assert_eq!(plain.len(), sealed.len());
        let cipher = Aes128Gcm::new(&Array(KEY));
        for (index, (plain, sealed)) in plain.iter().zip(&sealed).enumerate() {
            let (prefix, body) = sealed.split_at(ENCRYPTION_PREFIX_BYTES);
            let nonce: [u8; GCM_NONCE_BYTES] = prefix[..GCM_NONCE_BYTES].try_into().unwrap();
            assert_eq!(
                u64::from_le_bytes(nonce[..8].try_into().unwrap()),
                index as u64
            );
            assert_eq!(nonce[GCM_NONCE_BYTES - 1], VIDEO_NONCE_MARKER);
            assert_eq!(
                prefix[GCM_NONCE_BYTES..GCM_NONCE_BYTES + 4],
                1u32.to_le_bytes()
            );
            let tag: [u8; GCM_TAG_BYTES] = prefix[GCM_NONCE_BYTES + 4..].try_into().unwrap();
            let mut opened = body.to_vec();
            cipher
                .decrypt_inout_detached(
                    &Array(nonce),
                    &[],
                    opened.as_mut_slice().into(),
                    &Array(tag),
                )
                .expect("the tag authenticates the packet");
            assert_eq!(&opened, plain);
        }
    }

    #[test]
    fn a_keyframe_request_reaches_the_encoder_once() {
        let mut pay = MoonlightVideoPay::default();
        pay.recovery_requests().request_keyframe();
        assert_eq!(pay.take_reconfigure(), Some(Reconfigure::ForceKeyframe));
        assert_eq!(pay.take_reconfigure(), None);
    }

    struct DiscardSink;

    impl OutputSink for DiscardSink {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<g2g_core::PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(g2g_core::PushOutcome::Accepted))
        }
    }

    async fn send_frames(pay: &mut MoonlightVideoPay, count: u64) {
        pay.set_property("encrypt", PropValue::Bool(false)).unwrap();
        pay.configure_pipeline(&input_caps()[0]).unwrap();
        for pts_ns in 0..count {
            let frame = Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(
                    access_unit_filling(1).into_boxed_slice(),
                )),
                g2g_core::FrameTiming {
                    pts_ns: pts_ns * 1000,
                    ..Default::default()
                },
                pts_ns,
            );
            pay.process(PipelinePacket::DataFrame(frame), &mut DiscardSink)
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn an_invalidation_reaches_the_encoder_as_the_presentation_time_of_the_first_lost_frame()
    {
        let mut pay = MoonlightVideoPay::default();
        send_frames(&mut pay, 10).await;
        // moonlight numbers frames from 1, so frame 7 is the one with pts 6000
        pay.recovery_requests().invalidate_from(8);
        pay.recovery_requests().invalidate_from(7);
        assert_eq!(
            pay.take_reconfigure(),
            Some(Reconfigure::InvalidateReferences { from_pts_ns: 6000 })
        );
        assert_eq!(pay.take_reconfigure(), None);
    }

    #[tokio::test]
    async fn a_loss_older_than_the_kept_frames_asks_for_a_key_frame() {
        let mut pay = MoonlightVideoPay::default();
        send_frames(&mut pay, SENT_FRAMES_KEPT as u64 + 2).await;
        pay.recovery_requests().invalidate_from(1);
        assert_eq!(pay.take_reconfigure(), Some(Reconfigure::ForceKeyframe));
    }

    #[tokio::test]
    async fn a_mid_stream_caps_change_keeps_the_frame_numbers_going() {
        let mut pay = MoonlightVideoPay::default();
        send_frames(&mut pay, 3).await;
        // send_frames configures again first, as the runner does on a caps change
        send_frames(&mut pay, 1).await;
        let packetizer = pay.packetizer.as_ref().expect("configured");
        assert_eq!(packetizer.frame_number, 4);
    }

    #[test]
    fn encryption_without_a_key_refuses_to_configure() {
        let mut pay = MoonlightVideoPay::default();
        assert!(pay.configure_pipeline(&input_caps()[0]).is_err());
        pay.set_property("key", PropValue::Str(hex_encode(&KEY)))
            .unwrap();
        assert!(pay.configure_pipeline(&input_caps()[0]).is_ok());
    }
}
