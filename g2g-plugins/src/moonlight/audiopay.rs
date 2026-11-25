// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use aes_gcm::aead::array::Array;
use aes_gcm::aes::cipher::{BlockCipherEncrypt, KeyInit};
use aes_gcm::aes::Aes128;
use g2g_core::log::LogSource;
use g2g_core::memory::{DomainSet, MemoryDomain, MemoryDomainKind, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, CapsConstraint, CapsSet, ConfigureOutcome,
    ElementMetadata, Frame, G2gError, OutputSink, PadTemplate, PadTemplates, PipelinePacket,
    PropError, PropKind, PropValue, PropertySpec,
};

use super::fec::{ParityMatrix, MOONLIGHT_AUDIO_DATA_SHARDS};
use crate::opusparse::OPUS_RATE_HZ;
use crate::textencoding::hex_decode;

const RTP_HEADER_BYTES: usize = 12;
const RTP_VERSION_BYTE: u8 = 0x80;
const AUDIO_PAYLOAD_TYPE: u8 = 97;
const FEC_PAYLOAD_TYPE: u8 = 127;
const FEC_HEADER_BYTES: usize = 12;
const PARITY_SHARDS: usize = 2;
const AES_BLOCK_BYTES: usize = 16;
const AES_KEY_BYTES: usize = 16;
const DEFAULT_PACKET_DURATION_MS: u32 = 5;

struct CbcSealer {
    cipher: Aes128,
    key_id: u32,
}

impl core::fmt::Debug for CbcSealer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CbcSealer").finish_non_exhaustive()
    }
}

// AES-128-CBC with PKCS7 padding
fn cbc_encrypt(cipher: &Aes128, iv: [u8; AES_BLOCK_BYTES], plaintext: &[u8]) -> Vec<u8> {
    let padding = AES_BLOCK_BYTES - plaintext.len() % AES_BLOCK_BYTES;
    let mut buffer = Vec::with_capacity(plaintext.len() + padding);
    buffer.extend_from_slice(plaintext);
    buffer.resize(plaintext.len() + padding, padding as u8);
    let mut previous = iv;
    for block in buffer.as_chunks_mut::<AES_BLOCK_BYTES>().0 {
        block
            .iter_mut()
            .zip(previous)
            .for_each(|(byte, prior)| *byte ^= prior);
        cipher.encrypt_block(block.into());
        previous = *block;
    }
    buffer
}

impl CbcSealer {
    fn seal(&self, sequence_number: u16, opus: &[u8]) -> Vec<u8> {
        // moonlight's IV is the key id plus the RTP sequence number, big-endian, then zeros
        let mut iv = [0u8; AES_BLOCK_BYTES];
        let counter = self.key_id.wrapping_add(u32::from(sequence_number));
        iv[..4].copy_from_slice(&counter.to_be_bytes());
        cbc_encrypt(&self.cipher, iv, opus)
    }
}

fn rtp_header(payload_type: u8, sequence_number: u16, timestamp: u32) -> [u8; RTP_HEADER_BYTES] {
    let mut header = [0u8; RTP_HEADER_BYTES];
    header[0] = RTP_VERSION_BYTE;
    header[1] = payload_type;
    header[2..4].copy_from_slice(&sequence_number.to_be_bytes());
    header[4..8].copy_from_slice(&timestamp.to_be_bytes());
    header
}

#[derive(Debug)]
pub(crate) struct AudioPacketizer {
    packet_duration_ms: u32,
    sealer: Option<CbcSealer>,
    parity: ParityMatrix,
    sequence_number: u16,
    timestamp_ms: u32,
    block_payloads: Vec<Vec<u8>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UnequalBlock;

impl AudioPacketizer {
    pub(crate) fn new(packet_duration_ms: u32, key: Option<([u8; AES_KEY_BYTES], u32)>) -> Self {
        Self {
            packet_duration_ms,
            sealer: key.map(|(key, key_id)| CbcSealer {
                cipher: Aes128::new(&Array(key)),
                key_id,
            }),
            parity: ParityMatrix::moonlight_audio(),
            sequence_number: 0,
            timestamp_ms: 0,
            block_payloads: Vec::with_capacity(MOONLIGHT_AUDIO_DATA_SHARDS),
        }
    }

    // one data datagram, then the two parity datagrams once a block of four is complete
    pub(crate) fn packetize(&mut self, opus: &[u8]) -> Result<Vec<Vec<u8>>, UnequalBlock> {
        let payload = match &self.sealer {
            Some(sealer) => sealer.seal(self.sequence_number, opus),
            None => opus.to_vec(),
        };
        let header = rtp_header(AUDIO_PAYLOAD_TYPE, self.sequence_number, self.timestamp_ms);
        let datagram = [header.as_slice(), &payload].concat();
        let base_sequence_number =
            self.sequence_number - self.sequence_number % MOONLIGHT_AUDIO_DATA_SHARDS as u16;
        let base_timestamp = self.timestamp_ms.wrapping_sub(
            u32::from(self.sequence_number - base_sequence_number) * self.packet_duration_ms,
        );
        self.sequence_number = self.sequence_number.wrapping_add(1);
        self.timestamp_ms = self.timestamp_ms.wrapping_add(self.packet_duration_ms);
        self.block_payloads.push(payload);
        let mut datagrams = Vec::from([datagram]);
        if self.block_payloads.len() < MOONLIGHT_AUDIO_DATA_SHARDS {
            return Ok(datagrams);
        }
        let payloads = core::mem::take(&mut self.block_payloads);
        let size = payloads[0].len();
        // moonlight drops a block whose packets differ in size
        if payloads.iter().any(|payload| payload.len() != size) {
            return Err(UnequalBlock);
        }
        let mut parity: [Vec<u8>; PARITY_SHARDS] = core::array::from_fn(|_| alloc::vec![0u8; size]);
        self.parity.encode(&payloads, &mut parity);
        for (index, shard) in parity.iter().enumerate() {
            let header = rtp_header(
                FEC_PAYLOAD_TYPE,
                self.sequence_number.wrapping_add(index as u16),
                0,
            );
            let mut fec_header = [0u8; FEC_HEADER_BYTES];
            fec_header[0] = index as u8;
            fec_header[1] = AUDIO_PAYLOAD_TYPE;
            fec_header[2..4].copy_from_slice(&base_sequence_number.to_be_bytes());
            fec_header[4..8].copy_from_slice(&base_timestamp.to_be_bytes());
            datagrams.push([header.as_slice(), &fec_header, shard].concat());
        }
        Ok(datagrams)
    }
}

static PROPERTIES: &[PropertySpec] = &[
    PropertySpec::new(
        "packet-duration",
        PropKind::Uint,
        "milliseconds of audio in each Opus packet, the packetDuration Moonlight announces",
    )
    .with_enum_values("5 | 10")
    .with_default("5"),
    PropertySpec::new(
        "key",
        PropKind::Str,
        "AES-128 key as 32 hexadecimal digits, the rikey of the Moonlight launch",
    ),
    PropertySpec::new(
        "key-id",
        PropKind::Uint,
        "rikeyid of the Moonlight launch, the start of every packet's IV",
    )
    .with_range("0", "4294967295")
    .with_default("0"),
    PropertySpec::new(
        "encrypt",
        PropKind::Bool,
        "encrypt every Opus packet with AES-CBC under key",
    )
    .with_default("true"),
];

const OUTPUT_CAPS: Caps = Caps::ByteStream {
    encoding: ByteStreamEncoding::Datagram,
};

// moonlight plays stereo, 5.1 and 7.1
const MOONLIGHT_CHANNEL_COUNTS: [u8; 3] = [2, 6, 8];

fn input_caps() -> Vec<Caps> {
    MOONLIGHT_CHANNEL_COUNTS
        .into_iter()
        .map(|channels| Caps::Audio {
            format: AudioFormat::Opus,
            channels,
            sample_rate: OPUS_RATE_HZ,
            channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
        })
        .collect()
}

#[derive(Debug)]
pub struct MoonlightAudioPay {
    packet_duration_ms: u32,
    key: Option<[u8; AES_KEY_BYTES]>,
    key_id: u32,
    encrypt: bool,
    packetizer: Option<AudioPacketizer>,
}

impl Default for MoonlightAudioPay {
    fn default() -> Self {
        Self {
            packet_duration_ms: DEFAULT_PACKET_DURATION_MS,
            key: None,
            key_id: 0,
            encrypt: true,
            packetizer: None,
        }
    }
}

impl LogSource for MoonlightAudioPay {
    fn log_category(&self) -> &'static str {
        "moonlightaudiopay"
    }
}

impl MoonlightAudioPay {
    async fn send_packet(
        &mut self,
        frame: Frame,
        out: &mut dyn OutputSink,
    ) -> Result<(), G2gError> {
        let opus = frame
            .domain
            .require_system_slice(g2g_core::log::short_type_name::<Self>())?;
        let packetizer = self.packetizer.as_mut().ok_or(G2gError::NotConfigured)?;
        let datagrams = match packetizer.packetize(opus) {
            Ok(datagrams) => datagrams,
            Err(UnequalBlock) => {
                g2g_core::g2g_error!(
                    self,
                    "Opus packets in one FEC block differ in size: encode with bitrate-type=cbr"
                );
                return Err(G2gError::InputRefused);
            }
        };
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

impl AsyncElement for MoonlightAudioPay {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::System)
    }

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "Moonlight audio payloader",
            "Codec/Payloader/Network",
            "Puts Opus packets into the FEC-protected, AES-CBC encrypted RTP packets Moonlight \
             clients read",
            crate::AARON_BOXER,
        )
    }

    fn properties(&self) -> &'static [PropertySpec] {
        PROPERTIES
    }

    fn set_property(&mut self, name: &str, value: PropValue) -> Result<(), PropError> {
        match name {
            "packet-duration" => {
                let duration = value.as_uint().ok_or(PropError::Type)?;
                self.packet_duration_ms = match duration {
                    5 | 10 => duration as u32,
                    _ => return Err(PropError::Value),
                };
            }
            "key" => {
                let text = value.as_str().ok_or(PropError::Type)?;
                let key = hex_decode(text).ok_or(PropError::Value)?;
                self.key = Some(key.try_into().map_err(|_| PropError::Value)?);
            }
            "key-id" => {
                let key_id = value.as_uint().ok_or(PropError::Type)?;
                self.key_id = u32::try_from(key_id).map_err(|_| PropError::Value)?;
            }
            "encrypt" => self.encrypt = value.as_bool().ok_or(PropError::Type)?,
            _ => return Err(PropError::Unknown),
        }
        Ok(())
    }

    fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "packet-duration" => Some(PropValue::Uint(u64::from(self.packet_duration_ms))),
            // the key is a session secret
            "key" => Some(PropValue::Str(String::new())),
            "key-id" => Some(PropValue::Uint(u64::from(self.key_id))),
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
        let key = match (self.encrypt, self.key) {
            (false, _) => None,
            (true, Some(key)) => Some((key, self.key_id)),
            (true, None) => {
                g2g_core::g2g_error!(self, "encrypt is set but no key is: set the `key` property");
                return Err(G2gError::NotConfigured);
            }
        };
        self.packetizer = Some(AudioPacketizer::new(self.packet_duration_ms, key));
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            match packet {
                PipelinePacket::DataFrame(frame) => self.send_packet(frame, out).await?,
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
}

impl PadTemplates for MoonlightAudioPay {
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

    // FIPS-197's example key
    const KEY: [u8; AES_KEY_BYTES] = [
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f,
        0x3c,
    ];
    const PACKET_DURATION_MS: u32 = 5;

    #[test]
    fn cbc_matches_openssl() {
        let iv: [u8; AES_BLOCK_BYTES] = core::array::from_fn(|index| index as u8);
        let plaintext: Vec<u8> = (0..36).collect();
        // openssl enc -aes-128-cbc -K <KEY> -iv 000102...0f over the bytes 0 to 35
        let expected = "7df76b0c1ab899b33e42f047b91b546f1caa8018c80b15b8e7aea82794adcb00\
                        7954069e7336ba56181adc43ab278b26";
        let ciphertext = cbc_encrypt(&Aes128::new(&Array(KEY)), iv, &plaintext);
        assert_eq!(ciphertext, hex_decode(expected).unwrap());
    }

    #[test]
    fn four_packets_make_a_block_and_two_parity_packets() {
        let mut packetizer = AudioPacketizer::new(PACKET_DURATION_MS, None);
        let packets: Vec<Vec<u8>> = (0..4u8).map(|index| alloc::vec![index; 20]).collect();
        let mut datagrams = Vec::new();
        for (index, packet) in packets.iter().enumerate() {
            let sent = packetizer.packetize(packet).unwrap();
            assert_eq!(sent.len(), if index == 3 { 3 } else { 1 });
            datagrams.extend(sent);
        }
        for (index, datagram) in datagrams[..4].iter().enumerate() {
            assert_eq!(datagram[1], AUDIO_PAYLOAD_TYPE);
            assert_eq!(datagram[2..4], (index as u16).to_be_bytes());
            assert_eq!(
                datagram[4..8],
                (index as u32 * PACKET_DURATION_MS).to_be_bytes()
            );
            assert_eq!(datagram[RTP_HEADER_BYTES..], packets[index]);
        }
        let mut parity = [alloc::vec![0u8; 20], alloc::vec![0u8; 20]];
        ParityMatrix::moonlight_audio().encode(&packets, &mut parity);
        for (index, datagram) in datagrams[4..].iter().enumerate() {
            let fec_header = &datagram[RTP_HEADER_BYTES..RTP_HEADER_BYTES + FEC_HEADER_BYTES];
            assert_eq!(datagram[1], FEC_PAYLOAD_TYPE);
            assert_eq!(
                fec_header[..8],
                [index as u8, AUDIO_PAYLOAD_TYPE, 0, 0, 0, 0, 0, 0]
            );
            assert_eq!(
                datagram[RTP_HEADER_BYTES + FEC_HEADER_BYTES..],
                parity[index]
            );
        }
    }

    #[test]
    fn the_second_block_names_its_own_base() {
        let mut packetizer = AudioPacketizer::new(PACKET_DURATION_MS, None);
        let packet = [0u8; 20];
        let datagrams: Vec<Vec<u8>> = (0..8)
            .flat_map(|_| packetizer.packetize(&packet).unwrap())
            .collect();
        let fec_header = &datagrams.last().unwrap()[RTP_HEADER_BYTES..];
        assert_eq!(fec_header[2..4], 4u16.to_be_bytes());
        assert_eq!(fec_header[4..8], (4 * PACKET_DURATION_MS).to_be_bytes());
    }

    #[test]
    fn an_encrypted_payload_uses_the_key_id_plus_sequence_as_iv() {
        const KEY_ID: u32 = 0xffff_fffe;
        let mut packetizer = AudioPacketizer::new(PACKET_DURATION_MS, Some((KEY, KEY_ID)));
        let opus = [7u8; 20];
        packetizer.packetize(&opus).unwrap();
        let second = packetizer.packetize(&opus).unwrap().remove(0);
        // the counter wraps: key id 0xfffffffe plus sequence 1
        let mut iv = [0u8; AES_BLOCK_BYTES];
        iv[..4].copy_from_slice(&u32::MAX.to_be_bytes());
        let expected = cbc_encrypt(&Aes128::new(&Array(KEY)), iv, &opus);
        assert_eq!(second[RTP_HEADER_BYTES..], expected);
    }

    #[test]
    fn a_block_of_unequal_packets_is_refused() {
        let mut packetizer = AudioPacketizer::new(PACKET_DURATION_MS, None);
        for size in [20, 20, 20] {
            packetizer.packetize(&alloc::vec![0u8; size]).unwrap();
        }
        assert_eq!(packetizer.packetize(&[0u8; 21]), Err(UnequalBlock));
    }
}
