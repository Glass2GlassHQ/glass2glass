//! M1556: an AAC-ELD track demuxed as raw access units behind its
//! AudioSpecificConfig remuxes into MP4 with that config as the `esds` and the
//! units as the samples, and the MPEG-TS muxer refuses it rather than writing
//! it as ADTS.
#![cfg(feature = "std")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/eld_mp4.rs"]
mod eld_mp4;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::memory::{MemoryDomain, SystemSlice};
use g2g_core::{
    AsyncElement, AudioFormat, ByteStreamEncoding, Caps, G2gError, MultiInputElement, PropValue,
};
use g2g_plugins::mp4demux::Mp4Demux;
use g2g_plugins::mp4muxn::Mp4MuxN;
use g2g_plugins::tsmux::TsMux;

use collect_sink::CollectSink;
use eld_mp4::{access_units, progressive_eld_mp4, ELD_CONFIG};

const BOX_HEADER_LEN: usize = 8;
const FULL_BOX_HEADER_LEN: usize = 4;
// MPEG-4 descriptor tags (ISO/IEC 14496-1 7.2.2.1)
const ES_DESCRIPTOR_TAG: u8 = 0x03;
const DECODER_CONFIG_DESCRIPTOR_TAG: u8 = 0x04;
const DECODER_SPECIFIC_INFO_TAG: u8 = 0x05;
// ES_ID and flags ahead of an ES_Descriptor's children
const ES_DESCRIPTOR_FIXED_LEN: usize = 3;
// objectTypeIndication through avgBitrate ahead of a DecoderConfigDescriptor's children
const DECODER_CONFIG_FIXED_LEN: usize = 13;

fn aac_caps() -> Caps {
    Caps::Audio {
        format: AudioFormat::Aac,
        channels: 0,
        sample_rate: 0,
        channel_layout: g2g_core::ChannelLayout::UNSPECIFIED,
    }
}

async fn demux_eld() -> Vec<PipelinePacket> {
    let mut demux = Mp4Demux::new();
    demux
        .set_property("stream", PropValue::Str("aac".into()))
        .expect("stream=aac");
    demux
        .configure_pipeline(&Caps::ByteStream {
            encoding: ByteStreamEncoding::Mp4,
        })
        .expect("configure");
    let file = progressive_eld_mp4();
    let mut sink = CollectSink::default();
    demux
        .process(
            PipelinePacket::DataFrame(Frame::new(
                MemoryDomain::System(SystemSlice::from_boxed(file.into_boxed_slice())),
                FrameTiming::default(),
                0,
            )),
            &mut sink,
        )
        .await
        .expect("demux");
    demux
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    sink.packets
        .into_iter()
        .filter(|packet| !matches!(packet, PipelinePacket::Eos))
        .collect()
}

async fn remux_into_mp4(packets: Vec<PipelinePacket>) -> Vec<u8> {
    let mut mux = Mp4MuxN::new(1);
    mux.set_property("fragmented", PropValue::Bool(false))
        .expect("fragmented=false");
    mux.configure_pipeline(0, &aac_caps()).expect("configure");
    let mut sink = CollectSink::default();
    for packet in packets {
        mux.process(0, packet, &mut sink).await.expect("mux");
    }
    mux.process(0, PipelinePacket::Eos, &mut sink)
        .await
        .expect("mux eos");
    sink.packets
        .iter()
        .filter_map(|packet| match packet {
            PipelinePacket::DataFrame(frame) => frame.domain.as_system_slice(),
            _ => None,
        })
        .flatten()
        .copied()
        .collect()
}

// the payload of the first box of `kind` among the boxes laid out in `data`
fn box_payload<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    let mut offset = 0;
    while offset + BOX_HEADER_LEN <= data.len() {
        let size = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        if size < BOX_HEADER_LEN {
            return None;
        }
        let body = data.get(offset + BOX_HEADER_LEN..offset + size)?;
        if &data[offset + 4..offset + BOX_HEADER_LEN] == kind {
            return Some(body);
        }
        offset += size;
    }
    None
}

fn nested_box<'a>(data: &'a [u8], path: &[&[u8; 4]]) -> &'a [u8] {
    path.iter().fold(data, |parent, kind| {
        box_payload(parent, kind)
            .unwrap_or_else(|| panic!("no {} box", String::from_utf8_lossy(&kind[..])))
    })
}

// the payload of a single-byte-sized descriptor with `tag` at the start of `data`
fn descriptor(data: &[u8], tag: u8) -> &[u8] {
    assert_eq!(data[0], tag, "descriptor tag");
    let len = data[1] as usize;
    assert!(len < 0x80, "single-byte size");
    &data[2..2 + len]
}

fn esds_audio_specific_config(esds: &[u8]) -> &[u8] {
    let es = descriptor(&esds[FULL_BOX_HEADER_LEN..], ES_DESCRIPTOR_TAG);
    let decoder_config = descriptor(
        &es[ES_DESCRIPTOR_FIXED_LEN..],
        DECODER_CONFIG_DESCRIPTOR_TAG,
    );
    descriptor(
        &decoder_config[DECODER_CONFIG_FIXED_LEN..],
        DECODER_SPECIFIC_INFO_TAG,
    )
}

fn stsz_sample_sizes(stsz: &[u8]) -> Vec<usize> {
    let body = &stsz[FULL_BOX_HEADER_LEN..];
    let count = u32::from_be_bytes(body[4..8].try_into().unwrap()) as usize;
    body[8..8 + 4 * count]
        .chunks(4)
        .map(|size| u32::from_be_bytes(size.try_into().unwrap()) as usize)
        .collect()
}

#[tokio::test]
async fn mp4mux_writes_the_eld_config_as_the_esds_and_the_raw_units_as_samples() {
    let file = remux_into_mp4(demux_eld().await).await;

    let stbl = nested_box(&file, &[b"moov", b"trak", b"mdia", b"minf", b"stbl"]);
    // stsd: full box header and entry count, then the mp4a entry
    let stsd = nested_box(stbl, &[b"stsd"]);
    let mp4a = box_payload(&stsd[FULL_BOX_HEADER_LEN + 4..], b"mp4a").expect("mp4a entry");
    // an audio sample entry's fixed fields come before its child boxes
    const AUDIO_SAMPLE_ENTRY_FIXED_LEN: usize = 28;
    let esds = box_payload(&mp4a[AUDIO_SAMPLE_ENTRY_FIXED_LEN..], b"esds").expect("esds");
    assert_eq!(
        esds_audio_specific_config(esds),
        ELD_CONFIG,
        "the esds carries the demuxed config verbatim"
    );

    let units = access_units();
    assert_eq!(
        stsz_sample_sizes(nested_box(stbl, &[b"stsz"])),
        units.iter().map(Vec::len).collect::<Vec<_>>(),
        "one sample per raw unit, the config not among them"
    );
    assert_eq!(
        nested_box(&file, &[b"mdat"]),
        units.concat(),
        "the samples are the raw units unframed"
    );
}

#[tokio::test]
async fn tsmux_refuses_the_raw_eld_stream() {
    let mut mux = TsMux::new();
    mux.configure_pipeline(&aac_caps()).expect("configure");
    let mut sink = CollectSink::default();
    let mut outcome = Ok(());
    for packet in demux_eld().await {
        outcome = mux.process(packet, &mut sink).await;
        if outcome.is_err() {
            break;
        }
    }
    assert!(
        matches!(outcome, Err(G2gError::CapsMismatch)),
        "ADTS cannot carry ELD and no LOAS writer exists: {outcome:?}"
    );
}
