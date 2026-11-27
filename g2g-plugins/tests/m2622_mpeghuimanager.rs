// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

//! M2622: `mpeghuimanager` reports the audio scene, applies ActionEvents to the MHAS
//! stream and keeps the user's choices across runs. Reads the interactive stream
//! tools/mpegh-fixtures.sh fetches and skips without it.
#![cfg(feature = "mpegh")]

#[path = "common/collect_sink.rs"]
mod collect_sink;
#[path = "common/mpegh_fixtures.rs"]
mod mpegh_fixtures;

use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};
use g2g_core::{AsyncElement, Bus, BusMessage, Caps, PropError, PropValue};
use g2g_plugins::mpeghdec::MpeghDec;
use g2g_plugins::mpeghuimanager::MpeghUiManager;

use collect_sink::CollectSink;
use mpegh_fixtures::{
    data_frames, demux_mp4_audio, demux_ts_mpegh, first_caps, mpegh_fixture, MHA1_MP4, STEREO_MP4,
};

// three presets: 0 "Default", 1 "Dialog+" and a third, over a 5.1+4H bed with dialogue and an object
const INTERACTIVE_MP4: &str = "TRI_Fileset_17_514H_D1_D2_D3_O1_24bit2160p50.audio.mp4";
// two seconds of 50 Hz access units, enough to hear a preset
const ACCESS_UNITS: usize = 100;
// the access units the scene takes to arrive before the application answers it
const ACCESS_UNITS_BEFORE_ACTION: usize = 10;
const DIALOG_PRESET: u32 = 1;
const BUS_CAPACITY: usize = 4096;
const CICP_STEREO: i64 = 2;

fn select_preset(scene_uuid: &str, preset: u32) -> String {
    format!(
        "<ActionEvent uuid=\"{scene_uuid}\" version=\"11.1\" actionType=\"30\" paramInt=\"{preset}\" />"
    )
}

fn scene_uuid(config: &str) -> &str {
    let start = config.find("uuid=\"").expect("a scene uuid") + "uuid=\"".len();
    let length = config[start..].find('"').expect("a closing quote");
    &config[start..start + length]
}

fn preset_is_active(config: &str, preset: u32) -> bool {
    config.contains(&format!("<preset id=\"{preset}\" isActive=\"true\""))
}

fn string_property(element: &MpeghUiManager, name: &str) -> String {
    match element.get_property(name) {
        Some(PropValue::Str(value)) => value,
        other => panic!("{name} is a string property, read {other:?}"),
    }
}

struct Stream {
    caps: Caps,
    access_units: Vec<(Vec<u8>, FrameTiming)>,
}

// the stream's caps and its first frames, as the demuxer sends them
fn first_frames(packets: &[PipelinePacket]) -> Stream {
    let mut access_units = data_frames(packets);
    access_units.truncate(ACCESS_UNITS);
    Stream {
        caps: first_caps(packets).expect("MPEG-H caps"),
        access_units,
    }
}

async fn interactive_stream() -> Option<Stream> {
    let packets = demux_mp4_audio(mpegh_fixture(INTERACTIVE_MP4)?, "mpegh").await;
    Some(first_frames(&packets))
}

fn packets(stream: &Stream) -> Vec<PipelinePacket> {
    std::iter::once(PipelinePacket::CapsChanged(stream.caps.clone()))
        .chain(
            stream
                .access_units
                .iter()
                .enumerate()
                .map(|(index, (bytes, timing))| {
                    PipelinePacket::DataFrame(Frame::from_system_bytes(
                        bytes.clone(),
                        *timing,
                        index as u64,
                    ))
                }),
        )
        .collect()
}

struct Run {
    element: MpeghUiManager,
    output: Vec<PipelinePacket>,
    messages: Vec<BusMessage>,
}

// runs the stream through a UI manager, which answers the first scene with `answer`
async fn manage(
    stream: &Stream,
    input_persistence: Option<String>,
    answer: impl Fn(&str) -> Option<String>,
) -> Run {
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    let mut element = MpeghUiManager::new();
    element.set_bus(handle);
    if let Some(stored) = input_persistence {
        element
            .set_property("input-persistence", PropValue::Str(stored))
            .expect("persistence data");
    }
    element.configure_pipeline(&stream.caps).expect("MHAS caps");
    let mut sink = CollectSink::default();
    for (index, packet) in packets(stream).into_iter().enumerate() {
        // the first packet is the caps, so the access unit index is one less
        if index == ACCESS_UNITS_BEFORE_ACTION + 1 {
            let config = string_property(&element, "config");
            if let Some(action_event) = answer(&config) {
                element
                    .set_property("action-event", PropValue::Str(action_event))
                    .expect("an ActionEvent");
            }
        }
        element.process(packet, &mut sink).await.expect("manage");
    }
    element
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("eos");
    Run {
        element,
        output: sink.packets,
        messages: std::iter::from_fn(|| bus.try_recv()).collect(),
    }
}

fn configs(messages: &[BusMessage]) -> Vec<&str> {
    messages
        .iter()
        .filter_map(|message| match message {
            BusMessage::MpeghUiConfig { config, .. } => Some(config.as_str()),
            _ => None,
        })
        .collect()
}

async fn decoded_samples(managed: Vec<PipelinePacket>) -> Vec<u8> {
    let mut decoder = MpeghDec::new();
    decoder
        .set_property("target-layout", PropValue::Int(CICP_STEREO))
        .expect("stereo");
    decoder
        .configure_pipeline(&first_caps(&managed).expect("MPEG-H caps"))
        .expect("MHAS caps");
    let mut sink = CollectSink::default();
    for packet in managed {
        decoder.process(packet, &mut sink).await.expect("decode");
    }
    decoder
        .process(PipelinePacket::Eos, &mut sink)
        .await
        .expect("drain");
    data_frames(&sink.packets)
        .into_iter()
        .flat_map(|(bytes, _)| bytes)
        .collect()
}

#[tokio::test]
async fn the_scene_is_posted_and_kept_in_the_config_property() {
    let Some(stream) = interactive_stream().await else {
        return;
    };
    let run = manage(&stream, None, |_| None).await;
    let configs = configs(&run.messages);
    let latest = configs.last().expect("a posted AudioSceneConfig");
    assert!(latest.contains("<presets>"), "a full scene");
    assert!(preset_is_active(latest, 0), "the default preset plays");
    assert_eq!(string_property(&run.element, "config"), *latest);
}

#[tokio::test]
async fn a_selected_preset_changes_the_scene_and_the_decoded_audio() {
    let Some(stream) = interactive_stream().await else {
        return;
    };
    let untouched = manage(&stream, None, |_| None).await;
    let selected = manage(&stream, None, |config| {
        Some(select_preset(scene_uuid(config), DIALOG_PRESET))
    })
    .await;

    let results: Vec<&Result<(), i32>> = selected
        .messages
        .iter()
        .filter_map(|message| match message {
            BusMessage::MpeghUiEventResult { outcome, .. } => Some(outcome),
            _ => None,
        })
        .collect();
    assert_eq!(results, [&Ok(())], "the ActionEvent applied");
    let latest = *configs(&selected.messages).last().expect("a scene");
    assert!(preset_is_active(latest, DIALOG_PRESET), "{latest}");
    assert!(
        !string_property(&selected.element, "output-persistence").is_empty(),
        "the choice is stored"
    );
    assert_ne!(
        decoded_samples(selected.output).await,
        decoded_samples(untouched.output).await,
        "the dialogue preset sounds different"
    );
}

#[tokio::test]
async fn stored_settings_bring_the_preset_back_in_a_later_run() {
    let Some(stream) = interactive_stream().await else {
        return;
    };
    let first = manage(&stream, None, |config| {
        Some(select_preset(scene_uuid(config), DIALOG_PRESET))
    })
    .await;
    let stored = string_property(&first.element, "output-persistence");
    let later = manage(&stream, Some(stored), |_| None).await;
    let first_scene = *configs(&later.messages).first().expect("a scene");
    assert!(
        preset_is_active(first_scene, DIALOG_PRESET),
        "{first_scene}"
    );
}

#[tokio::test]
async fn stored_settings_are_refused_once_the_stream_runs() {
    let Some(stream) = interactive_stream().await else {
        return;
    };
    let mut run = manage(&stream, None, |_| None).await;
    assert_eq!(
        run.element
            .set_property("input-persistence", PropValue::Str("AAAAAA==".into())),
        Err(PropError::Value)
    );
}

#[tokio::test]
async fn without_action_events_the_stream_decodes_as_without_the_ui_manager() {
    let (Some(mp4), Some(ts)) = (
        mpegh_fixture(STEREO_MP4),
        mpegh_fixture("mpegh_cicp_02_baseline_32kbps_cont_2000.ts"),
    ) else {
        return;
    };
    for demuxed in [
        demux_mp4_audio(mp4, "mpegh").await,
        demux_ts_mpegh(&ts).await,
    ] {
        let stream = first_frames(&demuxed);
        let managed = manage(&stream, None, |_| None).await;
        assert_eq!(
            decoded_samples(managed.output).await,
            decoded_samples(packets(&stream)).await
        );
    }
}

#[tokio::test]
async fn raw_access_units_are_refused() {
    let Some(file) = mpegh_fixture(MHA1_MP4) else {
        return;
    };
    let caps = first_caps(&demux_mp4_audio(file, "mpegh-raw").await).expect("MPEG-H caps");
    assert!(MpeghUiManager::new().configure_pipeline(&caps).is_err());
}
