// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(feature = "std")]

use std::time::Duration;

use g2g_core::runtime::{parse_launch, RunBuilder};
use g2g_core::{Bus, BusMessage};
use g2g_plugins::clock::WallClock;
use g2g_plugins::registry::default_registry;

const BUS_CAPACITY: usize = 256;
const LINK_CAPACITY: usize = 4;
const RUN_TIMEOUT: Duration = Duration::from_secs(10);
const DB_TOLERANCE: f64 = 0.01;

// audiotestsrc's square wave swings between plus and minus half of i16 full scale
fn half_scale_square_wave_db() -> f64 {
    let half_scale = f64::from(i16::MAX / 2) / f64::from(i16::MAX as i32 + 1);
    20.0 * half_scale.log10()
}

async fn meter_readings(line: &str) -> Vec<(String, Vec<f64>, Vec<f64>)> {
    let graph = parse_launch(&default_registry(), line).expect("the line builds");
    let (bus, handle) = Bus::new(BUS_CAPACITY);
    tokio::time::timeout(
        RUN_TIMEOUT,
        RunBuilder::new(graph, &WallClock::new(), LINK_CAPACITY)
            .with_bus(&handle)
            .run(),
    )
    .await
    .expect("the run ends")
    .expect("the pipeline runs");
    std::iter::from_fn(|| bus.try_recv())
        .filter_map(|message| match message {
            BusMessage::Level {
                element,
                peak_db,
                rms_db,
                ..
            } => Some((element, peak_db, rms_db)),
            _ => None,
        })
        .collect()
}

fn assert_every_reading_is(readings: Vec<(String, Vec<f64>, Vec<f64>)>, expected_db: f64) {
    assert!(!readings.is_empty(), "level posted nothing");
    for (element, peak_db, rms_db) in readings {
        assert_eq!(element, "meter");
        for level_db in peak_db.iter().chain(&rms_db) {
            assert!(
                (level_db - expected_db).abs() < DB_TOLERANCE,
                "{level_db} dB, a full-swing square wave at half scale reads {expected_db} dB"
            );
        }
    }
}

#[tokio::test]
async fn level_posts_the_peak_and_rms_of_each_interval_in_db() {
    let readings = meter_readings(
        "audiotestsrc wave=square num-buffers=10 ! level name=meter interval=50000000 ! fakesink",
    )
    .await;
    assert_every_reading_is(readings, half_scale_square_wave_db());
}

#[tokio::test]
async fn level_reads_24_bit_audio() {
    let readings = meter_readings(
        "audiotestsrc wave=square num-buffers=10 ! audioconvert ! audio/x-raw,format=S24LE \
         ! level name=meter interval=50000000 ! fakesink",
    )
    .await;
    assert_every_reading_is(readings, half_scale_square_wave_db());
}
