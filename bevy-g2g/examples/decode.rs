//! Video playback with zero pipeline code: a stock windowed Bevy app plus
//! `VideoPlayerPlugin`; the cube is tagged `VideoScreen` so it plays the clip
//! (decoded by g2g onto Bevy's own wgpu device, zero-copy).
//!
//! ```sh
//! cargo run --release --features decode --example decode            # bundled clip
//! cargo run --release --features decode --example decode -- my.h264
//! ```
//!
//! `G2G_EXIT_AFTER_SECS=8` exits by itself (a smoke run): success once frames
//! were decoded and bound.

use std::time::Instant;

use bevy::prelude::*;
use bevy_g2g::{VideoPlayback, VideoPlayerPlugin, VideoScreen};

/// The 640x480 baseline clip the g2g tests use (two GOPs of IDR + P frames).
const BUNDLED_CLIP: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../g2g-plugins/tests/fixtures/h264_640x480.h264"
);

fn main() {
    let clip = std::env::args()
        .nth(1)
        .unwrap_or_else(|| BUNDLED_CLIP.to_string());
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "g2g decode -> Bevy (shared wgpu device)".into(),
                resolution: (960, 720).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins(VideoPlayerPlugin::new(clip))
        .insert_resource(SmokeExit {
            started: Instant::now(),
            exit_after_secs: std::env::var("G2G_EXIT_AFTER_SECS")
                .ok()
                .and_then(|s| s.parse::<f64>().ok()),
        })
        .add_systems(Startup, setup)
        .add_systems(Update, (spin, smoke_exit))
        .run();
}

#[derive(Component)]
struct Spin;

/// Optional self-exit for unattended smoke runs (`G2G_EXIT_AFTER_SECS`).
#[derive(Resource)]
struct SmokeExit {
    started: Instant,
    exit_after_secs: Option<f64>,
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // The cube the video plays on. Starts white until the first frame arrives.
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(2.0, 1.5, 2.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            ..default()
        })),
        Transform::from_xyz(0.0, 1.0, 0.0),
        Spin,
        VideoScreen,
    ));
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.3, 0.35))),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    commands.spawn((
        PointLight {
            intensity: 4_000_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    commands.spawn((
        Camera3d::default(),
        // Per-view ambient so the video texture stays readable on the faces
        // away from the light.
        AmbientLight {
            brightness: 400.0,
            ..default()
        },
        Transform::from_xyz(-2.0, 2.5, 5.0).looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
    ));
}

fn spin(time: Res<Time>, mut q: Query<&mut Transform, With<Spin>>) {
    for mut t in &mut q {
        t.rotate_y(time.delta_secs() * 0.8);
    }
}

fn smoke_exit(
    smoke: Res<SmokeExit>,
    playback: Res<VideoPlayback>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(secs) = smoke.exit_after_secs else {
        return;
    };
    if smoke.started.elapsed().as_secs_f64() < secs {
        return;
    }
    if playback.frames_bound() > 0 {
        info!(
            "smoke exit: {} decoded frames, {} bound to the material",
            playback.frames_decoded(),
            playback.frames_bound()
        );
        exit.write(AppExit::Success);
    } else {
        error!("smoke exit: no decoded frame was ever bound");
        exit.write(AppExit::error());
    }
}
