//! Remote rendering with zero streaming code: a spinning cube on a lit plane,
//! rendered headless and streamed by `RemoteRenderPlugins`. With the `nvenc`
//! feature on an NVIDIA GPU the frames are encoded without ever leaving the
//! GPU; otherwise they are read back and encoded with libx264.
//!
//! `G2G_WHIP_URL` streams to a WHIP endpoint (e.g. MediaMTX) and `G2G_MOQT_URL`
//! to a MoQ Transport relay (`G2G_MOQT_NAMESPACE`, `G2G_MOQT_CERT_HASHES`);
//! with neither, the H.264 goes to `bevy_g2g.h264`. `G2G_FRAMES` caps the run
//! (default 900 (15 s), `0` = forever).

use bevy::prelude::*;

fn main() {
    // G2G_WINDOW=1 streams from a normal windowed run (the window mirrors the
    // stream); default is headless.
    let plugins = if std::env::var("G2G_WINDOW").is_ok() {
        bevy_g2g::RemoteRenderPlugins::windowed(bevy_g2g::StreamSettings::from_env())
    } else {
        bevy_g2g::RemoteRenderPlugins::from_env()
    };
    let mut app = App::new();
    app.add_plugins(plugins)
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(Update, (spin, drive));
    bevy_g2g::run(app);
}

/// Viewer input drives the cube (WASD / arrows via the `G2G_INPUT_PORT`
/// backchannel): ordinary `ButtonInput<KeyCode>` code, nothing
/// backchannel-specific.
fn drive(
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut q: Query<&mut Transform, With<Spin>>,
) {
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        dir.z -= 1.0;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        dir.z += 1.0;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        dir.x -= 1.0;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        dir.x += 1.0;
    }
    if dir == Vec3::ZERO {
        return;
    }
    let announce = keys.get_just_pressed().next().is_some();
    for mut t in &mut q {
        t.translation += dir * time.delta_secs() * 3.0;
        if announce {
            info!("cube moving, now at {:?}", t.translation);
        }
    }
}

#[derive(Component)]
struct Spin;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(1.0, 1.0, 1.0))),
        MeshMaterial3d(materials.add(Color::srgb_u8(124, 144, 255))),
        Transform::from_xyz(0.0, 0.5, 0.0),
        Spin,
    ));
    commands.spawn((
        Mesh3d(meshes.add(Circle::new(4.0))),
        MeshMaterial3d(materials.add(Color::WHITE)),
        Transform::from_rotation(Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)),
    ));
    commands.spawn((
        PointLight {
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0),
    ));
    // A plain camera: the plugin retargets it onto the stream texture.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(-2.5, 4.5, 9.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

fn spin(time: Res<Time>, mut q: Query<&mut Transform, With<Spin>>) {
    for mut t in &mut q {
        t.rotate_y(time.delta_secs() * 1.2);
    }
}
