//! Remote rendering with zero streaming code: a spinning cube on a lit plane,
//! rendered headless and streamed by `RemoteRenderPlugins`. With the `nvenc`
//! feature on an NVIDIA GPU the frames are encoded without ever leaving the
//! GPU; otherwise they are read back and encoded with libx264.
//!
//! `G2G_WHIP_URL` streams to a WHIP endpoint (e.g. MediaMTX); unset, the
//! H.264 goes to `bevy_g2g.h264`. `G2G_FRAMES` caps the run (default 240,
//! `0` = forever).

use bevy::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(bevy_g2g::RemoteRenderPlugins::from_env())
        .insert_resource(ClearColor(Color::srgb(0.05, 0.05, 0.1)))
        .add_systems(Startup, setup)
        .add_systems(Update, spin);
    bevy_g2g::run(app);
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
