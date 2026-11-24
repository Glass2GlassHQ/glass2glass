// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#[cfg(all(target_os = "linux", feature = "display"))]
mod display;
#[cfg(all(target_os = "linux", feature = "display"))]
mod gpu;
#[cfg(all(target_os = "linux", feature = "display"))]
pub mod waylanddisplaysrc;

#[cfg(all(target_os = "linux", feature = "display"))]
pub fn register(registry: &mut g2g_core::runtime::Registry) -> &mut g2g_core::runtime::Registry {
    use g2g_core::runtime::SourceFactory;
    use waylanddisplaysrc::WaylandDisplaySrc;

    let declared = WaylandDisplaySrc::new()
        .rgba_caps()
        .expect("the default size and rate form valid caps");
    registry.register_source(
        SourceFactory::new("waylanddisplaysrc", declared, || {
            Box::new(WaylandDisplaySrc::new())
        })
        .with_template_outputs::<WaylandDisplaySrc>()
        .with_experimental(),
    )
}
