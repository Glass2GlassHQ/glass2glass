//! Shared wgpu device context for the GPU elements (M103): the Vello overlay
//! producer ([`crate::vellooverlay`]) and the [`WgpuSink`](crate::wgpusink)
//! consumer.
//!
//! A `wgpu::Texture` is bound to the device that created it, so a producer and a
//! sink can only hand a [`MemoryDomain::WgpuTexture`](g2g_core::MemoryDomain)
//! frame across with no copy if they share **one** device. [`GpuContext`] is that
//! shared handle: build it once, clone it into both elements. `wgpu::Device` /
//! `Queue` / `Adapter` / `Instance` are all cheap `Clone`s (reference-counted
//! inside wgpu), so cloning a `GpuContext` shares the GPU rather than duplicating
//! it.
//!
//! Gated on the GPU features (`vello-overlay` / `wgpu-sink`).

use g2g_core::memory::OwnedWgpuTexture;
use g2g_core::{G2gError, HardwareError, WgpuKeepAlive};

/// A shared wgpu device context. Clone it into each GPU element so they render
/// and present on the same device (the prerequisite for a copy-free
/// `WgpuTexture` handoff between a producer and a sink).
#[derive(Debug, Clone)]
pub struct GpuContext {
    /// The wgpu instance (used by an application to create a surface).
    pub instance: wgpu::Instance,
    /// The adapter the device was opened on (used for surface capabilities).
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Build a headless context (no surface): for the overlay, the offscreen
    /// sink, and tests. Picks a high-performance adapter.
    pub async fn headless() -> Result<Self, G2gError> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(gpu_err)?;
        Self::from_adapter(instance, adapter).await
    }

    /// Build a context whose adapter can present to `surface`. Use this for the
    /// on-screen sink: an application creates the window's `wgpu::Surface` from
    /// `instance`, then opens a compatible device here.
    pub async fn for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
    ) -> Result<Self, G2gError> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                ..Default::default()
            })
            .await
            .map_err(gpu_err)?;
        Self::from_adapter(instance, adapter).await
    }

    async fn from_adapter(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
    ) -> Result<Self, G2gError> {
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("g2g-gpu"),
                // The full pipeline fits within the default limits on a discrete
                // GPU; pass the adapter's so a constrained default tier does not
                // reject Vello's renderer construction.
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(gpu_err)?;
        Ok(Self { instance, adapter, device, queue })
    }
}

/// Keep-alive owner for a rendered wgpu texture (the [`WgpuKeepAlive`] payload of
/// [`MemoryDomain::WgpuTexture`](g2g_core::MemoryDomain)). Owns the
/// `wgpu::Texture`; the consuming sink recovers it via [`texture_of`]. Shared so
/// the overlay producer and the sink agree on the concrete type to downcast to.
#[derive(Debug)]
pub struct WgpuTextureKeepAlive(pub wgpu::Texture);

impl WgpuKeepAlive for WgpuTextureKeepAlive {
    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

/// Recover the `wgpu::Texture` from a [`OwnedWgpuTexture`] produced with a
/// [`WgpuTextureKeepAlive`]. Returns `None` if the frame's keep-alive is some
/// other producer's type (a foreign GPU domain this sink cannot present).
pub fn texture_of(owned: &OwnedWgpuTexture) -> Option<&wgpu::Texture> {
    owned
        .keep_alive()
        .as_any()
        .downcast_ref::<WgpuTextureKeepAlive>()
        .map(|k| &k.0)
}

/// Map any wgpu / Vello failure to a structured hardware error.
pub(crate) fn gpu_err<E>(_e: E) -> G2gError {
    G2gError::Hardware(HardwareError::Other)
}
