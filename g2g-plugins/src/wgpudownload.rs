use core::fmt::Debug;
use core::future::Future;
use core::pin::Pin;

use alloc::boxed::Box;
use alloc::vec::Vec;

use g2g_core::log::{short_type_name, Target};
use g2g_core::memory::{
    DomainSet, MemoryDomainKind, OwnedWgpuBuffer, OwnedWgpuTexture, SystemSlice,
};
use g2g_core::meta::PlaneLayout;
use g2g_core::{
    g2g_error, AsyncElement, Caps, CapsConstraint, ConfigureOutcome, Dim, ElementMetadata,
    G2gError, MemoryDomain, OutputSink, PipelinePacket, RawVideoFormat,
};

use crate::gpu::{
    gpu_err, read_rgba_texture_dq, read_texture_plane, WgpuNv12Texture, WgpuTextureKeepAlive,
};

#[derive(Debug, Default)]
pub struct WgpuDownload {
    configured: bool,
    downloaded: u64,
    raw_video: Option<RawVideoGeometry>,
}

#[derive(Debug, Clone, Copy)]
struct RawVideoGeometry {
    format: RawVideoFormat,
    width: usize,
    height: usize,
}

impl RawVideoGeometry {
    fn of(caps: &Caps) -> Option<Self> {
        match caps {
            Caps::RawVideo {
                format,
                width: Dim::Fixed(width),
                height: Dim::Fixed(height),
                ..
            } => Some(Self {
                format: *format,
                width: *width as usize,
                height: *height as usize,
            }),
            _ => None,
        }
    }
}

impl WgpuDownload {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn downloaded(&self) -> u64 {
        self.downloaded
    }
}

fn download_texture(owned: &OwnedWgpuTexture) -> Result<Vec<u8>, G2gError> {
    let owner = owned.keep_alive();
    let any = owner.as_any();
    if let Some(k) = any.downcast_ref::<WgpuTextureKeepAlive>() {
        return read_texture_dq(k.device(), k.queue(), k.texture());
    }
    if let Some(k) = any.downcast_ref::<WgpuNv12Texture>() {
        return read_texture_dq(k.device(), k.queue(), k.texture());
    }
    #[cfg(all(target_os = "android", feature = "mediacodec-wgpu"))]
    if let Some(k) = any.downcast_ref::<crate::mediacodec_wgpu::WgpuRgbaTexture>() {
        return crate::mediacodec_wgpu::readback_rgba_texture(k);
    }
    Err(unreadable(MemoryDomainKind::WgpuTexture, owner))
}

fn read_texture_dq(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
) -> Result<Vec<u8>, G2gError> {
    let format = texture.format();
    let Some(planes) = format.planes() else {
        return read_rgba_texture_dq(device, queue, texture);
    };
    let mut out = Vec::new();
    for plane in 0..planes {
        let aspect = wgpu::TextureAspect::from_plane(plane).ok_or_else(|| gpu_err(()))?;
        let bpp = format
            .block_copy_size(Some(aspect))
            .ok_or_else(|| gpu_err(()))?;
        let (x_factor, y_factor) = format.subsampling_factors(Some(plane));
        let extent = (
            texture.width().div_ceil(x_factor),
            texture.height().div_ceil(y_factor),
        );
        out.extend(read_texture_plane(
            device, queue, texture, aspect, extent, bpp,
        )?);
    }
    Ok(out)
}

fn download_buffer(
    owned: &OwnedWgpuBuffer,
    raw_video: Option<RawVideoGeometry>,
) -> Result<Vec<u8>, G2gError> {
    let bytes = read_owned_buffer(owned)?;
    let Some(layout) = owned.plane_layout() else {
        return Ok(bytes);
    };
    let raw_video = raw_video.ok_or(G2gError::CapsMismatch)?;
    pack_planes(&bytes, layout, raw_video).ok_or(G2gError::CapsMismatch)
}

// `None` when the layout and the caps disagree or a row falls outside the buffer.
fn pack_planes(
    padded: &[u8],
    layout: &PlaneLayout,
    raw_video: RawVideoGeometry,
) -> Option<Vec<u8>> {
    let RawVideoGeometry {
        format,
        width,
        height,
    } = raw_video;
    let shapes = crate::paddedrows::plane_shapes_with_stride_shift(format, width, height);
    if shapes.len() != layout.count() {
        return None;
    }
    let mut packed = Vec::new();
    for (plane, (row_bytes, rows, _)) in shapes.into_iter().enumerate() {
        for row in 0..rows {
            packed.extend_from_slice(padded.get(layout.row_range(plane, row, row_bytes)?)?);
        }
    }
    Some(packed)
}

fn read_owned_buffer(owned: &OwnedWgpuBuffer) -> Result<Vec<u8>, G2gError> {
    let owner = owned.keep_alive();
    #[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
    {
        let any = owner.as_any();
        if let Some(b) = any.downcast_ref::<crate::wgpudmabuf::PlainWgpuBuffer>() {
            return read_buffer(b.device(), b.queue(), b.buffer(), owned.len);
        }
        if let Some(b) = any.downcast_ref::<crate::dmabufwgpu::DmaBufWgpuBuffer>() {
            return read_buffer(b.device(), b.queue(), b.buffer(), owned.len);
        }
    }
    Err(unreadable(MemoryDomainKind::WgpuBuffer, owner))
}

fn unreadable(domain: MemoryDomainKind, owner: &dyn Debug) -> G2gError {
    g2g_error!(
        Target::category(short_type_name::<WgpuDownload>()),
        "cannot read back a {domain:?} frame owned by {owner:?}: the owner type is not one this element knows"
    );
    G2gError::UnsupportedDomain
}

#[cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]
fn read_buffer(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Result<Vec<u8>, G2gError> {
    let copy_size = crate::dmabufwgpu::whole_word_size(len as u64).ok_or(G2gError::CapsMismatch)?;
    if copy_size > buffer.size() {
        return Err(G2gError::CapsMismatch);
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wgpudownload-staging"),
        size: copy_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, copy_size);
    queue.submit([encoder.finish()]);
    let mapped = crate::gpu::map_for_read(device, &staging)?;
    let bytes = mapped[..len].to_vec();
    drop(mapped);
    staging.unmap();
    Ok(bytes)
}

impl AsyncElement for WgpuDownload {
    type ProcessFuture<'a>
        = Pin<Box<dyn Future<Output = Result<(), G2gError>> + 'a>>
    where
        Self: 'a;

    fn metadata(&self) -> ElementMetadata {
        ElementMetadata::new(
            "wgpu download",
            "Filter/Converter/Video/GPU",
            "Reads a GPU-resident wgpu texture or buffer back to system memory",
            "g2g",
        )
    }

    fn intercept_caps(&self, upstream_caps: &Caps) -> Result<Caps, G2gError> {
        Ok(upstream_caps.clone())
    }

    fn caps_constraint_as_transform(&self) -> CapsConstraint<'_> {
        CapsConstraint::IdentityAny
    }

    fn input_domains(&self) -> DomainSet {
        DomainSet::only(MemoryDomainKind::WgpuTexture).with(MemoryDomainKind::WgpuBuffer)
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        self.raw_video = RawVideoGeometry::of(absolute_caps);
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn process<'a>(
        &'a mut self,
        packet: PipelinePacket,
        out: &'a mut dyn OutputSink,
    ) -> Self::ProcessFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            match packet {
                PipelinePacket::DataFrame(mut frame) => {
                    let bytes = match &frame.domain {
                        MemoryDomain::WgpuTexture(owned) => Some(download_texture(owned)?),
                        MemoryDomain::WgpuBuffer(owned) => {
                            Some(download_buffer(owned, self.raw_video)?)
                        }
                        _ => None,
                    };
                    if let Some(bytes) = bytes {
                        frame.domain =
                            MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice()));
                        self.downloaded += 1;
                    }
                    out.push(PipelinePacket::DataFrame(frame)).await?;
                }
                // The runner forwards end-of-stream after `process` returns.
                PipelinePacket::Eos => {}
                PipelinePacket::CapsChanged(caps) => {
                    self.raw_video = RawVideoGeometry::of(&caps);
                    out.push(PipelinePacket::CapsChanged(caps)).await?;
                }
                other => {
                    out.push(other).await?;
                }
            }
            Ok(())
        })
    }
}
