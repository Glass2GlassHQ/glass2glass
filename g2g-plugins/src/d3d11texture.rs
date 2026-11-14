// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#[cfg(any(
    feature = "d3d11-sink",
    feature = "d3d11-convert",
    feature = "mf-encode"
))]
pub(crate) use consume::{texture_objects, texture_proposal};
#[cfg(all(test, feature = "d3d11-sink"))]
pub(crate) use consume::{D3D11_ALIGN, D3D11_POOL_HEADROOM};
#[cfg(any(feature = "d3d11-convert", feature = "d3d11-screen-capture"))]
pub(crate) use produce::{texture_frame, TextureOwner};

#[cfg(any(feature = "d3d11-convert", feature = "d3d11-screen-capture"))]
mod produce {
    use alloc::sync::Arc;

    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;

    use g2g_core::memory::D3D11KeepAlive;
    use g2g_core::{MemoryDomain, OwnedD3D11Texture};

    #[derive(Debug)]
    pub(crate) struct TextureOwner(pub(crate) ID3D11Texture2D);

    impl D3D11KeepAlive for TextureOwner {}

    pub(crate) fn texture_frame(
        device: &ID3D11Device,
        texture: Arc<TextureOwner>,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> MemoryDomain {
        MemoryDomain::D3D11Texture(OwnedD3D11Texture::new(
            texture.0.as_raw() as u64,
            0,
            width,
            height,
            format.0 as u32,
            device.as_raw() as u64,
            texture,
        ))
    }
}

#[cfg(any(
    feature = "d3d11-sink",
    feature = "d3d11-convert",
    feature = "mf-encode"
))]
mod consume {
    use windows::core::Interface;
    use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};

    use g2g_core::memory::DomainSet;
    use g2g_core::{AllocationParams, Caps, Dim, G2gError, HardwareError, OwnedD3D11Texture};

    // the frame in flight in the consumer plus the one the runner link holds, so a decoder's texture pool does not starve under live pacing
    pub(crate) const D3D11_POOL_HEADROOM: usize = 3;
    // 256 bytes is the common D3D11 and DXVA texture alignment
    pub(crate) const D3D11_ALIGN: usize = 256;

    pub(crate) fn texture_objects(
        texture: &OwnedD3D11Texture,
    ) -> Result<(ID3D11Device, ID3D11Texture2D), G2gError> {
        // SAFETY: both pointers come from a live frame whose keep-alive holds the
        // texture, and cloning takes a reference of our own.
        unsafe {
            let device_raw = texture.device as *mut core::ffi::c_void;
            let texture_raw = texture.texture as *mut core::ffi::c_void;
            let device = ID3D11Device::from_raw_borrowed(&device_raw)
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            let input = ID3D11Texture2D::from_raw_borrowed(&texture_raw)
                .ok_or(G2gError::Hardware(HardwareError::Other))?;
            Ok((device.clone(), input.clone()))
        }
    }

    pub(crate) fn texture_proposal(caps: &Caps, accepts: DomainSet) -> Option<AllocationParams> {
        let Caps::RawVideo {
            format,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            ..
        } = caps
        else {
            return None;
        };
        let size = crate::pixel::frame_byte_size(*format, *width, *height);
        let mut params = AllocationParams::d3d11(size, D3D11_POOL_HEADROOM, D3D11_ALIGN);
        params.accepts = accepts;
        Some(params)
    }
}
