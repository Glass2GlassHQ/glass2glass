// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use core::mem::ManuallyDrop;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, ID3D11VideoContext, ID3D11VideoDevice,
    ID3D11VideoProcessor, ID3D11VideoProcessorEnumerator, ID3D11VideoProcessorInputView,
    ID3D11VideoProcessorOutputView, D3D11_TEX2D_VPIV, D3D11_TEX2D_VPOV,
    D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE, D3D11_VIDEO_PROCESSOR_CONTENT_DESC,
    D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0,
    D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC, D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0,
    D3D11_VIDEO_PROCESSOR_STREAM, D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
    D3D11_VPIV_DIMENSION_TEXTURE2D, D3D11_VPOV_DIMENSION_TEXTURE2D,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_RATIONAL;

use g2g_core::{G2gError, HardwareError};

use crate::mfcommon::mf_err;

pub(crate) struct VideoProcessor {
    pub(crate) video_device: ID3D11VideoDevice,
    pub(crate) video_context: ID3D11VideoContext,
    pub(crate) vp_enum: ID3D11VideoProcessorEnumerator,
    pub(crate) processor: ID3D11VideoProcessor,
}

impl core::fmt::Debug for VideoProcessor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VideoProcessor").finish_non_exhaustive()
    }
}

impl VideoProcessor {
    pub(crate) fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        (input_width, input_height): (u32, u32),
        (output_width, output_height): (u32, u32),
    ) -> Result<Self, G2gError> {
        // SAFETY: COM object creation, every object from `device`.
        unsafe {
            let video_device: ID3D11VideoDevice = device.cast().map_err(mf_err)?;
            let video_context: ID3D11VideoContext = context.cast().map_err(mf_err)?;
            let rate = DXGI_RATIONAL {
                Numerator: 60,
                Denominator: 1,
            };
            let content = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
                InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
                InputFrameRate: rate,
                InputWidth: input_width,
                InputHeight: input_height,
                OutputFrameRate: rate,
                OutputWidth: output_width,
                OutputHeight: output_height,
                Usage: D3D11_VIDEO_USAGE_PLAYBACK_NORMAL,
            };
            let vp_enum = video_device
                .CreateVideoProcessorEnumerator(&content)
                .map_err(mf_err)?;
            let processor = video_device
                .CreateVideoProcessor(&vp_enum, 0)
                .map_err(mf_err)?;
            Ok(Self {
                video_device,
                video_context,
                vp_enum,
                processor,
            })
        }
    }

    pub(crate) fn blt(
        &self,
        input: &ID3D11Texture2D,
        array_slice: u32,
        output: &ID3D11Texture2D,
    ) -> Result<(), G2gError> {
        // SAFETY: both textures hold their own references for this call and belong to the processor's device.
        unsafe {
            let in_desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
                FourCC: 0,
                ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_VPIV {
                        MipSlice: 0,
                        ArraySlice: array_slice,
                    },
                },
            };
            let mut input_view: Option<ID3D11VideoProcessorInputView> = None;
            self.video_device
                .CreateVideoProcessorInputView(
                    input,
                    &self.vp_enum,
                    &in_desc,
                    Some(&mut input_view),
                )
                .map_err(mf_err)?;
            let input_view = input_view.ok_or(G2gError::Hardware(HardwareError::Other))?;

            let out_desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
                ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 },
                },
            };
            let mut output_view: Option<ID3D11VideoProcessorOutputView> = None;
            self.video_device
                .CreateVideoProcessorOutputView(
                    output,
                    &self.vp_enum,
                    &out_desc,
                    Some(&mut output_view),
                )
                .map_err(mf_err)?;
            let output_view = output_view.ok_or(G2gError::Hardware(HardwareError::Other))?;

            let mut streams = [D3D11_VIDEO_PROCESSOR_STREAM {
                Enable: true.into(),
                OutputIndex: 0,
                InputFrameOrField: 0,
                PastFrames: 0,
                FutureFrames: 0,
                ppPastSurfaces: core::ptr::null_mut(),
                pInputSurface: ManuallyDrop::new(Some(input_view)),
                ppFutureSurfaces: core::ptr::null_mut(),
                ppPastSurfacesRight: core::ptr::null_mut(),
                pInputSurfaceRight: ManuallyDrop::new(None),
                ppFutureSurfacesRight: core::ptr::null_mut(),
            }];
            let blt =
                self.video_context
                    .VideoProcessorBlt(&self.processor, &output_view, 0, &streams);
            // the stream struct holds the input view in ManuallyDrop, so it is released by hand
            drop(ManuallyDrop::into_inner(core::mem::replace(
                &mut streams[0].pInputSurface,
                ManuallyDrop::new(None),
            )));
            drop(ManuallyDrop::into_inner(core::mem::replace(
                &mut streams[0].pInputSurfaceRight,
                ManuallyDrop::new(None),
            )));
            blt.map_err(mf_err)
        }
    }
}
