#![cfg(all(target_os = "linux", feature = "dmabuf-wgpu"))]

use g2g_ml::wgpupreprocess::gpu_available;

include!("util/dmabuf_preprocess.rs");

// odd on both axes, then on one axis alone
const ODD_SIZES: [(u32, u32); 3] = [(7, 5), (7, 4), (6, 5)];
const LUMA_PLANE: usize = 0;
const CHROMA_PLANE: usize = 1;
// fills the byte past each odd-width luma row, which shifts the tensor if the shader reads it
const PADDING_BYTE: u8 = 0xFF;

fn nv12_caps(width: u32, height: u32) -> Caps {
    raw_caps(RawVideoFormat::Nv12, width, height)
}

// NV12 with every row, chroma included, at `RawVideoFormat::row_stride`, the dma-buf layout
fn single_stride_nv12(width: u32, height: u32) -> (Vec<u8>, u32) {
    let format = RawVideoFormat::Nv12;
    let stride = format.row_stride(width).unwrap();
    let frame_bytes = format
        .frame_bytes(u64::from(stride), u64::from(height))
        .unwrap() as usize;
    let luma_row = format.plane_stride(LUMA_PLANE, width).unwrap() as usize;
    let chroma_row = format.plane_stride(CHROMA_PLANE, width).unwrap() as usize;
    let chroma_rows = format.plane_rows(CHROMA_PLANE, height).unwrap() as usize;
    let (row_pitch, luma_rows) = (stride as usize, height as usize);
    let chroma_base = row_pitch * luma_rows;
    let mut frame = vec![PADDING_BYTE; frame_bytes];
    for y in 0..luma_rows {
        for x in 0..luma_row {
            frame[y * row_pitch + x] = (16 + (x * 13 + y * 29) % 220) as u8;
        }
    }
    for row in 0..chroma_rows {
        for byte in 0..chroma_row {
            frame[chroma_base + row * row_pitch + byte] = ((byte * 23 + row * 7) % 256) as u8;
        }
    }
    (frame, stride)
}

// the same pixels in the tight layout the System upload path takes
fn tight_nv12(single_stride: &[u8], stride: u32, width: u32, height: u32) -> Vec<u8> {
    let format = RawVideoFormat::Nv12;
    let row_pitch = stride as usize;
    let chroma_base = row_pitch * height as usize;
    let mut tight = repack_tight(
        &single_stride[..chroma_base],
        row_pitch,
        format.plane_stride(LUMA_PLANE, width).unwrap() as usize,
        height as usize,
    );
    tight.extend(repack_tight(
        &single_stride[chroma_base..],
        row_pitch,
        format.plane_stride(CHROMA_PLANE, width).unwrap() as usize,
        format.plane_rows(CHROMA_PLANE, height).unwrap() as usize,
    ));
    tight
}

#[tokio::test]
async fn odd_nv12_dmabuf_import_matches_system_upload() {
    let _gpu = GPU_LOCK.lock().await;
    if !gpu_available().await {
        eprintln!("skipping: no wgpu adapter on this host");
        return;
    }
    for (width, height) in ODD_SIZES {
        let (single_stride, stride) = single_stride_nv12(width, height);
        // the export derives its stride from the caps width
        let Some(frame) = export_dmabuf(&single_stride, &nv12_caps(stride, height)).await else {
            eprintln!("skipping: no GPU dma-buf export on this host");
            return;
        };
        let caps = nv12_caps(width, height);
        let Some(tensor) = preprocess_dmabuf(frame, &caps, ImportAdapter::default(), false).await
        else {
            return;
        };
        let tight = tight_nv12(&single_stride, stride, width, height);
        assert_eq!(
            frame_f32(&tensor),
            preprocess_system(&caps, tight).await,
            "{width}x{height} at stride {stride}"
        );
    }
}
