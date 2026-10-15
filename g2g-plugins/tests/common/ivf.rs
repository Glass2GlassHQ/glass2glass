const IVF_FILE_HEADER_LEN: usize = 32;
const IVF_FRAME_HEADER_LEN: usize = 12;
const IVF_FRAME_SIZE_LEN: usize = 4;

pub(crate) fn ivf_frame_count(file: &[u8]) -> u64 {
    let mut offset = IVF_FILE_HEADER_LEN;
    let mut frames = 0;
    while let Some(frame_header) = file.get(offset..offset + IVF_FRAME_HEADER_LEN) {
        let size_bytes: [u8; IVF_FRAME_SIZE_LEN] = frame_header[..IVF_FRAME_SIZE_LEN]
            .try_into()
            .expect("a four byte frame size");
        offset += IVF_FRAME_HEADER_LEN + u32::from_le_bytes(size_bytes) as usize;
        frames += 1;
    }
    frames
}
