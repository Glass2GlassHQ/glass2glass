#![allow(dead_code)] // no one test file uses every helper here

const IVF_FILE_HEADER_LEN: usize = 32;
const IVF_FRAME_HEADER_LEN: usize = 12;
const IVF_FRAME_SIZE_LEN: usize = 4;

pub(crate) fn ivf_frames(file: &[u8]) -> Vec<&[u8]> {
    let mut offset = IVF_FILE_HEADER_LEN;
    let mut frames = Vec::new();
    while let Some(frame_header) = file.get(offset..offset + IVF_FRAME_HEADER_LEN) {
        let size_bytes: [u8; IVF_FRAME_SIZE_LEN] = frame_header[..IVF_FRAME_SIZE_LEN]
            .try_into()
            .expect("a four byte frame size");
        let start = offset + IVF_FRAME_HEADER_LEN;
        let end = start + u32::from_le_bytes(size_bytes) as usize;
        frames.push(file.get(start..end).expect("the frame fits in the file"));
        offset = end;
    }
    frames
}

pub(crate) fn ivf_frame_count(file: &[u8]) -> u64 {
    ivf_frames(file).len() as u64
}
