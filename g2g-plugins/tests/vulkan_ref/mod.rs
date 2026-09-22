//! The software reference dumps the Vulkan decode tests compare against:
//! `G2G_VULKAN_REF_DIR` names a directory of raw planar dumps, one per fixture,
//! named after the fixture with a `.yuv` extension. `tools/vulkan-refs.sh`
//! generates them with ffmpeg.

const REFERENCE_DIRECTORY_VAR: &str = "G2G_VULKAN_REF_DIR";
const REFERENCE_EXTENSION: &str = "yuv";

// The dump for `fixture_file_name`, or `None` when the variable is unset (the
// caller then checks geometry only). Panics when the variable is set and the
// dump is missing, so a misnamed dump cannot pass vacuously.
pub(crate) fn reference_yuv(fixture_file_name: &str) -> Option<Vec<u8>> {
    let directory = std::env::var(REFERENCE_DIRECTORY_VAR).ok()?;
    let stem = fixture_file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(fixture_file_name);
    let path = std::path::Path::new(&directory).join(format!("{stem}.{REFERENCE_EXTENSION}"));
    match std::fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) => panic!(
            "{REFERENCE_DIRECTORY_VAR} is set but {} is unreadable: {error}",
            path.display()
        ),
    }
}
