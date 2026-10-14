use std::process::Command;

pub(crate) fn have_ffmpeg() -> bool {
    Command::new("ffmpeg").arg("-version").output().is_ok()
}
