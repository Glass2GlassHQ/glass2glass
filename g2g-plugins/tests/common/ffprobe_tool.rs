use std::process::Command;

pub(crate) fn have_ffprobe() -> bool {
    Command::new("ffprobe").arg("-version").output().is_ok()
}
