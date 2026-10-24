use std::io::Write;

// SIGPIPE stays ignored for the sockets, so a closed stdout (`| head`) ends the process here as gst-launch does
pub fn write_stdout(text: core::fmt::Arguments) {
    let Err(err) = std::io::stdout().write_fmt(text) else {
        return;
    };
    #[cfg(unix)]
    if err.kind() == std::io::ErrorKind::BrokenPipe {
        // SAFETY: SIG_DFL is a valid disposition and raise signals only this process
        unsafe {
            libc::signal(libc::SIGPIPE, libc::SIG_DFL);
            libc::raise(libc::SIGPIPE);
        }
    }
    panic!("failed printing to stdout: {err}");
}
