//! Bevy + g2g integration: remote rendering over WebRTC and zero-copy video
//! decode, with no pipeline code in the app.
//!
//! Streaming ([`RemoteRenderPlugins`]): the app renders headless and every
//! frame is encoded to H.264 and sent to a WHIP endpoint (or a file). With the
//! `nvenc` feature on an NVIDIA GPU the frames never leave the GPU (Bevy
//! renders on g2g's interop device, `WgpuToCuda` -> `NvEnc`); otherwise a
//! GPU -> CPU readback feeds the libx264 software encoder, which works on any
//! adapter.
//!
//! ```no_run
//! use bevy::prelude::*;
//!
//! fn main() {
//!     let mut app = App::new();
//!     app.add_plugins(bevy_g2g::RemoteRenderPlugins::from_env())
//!         .add_systems(Startup, |mut c: Commands| { /* spawn the scene */ });
//!     bevy_g2g::run(app); // runs, flushes the stream, exits the process
//! }
//! ```
//!
//! Decode (`decode` feature, [`VideoPlayerPlugin`]): a stock windowed Bevy app
//! keeps its own device; g2g decodes a clip onto it and any mesh tagged
//! [`VideoScreen`] plays the video, zero-copy.

mod input;
mod stream;

pub use input::RemoteInputPlugin;
pub use stream::{run, RemoteRenderPlugins, StreamOutput, StreamSettings, StreamTarget};

#[cfg(feature = "decode")]
mod decode;

#[cfg(feature = "decode")]
pub use decode::{VideoPlayback, VideoPlayerPlugin, VideoScreen};
