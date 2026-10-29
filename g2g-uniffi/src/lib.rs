use std::fmt;
use std::sync::Arc;

use g2g_core::runtime::block_on;
use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull, BusKind, HostFrame};
use g2g_plugins::appsrc::{register_appsrc, AppSrcFeed};
use g2g_plugins::launched::{EditError, LaunchError, LaunchedPipeline, RunFailure};

#[cfg(target_os = "android")]
mod surfaces;

uniffi::setup_scaffolding!();

#[derive(Debug, uniffi::Error)]
#[uniffi(flat_error)]
pub enum PipelineError {
    Launch(LaunchError),
    Edit(EditError),
    Run(RunFailure),
    NotHostMemory,
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineError::Launch(error) => error.fmt(f),
            PipelineError::Edit(error) => error.fmt(f),
            PipelineError::Run(error) => error.fmt(f),
            PipelineError::NotHostMemory => f.write_str("the frame is not in host memory"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, uniffi::Enum)]
pub enum BusMessageKind {
    StreamStart,
    Eos,
    Error,
    Warning,
    Info,
    StateChanged,
    Buffering,
    DurationChanged,
    Other,
}

impl From<BusKind> for BusMessageKind {
    fn from(kind: BusKind) -> Self {
        match kind {
            BusKind::StreamStart => BusMessageKind::StreamStart,
            BusKind::Eos => BusMessageKind::Eos,
            BusKind::Error => BusMessageKind::Error,
            BusKind::Warning => BusMessageKind::Warning,
            BusKind::Info => BusMessageKind::Info,
            BusKind::StateChanged => BusMessageKind::StateChanged,
            BusKind::Buffering => BusMessageKind::Buffering,
            BusKind::DurationChanged => BusMessageKind::DurationChanged,
            BusKind::Other => BusMessageKind::Other,
        }
    }
}

// a and b mean what G2gBusMessage in g2g.h says for the same kind
#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct BusMessage {
    pub kind: BusMessageKind,
    pub text: Option<String>,
    pub a: u64,
    pub b: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct Stats {
    pub frames_emitted: u64,
    pub frames_consumed: u64,
    pub frames_dropped: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, uniffi::Record)]
pub struct Sample {
    pub data: Vec<u8>,
    pub pts_ns: u64,
}

#[derive(Debug, uniffi::Object)]
pub struct Pipeline {
    launched: LaunchedPipeline,
}

#[uniffi::export]
impl Pipeline {
    #[uniffi::constructor]
    pub fn new(description: &str) -> Result<Arc<Self>, PipelineError> {
        LaunchedPipeline::launch(description, "g2g-uniffi-run")
            .map(|launched| Arc::new(Pipeline { launched }))
            .map_err(PipelineError::Launch)
    }

    pub fn poll_bus(&self) -> Option<BusMessage> {
        self.launched.bus_poll().map(|flat| BusMessage {
            kind: flat.kind.into(),
            text: flat.text,
            a: flat.a,
            b: flat.b,
        })
    }

    pub fn is_done(&self) -> bool {
        self.launched.is_done()
    }

    // The error itself is also posted on the bus.
    pub fn wait(&self) -> Result<Stats, PipelineError> {
        let stats = self.launched.wait().map_err(PipelineError::Run)?;
        Ok(Stats {
            frames_emitted: stats.frames_emitted,
            frames_consumed: stats.frames_consumed,
            frames_dropped: stats.frames_dropped,
        })
    }

    pub fn insert_after(&self, node: &str, description: &str) -> Result<String, PipelineError> {
        self.launched
            .insert_after(node, description)
            .map_err(PipelineError::Edit)
    }

    pub fn insert_before(&self, node: &str, description: &str) -> Result<String, PipelineError> {
        self.launched
            .insert_before(node, description)
            .map_err(PipelineError::Edit)
    }

    pub fn insert_chain_after(
        &self,
        node: &str,
        description: &str,
    ) -> Result<Vec<String>, PipelineError> {
        self.launched
            .insert_chain_after(node, description)
            .map_err(PipelineError::Edit)
    }

    pub fn insert_chain_before(
        &self,
        node: &str,
        description: &str,
    ) -> Result<Vec<String>, PipelineError> {
        self.launched
            .insert_chain_before(node, description)
            .map_err(PipelineError::Edit)
    }

    pub fn remove(&self, node: &str) -> Result<(), PipelineError> {
        self.launched.remove(node).map_err(PipelineError::Edit)
    }

    pub fn add_branch(&self, tee: &str, description: &str) -> Result<Vec<String>, PipelineError> {
        self.launched
            .add_branch(tee, description)
            .map_err(PipelineError::Edit)
    }

    pub fn remove_branch(&self, sink: &str) -> Result<(), PipelineError> {
        self.launched
            .remove_branch(sink)
            .map_err(PipelineError::Edit)
    }

    pub fn add_input(&self, muxer: &str, description: &str) -> Result<String, PipelineError> {
        self.launched
            .add_input(muxer, description)
            .map_err(PipelineError::Edit)
    }

    pub fn replace_sink(&self, node: &str, description: &str) -> Result<String, PipelineError> {
        self.launched
            .replace_sink(node, description)
            .map_err(PipelineError::Edit)
    }

    pub fn replace_source(&self, node: &str, description: &str) -> Result<String, PipelineError> {
        self.launched
            .replace_source(node, description)
            .map_err(PipelineError::Edit)
    }

    pub fn set_property(&self, node: &str, name: &str, value: &str) -> Result<(), PipelineError> {
        self.launched
            .set_property(node, name, value)
            .map_err(PipelineError::Edit)
    }

    // None when the element has no such property.
    pub fn get_property(&self, node: &str, name: &str) -> Result<Option<String>, PipelineError> {
        self.launched
            .get_property(node, name)
            .map_err(PipelineError::Edit)
    }
}

// Create it before the pipeline that names its channel.
#[derive(Debug, uniffi::Object)]
pub struct AppSrc {
    feed: AppSrcFeed,
}

#[uniffi::export]
impl AppSrc {
    #[uniffi::constructor(default(channel = "default"))]
    pub fn new(channel: &str) -> Arc<Self> {
        Arc::new(AppSrc {
            feed: register_appsrc(channel),
        })
    }

    // False when the feed is full (retry) or the pipeline is gone.
    pub fn push(&self, data: Vec<u8>, pts_ns: u64) -> bool {
        self.feed.push(&data, pts_ns)
    }

    pub fn end_of_stream(&self) -> bool {
        self.feed.end_of_stream()
    }
}

// Create it before the pipeline that names its channel.
#[derive(Debug, uniffi::Object)]
pub struct AppSink {
    pull: AppSinkPull,
}

#[uniffi::export]
impl AppSink {
    #[uniffi::constructor(default(channel = "default"))]
    pub fn new(channel: &str) -> Arc<Self> {
        Arc::new(AppSink {
            pull: register_appsink_pull(channel),
        })
    }

    // Blocks for the next sample, null once the stream has ended.
    pub fn pull(&self) -> Result<Option<Sample>, PipelineError> {
        let Some(frame) = block_on(self.pull.pull()) else {
            return Ok(None);
        };
        let host = HostFrame::new(frame);
        let data = host.bytes().ok_or(PipelineError::NotHostMemory)?.to_vec();
        Ok(Some(Sample {
            data,
            pts_ns: host.frame().timing.pts_ns,
        }))
    }
}
