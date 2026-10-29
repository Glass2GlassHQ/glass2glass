use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Mutex, OnceLock};
use std::thread::JoinHandle;

use g2g_core::runtime::{
    block_on, parse_chain, parse_element, parse_launch, parse_source, set_property_text,
    GraphMutator, MutationError, ParseError, RunBuilder, RunStats,
};
use g2g_core::{Bus, G2gError};

use crate::appsink::FlatBusMessage;
use crate::clock::WallClock;
use crate::registry::{default_registry, mutator_with_defaults};

// matches g2g-launch, low latency without starving the source
const LINK_CAPACITY: usize = 4;
// a full bus drops control messages instead of stalling the data path
const BUS_CAPACITY: usize = 64;

// A launch line running on its own thread, as the language bindings drive it.
#[derive(Debug)]
pub struct LaunchedPipeline {
    bus: Bus,
    mutator: GraphMutator<'static>,
    join: Mutex<Option<JoinHandle<()>>>,
    // the run thread sets it last, so is_done needs no lock that wait holds
    outcome: Arc<OnceLock<Result<RunStats, RunFailure>>>,
}

#[derive(Debug)]
pub enum LaunchError {
    Parse(ParseError),
    Spawn(std::io::Error),
    NotStarted,
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaunchError::Parse(error) => write!(f, "parse error: {error}"),
            LaunchError::Spawn(error) => write!(f, "failed to spawn run thread: {error}"),
            LaunchError::NotStarted => f.write_str("pipeline run thread did not start"),
        }
    }
}

#[derive(Debug)]
pub enum EditError {
    Parse(ParseError),
    Refused(MutationError),
}

// The refusal prints as its variant, which callers match on ("UnknownNode").
impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EditError::Parse(error) => write!(f, "parse error: {error}"),
            EditError::Refused(error) => write!(f, "{error:?}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunFailure {
    Error(G2gError),
    Panic(String),
}

impl fmt::Display for RunFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunFailure::Error(error) => f.write_str(&describe_error(error)),
            RunFailure::Panic(message) => write!(f, "pipeline run thread panicked: {message}"),
        }
    }
}

fn describe_error(error: &G2gError) -> String {
    match error {
        G2gError::CapsMismatch => {
            "caps negotiation failed (no common format between linked elements)".into()
        }
        G2gError::NotConfigured => "an element received data before configuration completed".into(),
        G2gError::FixationFailed => "caps fixation failed".into(),
        G2gError::PoolExhausted => "buffer pool exhausted".into(),
        G2gError::UnsupportedDomain => {
            "an element was handed a memory domain it cannot consume".into()
        }
        G2gError::AllocationConflict => "fan-out branches share no common memory domain".into(),
        G2gError::Hardware(hardware) => format!("hardware/driver failure: {hardware:?}"),
        G2gError::Shutdown => "pipeline shut down before completing".into(),
        other => format!("pipeline error: {other:?}"),
    }
}

fn panic_message(payload: &(dyn core::any::Any + Send)) -> String {
    if let Some(text) = payload.downcast_ref::<&str>() {
        (*text).into()
    } else if let Some(text) = payload.downcast_ref::<String>() {
        text.clone()
    } else {
        "unknown panic".into()
    }
}

impl LaunchedPipeline {
    pub fn launch(description: &str, thread_name: &str) -> Result<Self, LaunchError> {
        let graph = parse_launch(&default_registry(), description).map_err(LaunchError::Parse)?;
        let (bus, bus_handle) = Bus::new(BUS_CAPACITY);
        let (mutator_sender, mutator_receiver) = mpsc::sync_channel(1);
        let outcome = Arc::new(OnceLock::new());
        let run_outcome = Arc::clone(&outcome);
        let join = std::thread::Builder::new()
            .name(thread_name.into())
            .spawn(move || {
                let run = catch_unwind(AssertUnwindSafe(|| {
                    let runtime = tokio::runtime::Builder::new_current_thread()
                        .enable_time()
                        .build()
                        .expect("build tokio runtime");
                    let clock = WallClock::new();
                    let (mutator, run) = RunBuilder::new(graph, &clock, LINK_CAPACITY)
                        .with_bus(&bus_handle)
                        .run_mutable();
                    if mutator_sender.send(mutator).is_err() {
                        return Err(G2gError::Shutdown);
                    }
                    runtime.block_on(run)
                }));
                let _ = run_outcome.set(match run {
                    Ok(result) => result.map_err(RunFailure::Error),
                    Err(panic) => Err(RunFailure::Panic(panic_message(&*panic))),
                });
            })
            .map_err(LaunchError::Spawn)?;
        let Ok(mutator) = mutator_receiver.recv() else {
            let _ = join.join();
            return Err(LaunchError::NotStarted);
        };
        Ok(LaunchedPipeline {
            bus,
            mutator: mutator_with_defaults(mutator),
            join: Mutex::new(Some(join)),
            outcome,
        })
    }

    pub fn bus_poll(&self) -> Option<FlatBusMessage> {
        self.bus
            .try_recv()
            .map(|message| FlatBusMessage::new(&message))
    }

    pub fn is_done(&self) -> bool {
        self.outcome.get().is_some()
    }

    // Blocks until the run ends, then returns the same outcome on every call.
    pub fn wait(&self) -> Result<RunStats, RunFailure> {
        let mut join = self.join.lock().expect("a wait panicked while joining");
        if let Some(handle) = join.take() {
            handle
                .join()
                .expect("the run thread catches its own panics");
        }
        self.outcome
            .get()
            .cloned()
            .expect("the run thread sets the outcome before it ends")
    }

    pub fn insert_after(&self, node: &str, description: &str) -> Result<String, EditError> {
        let element = parse_element(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.insert_after(node, element)).map_err(EditError::Refused)
    }

    pub fn insert_before(&self, node: &str, description: &str) -> Result<String, EditError> {
        let element = parse_element(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.insert_before(node, element)).map_err(EditError::Refused)
    }

    pub fn insert_chain_after(
        &self,
        node: &str,
        description: &str,
    ) -> Result<Vec<String>, EditError> {
        let chain = parse_chain(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.insert_chain_after(node, chain)).map_err(EditError::Refused)
    }

    pub fn insert_chain_before(
        &self,
        node: &str,
        description: &str,
    ) -> Result<Vec<String>, EditError> {
        let chain = parse_chain(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.insert_chain_before(node, chain)).map_err(EditError::Refused)
    }

    pub fn add_branch(&self, tee: &str, description: &str) -> Result<Vec<String>, EditError> {
        let branch = parse_chain(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.add_branch(tee, branch)).map_err(EditError::Refused)
    }

    pub fn remove_branch(&self, sink: &str) -> Result<(), EditError> {
        block_on(self.mutator.remove_branch(sink))
            .map(drop)
            .map_err(EditError::Refused)
    }

    pub fn add_input(&self, muxer: &str, description: &str) -> Result<String, EditError> {
        let source = parse_source(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.add_input(muxer, source)).map_err(EditError::Refused)
    }

    pub fn remove(&self, node: &str) -> Result<(), EditError> {
        block_on(self.mutator.remove(node))
            .map(drop)
            .map_err(EditError::Refused)
    }

    pub fn replace_sink(&self, node: &str, description: &str) -> Result<String, EditError> {
        let element = parse_element(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.replace_sink(node, element))
            .map(|(name, _old)| name)
            .map_err(EditError::Refused)
    }

    pub fn replace_source(&self, node: &str, description: &str) -> Result<String, EditError> {
        let source = parse_source(&default_registry(), description).map_err(EditError::Parse)?;
        block_on(self.mutator.replace_source(node, source))
            .map(|(name, _old)| name)
            .map_err(EditError::Refused)
    }

    pub fn set_property(&self, node: &str, name: &str, value: &str) -> Result<(), EditError> {
        block_on(set_property_text(&self.mutator, node, name, value)).map_err(EditError::Refused)
    }

    // None when the element has no such property.
    pub fn get_property(&self, node: &str, name: &str) -> Result<Option<String>, EditError> {
        block_on(self.mutator.get_property(node, name))
            .map(|value| value.map(|value| format!("{value}")))
            .map_err(EditError::Refused)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::boxed::Box;
    use g2g_core::HardwareError;

    #[test]
    fn every_error_variant_has_its_own_message() {
        let cases = [
            G2gError::CapsMismatch,
            G2gError::NotConfigured,
            G2gError::FixationFailed,
            G2gError::PoolExhausted,
            G2gError::UnsupportedDomain,
            G2gError::AllocationConflict,
            G2gError::Hardware(HardwareError::Cuda(2)),
            G2gError::Shutdown,
        ];
        let messages: Vec<String> = cases.iter().map(describe_error).collect();
        for (index, message) in messages.iter().enumerate() {
            assert!(!message.is_empty());
            assert!(
                !messages[index + 1..].contains(message),
                "{message} is shared by two variants"
            );
        }
        assert!(describe_error(&G2gError::Hardware(HardwareError::Cuda(2))).contains('2'));
        assert!(describe_error(&G2gError::CapsMismatch).contains("caps"));
    }

    #[test]
    fn a_panic_payload_keeps_its_text() {
        let text: Box<dyn core::any::Any + Send> = Box::new("boom");
        assert_eq!(panic_message(&*text), "boom");
        let owned: Box<dyn core::any::Any + Send> = Box::new(String::from("kaboom"));
        assert_eq!(panic_message(&*owned), "kaboom");
        let other: Box<dyn core::any::Any + Send> = Box::new(42i32);
        assert_eq!(panic_message(&*other), "unknown panic");
    }
}
