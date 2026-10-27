use std::convert::Infallible;
use std::future::Future;
use std::pin::pin;
use std::process::ExitCode;
use std::time::Duration;

use g2g_core::graph::{Graph, GraphError, NodeId};
use g2g_core::runtime::{
    GraphMutator, GraphNode, MutationError, Observer, RunBuilder, RunStats, StateController,
};
use g2g_core::{Bus, BusHandle, G2gError, PipelineState, PropValue};
use g2g_plugins::checksumsink::ChecksumSink;
use g2g_plugins::clock::WallClock;
use g2g_plugins::dashboard;
use g2g_plugins::ffmpegdec::FfmpegH264Dec;
use g2g_plugins::h264parse::H264Parse;
use g2g_plugins::identity::IdentityTransform;
use g2g_plugins::rtspsrc::RtspSrc;
use g2g_plugins::videoscale::VideoScale;
use tokio::sync::broadcast;

const USAGE: &str = "usage: g2g-api-soak [-q] --observe <port> <rebuild | mutate>";
const SOURCE_URL: &str = "rtsp://127.0.0.1:18554/baseline";
const OBSERVE_HOST: &str = "127.0.0.1";
// g2g-launch's default link depth
const LINK_CAPACITY: usize = 4;
const BUS_DEPTH: usize = 256;
const EVENT_DEPTH: usize = 256;

const PARSER: &str = "parse";
const DECODER: &str = "dec";
const TEE: &str = "tee";
const VIDEO_SINK: &str = "video";

#[derive(Debug, Clone, Copy)]
enum Scenario {
    Rebuild,
    Mutate,
}

#[derive(Debug)]
struct Options {
    quiet: bool,
    observe_port: u16,
    scenario: Scenario,
}

#[derive(Debug)]
enum SoakError {
    Server(std::io::Error),
    Graph(GraphError),
    // the run returned while the scenario still had it running
    RunEnded {
        frames_consumed: u64,
    },
    RunFailed(G2gError),
    Stop(G2gError),
    StopTimedOut,
    Mutation {
        operation: &'static str,
        error: MutationError,
    },
    MutationTimedOut(&'static str),
    PropertyNotApplied {
        written: PropValue,
        read: Option<PropValue>,
    },
}

impl std::fmt::Display for SoakError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SoakError::Server(error) => write!(f, "observe server: {error}"),
            SoakError::Graph(error) => write!(f, "graph build: {error:?}"),
            SoakError::RunEnded { frames_consumed } => write!(
                f,
                "run ended on its own after {frames_consumed} frames into the sinks"
            ),
            SoakError::RunFailed(error) => write!(f, "pipeline error: {error:?}"),
            SoakError::Stop(error) => write!(f, "stopping the run: {error:?}"),
            SoakError::StopTimedOut => {
                write!(f, "the run did not end within {STOP_TIMEOUT:?} of NULL")
            }
            SoakError::Mutation { operation, error } => write!(f, "{operation}: {error:?}"),
            SoakError::MutationTimedOut(operation) => {
                write!(f, "{operation}: no answer within {MUTATION_TIMEOUT:?}")
            }
            SoakError::PropertyNotApplied { written, read } => write!(
                f,
                "{PARSER} {CONFIG_INTERVAL}: wrote {written:?}, read back {read:?}"
            ),
        }
    }
}

fn run_ended(ended: Result<RunStats, G2gError>) -> SoakError {
    match ended {
        Ok(stats) => SoakError::RunEnded {
            frames_consumed: stats.frames_consumed,
        },
        Err(error) => SoakError::RunFailed(error),
    }
}

fn parse_options(mut args: impl Iterator<Item = String>) -> Option<Options> {
    let mut quiet = false;
    let mut observe_port = None;
    let mut scenario = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-q" => quiet = true,
            "--observe" => observe_port = Some(args.next()?.parse().ok()?),
            "rebuild" => scenario = Some(Scenario::Rebuild),
            "mutate" => scenario = Some(Scenario::Mutate),
            _ => return None,
        }
    }
    Some(Options {
        quiet,
        observe_port: observe_port?,
        scenario: scenario?,
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    g2g_core::log::init_from_env();
    let Some(options) = parse_options(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match soak(options).await {
        Ok(never) => match never {},
        Err(error) => {
            eprintln!("g2g-api-soak: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn soak(options: Options) -> Result<Infallible, SoakError> {
    let observer = Observer::new();
    let (bus, bus_handle) = Bus::new(BUS_DEPTH);
    let (events, _) = broadcast::channel::<String>(EVENT_DEPTH);
    let forwarded = events.clone();
    tokio::spawn(async move {
        while let Some(message) = bus.recv().await {
            if let Some(json) = dashboard::event_json(&message) {
                let _ = forwarded.send(json);
            }
        }
    });
    let server = dashboard::serve(
        observer.clone(),
        events,
        OBSERVE_HOST,
        options.observe_port,
        std::future::pending(),
    );
    let scenario = async {
        match options.scenario {
            Scenario::Rebuild => rebuild(&observer, &bus_handle, options.quiet).await,
            Scenario::Mutate => mutate(&observer, &bus_handle, options.quiet).await,
        }
    };
    tokio::select! {
        served = server => match served {
            Err(error) => Err(SoakError::Server(error)),
            Ok(()) => unreachable!("the server stops only on a shutdown that never comes"),
        },
        failed = scenario => failed,
    }
}

// rtspsrc ! h264parse ! avdec_h264, the chain the baseline table was decoded through
fn add_decode_chain(graph: &mut Graph<GraphNode>) -> Result<NodeId, GraphError> {
    let source = graph.add_source(GraphNode::source(RtspSrc::new(SOURCE_URL)));
    let parser = graph.add_transform(GraphNode::element(H264Parse::reframing()));
    graph.set_node_name(parser, PARSER.into());
    let decoder = graph.add_transform(GraphNode::element(FfmpegH264Dec::new()));
    graph.set_node_name(decoder, DECODER.into());
    graph.link(source, parser)?;
    graph.link(parser, decoder)?;
    Ok(decoder)
}

fn add_video_sink(graph: &mut Graph<GraphNode>) -> NodeId {
    let sink = graph.add_sink(GraphNode::element(ChecksumSink::new()));
    graph.set_node_name(sink, VIDEO_SINK.into());
    sink
}

fn decode_graph() -> Result<Graph<GraphNode>, GraphError> {
    let mut graph = Graph::new();
    let decoder = add_decode_chain(&mut graph)?;
    let sink = add_video_sink(&mut graph);
    graph.link(decoder, sink)?;
    Ok(graph)
}

// the decoder feeds a tee, so branches can be added beside the video sink
fn tee_graph() -> Result<Graph<GraphNode>, GraphError> {
    let mut graph = Graph::new();
    let decoder = add_decode_chain(&mut graph)?;
    let tee = graph.add_tee(1);
    graph.set_node_name(tee.node(), TEE.into());
    let sink = add_video_sink(&mut graph);
    graph.link(decoder, tee.input())?;
    graph.link(tee.out(0), sink)?;
    Ok(graph)
}

const REBUILD_RUN: Duration = Duration::from_secs(60);
const STOP_TIMEOUT: Duration = Duration::from_secs(10);

// builds the graph, runs it, stops it with a state change to NULL and drops it, forever
async fn rebuild(
    observer: &Observer,
    bus: &BusHandle,
    quiet: bool,
) -> Result<Infallible, SoakError> {
    let clock = WallClock::new();
    let mut cycle = 0u64;
    loop {
        cycle += 1;
        let graph = decode_graph().map_err(SoakError::Graph)?;
        let state = StateController::new(PipelineState::Playing);
        let run = RunBuilder::new(graph, &clock, LINK_CAPACITY)
            .with_observer(observer)
            .with_bus(bus)
            .with_state(&state)
            .run();
        let mut run = pin!(run);
        tokio::select! {
            ended = &mut run => return Err(run_ended(ended)),
            () = tokio::time::sleep(REBUILD_RUN) => {}
        }
        state.set_state(PipelineState::Null);
        let stats = tokio::time::timeout(STOP_TIMEOUT, run)
            .await
            .map_err(|_| SoakError::StopTimedOut)?
            .map_err(SoakError::Stop)?;
        if !quiet {
            println!(
                "rebuild {cycle}: stopped after {} frames into the sinks",
                stats.frames_consumed
            );
        }
    }
}

const MUTATE_STEP: Duration = Duration::from_secs(4);
const MUTATION_TIMEOUT: Duration = Duration::from_secs(10);
const SCALED_WIDTH: u32 = 320;
const SCALED_HEIGHT: u32 = 180;
const CONFIG_INTERVAL: &str = "config-interval";
// h264parse repeats sps and pps before every idr, then stops again
const CONFIG_INTERVAL_EVERY_IDR: i64 = -1;
const CONFIG_INTERVAL_OFF: i64 = 0;

// one graph that runs forever while the main chain and a tee branch change under it
async fn mutate(
    observer: &Observer,
    bus: &BusHandle,
    quiet: bool,
) -> Result<Infallible, SoakError> {
    let clock = WallClock::new();
    let graph = tee_graph().map_err(SoakError::Graph)?;
    let (mutator, run) = RunBuilder::new(graph, &clock, LINK_CAPACITY)
        .with_observer(observer)
        .with_bus(bus)
        .run_mutable();
    tokio::select! {
        ended = run => Err(run_ended(ended)),
        failed = mutate_forever(&mutator, quiet) => failed,
    }
}

async fn mutate_forever(mutator: &GraphMutator<'_>, quiet: bool) -> Result<Infallible, SoakError> {
    let mut cycle = 0u64;
    loop {
        cycle += 1;
        tokio::time::sleep(MUTATE_STEP).await;
        let branch = answered(
            "add_branch",
            mutator.add_branch(
                TEE,
                vec![
                    Box::new(VideoScale::new(SCALED_WIDTH, SCALED_HEIGHT)),
                    Box::new(ChecksumSink::new()),
                ],
            ),
        )
        .await?;
        tokio::time::sleep(MUTATE_STEP).await;

        let config_interval = PropValue::Int(if cycle.is_multiple_of(2) {
            CONFIG_INTERVAL_OFF
        } else {
            CONFIG_INTERVAL_EVERY_IDR
        });
        answered(
            "set_property",
            mutator.set_property(PARSER, CONFIG_INTERVAL, config_interval.clone()),
        )
        .await?;
        let read = answered(
            "get_property",
            mutator.get_property(PARSER, CONFIG_INTERVAL),
        )
        .await?;
        if read.as_ref() != Some(&config_interval) {
            return Err(SoakError::PropertyNotApplied {
                written: config_interval,
                read,
            });
        }

        let spliced = answered(
            "insert_after",
            mutator.insert_after(DECODER, Box::new(IdentityTransform::new())),
        )
        .await?;
        tokio::time::sleep(MUTATE_STEP).await;
        answered("remove", mutator.remove(&spliced)).await?;

        let branch_sink = branch
            .last()
            .expect("add_branch names every element it adds");
        answered("remove_branch", mutator.remove_branch(branch_sink)).await?;
        if !quiet {
            println!("mutate {cycle}: added {branch:?}, spliced {spliced}, both removed");
        }
    }
}

// a mutation that never answers is a failure too
async fn answered<T>(
    operation: &'static str,
    call: impl Future<Output = Result<T, MutationError>>,
) -> Result<T, SoakError> {
    match tokio::time::timeout(MUTATION_TIMEOUT, call).await {
        Err(_) => Err(SoakError::MutationTimedOut(operation)),
        Ok(answer) => answer.map_err(|error| SoakError::Mutation { operation, error }),
    }
}
