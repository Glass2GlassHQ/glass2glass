// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

// Times ovxinference per frame on a board with a VeriSilicon NPU. Run through
// tools/stm32mp2-npu-bench.sh, which also times ST's stai_mpu on the same model.
//
// Usage: ovxinference_bench <model.nb>

#[cfg(target_os = "linux")]
mod bench {
    use std::time::{Duration, Instant};

    use g2g_core::element::{AsyncElement, OutputSink, PushOutcome};
    use g2g_core::frame::{Frame, FrameTiming, PipelinePacket};

    use g2g_core::{
        Caps, Colorimetry, Dim, G2gError, Interlace, PixelAspect, PropValue, Rate, RawVideoFormat,
    };
    use g2g_ml::ovxinfer::OvxInference;
    use g2g_plugins::openvx::NetworkBinaryGraph;

    const WARMUP_RUNS: usize = 10;
    const TIMED_RUNS: usize = 200;

    #[derive(Debug)]
    struct Discard;

    impl OutputSink for Discard {
        fn poll_push(
            &mut self,
            _cx: &mut core::task::Context<'_>,
            packet_slot: &mut Option<PipelinePacket>,
        ) -> core::task::Poll<Result<PushOutcome, G2gError>> {
            packet_slot.take();
            core::task::Poll::Ready(Ok(PushOutcome::Accepted))
        }
    }

    fn timed(mut run: impl FnMut()) -> Vec<Duration> {
        for _ in 0..WARMUP_RUNS {
            run();
        }
        (0..TIMED_RUNS)
            .map(|_| {
                let start = Instant::now();
                run();
                start.elapsed()
            })
            .collect()
    }

    fn print_row(stage: &str, mut durations: Vec<Duration>) {
        durations.sort_unstable();
        let milliseconds = |duration: Duration| duration.as_secs_f64() * 1000.0;
        let mean = milliseconds(durations.iter().sum::<Duration>()) / durations.len() as f64;
        let p50 = milliseconds(durations[durations.len() / 2]);
        let p99 = milliseconds(durations[durations.len() * 99 / 100]);
        println!("{stage:<22} {p50:>8.2} {p99:>8.2} {mean:>8.2}");
    }

    pub(crate) fn run(model: &str) {
        let mut graph = NetworkBinaryGraph::load(model).expect("graph loads");
        let input_len = graph
            .inputs()
            .next()
            .and_then(|info| info.byte_len())
            .expect("one sized input");
        let input = vec![0u8; input_len];
        let graph_run = timed(|| graph.run(&[&input]).expect("graph runs"));
        let output_count = graph.outputs().count();
        let output_copy = timed(|| {
            for index in 0..output_count {
                graph.output(index).expect("output copies");
            }
        });
        drop(graph);

        let mut inference = OvxInference::from_file(model).expect("model loads");
        inference
            .set_property("attach-tensor", PropValue::Bool(true))
            .expect("attach-tensor is a property");
        let (width, height) = inference.input_dims();
        let video = Caps::RawVideo {
            format: RawVideoFormat::Rgb8,
            width: Dim::Fixed(width),
            height: Dim::Fixed(height),
            framerate: Rate::Any,
            interlace: Interlace::Any,
            colorimetry: Colorimetry::UNKNOWN,
            pixel_aspect: PixelAspect::Any,
        };
        let narrowed = inference
            .intercept_caps(&video)
            .expect("RGB video is accepted");
        inference.configure_pipeline(&narrowed).expect("configure");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .build()
            .expect("tokio runtime");
        let mut sequence = 0;
        let element = timed(|| {
            let frame = Frame::from_system_bytes(input.clone(), FrameTiming::default(), sequence);
            sequence += 1;
            runtime
                .block_on(inference.process(PipelinePacket::DataFrame(frame), &mut Discard))
                .expect("inference runs");
        });

        println!("{TIMED_RUNS} runs after {WARMUP_RUNS} warmup, milliseconds");
        println!("{:<22} {:>8} {:>8} {:>8}", "stage", "p50", "p99", "mean");
        print_row("copy in + graph", graph_run);
        print_row("copy out", output_copy);
        print_row("ovxinference frame", element);
    }
}

#[cfg(target_os = "linux")]
fn main() {
    let model = std::env::args()
        .nth(1)
        .expect("usage: ovxinference_bench <model.nb>");
    bench::run(&model);
}

#[cfg(not(target_os = "linux"))]
fn main() {}
