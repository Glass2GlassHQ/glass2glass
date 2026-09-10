# Timing: clocks, sync and QoS

Clock election and health, mid-graph pacing, the audio master clock, PTP, how a
sink anchors presentation, the latency fold, and the QoS report that travels back
upstream. Part of the design in [README.md](README.md), which defines
`FrameTiming` and the clock traits.

## Clock election and distribution

A pipeline runs against one elected clock. `elect_clock` orders candidates by
`ClockPriority`: a PTP grandmaster-disciplined clock (`PtpGrandmaster`) outranks a
live source's hardware clock (`LiveSource`), which outranks an audio sink's DAC
clock (`AudioProvider`), which outranks a plain monotonic provider such as a video
display sink (`Provider`), which outranks the system fallback.

The runner samples the elected clock's `now_ns()` once at startup as the base
time, the clock reading at running-time zero, and hands both to each sink via
`set_clock_sync(ClockSync { clock, base_time_ns })`, called once after election.
Both the linear runners and the DAG runner deliver it, the latter walking its sink
nodes after election, so a display sink PTS-paces in any topology. A sink that
synchronises presents a frame when the elected clock reaches
`base_time_ns + running_time`, where running time is the frame's `pts_ns` mapped
through the active `Segment`, and a sink that ignores the hook presents as fast as
backpressure allows.

An elected clock can lose the reference it is disciplined to, a PTP servo going
free-running when its grandmaster disappears, which `PipelineClock::healthy`
reports: true by default, since a clock reading a monotonic counter or a DAC has
nothing to lose, and overridden by `PtpClock`, healthy in `Locked` and `Holdover`
but not in `FreeRunning`.

When a bus is attached and the runner has a timer to sleep on, `run_graph`,
cooperative and thread-per-arm alike, runs a health monitor alongside the arms. It
reads the elected clock once a second and, on a loss, posts
`BusMessage::ClockLost`, elects again over the candidates that are still healthy,
and retargets every sink. The retarget works because in that mode the sinks'
`ClockSync` points at an `ElectedClock`, a shared handle over a swappable target,
rather than at the clock itself, since the elements are already inside their arms,
on other threads under the thread-per-arm runner, so there is no second
`set_clock_sync`. A sink re-anchors on its next frame, as it does for any epoch
change. `ElectedClock` answers `shared_ticker`, which is owned, but not
`as_ticker`, a borrow into a target that can be replaced, which is why the
indirection is installed only when the monitor runs. With no healthy candidate
left the pipeline keeps the clock it has: it still tells time, it is just not
disciplined, and a later re-lock is picked up by the same check.

## Pacing mid-graph

Presentation is not the only place a stream needs to run at real time. A publisher
muxing to a live transport, such as
`videotestsrc ! x264enc ! mp4mux ! moqtsink`, has no sync sink at all, so nothing
stops it producing minutes of media per minute of wall clock.

`ClockSyncTransform` (`clocksync`) is the sink's pacing as a pass-through
transform: it holds each buffer until its PTS, anchored on the first one, is due
on the clock, and forwards everything else unchanged. It shares the display sinks'
`PresentationPacer`, so the anchor, the segment mapping and the seek re-anchor
behave identically, and differs in two ways. It never drops, because a late or
segment-clipped buffer is forwarded immediately since a hole in a transform's
output is one downstream cannot recover. And it supplies its own monotonic clock
when none was handed to it, which is both GStreamer's fallback to the pipeline
system clock and a necessity here, since the runners deliver `ClockSync` to sink
nodes only and a `clocksync` sits mid-graph. `sync=false` reduces it to an
identity, and `ts-offset` shifts the whole schedule.

## Audio as the sync master

For playback the audio sink should drive timing, because samples leave the DAC at
the hardware's real rate, which drifts from wall time by tens to hundreds of ppm.

`DriftClock` (`g2g-core`) turns that into a usable pipeline clock. It is fed
`(local_ns, master_ns)` observations, `local_ns` from a monotonic reference and
`master_ns` the true playout position, and fits
`master ~= slope * local + offset` by least squares over a sliding window, so
`now_ns()` projects the current reference time through the fit, both estimating
the playout rate and smoothing the coarse, jittery per-observation readings.

The fit is exponentially weighted, each sample 0.95x the weight of the one after
it, newest first, so half of a rate step is taken up 24 samples after it rather
than 32, at the cost of 37% more slope noise. An observation landing further from
the current fit than the outlier gate, 10 ms by default and 20 ms for the PTP
servo below which sees delayed packets rather than delay jitter, is dropped rather
than folded in, so an underrun recovery or a stale `snd_pcm_delay()` reading cannot
bend the fit for a whole window. Eight rejections in a row mean the timeline
genuinely moved, a device re-open, so the window clears and the fit restarts from
the new samples.

`AlsaSink`'s worker samples `frames_written - snd_pcm_delay()` after each blocking
`writei` and feeds the clock, offering it to election at the `AudioProvider` tier,
gated by a `provide-clock` property. A video sink then slaves to it: because the
elected clock is the disciplined audio timeline rather than raw wall time, video
presentation follows audio, giving true A/V sync. A `LiveSource` capture clock
still wins when present, so a live pipeline paces to capture.

`PipeWireSink` offers the same `AudioProvider` clock on the PipeWire path. Its
realtime process callback probes `pw_stream_get_time_n` via `pipewire-sys`, the
0.8 safe binding having no wrapper, and feeds the graph tick counter, scaled
through the stream rate, as `master_ns`. The tick counter is the graph driver's
sample position, advancing at the device's real rate and independent of the sink's
leaky byte queue, so producer-side drops never skew the fit and the constant
graph-to-speaker delay lands in the affine offset.

## PTP

For facility-wide sync in Pro AV and SMPTE ST 2110, the shared reference is a PTP
grandmaster and every device slaves to it, so a `PtpGrandmaster` clock outranks
all of the above.

`PtpServo` (`g2g-core::ptp`) is the servo: fed the four timestamps of each PTP
delay request-response, it computes the standard `offset` and `mean_path_delay` and
folds `(local, master)` into the same `DriftClock` machinery, disciplining the
local monotonic reference to the grandmaster's TAI timeline with lock, holdover
and outlier-rejection state. `PtpClock` wraps it, interior-mutable so one worker
drives it while sinks read `now_ns` through a shared `Arc`, and offers itself to
election only once locked. Because the elected timeline is grandmaster-derived, two
machines locked to the same grandmaster read the same clock, so the A/V pacing
above holds across devices, not just within one process.

Two sources feed the servo: raw PTP message timestamps (`sync_exchange`) or a
direct absolute-time observation (`observe_master`). Two backends supply them.
`PtpSystemClock` (`g2g-plugins`, Linux) delegates to an OS PTP-disciplined
`CLOCK_TAI` from `linuxptp` or `phc2sys`, sampled on a worker. `PtpClient`
(`g2g-plugins`) is a from-scratch software PTP SLAVE that speaks PTP over UDP
itself, the `ptp::wire` message parser plus the `ptp::slave` delay-request-response
state machine plus a UDP transport, so an endpoint with no OS PTP daemon can still
lock. The wire parser and slave state machine are `no_std` and CI-tested end to end
from parse through slave to servo without sockets.

Both backends coexist with a host `ptp4l`. `PtpClient`'s sockets take
`SO_REUSEADDR` and `SO_REUSEPORT` so the daemon keeps receiving its own copy of
each multicast message, and `PtpSystemClock` polls the daemon over its management
socket (`ptp::management` builds the same GET `pmc` sends, and
`g2g-plugins::ptp4l` carries it over the Unix datagram socket) so
`grandmaster_locked` reports the port state behind `CLOCK_TAI` rather than trusting
a clock that is readable either way.

Distinct time newtypes guard the seam where three "just an integer" times meet:
`TaiNs` for PTP/TAI nanoseconds, absolute, `RtpTs` for the 32-bit wrapping RTP
media-clock timestamp on the wire, and `RefNs` for the pipeline's monotonic
reference nanoseconds, a relative timeline with an arbitrary epoch. `MediaClock`
takes a `TaiNs` and returns an `RtpTs`, so the compiler rejects handing it the
wrong clock, a monotonic reference minus a TAI master being meaningless. The PTP
servo's own seam is typed the same way: `sync_exchange` takes
`(TaiNs, RefNs, RefNs, TaiNs)` and `observe_master` takes `(RefNs, TaiNs)`, so
master and reference cannot be swapped. Durations stay a plain `u64`. The
ST 2110 media transport built over `MediaClock` is in
[transports.md](transports.md).

## Anchoring and the latency fold

`WaylandSink` holds each frame until its running-time deadline, tracking the
`Segment`, clipping pre-target frames after an accurate seek, and re-anchoring on
`Flush`. It also does QoS late-drop, matching `SyncSink`: a frame already past its
deadline by more than a configurable `max_lateness` bound is dropped instead of
presented late, so the sink catches up instead of accumulating lag, posting a
`BusMessage::Qos` with running time, jitter and cumulative processed and dropped
counts per drop.

`SyncSink` is the same sink without a display, so clock slaving is CI-testable with
no hardware. It paces through the shared `PresentationPacer` and adopts the elected
`ClockSync`, so in an A/V graph its deadlines land on the audio master's
`DriftClock` timeline. Its own clock stays the timer, the only thing that can sleep
in `no_std`, and the pacer's wait is relative so the two can be different
timelines. Until a clock is elected it paces on its own clock with the anchor
pinned at zero (`PresentationPacer::set_anchor_ns`), which makes a frame's deadline
its running time and the recorded drift a real end-to-end latency reading, and
adopting an elected clock drops the pin since that clock's epoch is its own.

The startup base time is sampled before the data plane and before the application
presses play. For a non-live, prerolled pipeline that sits in `Paused` for a while,
that is the wrong epoch: the preroll frame is consumed during `Paused`, so a sink
that anchored on the startup base, or on that first frame, then rushes or drops
once `Playing` finally arrives.

So when a `StateController` drives the run, the runner arms a `PlayAnchor`, a
shared cell, on the elected clock and hands each sink `ClockSync::with_play_anchor`.
`set_state(Playing)` stamps the anchor with `clock.now_ns()` at the exact play
edge, and a transition down to `Ready` or `Null` clears it so a replay re-bases.
`ClockSync::base_time()` then resolves to the play-edge stamp once armed, else the
eager startup base time. `WaylandSink` reads it per frame: it first-frame-anchors a
preroll frame consumed during `Paused`, presenting it immediately, then re-bases
onto the play edge once `Playing` stamps it, and a seek `Flush` forces a
first-frame re-anchor so the seek target presents immediately rather than against
the stale play base. The non-stateful runners keep the eager base time, having no
`StateController` and no play edge to anchor to.

Live paths never first-frame-anchor. Each runner folds the path's `LatencyReport`
into the sink's `ClockSync` through `with_path_latency`. When the aggregate says
live, the pacer anchors on `base_time()` unconditionally, stamped or eager, and
every deadline adds the aggregated minimum latency, GStreamer's
`base_time + running_time + latency` model. A startup stall, a hardware decoder's
first-frame initialization, then makes the opening frames late, they present
immediately and the backlog drains, instead of the stall being latched into a
first-frame anchor as the run's standing latency. First-frame anchoring remains the
non-live behaviour, where frames arrive at read speed and an absolute anchor would
defeat pacing entirely.

The DAG runner folds every node that carries an element: sources, transforms, sinks
and fan-ins, which contribute `MultiInputElement::latency()` the way a transform
contributes its own, so a `fallbackswitch` declares its stall slack there. A tee is
structural and contributes nothing, and a demux contributes nothing either since
`MultiOutputElement` has no `latency()` to declare one with.

The fold follows paths, not the node list. Each node's upstream aggregate is its
inputs' merged, plus its own contribution (`LatencyReport::combine`, the sum a
chain has always used). Where branches meet, they merge instead of summing
(`LatencyReport::join_branches`): a fan-in cannot produce until its slowest branch
delivers, so the minimums take the larger, and the branch that overflows first sets
the ceiling, so an unbounded branch never lifts a finite one. The run's figure
merges the same way over its sinks, GStreamer's rule for aggregating a latency
query across a bin. A branched graph therefore reports the slowest path rather than
the sum of every element in it, and a fan-in has the aggregate of the branches
feeding it, which is what a per-pad or minimum-upstream figure has to be measured
against.

The aggregate's `live` flag does not only fold into the sink's `ClockSync`, it also
reaches every element: each runner calls `AsyncElement::configure_liveness` after
the fold and before any arm starts, unconditionally, so a path with no live source
is told `false` rather than left to assume one. A `GraphMutator` splice hands the
run's flag to the spliced element beside its caps. `ffmpegdec` is what reads it
today: `thread-type=auto` resolves to slice threading off a live source and to
frame threading otherwise, gst-libav's rule, which trades the `thread_count - 1`
pictures libavcodec then holds back for throughput when nothing is pacing the
stream. The fold reads `latency()` first, so the pictures a decoder starts holding
because of the answer are not in that run's reported aggregate.

A first-frame anchor is `clock.now_ns() - running_time`, and a stream whose PTS
epoch runs ahead of the elected clock's puts that below zero: an audio `DriftClock`
reads the playout position, seconds since the device opened, while an HLS feed's
PTS is the publisher's uptime, hours. `PresentationPacer` keeps `anchor_ns` as an
`i64` for exactly that case, clamping only the finished deadline at zero, so a
video sink slaved to an audio master presents its first frame on arrival rather
than holding it for the difference between the two epochs.

## Deadline telemetry

Once a frame is on screen `WaylandSink` reads the elected clock again and records
the signed difference from that frame's deadline
(`PresentationPacer::last_deadline_ns`, captured where `judge` derived it, so the
reading is against the anchor the frame was actually held to).
`deadline_error_samples()` returns the run's samples, capped like the latency
samples and empty for an unpaced sink. In an A/V graph the elected clock is the
audio sink's, which makes the series the video-against-audio sync error: its p95 is
the presentation jitter and its drift across a long run is lip sync.
`g2g-plugins/tests/av_lipsync_soak.rs` is the live harness that bounds both over a
real display and a real audio device.

A frame that misses its deadline looks the same whether the sink is presenting
early or the feed is arriving late, so the deadline error alone cannot name the
culprit. Under `G2G_DEBUG=WaylandSink` the sink prints one line a second holding
the elected clock, the frame PTS and the frame deadline as cumulative differences
against this host's monotonic clock, next to that frame's pacer wait and its age
since the source stamped it. An elected clock running at the wrong rate moves the
first, a presentation anchor moving under the stream separates the third from the
second, and a feed delivering slower than the media rate it stamps leaves all three
flat while the age decays toward zero, and only once it reaches zero does the media
line sag away from the clock line, which is where the late drops start.
`PipeWireSink` prints the master clock's own discipline, playout position, fitted
slope and window depth, once a second under `G2G_DEBUG=PipeWireSink`, which is the
same question asked at the source of the timeline.

## Upstream QoS

Upstream QoS carries that lateness back to the producer so it sheds load too, not
just the sink. It rides the same per-link reverse channel as `Reconfigure`: a sink
returns a `QosMessage` from `AsyncElement::take_qos`, the runner stores it into the
incoming link's reverse `QosSlot`, and the producer observes it as
`PushOutcome::Qos` on its next push. Reconfigure wins when both are pending, and
QoS is advisory and never holds the packet back. `SyncSink` originates it on a
late-drop and `VideoTestSrc` reacts by skipping about `jitter / frame_period`
frames, advancing PTS without generating them.

Relaying through a transform carries the report the rest of the way to the source
in a multi-element pipeline. A transform observes a downstream QoS as a
`PushOutcome::Qos` inside `process`, but that outcome is discarded by a generic
transform, and the runner rather than the element owns the reverse slots, so the
relay is runner-mediated: the runner wires the transform's output `SenderSink` with
a relay handle to its input link's `QosSlot` (`relay_qos_to`). When the output
adapter then sees a downstream QoS it stores it onto the input link instead of
surfacing it, so the upstream neighbour observes it on its next push, and across N
transforms the report walks one hop at a time back to the source. The element's
`process` is unaffected.

Acting on the report is the other half. An element that returns `true` from
`AsyncElement::handles_qos` is not relayed past, so it observes the report as
`PushOutcome::Qos` from its own `push` and sheds work itself, the same opt-out
`handles_keyframe_requests` and `handles_bitrate_requests` give an encoder.

`FfmpegVideoDec` does that under its `qos` property: a report arms a skip budget of
`jitter / frame_period` pictures, capped by `max-skip-frames`, during which the
codec context runs with `AVDISCARD_NONREF`, so libavcodec stops decoding the
pictures nothing references. Decode cost drops without touching a reference chain,
so every frame still emitted is bit-identical to a full decode, and the budget
counting down to zero is the recovery. It is wired in the bespoke
`run_source_transform_sink` runner and in the DAG runner (`run_graph` and
`run_linear_chain`, which the `WaylandSink` demo uses), so the sink's own load-shed
reaches the source through interior transforms such as an overlay or a convert.
