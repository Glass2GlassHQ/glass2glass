# Properties, introspection and the launch DSL

The runtime face of an element: string-keyed properties, `gst-inspect`-style
introspection, the `gst-launch` text parser, declarative and scripted graphs,
dynamically loaded native plugins, and the hosted Python and Rhai element
shells. Part of the design in [README.md](README.md).

The typed `with_*` builders are the zero-cost construction path and the only one
the `no_std` and RTOS baseline needs. Tooling, meaning a text-pipeline parser, an
inspector or a GUI, needs a runtime face instead: set a property by string name,
read it back, enumerate what an element exposes. There are three layers, each
building on the last.

## The property bag

`g2g-core::property` (`no_std + alloc`) holds `PropValue` (`Bool`, `Int`, `Uint`,
`Double`, `Fraction`, `Str`), `PropKind`, a static `PropertySpec` of name, kind
and blurb, and `PropError`, plus `PropValue::parse(kind, "text")` for the
`key=value` syntax.

`AsyncElement` and `SourceLoop`, and their dyn mirrors, gain `properties()`,
`set_property()` and `get_property()`, all defaulting to no properties the same
zero-cost way `latency()` defaults to zero, so the baseline pays nothing and an
element opts in only by overriding them. This is the GObject-property analog. The
builders stay the type-checked path and this is the string-keyed one.

## By-name construction and introspection

`Registry` (`std`) registers a transform or sink under a name through
`LaunchFactory`, with a parameterless constructor and its pad templates, and
sources reuse the parameterless `SourceFactory`. `make_source` and `make_element`
build by name.

`inspect(name)` dumps an element's role, properties and pad templates, the
`gst-inspect` analog. The dump is GStreamer-shaped: a Factory Details header from
the element type's `metadata()` (`ElementMetadata { long_name, klass,
description, author }`, the `gst_element_class_set_static_metadata` analog and a
zero-cost opt-in like `properties()`), then pad templates, then an Element
Properties section where each `PropertySpec` carries its `default`, numeric
`range`, enum `values` and read/write `flags` alongside the blurb.
`element_listing()` is the no-arg index, one `name: Long-name` per element.

A factory can declare `with_experimental()` when its runtime is host-validated or
device-validated rather than a CI promise. The dump then includes
`Stability   experimental` and the listing suffixes `[experimental]`.

## The text parser

`runtime::parse_launch` (`std`) turns
`"videotestsrc num-buffers=3 ! videoflip method=rotate-180 ! fakesink"` into a
runnable `Graph`. Each `!`-separated stage is `element-name key=value ...`, the
element is built by name, each value is parsed for its property's `PropKind` and
applied, and the stages are linked source to transforms to sink. The result drops
straight onto `run_graph`, so a pipeline is expressible as text without
hand-written Rust.

A bare `media/type,field=value,...` stage is the inline caps-filter shorthand.
`parse_launch` rewrites it to a `capsfilter` whose `caps` property is parsed by
`capsfilter::parse_caps`, the `Caps` text grammar, so
`videotestsrc ! video/x-raw,format=nv12,width=320 ! ...` pins a format and
geometry as text.

Branching makes this a chain parser. `name=t` names an element and a `t.`
reference opens a branch, with `tee` the structural fan-out node broadcasting to
every branch, its width derived from the branch count. Roles follow
connectivity.

The tokenizer is quote-aware: a double-quoted value is one token, so whitespace
and `!` inside it are literal, as in `gstwrap element="x264enc bitrate=4000"` and
`filesrc location="/my file.ts"`. The surrounding quotes are stripped from the
value.

## ML elements by name

The stock registry is assembled in `g2g-plugins`, which does not depend on
`g2g-ml`, so an app that wants the ML elements in a launch line calls
`g2g_ml::register(&mut reg)` (`launch` feature) on the registry it built. That
adds `ortinfer` (`ort`), `wgpupreprocess` (`wgpu`) and `detectionpostprocess`
(`analytics`), each only when its feature builds the element, so
`... ! ortinfer model=yolov8n.onnx tensor-input=true ! detectionpostprocess
conf-threshold=0.3 ! ...` parses.

`OrtInference` is constructible without a model for this: the `model` property
loads the session through the same v1 contract check, `tensor-input` survives the
load either side of it, and until a model is loaded negotiation and `process`
fail with `NotConfigured`. `WgpuInference` stays out, because it is built from
weight tensors and shapes, which a text line cannot express.

## Declarative graph documents

A launch string is the ergonomic one-liner. A JSON or YAML document
(`g2g_plugins::declarative`, `declarative` and `declarative-yaml` features) is
the version-controllable, tool-generated, comment-carrying form.

A document is `nodes`, each `{ id, element, props }` or a `{ id, caps }`
capsfilter shorthand, plus `edges`, each `{ from, to }` with an optional
backpressure `policy` and `capacity`. It reaches the graph through exactly the
launch parser's machinery: roles follow link degree, so no inbound means source,
several inbound means a `MuxerFactory` muxer, and a fan-out node gets the
auto-tee spliced in. Every property value is typed by the target element's
`PropertySpec` and parsed with the same `PropValue::parse`, so a
`num-buffers: 30` in JSON means exactly what `num-buffers=30` does in a launch
string.

A top-level `pipeline:` string is an escape hatch that defers to `parse_launch`.
Both formats deserialize into one shared `GraphSpec`, a format-agnostic serde
model, and `build_spec` turns that into the runnable `Graph`.
`g2g-launch --graph <file>` runs one.

## Rhai graph-building scripts

Where a document describes a fixed graph, a script computes one: the shape can
depend on a loop, a parameter, or the environment, such as fanning N cameras into
a compositor or gating a branch on a flag. The script (`g2g_plugins::script`,
`script-rhai` feature) drives a small builder API (`add`, `caps`, `set`, `link`,
`link_leaky`) that accumulates into the same `GraphSpec`, so a script and a
document reach the graph through one builder and one set of role, caps and policy
rules.

Rhai is pure Rust with no C toolchain, so scripting reaches the browser
(`wasm32`, CI-guarded) and every other `std` target without compromising the
portability story, and its `sync` feature makes its values `Send`. It is a
`std`-tier capability: `script-rhai` implies `std` for Rhai's own `std` feature
and `std::fs` behind `location=`, so the bare-metal `no_std` and RTOS baseline
does not get scripting, by design, since an MCU builds a fixed graph in Rust.
`g2g-launch --script <file>` runs one. These are construction-time scripts, run
once to emit a graph. The per-frame `scriptelement` below is the runtime
complement.

## Animated properties

The layers above set a property once, at build time. A controller
(`g2g-core::controller`, `runtime` feature) makes it a function of stream time,
the `gst-controller` analog.

A `ControlSource` is a keyframed curve over `(pts_ns, value)` pairs, either
`Step`, holding each keyframe, or `Linear`, interpolating, clamped to its end
values outside the keyframe range. A `ControlProgram` binds curves to one node's
property names and attaches with `Graph::set_node_control(node, program)`, so a
`parse_launch` line's `name=` node can be animated through `Graph::node_by_name`.

When the run starts, before negotiation and before any frame flows, each program
is resolved against its element's own `PropertySpec` table. An unknown name, a
kind with no number to animate (`Fraction`, `Str`, `Flags`), an empty curve, or a
node whose arm has no per-frame hook, since a source drives itself and a tee
carries no element, fails the run with `G2gError::ControlBinding` rather than
animating nothing.

At runtime the arm that owns the element samples every binding at each
`DataFrame`'s PTS and sets it before handing that frame over, so a frame is
always processed under the values its own timestamp calls for. The sample is
rounded and clamped into the property's kind, so a negative sample cannot wrap a
`Uint`, and a value the element refuses fails the run loud. Transform, sink and
fan-in nodes carry controllers, under both the cooperative and the
thread-per-arm runner, since a resolved controller is owned data and rides the
arm's builder closure onto its thread.

Two deliberate limits: a zero-order-hold `Tick` frame samples nothing, because
the held frame's advanced timestamp lives inside the element rather than in the
runner, and samples use the raw PTS, not segment-mapped running time.

## Dynamic plugin loading

Beyond build-time registration, meaning a crate that calls
`Registry::register_*` and the primary extension path, a third party can ship a
native element as a dynamically loaded `.so`, the analog of GStreamer's scanned
plugin path.

They build a `cdylib` against the published `g2g-core` plus the `g2g-plugin` SDK
and use its `declare_plugin! { elements: [ (name, Type, build) ] }` macro, which
emits two C-ABI entry points: `g2g_plugin_abi`, returning the ABI tag, and
`g2g_plugin_register(&mut Registry)`, registering the elements with its body in
`catch_unwind` because unwinding across `extern "C"` is undefined behaviour. A
host built with the `plugin-loader` feature (`g2g_plugins::plugin_loader`, over
`libloading`) `dlopen`s the object, reads its tag, and registers it only on an
exact match. `g2g-launch` and `g2g-inspect` expose this via `--plugin <path>` and
`$G2G_PLUGIN_PATH`.

The hard constraint is that Rust has no stable ABI, so a plugin and host must
share the same `g2g-core` version, the same `rustc`, and the same
layout-affecting features. Two features change in-memory layout across the
boundary: `metadata` resizes `Frame` through the `FrameMetaSet` side-channel, and
`multi-thread` changes the `Send` bound on the boxed element trait objects.
`g2g_core::ABI_VERSION`, a `build.rs`-computed string folding version, `rustc`
and those features, is embedded in each plugin and checked by the loader, which
refuses a mismatch with a clear `AbiMismatch` error rather than risk passing a
differently-laid-out `Frame` or trait object across the boundary.

Each loaded `libloading::Library` is held for the life of the process, because
the registered factories are `fn` pointers into its mapped code and dropping it
would be a use-after-free with no back-pointer to catch it. The whole path is
exercised out-of-tree by `g2g-plugins/tests/fixtures/example-plugin` and
`tests/plugin_loader_dlopen.rs`.

### Plugin ABI v2, the cross-toolchain tier

The version lock above is the price of passing Rust types across `dlopen`. v2 is
the other trade: a frozen `repr(C)` boundary (`g2g-plugin::abi`, header
`g2g-plugin/include/g2g_plugin_v2.h`) that carries a smaller surface but loads
into a host built by a different compiler, and can be written in C.

The model is GStreamer's `gst_plugin_desc`, a versioned descriptor plus vtables,
hand-rolled rather than taken from `abi_stable`, which is dormant, or `stabby`,
which leaks a heap vtable registry on stable Rust. `async-ffi` supplies the one
thing a hand-rolled C ABI cannot express, an FFI-safe `Future` (`FfiPoll`,
`FfiContext`, a three-pointer future struct), so `process` stays
backpressure-aware across the boundary.

The descriptor is data, not code. A v2 plugin exports one data symbol,
`g2g_plugin_v2_descriptor`, holding a magic, an ABI generation, and the list of
element names and kinds it will register. The host reads and validates it with
`dlsym` before calling any plugin function, which is what makes the capability
gate meaningful: `load_plugin_with_policy` hands the declaration to a
caller-supplied policy before the plugin gets control, and the default policy
refuses a declaration carrying a capability kind this host does not understand.
The declaration is then binding. The registrar stages elements rather than
writing them into the `Registry`, checks each against the declaration, and
commits only if every one matched, so a plugin that registers three declared
elements and one undeclared one contributes nothing.

What crosses: `configure_pipeline`, `configure_output`, `process`,
`set_property`, `get_property`, `destroy`, plus a `create` on the registration.
Caps cross as a `repr(C)` tagged union over a frozen numeric code table, since
the host's caps enums are `#[non_exhaustive]` and their discriminants can never
be an ABI, and property values likewise. Frames cross as pointer plus length plus
an owner-side `free`, which maps exactly onto `SystemSlice::from_foreign`, so a
frame moves in either direction without a copy.

What does not cross: v2 elements are System memory only. The wrapper narrows
`input_domains` to `System`, so a GPU-resident producer upstream gets a domain
converter spliced in rather than a frame the plugin cannot read. GPU domains, and
the roughly 50 exotic `AsyncElement` hooks covering clock election, QoS, metadata
propagation, the allocation cascade and the reverse-channel signals, stay v1 and
host-native, with the host-side wrapper element answering them with the trait
defaults. The flag-set property kind and the tensor, KLV, closed-caption and
sub-picture caps kinds do not cross either, and a registration that names one is
refused rather than approximated.

Growing it uses two mechanisms. `abi_version` gates the whole surface, so a
semantic change to an existing field bumps it. Inside one generation, every
versioned struct carries its own `struct_size` and the host reads
`min(plugin, host)` bytes into a zeroed local, so an older plugin's shorter
vtable leaves the host's newer entries absent and the host uses its defaults, and
trailing reserved fn-pointer slots let a future entry appear without the size
changing, which an older host ignores.

The loader probes the v2 symbol first and falls back to the v1 pair, so existing
v1 plugins load unchanged. v1 remains the path for a plugin that needs the whole
trait surface or GPU memory and ships alongside the host build it was compiled
against.

`LaunchFactory` builds an element from a context-free `fn()` pointer, and a v2
element's constructor needs to know which plugin vtable it belongs to. The host
therefore keeps a fixed table of 64 const-generic trampolines
(`MAX_V2_ELEMENT_SLOTS`), and past that a load is refused rather than silently
dropping an element. Slots are never freed, matching the loaded-forever library.

### Detached signatures

A host built with the `plugin-signing` feature can be handed a set of trusted
Ed25519 public keys, from `$G2G_PLUGIN_TRUSTED_KEYS` (a `:`-separated list of key
files), `g2g-inspect --trusted-key`, or `TrustedKeys` in code. An empty set is
the default and means no verification, so signed and unsigned plugins load
exactly as before. With one key or more, every plugin the host loads must carry a
sibling `<plugin>.sig` verifying under one of them, checked before `dlopen`, so a
refusal happens with none of the plugin's code run, initialisers included.

The `.sig` is 103 fixed bytes: magic, format version, the signer's 32-byte public
key, and the 64-byte signature over the plugin file. The signer's key is present
so one directory can hold plugins from several signers and an error can name the
offending one. That key is a selector, not a credential, since it must already be
in the trust set for its signature to be checked. `ring` supplies the Ed25519,
the same implementation the TLS stack already links. `g2g-plugin-sign` does
keygen, sign and verify, writing private keys 0600.

Verifying bytes read from a path and then handing `dlopen` the same path would
leave a window to swap the file. On Linux the verified bytes are written to a
`memfd_create` object, sealed against shrink, grow, write and further seals, and
loaded through `/proc/self/fd/N`, so what runs is what was checked. A plugin's
`$ORIGIN` rpath then resolves against `/proc/self` rather than its directory,
which is the price of that closure and applies only on the verified path. Other
platforms verify and then open the path, with the window left open and
documented. A trust set that cannot be read, meaning a missing or malformed key
file, or keys configured in a build without `plugin-signing`, is an error, never
a silently empty set.

### Security posture of the loader

The loader defends against a malformed plugin, not a malicious one, and the
difference is worth stating plainly. `dlopen` runs the library's initialisers
before the loader reads a single field, and a loaded plugin shares the host's
address space with no boundary at all: it can make any syscall the host can, read
the host's memory, and ignore every rule in the ABI.

A signature proves the bytes came from a holder of a trusted key and that they
did not change afterwards. It says nothing about what those bytes do, so a signed
malicious plugin is still malicious, and a trusted key that leaks signs anything.
The capability gate decides whether to load a file and what it may register, and
cannot constrain what loaded code does. It is policy, not sandboxing. Anything
stronger, a separate process or seccomp, is out of scope and deliberately has no
half-built stubs.

What the loader does do is treat every byte reachable from the descriptor as
untrusted input, on the same rules as a bitstream parser: bound every count
before using it as a length, null-check before dereferencing, UTF-8 check before
a byte range becomes a `str`, restrict element and property names to a
`gst-launch`-safe character set, and refuse any unknown discriminant instead of
reinterpreting it. Two things it cannot check and takes on the plugin's contract:
that a pointer and length pair really addresses that many readable bytes, and
that a `struct_size` really matches what the plugin wrote. The wrapper also
asserts `Send` for a plugin instance under a documented contract, that the runner
owns an element exclusively but may move it between threads, so a thread-affine
plugin is outside the ABI.

The path is exercised by `g2g-plugins/tests/plugin_loader_v2.rs`, a Rust plugin
built with a deliberately mismatched `g2g-core` feature set that v1 refuses and
v2 does not care about, `tests/plugin_c_abi.rs`, a plugin written in C and
compiled against the hand-written header including a `sizeof` comparison of every
ABI struct against its Rust type so the two cannot drift, and
`tests/m1061_plugin_signing.rs`, the same fixture signed, unsigned, signed by an
untrusted key, and modified after signing, each case checking that the element
never reached the registry.

## Hosted Python elements

`pyelement`, `pysrc` and `pyaggregator` (`g2g-python`) run a gst-python-ml
element shell as a first-class g2g element. `g2g-python` embeds CPython (pyo3,
`auto-initialize`), exposes a native `g2g` module the `backend/g2g` package
imports, and negotiates as a same-format passthrough.

Each hosted instance owns a dedicated GIL-holding OS thread. The element hands it
the frame and awaits the reply over a Waker channel, so the cooperative executor
keeps polling other arms while Python runs. A frame reaches Python without a copy
on either of two paths.

**System memory.** `g2g_process(buf, width, height, fmt, meta)` gets a writable
buffer-protocol object over the frame's own bytes, so `memoryview` and numpy read
and overwrite pixels in place. The host counts outstanding buffer exports and
fails the frame if the element kept a view past return, since its pointer would
dangle once the frame is freed downstream. `g2g_process_batch` and `g2g_produce`
are the aggregator and source shapes of the same contract.

**Payloads with no picture shape.** Audio into a transcriber, or text into
speech: the frame reaches `g2g_process_payload(buffers, caps, meta)`, and the
element hands back buffers of its own through
`meta.emit(payload, duration_ns=None, pts_ns=None)` instead of overwriting the
one it read. Each emitted buffer inherits the anchor's timing unless it says
otherwise. A streaming element gives every chunk its own `pts_ns`, usually the
previous chunk's pts plus its duration, so the chunks play one after another,
while outputs that run in parallel, such as the separation family's stems, leave
it unset and share the anchor's. `g2g.PTS_NONE`, which is
`FrameTiming::PTS_NONE`, emits a buffer with no presentation time, which a sink
presents on arrival.

### CUDA device memory

A `MemoryDomain::Cuda` frame has no CPU bytes, so its two semi-planar planes are
described to `g2g_process_cuda(luma, chroma, width, height, meta)` as
`g2g.CudaPlane` objects exposing `__cuda_array_interface__` v3: luma
`(height, width)` and interleaved chroma `(height/2, width/2, 2)`, byte strides
carrying the producer's row pitch so a pitch that is not the width is described
rather than repacked, `|u1` for NV12 and `<u2` for P010, and `stream: None`,
since the CUDA domain carries no stream and a producer hands the frame over once
the decode into it completed. `cupy.asarray(luma)` then aliases the decoder's
surface with no PCIe round-trip.

The `data` flag is read-only: the device memory belongs to the producer and a
teed frame shares it under a read-only guarantee, with no copy-on-write to fall
back on as the System path has. cupy treats the flag as advisory and aliases
anyway. torch's CAI importer refuses a read-only export, so a torch consumer
takes the plane's DLPack export instead, which carries the same read-only bit and
torch accepts. The flag stays set rather than being cleared to widen torch's CAI
path, because the surface is the producer's and the flag is what says so.

CAI carries no CUDA context, so the pointers are valid only in the context the
producer decoded into, exposed as the plane's `cuda_context` property for a
consumer that must push it, and cupy and torch use the device's primary context.
Plane lifetime is the call, enforced by a refcount check after it: a retained
plane, including one a cupy array holds as its base or a consumed DLPack tensor
holds as its manager context, fails the frame. An element that defines no hook
for its shape gets `UnsupportedDomain` for a GPU frame rather than a silent
readback, because `g2g-python` links no CUDA, so a CPU-only element needs an
explicit `cudadownload` upstream.

A GPU batch reaches a hosted aggregator as
`g2g_process_cuda_batch(planes, width, height, meta)`, one `(luma, chroma)` pair
per contributing input, so a batched detector reads every stream's decoded
surface in place, and the anchor flows on device-resident. A hosted source runs
the handoff backwards: `g2g_produce_cuda(width, height, meta)` returns the two
planes as any CAI-exporting objects, a cupy or torch allocation, or `None` for
end of stream, because this crate links no CUDA and cannot allocate device memory
itself. The returned planes are validated against the negotiated caps for shape,
sample type, packed-within-each-row layout, only the row pitch free, and a
non-null pointer, before they become a frame, and the frame's keep-alive holds
the Python objects so the memory outlives it. The source stamps timing and
reports its `cuda_context` through an optional attribute for a downstream
consumer that must push it.

The same plane also answers `__dlpack__` and `__dlpack_device__`, for the
frameworks that prefer it (`torch.from_dlpack`, `cupy.from_dlpack`). DLPack
carries a device and stream contract CAI does not: the device is `(kDLCUDA, 0)`,
since the CUDA domain carries the producing context but no device ordinal. A
consumer asking for 1.0 or newer through `max_version` gets a
`DLManagedTensorVersioned` capsule with the read-only flag set, and one asking
for nothing gets the pre-1.0 `DLManagedTensor`. `copy=True` or another
`dl_device` is refused rather than silently ignored, and `stream` is ignored
because the domain carries no stream. DLPack strides count elements rather than
bytes, so a row pitch that is not a whole number of samples is refused instead of
rounded. The capsule's destructor frees the tensor only while the capsule still
carries the unconsumed name, since a consumer that takes ownership renames it and
calls the deleter itself.

### One domain on both pads

The frame is read where it lies and forwarded untouched, so a hosted transform
carries one memory domain on both pads: System, or CUDA under `cuda-frames=true`,
the property that says the hosted class reads device memory. Declaring the same
domain on input and output keeps the relation honest, since the domain a frame
leaves in is the one it arrived in.

Two things follow. The domain-converter auto-plug splices a download or upload on
the edge into the element when upstream cannot deliver what the hosted code
reads, and never after it. And `propose_allocation` names that domain upstream,
so a multi-domain producer, an NVDEC that can keep frames on the device or
download them, settles on it and no converter node is needed at all. The proposal
constrains only the domain and the frame size, since the element allocates
nothing itself. `format` is a property too, so a launch-built `pyelement` can
accept the NV12 a decoder emits rather than only its RGBA default.

### Free-threading

One worker thread per element is the free-threading unit, and that is measured
rather than assumed, by the ignored `m988_gil_offload` test whose module docs
carry the invocation for each interpreter. Four hosted elements each running one
compute-bound pure-Python callback recover 3.6x of the ideal 4x on free-threaded
CPython 3.14, where `sys._is_gil_enabled()` is false in-process, and 0.9x on
stock 3.14, with no code change between the two. pyo3 picks the interpreter up at
build time through `PYO3_PYTHON`, free-threaded rules out `abi3`, and the whole
crate's test suite passes on both.

The native `g2g` module declares `gil_used = false`, which is load-bearing rather
than decorative: CPython re-enables the GIL process-wide when it imports a module
that has not declared it, so without the declaration the `import g2g` inside a
hosted element drops the same measurement back to 0.9x.

The sizing consequence on a stock interpreter is that N hosted elements do not
overlap, so a chain's Python cost is the sum of their per-frame times rather than
the slowest one, and `link_capacity` on those links has to absorb the wait while
the other elements hold the GIL. The `g2g-python` `host` module docs carry this
next to the worker design it follows from.

## Runtime scripting

The construction scripts above run once to emit a graph. `scriptelement`
(`script-rhai` feature) is the per-frame complement: a raw-video transform whose
`process(frame)` is a Rhai function, the pure-Rust cousin of the `pyelement`
CPython host. It negotiates as a same-format passthrough under the
`DerivedOutput` constraint, like `pyelement`.

On each `System`-memory frame it hands the script a zero-copy handle
(`FrameBuf`). The script indexes the live buffer in place
(`frame[i] = 255 - frame[i]`) and reads `frame.width`, `.height`, `.format`,
`.pts`, `.sequence` and `.len`, with no bulk copy in or out. The copy-free path
is a custom-type receiver rather than a byte blob because Rhai clones a value
argument on entry, so a blob argument is copied regardless, while a custom type
is passed by reference. The handle reaches the buffer through an atomic guard of
pointer plus length, armed for the call and nulled the instant it returns, so it
is `Send` and `Sync` with no `unsafe impl`, and a handle kept past the call reads
and writes nothing, a clean error, instead of dereferencing freed memory.

Per-pixel `frame[i]` is interpreted, which is fine for logic, metadata and small
regions, so whole-frame work goes through native bulk methods (`invert`, `fill`,
`apply_lut`) the script calls once and Rust loops at native speed, the control
plane and data plane split. Rhai is synchronous pure Rust, so the call runs inline
on the pipeline thread, with no GIL and hence none of the worker-thread isolation
the Python host needs. The compiled `Engine`, `AST` and `Scope` are held on the
element and are `Send` under rhai's `sync` feature, so it runs under the
multi-thread runner too. It is registered by name, so
`scriptelement script=... ! ...` parses in a launch line or a declarative
document. A GPU-resident frame yields `UnsupportedDomain`, since a script cannot
touch device memory.

`scriptrouter` is the fan-out sibling, a Rhai-scripted routing demux, a
`MultiOutputElement` registered via `register_demux` so
`scriptrouter name=r r.0 ! ... r.1 ! ...` builds a 1-to-N node. Its
`route(frame)` returns the output port each `DataFrame` goes to: a single index,
where negative means drop, or an array of indices to multicast one frame to
several ports at once. A multicast is a shared duplicate per port via
`Frame::share`, the same fan-out primitive a broadcast tee uses, so the buffer
refcounts where the memory domain allows and deep-copies owned CPU bytes, which
makes the cost honest. Control packets broadcast to every branch and the runner
broadcasts EOS, exactly like the built-in `Router` that it is the scripted analog
of.

It is the route-buffers-into-my-own-pipeline seam: an `appsink channel=...` on
each output pad turns each route into a separate consumer the app `pull()`s live
while the pipeline runs, with the control plane in the script, buffers moved
natively, and no interpreter on the data path. See the
`scriptrouter_appsink_egress` example. The `route` handle is read-only and
media-agnostic, routing audio, video and byte streams by `pts`, `sequence`,
`keyframe` and `len`, with a `frame[i]` byte peek for content routing, reusing
the `scriptelement` `FrameGuard`. Rhai is a sandboxed interpreter with no I/O or
FFI, so buffer egress to an external system stays the host's job, an `appsink`
plus a binding or a native callback. The script decides routing, it does not
perform the handoff.
