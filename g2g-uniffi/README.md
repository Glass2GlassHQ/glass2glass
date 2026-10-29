# g2g-uniffi

Kotlin and Swift bindings for glass2glass, generated with [UniFFI](https://mozilla.github.io/uniffi-rs/). They cover launching a pipeline, the bus, editing a running pipeline, pushing buffers into `appsrc` and pulling them from `appsink`. On Android they also show video in an app's `Surface`. The generated Kotlin is in package `dev.g2g` and loads the library through [JNA](https://github.com/java-native-access/jna).

## Desktop JVM

```sh
tools/kotlin-jvm-example.sh
```

This builds `libg2g_uniffi.so`, generates the bindings, and runs [`examples/jvm/Example.kt`](examples/jvm/Example.kt). It needs a JDK, and downloads kotlinc and JNA once into `~/.cache/g2g-kotlin`, checking their sha256.

To generate the bindings for your own project:

```sh
cargo build --release -p g2g-uniffi
cargo run --release -p g2g-uniffi --features bindgen --bin uniffi-bindgen -- \
    generate --library target/release/libg2g_uniffi.so --language kotlin --out-dir <dir>
```

## Swift

```sh
tools/swift-example.sh
```

This builds `libg2g_uniffi.so`, generates the Swift bindings, and runs [`examples/swift/main.swift`](examples/swift/main.swift). It needs `swiftc` on PATH. The bindings are `g2g_uniffi.swift` plus the C header and modulemap for the `g2g_uniffiFFI` module. Compile them with `-Xcc -fmodule-map-file=<dir>/g2g_uniffiFFI.modulemap -I <dir>` and link with `-lg2g_uniffi`. Byte arguments and `Sample.data` are `Data`, and a refused call throws `PipelineError.Edit(message:)`.

## Android

Build the library with `cargo ndk --platform 26 -t arm64-v8a build --release -p g2g-uniffi`. Package `libg2g_uniffi.so` with the generated Kotlin, [`android/Surfaces.kt`](android/Surfaces.kt), and JNA's Android archive (`net.java.dev.jna:jna:<version>@aar`). Add the elements of a `g2g-plugins` feature with `--features g2g-plugins/camera2` (or `mediacodec`, `aaudio`).

`surfacesink channel=<name>` draws RGBA frames into the `Surface` attached under that name and drops frames while none is attached. It copies frames in host memory. Built with `g2g-plugins/mediacodec-wgpu`, it also presents the GPU textures of `mediacodecdec gpu-output=true` with no copy. The first frame decides which kind a `Surface` takes until it is detached, and a frame of the other kind fails the run. Attach in `surfaceCreated` and detach in `surfaceDestroyed`. `detach` returns once the sink has stopped drawing. The frame is scaled to the view, so size the view to the frame's aspect ratio.

```kotlin
override fun surfaceCreated(holder: SurfaceHolder) = Surfaces.attach("preview", holder.surface)
override fun surfaceDestroyed(holder: SurfaceHolder) = Surfaces.detach("preview")

Thread {
    Pipeline("videotestsrc ! videoconvert ! surfacesink channel=preview").use { it.wait() }
}.start()
```

[`examples/android`](examples/android) is a complete app. `tools/android-kotlin-smoke.sh` builds it without Gradle (`tools/android-kotlin-apk.sh`), installs it on the connected device, and runs its test bars, camera and GPU decode modes, checking that each one reaches the screen. The device must be unlocked, and the back camera must face something lit.

## Behaviour

Create an `AppSrc` or `AppSink` before the pipeline that names its channel. A refused call throws `PipelineException.Edit` with the reason, such as `UnknownNode`. `Pipeline.wait` throws `PipelineException.Run` when the run ends in an error. `pull` and `wait` block, so call them off the main thread. `AppSink.pull` copies each frame into a `ByteArray`. `AppSrc.pushBuffer` takes a direct `ByteBuffer` and copies once, where `push` copies a `ByteArray` twice. The buffer's position must be 0, since the bindings send the bytes from its start up to `remaining()`.
