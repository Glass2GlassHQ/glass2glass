# g2g-uniffi

Kotlin bindings for glass2glass, generated with [UniFFI](https://mozilla.github.io/uniffi-rs/). They cover launching a pipeline, the bus, editing a running pipeline, pushing buffers into `appsrc` and pulling them from `appsink`. On Android they also show video in an app's `Surface`. The generated Kotlin is in package `dev.g2g` and loads the library through [JNA](https://github.com/java-native-access/jna).

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

## Android

Build the library with `cargo ndk --platform 26 -t arm64-v8a build --release -p g2g-uniffi`. Package `libg2g_uniffi.so` with the generated Kotlin, [`android/Surfaces.kt`](android/Surfaces.kt), and JNA's Android archive (`net.java.dev.jna:jna:<version>@aar`). Add the elements of a `g2g-plugins` feature with `--features g2g-plugins/camera2` (or `mediacodec`, `aaudio`).

`surfacesink channel=<name>` copies RGBA frames into the `Surface` attached under that name and drops frames while none is attached. Attach in `surfaceCreated` and detach in `surfaceDestroyed`. `detach` returns once the sink has stopped drawing. The frame is scaled to the view, so size the view to the frame's aspect ratio.

```kotlin
override fun surfaceCreated(holder: SurfaceHolder) = Surfaces.attach("preview", holder.surface)
override fun surfaceDestroyed(holder: SurfaceHolder) = Surfaces.detach("preview")

Thread {
    Pipeline("videotestsrc ! videoconvert ! surfacesink channel=preview").use { it.wait() }
}.start()
```

[`examples/android`](examples/android) is a complete app. `tools/android-kotlin-smoke.sh` builds it without Gradle (`tools/android-kotlin-apk.sh`), installs it on the connected device, and checks that the test bars reach the screen. The device must be unlocked.

## Behaviour

Create an `AppSrc` or `AppSink` before the pipeline that names its channel. A refused call throws `PipelineException.Edit` with the reason, such as `UnknownNode`. `Pipeline.wait` throws `PipelineException.Run` when the run ends in an error. `pull` and `wait` block, so call them off the main thread. `AppSink.pull` copies each frame into a `ByteArray`.
