# g2g-dotnet

C# bindings for glass2glass over the [g2g-capi](../g2g-capi/README.md) C ABI. They cover launching a pipeline, the bus, editing a running pipeline, pushing buffers into `appsrc` and pulling them from `appsink`. Targets .NET 10.

## Build and run

```sh
cargo build --release -p g2g-capi
dotnet run --project g2g-dotnet/Example
```

`G2g.csproj` copies the native library from `target/release` into the output of every project that references it. Pass `-p:G2gNativeDir=<dir>/` to take it from another directory.

The library contains the elements of `g2g-plugins` built with only `std`. Add the elements of a `g2g-plugins` feature at build time:

```sh
cargo build --release -p g2g-capi --features g2g-plugins/rtsp
```

## Use

```csharp
using G2g;

using var sink = new AppSink("out");
using var pipeline = new Pipeline("videotestsrc num-buffers=30 ! appsink channel=out");
while (sink.Pull() is { } sample)
{
    using (sample)
    {
        Console.WriteLine($"{sample.Data.Length} bytes at {sample.PtsNs} ns");
    }
}
pipeline.Wait();
```

Create an `AppSrc` or `AppSink` before the pipeline that names its channel. A refused call throws `G2gException` with the reason from the C ABI, such as `UnknownNode`. `Pipeline.Wait` throws when the run ends in an error, and the error itself is on the bus. `Pipeline.Dispose` blocks until the pipeline reaches end of stream.

The C ABI's zero-copy `appsrc` lend and per-frame `appsink` callback have no C# wrapper yet.
