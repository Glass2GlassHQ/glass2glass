namespace G2g;

// Create it before the pipeline that names its channel.
public sealed class AppSrc(string channel = "default") : IDisposable
{
    private readonly AppSrcHandle handle = Native.g2g_appsrc_new(channel);

    // False when the feed is full (retry) or the pipeline is gone.
    public unsafe bool TryPush(ReadOnlySpan<byte> data, ulong ptsNs)
    {
        fixed (byte* bytes = data)
        {
            return Native.g2g_appsrc_push(handle, bytes, (nuint)data.Length, ptsNs) == 1;
        }
    }

    public bool EndOfStream() => Native.g2g_appsrc_end_of_stream(handle) == 1;

    public void Dispose() => handle.Dispose();
}

// Create it before the pipeline that names its channel.
public sealed class AppSink(string channel = "default") : IDisposable
{
    private readonly AppSinkHandle handle = Native.g2g_appsink_new(channel);

    // Blocks for the next sample, null once the stream has ended.
    public Sample? Pull()
    {
        if (Native.g2g_appsink_pull(handle, out var sample) == 0)
        {
            sample.Dispose();
            return null;
        }
        return new Sample(sample);
    }

    public void Dispose() => handle.Dispose();
}

// Data points into the pipeline's buffer and is invalid after Dispose.
public sealed class Sample : IDisposable
{
    private readonly SampleHandle handle;

    internal Sample(SampleHandle handle) => this.handle = handle;

    public unsafe ReadOnlySpan<byte> Data =>
        new(Native.g2g_sample_data(handle), checked((int)Native.g2g_sample_len(handle)));

    public ulong PtsNs => Native.g2g_sample_pts(handle);

    public void Dispose() => handle.Dispose();
}
