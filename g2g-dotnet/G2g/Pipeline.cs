using System.Runtime.InteropServices;

namespace G2g;

public sealed class G2gException(string message) : Exception(message);

public enum BusMessageKind
{
    StreamStart = 0,
    Eos = 1,
    Error = 2,
    Warning = 3,
    Info = 4,
    StateChanged = 5,
    Buffering = 6,
    DurationChanged = 7,
    Other = 99,
}

// A and B mean what g2g.h documents for G2gBusMessage of the same kind.
public readonly record struct BusMessage(BusMessageKind Kind, string? Text, ulong A, ulong B);

[StructLayout(LayoutKind.Sequential)]
public readonly record struct Stats(ulong FramesEmitted, ulong FramesConsumed, ulong FramesDropped);

// Calls on one pipeline must come from one thread at a time.
public sealed class Pipeline : IDisposable
{
    private readonly PipelineHandle handle;

    public Pipeline(string description)
    {
        handle = Native.g2g_pipeline_launch(description, out var error);
        if (handle.IsInvalid)
        {
            throw new G2gException(Native.TakeString(error) ?? $"cannot launch {description}");
        }
    }

    public bool TryPollBus(out BusMessage message)
    {
        if (Native.g2g_pipeline_bus_poll(handle, out var raw) == 0)
        {
            message = default;
            return false;
        }
        var text = raw.Text == 0 ? null : Marshal.PtrToStringUTF8(raw.Text);
        message = new BusMessage((BusMessageKind)raw.Kind, text, raw.A, raw.B);
        return true;
    }

    public bool IsDone => Native.g2g_pipeline_is_done(handle) == 1;

    // The error itself is posted on the bus.
    public Stats Wait()
    {
        if (Native.g2g_pipeline_wait(handle, out var stats) != 0)
        {
            throw new G2gException("pipeline ended with an error");
        }
        return stats;
    }

    public string InsertAfter(string node, string description) =>
        TakeName(Native.g2g_pipeline_insert_after(handle, node, description, out var error), error);

    public string InsertBefore(string node, string description) =>
        TakeName(Native.g2g_pipeline_insert_before(handle, node, description, out var error), error);

    public string[] InsertChainAfter(string node, string description) =>
        TakeNames(Native.g2g_pipeline_insert_chain_after(handle, node, description, out var error), error);

    public string[] InsertChainBefore(string node, string description) =>
        TakeNames(Native.g2g_pipeline_insert_chain_before(handle, node, description, out var error), error);

    public void Remove(string node) =>
        ThrowIfRefused(Native.g2g_pipeline_remove(handle, node, out var error), error);

    public string[] AddBranch(string tee, string description) =>
        TakeNames(Native.g2g_pipeline_add_branch(handle, tee, description, out var error), error);

    public void RemoveBranch(string sink) =>
        ThrowIfRefused(Native.g2g_pipeline_remove_branch(handle, sink, out var error), error);

    public string AddInput(string muxer, string description) =>
        TakeName(Native.g2g_pipeline_add_input(handle, muxer, description, out var error), error);

    public string ReplaceSink(string node, string description) =>
        TakeName(Native.g2g_pipeline_replace_sink(handle, node, description, out var error), error);

    public string ReplaceSource(string node, string description) =>
        TakeName(Native.g2g_pipeline_replace_source(handle, node, description, out var error), error);

    public void SetProperty(string node, string name, string value) =>
        ThrowIfRefused(Native.g2g_pipeline_set_property(handle, node, name, value, out var error), error);

    // Null when the element has no such property.
    public string? GetProperty(string node, string name)
    {
        var value = Native.g2g_pipeline_get_property(handle, node, name, out var error);
        if (error != 0)
        {
            throw new G2gException(Native.TakeString(error)!);
        }
        return Native.TakeString(value);
    }

    // Blocks until the pipeline reaches end of stream.
    public void Dispose() => handle.Dispose();

    private static string TakeName(nint name, nint error)
    {
        if (name == 0)
        {
            throw new G2gException(Native.TakeString(error)!);
        }
        return Native.TakeString(name)!;
    }

    private static string[] TakeNames(nint names, nint error)
    {
        if (names == 0)
        {
            throw new G2gException(Native.TakeString(error)!);
        }
        return Native.TakeNames(names);
    }

    private static void ThrowIfRefused(int succeeded, nint error)
    {
        if (succeeded == 0)
        {
            throw new G2gException(Native.TakeString(error)!);
        }
    }
}
