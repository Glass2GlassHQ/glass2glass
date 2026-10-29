using System.Runtime.InteropServices;

namespace G2g;

internal sealed class PipelineHandle() : SafeHandle(0, ownsHandle: true)
{
    public override bool IsInvalid => handle == 0;

    protected override bool ReleaseHandle()
    {
        Native.g2g_pipeline_free(handle);
        return true;
    }
}

internal sealed class AppSrcHandle() : SafeHandle(0, ownsHandle: true)
{
    public override bool IsInvalid => handle == 0;

    protected override bool ReleaseHandle()
    {
        Native.g2g_appsrc_free(handle);
        return true;
    }
}

internal sealed class AppSinkHandle() : SafeHandle(0, ownsHandle: true)
{
    public override bool IsInvalid => handle == 0;

    protected override bool ReleaseHandle()
    {
        Native.g2g_appsink_free(handle);
        return true;
    }
}

internal sealed class SampleHandle() : SafeHandle(0, ownsHandle: true)
{
    public override bool IsInvalid => handle == 0;

    protected override bool ReleaseHandle()
    {
        Native.g2g_sample_free(handle);
        return true;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal struct NativeBusMessage
{
    public int Kind;
    public nint Text;
    public ulong A;
    public ulong B;
}

internal static unsafe partial class Native
{
    private const string Library = "g2g_capi";

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial PipelineHandle g2g_pipeline_launch(string description, out nint error);

    [LibraryImport(Library)]
    internal static partial int g2g_pipeline_bus_poll(PipelineHandle pipeline, out NativeBusMessage message);

    [LibraryImport(Library)]
    internal static partial int g2g_pipeline_is_done(PipelineHandle pipeline);

    [LibraryImport(Library)]
    internal static partial int g2g_pipeline_wait(PipelineHandle pipeline, out Stats stats);

    [LibraryImport(Library)]
    internal static partial void g2g_pipeline_free(nint pipeline);

    [LibraryImport(Library)]
    internal static partial void g2g_string_free(nint text);

    [LibraryImport(Library)]
    internal static partial void g2g_strv_free(nint names);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_insert_after(PipelineHandle pipeline, string after, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_insert_before(PipelineHandle pipeline, string before, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_insert_chain_after(PipelineHandle pipeline, string after, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_insert_chain_before(PipelineHandle pipeline, string before, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_add_branch(PipelineHandle pipeline, string tee, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int g2g_pipeline_remove_branch(PipelineHandle pipeline, string sink, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_add_input(PipelineHandle pipeline, string muxer, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int g2g_pipeline_remove(PipelineHandle pipeline, string node, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_replace_sink(PipelineHandle pipeline, string node, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_replace_source(PipelineHandle pipeline, string node, string description, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial int g2g_pipeline_set_property(PipelineHandle pipeline, string node, string name, string value, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial nint g2g_pipeline_get_property(PipelineHandle pipeline, string node, string name, out nint error);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial AppSrcHandle g2g_appsrc_new(string channel);

    [LibraryImport(Library)]
    internal static partial int g2g_appsrc_push(AppSrcHandle source, byte* data, nuint length, ulong ptsNs);

    [LibraryImport(Library)]
    internal static partial int g2g_appsrc_end_of_stream(AppSrcHandle source);

    [LibraryImport(Library)]
    internal static partial void g2g_appsrc_free(nint source);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    internal static partial AppSinkHandle g2g_appsink_new(string channel);

    [LibraryImport(Library)]
    internal static partial int g2g_appsink_pull(AppSinkHandle sink, out SampleHandle sample);

    [LibraryImport(Library)]
    internal static partial void g2g_appsink_free(nint sink);

    [LibraryImport(Library)]
    internal static partial byte* g2g_sample_data(SampleHandle sample);

    [LibraryImport(Library)]
    internal static partial nuint g2g_sample_len(SampleHandle sample);

    [LibraryImport(Library)]
    internal static partial ulong g2g_sample_pts(SampleHandle sample);

    [LibraryImport(Library)]
    internal static partial void g2g_sample_free(nint sample);

    internal static string? TakeString(nint text)
    {
        if (text == 0)
        {
            return null;
        }
        var owned = Marshal.PtrToStringUTF8(text);
        g2g_string_free(text);
        return owned;
    }

    internal static string[] TakeNames(nint names)
    {
        var owned = new List<string>();
        for (var entry = (nint*)names; *entry != 0; entry++)
        {
            owned.Add(Marshal.PtrToStringUTF8(*entry)!);
        }
        g2g_strv_free(names);
        return owned.ToArray();
    }
}
