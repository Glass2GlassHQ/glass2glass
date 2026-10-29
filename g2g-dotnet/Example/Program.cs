using G2g;

const int FrameBytes = 2 * 2 * 4;
const int FrameCount = 3;
const ulong FramePeriodNs = 33_333_333;
const int EditedFrameCount = 90;

using (var source = new AppSrc("in"))
using (var sink = new AppSink("out"))
using (var pipeline = new Pipeline(
    "appsrc channel=in caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1 ! appsink channel=out"))
{
    var frame = new byte[FrameBytes];
    for (var index = 0; index < FrameCount; index++)
    {
        Array.Fill(frame, (byte)(index + 1));
        while (!source.TryPush(frame, (ulong)index * FramePeriodNs))
        {
            Thread.Yield();
        }
    }
    source.EndOfStream();

    var pulled = 0;
    while (sink.Pull() is { } sample)
    {
        using (sample)
        {
            Console.WriteLine($"frame {pulled}: {sample.Data.Length} bytes, first=0x{sample.Data[0]:X2}, pts={sample.PtsNs} ns");
            Expect(sample.Data[0] == pulled + 1, $"frame {pulled} carries the bytes pushed for it");
        }
        pulled++;
    }

    while (pipeline.TryPollBus(out var message))
    {
        Console.WriteLine($"bus: {message.Kind} {message.Text}");
    }

    var stats = pipeline.Wait();
    Console.WriteLine($"done: emitted={stats.FramesEmitted} consumed={stats.FramesConsumed} dropped={stats.FramesDropped}");
    Expect(pulled == FrameCount, $"pulled {pulled} of {FrameCount} frames");
}

using (var pipeline = new Pipeline(
    $"videotestsrc name=src num-buffers={EditedFrameCount} ! clocksync ! tee name=t ! fakesink"))
{
    var valve = pipeline.InsertAfter("src", "valve drop=false");
    pipeline.SetProperty(valve, "drop-mode", "forward-sticky-events");
    Expect(pipeline.GetProperty(valve, "drop-mode") == "forward-sticky-events", "drop-mode reads back");
    Expect(pipeline.GetProperty(valve, "no-such-property") is null, "an unknown property reads as null");
    pipeline.Remove(valve);

    var branch = pipeline.AddBranch("t", "identity ! fakesink");
    Expect(branch.Length == 2, $"the branch has two names, got {string.Join(", ", branch)}");
    pipeline.RemoveBranch(branch[1]);

    try
    {
        pipeline.InsertAfter("no-such-node", "valve");
        Expect(false, "inserting after an unknown node is refused");
    }
    catch (G2gException refused) when (refused.Message.Contains("UnknownNode"))
    {
        Console.WriteLine($"refused: {refused.Message}");
    }

    var stats = pipeline.Wait();
    Console.WriteLine($"edited run: consumed={stats.FramesConsumed}");
    Expect(stats.FramesConsumed == EditedFrameCount, "no frame is lost to the edits");
}

static void Expect(bool condition, string what)
{
    if (!condition)
    {
        throw new InvalidOperationException($"expected: {what}");
    }
}
