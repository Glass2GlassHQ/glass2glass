import Foundation

let frameBytes = 2 * 2 * 4
let frameCount = 3
let framePeriodNs: UInt64 = 33_333_333
let editedFrameCount: UInt64 = 90

func expect(_ condition: Bool, _ what: String) {
    if !condition {
        FileHandle.standardError.write(Data("expected: \(what)\n".utf8))
        exit(1)
    }
}

func roundTrip() throws {
    let source = AppSrc(channel: "in")
    let sink = AppSink(channel: "out")
    let pipeline = try Pipeline(
        description: "appsrc channel=in caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1 ! appsink channel=out"
    )
    for index in 0..<frameCount {
        let fill = UInt8(index + 1)
        let pts = UInt64(index) * framePeriodNs
        let frame = Data(repeating: fill, count: frameBytes)
        if index % 2 == 0 {
            while !source.push(data: frame, ptsNs: pts) {
                sched_yield()
            }
        } else {
            while !source.pushBuffer(data: frame, ptsNs: pts) {
                sched_yield()
            }
        }
    }
    _ = source.endOfStream()

    var pulled = 0
    while let sample = try sink.pull() {
        print("frame \(pulled): \(sample.data.count) bytes, first=\(sample.data[0]), pts=\(sample.ptsNs) ns")
        expect(sample.data.allSatisfy { $0 == UInt8(pulled + 1) }, "frame \(pulled) carries the bytes pushed for it")
        pulled += 1
    }

    while let message = pipeline.pollBus() {
        print("bus: \(message.kind) \(message.text ?? "")")
    }

    let stats = try pipeline.wait()
    print("done: emitted=\(stats.framesEmitted) consumed=\(stats.framesConsumed) dropped=\(stats.framesDropped)")
    expect(pulled == frameCount, "pulled \(pulled) of \(frameCount) frames")
}

func edits() throws {
    let pipeline = try Pipeline(
        description: "videotestsrc name=src num-buffers=\(editedFrameCount) ! clocksync ! tee name=t ! fakesink"
    )
    let valve = try pipeline.insertAfter(node: "src", description: "valve drop=false")
    try pipeline.setProperty(node: valve, name: "drop-mode", value: "forward-sticky-events")
    expect(try pipeline.getProperty(node: valve, name: "drop-mode") == "forward-sticky-events", "drop-mode reads back")
    expect(try pipeline.getProperty(node: valve, name: "no-such-property") == nil, "an unknown property reads as nil")
    try pipeline.remove(node: valve)

    let branch = try pipeline.addBranch(tee: "t", description: "identity ! fakesink")
    expect(branch.count == 2, "the branch has two names, got \(branch)")
    try pipeline.removeBranch(sink: branch[1])

    var refused: String?
    do {
        _ = try pipeline.insertAfter(node: "no-such-node", description: "valve")
    } catch PipelineError.Edit(let message) {
        refused = message
    }
    print("refused: \(refused ?? "nothing")")
    expect(refused?.contains("UnknownNode") == true, "inserting after an unknown node is refused with UnknownNode")

    let stats = try pipeline.wait()
    print("edited run: consumed=\(stats.framesConsumed)")
    expect(stats.framesConsumed == editedFrameCount, "no frame is lost to the edits")
}

try roundTrip()
try edits()
