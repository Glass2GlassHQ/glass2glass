import dev.g2g.AppSink
import dev.g2g.AppSrc
import dev.g2g.Pipeline
import dev.g2g.PipelineException
import java.nio.ByteBuffer

const val FRAME_BYTES = 2 * 2 * 4
const val FRAME_COUNT = 3
const val FRAME_PERIOD_NS = 33_333_333UL
const val EDITED_FRAME_COUNT = 90UL

fun expect(condition: Boolean, what: String) {
    check(condition) { "expected: $what" }
}

fun roundTrip() {
    AppSrc("in").use { source ->
        AppSink("out").use { sink ->
            Pipeline(
                "appsrc channel=in caps=video/x-raw,format=RGBA,width=2,height=2,framerate=30/1 ! appsink channel=out",
            ).use { pipeline ->
                val direct = ByteBuffer.allocateDirect(FRAME_BYTES)
                for (index in 0 until FRAME_COUNT) {
                    val fill = (index + 1).toByte()
                    val pts = index.toULong() * FRAME_PERIOD_NS
                    if (index % 2 == 0) {
                        val frame = ByteArray(FRAME_BYTES) { fill }
                        while (!source.push(frame, pts)) {
                            Thread.yield()
                        }
                    } else {
                        direct.clear()
                        repeat(FRAME_BYTES) { direct.put(fill) }
                        direct.flip()
                        while (!source.pushBuffer(direct, pts)) {
                            Thread.yield()
                        }
                    }
                }
                source.endOfStream()

                var pulled = 0
                while (true) {
                    val sample = sink.pull() ?: break
                    println("frame $pulled: ${sample.data.size} bytes, first=${sample.data[0]}, pts=${sample.ptsNs} ns")
                    expect(sample.data.all { it == (pulled + 1).toByte() }, "frame $pulled carries the bytes pushed for it")
                    pulled++
                }

                while (true) {
                    val message = pipeline.pollBus() ?: break
                    println("bus: ${message.kind} ${message.text ?: ""}")
                }

                val stats = pipeline.wait()
                println("done: emitted=${stats.framesEmitted} consumed=${stats.framesConsumed} dropped=${stats.framesDropped}")
                expect(pulled == FRAME_COUNT, "pulled $pulled of $FRAME_COUNT frames")
            }
        }
    }
}

fun edits() {
    Pipeline("videotestsrc name=src num-buffers=$EDITED_FRAME_COUNT ! clocksync ! tee name=t ! fakesink").use { pipeline ->
        val valve = pipeline.insertAfter("src", "valve drop=false")
        pipeline.setProperty(valve, "drop-mode", "forward-sticky-events")
        expect(pipeline.getProperty(valve, "drop-mode") == "forward-sticky-events", "drop-mode reads back")
        expect(pipeline.getProperty(valve, "no-such-property") == null, "an unknown property reads as null")
        pipeline.remove(valve)

        val branch = pipeline.addBranch("t", "identity ! fakesink")
        expect(branch.size == 2, "the branch has two names, got $branch")
        pipeline.removeBranch(branch[1])

        val refused = try {
            pipeline.insertAfter("no-such-node", "valve")
            null
        } catch (error: PipelineException.Edit) {
            error.message
        }
        println("refused: $refused")
        expect(refused?.contains("UnknownNode") == true, "inserting after an unknown node is refused with UnknownNode")

        val stats = pipeline.wait()
        println("edited run: consumed=${stats.framesConsumed}")
        expect(stats.framesConsumed == EDITED_FRAME_COUNT, "no frame is lost to the edits")
    }
}

fun main() {
    roundTrip()
    edits()
}
