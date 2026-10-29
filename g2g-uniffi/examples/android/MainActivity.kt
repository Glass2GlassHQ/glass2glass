package dev.g2g.example

import android.app.Activity
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import dev.g2g.Pipeline
import dev.g2g.PipelineException
import dev.g2g.Surfaces

const val TAG = "g2g-kotlin"
const val CHANNEL = "preview"
const val FRAME_COUNT = 300

class MainActivity : Activity(), SurfaceHolder.Callback {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val view = SurfaceView(this)
        view.holder.addCallback(this)
        setContentView(view)
        Thread(::run).start()
    }

    private fun run() {
        try {
            Pipeline("videotestsrc pattern=smpte num-buffers=$FRAME_COUNT ! videoconvert ! surfacesink channel=$CHANNEL").use { pipeline ->
                val stats = pipeline.wait()
                Log.i(TAG, "consumed ${stats.framesConsumed} of $FRAME_COUNT frames")
            }
        } catch (error: PipelineException) {
            Log.e(TAG, "pipeline failed: ${error.message}")
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        Surfaces.attach(CHANNEL, holder.surface)
        Log.i(TAG, "surface attached")
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Surfaces.detach(CHANNEL)
    }
}
