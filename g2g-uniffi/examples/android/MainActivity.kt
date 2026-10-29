package dev.g2g.example

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.SurfaceView
import dev.g2g.Pipeline
import dev.g2g.PipelineException
import dev.g2g.Surfaces
import java.io.File
import java.util.concurrent.CountDownLatch

const val TAG = "g2g-kotlin"
const val CHANNEL = "preview"
const val FRAME_COUNT = 300
const val MODE_EXTRA = "mode"
const val CLIP_ASSET = "h264_640x480.h264"
const val CLIP_FRAME_COUNT = 10

class ExampleMode(val description: String, val frameCount: Int)

// The back camera's sensor is mounted landscape, so its frames are rotated upright.
fun exampleModes(clip: File) = mapOf(
    "bars" to ExampleMode(
        "videotestsrc pattern=smpte num-buffers=$FRAME_COUNT ! timeoverlay time-mode=buffer-count halignment=center valignment=center scale=8 ! videoconvert ! surfacesink channel=$CHANNEL",
        FRAME_COUNT,
    ),
    "camera" to ExampleMode(
        "camera2src num-buffers=$FRAME_COUNT ! videoflip method=clockwise ! videoconvert ! surfacesink channel=$CHANNEL",
        FRAME_COUNT,
    ),
    "decode" to ExampleMode(
        "filesrc location=${clip.path} ! h264parse ! mediacodecdec gpu-output=true ! surfacesink channel=$CHANNEL",
        CLIP_FRAME_COUNT,
    ),
)

class MainActivity : Activity(), SurfaceHolder.Callback {
    private lateinit var mode: String
    private val surfaceAttached = CountDownLatch(1)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val view = SurfaceView(this)
        view.holder.addCallback(this)
        setContentView(view)
        mode = intent.getStringExtra(MODE_EXTRA) ?: "bars"
        val needsCamera = mode == "camera" &&
            checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
        if (needsCamera) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 0)
            return
        }
        Thread(::run).start()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        if (grantResults.firstOrNull() != PackageManager.PERMISSION_GRANTED) {
            Log.e(TAG, "pipeline failed: camera permission denied")
            return
        }
        Thread(::run).start()
    }

    private fun run() {
        val clip = File(filesDir, CLIP_ASSET)
        val selected = exampleModes(clip)[mode]
        if (selected == null) {
            Log.e(TAG, "pipeline failed: unknown mode $mode")
            return
        }
        if (mode == "decode") {
            assets.open(CLIP_ASSET).use { asset -> clip.outputStream().use { asset.copyTo(it) } }
        }
        // surfacesink drops frames that arrive before the surface is attached
        surfaceAttached.await()
        try {
            Pipeline(selected.description).use { pipeline ->
                val stats = pipeline.wait()
                Log.i(TAG, "$mode consumed ${stats.framesConsumed} of ${selected.frameCount} frames")
            }
        } catch (error: PipelineException) {
            Log.e(TAG, "pipeline failed: ${error.message}")
        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        Surfaces.attach(CHANNEL, holder.surface)
        surfaceAttached.countDown()
        Log.i(TAG, "surface attached")
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Surfaces.detach(CHANNEL)
    }
}
