package dev.g2g

import android.view.Surface

object Surfaces {
    init {
        System.loadLibrary("g2g_uniffi")
    }

    // Frames for `surfacesink channel=<channel>` draw into the surface from the next frame on.
    @JvmStatic
    external fun attach(channel: String, surface: Surface)

    // Returns once no sink draws into the surface, so call it from surfaceDestroyed.
    @JvmStatic
    external fun detach(channel: String)
}
