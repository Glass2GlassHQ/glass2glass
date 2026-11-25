# Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
#
# SPDX-License-Identifier: MPL-2.0

import sys

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, Gtk  # noqa: E402

log = open(sys.argv[1], "w", buffering=1)


def draw(area, context, width, height):
    # damage-driven capture sends a frame only when the picture changes
    shade = (area.get_frame_clock().get_frame_counter() % 256) / 255
    context.set_source_rgb(shade, 0.3, 0.6)
    context.paint()


def activate(app):
    window = Gtk.ApplicationWindow(application=app)
    area = Gtk.DrawingArea()
    area.set_draw_func(draw)
    area.add_tick_callback(lambda widget, clock: widget.queue_draw() or True)
    window.set_child(area)

    keys = Gtk.EventControllerKey()
    keys.connect(
        "key-pressed",
        lambda controller, keyval, code, state: print("key", Gdk.keyval_name(keyval), file=log),
    )
    window.add_controller(keys)
    clicks = Gtk.GestureClick(button=0)
    clicks.connect(
        "pressed",
        lambda gesture, count, x, y: print("button", gesture.get_current_button(), file=log),
    )
    window.add_controller(clicks)
    motion = Gtk.EventControllerMotion()
    motion.connect("motion", lambda controller, x, y: print("motion", int(x), int(y), file=log))
    window.add_controller(motion)

    window.fullscreen()
    window.present()


app = Gtk.Application()
app.connect("activate", activate)
app.run([])
