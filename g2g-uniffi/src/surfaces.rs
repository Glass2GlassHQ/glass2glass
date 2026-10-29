use std::ffi::CStr;
use std::ptr;

use g2g_plugins::surfacesink::{attach_surface, detach_surface};
use jni_sys::{jclass, jobject, jstring, JNIEnv};
use ndk::native_window::NativeWindow;

// SAFETY: the JVM calls this as dev.g2g.Surfaces.attach with a live env, a String and a Surface.
#[no_mangle]
pub(crate) unsafe extern "system" fn Java_dev_g2g_Surfaces_attach(
    env: *mut JNIEnv,
    _class: jclass,
    channel: jstring,
    surface: jobject,
) {
    // SAFETY: `surface` is the android.view.Surface the JVM passed in.
    let Some(window) = (unsafe { NativeWindow::from_surface(env, surface) }) else {
        // SAFETY: `env` is the live env of this call.
        unsafe { throw_illegal_argument(env, c"the Surface has no native window") };
        return;
    };
    // SAFETY: `channel` is the String the JVM passed in.
    if let Some(channel) = unsafe { read_string(env, channel) } {
        attach_surface(&channel, window);
    }
}

// SAFETY: the JVM calls this as dev.g2g.Surfaces.detach with a live env and a String.
#[no_mangle]
pub(crate) unsafe extern "system" fn Java_dev_g2g_Surfaces_detach(
    env: *mut JNIEnv,
    _class: jclass,
    channel: jstring,
) {
    // SAFETY: `channel` is the String the JVM passed in.
    if let Some(channel) = unsafe { read_string(env, channel) } {
        detach_surface(&channel);
    }
}

// None with a Java exception pending.
unsafe fn read_string(env: *mut JNIEnv, text: jstring) -> Option<String> {
    // SAFETY: `env` is a live env, so its function table is filled in.
    let table = unsafe { &**env };
    let get = table.GetStringUTFChars.expect("JNI GetStringUTFChars");
    let release = table
        .ReleaseStringUTFChars
        .expect("JNI ReleaseStringUTFChars");
    // SAFETY: `text` is a live java.lang.String reference.
    let chars = unsafe { get(env, text, ptr::null_mut()) };
    if chars.is_null() {
        return None;
    }
    // SAFETY: GetStringUTFChars returns a NUL-terminated string valid until the release below.
    let owned = unsafe { CStr::from_ptr(chars) }.to_str().map(str::to_owned);
    // SAFETY: `chars` came from GetStringUTFChars on `text` and is released once.
    unsafe { release(env, text, chars) };
    if owned.is_err() {
        // SAFETY: `env` is the live env of this call.
        unsafe { throw_illegal_argument(env, c"the channel name is not valid UTF-8") };
    }
    owned.ok()
}

unsafe fn throw_illegal_argument(env: *mut JNIEnv, message: &CStr) {
    // SAFETY: `env` is a live env, so its function table is filled in.
    let table = unsafe { &**env };
    let find_class = table.FindClass.expect("JNI FindClass");
    let throw_new = table.ThrowNew.expect("JNI ThrowNew");
    // SAFETY: the class name is a NUL-terminated JNI class descriptor.
    let class = unsafe { find_class(env, c"java/lang/IllegalArgumentException".as_ptr()) };
    // SAFETY: `class` is the class FindClass just returned.
    unsafe { throw_new(env, class, message.as_ptr()) };
}
