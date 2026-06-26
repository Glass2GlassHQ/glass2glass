//! M308: Android camera capture via the NDK Camera2 API (`libcamera2ndk`).
//!
//! `Camera2Src` is a `SourceLoop` that captures `YUV_420_888` frames from a
//! camera and emits tight NV12 (`MemoryDomain::System`, `Caps::RawVideo`), the
//! Android analog of `V4l2Src` (Linux) / `avfvideosrc` (macOS). It is the capture
//! mirror of the MediaCodec decode path and pairs naturally with `MediaCodecEnc`
//! for an on-device camera -> H.264 encode pipeline.
//!
//! **Why raw FFI.** The safe `ndk` crate wraps MediaCodec, ImageReader and AAudio
//! but not Camera2, so the camera control surface (`ACameraManager`,
//! `ACameraDevice`, `ACameraCaptureSession`, `ACaptureRequest`) is called through
//! `ndk-sys` directly. The frame delivery side reuses a safe `ndk` `ImageReader`:
//! the capture session targets the reader's `ANativeWindow`, and we acquire
//! `Image`s and pack their `YUV_420_888` planes to NV12 (row/pixel strides handle
//! any vendor layout), exactly as the decoder does. `ACameraManager_openCamera`
//! is synchronous (its state callbacks are only for disconnect / error), so the
//! whole session is set up inline in `configure_pipeline`, no async open handshake.
//!
//! **Headless.** Camera access needs the `CAMERA` runtime permission, which a
//! bare native binary does not hold; on-device validation of real capture needs
//! an APK harness. The element + FFI cross-compile and are exercised by the
//! `android_camera2_probe` (which reports a permission denial rather than
//! failing). See `tools/android-camera2-smoke.sh`.
//!
//! `camera2` feature (implies `std`).

use core::ffi::{c_char, c_int, c_void};
use core::future::Future;
use core::pin::Pin;
use core::time::Duration;

use ndk::media::image_reader::{AcquireResult, Image, ImageFormat, ImageReader};
use ndk::native_window::NativeWindow;

use g2g_core::frame::Frame;
use g2g_core::memory::SystemSlice;
use g2g_core::runtime::SourceLoop;
use g2g_core::{
    Caps, CapsConstraint, CapsSet, ConfigureOutcome, Dim, FrameTiming, G2gError, HardwareError,
    MemoryDomain, OutputSink, PadTemplate, PadTemplates, PipelinePacket, RawVideoFormat, Rate,
};

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::ffi::CString;
use alloc::string::String;
use alloc::vec::Vec;

// `ndk-sys` declares the Camera2 externs but links no library for them, so pull
// in libcamera2ndk explicitly (the camera control surface). ImageReader's
// libmediandk is already linked by the ndk crate's `media` feature.
#[link(name = "camera2ndk")]
extern "C" {}

/// Images the `ImageReader` may hold; a small queue so capture never stalls
/// waiting for us to acquire, without hoarding camera buffers.
const MAX_IMAGES: i32 = 4;

/// Bounded poll attempts per frame while the camera has not delivered one yet
/// (v1 has no image-available callback; see the module note). At `POLL_SLEEP`
/// apart this waits ~2 s, far longer than a frame interval, before giving up.
const MAX_ACQUIRE_ATTEMPTS: u32 = 400;

/// Sleep between acquire attempts. Critical: the camera HAL delivers buffers via
/// binder callbacks into this process, so the run loop must YIELD the CPU (a
/// tight spin starves those threads and no frame ever lands).
const POLL_SLEEP: Duration = Duration::from_millis(5);

/// `ACameraDevice_request_template::TEMPLATE_PREVIEW`.
const TEMPLATE_PREVIEW: ndk_sys::ACameraDevice_request_template =
    ndk_sys::ACameraDevice_request_template::TEMPLATE_PREVIEW;

/// Map a Camera2 status to a structured error.
fn ck(status: ndk_sys::camera_status_t) -> Result<(), G2gError> {
    if status == ndk_sys::camera_status_t::ACAMERA_OK {
        Ok(())
    } else {
        Err(G2gError::Hardware(HardwareError::Other))
    }
}

// No-op camera / session state callbacks. The pointers passed to the NDK must be
// non-null and outlive the device / session; they are boxed in `CameraSession`.
// SAFETY: trivial no-ops; they touch none of their arguments.
unsafe extern "C" fn on_disconnected(_ctx: *mut c_void, _dev: *mut ndk_sys::ACameraDevice) {}
unsafe extern "C" fn on_error(_ctx: *mut c_void, _dev: *mut ndk_sys::ACameraDevice, _err: c_int) {}
unsafe extern "C" fn on_session_state(
    _ctx: *mut c_void,
    _session: *mut ndk_sys::ACameraCaptureSession,
) {
}

/// Owns one live capture session and every raw Camera2 resource backing it, plus
/// the safe `ImageReader` frames land in. Torn down in order by `Drop`.
struct CameraSession {
    manager: *mut ndk_sys::ACameraManager,
    device: *mut ndk_sys::ACameraDevice,
    session: *mut ndk_sys::ACameraCaptureSession,
    request: *mut ndk_sys::ACaptureRequest,
    target: *mut ndk_sys::ACameraOutputTarget,
    container: *mut ndk_sys::ACaptureSessionOutputContainer,
    output: *mut ndk_sys::ACaptureSessionOutput,
    reader: ImageReader,
    // The reader's Surface, handed to the capture session; kept resident.
    _window: NativeWindow,
    // Boxed state-callback structs the NDK holds pointers to; must outlive the
    // device / session, so they live here and drop after teardown.
    _dev_cbs: Box<ndk_sys::ACameraDevice_StateCallbacks>,
    _sess_cbs: Box<ndk_sys::ACameraCaptureSession_stateCallbacks>,
}

impl Drop for CameraSession {
    fn drop(&mut self) {
        // SAFETY: each pointer was created by the matching NDK constructor in
        // `open` and is freed exactly once here, in dependency order: stop the
        // repeating request and close the session before the device, then free
        // the request / targets / container, then the manager. The ImageReader
        // and window drop after (their Surface is no longer a session target).
        unsafe {
            if !self.session.is_null() {
                ndk_sys::ACameraCaptureSession_stopRepeating(self.session);
                ndk_sys::ACameraCaptureSession_close(self.session);
            }
            if !self.device.is_null() {
                ndk_sys::ACameraDevice_close(self.device);
            }
            if !self.request.is_null() {
                ndk_sys::ACaptureRequest_free(self.request);
            }
            if !self.target.is_null() {
                ndk_sys::ACameraOutputTarget_free(self.target);
            }
            if !self.container.is_null() {
                ndk_sys::ACaptureSessionOutputContainer_free(self.container);
            }
            if !self.output.is_null() {
                ndk_sys::ACaptureSessionOutput_free(self.output);
            }
            if !self.manager.is_null() {
                ndk_sys::ACameraManager_delete(self.manager);
            }
        }
    }
}

impl core::fmt::Debug for CameraSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CameraSession").finish_non_exhaustive()
    }
}

/// Android camera capture source: `YUV_420_888` in, tight NV12 out.
#[derive(Debug)]
pub struct Camera2Src {
    width: u32,
    height: u32,
    target_buffers: u64,
    camera_id: Option<String>,
    configured: bool,
    cam: Option<CameraSession>,
    emitted: u64,
}

// SAFETY: the raw Camera2 handles and the `ImageReader` are only touched from the
// element's owning task on a single-thread executor (the same contract as the
// MediaCodec / AAudio elements); never from two threads at once.
unsafe impl Send for Camera2Src {}

impl Camera2Src {
    /// A capture source at `width` x `height`, emitting `target_buffers` frames
    /// then EOS (`u64::MAX` = capture until stopped). Resolution must be one the
    /// camera supports for `YUV_420_888` (640x480 / 1280x720 are widely safe).
    pub fn new(width: u32, height: u32, target_buffers: u64) -> Self {
        Self {
            width,
            height,
            target_buffers,
            camera_id: None,
            configured: false,
            cam: None,
            emitted: 0,
        }
    }

    /// Select a specific camera id (default: the first the manager reports).
    pub fn with_camera_id(mut self, id: impl Into<String>) -> Self {
        self.camera_id = Some(id.into());
        self
    }

    /// Count of NV12 frames pushed downstream. Useful in tests.
    pub fn emitted(&self) -> u64 {
        self.emitted
    }

    fn caps(&self) -> Caps {
        Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(self.width),
            height: Dim::Fixed(self.height),
            framerate: Rate::Any,
        }
    }

    /// Pick the camera id to open: the configured one, else the first reported.
    fn resolve_camera_id(&self, manager: *mut ndk_sys::ACameraManager) -> Result<CString, G2gError> {
        if let Some(id) = &self.camera_id {
            return CString::new(id.as_str()).map_err(|_| G2gError::CapsMismatch);
        }
        let mut list: *mut ndk_sys::ACameraIdList = core::ptr::null_mut();
        // SAFETY: `manager` is live; `list` is an out-param filled on success and
        // freed below.
        unsafe {
            ck(ndk_sys::ACameraManager_getCameraIdList(manager, &mut list))?;
            if list.is_null() || (*list).numCameras < 1 {
                if !list.is_null() {
                    ndk_sys::ACameraManager_deleteCameraIdList(list);
                }
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            // `cameraIds[0]` is a NUL-terminated string owned by the list; copy it
            // before freeing the list.
            let first = *(*list).cameraIds;
            let owned = core::ffi::CStr::from_ptr(first).to_owned();
            ndk_sys::ACameraManager_deleteCameraIdList(list);
            Ok(owned)
        }
    }

    /// Open the camera, build the preview capture request targeting the
    /// `ImageReader`'s Surface, create the session, and start the repeating
    /// request. All Camera2 calls are synchronous here.
    fn open(&mut self) -> Result<(), G2gError> {
        let reader = ImageReader::new(
            self.width as i32,
            self.height as i32,
            ImageFormat::YUV_420_888,
            MAX_IMAGES,
        )
        .map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let window = reader.window().map_err(|_| G2gError::Hardware(HardwareError::Other))?;
        let anw = window.ptr().as_ptr() as *mut ndk_sys::ACameraWindowType;

        let dev_cbs = Box::new(ndk_sys::ACameraDevice_StateCallbacks {
            context: core::ptr::null_mut(),
            onDisconnected: Some(on_disconnected),
            onError: Some(on_error),
        });
        let sess_cbs = Box::new(ndk_sys::ACameraCaptureSession_stateCallbacks {
            context: core::ptr::null_mut(),
            onClosed: Some(on_session_state),
            onReady: Some(on_session_state),
            onActive: Some(on_session_state),
        });

        // SAFETY: every out-param below is null-initialised and filled by its NDK
        // constructor on `ACAMERA_OK` (checked via `ck`); `anw` is the live
        // ImageReader Surface; the callback structs outlive the device / session
        // (boxed into the returned `CameraSession`). On any early error the
        // already-created resources are dropped by `partial` (a CameraSession with
        // the nulls left for what was not built).
        unsafe {
            let manager = ndk_sys::ACameraManager_create();
            if manager.is_null() {
                return Err(G2gError::Hardware(HardwareError::Other));
            }
            // A partially-built session so any early-return frees what exists.
            let mut partial = CameraSession {
                manager,
                device: core::ptr::null_mut(),
                session: core::ptr::null_mut(),
                request: core::ptr::null_mut(),
                target: core::ptr::null_mut(),
                container: core::ptr::null_mut(),
                output: core::ptr::null_mut(),
                reader,
                _window: window,
                _dev_cbs: dev_cbs,
                _sess_cbs: sess_cbs,
            };

            let id = self.resolve_camera_id(manager)?;
            ck(ndk_sys::ACameraManager_openCamera(
                manager,
                id.as_ptr() as *const c_char,
                partial._dev_cbs.as_mut() as *mut _,
                &mut partial.device,
            ))?;
            ck(ndk_sys::ACameraDevice_createCaptureRequest(
                partial.device,
                TEMPLATE_PREVIEW,
                &mut partial.request,
            ))?;
            ck(ndk_sys::ACameraOutputTarget_create(anw, &mut partial.target))?;
            ck(ndk_sys::ACaptureRequest_addTarget(partial.request, partial.target))?;
            ck(ndk_sys::ACaptureSessionOutputContainer_create(&mut partial.container))?;
            ck(ndk_sys::ACaptureSessionOutput_create(anw, &mut partial.output))?;
            ck(ndk_sys::ACaptureSessionOutputContainer_add(partial.container, partial.output))?;
            ck(ndk_sys::ACameraDevice_createCaptureSession(
                partial.device,
                partial.container,
                partial._sess_cbs.as_mut() as *mut _,
                &mut partial.session,
            ))?;
            ck(ndk_sys::ACameraCaptureSession_setRepeatingRequest(
                partial.session,
                core::ptr::null_mut(),
                1,
                &mut partial.request,
                core::ptr::null_mut(),
            ))?;

            self.cam = Some(partial);
        }
        Ok(())
    }
}

impl SourceLoop for Camera2Src {
    type RunFuture<'a> = Pin<Box<dyn Future<Output = Result<u64, G2gError>> + 'a>>
    where
        Self: 'a;

    type CapsFuture<'a> = core::future::Ready<Result<Caps, G2gError>>
    where
        Self: 'a;

    fn intercept_caps<'a>(&'a mut self) -> Self::CapsFuture<'a> {
        core::future::ready(Ok(self.caps()))
    }

    fn caps_constraint<'a>(
        &'a mut self,
    ) -> impl Future<Output = Result<CapsConstraint<'a>, G2gError>> + 'a {
        core::future::ready(Ok(CapsConstraint::Produces(CapsSet::one(self.caps()))))
    }

    fn configure_pipeline(&mut self, absolute_caps: &Caps) -> Result<ConfigureOutcome, G2gError> {
        match absolute_caps {
            Caps::RawVideo { format: RawVideoFormat::Nv12, width, height, .. } => {
                if let Dim::Fixed(w) = width {
                    self.width = *w;
                }
                if let Dim::Fixed(h) = height {
                    self.height = *h;
                }
            }
            _ => return Err(G2gError::CapsMismatch),
        }
        self.open()?;
        self.configured = true;
        Ok(ConfigureOutcome::Accepted)
    }

    fn run<'a>(&'a mut self, out: &'a mut dyn OutputSink) -> Self::RunFuture<'a> {
        Box::pin(async move {
            if !self.configured {
                return Err(G2gError::NotConfigured);
            }
            let mut seq = 0u64;
            while seq < self.target_buffers {
                // Poll the reader until the camera delivers a frame (v1 has no
                // image-available callback). `acquire_latest_image` drops any
                // backlog so we always emit the freshest frame. Sleep between
                // attempts so the binder threads delivering buffers get the CPU.
                let mut nv12: Option<(Vec<u8>, u64)> = None;
                for _ in 0..MAX_ACQUIRE_ATTEMPTS {
                    let cam = self.cam.as_ref().ok_or(G2gError::NotConfigured)?;
                    match cam
                        .reader
                        .acquire_latest_image()
                        .map_err(|_| G2gError::Hardware(HardwareError::Other))?
                    {
                        AcquireResult::Image(img) => {
                            let pts_ns = img.timestamp().unwrap_or(0).max(0) as u64;
                            if let Some(packed) = image_to_nv12(&img) {
                                nv12 = Some((packed, pts_ns));
                                break;
                            }
                        }
                        // No frame yet / our queue is momentarily full: wait a beat.
                        _ => std::thread::sleep(POLL_SLEEP),
                    }
                }
                let Some((bytes, pts_ns)) = nv12 else {
                    return Err(G2gError::Hardware(HardwareError::Other));
                };

                #[cfg(feature = "std")]
                let arrival_ns = g2g_core::metrics::monotonic_ns();
                #[cfg(not(feature = "std"))]
                let arrival_ns: u64 = 0;
                let frame = Frame {
                    domain: MemoryDomain::System(SystemSlice::from_boxed(bytes.into_boxed_slice())),
                    timing: FrameTiming {
                        pts_ns,
                        dts_ns: pts_ns,
                        duration_ns: 0,
                        capture_ns: pts_ns,
                        arrival_ns,
                        keyframe: true, // raw frames are independent
                    },
                    sequence: seq,
                    meta: Default::default(),
                };
                self.emitted += 1;
                seq += 1;
                out.push(PipelinePacket::DataFrame(frame)).await?;
            }

            // Drop the session (stops the repeating request, closes device).
            self.cam = None;
            out.push(PipelinePacket::Eos).await?;
            Ok(seq)
        })
    }
}

impl PadTemplates for Camera2Src {
    fn pad_templates() -> Vec<PadTemplate> {
        Vec::from([PadTemplate::source(CapsSet::one(Caps::RawVideo {
            format: RawVideoFormat::Nv12,
            width: Dim::Fixed(640),
            height: Dim::Fixed(480),
            framerate: Rate::Any,
        }))])
    }
}

/// Pack a decoded `YUV_420_888` image to tight NV12 (Y plane then interleaved
/// UV). The per-plane row / pixel strides describe whatever layout the camera
/// chose, so this one path handles planar I420, semi-planar, and vendor formats.
fn image_to_nv12(img: &Image) -> Option<Vec<u8>> {
    let w = img.width().ok()?.max(0) as usize;
    let h = img.height().ok()?.max(0) as usize;
    if w == 0 || h == 0 {
        return None;
    }
    let y = img.plane_data(0).ok()?;
    let y_rs = img.plane_row_stride(0).ok()? as usize;
    let u = img.plane_data(1).ok()?;
    let u_rs = img.plane_row_stride(1).ok()? as usize;
    let u_ps = img.plane_pixel_stride(1).ok()? as usize;
    let v = img.plane_data(2).ok()?;
    let v_rs = img.plane_row_stride(2).ok()? as usize;
    let v_ps = img.plane_pixel_stride(2).ok()? as usize;

    let (cw, ch) = (w / 2, h / 2);
    let mut nv12 = Vec::with_capacity(w * h + 2 * cw * ch);
    for row in 0..h {
        let off = row * y_rs;
        nv12.extend_from_slice(y.get(off..off + w)?);
    }
    for row in 0..ch {
        for col in 0..cw {
            nv12.push(*u.get(row * u_rs + col * u_ps)?);
            nv12.push(*v.get(row * v_rs + col * v_ps)?);
        }
    }
    Some(nv12)
}
