// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Gdi::{GetStockObject, BLACK_BRUSH, HBRUSH};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CreateWindowExW, DefWindowProcW, DispatchMessageW, GetSystemMetrics,
    GetWindowLongW, LoadCursorW, PeekMessageW, PostQuitMessage, RegisterClassExW, SetWindowPos,
    TranslateMessage, CW_USEDEFAULT, GWL_EXSTYLE, GWL_STYLE, IDC_ARROW, MSG, PM_REMOVE,
    SM_CXSCREEN, SM_CYSCREEN, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, WINDOW_EX_STYLE,
    WINDOW_STYLE, WM_DESTROY, WM_QUIT, WNDCLASSEXW, WS_OVERLAPPEDWINDOW, WS_POPUP, WS_VISIBLE,
};

#[cfg(feature = "wgpu-present")]
use crate::cursorshape::{CursorShape, CURSOR_BYTES_PER_PIXEL};
#[cfg(feature = "wgpu-present")]
use crate::windowinput::{PointerButton, WindowInput};
#[cfg(feature = "wgpu-present")]
use crate::windowworker::{
    PresentReports, WindowParams, WindowRenderer, WorkerChannels, WorkerCmd,
};

/// Drain pending window messages so the window stays responsive. False once
/// the window has been closed.
pub(crate) fn pump_messages() -> bool {
    let mut msg = MSG::default();
    // SAFETY: standard message-pump calls; `msg` is a valid local.
    unsafe {
        while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
            if msg.message == WM_QUIT {
                return false;
            }
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    true
}

#[cfg(feature = "wgpu-present")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowMessage {
    Input(WindowInput),
    PointerAt { x: i32, y: i32 },
}

#[cfg(feature = "wgpu-present")]
std::thread_local! {
    // win32 runs wndproc on the thread that created the window, the present worker
    static WINDOW_MESSAGES: core::cell::RefCell<alloc::vec::Vec<WindowMessage>> =
        const { core::cell::RefCell::new(alloc::vec::Vec::new()) };
}

#[cfg(feature = "wgpu-present")]
#[derive(Debug)]
struct OwnedCursor(windows::Win32::UI::WindowsAndMessaging::HCURSOR);

#[cfg(feature = "wgpu-present")]
impl Drop for OwnedCursor {
    fn drop(&mut self) {
        // SAFETY: the handle came from CreateIconIndirect and nothing else destroys it.
        unsafe {
            let _ = windows::Win32::UI::WindowsAndMessaging::DestroyCursor(self.0);
        }
    }
}

#[cfg(feature = "wgpu-present")]
#[derive(Debug)]
enum HostCursor {
    Hidden,
    Shape(OwnedCursor),
}

#[cfg(feature = "wgpu-present")]
std::thread_local! {
    // win32 asks wndproc for the client area's cursor on every pointer move
    static HOST_CURSOR: core::cell::RefCell<Option<HostCursor>> =
        const { core::cell::RefCell::new(None) };
}

#[cfg(feature = "wgpu-present")]
fn create_cursor(shape: &CursorShape) -> windows::core::Result<OwnedCursor> {
    use windows::Win32::Graphics::Gdi::{
        CreateBitmap, CreateDIBSection, DeleteObject, BITMAPINFO, BITMAPINFOHEADER, BI_RGB,
        DIB_RGB_COLORS,
    };
    use windows::Win32::UI::WindowsAndMessaging::{CreateIconIndirect, HCURSOR, ICONINFO};

    let (width, height) = (shape.width as i32, shape.height as i32);
    let header = BITMAPINFOHEADER {
        biSize: core::mem::size_of::<BITMAPINFOHEADER>() as u32,
        biWidth: width,
        // gdi reads a negative height as rows stored top down
        biHeight: -height,
        biPlanes: 1,
        biBitCount: 32,
        biCompression: BI_RGB.0,
        ..Default::default()
    };
    let info = BITMAPINFO {
        bmiHeader: header,
        ..Default::default()
    };
    let mut bits: *mut core::ffi::c_void = core::ptr::null_mut();
    // SAFETY: `info` describes a 32-bit top-down bitmap, and CreateDIBSection points `bits` at its pixels.
    let color = unsafe { CreateDIBSection(None, &info, DIB_RGB_COLORS, &mut bits, None, 0)? };
    let pixel_bytes = shape.width as usize * shape.height as usize * CURSOR_BYTES_PER_PIXEL;
    // SAFETY: the section holds width * height pixels of CURSOR_BYTES_PER_PIXEL bytes.
    let pixels = unsafe { core::slice::from_raw_parts_mut(bits.cast::<u8>(), pixel_bytes) };
    let (pixels, _) = pixels.as_chunks_mut::<CURSOR_BYTES_PER_PIXEL>();
    let (rgba, _) = shape.rgba.as_chunks::<CURSOR_BYTES_PER_PIXEL>();
    for (bgra, [red, green, blue, alpha]) in pixels.iter_mut().zip(rgba) {
        *bgra = [*blue, *green, *red, *alpha];
    }
    // win32 rows of a one-bit bitmap are padded to two bytes, and an all-zero mask leaves the colour's alpha in charge
    let mask_row_bytes = (shape.width as usize).div_ceil(16) * 2;
    let mask_bits = alloc::vec![0u8; mask_row_bytes * shape.height as usize];
    // SAFETY: `mask_bits` holds one padded row per line of the bitmap.
    let mask = unsafe { CreateBitmap(width, height, 1, 1, Some(mask_bits.as_ptr().cast())) };
    let icon_info = ICONINFO {
        fIcon: false.into(),
        xHotspot: shape.hotspot_x as u32,
        yHotspot: shape.hotspot_y as u32,
        hbmMask: mask,
        hbmColor: color,
    };
    // SAFETY: both bitmaps are live, and CreateIconIndirect copies them.
    let cursor = unsafe { CreateIconIndirect(&icon_info) };
    // SAFETY: the bitmaps were created above and are not used after this.
    unsafe {
        let _ = DeleteObject(color.into());
        let _ = DeleteObject(mask.into());
    }
    Ok(OwnedCursor(HCURSOR(cursor?.0)))
}

#[cfg(feature = "wgpu-present")]
fn show_host_cursor() {
    use windows::Win32::UI::WindowsAndMessaging::SetCursor;
    HOST_CURSOR.with_borrow(|cursor| {
        let Some(cursor) = cursor else {
            return;
        };
        let handle = match cursor {
            HostCursor::Hidden => None,
            HostCursor::Shape(shape) => Some(shape.0),
        };
        // SAFETY: the handle stays alive while HOST_CURSOR holds it.
        unsafe { SetCursor(handle) };
    });
}

#[cfg(feature = "wgpu-present")]
fn pointer_in_client_area(hwnd: HWND) -> bool {
    use windows::Win32::Foundation::POINT;
    use windows::Win32::Graphics::Gdi::ScreenToClient;
    use windows::Win32::UI::WindowsAndMessaging::{GetCursorPos, WindowFromPoint};
    let mut point = POINT::default();
    // SAFETY: `point` is a valid local.
    if unsafe { GetCursorPos(&mut point) }.is_err() {
        return false;
    }
    // SAFETY: plain queries on a live window.
    unsafe {
        if WindowFromPoint(point) != hwnd || !ScreenToClient(hwnd, &mut point).as_bool() {
            return false;
        }
    }
    let Ok((width, height)) = client_size(hwnd) else {
        return false;
    };
    (0..width as i32).contains(&point.x) && (0..height as i32).contains(&point.y)
}

#[cfg(feature = "wgpu-present")]
fn follow_host_cursor(hwnd: HWND, shape: Option<&CursorShape>, log_tag: &'static str) {
    let cursor = match shape.map(create_cursor).transpose() {
        Ok(cursor) => cursor.map_or(HostCursor::Hidden, HostCursor::Shape),
        Err(e) => {
            g2g_core::g2g_warn!(
                g2g_core::log::Target::category(log_tag),
                "cannot make the host's cursor: {e}"
            );
            return;
        }
    };
    HOST_CURSOR.set(Some(cursor));
    // win32 sets the client area's cursor only on the next pointer move
    if pointer_in_client_area(hwnd) {
        show_host_cursor();
    }
}

#[cfg(feature = "wgpu-present")]
fn low_word(value: usize) -> i32 {
    i32::from(value as u16 as i16)
}

#[cfg(feature = "wgpu-present")]
fn high_word(value: usize) -> i32 {
    low_word(value >> 16)
}

#[cfg(feature = "wgpu-present")]
fn window_message(msg: u32, wp: WPARAM, lp: LPARAM) -> Option<WindowMessage> {
    use windows::Win32::UI::WindowsAndMessaging::{
        KF_EXTENDED, KF_REPEAT, WM_KEYDOWN, WM_KEYUP, WM_KILLFOCUS, WM_LBUTTONDOWN, WM_LBUTTONUP,
        WM_MBUTTONDOWN, WM_MBUTTONUP, WM_MOUSEHWHEEL, WM_MOUSEMOVE, WM_MOUSEWHEEL, WM_RBUTTONDOWN,
        WM_RBUTTONUP, WM_SETFOCUS, WM_SYSKEYDOWN, WM_SYSKEYUP, WM_XBUTTONDOWN, WM_XBUTTONUP,
        XBUTTON1,
    };
    // the windows scancode set marks an extended key with an E0 prefix byte
    const EXTENDED_PREFIX: u32 = 0xE000;
    const SCANCODE_MASK: u32 = 0xFF;
    let button = |button, down| Some(WindowMessage::Input(WindowInput::Button { button, down }));
    match msg {
        WM_KEYDOWN | WM_SYSKEYDOWN | WM_KEYUP | WM_SYSKEYUP => {
            let key_flags = (lp.0 as usize >> 16) as u32;
            let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
            // the host repeats a held key itself
            if down && key_flags & KF_REPEAT != 0 {
                return None;
            }
            let prefix = if key_flags & KF_EXTENDED != 0 {
                EXTENDED_PREFIX
            } else {
                0
            };
            let scancode = prefix | (key_flags & SCANCODE_MASK);
            Some(WindowMessage::Input(WindowInput::Key { scancode, down }))
        }
        WM_MOUSEMOVE => Some(WindowMessage::PointerAt {
            x: low_word(lp.0 as usize),
            y: high_word(lp.0 as usize),
        }),
        WM_LBUTTONDOWN => button(PointerButton::Left, true),
        WM_LBUTTONUP => button(PointerButton::Left, false),
        WM_RBUTTONDOWN => button(PointerButton::Right, true),
        WM_RBUTTONUP => button(PointerButton::Right, false),
        WM_MBUTTONDOWN => button(PointerButton::Middle, true),
        WM_MBUTTONUP => button(PointerButton::Middle, false),
        WM_XBUTTONDOWN | WM_XBUTTONUP => {
            let side = if high_word(wp.0) == i32::from(XBUTTON1) {
                PointerButton::Back
            } else {
                PointerButton::Forward
            };
            button(side, msg == WM_XBUTTONDOWN)
        }
        WM_MOUSEWHEEL => Some(WindowMessage::Input(WindowInput::Wheel {
            dx: 0,
            dy: high_word(wp.0),
        })),
        WM_MOUSEHWHEEL => Some(WindowMessage::Input(WindowInput::Wheel {
            dx: high_word(wp.0),
            dy: 0,
        })),
        WM_SETFOCUS => Some(WindowMessage::Input(WindowInput::Focus(true))),
        WM_KILLFOCUS => Some(WindowMessage::Input(WindowInput::Focus(false))),
        _ => None,
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    #[cfg(feature = "wgpu-present")]
    if let Some(message) = window_message(msg, wp, lp) {
        WINDOW_MESSAGES.with_borrow_mut(|messages| messages.push(message));
    }
    #[cfg(feature = "wgpu-present")]
    {
        use windows::Win32::UI::WindowsAndMessaging::{HTCLIENT, WM_SETCURSOR};
        if msg == WM_SETCURSOR
            && low_word(lp.0 as usize) == HTCLIENT as i32
            && HOST_CURSOR.with_borrow(Option::is_some)
        {
            show_host_cursor();
            // win32 keeps the class cursor unless WM_SETCURSOR returns TRUE
            return LRESULT(1);
        }
    }
    // win32 sends WM_DESTROY when the user closes the window
    if msg == WM_DESTROY {
        // SAFETY: called on the thread that owns the window's message queue.
        unsafe { PostQuitMessage(0) };
        return LRESULT(0);
    }
    // SAFETY: forwarded straight to the default handler.
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

fn outer_size(
    style: WINDOW_STYLE,
    extended_style: WINDOW_EX_STYLE,
    (width, height): (u32, u32),
) -> windows::core::Result<(i32, i32)> {
    let mut rect = RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    };
    // SAFETY: `rect` is a valid local the call adjusts in place.
    unsafe { AdjustWindowRectEx(&mut rect, style, false, extended_style)? };
    Ok((rect.right - rect.left, rect.bottom - rect.top))
}

pub(crate) fn create_window(
    title: &str,
    client_size: (u32, u32),
    fullscreen: bool,
) -> windows::core::Result<HWND> {
    // SAFETY: standard Win32 window creation on the worker thread.
    unsafe {
        let hinstance = GetModuleHandleW(None)?;
        let class_name = w!("g2g_window");
        let wc = WNDCLASSEXW {
            cbSize: core::mem::size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(wndproc),
            hInstance: hinstance.into(),
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            hbrBackground: HBRUSH(GetStockObject(BLACK_BRUSH).0),
            lpszClassName: class_name,
            ..Default::default()
        };
        // RegisterClassExW returns 0 on failure; a duplicate registration
        // (second sink) also "fails" but is harmless, so we ignore the result
        // and let CreateWindowExW surface a real problem.
        let _ = RegisterClassExW(&wc);

        let title_w: alloc::vec::Vec<u16> =
            title.encode_utf16().chain(core::iter::once(0)).collect();
        let (style, x, y, width, height) = if fullscreen {
            (
                WS_POPUP | WS_VISIBLE,
                0,
                0,
                GetSystemMetrics(SM_CXSCREEN),
                GetSystemMetrics(SM_CYSCREEN),
            )
        } else {
            let style = WS_OVERLAPPEDWINDOW | WS_VISIBLE;
            let (width, height) = outer_size(style, WINDOW_EX_STYLE::default(), client_size)?;
            (style, CW_USEDEFAULT, CW_USEDEFAULT, width, height)
        };
        CreateWindowExW(
            WINDOW_EX_STYLE::default(),
            class_name,
            windows::core::PCWSTR(title_w.as_ptr()),
            style,
            x,
            y,
            width,
            height,
            None,
            None,
            Some(hinstance.into()),
            None,
        )
    }
}

pub(crate) fn resize_client(hwnd: HWND, client_size: (u32, u32)) -> windows::core::Result<()> {
    // SAFETY: `hwnd` is a live window owned by this thread.
    unsafe {
        let style = WINDOW_STYLE(GetWindowLongW(hwnd, GWL_STYLE) as u32);
        let extended_style = WINDOW_EX_STYLE(GetWindowLongW(hwnd, GWL_EXSTYLE) as u32);
        let (width, height) = outer_size(style, extended_style, client_size)?;
        SetWindowPos(
            hwnd,
            None,
            0,
            0,
            width,
            height,
            SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
        )
    }
}

#[cfg(feature = "wgpu-present")]
fn client_size(hwnd: HWND) -> windows::core::Result<(u32, u32)> {
    let mut rect = RECT::default();
    // SAFETY: `hwnd` is a live window and `rect` a valid local.
    unsafe { windows::Win32::UI::WindowsAndMessaging::GetClientRect(hwnd, &mut rect)? };
    Ok((
        (rect.right - rect.left) as u32,
        (rect.bottom - rect.top) as u32,
    ))
}

#[cfg(feature = "wgpu-present")]
const MESSAGE_POLL: core::time::Duration = core::time::Duration::from_millis(8);

#[cfg(feature = "wgpu-present")]
pub(crate) fn run_window<R: WindowRenderer>(
    build_renderer: impl FnOnce(HWND, u32, u32) -> Result<R, alloc::boxed::Box<dyn std::error::Error>>,
    params: WindowParams,
    channels: WorkerChannels<std::sync::mpsc::Receiver<WorkerCmd<R::Frame>>>,
) -> Result<(), alloc::boxed::Box<dyn std::error::Error>> {
    let hwnd = create_window(
        &params.title,
        (params.width, params.height),
        params.fullscreen,
    )?;
    let presented = present_loop(hwnd, build_renderer, &params, channels);
    // SAFETY: the window was created on this thread, and the renderer built over
    // it was dropped when `present_loop` returned.
    unsafe {
        let _ = windows::Win32::UI::WindowsAndMessaging::DestroyWindow(hwnd);
    }
    presented
}

#[cfg(feature = "wgpu-present")]
fn present_loop<R: WindowRenderer>(
    hwnd: HWND,
    build_renderer: impl FnOnce(HWND, u32, u32) -> Result<R, alloc::boxed::Box<dyn std::error::Error>>,
    params: &WindowParams,
    channels: WorkerChannels<std::sync::mpsc::Receiver<WorkerCmd<R::Frame>>>,
) -> Result<(), alloc::boxed::Box<dyn std::error::Error>> {
    use core::sync::atomic::Ordering;
    use std::sync::mpsc::RecvTimeoutError;

    let mut size = client_size(hwnd)?;
    let mut renderer = build_renderer(hwnd, size.0, size.1)?;
    let reports = PresentReports {
        presented: channels.presented,
        latency: channels.latency,
        log_tag: params.log_tag,
    };
    channels.ready.notify();
    let mut input = channels.input;
    let mut cursor_shapes = channels.cursor_shapes;
    loop {
        if let Some(shapes) = &mut cursor_shapes {
            match shapes.has_changed() {
                Ok(true) => {
                    let shape = shapes.borrow_and_update().clone();
                    follow_host_cursor(hwnd, shape.as_ref(), params.log_tag);
                }
                Ok(false) => {}
                Err(_) => cursor_shapes = None,
            }
        }
        if !pump_messages() {
            channels.window_closed.store(true, Ordering::Release);
            return Ok(());
        }
        let messages = WINDOW_MESSAGES.with_borrow_mut(core::mem::take);
        if let Some(sender) = &input {
            let window_inputs = messages.into_iter().filter_map(|message| match message {
                WindowMessage::Input(window_input) => Some(window_input),
                WindowMessage::PointerAt { x, y } => renderer
                    .picture_point(f64::from(x), f64::from(y))
                    .map(|(x, y)| WindowInput::Pointer { x, y }),
            });
            for window_input in window_inputs {
                if sender.send(window_input).is_err() {
                    input = None;
                    break;
                }
            }
        }
        let current = client_size(hwnd)?;
        // a minimized window has an empty client area
        if current != size && current.0 > 0 && current.1 > 0 {
            renderer.resize(current.0, current.1);
            size = current;
        }
        match channels.rx.recv_timeout(MESSAGE_POLL) {
            Ok(WorkerCmd::Frame {
                frame,
                arrival_ns,
                ack,
            }) => {
                let drawn = renderer.present(&frame);
                reports.acknowledge(drawn, frame, arrival_ns, ack);
            }
            Ok(WorkerCmd::Reconfigure {
                caps,
                width,
                height,
            }) => {
                if let Err(e) = renderer.reconfigure(&caps) {
                    g2g_core::g2g_error!(
                        g2g_core::log::Target::category(params.log_tag),
                        "cannot apply the new caps to the window: {e:?}"
                    );
                    return Ok(());
                }
                if !params.fullscreen {
                    resize_client(hwnd, (width, height))?;
                }
            }
            Ok(WorkerCmd::Shutdown) | Err(RecvTimeoutError::Disconnected) => return Ok(()),
            Err(RecvTimeoutError::Timeout) => {}
        }
    }
}
