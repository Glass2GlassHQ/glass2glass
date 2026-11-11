// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use windows::core::w;
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::{
    AdjustWindowRectEx, CreateWindowExW, DefWindowProcW, DispatchMessageW, GetSystemMetrics,
    GetWindowLongW, PeekMessageW, PostQuitMessage, RegisterClassExW, SetWindowPos,
    TranslateMessage, CW_USEDEFAULT, GWL_EXSTYLE, GWL_STYLE, MSG, PM_REMOVE, SM_CXSCREEN,
    SM_CYSCREEN, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, WINDOW_EX_STYLE, WINDOW_STYLE,
    WM_DESTROY, WM_QUIT, WNDCLASSEXW, WS_OVERLAPPEDWINDOW, WS_POPUP, WS_VISIBLE,
};

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

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
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
    loop {
        if !pump_messages() {
            channels.window_closed.store(true, Ordering::Release);
            return Ok(());
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
