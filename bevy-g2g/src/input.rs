//! Input backchannel: viewer input flows back into Bevy over a WebSocket.
//!
//! WHIP through an SFU is publish-only and the viewer's WHEP session is a
//! separate peer connection, so a WebRTC data channel cannot reach the
//! publisher through the media server; a side-channel WebSocket is the
//! standard pixel-streaming shape. [`RemoteInputPlugin`] listens on a port,
//! parses JSON input messages, and injects them as ordinary Bevy input
//! messages (`KeyboardInput`, `MouseMotion`, ...), so the app's existing
//! input systems (`ButtonInput<KeyCode>`, `AccumulatedMouseMotion`, ...)
//! work unchanged.
//!
//! Protocol (one JSON object per WebSocket text frame):
//!
//! ```json
//! {"type":"key","code":"KeyW","down":true}
//! {"type":"mouse_move","dx":3.5,"dy":-1.0}
//! {"type":"mouse_button","button":"left","down":true}
//! {"type":"wheel","dx":0.0,"dy":-1.0}
//! ```
//!
//! `code` is the W3C `KeyboardEvent.code` string, which is also the Bevy
//! `KeyCode` variant name, so the browser value passes through unmapped.

use bevy::input::keyboard::{Key, KeyCode, KeyboardInput, NativeKey};
use bevy::input::mouse::{MouseButton, MouseButtonInput, MouseMotion, MouseScrollUnit, MouseWheel};
use bevy::input::touch::TouchPhase;
use bevy::input::ButtonState;
use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use futures_util::StreamExt;
use serde::Deserialize;

/// Serves the input WebSocket and injects received input into the app. Added
/// automatically by `RemoteRenderPlugins` when `StreamSettings::input_port`
/// is set; add it directly to compose with your own plugin stack.
#[derive(Debug)]
pub struct RemoteInputPlugin {
    pub port: u16,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum InputMsg {
    Key { code: String, down: bool },
    MouseMove { dx: f32, dy: f32 },
    MouseButton { button: String, down: bool },
    Wheel { dx: f32, dy: f32 },
}

#[derive(Resource)]
struct InputReceiver(Receiver<InputMsg>);

impl Plugin for RemoteInputPlugin {
    fn build(&self, app: &mut App) {
        let (tx, rx) = crossbeam_channel::unbounded();
        let port = self.port;
        std::thread::spawn(move || serve(port, tx));
        app.insert_resource(InputReceiver(rx))
            // PreUpdate, before the input systems that fold messages into
            // `ButtonInput` etc., so an injected press is visible to the
            // app's Update systems the same frame.
            .add_systems(PreUpdate, inject.before(bevy::input::InputSystems));
    }
}

/// Accept WebSocket clients and forward their parsed messages. Multiple
/// viewers may connect; their input interleaves.
fn serve(port: u16, tx: Sender<InputMsg>) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("input tokio runtime");
    rt.block_on(async move {
        let listener = match tokio::net::TcpListener::bind(("0.0.0.0", port)).await {
            Ok(l) => l,
            Err(e) => {
                error!("input backchannel: bind port {port} failed: {e}");
                return;
            }
        };
        info!("input backchannel listening on ws://0.0.0.0:{port}");
        loop {
            let Ok((stream, peer)) = listener.accept().await else {
                continue;
            };
            let tx = tx.clone();
            tokio::spawn(async move {
                let Ok(mut ws) = tokio_tungstenite::accept_async(stream).await else {
                    return;
                };
                info!("input client connected: {peer}");
                while let Some(Ok(msg)) = ws.next().await {
                    // Only text frames carry protocol messages; ping/pong and
                    // the close frame pass through here too.
                    let tokio_tungstenite::tungstenite::Message::Text(text) = msg else {
                        continue;
                    };
                    match serde_json::from_str::<InputMsg>(&text) {
                        Ok(parsed) => {
                            if tx.send(parsed).is_err() {
                                return;
                            }
                        }
                        Err(e) => warn!("input backchannel: bad message ({e}): {text}"),
                    }
                }
                info!("input client disconnected: {peer}");
            });
        }
    });
}

/// Drain the channel into Bevy input messages. The window entity is a
/// placeholder: the input-resource systems (`keyboard_input_system`,
/// `mouse_button_input_system`) ignore it.
fn inject(
    receiver: Res<InputReceiver>,
    mut keys: MessageWriter<KeyboardInput>,
    mut motion: MessageWriter<MouseMotion>,
    mut buttons: MessageWriter<MouseButtonInput>,
    mut wheel: MessageWriter<MouseWheel>,
    mut first: Local<bool>,
) {
    while let Ok(msg) = receiver.0.try_recv() {
        if !*first {
            *first = true;
            info!("first remote input message injected");
        }
        match msg {
            InputMsg::Key { code, down } => {
                let Some(key_code) = keycode(&code) else {
                    warn!("input backchannel: unknown key code {code}");
                    continue;
                };
                keys.write(KeyboardInput {
                    key_code,
                    logical_key: Key::Unidentified(NativeKey::Unidentified),
                    state: state(down),
                    text: None,
                    repeat: false,
                    window: Entity::PLACEHOLDER,
                });
            }
            InputMsg::MouseMove { dx, dy } => {
                motion.write(MouseMotion {
                    delta: Vec2::new(dx, dy),
                });
            }
            InputMsg::MouseButton { button, down } => {
                let button = match button.as_str() {
                    "left" => MouseButton::Left,
                    "right" => MouseButton::Right,
                    "middle" => MouseButton::Middle,
                    "back" => MouseButton::Back,
                    "forward" => MouseButton::Forward,
                    other => {
                        warn!("input backchannel: unknown mouse button {other}");
                        continue;
                    }
                };
                buttons.write(MouseButtonInput {
                    button,
                    state: state(down),
                    window: Entity::PLACEHOLDER,
                });
            }
            InputMsg::Wheel { dx, dy } => {
                wheel.write(MouseWheel {
                    unit: MouseScrollUnit::Line,
                    x: dx,
                    y: dy,
                    window: Entity::PLACEHOLDER,
                    phase: TouchPhase::Moved,
                });
            }
        }
    }
}

fn state(down: bool) -> ButtonState {
    if down {
        ButtonState::Pressed
    } else {
        ButtonState::Released
    }
}

/// W3C `KeyboardEvent.code` -> `KeyCode`: the variant names ARE the W3C code
/// strings, so this is a pass-through lookup over the supported set.
fn keycode(code: &str) -> Option<KeyCode> {
    macro_rules! codes {
        ($($k:ident)*) => {
            match code {
                $(stringify!($k) => Some(KeyCode::$k),)*
                _ => None,
            }
        };
    }
    codes!(
        KeyA KeyB KeyC KeyD KeyE KeyF KeyG KeyH KeyI KeyJ KeyK KeyL KeyM
        KeyN KeyO KeyP KeyQ KeyR KeyS KeyT KeyU KeyV KeyW KeyX KeyY KeyZ
        Digit0 Digit1 Digit2 Digit3 Digit4 Digit5 Digit6 Digit7 Digit8 Digit9
        F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12
        ArrowUp ArrowDown ArrowLeft ArrowRight
        Space Enter Escape Tab Backspace Delete Insert Home End PageUp PageDown
        ShiftLeft ShiftRight ControlLeft ControlRight AltLeft AltRight
        SuperLeft SuperRight CapsLock ContextMenu
        Minus Equal BracketLeft BracketRight Backslash Semicolon Quote
        Backquote Comma Period Slash
        Numpad0 Numpad1 Numpad2 Numpad3 Numpad4 Numpad5 Numpad6 Numpad7
        Numpad8 Numpad9 NumpadAdd NumpadSubtract NumpadMultiply NumpadDivide
        NumpadDecimal NumpadEnter NumLock
    )
}
