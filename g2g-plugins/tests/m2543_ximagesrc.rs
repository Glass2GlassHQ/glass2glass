// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

#![cfg(all(target_os = "linux", feature = "x11"))]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use g2g_core::frame::Frame;
use g2g_core::runtime::{parse_launch, run_graph};
use g2g_core::PipelineClock;
use g2g_plugins::appsink::{register_appsink_pull, AppSinkPull};
use g2g_plugins::registry::default_registry;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    AtomEnum, ChangeWindowAttributesAux, ConnectionExt as _, CreateWindowAux, PropMode, Window,
    WindowClass,
};
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;
use x11rb::COPY_DEPTH_FROM_PARENT;

const SCREEN_WIDTH: u16 = 320;
const SCREEN_HEIGHT: u16 = 240;
const WINDOW_X: i16 = 40;
const WINDOW_Y: i16 = 30;
const WINDOW_WIDTH: u16 = 64;
const WINDOW_HEIGHT: u16 = 48;
const WINDOW_NAME: &str = "g2g ximagesrc test window";
// X TrueColor pixels at depth 24 are 0x00RRGGBB
const FIRST_COLOR: u32 = 0x0020_80C0;
const SECOND_COLOR: u32 = 0x00C0_4010;
// Xvfb -br paints the root window black
const ROOT_COLOR: u32 = 0;
const BYTES_PER_PIXEL: usize = 4;
const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
const DEFAULT_FPS: u64 = 25;
const TIMEOUT: Duration = Duration::from_secs(10);

struct ZeroClock;
impl PipelineClock for ZeroClock {
    fn now_ns(&self) -> u64 {
        0
    }
}

struct Xvfb {
    server: Child,
    display: String,
}

impl Xvfb {
    fn start() -> Self {
        let screen = format!("{SCREEN_WIDTH}x{SCREEN_HEIGHT}x24");
        // Xvfb writes the display number it picked to the -displayfd descriptor
        let mut server = Command::new("sh")
            .args([
                "-c",
                &format!(
                    "exec Xvfb -displayfd 3 -screen 0 {screen} -br -nolisten tcp 3>&1 >/dev/null 2>&1"
                ),
            ])
            .stdout(Stdio::piped())
            .spawn()
            .expect("starts Xvfb");
        let mut number = String::new();
        BufReader::new(server.stdout.take().expect("piped"))
            .read_line(&mut number)
            .expect("Xvfb reports its display");
        Self {
            server,
            display: format!(":{}", number.trim()),
        }
    }
}

impl Drop for Xvfb {
    fn drop(&mut self) {
        let _ = self.server.kill();
        let _ = self.server.wait();
    }
}

struct TestWindow {
    connection: RustConnection,
    window: Window,
}

impl TestWindow {
    fn map(display: &str) -> Self {
        let (connection, screen) = x11rb::connect(Some(display)).expect("connects to Xvfb");
        let root = connection.setup().roots[screen].root;
        let window = connection.generate_id().expect("window id");
        connection
            .create_window(
                COPY_DEPTH_FROM_PARENT,
                window,
                root,
                WINDOW_X,
                WINDOW_Y,
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                0,
                WindowClass::INPUT_OUTPUT,
                0,
                &CreateWindowAux::new().background_pixel(FIRST_COLOR),
            )
            .expect("creates the window");
        connection
            .change_property8(
                PropMode::REPLACE,
                window,
                AtomEnum::WM_NAME,
                AtomEnum::STRING,
                WINDOW_NAME.as_bytes(),
            )
            .expect("names the window");
        connection.map_window(window).expect("maps the window");
        let test_window = Self { connection, window };
        test_window.sync();
        test_window
    }

    fn sync(&self) {
        self.connection
            .get_input_focus()
            .expect("sends")
            .reply()
            .expect("the server answers");
    }

    fn repaint(&self, color: u32) {
        self.connection
            .change_window_attributes(
                self.window,
                &ChangeWindowAttributesAux::new().background_pixel(color),
            )
            .expect("changes the background");
        self.connection
            .clear_area(false, self.window, 0, 0, 0, 0)
            .expect("repaints the window");
        self.sync();
    }

    fn move_pointer(&self, x: i16, y: i16) {
        self.connection
            .warp_pointer(x11rb::NONE, self.window, 0, 0, 0, 0, x, y)
            .expect("moves the pointer");
        self.sync();
    }
}

fn bgr(color: u32) -> [u8; 3] {
    let [blue, green, red, _] = color.to_le_bytes();
    [blue, green, red]
}

fn pixel(picture: &[u8], width: usize, x: usize, y: usize) -> [u8; 3] {
    let offset = (y * width + x) * BYTES_PER_PIXEL;
    [picture[offset], picture[offset + 1], picture[offset + 2]]
}

fn picture(frame: &Frame) -> &[u8] {
    frame.domain.as_system_slice().expect("system memory")
}

fn launch(
    channel: &str,
    source: &str,
) -> (AppSinkPull, g2g_core::Graph<g2g_core::runtime::GraphNode>) {
    let pull = register_appsink_pull(channel);
    let graph = parse_launch(
        &default_registry(),
        &format!("{source} ! appsink channel={channel}"),
    )
    .expect("parses");
    (pull, graph)
}

async fn single_frame(channel: &str, source: &str) -> Frame {
    let (pull, graph) = launch(channel, source);
    tokio::time::timeout(TIMEOUT, run_graph(graph, &ZeroClock, 2))
        .await
        .expect("one frame ends the run")
        .expect("runs");
    pull.pull().await.expect("the frame reaches the sink")
}

#[tokio::test]
async fn the_screen_shows_a_window_and_damage_brings_its_change() {
    let server = Xvfb::start();
    let window = TestWindow::map(&server.display);
    let (pull, graph) = launch(
        "m2543_screen",
        &format!(
            "ximagesrc display-name={} show-pointer=false",
            server.display
        ),
    );
    let width = usize::from(SCREEN_WIDTH);
    let inside = (WINDOW_X as usize + 1, WINDOW_Y as usize + 1);
    let checks = async {
        let first = pull.pull().await.expect("the first frame");
        assert_eq!(
            picture(&first).len(),
            width * usize::from(SCREEN_HEIGHT) * BYTES_PER_PIXEL
        );
        assert_eq!(
            pixel(picture(&first), width, inside.0, inside.1),
            bgr(FIRST_COLOR)
        );
        assert_eq!(
            first.timing.duration_ns,
            NANOSECONDS_PER_SECOND / DEFAULT_FPS
        );
        window.repaint(SECOND_COLOR);
        loop {
            let frame = pull.pull().await.expect("the source keeps producing");
            if pixel(picture(&frame), width, inside.0, inside.1) == bgr(SECOND_COLOR) {
                return;
            }
        }
    };
    tokio::select! {
        result = run_graph(graph, &ZeroClock, 2) => panic!("the graph ended first: {result:?}"),
        shown = tokio::time::timeout(TIMEOUT, checks) => shown.expect("a frame shows the repainted window"),
    }
}

#[tokio::test]
async fn a_region_is_captured_at_the_rate_a_caps_filter_asks_for() {
    const MARGIN: usize = 8;
    const START_X: usize = WINDOW_X as usize - MARGIN;
    const START_Y: usize = WINDOW_Y as usize - MARGIN;
    const END_X: usize = START_X + 31;
    const END_Y: usize = START_Y + 23;
    const FPS: u64 = 10;
    let server = Xvfb::start();
    let _window = TestWindow::map(&server.display);
    let frame = single_frame(
        "m2543_region",
        &format!(
            "ximagesrc display-name={} startx={START_X} starty={START_Y} endx={END_X} \
             endy={END_Y} show-pointer=false num-buffers=1 ! video/x-raw,format=BGRx,framerate={FPS}/1",
            server.display
        ),
    )
    .await;
    let width = END_X - START_X + 1;
    let height = END_Y - START_Y + 1;
    let picture = picture(&frame);
    assert_eq!(picture.len(), width * height * BYTES_PER_PIXEL);
    assert_eq!(pixel(picture, width, MARGIN - 1, MARGIN), bgr(ROOT_COLOR));
    assert_eq!(pixel(picture, width, MARGIN, MARGIN - 1), bgr(ROOT_COLOR));
    assert_eq!(pixel(picture, width, MARGIN, MARGIN), bgr(FIRST_COLOR));
    assert_eq!(frame.timing.duration_ns, NANOSECONDS_PER_SECOND / FPS);
}

#[tokio::test]
async fn a_window_is_found_by_name_and_shown_with_the_pointer() {
    let server = Xvfb::start();
    let window = TestWindow::map(&server.display);
    let pointer = (WINDOW_WIDTH as i16 / 2, WINDOW_HEIGHT as i16 / 2);
    window.move_pointer(pointer.0, pointer.1);
    let source = |show_pointer: bool| {
        format!(
            "ximagesrc display-name={} xname=\"{WINDOW_NAME}\" show-pointer={show_pointer} num-buffers=1",
            server.display
        )
    };
    let width = usize::from(WINDOW_WIDTH);
    let without_pointer = single_frame("m2543_name", &source(false)).await;
    assert_eq!(
        picture(&without_pointer).len(),
        width * usize::from(WINDOW_HEIGHT) * BYTES_PER_PIXEL
    );
    assert!(picture(&without_pointer)
        .as_chunks::<BYTES_PER_PIXEL>()
        .0
        .iter()
        .all(|pixel| pixel[..3] == bgr(FIRST_COLOR)));
    let by_id = single_frame(
        "m2543_xid",
        &format!(
            "ximagesrc display-name={} xid={} show-pointer=false num-buffers=1",
            server.display, window.window
        ),
    )
    .await;
    assert_eq!(picture(&by_id), picture(&without_pointer));
    let with_pointer = single_frame("m2543_name_pointer", &source(true)).await;
    let (x, y) = (pointer.0 as usize, pointer.1 as usize);
    let around_pointer = (y.saturating_sub(4)..y + 4)
        .flat_map(|row| (x.saturating_sub(4)..x + 4).map(move |column| (column, row)));
    assert!(
        around_pointer.into_iter().any(|(column, row)| pixel(
            picture(&with_pointer),
            width,
            column,
            row
        ) != bgr(FIRST_COLOR)),
        "the pointer is drawn into the frame"
    );
}
