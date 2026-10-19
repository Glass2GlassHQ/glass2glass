#![cfg(all(target_os = "linux", feature = "decklink"))]

use g2g_core::{G2gError, HardwareError};
use g2g_plugins::decklink::sdk_opener;

#[path = "common/decklink_board.rs"]
mod decklink_board;
use decklink_board::driver_installed;

#[test]
fn a_missing_driver_library_is_reported_by_name() {
    if driver_installed() {
        eprintln!("skip m1640: the driver library is installed on this host");
        return;
    }
    let open = sdk_opener();
    assert_eq!(
        open(0).err(),
        Some(G2gError::Hardware(HardwareError::LibraryNotFound(
            "libDeckLinkAPI.so"
        )))
    );
}
