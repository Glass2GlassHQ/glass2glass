// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

/// The largest single message str0m accepts on a data channel (sctp-proto's
/// default max receive message size, 64 KiB). SCTP fragments and reassembles up
/// to this; a larger payload is rejected on write.
pub const MAX_MESSAGE_SIZE: usize = 65536;

#[cfg(feature = "std")]
pub use ends::DataChannelEnds;
#[cfg(any(feature = "webrtc", feature = "webtransport"))]
pub(crate) use ends::{data_channel, DataChannelTransportEnds};

#[cfg(feature = "std")]
mod ends {
    use alloc::vec::Vec;

    use tokio::sync::mpsc;

    #[derive(Debug)]
    pub struct DataChannelEnds {
        pub outgoing: mpsc::UnboundedSender<Vec<u8>>,
        pub incoming: mpsc::UnboundedReceiver<Vec<u8>>,
    }

    #[cfg(any(feature = "webrtc", feature = "webtransport"))]
    #[derive(Debug)]
    pub(crate) struct DataChannelTransportEnds {
        pub(crate) outgoing: mpsc::UnboundedReceiver<Vec<u8>>,
        pub(crate) incoming: mpsc::UnboundedSender<Vec<u8>>,
    }

    #[cfg(any(feature = "webrtc", feature = "webtransport"))]
    pub(crate) fn data_channel() -> (DataChannelEnds, DataChannelTransportEnds) {
        let (outgoing, outgoing_receiver) = mpsc::unbounded_channel();
        let (incoming_sender, incoming) = mpsc::unbounded_channel();
        (
            DataChannelEnds { outgoing, incoming },
            DataChannelTransportEnds {
                outgoing: outgoing_receiver,
                incoming: incoming_sender,
            },
        )
    }
}
