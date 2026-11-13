// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

/// The largest single message str0m accepts on a data channel (sctp-proto's
/// default max receive message size, 64 KiB). SCTP fragments and reassembles up
/// to this; a larger payload is rejected on write.
pub const MAX_MESSAGE_SIZE: usize = 65536;
