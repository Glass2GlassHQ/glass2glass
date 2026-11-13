// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

use crate::signaling::{ClientEvent, ClientSignaling, SignalingError};

const SIGNALING_READ_BYTES: usize = 4096;

#[derive(Debug)]
pub enum ExchangeError {
    Signaling(SignalingError),
    Io(std::io::Error),
    SessionGone,
}

impl core::fmt::Display for ExchangeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ExchangeError::Signaling(e) => write!(f, "signaling: {e}"),
            ExchangeError::Io(e) => write!(f, "signaling connection: {e}"),
            ExchangeError::SessionGone => write!(f, "the session ended before the SDP exchange"),
        }
    }
}

impl std::error::Error for ExchangeError {}

impl From<SignalingError> for ExchangeError {
    fn from(e: SignalingError) -> Self {
        ExchangeError::Signaling(e)
    }
}

impl From<std::io::Error> for ExchangeError {
    fn from(e: std::io::Error) -> Self {
        ExchangeError::Io(e)
    }
}

async fn next_event(
    stream: &mut TcpStream,
    signaling: &mut ClientSignaling,
) -> Result<ClientEvent, ExchangeError> {
    let mut buffer = [0u8; SIGNALING_READ_BYTES];
    loop {
        let read = stream.read(&mut buffer).await?;
        if read == 0 {
            return Err(ExchangeError::Signaling(SignalingError::HungUp));
        }
        if let Some(event) = signaling.receive(&buffer[..read])?.into_iter().next() {
            return Ok(event);
        }
    }
}

pub async fn exchange_sdp(
    mut stream: TcpStream,
    token: &str,
    mut offers: mpsc::Receiver<String>,
    answers: mpsc::Sender<String>,
) -> Result<(), ExchangeError> {
    let (mut signaling, framed_token) = ClientSignaling::new(token)?;
    stream.write_all(&framed_token).await?;
    if next_event(&mut stream, &mut signaling).await? != ClientEvent::Accepted {
        return Err(ExchangeError::Signaling(SignalingError::UnexpectedMessage));
    }
    let offer = offers.recv().await.ok_or(ExchangeError::SessionGone)?;
    stream.write_all(&signaling.offer(&offer)?).await?;
    let ClientEvent::Answer(answer) = next_event(&mut stream, &mut signaling).await? else {
        return Err(ExchangeError::Signaling(SignalingError::UnexpectedMessage));
    };
    answers
        .send(answer)
        .await
        .map_err(|_| ExchangeError::SessionGone)?;
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).await?;
    signaling.receive(&rest)?;
    Ok(signaling.connection_closed()?)
}
