// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockDecrypt, BlockEncrypt, KeyInit};
use aes::Aes128;
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{UnparsedPublicKey, RSA_PKCS1_2048_8192_SHA256};
use sha2::{Digest, Sha256};
use tokio::sync::Notify;

use super::identity::{certificate_signature, HostIdentity};

const BLOCK: usize = 16;
const SECRET: usize = 16;
const HASH: usize = 32;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PairingError {
    NotPairing,
    OutOfOrder,
    BadLength,
    BadClientCertificate,
    HashMismatch,
    BadSignature,
    Random,
}

impl fmt::Display for PairingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NotPairing => "no pairing in progress for this client",
            Self::OutOfOrder => "pairing step arrived out of order",
            Self::BadLength => "pairing value has the wrong length",
            Self::BadClientCertificate => "client certificate does not parse",
            Self::HashMismatch => "client hash does not match, wrong PIN or a tampered exchange",
            Self::BadSignature => "client signature does not verify",
            Self::Random => "no random bytes from the system",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PairingError {}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PinError {
    NoneWaiting,
    SeveralWaiting,
    UnknownClient,
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::NoneWaiting => "no Moonlight client is waiting for a PIN",
            Self::SeveralWaiting => {
                "several Moonlight clients wait for a PIN, name one with uniqueid"
            }
            Self::UnknownClient => "no Moonlight client with that uniqueid waits for a PIN",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PinError {}

#[derive(Debug)]
enum Stage {
    WaitingForPin {
        salt: [u8; SECRET],
    },
    Keyed {
        key: [u8; BLOCK],
    },
    Challenged {
        key: [u8; BLOCK],
        server_secret: [u8; SECRET],
        server_challenge: [u8; SECRET],
    },
    Answered {
        server_challenge: [u8; SECRET],
        client_hash: Vec<u8>,
    },
}

#[derive(Debug)]
struct Pending {
    client_certificate_der: Vec<u8>,
    pin_entered: Arc<Notify>,
    stage: Stage,
}

#[derive(Debug, Default)]
pub(crate) struct Pairings {
    pending: Mutex<HashMap<String, Pending>>,
}

impl Pairings {
    pub(crate) fn begin(
        &self,
        unique_id: &str,
        client_certificate_der: Vec<u8>,
        salt: [u8; SECRET],
    ) -> Arc<Notify> {
        let pin_entered = Arc::new(Notify::new());
        self.pending.lock().unwrap().insert(
            unique_id.to_string(),
            Pending {
                client_certificate_der,
                pin_entered: Arc::clone(&pin_entered),
                stage: Stage::WaitingForPin { salt },
            },
        );
        pin_entered
    }

    pub(crate) fn has_key(&self, unique_id: &str) -> bool {
        matches!(
            self.pending.lock().unwrap().get(unique_id),
            Some(Pending {
                stage: Stage::Keyed { .. },
                ..
            })
        )
    }

    pub(crate) fn cancel(&self, unique_id: &str) {
        self.pending.lock().unwrap().remove(unique_id);
    }

    pub(crate) fn enter_pin(&self, pin: &str, unique_id: Option<&str>) -> Result<(), PinError> {
        let mut pending = self.pending.lock().unwrap();
        let mut waiting = pending
            .iter_mut()
            .filter(|(_, p)| matches!(p.stage, Stage::WaitingForPin { .. }));
        let entry = match unique_id {
            Some(id) => waiting
                .find(|(waiting_id, _)| *waiting_id == id)
                .ok_or(PinError::UnknownClient)?,
            None => {
                let first = waiting.next().ok_or(PinError::NoneWaiting)?;
                if waiting.next().is_some() {
                    return Err(PinError::SeveralWaiting);
                }
                first
            }
        };
        let Stage::WaitingForPin { salt } = entry.1.stage else {
            unreachable!("filtered to waiting clients");
        };
        entry.1.stage = Stage::Keyed {
            key: pin_key(&salt, pin),
        };
        entry.1.pin_entered.notify_one();
        Ok(())
    }

    pub(crate) fn client_challenge(
        &self,
        unique_id: &str,
        challenge: &[u8],
        host: &HostIdentity,
    ) -> Result<Vec<u8>, PairingError> {
        let mut pending = self.pending.lock().unwrap();
        let pending = pending.get_mut(unique_id).ok_or(PairingError::NotPairing)?;
        let Stage::Keyed { key } = pending.stage else {
            return Err(PairingError::OutOfOrder);
        };
        if challenge.len() != BLOCK {
            return Err(PairingError::BadLength);
        }
        let server_secret = random()?;
        let server_challenge = random()?;
        let mut hashed = ecb_decrypt(&key, challenge);
        hashed.extend_from_slice(&host.certificate_signature);
        hashed.extend_from_slice(&server_secret);
        let mut response = Sha256::digest(&hashed).to_vec();
        response.extend_from_slice(&server_challenge);
        pending.stage = Stage::Challenged {
            key,
            server_secret,
            server_challenge,
        };
        Ok(ecb_encrypt(&key, &response))
    }

    pub(crate) fn server_challenge_response(
        &self,
        unique_id: &str,
        response: &[u8],
        host: &HostIdentity,
    ) -> Result<Vec<u8>, PairingError> {
        let mut pending = self.pending.lock().unwrap();
        let pending = pending.get_mut(unique_id).ok_or(PairingError::NotPairing)?;
        let Stage::Challenged {
            key,
            server_secret,
            server_challenge,
        } = pending.stage
        else {
            return Err(PairingError::OutOfOrder);
        };
        if response.len() < HASH || !response.len().is_multiple_of(BLOCK) {
            return Err(PairingError::BadLength);
        }
        let mut client_hash = ecb_decrypt(&key, response);
        client_hash.truncate(HASH);
        let mut pairing_secret = server_secret.to_vec();
        pairing_secret.extend(
            host.sign(&server_secret)
                .map_err(|_| PairingError::Random)?,
        );
        pending.stage = Stage::Answered {
            server_challenge,
            client_hash,
        };
        Ok(pairing_secret)
    }

    // returns the client certificate to remember once the client proves it holds the PIN and its key
    pub(crate) fn client_pairing_secret(
        &self,
        unique_id: &str,
        pairing_secret: &[u8],
    ) -> Result<Vec<u8>, PairingError> {
        let pending = self
            .pending
            .lock()
            .unwrap()
            .remove(unique_id)
            .ok_or(PairingError::NotPairing)?;
        let Stage::Answered {
            server_challenge,
            client_hash,
        } = pending.stage
        else {
            return Err(PairingError::OutOfOrder);
        };
        if pairing_secret.len() <= SECRET {
            return Err(PairingError::BadLength);
        }
        let (client_secret, signature) = pairing_secret.split_at(SECRET);
        let certificate = &pending.client_certificate_der;
        let mut hashed = server_challenge.to_vec();
        hashed
            .extend(certificate_signature(certificate).ok_or(PairingError::BadClientCertificate)?);
        hashed.extend_from_slice(client_secret);
        if Sha256::digest(&hashed).as_slice() != client_hash.as_slice() {
            return Err(PairingError::HashMismatch);
        }
        let (_, parsed) = x509_parser::parse_x509_certificate(certificate)
            .map_err(|_| PairingError::BadClientCertificate)?;
        UnparsedPublicKey::new(
            &RSA_PKCS1_2048_8192_SHA256,
            &parsed.public_key().subject_public_key.data,
        )
        .verify(client_secret, signature)
        .map_err(|_| PairingError::BadSignature)?;
        Ok(pending.client_certificate_der)
    }
}

// moonlight derives the AES key from the first 16 bytes of SHA-256(salt || PIN)
fn pin_key(salt: &[u8; SECRET], pin: &str) -> [u8; BLOCK] {
    let mut hasher = Sha256::new();
    hasher.update(salt);
    hasher.update(pin.as_bytes());
    let mut key = [0u8; BLOCK];
    key.copy_from_slice(&hasher.finalize()[..BLOCK]);
    key
}

fn random() -> Result<[u8; SECRET], PairingError> {
    let mut bytes = [0u8; SECRET];
    SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| PairingError::Random)?;
    Ok(bytes)
}

// moonlight pads every pairing value to whole AES blocks and uses ECB without padding
fn ecb_encrypt(key: &[u8; BLOCK], data: &[u8]) -> Vec<u8> {
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let mut out = data.to_vec();
    for block in out.as_chunks_mut::<BLOCK>().0 {
        cipher.encrypt_block(GenericArray::from_mut_slice(block));
    }
    out
}

fn ecb_decrypt(key: &[u8; BLOCK], data: &[u8]) -> Vec<u8> {
    let cipher = Aes128::new(GenericArray::from_slice(key));
    let mut out = data.to_vec();
    for block in out.as_chunks_mut::<BLOCK>().0 {
        cipher.decrypt_block(GenericArray::from_mut_slice(block));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{RsaKeyPair, RSA_PKCS1_SHA256};
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::CertificateDer;

    const HOST_KEY: &[u8] = include_bytes!("testdata/host-key.der");
    const HOST_CERTIFICATE: &str = include_str!("testdata/host-certificate.pem");
    const CLIENT_KEY: &[u8] = include_bytes!("testdata/client-key.der");
    const CLIENT_CERTIFICATE: &str = include_str!("testdata/client-certificate.pem");
    const UNIQUE_ID: &str = "0123456789ABCDEF";
    const SALT: [u8; SECRET] = [7; SECRET];
    const HOST_PIN: &str = "4721";
    const CLIENT_CHALLENGE: [u8; SECRET] = [1; SECRET];
    const CLIENT_SECRET: [u8; SECRET] = [2; SECRET];

    fn host() -> HostIdentity {
        HostIdentity::from_parts(
            HOST_KEY.to_vec(),
            HOST_CERTIFICATE.to_string(),
            "host".to_string(),
        )
        .expect("host fixture")
    }

    fn client_certificate() -> Vec<u8> {
        CertificateDer::from_pem_slice(CLIENT_CERTIFICATE.as_bytes())
            .expect("client fixture")
            .to_vec()
    }

    struct ClientEnd {
        // what moonlight's own check of the host's challenge response compares
        host_response_matches: bool,
        result: Result<Vec<u8>, PairingError>,
    }

    // the client side of moonlight's pairing exchange, with the PIN the person typed
    fn pair_as_client(pairings: &Pairings, host: &HostIdentity, client_pin: &str) -> ClientEnd {
        let key = pin_key(&SALT, client_pin);
        let response = pairings
            .client_challenge(UNIQUE_ID, &ecb_encrypt(&key, &CLIENT_CHALLENGE), host)
            .expect("challenge answered");
        let response = ecb_decrypt(&key, &response);
        let (host_hash, server_challenge) = response.split_at(HASH);

        let certificate = client_certificate();
        let mut client_hash = server_challenge.to_vec();
        client_hash.extend(certificate_signature(&certificate).expect("client signature"));
        client_hash.extend_from_slice(&CLIENT_SECRET);
        let client_hash = Sha256::digest(&client_hash);
        let pairing_secret = pairings
            .server_challenge_response(UNIQUE_ID, &ecb_encrypt(&key, &client_hash), host)
            .expect("pairing secret");
        let (server_secret, server_signature) = pairing_secret.split_at(SECRET);
        let (_, host_certificate) =
            x509_parser::parse_x509_certificate(&host.certificate_der).expect("host certificate");
        UnparsedPublicKey::new(
            &RSA_PKCS1_2048_8192_SHA256,
            &host_certificate.public_key().subject_public_key.data,
        )
        .verify(server_secret, server_signature)
        .expect("the host signs its secret");
        let mut expected = CLIENT_CHALLENGE.to_vec();
        expected.extend_from_slice(&host.certificate_signature);
        expected.extend_from_slice(server_secret);

        let client_key = RsaKeyPair::from_pkcs8(CLIENT_KEY).expect("client key");
        let mut signature = vec![0; client_key.public().modulus_len()];
        client_key
            .sign(
                &RSA_PKCS1_SHA256,
                &SystemRandom::new(),
                &CLIENT_SECRET,
                &mut signature,
            )
            .expect("signed");
        let mut client_pairing_secret = CLIENT_SECRET.to_vec();
        client_pairing_secret.extend(signature);
        ClientEnd {
            host_response_matches: Sha256::digest(&expected).as_slice() == host_hash,
            result: pairings.client_pairing_secret(UNIQUE_ID, &client_pairing_secret),
        }
    }

    #[test]
    fn the_right_pin_pairs_the_client_certificate() {
        let (pairings, host) = (Pairings::default(), host());
        pairings.begin(UNIQUE_ID, client_certificate(), SALT);
        pairings.enter_pin(HOST_PIN, None).expect("a client waits");
        let end = pair_as_client(&pairings, &host, HOST_PIN);
        assert!(end.host_response_matches);
        assert_eq!(end.result, Ok(client_certificate()));
    }

    #[test]
    fn a_wrong_pin_fails_on_both_sides() {
        let (pairings, host) = (Pairings::default(), host());
        pairings.begin(UNIQUE_ID, client_certificate(), SALT);
        pairings.enter_pin(HOST_PIN, None).expect("a client waits");
        let end = pair_as_client(&pairings, &host, "0000");
        assert!(!end.host_response_matches);
        assert_eq!(end.result, Err(PairingError::HashMismatch));
    }

    #[test]
    fn a_challenge_before_the_pin_is_refused() {
        let (pairings, host) = (Pairings::default(), host());
        pairings.begin(UNIQUE_ID, client_certificate(), SALT);
        assert_eq!(
            pairings.client_challenge(UNIQUE_ID, &[0; BLOCK], &host),
            Err(PairingError::OutOfOrder)
        );
        assert_eq!(
            pairings.enter_pin(HOST_PIN, Some("other")),
            Err(PinError::UnknownClient)
        );
    }
}
