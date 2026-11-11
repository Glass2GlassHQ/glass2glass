// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use alloc::format;
use alloc::string::{String, ToString};

use md5::{Digest, Md5};

use crate::rtmp::PublishRefusal;
use crate::rtspclient::md5_hex;
use crate::textencoding::base64_encode;

const CONNECT_REJECTED: &str = "NetConnection.Connect.Rejected";
const LLNW_REALM: &str = "live";
const LLNW_NONCE_COUNT: &str = "00000001";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RtmpAuthmod {
    None,
    // take the mode the server names in its first refusal
    Auto,
    Adobe,
    Llnw,
}

impl RtmpAuthmod {
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "auto" => Some(Self::Auto),
            "adobe" => Some(Self::Adobe),
            "llnw" => Some(Self::Llnw),
            _ => None,
        }
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Auto => "auto",
            Self::Adobe => "adobe",
            Self::Llnw => "llnw",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AuthFailure {
    // a refusal that asks for no authentication this sink can give
    NotAChallenge,
    MissingCredentials,
    // the server's authmod differs from the configured one
    WrongAuthmod,
    // reason=authfailed, any reason other than needauth, or a second challenge after the answer
    Denied,
    // a needauth challenge without the adobe salt or the llnw nonce
    MalformedChallenge,
}

#[derive(Debug)]
pub(crate) struct RtmpAuth {
    authmod: RtmpAuthmod,
    username: String,
    password: String,
    answer: Option<String>,
}

impl RtmpAuth {
    pub(crate) fn new(authmod: RtmpAuthmod, username: &str, password: &str) -> Self {
        Self {
            authmod,
            username: username.to_string(),
            password: password.to_string(),
            answer: None,
        }
    }

    pub(crate) fn connect_query(&self) -> Result<Option<String>, AuthFailure> {
        if let Some(answer) = &self.answer {
            return Ok(Some(answer.clone()));
        }
        if matches!(self.authmod, RtmpAuthmod::None | RtmpAuthmod::Auto) {
            return Ok(None);
        }
        if self.username.is_empty() || self.password.is_empty() {
            return Err(AuthFailure::MissingCredentials);
        }
        Ok(Some(format!(
            "authmod={}&user={}",
            self.authmod.name(),
            self.username
        )))
    }

    pub(crate) fn answer(
        &mut self,
        refusal: &PublishRefusal,
        app: &str,
        client_nonce: u32,
    ) -> Result<(), AuthFailure> {
        if refusal.code != CONNECT_REJECTED {
            return Err(AuthFailure::NotAChallenge);
        }
        let description = refusal.description.as_str();
        if self.authmod == RtmpAuthmod::Auto && description.contains("code=403 need auth") {
            self.authmod = if description.contains("authmod=adobe") {
                RtmpAuthmod::Adobe
            } else if description.contains("authmod=llnw") {
                RtmpAuthmod::Llnw
            } else {
                return Err(AuthFailure::NotAChallenge);
            };
            return Ok(());
        }
        let (server_authmod, query) =
            access_manager_reject(description).ok_or(AuthFailure::NotAChallenge)?;
        let authmod_matches = matches!(
            (self.authmod, server_authmod),
            (RtmpAuthmod::Adobe, "adobe") | (RtmpAuthmod::Llnw, "llnw")
        );
        if !authmod_matches {
            return Err(AuthFailure::WrongAuthmod);
        }
        if query_value(query, "reason") != Some("needauth") || self.answer.is_some() {
            return Err(AuthFailure::Denied);
        }
        let client_nonce = format!("{client_nonce:08x}");
        let username = &self.username;
        let answer = if self.authmod == RtmpAuthmod::Adobe {
            let salt = query_value(query, "salt").ok_or(AuthFailure::MalformedChallenge)?;
            let opaque = query_value(query, "opaque");
            let server_challenge = opaque.or_else(|| query_value(query, "challenge"));
            let response = adobe_response(
                username,
                &self.password,
                salt,
                server_challenge,
                &client_nonce,
            );
            let answer = format!(
                "authmod=adobe&user={username}&challenge={client_nonce}&response={response}"
            );
            match opaque {
                Some(opaque) => format!("{answer}&opaque={opaque}"),
                None => answer,
            }
        } else {
            let nonce = query_value(query, "nonce").ok_or(AuthFailure::MalformedChallenge)?;
            let response = llnw_response(username, &self.password, app, nonce, &client_nonce);
            format!(
                "authmod=llnw&user={username}&nonce={nonce}&cnonce={client_nonce}&nc={LLNW_NONCE_COUNT}&response={response}"
            )
        };
        self.answer = Some(answer);
        Ok(())
    }
}

fn access_manager_reject(description: &str) -> Option<(&str, &str)> {
    // servers write "[ AccessManager.Reject ] : [ authmod=adobe ] : ?reason=needauth&..."
    let (_, mut rest) = description.split_once('[')?;
    for token in ["AccessManager.Reject", "]", ":", "[", "authmod="] {
        rest = rest.trim_start_matches(' ').strip_prefix(token)?;
    }
    let (authmod, mut query) = rest.split_once(']')?;
    for token in [":", "?"] {
        query = query.trim_start_matches(' ').strip_prefix(token)?;
    }
    Some((authmod.trim_end_matches(' '), query))
}

pub(crate) fn query_value<'a>(query: &'a str, key: &str) -> Option<&'a str> {
    query
        .split('&')
        .find_map(|pair| pair.strip_prefix(key)?.strip_prefix('='))
}

pub(crate) fn adobe_response(
    username: &str,
    password: &str,
    salt: &str,
    server_challenge: Option<&str>,
    client_challenge: &str,
) -> String {
    let salted = base64_encode(&Md5::digest(
        format!("{username}{salt}{password}").as_bytes(),
    ));
    let mut hash = Md5::new();
    hash.update(salted.as_bytes());
    if let Some(server_challenge) = server_challenge {
        hash.update(server_challenge.as_bytes());
    }
    hash.update(client_challenge.as_bytes());
    base64_encode(&hash.finalize())
}

pub(crate) fn llnw_response(
    username: &str,
    password: &str,
    app: &str,
    nonce: &str,
    client_nonce: &str,
) -> String {
    // limelight hashes a plain "appname" as "appname/_definst_"
    let app_instance = if app.contains('/') {
        app.to_string()
    } else {
        format!("{app}/_definst_")
    };
    let credentials = md5_hex(&format!("{username}:{LLNW_REALM}:{password}"));
    let method = md5_hex(&format!("publish:/{app_instance}"));
    md5_hex(&format!(
        "{credentials}:{nonce}:{LLNW_NONCE_COUNT}:{client_nonce}:auth:{method}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const USERNAME: &str = "user";
    const PASSWORD: &str = "secret";
    const CLIENT_NONCE: u32 = 0x1234_abcd;

    fn rejected(description: &str) -> PublishRefusal {
        PublishRefusal {
            code: CONNECT_REJECTED.to_string(),
            description: description.to_string(),
        }
    }

    #[test]
    fn adobe_response_matches_an_independent_md5() {
        // expected digests come from python hashlib
        assert_eq!(
            adobe_response(USERNAME, PASSWORD, "c2FsdA==", Some("b3BhcXVl"), "1234abcd"),
            "1D3rjeY+r9SOkHivdH0+SA=="
        );
        assert_eq!(
            adobe_response(USERNAME, PASSWORD, "c2FsdA==", Some("Y2hhbA=="), "1234abcd"),
            "wakav5P0VTQoy6v5rWTDEA=="
        );
    }

    #[test]
    fn llnw_response_matches_an_independent_md5() {
        assert_eq!(
            llnw_response(USERNAME, PASSWORD, "live", "abc123", "1234abcd"),
            "47d6b1c52e3a15a9007b97b43cf22639"
        );
        assert_eq!(
            llnw_response(USERNAME, PASSWORD, "live/sub", "abc123", "1234abcd"),
            "2a34fb8136b403f8ac99adde6b78a918"
        );
    }

    #[test]
    fn auto_follows_the_adobe_challenge_to_an_answer() {
        let mut auth = RtmpAuth::new(RtmpAuthmod::Auto, USERNAME, PASSWORD);
        assert_eq!(auth.connect_query(), Ok(None));
        auth.answer(
            &rejected("[ AccessManager.Reject ] : [ code=403 need auth; authmod=adobe ] : "),
            "live",
            CLIENT_NONCE,
        )
        .unwrap();
        assert_eq!(
            auth.connect_query().unwrap().as_deref(),
            Some("authmod=adobe&user=user")
        );
        auth.answer(
            &rejected("[ AccessManager.Reject ] : [ authmod=adobe ] : ?reason=needauth&user=user&salt=c2FsdA==&challenge=Y2hhbA==&opaque=b3BhcXVl"),
            "live",
            CLIENT_NONCE,
        )
        .unwrap();
        assert_eq!(
            auth.connect_query().unwrap().as_deref(),
            Some("authmod=adobe&user=user&challenge=1234abcd&response=1D3rjeY+r9SOkHivdH0+SA==&opaque=b3BhcXVl")
        );
        assert_eq!(
            auth.answer(
                &rejected("[ AccessManager.Reject ] : [ authmod=adobe ] : ?reason=needauth&salt=x"),
                "live",
                CLIENT_NONCE,
            ),
            Err(AuthFailure::Denied),
            "a second challenge after the answer ends the retries"
        );
    }

    #[test]
    fn llnw_answers_the_nonce() {
        let mut auth = RtmpAuth::new(RtmpAuthmod::Llnw, USERNAME, PASSWORD);
        assert_eq!(
            auth.connect_query().unwrap().as_deref(),
            Some("authmod=llnw&user=user")
        );
        auth.answer(
            &rejected("[AccessManager.Reject]:[authmod=llnw]:?reason=needauth&nonce=abc123"),
            "live",
            CLIENT_NONCE,
        )
        .unwrap();
        assert_eq!(
            auth.connect_query().unwrap().as_deref(),
            Some("authmod=llnw&user=user&nonce=abc123&cnonce=1234abcd&nc=00000001&response=47d6b1c52e3a15a9007b97b43cf22639")
        );
    }

    #[test]
    fn refusals_that_cannot_be_answered_fail() {
        let challenge =
            "[ AccessManager.Reject ] : [ authmod=adobe ] : ?reason=needauth&salt=c2FsdA==";
        let cases = [
            (
                RtmpAuthmod::Adobe,
                rejected("[ AccessManager.Reject ] : [ authmod=adobe ] : ?reason=authfailed"),
                AuthFailure::Denied,
            ),
            (
                RtmpAuthmod::Adobe,
                rejected("[ AccessManager.Reject ] : [ authmod=adobe ] : ?reason=nosuchuser"),
                AuthFailure::Denied,
            ),
            (
                RtmpAuthmod::Adobe,
                rejected("[ AccessManager.Reject ] : [ authmod=adobe ] : ?reason=needauth"),
                AuthFailure::MalformedChallenge,
            ),
            (
                RtmpAuthmod::Llnw,
                rejected(challenge),
                AuthFailure::WrongAuthmod,
            ),
            (
                RtmpAuthmod::None,
                rejected(challenge),
                AuthFailure::WrongAuthmod,
            ),
            (
                RtmpAuthmod::Auto,
                rejected("[ AccessManager.Reject ] : [ code=403 need auth; authmod=other ] : "),
                AuthFailure::NotAChallenge,
            ),
            (
                RtmpAuthmod::Adobe,
                rejected("Connection refused"),
                AuthFailure::NotAChallenge,
            ),
            (
                RtmpAuthmod::Adobe,
                PublishRefusal {
                    code: "NetStream.Publish.BadName".to_string(),
                    description: challenge.to_string(),
                },
                AuthFailure::NotAChallenge,
            ),
        ];
        for (authmod, refusal, failure) in cases {
            let mut auth = RtmpAuth::new(authmod, USERNAME, PASSWORD);
            assert_eq!(
                auth.answer(&refusal, "live", CLIENT_NONCE),
                Err(failure),
                "{refusal:?}"
            );
        }
        assert_eq!(
            RtmpAuth::new(RtmpAuthmod::Adobe, USERNAME, "").connect_query(),
            Err(AuthFailure::MissingCredentials)
        );
    }
}
