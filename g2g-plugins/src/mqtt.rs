//! What `mqttsink` and `mqttsrc` share: the broker connection settings and
//! their properties, the QoS level mapping, and the reconnect delay.

use core::time::Duration;

use alloc::string::{String, ToString};

use rumqttc::{MqttOptions, QoS, TlsConfiguration, Transport};

use g2g_core::{PropError, PropValue};

pub(crate) const DEFAULT_HOST: &str = "localhost";
pub(crate) const DEFAULT_PORT: u16 = 1883;
pub(crate) const DEFAULT_PORT_TEXT: &str = "1883";
pub(crate) const DEFAULT_QOS: QoS = QoS::AtLeastOnce;
pub(crate) const DEFAULT_QOS_TEXT: &str = "1";
pub(crate) const QOS_MAX_TEXT: &str = "2";
pub(crate) const DEFAULT_KEEP_ALIVE_SECONDS: u64 = 30;
pub(crate) const DEFAULT_KEEP_ALIVE_TEXT: &str = "30";

/// How long an element waits after a failed connection before it retries.
pub(crate) const RECONNECT_DELAY: Duration = Duration::from_secs(2);

pub(crate) fn qos_from_level(level: u64) -> Option<QoS> {
    match level {
        0 => Some(QoS::AtMostOnce),
        1 => Some(QoS::AtLeastOnce),
        2 => Some(QoS::ExactlyOnce),
        _ => None,
    }
}

pub(crate) fn qos_level(qos: QoS) -> u64 {
    match qos {
        QoS::AtMostOnce => 0,
        QoS::AtLeastOnce => 1,
        QoS::ExactlyOnce => 2,
    }
}

/// The broker an element connects to and how: the `host`, `port`, `client-id`,
/// `username`, `password`, `tls` and `keep-alive` properties.
#[derive(Debug)]
pub(crate) struct BrokerSettings {
    pub host: String,
    pub port: u16,
    pub client_id: String,
    pub username: String,
    pub password: String,
    pub tls: bool,
    pub keep_alive_seconds: u64,
}

impl BrokerSettings {
    pub(crate) fn new() -> Self {
        Self {
            host: String::from(DEFAULT_HOST),
            port: DEFAULT_PORT,
            client_id: String::new(),
            username: String::new(),
            password: String::new(),
            tls: false,
            keep_alive_seconds: DEFAULT_KEEP_ALIVE_SECONDS,
        }
    }

    /// The connection options, with `fallback_client_id` standing in for an
    /// unset `client-id`.
    pub(crate) fn mqtt_options(&self, fallback_client_id: &str) -> MqttOptions {
        let client_id = if self.client_id.is_empty() {
            fallback_client_id
        } else {
            &self.client_id
        };
        let mut options = MqttOptions::new(client_id, self.host.clone(), self.port);
        options.set_keep_alive(Duration::from_secs(self.keep_alive_seconds));
        if !self.username.is_empty() {
            options.set_credentials(self.username.clone(), self.password.clone());
        }
        if self.tls {
            options.set_transport(Transport::tls_with_config(TlsConfiguration::Native));
        }
        options
    }

    /// Apply a broker property; `None` when `name` is not one of them.
    pub(crate) fn set_property(
        &mut self,
        name: &str,
        value: &PropValue,
    ) -> Option<Result<(), PropError>> {
        let text = || value.as_str().map(str::to_string).ok_or(PropError::Type);
        let result = match name {
            "host" => text().map(|host| self.host = host),
            "port" => value
                .as_uint()
                .ok_or(PropError::Type)
                .and_then(|port| u16::try_from(port).map_err(|_| PropError::Value))
                .map(|port| self.port = port),
            "client-id" => text().map(|client_id| self.client_id = client_id),
            "username" => text().map(|username| self.username = username),
            "password" => text().map(|password| self.password = password),
            "tls" => value
                .as_bool()
                .ok_or(PropError::Type)
                .map(|tls| self.tls = tls),
            "keep-alive" => value
                .as_uint()
                .ok_or(PropError::Type)
                .map(|seconds| self.keep_alive_seconds = seconds),
            _ => return None,
        };
        Some(result)
    }

    /// Read a broker property; `None` when `name` is not one of them.
    pub(crate) fn get_property(&self, name: &str) -> Option<PropValue> {
        match name {
            "host" => Some(PropValue::Str(self.host.clone())),
            "port" => Some(PropValue::Uint(u64::from(self.port))),
            "client-id" => Some(PropValue::Str(self.client_id.clone())),
            "username" => Some(PropValue::Str(self.username.clone())),
            "password" => Some(PropValue::Str(self.password.clone())),
            "tls" => Some(PropValue::Bool(self.tls)),
            "keep-alive" => Some(PropValue::Uint(self.keep_alive_seconds)),
            _ => None,
        }
    }
}

/// A property table holding the broker properties `BrokerSettings` handles,
/// followed by the element's own entries.
macro_rules! broker_property_table {
    ($($own:expr),* $(,)?) => {
        &[
            g2g_core::PropertySpec::new("host", g2g_core::PropKind::Str, "broker host name")
                .with_default($crate::mqtt::DEFAULT_HOST),
            g2g_core::PropertySpec::new("port", g2g_core::PropKind::Uint, "broker port")
                .with_default($crate::mqtt::DEFAULT_PORT_TEXT),
            g2g_core::PropertySpec::new(
                "client-id",
                g2g_core::PropKind::Str,
                "client id sent to the broker, the element's instance name when empty",
            )
            .with_default(""),
            g2g_core::PropertySpec::new(
                "username",
                g2g_core::PropKind::Str,
                "broker login, none when empty",
            )
            .with_default(""),
            g2g_core::PropertySpec::new("password", g2g_core::PropKind::Str, "broker password")
                .with_default(""),
            g2g_core::PropertySpec::new("tls", g2g_core::PropKind::Bool, "connect over TLS")
                .with_default("false"),
            g2g_core::PropertySpec::new(
                "keep-alive",
                g2g_core::PropKind::Uint,
                "seconds between keep-alive pings to the broker",
            )
            .with_default($crate::mqtt::DEFAULT_KEEP_ALIVE_TEXT),
            $($own),*
        ]
    };
}
pub(crate) use broker_property_table;
