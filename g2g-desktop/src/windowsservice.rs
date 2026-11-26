// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::iter::repeat_n;
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::time::Duration;

use g2g_plugins::remotewtio::{certificate_from_pem, generate_certificate};
use g2g_plugins::textencoding::hex_encode;
use windows::core::{HSTRING, PCWSTR, PWSTR};
use windows::Win32::Foundation::{
    CloseHandle, LocalFree, SetHandleInformation, HANDLE, HANDLE_FLAG_INHERIT, HLOCAL,
    WAIT_OBJECT_0,
};
use windows::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows::Win32::Security::Cryptography::{BCryptGenRandom, BCRYPT_USE_SYSTEM_PREFERRED_RNG};
use windows::Win32::Security::{
    DuplicateTokenEx, SecurityImpersonation, SetFileSecurityW, SetTokenInformation, TokenPrimary,
    TokenSessionId, DACL_SECURITY_INFORMATION, PROTECTED_DACL_SECURITY_INFORMATION,
    PSECURITY_DESCRIPTOR, TOKEN_ALL_ACCESS, TOKEN_DUPLICATE,
};
use windows::Win32::System::RemoteDesktop::WTSGetActiveConsoleSessionId;
use windows::Win32::System::Threading::{
    CreateProcessAsUserW, GetCurrentProcess, GetExitCodeProcess, OpenProcessToken,
    TerminateProcess, WaitForSingleObject, CREATE_NO_WINDOW, PROCESS_INFORMATION,
    STARTF_USESTDHANDLES, STARTUPINFOW,
};
use windows_service::service::{
    ServiceAccess, ServiceControl, ServiceControlAccept, ServiceErrorControl, ServiceExitCode,
    ServiceInfo, ServiceStartType, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
use windows_service::service_dispatcher;
use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

use crate::host::RemotewtIdentity;

const SERVICE_NAME: &str = "g2g-desktop";
const SERVICE_DISPLAY_NAME: &str = "g2g-desktop remote desktop host";
const SERVICE_RUN_ARGUMENTS: [&str; 2] = ["service", "run"];
const REMOTEWT_DIRECTORY: &str = "remotewt";
const CERTIFICATE_FILE: &str = "certificate.pem";
const KEY_FILE: &str = "key.pem";
const TOKEN_FILE: &str = "token";
const TOKEN_BYTES: usize = 16;
const SERVICE_LOG_FILE: &str = "service.log";
const HOST_LOG_FILE: &str = "host.log";

pub fn state_directory() -> Result<PathBuf, Box<dyn Error>> {
    let program_data =
        std::env::var_os("ProgramData").ok_or("the ProgramData variable is not set")?;
    Ok(PathBuf::from(program_data).join(SERVICE_NAME))
}

fn create_state_directory(directory: &Path) -> Result<(), Box<dyn Error>> {
    // SDDL: a DACL that ignores inherited entries and gives SYSTEM and Administrators full control
    const SYSTEM_AND_ADMINISTRATORS_ONLY: &str = "D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)";
    fs::create_dir_all(directory)?;
    let mut descriptor = PSECURITY_DESCRIPTOR::default();
    // SAFETY: the out-param is a valid local, and the descriptor it receives is freed below.
    unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            &HSTRING::from(SYSTEM_AND_ADMINISTRATORS_ONLY),
            SDDL_REVISION_1,
            &mut descriptor,
            None,
        )?;
        let applied = SetFileSecurityW(
            &HSTRING::from(directory.as_os_str()),
            DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
            descriptor,
        );
        LocalFree(Some(HLOCAL(descriptor.0)));
        applied.ok()?;
    }
    Ok(())
}

fn random_token() -> Result<String, Box<dyn Error>> {
    let mut bytes = [0u8; TOKEN_BYTES];
    // SAFETY: the buffer is a valid local slice.
    unsafe { BCryptGenRandom(None, &mut bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG) }.ok()?;
    Ok(hex_encode(&bytes))
}

pub fn load_or_create_remotewt_identity(
    state_directory: &Path,
) -> Result<RemotewtIdentity, Box<dyn Error>> {
    let directory = state_directory.join(REMOTEWT_DIRECTORY);
    if !directory.join(TOKEN_FILE).exists() {
        fs::create_dir_all(&directory)?;
        let certificate = generate_certificate()?;
        fs::write(
            directory.join(CERTIFICATE_FILE),
            &certificate.certificate_pem,
        )?;
        fs::write(directory.join(KEY_FILE), &certificate.private_key_pem)?;
        fs::write(directory.join(TOKEN_FILE), random_token()?)?;
    }
    let certificate = certificate_from_pem(
        fs::read_to_string(directory.join(CERTIFICATE_FILE))?,
        fs::read_to_string(directory.join(KEY_FILE))?,
    )
    .map_err(|e| format!("{}: {e:?}", directory.display()))?;
    Ok(RemotewtIdentity {
        certificate,
        token: fs::read_to_string(directory.join(TOKEN_FILE))?
            .trim()
            .to_string(),
    })
}

pub fn install(host_arguments: &[String], remotewt: bool) -> Result<(), Box<dyn Error>> {
    let directory = state_directory()?;
    create_state_directory(&directory)?;
    if remotewt {
        let identity = load_or_create_remotewt_identity(&directory)?;
        println!("certificate sha256 {}", identity.certificate.sha256_hex);
        println!("token {}", identity.token);
    }
    let manager = ServiceManager::local_computer(
        None::<&str>,
        ServiceManagerAccess::CONNECT | ServiceManagerAccess::CREATE_SERVICE,
    )?;
    let launch_arguments = SERVICE_RUN_ARGUMENTS
        .iter()
        .map(OsString::from)
        .chain(host_arguments.iter().map(OsString::from))
        .collect();
    let service = manager.create_service(
        &ServiceInfo {
            name: SERVICE_NAME.into(),
            display_name: SERVICE_DISPLAY_NAME.into(),
            service_type: ServiceType::OWN_PROCESS,
            start_type: ServiceStartType::AutoStart,
            error_control: ServiceErrorControl::Normal,
            executable_path: std::env::current_exe()?,
            launch_arguments,
            dependencies: Vec::new(),
            account_name: None,
            account_password: None,
        },
        ServiceAccess::START,
    )?;
    service.start::<&OsStr>(&[])?;
    println!(
        "g2g-desktop: service {SERVICE_NAME} started, logs in {}",
        directory.display()
    );
    Ok(())
}

pub fn uninstall() -> Result<(), Box<dyn Error>> {
    let manager = ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CONNECT)?;
    let service = manager.open_service(
        SERVICE_NAME,
        ServiceAccess::QUERY_STATUS | ServiceAccess::STOP | ServiceAccess::DELETE,
    )?;
    if service.query_status()?.current_state != ServiceState::Stopped {
        service.stop()?;
    }
    service.delete()?;
    println!("g2g-desktop: service {SERVICE_NAME} removed");
    Ok(())
}

windows_service::define_windows_service!(ffi_service_main, service_main);

pub fn run() -> Result<(), Box<dyn Error>> {
    Ok(service_dispatcher::start(SERVICE_NAME, ffi_service_main)?)
}

fn append_service_log(directory: &Path, message: &str) {
    let opened = OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join(SERVICE_LOG_FILE));
    // the service has no console to report a failed log write to
    if let Ok(mut log) = opened {
        let _ = writeln!(log, "{message}");
    }
}

fn service_main(_start_arguments: Vec<OsString>) {
    let Ok(directory) = state_directory() else {
        return;
    };
    if let Err(e) = run_service(&directory) {
        append_service_log(&directory, &format!("service failed: {e}"));
    }
}

#[derive(Debug)]
enum ServiceEvent {
    Stop,
    SessionChange,
}

fn service_status(state: ServiceState, exit_code: u32) -> ServiceStatus {
    ServiceStatus {
        service_type: ServiceType::OWN_PROCESS,
        current_state: state,
        controls_accepted: match state {
            ServiceState::Running => {
                ServiceControlAccept::STOP | ServiceControlAccept::SESSION_CHANGE
            }
            _ => ServiceControlAccept::empty(),
        },
        exit_code: ServiceExitCode::Win32(exit_code),
        checkpoint: 0,
        wait_hint: Duration::ZERO,
        process_id: None,
    }
}

fn run_service(directory: &Path) -> Result<(), Box<dyn Error>> {
    let (events, received) = mpsc::channel();
    let status = service_control_handler::register(SERVICE_NAME, move |control| {
        let event = match control {
            ServiceControl::Stop => ServiceEvent::Stop,
            ServiceControl::SessionChange(_) => ServiceEvent::SessionChange,
            ServiceControl::Interrogate => return ServiceControlHandlerResult::NoError,
            _ => return ServiceControlHandlerResult::NotImplemented,
        };
        let _ = events.send(event);
        ServiceControlHandlerResult::NoError
    })?;
    status.set_service_status(service_status(ServiceState::Running, 0))?;
    // the service control manager starts the service as `g2g-desktop service run HOST-FLAGS...`
    let host_arguments: Vec<String> = std::env::args()
        .skip(1 + SERVICE_RUN_ARGUMENTS.len())
        .collect();
    let supervised = supervise_host(&received, directory, &host_arguments);
    let exit_code = match &supervised {
        Ok(()) => 0,
        Err(_) => 1,
    };
    status.set_service_status(service_status(ServiceState::Stopped, exit_code))?;
    supervised
}

fn quote_argument(argument: &str) -> String {
    // CommandLineToArgvW and the msvc runtime split a command line with these quoting rules
    if !argument.is_empty() && !argument.contains([' ', '\t', '\n', '\u{b}', '"']) {
        return argument.to_string();
    }
    let mut quoted = String::from('"');
    let mut backslashes = 0;
    for character in argument.chars() {
        match character {
            '\\' => backslashes += 1,
            '"' => {
                quoted.extend(repeat_n('\\', backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.extend(repeat_n('\\', backslashes));
                quoted.push(character);
                backslashes = 0;
            }
        }
    }
    quoted.extend(repeat_n('\\', backslashes * 2));
    quoted.push('"');
    quoted
}

fn host_command_line(
    directory: &Path,
    host_arguments: &[String],
) -> Result<String, Box<dyn Error>> {
    let executable = std::env::current_exe()?;
    let directory = directory
        .to_str()
        .ok_or("the state directory is not UTF-8")?;
    let executable = executable
        .to_str()
        .ok_or("the executable path is not UTF-8")?;
    let arguments = [executable, "host", "--state-dir", directory]
        .into_iter()
        .chain(host_arguments.iter().map(String::as_str))
        .map(quote_argument)
        .collect::<Vec<_>>();
    Ok(arguments.join(" "))
}

const NO_CONSOLE_SESSION: u32 = u32::MAX;
const HOST_POLL_INTERVAL: Duration = Duration::from_millis(500);
const RESTART_DELAY: Duration = Duration::from_secs(5);
const TERMINATE_WAIT_MS: u32 = 5000;

#[derive(Debug)]
struct HostProcess {
    process: HANDLE,
    session: u32,
}

impl Drop for HostProcess {
    fn drop(&mut self) {
        // SAFETY: the handle came from CreateProcessAsUserW and is closed once.
        let _ = unsafe { CloseHandle(self.process) };
    }
}

impl HostProcess {
    fn launch(session: u32, command_line: &str, log: &File) -> Result<Self, Box<dyn Error>> {
        let log_handle = HANDLE(log.as_raw_handle());
        let mut command_line: Vec<u16> = command_line.encode_utf16().chain([0]).collect();
        let mut desktop: Vec<u16> = "winsta0\\default".encode_utf16().chain([0]).collect();
        let mut process_token = HANDLE::default();
        let mut token = HANDLE::default();
        let mut process = PROCESS_INFORMATION::default();
        // SAFETY: every out-param is a valid local, the wide strings stay alive across the call, and each handle opened here is closed below.
        unsafe {
            SetHandleInformation(log_handle, HANDLE_FLAG_INHERIT.0, HANDLE_FLAG_INHERIT)?;
            OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE, &mut process_token)?;
            let duplicated = DuplicateTokenEx(
                process_token,
                TOKEN_ALL_ACCESS,
                None,
                SecurityImpersonation,
                TokenPrimary,
                &mut token,
            );
            CloseHandle(process_token)?;
            duplicated?;
            let startup = STARTUPINFOW {
                cb: size_of::<STARTUPINFOW>() as u32,
                lpDesktop: PWSTR(desktop.as_mut_ptr()),
                dwFlags: STARTF_USESTDHANDLES,
                hStdOutput: log_handle,
                hStdError: log_handle,
                ..Default::default()
            };
            // processes with a SYSTEM token in the console session can capture and inject on the secure desktop
            let launched = SetTokenInformation(
                token,
                TokenSessionId,
                (&session as *const u32).cast(),
                size_of::<u32>() as u32,
            )
            .and_then(|()| {
                CreateProcessAsUserW(
                    Some(token),
                    PCWSTR::null(),
                    Some(PWSTR(command_line.as_mut_ptr())),
                    None,
                    None,
                    true,
                    CREATE_NO_WINDOW,
                    None,
                    PCWSTR::null(),
                    &startup,
                    &mut process,
                )
            });
            CloseHandle(token)?;
            launched?;
            CloseHandle(process.hThread)?;
        }
        Ok(Self {
            process: process.hProcess,
            session,
        })
    }

    fn exit_code(&self) -> Result<Option<u32>, Box<dyn Error>> {
        // SAFETY: the process handle is open for the lifetime of self, and the out-param is a valid local.
        unsafe {
            if WaitForSingleObject(self.process, 0) != WAIT_OBJECT_0 {
                return Ok(None);
            }
            let mut code = 0;
            GetExitCodeProcess(self.process, &mut code)?;
            Ok(Some(code))
        }
    }

    fn terminate(&self) -> Result<(), Box<dyn Error>> {
        // SAFETY: the process handle is open for the lifetime of self.
        unsafe {
            TerminateProcess(self.process, 1)?;
            // the next host binds the same ports
            WaitForSingleObject(self.process, TERMINATE_WAIT_MS);
        }
        Ok(())
    }
}

fn console_session() -> u32 {
    // SAFETY: the call takes no arguments.
    unsafe { WTSGetActiveConsoleSessionId() }
}

fn supervise_host(
    events: &Receiver<ServiceEvent>,
    directory: &Path,
    host_arguments: &[String],
) -> Result<(), Box<dyn Error>> {
    let command_line = host_command_line(directory, host_arguments)?;
    let log = File::create(directory.join(HOST_LOG_FILE))?;
    loop {
        let session = console_session();
        if session == NO_CONSOLE_SESSION {
            match events.recv()? {
                ServiceEvent::Stop => return Ok(()),
                ServiceEvent::SessionChange => continue,
            }
        }
        let host = match HostProcess::launch(session, &command_line, &log) {
            Ok(host) => host,
            Err(e) => {
                append_service_log(
                    directory,
                    &format!("cannot start the host in session {session}: {e}"),
                );
                if let Ok(ServiceEvent::Stop) = events.recv_timeout(RESTART_DELAY) {
                    return Ok(());
                }
                continue;
            }
        };
        append_service_log(directory, &format!("started the host in session {session}"));
        loop {
            match events.recv_timeout(HOST_POLL_INTERVAL) {
                Ok(ServiceEvent::Stop) => {
                    host.terminate()?;
                    return Ok(());
                }
                Ok(ServiceEvent::SessionChange) if console_session() != host.session => {
                    append_service_log(
                        directory,
                        "the console session changed, restarting the host",
                    );
                    host.terminate()?;
                    break;
                }
                Ok(ServiceEvent::SessionChange) | Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    return Err("the service control handler is gone".into())
                }
            }
            if let Some(code) = host.exit_code()? {
                append_service_log(directory, &format!("the host exited with code {code}"));
                if let Ok(ServiceEvent::Stop) = events.recv_timeout(RESTART_DELAY) {
                    return Ok(());
                }
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_arguments_survive_the_windows_command_line_rules() {
        assert_eq!(quote_argument("--port"), "--port");
        assert_eq!(quote_argument(""), "\"\"");
        assert_eq!(
            quote_argument(r"C:\Program Files\g2g"),
            r#""C:\Program Files\g2g""#
        );
        assert_eq!(quote_argument(r#"a "b""#), r#""a \"b\"""#);
        assert_eq!(quote_argument(r"trailing slash\ "), r#""trailing slash\ ""#);
        assert_eq!(
            quote_argument(r"C:\dir with space\"),
            r#""C:\dir with space\\""#
        );
    }
}
