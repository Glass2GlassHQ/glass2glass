// Copyright (C) 2026 Aaron Boxer <boxerab@protonmail.com>
//
// SPDX-License-Identifier: MPL-2.0

use std::cell::RefCell;
use std::fmt;
use std::io;

use tokio::process::Child;

pub(crate) const DESKTOP_APP_ID: u32 = 1;
const DESKTOP_APP_TITLE: &str = "Desktop";
const FIRST_LAUNCHABLE_APP_ID: u32 = 2;

pub type AppStarter = Box<dyn Fn(&str) -> io::Result<Child>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchableApp {
    pub title: String,
    pub command: String,
}

pub struct ServedApps {
    pub listed: Vec<LaunchableApp>,
    pub start: AppStarter,
}

impl fmt::Debug for ServedApps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ServedApps")
            .field("listed", &self.listed)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub(crate) enum LaunchError {
    UnknownApp(u32),
    Start { title: String, error: io::Error },
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownApp(id) => write!(f, "no app has id {id}"),
            Self::Start { title, error } => write!(f, "could not start {title}: {error}"),
        }
    }
}

#[derive(Debug)]
struct RunningApp {
    id: u32,
    process: Child,
}

#[derive(Debug)]
pub(crate) struct AppLauncher {
    served: Option<ServedApps>,
    running: RefCell<Option<RunningApp>>,
}

impl AppLauncher {
    pub(crate) fn new(served: Option<ServedApps>) -> Self {
        Self {
            served,
            running: RefCell::new(None),
        }
    }

    pub(crate) fn listed(&self) -> impl Iterator<Item = (u32, &str)> {
        let launchable = self.served.iter().flat_map(|served| &served.listed);
        std::iter::once((DESKTOP_APP_ID, DESKTOP_APP_TITLE)).chain(
            (FIRST_LAUNCHABLE_APP_ID..)
                .zip(launchable)
                .map(|(id, app)| (id, app.title.as_str())),
        )
    }

    pub(crate) fn running_id(&self) -> Option<u32> {
        let mut running = self.running.borrow_mut();
        let exited = match running.as_mut()?.process.try_wait() {
            Ok(status) => status.is_some(),
            Err(e) => {
                eprintln!("g2g-desktop host: lost track of a launched app: {e}");
                true
            }
        };
        if exited {
            running.take();
        }
        running.as_ref().map(|app| app.id)
    }

    pub(crate) async fn launch(&self, id: u32) -> Result<(), LaunchError> {
        if id == DESKTOP_APP_ID || self.running_id() == Some(id) {
            return Ok(());
        }
        let app = id
            .checked_sub(FIRST_LAUNCHABLE_APP_ID)
            .and_then(|index| Some((self.served.as_ref()?, usize::try_from(index).ok()?)))
            .and_then(|(served, index)| Some((served, served.listed.get(index)?)));
        let Some((served, app)) = app else {
            return Err(LaunchError::UnknownApp(id));
        };
        self.quit().await;
        let process = (served.start)(&app.command).map_err(|error| LaunchError::Start {
            title: app.title.clone(),
            error,
        })?;
        eprintln!("g2g-desktop host: started {}", app.title);
        *self.running.borrow_mut() = Some(RunningApp { id, process });
        Ok(())
    }

    pub(crate) async fn quit(&self) {
        let running = self.running.borrow_mut().take();
        let Some(mut app) = running else {
            return;
        };
        if let Err(e) = app.process.kill().await {
            eprintln!("g2g-desktop host: could not stop app {}: {e}", app.id);
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use std::path::Path;

    use super::*;

    const SLEEPING: &str = "exec sleep 30";

    fn launcher() -> AppLauncher {
        let app = |title: &str| LaunchableApp {
            title: title.to_owned(),
            command: SLEEPING.to_owned(),
        };
        AppLauncher::new(Some(ServedApps {
            listed: vec![app("First"), app("Second")],
            start: Box::new(|command| {
                tokio::process::Command::new("sh")
                    .arg("-c")
                    .arg(command)
                    .kill_on_drop(true)
                    .spawn()
            }),
        }))
    }

    fn running_pid(launcher: &AppLauncher) -> u32 {
        let running = launcher.running.borrow();
        let app = running.as_ref().expect("an app runs");
        app.process.id().expect("the app has not been waited on")
    }

    fn alive(pid: u32) -> bool {
        Path::new(&format!("/proc/{pid}")).exists()
    }

    #[test]
    fn the_desktop_comes_first_and_each_app_takes_the_next_id() {
        let launcher = launcher();
        let listed: Vec<_> = launcher.listed().collect();
        assert_eq!(
            listed,
            [(DESKTOP_APP_ID, "Desktop"), (2, "First"), (3, "Second")]
        );
    }

    #[tokio::test]
    async fn launching_another_app_stops_the_running_one_and_quit_stops_it_too() {
        let launcher = launcher();
        launcher.launch(2).await.unwrap();
        let first = running_pid(&launcher);
        launcher.launch(2).await.unwrap();
        assert_eq!(running_pid(&launcher), first, "a relaunch keeps the app");
        launcher.launch(3).await.unwrap();
        assert_eq!(launcher.running_id(), Some(3));
        assert!(!alive(first), "the first app was stopped");
        let second = running_pid(&launcher);
        launcher.quit().await;
        assert_eq!(launcher.running_id(), None);
        assert!(!alive(second), "quit stopped the second app");
    }

    #[tokio::test]
    async fn an_unknown_id_is_refused() {
        assert!(matches!(
            launcher().launch(4).await,
            Err(LaunchError::UnknownApp(4))
        ));
    }
}
