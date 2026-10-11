//! Tokio-backed [`GraphSpawner`] for the opt-in thread-per-arm graph runner
//! ([`g2g_core::runtime::run_graph_threaded`]).
//!
//! Each graph arm gets its own OS thread running a dedicated **current-thread**
//! tokio runtime (`enable_all`, so a network source / paced element has its own
//! reactor + timer). The arm's `!Send` future is built and driven entirely on
//! that thread; only the element and its channels (all `Send`) crossed the
//! boundary at setup. This is the GStreamer streaming-thread model: CPU-bound
//! stages (software decode/encode) run on separate cores instead of serialising
//! on one cooperative executor, at the cost of a per-stage thread handoff (so it
//! is opt-in; cooperative `run_graph` stays the low-latency default).
//!
//! Prefer this over core's `ThreadSpawner` whenever any element uses tokio
//! (network, timers): `ThreadSpawner` drives arms on a bare park-based executor
//! with no reactor.

use std::boxed::Box;
use std::format;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::vec::Vec;

use g2g_core::element::BoxFuture;
use g2g_core::error::{G2gError, HardwareError};
use g2g_core::runtime::{GraphSpawner, LocalArmFuture};

const SYSFS_CPUS: &str = "/sys/devices/system/cpu";
const L2_CACHE_LEVEL: u32 = 2;
const L3_CACHE_LEVEL: u32 = 3;

/// Runs each graph arm on its own OS thread with a private current-thread tokio
/// runtime. See the module docs.
#[derive(Debug, Default, Clone, Copy)]
pub struct TokioThreadSpawner;

impl GraphSpawner for TokioThreadSpawner {
    fn spawn_arm(
        &self,
        build: Box<dyn FnOnce() -> LocalArmFuture + Send>,
    ) -> BoxFuture<'static, Result<u64, G2gError>> {
        drive_on_a_thread(build, || Ok(()))
    }
}

#[derive(Debug)]
pub struct PinnedThreadSpawner {
    cores: Vec<usize>,
    next_arm: AtomicUsize,
}

impl PinnedThreadSpawner {
    pub fn new(cores: Vec<usize>) -> Self {
        assert!(!cores.is_empty(), "pinned arms need at least one core");
        Self {
            cores,
            next_arm: AtomicUsize::new(0),
        }
    }

    pub fn from_sysfs() -> io::Result<Self> {
        Ok(Self::new(core_order(Path::new(SYSFS_CPUS))?))
    }

    pub fn cores(&self) -> &[usize] {
        &self.cores
    }
}

impl GraphSpawner for PinnedThreadSpawner {
    fn spawn_arm(
        &self,
        build: Box<dyn FnOnce() -> LocalArmFuture + Send>,
    ) -> BoxFuture<'static, Result<u64, G2gError>> {
        let arm = self.next_arm.fetch_add(1, Ordering::Relaxed);
        let cpu = self.cores[arm % self.cores.len()];
        drive_on_a_thread(build, move || pin_current_thread(cpu))
    }
}

fn drive_on_a_thread(
    build: Box<dyn FnOnce() -> LocalArmFuture + Send>,
    prepare: impl FnOnce() -> io::Result<()> + Send + 'static,
) -> BoxFuture<'static, Result<u64, G2gError>> {
    // a panicked thread or dropped handle reads as Shutdown, like a closed link
    let (tx, rx) = tokio::sync::oneshot::channel::<Result<u64, G2gError>>();
    thread::spawn(move || {
        let result = match prepare() {
            Err(error) => Err(G2gError::Hardware(HardwareError::Io(
                error.raw_os_error().unwrap_or(0),
            ))),
            Ok(()) => match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                // `build()` is invoked inside `block_on` so the arm's future is
                // constructed with this runtime as its ambient context.
                Ok(rt) => rt.block_on(async move { build().await }),
                Err(_) => Err(G2gError::Shutdown),
            },
        };
        let _ = tx.send(result);
    });
    Box::pin(async move { rx.await.unwrap_or(Err(G2gError::Shutdown)) })
}

#[cfg(target_os = "linux")]
fn pin_current_thread(cpu: usize) -> io::Result<()> {
    // SAFETY: a zeroed cpu_set_t is the empty set and CPU_SET writes within it
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu, &mut set);
        // pid 0 is the calling thread
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread(_cpu: usize) -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "pinned arm threads are linux only",
    ))
}

fn core_order(root: &Path) -> io::Result<Vec<usize>> {
    let online = cpu_list(&fs::read_to_string(root.join("online"))?);
    let mut keyed = Vec::with_capacity(online.len());
    for &cpu in &online {
        let cpu_dir = root.join(format!("cpu{cpu}"));
        let siblings = fs::read_to_string(cpu_dir.join("topology/thread_siblings_list"))?;
        let core = cpu_list(&siblings).into_iter().min().unwrap_or(cpu);
        let l2 = shared_cache_group(&cpu_dir, L2_CACHE_LEVEL)?.unwrap_or(core);
        let l3 = shared_cache_group(&cpu_dir, L3_CACHE_LEVEL)?.unwrap_or(l2);
        keyed.push((cpu != core, l3, l2, core, cpu));
    }
    keyed.sort_unstable();
    Ok(keyed.into_iter().map(|(_, _, _, _, cpu)| cpu).collect())
}

fn shared_cache_group(cpu_dir: &Path, level: u32) -> io::Result<Option<usize>> {
    let cache_dir = cpu_dir.join("cache");
    let Ok(entries) = fs::read_dir(&cache_dir) else {
        return Ok(None);
    };
    for entry in entries {
        let index_dir = entry?.path();
        let Ok(found) = fs::read_to_string(index_dir.join("level")) else {
            continue;
        };
        if found.trim().parse::<u32>().ok() != Some(level) {
            continue;
        }
        let shared = fs::read_to_string(index_dir.join("shared_cpu_list"))?;
        return Ok(cpu_list(&shared).into_iter().min());
    }
    Ok(None)
}

fn cpu_list(text: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        match part.split_once('-') {
            Some((first, last)) => {
                let (Ok(first), Ok(last)) = (first.parse::<usize>(), last.parse::<usize>()) else {
                    continue;
                };
                cpus.extend(first..=last);
            }
            None => cpus.extend(part.parse::<usize>().ok()),
        }
    }
    cpus
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::string::ToString;
    use std::vec;

    fn fake_cpu(root: &Path, cpu: usize, siblings: &str, l2: &str, l3: &str) {
        let cpu_dir = root.join(format!("cpu{cpu}"));
        fs::create_dir_all(cpu_dir.join("topology")).unwrap();
        fs::write(cpu_dir.join("topology/thread_siblings_list"), siblings).unwrap();
        for (index, level, shared) in [(0, 1, siblings), (1, 2, l2), (2, 3, l3)] {
            let index_dir = cpu_dir.join(format!("cache/index{index}"));
            fs::create_dir_all(&index_dir).unwrap();
            fs::write(index_dir.join("level"), level.to_string()).unwrap();
            fs::write(index_dir.join("shared_cpu_list"), shared).unwrap();
        }
    }

    fn fake_sysfs(name: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("g2g-core-order-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn cpu_lists_take_ranges_and_singles() {
        assert_eq!(cpu_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(cpu_list(""), Vec::<usize>::new());
    }

    #[test]
    fn second_smt_threads_come_after_every_first_one() {
        let root = fake_sysfs("smt");
        fs::write(root.join("online"), "0-3").unwrap();
        fake_cpu(&root, 0, "0-1", "0-1", "0-3");
        fake_cpu(&root, 1, "0-1", "0-1", "0-3");
        fake_cpu(&root, 2, "2-3", "2-3", "0-3");
        fake_cpu(&root, 3, "2-3", "2-3", "0-3");
        assert_eq!(core_order(&root).unwrap(), vec![0, 2, 1, 3]);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn cores_group_by_l3_then_l2() {
        let root = fake_sysfs("clusters");
        fs::write(root.join("online"), "0-5").unwrap();
        for cpu in 0..3 {
            fake_cpu(&root, cpu, &cpu.to_string(), "0-1", "0-2");
        }
        fake_cpu(&root, 2, "2", "2", "0-2");
        for cpu in 3..6 {
            fake_cpu(&root, cpu, &cpu.to_string(), "3-5", "3-5");
        }
        assert_eq!(core_order(&root).unwrap(), vec![0, 1, 2, 3, 4, 5]);
        fs::write(root.join("online"), "0,2,4").unwrap();
        assert_eq!(core_order(&root).unwrap(), vec![0, 2, 4]);
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn missing_cache_directories_fall_back_to_core_ids() {
        let root = fake_sysfs("nocache");
        fs::write(root.join("online"), "0-1").unwrap();
        for cpu in 0..2 {
            let cpu_dir = root.join(format!("cpu{cpu}/topology"));
            fs::create_dir_all(&cpu_dir).unwrap();
            fs::write(cpu_dir.join("thread_siblings_list"), cpu.to_string()).unwrap();
        }
        assert_eq!(core_order(&root).unwrap(), vec![0, 1]);
        fs::remove_dir_all(&root).unwrap();
    }

    #[cfg(target_os = "linux")]
    fn allowed_cpus() -> Vec<usize> {
        // SAFETY: a zeroed cpu_set_t is a valid set for sched_getaffinity to fill.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            assert_eq!(
                libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set),
                0
            );
            (0..libc::CPU_SETSIZE as usize)
                .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
                .collect()
        }
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn an_arm_runs_on_its_pinned_core() {
        let last_cpu = *core_order(Path::new(SYSFS_CPUS)).unwrap().last().unwrap();
        let spawner = PinnedThreadSpawner::new(vec![last_cpu]);
        let build: Box<dyn FnOnce() -> LocalArmFuture + Send> = Box::new(|| {
            Box::pin(async move {
                let allowed = allowed_cpus();
                assert_eq!(allowed.len(), 1);
                Ok(allowed[0] as u64)
            })
        });
        assert_eq!(spawner.spawn_arm(build).await.unwrap(), last_cpu as u64);
    }
}
