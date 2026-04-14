//! Diagnostics utilities for TOPP algorithms.

use std::fs::{File, OpenOptions, create_dir_all};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::{Duration, Instant};

use super::error::CoppError;

/// Output backend used by verbosity logging.
///
/// Default is [`Println`](VerbosityOutput::Println) to preserve current behavior.
#[derive(Default, Eq, PartialEq, Clone, Debug)]
pub enum VerbosityOutput {
    /// Emit messages with `println!`.
    ///
    /// Typical usage:
    /// ```rust,no_run
    /// use copp::diag::{VerbosityOutput, set_verbosity_output};
    ///
    /// set_verbosity_output(VerbosityOutput::Println)?;
    /// # Ok::<(), copp::diag::CoppError>(())
    /// ```
    #[default]
    Println,
    /// Emit messages through the `log` facade (`info/debug/trace`).
    ///
    /// Before selecting this mode, initialize a global logger in your app/test.
    ///
    /// Typical usage:
    /// ```rust,no_run
    /// use log::LevelFilter;
    /// use std::sync::Once;
    /// use copp::prelude::{VerbosityOutput, set_verbosity_output};
    ///
    /// static INIT: Once = Once::new();
    /// INIT.call_once(|| {
    ///     let _ = env_logger::Builder::new()
    ///         .filter_level(LevelFilter::Info)
    ///         .is_test(true)
    ///         .try_init();
    /// });
    /// set_verbosity_output(VerbosityOutput::Log)?;
    /// # Ok::<(), copp::diag::CoppError>(())
    /// ```
    Log,
    /// Emit messages to a configured log file.
    ///
    /// Typical usage:
    /// ```rust,no_run
    /// use copp::prelude::{VerbosityOutput, set_verbosity_output};
    ///
    /// // Auto-creates parent directories and opens file in append mode.
    /// set_verbosity_output(VerbosityOutput::File("logs/copp/run.log".into()))?;
    /// # Ok::<(), copp::prelude::CoppError>(())
    /// ```
    File(String),
}

const VERBOSITY_MODE_PRINTLN: u8 = 0;
const VERBOSITY_MODE_LOG: u8 = 1;
const VERBOSITY_MODE_FILE: u8 = 2;

static VERBOSITY_OUTPUT: AtomicU8 = AtomicU8::new(VERBOSITY_MODE_PRINTLN);
static VERBOSITY_FILE: Mutex<Option<File>> = Mutex::new(None);
static VERBOSITY_FILE_PATH: Mutex<Option<String>> = Mutex::new(None);

/// Set output backend for all verbosity messages.
///
/// This is a global process-wide switch.
pub fn set_verbosity_output(output: VerbosityOutput) -> Result<(), CoppError> {
    match output {
        VerbosityOutput::Println => {
            VERBOSITY_OUTPUT.store(VERBOSITY_MODE_PRINTLN, Ordering::Relaxed);
            Ok(())
        }
        VerbosityOutput::Log => {
            VERBOSITY_OUTPUT.store(VERBOSITY_MODE_LOG, Ordering::Relaxed);
            Ok(())
        }
        VerbosityOutput::File(path) => set_verbosity_log_file(path),
    }
}

/// Get current output backend for verbosity messages.
pub fn verbosity_output() -> VerbosityOutput {
    match VERBOSITY_OUTPUT.load(Ordering::Relaxed) {
        VERBOSITY_MODE_LOG => VerbosityOutput::Log,
        VERBOSITY_MODE_FILE => {
            let path = VERBOSITY_FILE_PATH
                .lock()
                .expect("verbosity_output: mutex poisoned")
                .clone()
                .unwrap_or_else(|| "<unknown>".to_string());
            VerbosityOutput::File(path)
        }
        _ => VerbosityOutput::Println,
    }
}

/// Configure a log file as verbosity output backend.
///
/// The parent directory is created automatically if it does not exist.
/// File is opened in append mode.
pub fn set_verbosity_log_file(path: impl AsRef<Path>) -> Result<(), CoppError> {
    let path = path.as_ref();
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        create_dir_all(parent)?;
    }
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let path_string = path.to_string_lossy().to_string();
    let mut file_slot = VERBOSITY_FILE
        .lock()
        .expect("set_verbosity_log_file: mutex poisoned");
    let mut path_slot = VERBOSITY_FILE_PATH
        .lock()
        .expect("set_verbosity_log_file(path): mutex poisoned");
    *file_slot = Some(file);
    *path_slot = Some(path_string);
    VERBOSITY_OUTPUT.store(VERBOSITY_MODE_FILE, Ordering::Relaxed);
    Ok(())
}

/// The verbosity level for logging.  
/// The default value is [`Silent`](`Verbosity::Silent`).  
/// [`Trace`](`Verbosity::Trace`) > [`Debug`](`Verbosity::Debug`) > [`Summary`](`Verbosity::Summary`) > [`Silent`](`Verbosity::Silent`).
#[repr(u8)]
#[derive(Default, Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
pub enum Verbosity {
    /// No log will be emitted.
    #[default]
    Silent = 0,
    /// Only summary information will be emitted, including:  
    /// + The beginning and end of each algorithm.  
    /// + The total computation time of each algorithm.  
    /// + The success or failure of each algorithm.
    Summary = 1,
    /// Detailed debug information will be emitted, including:  
    /// + The failure or degeneration of each algorithm at each grid point.
    Debug = 2,
    /// Very detailed trace information will be emitted, including:  
    /// + The values of reachable sets, optimal profiles, and other intermediate variables at each step of each algorithm.
    Trace = 3,
}

/// Emit one verbosity message using the configured backend.
pub fn emit_verbosity_line(level: Verbosity, message: impl AsRef<str>) {
    let message = message.as_ref();
    match verbosity_output() {
        VerbosityOutput::Println => println!("{message}"),
        VerbosityOutput::Log => match level {
            Verbosity::Silent => {}
            Verbosity::Summary => log::info!("{message}"),
            Verbosity::Debug => log::debug!("{message}"),
            Verbosity::Trace => log::trace!("{message}"),
        },
        VerbosityOutput::File(_) => {
            if matches!(level, Verbosity::Silent) {
                return;
            }
            let mut file_slot = VERBOSITY_FILE
                .lock()
                .expect("emit_verbosity_line: mutex poisoned");
            if let Some(file) = file_slot.as_mut() {
                let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f");
                let level_tag = match level {
                    Verbosity::Silent => "SILENT",
                    Verbosity::Summary => "INFO",
                    Verbosity::Debug => "DEBUG",
                    Verbosity::Trace => "TRACE",
                };
                let _ = writeln!(file, "[{now}] [{level_tag}] {message}");
                let _ = file.flush();
            } else {
                println!("{message}");
            }
        }
    }
}

/// Emit a formatted verbosity message through the global diagnostics backend.
///
/// This macro is a convenience wrapper over [`emit_verbosity_line`](crate::diag::emit_verbosity_line).
/// It accepts a [`Verbosity`](crate::diag::Verbosity) level plus `format!`-style arguments.
///
/// # Example
/// ```rust,no_run
/// use copp::diag::Verbosity;
///
/// copp::verbosity_log!(Verbosity::Summary, "solver finished in {:.3} ms", 12.34);
/// ```
#[macro_export]
macro_rules! verbosity_log {
    ($level:expr, $($arg:tt)*) => {
        $crate::diag::emit_verbosity_line($level, format!($($arg)*))
    };
}

pub(crate) trait Verboser {
    /// The verbosity level for logging.
    const LEVEL: Verbosity;
    /// Check if the current verbosity level is greater than or equal to the given level.
    #[inline(always)]
    fn is_enabled(&self, level: Verbosity) -> bool {
        Self::LEVEL >= level
    }
    fn record_start_time(&mut self);
    fn elapsed(&self) -> Duration;
}

pub(crate) struct SilentVerboser;

impl Verboser for SilentVerboser {
    const LEVEL: Verbosity = Verbosity::Silent;
    #[inline(always)]
    fn record_start_time(&mut self) {}
    #[inline(always)]
    fn elapsed(&self) -> Duration {
        Duration::ZERO
    }
}
pub(crate) struct SummaryVerboser {
    start_time: Instant,
}
impl SummaryVerboser {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }
}
impl Verboser for SummaryVerboser {
    const LEVEL: Verbosity = Verbosity::Summary;
    #[inline(always)]
    fn record_start_time(&mut self) {
        self.start_time = Instant::now();
    }
    #[inline(always)]
    fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}
pub(crate) struct DebugVerboser {
    start_time: Instant,
}
impl DebugVerboser {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }
}
impl Verboser for DebugVerboser {
    const LEVEL: Verbosity = Verbosity::Debug;
    #[inline(always)]
    fn record_start_time(&mut self) {
        self.start_time = Instant::now();
    }
    #[inline(always)]
    fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}
pub(crate) struct TraceVerboser {
    start_time: Instant,
}
impl TraceVerboser {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }
}
impl Verboser for TraceVerboser {
    const LEVEL: Verbosity = Verbosity::Trace;
    #[inline(always)]
    fn record_start_time(&mut self) {
        self.start_time = Instant::now();
    }
    #[inline(always)]
    fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Format a `Duration` into a human-readable string with appropriate units (ns, us, ms, s, min, h).
pub(crate) fn format_duration_human(d: Duration) -> String {
    let secs = d.as_secs_f64();

    fn fmt(v: f64, unit: &str) -> String {
        if v >= 100.0 {
            format!("{v:.0} {unit}")
        } else if v >= 10.0 {
            format!("{v:.1} {unit}")
        } else {
            format!("{v:.3} {unit}")
        }
    }

    if secs < 1e-6 {
        fmt(secs * 1e9, "ns")
    } else if secs < 1e-3 {
        fmt(secs * 1e6, "us")
    } else if secs < 1.0 {
        fmt(secs * 1e3, "ms")
    } else if secs < 60.0 {
        fmt(secs, "s")
    } else if secs < 3600.0 {
        fmt(secs / 60.0, "min")
    } else {
        fmt(secs / 3600.0, "h")
    }
}
