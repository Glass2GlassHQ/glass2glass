use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::string::String;
use std::sync::{Arc, Mutex};

use g2g_core::runtime::{StageVisit, VisitRecorder};

const NANOSECONDS_PER_MICROSECOND: f64 = 1.0e3;

#[derive(Debug)]
pub struct PerfettoTrace {
    state: Mutex<TraceState>,
}

#[derive(Debug)]
struct TraceState {
    out: BufWriter<File>,
    element_processes: HashMap<String, u32>,
    events_written: u64,
    wait_spans: u64,
    first_write_error: Option<io::Error>,
}

impl PerfettoTrace {
    pub fn create(path: &Path) -> io::Result<Arc<Self>> {
        let mut out = BufWriter::new(File::create(path)?);
        out.write_all(b"{\"traceEvents\":[\n")?;
        Ok(Arc::new(Self {
            state: Mutex::new(TraceState {
                out,
                element_processes: HashMap::new(),
                events_written: 0,
                wait_spans: 0,
                first_write_error: None,
            }),
        }))
    }

    pub fn finish(self: Arc<Self>) -> io::Result<()> {
        let trace = Arc::into_inner(self).expect("the finished run holds no trace handle");
        let mut state = trace.state.into_inner().expect("trace writer lock");
        if let Some(error) = state.first_write_error {
            return Err(error);
        }
        state.out.write_all(b"\n]}\n")?;
        state.out.flush()
    }
}

impl VisitRecorder for PerfettoTrace {
    fn record(&self, element: &str, visit: StageVisit) {
        let mut state = self.state.lock().expect("trace writer lock");
        if state.first_write_error.is_some() {
            return;
        }
        if let Err(error) = state.write_visit(element, visit) {
            state.first_write_error = Some(error);
        }
    }
}

impl TraceState {
    fn write_visit(&mut self, element: &str, visit: StageVisit) -> io::Result<()> {
        let process = match self.element_processes.get(element) {
            Some(&process) => process,
            None => self.add_element(element)?,
        };
        let sequence = visit.sequence;
        // waits overlap the previous frame's work and each other
        if visit.wait_ns > 0 {
            self.wait_spans += 1;
            let id = self.wait_spans;
            let begin = microseconds(visit.enter_ns - visit.wait_ns);
            let end = microseconds(visit.enter_ns);
            self.write_event(format_args!(
                r#"{{"ph":"b","cat":"wait","name":"wait","id":{id},"pid":{process},"tid":{process},"ts":{begin},"args":{{"sequence":{sequence}}}}}"#
            ))?;
            self.write_event(format_args!(
                r#"{{"ph":"e","cat":"wait","name":"wait","id":{id},"pid":{process},"tid":{process},"ts":{end}}}"#
            ))?;
        }
        let begin = microseconds(visit.enter_ns);
        let duration = microseconds(visit.exit_ns - visit.enter_ns);
        let push_wait_ns = visit.push_wait_ns;
        self.write_event(format_args!(
            r#"{{"ph":"X","cat":"work","name":"work","pid":{process},"tid":{process},"ts":{begin},"dur":{duration},"args":{{"sequence":{sequence},"push_wait_ns":{push_wait_ns}}}}}"#
        ))
    }

    fn add_element(&mut self, element: &str) -> io::Result<u32> {
        let process = self.element_processes.len() as u32 + 1;
        self.element_processes
            .insert(String::from(element), process);
        let name = JsonString(element);
        self.write_event(format_args!(
            r#"{{"ph":"M","name":"process_name","pid":{process},"args":{{"name":{name}}}}}"#
        ))?;
        self.write_event(format_args!(
            r#"{{"ph":"M","name":"process_sort_index","pid":{process},"args":{{"sort_index":{process}}}}}"#
        ))?;
        self.write_event(format_args!(
            r#"{{"ph":"M","name":"thread_name","pid":{process},"tid":{process},"args":{{"name":"work"}}}}"#
        ))?;
        Ok(process)
    }

    fn write_event(&mut self, event: fmt::Arguments) -> io::Result<()> {
        if self.events_written > 0 {
            self.out.write_all(b",\n")?;
        }
        self.out.write_fmt(event)?;
        self.events_written += 1;
        Ok(())
    }
}

fn microseconds(nanoseconds: u64) -> f64 {
    nanoseconds as f64 / NANOSECONDS_PER_MICROSECOND
}

struct JsonString<'a>(&'a str);

impl fmt::Display for JsonString<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('"')?;
        for character in self.0.chars() {
            match character {
                '"' => f.write_str("\\\"")?,
                '\\' => f.write_str("\\\\")?,
                control if control.is_control() => write!(f, "\\u{:04x}", control as u32)?,
                other => f.write_char(other)?,
            }
        }
        f.write_char('"')
    }
}
