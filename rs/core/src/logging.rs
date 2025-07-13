// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Genkit Logging
//!
//! This module provides logging functionalities for the Genkit framework.
//! It is a Rust-idiomatic equivalent of `logging.ts`, built on top of the
//! standard `log` crate facade.
//!
//! The library itself only uses the `log` macros (`info!`, `warn!`, etc.).
//! An application using this library is responsible for initializing a logger
//! implementation. This module provides a simple default logger that can be
// initialized with `init()`.

use log::{LevelFilter, SetLoggerError};
use once_cell::sync::Lazy;
use std::sync::Mutex;

// Re-export log macros for convenient use as `genkit_core::logging::info!`.
pub use log::{debug, error, info, trace, warn};

// Global state for the log level. This is only used by our simple logger.
static LOG_LEVEL: Lazy<Mutex<LevelFilter>> = Lazy::new(|| Mutex::new(LevelFilter::Info));

/// A simple logger that prints to standard output.
struct SimpleLogger;

impl log::Log for SimpleLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= *LOG_LEVEL.lock().unwrap()
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            println!("[{:<5}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static LOGGER: SimpleLogger = SimpleLogger;

/// Initializes a simple logger that writes to stdout.
///
/// This should be called only once at the beginning of the application.
/// If another logger has already been initialized, this will return an error.
/// The default log level is `Info`. Use `set_log_level` to change it.
pub fn init() -> Result<(), SetLoggerError> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Trace))
}

/// Sets the global log level for the simple logger.
///
/// Has no effect if a different logging implementation is used.
pub fn set_log_level(level: LevelFilter) {
    *LOG_LEVEL.lock().unwrap() = level;
}

/// Logs a structured message at the INFO level.
/// In this simple logger, it just formats the message. A more advanced
/// logger could serialize the metadata to JSON.
pub fn log_structured(msg: &str, metadata: serde_json::Value) {
    info!("{}: {}", msg, metadata);
}

/// Logs a structured message at the ERROR level.
pub fn log_structured_error(msg: &str, metadata: serde_json::Value) {
    error!("{}: {}", msg, metadata);
}
