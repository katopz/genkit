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

//! # Test Utilities
//!
//! This module provides shared utilities for integration tests, such as
//! mock telemetry exporters.

use opentelemetry_sdk::error::OTelSdkError;
use opentelemetry_sdk::trace::{SpanData, SpanExporter};
use std::fmt::Debug;

use std::sync::{Arc, Mutex};

/// A simple `SpanExporter` that stores exported spans in memory for inspection.
///
/// This is useful for tests that need to verify the telemetry data being generated.
#[derive(Debug, Clone)]
pub struct TestSpanExporter {
    exported_spans: Arc<Mutex<Vec<SpanData>>>,
}

impl Default for TestSpanExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl TestSpanExporter {
    /// Creates a new, empty `TestSpanExporter`.
    pub fn new() -> Self {
        Self {
            exported_spans: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a clone of the spans that have been exported to this exporter.
    pub fn get_exported_spans(&self) -> Vec<SpanData> {
        self.exported_spans.lock().unwrap().clone()
    }

    /// Clears all spans that have been exported.
    pub fn clear(&self) {
        self.exported_spans.lock().unwrap().clear();
    }
}

impl SpanExporter for TestSpanExporter {
    /// Exports a batch of spans. This implementation stores them in a shared vector.
    async fn export(&self, batch: Vec<SpanData>) -> Result<(), OTelSdkError> {
        if let Ok(mut spans) = self.exported_spans.lock() {
            spans.extend(batch);
        }
        Ok(())
    }

    /// Shuts down the exporter. This is a no-op for the test exporter.
    fn shutdown(&mut self) -> Result<(), OTelSdkError> {
        Ok(())
    }

    /// Flushes any buffered spans. This is a no-op for the test exporter.
    fn force_flush(&mut self) -> Result<(), OTelSdkError> {
        Ok(())
    }
}
