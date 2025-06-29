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

//! # Genkit Tracing
//!
//! This module provides the main entry point for configuring and using
//! OpenTelemetry for tracing within the Genkit framework.

// Declare modules.
pub mod exporter;
pub mod instrumentation;
pub mod types;

// Re-export key types and functions for the public API.
pub use self::instrumentation::in_new_span;

use crate::error::Result;
use crate::telemetry::TelemetryConfig;
use crate::utils::is_dev_env;
use exporter::TraceServerExporter;
use once_cell::sync::OnceCell;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::Span;
use opentelemetry_sdk::{
    trace::{
        BatchSpanProcessor, SdkTracerProvider, SimpleSpanProcessor, SpanData, SpanProcessor,
        TraceError,
    },
    Resource,
};
use std::fmt::{Debug, Formatter, Result as FmtResult};

// A wrapper to allow Box<dyn SpanProcessor> to be used with the new API.
struct BoxedSpanProcessor(Box<dyn SpanProcessor>);

impl Debug for BoxedSpanProcessor {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        self.0.fmt(f)
    }
}

impl SpanProcessor for BoxedSpanProcessor {
    fn on_start(&self, span: &mut Span, cx: &opentelemetry::Context) {
        self.0.on_start(span, cx)
    }

    fn on_end(&self, span: SpanData) {
        self.0.on_end(span)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.0.force_flush()
    }

    fn shutdown(&self) -> OTelSdkResult {
        self.0.shutdown()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.0.set_resource(resource);
    }

    fn shutdown_with_timeout(
        &self,
        _timeout: std::time::Duration,
    ) -> opentelemetry_sdk::error::OTelSdkResult {
        todo!()
    }
}

// A global static to ensure telemetry is initialized only once.
static TELEMETRY_INIT: OnceCell<()> = OnceCell::new();
// A global static to hold the tracer provider for calling shutdown.
static GENKIT_TRACER_PROVIDER: OnceCell<SdkTracerProvider> = OnceCell::new();

/// The OpenTelemetry trace context, containing Trace and Span IDs.
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
}

/// Enables and configures OpenTelemetry tracing.
///
/// This function should be called once at the beginning of the application's
/// lifecycle. It sets up the global tracer provider with the specified
/// configuration. If it is called more than once, subsequent calls will have
/// no effect and will return `Ok(())`.
///
/// # Arguments
///
/// * `config` - A `TelemetryConfig` struct containing custom configuration
///   for span processors, resources, and samplers.
pub fn enable_telemetry(mut config: TelemetryConfig) -> Result<(), TraceError> {
    TELEMETRY_INIT.get_or_try_init(|| {
        // 1. Create the default processor that exports to the Genkit Telemetry Server.
        let telemetry_processor: Box<dyn SpanProcessor> = if is_dev_env() {
            let exporter = TraceServerExporter::new();
            Box::new(SimpleSpanProcessor::new(exporter))
        } else {
            let exporter = TraceServerExporter::new();
            Box::new(BatchSpanProcessor::new(exporter, Default::default()))
        };

        // 2. Combine with any user-provided processors.
        let mut processors: Vec<Box<dyn SpanProcessor>> = Vec::new();
        processors.push(telemetry_processor);
        processors.append(&mut config.span_processors);

        // 3. Build the TracerProvider.
        let mut provider_builder = SdkTracerProvider::builder();
        for p in processors {
            provider_builder = provider_builder.with_span_processor(BoxedSpanProcessor(p));
        }
        if let Some(sampler) = config.sampler {
            provider_builder = provider_builder.with_sampler(sampler);
        }
        if let Some(resource) = config.resource {
            provider_builder = provider_builder.with_resource(resource);
        }
        let provider = provider_builder.build();

        // 4. Set the global tracer provider and store our handle for later shutdown.
        opentelemetry::global::set_tracer_provider(provider.clone());
        GENKIT_TRACER_PROVIDER
            .set(provider)
            .map_err(|_| TraceError::from("Failed to store SdkTracerProvider in global static."))?;

        Ok::<(), TraceError>(())
    })?;
    Ok(())
}

/// Shuts down the global tracer provider, flushing any buffered spans.
///
/// This should be called before the application exits to ensure all telemetry
/// data is sent.
pub fn flush_tracing() -> OTelSdkResult {
    if let Some(provider) = GENKIT_TRACER_PROVIDER.get() {
        provider.shutdown()
    } else {
        OTelSdkResult::Ok(())
    }
}
