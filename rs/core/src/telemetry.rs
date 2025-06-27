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

//! # Telemetry Configuration Types
//!
//! This module defines the configuration structures for setting up OpenTelemetry.
//! It is the Rust equivalent of `telemetryTypes.ts`.

use opentelemetry_sdk::trace;
use opentelemetry_sdk::Resource;

/// Configuration for setting up the OpenTelemetry SDK.
///
/// This struct allows for the customization of tracing and other telemetry
/// aspects. Plugins can provide instances of this configuration to specify
/// how and where telemetry data will be exported.
#[derive(Default)]
pub struct TelemetryConfig {
    /// A list of additional span processors to register with the tracer provider.
    /// The framework will add its own processors for internal telemetry, but
    /// plugins can add more to export traces to other systems (e.g., OTLP exporter).
    pub span_processors: Vec<Box<dyn trace::SpanProcessor>>,

    /// The OpenTelemetry `Resource` to associate with all telemetry.
    /// If not provided, a default resource may be created.
    pub resource: Option<Resource>,

    /// The sampler to use for tracing. If not provided, a default sampler
    /// (`AlwaysOnSampler`) is typically used.
    pub sampler: Option<trace::Sampler>,
}
