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

//! # Tracing Instrumentation
//!
//! This module provides the core functions for creating and managing trace spans.
//! It is the Rust equivalent of `tracing/instrumentation.ts`.

use super::types::{SpanMetadata, SpanState};
use crate::action::TelemetryInfo;
use crate::error::Result;
use opentelemetry::{
    trace::{FutureExt, Status, TraceContextExt, Tracer},
    Context, KeyValue,
};
use std::collections::HashMap;
use std::future::Future;

// Constants for OpenTelemetry attributes.
pub const ATTR_PREFIX: &str = "genkit";
pub const SPAN_TYPE_ATTR: &str = "genkit:type";
const TRACER_NAME: &str = "genkit-tracer";

/// Executes a future within a new OpenTelemetry span.
///
/// This function is the primary way to instrument code in Genkit. It creates a
/// new span, sets it as the active span for the duration of the future, and
/// records the outcome (success or error) and attributes.
pub async fn in_new_span<F, Fut, T>(
    name: String,
    attrs: Option<HashMap<String, serde_json::Value>>,
    f: F,
) -> Result<(T, TelemetryInfo)>
where
    F: FnOnce(crate::tracing::TraceContext) -> Fut,
    Fut: Future<Output = Result<T>> + Send,
    T: Send,
{
    let tracer = opentelemetry::global::tracer(TRACER_NAME);
    let span = tracer.start(name); // This returns a guard that must be kept alive.

    let cx = Context::current_with_span(span);

    let trace_context = {
        let span_ref = cx.span(); // Get a reference to the span inside the context
        let span_context = span_ref.span_context();
        crate::tracing::TraceContext {
            trace_id: span_context.trace_id().to_string(),
            span_id: span_context.span_id().to_string(),
        }
    };

    let telemetry_info = TelemetryInfo {
        trace_id: trace_context.trace_id.clone(),
        span_id: trace_context.span_id.clone(),
    };

    // Run the async operation within the span's context.
    let result = f(trace_context).with_context(cx.clone()).await;

    // Now that the future is complete, we can update the span's status before it's dropped.
    let span = cx.span();
    if let Some(a) = attrs {
        span.set_attributes(a.into_iter().map(|(k, v)| KeyValue::new(k, v.to_string())));
    }
    match &result {
        Ok(_) => span.set_status(Status::Ok),
        Err(e) => {
            span.set_status(Status::Error {
                description: format!("{e:#?}").into(),
            });
            span.record_error(e);
        }
    }

    // `span` (the guard from cx.span()) goes out of scope here, and its Drop impl will call `end()`.
    result.map(|r| (r, telemetry_info))
}

/// Converts `SpanMetadata` into a `Vec` of OpenTelemetry `KeyValue` pairs.
pub fn metadata_to_attributes(metadata: &SpanMetadata) -> Vec<KeyValue> {
    let mut attributes = Vec::new();
    attributes.push(KeyValue::new(
        format!("{}:name", ATTR_PREFIX),
        metadata.name.clone(),
    ));
    if let Some(state) = &metadata.state {
        attributes.push(KeyValue::new(
            format!("{}:state", ATTR_PREFIX),
            match state {
                SpanState::Success => "success",
                SpanState::Error => "error",
            }
            .to_string(),
        ));
    }
    if let Some(input) = &metadata.input {
        if let Ok(json_str) = serde_json::to_string(input) {
            attributes.push(KeyValue::new(format!("{}:input", ATTR_PREFIX), json_str));
        }
    }
    if let Some(output) = &metadata.output {
        if let Ok(json_str) = serde_json::to_string(output) {
            attributes.push(KeyValue::new(format!("{}:output", ATTR_PREFIX), json_str));
        }
    }
    if let Some(is_root) = metadata.is_root {
        attributes.push(KeyValue::new(format!("{}:isRoot", ATTR_PREFIX), is_root));
    }
    if let Some(path) = &metadata.path {
        attributes.push(KeyValue::new(format!("{}:path", ATTR_PREFIX), path.clone()));
    }
    if let Some(is_failure_source) = metadata.is_failure_source {
        attributes.push(KeyValue::new(
            format!("{}:isFailureSource", ATTR_PREFIX),
            is_failure_source,
        ));
    }
    if let Some(meta) = &metadata.metadata {
        for (key, value) in meta {
            attributes.push(KeyValue::new(
                format!("{}:metadata:{}", ATTR_PREFIX, key),
                value.clone(),
            ));
        }
    }

    attributes
}
