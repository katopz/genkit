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

use crate::action::TelemetryInfo;
use crate::error::Result;
use crate::tracing::TRACE_PATH;
use opentelemetry::trace::Span;
use opentelemetry::{
    trace::{FutureExt, Status, TraceContextExt, Tracer},
    Context, KeyValue,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;

// Constants for OpenTelemetry attributes.
pub const ATTR_PREFIX: &str = "genkit";
pub const SPAN_TYPE_ATTR: &str = "genkit:type";
const TRACER_NAME: &str = "genkit-tracer";

/// Converts a `serde_json::Value` to an `opentelemetry::Value`.
///
/// This handles the translation between JSON types and OpenTelemetry's primitive
/// attribute types, preventing issues like booleans being converted to strings.
fn convert_otel_value(v: Value) -> opentelemetry::Value {
    match v {
        Value::Bool(b) => opentelemetry::Value::from(b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                opentelemetry::Value::from(i)
            } else if let Some(f) = n.as_f64() {
                opentelemetry::Value::from(f)
            } else {
                opentelemetry::Value::from(n.to_string())
            }
        }
        Value::String(s) => opentelemetry::Value::from(s),
        // For complex types (Array, Object), serialize to a JSON string as
        // OpenTelemetry attributes are primitive.
        v @ (Value::Array(_) | Value::Object(_)) => {
            opentelemetry::Value::from(serde_json::to_string(&v).unwrap_or_else(|_| v.to_string()))
        }
        Value::Null => opentelemetry::Value::from("null"),
    }
}

/// Executes a future within a new OpenTelemetry span.
///
/// This function is the primary way to instrument code in Genkit. It creates a
/// new span, sets it as the active span for the duration of the future, and
/// records the outcome (success or error) and attributes. It also manages the
/// nested `genkit:path` attribute.
pub async fn in_new_span<F, Fut, T>(
    name: String,
    attrs: Option<HashMap<String, Value>>,
    f: F,
) -> Result<(T, TelemetryInfo)>
where
    F: FnOnce(crate::tracing::TraceContext) -> Fut,
    Fut: Future<Output = Result<T>> + Send,
    T: Serialize + Send,
{
    let tracer = opentelemetry::global::tracer(TRACER_NAME);
    let mut user_attrs = attrs.unwrap_or_default();

    // Determine path and if this is a root span by checking the TRACE_PATH task-local.
    let (path_for_span, path_for_scope, is_root) = TRACE_PATH
        .try_with(|parent_path| {
            let genkit_type = user_attrs
                .get("genkit:type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let subtype = user_attrs
                .get("genkit:metadata:subtype")
                .and_then(|v| v.as_str());

            let path_segment = match (genkit_type, subtype, parent_path.is_empty()) {
                ("action", Some(sub), true) => format!("{{{},t:{}}}", name, sub),
                ("action", Some(sub), false) => format!("{{{},t:action,s:{}}}", name, sub),
                (typ, _, _) => format!("{{{},t:{}}}", name, typ),
            };

            let mut new_path_segments = parent_path.clone();
            new_path_segments.push(path_segment);

            let path_string = format!("/{}", new_path_segments.join("/"));

            (path_string, new_path_segments, parent_path.is_empty())
        })
        .unwrap_or_else(|_| {
            // Fallback if TRACE_PATH is not set. This span becomes the root.
            let genkit_type = user_attrs
                .get("genkit:type")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let subtype = user_attrs
                .get("genkit:metadata:subtype")
                .and_then(|v| v.as_str());
            let type_for_path = subtype.unwrap_or(genkit_type);
            let segment = format!("{{{},t:{}}}", name, type_for_path);
            (format!("/{}", segment), vec![segment], true)
        });

    TRACE_PATH
        .scope(path_for_scope, async move {
            // Build attributes for the span
            user_attrs.insert("genkit:path".to_string(), Value::String(path_for_span));
            user_attrs.insert("genkit:name".to_string(), Value::String(name.clone()));
            if is_root {
                user_attrs.insert("genkit:isRoot".to_string(), Value::Bool(true));
            } else {
                // Per TS behavior, context is only added to the root span's telemetry.
                user_attrs.remove("genkit:metadata:context");
            }

            // Convert serde_json::Value to opentelemetry::Value for the span attributes
            let span_attributes: Vec<KeyValue> = user_attrs
                .into_iter()
                .map(|(k, v)| KeyValue::new(k, convert_otel_value(v)))
                .collect();

            let mut span = tracer.start_with_context(name, &Context::current());
            span.set_attributes(span_attributes);

            let cx = Context::current_with_span(span);

            let trace_context = {
                let span_ref = cx.span();
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

            // Run the provided async operation
            let result = f(trace_context).with_context(cx.clone()).await;

            // Finalize span based on result
            let span = cx.span();
            match &result {
                Ok(output) => {
                    span.set_status(Status::Ok);
                    span.set_attribute(KeyValue::new("genkit:state", "success"));
                    if let Ok(output_str) = serde_json::to_string(output) {
                        span.set_attribute(KeyValue::new("genkit:output", output_str));
                    }
                }
                Err(e) => {
                    span.set_status(Status::Error {
                        description: e.to_string().into(),
                    });
                    span.record_error(e);
                    span.set_attribute(KeyValue::new("genkit:state", "error"));
                }
            }

            result.map(|r| (r, telemetry_info))
        })
        .await
}
