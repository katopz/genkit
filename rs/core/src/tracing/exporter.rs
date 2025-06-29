// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is 'to' an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Trace Server Exporter
//!
//! This module provides a `SpanExporter` that sends trace data to a Genkit
//! telemetry server. It is the Rust equivalent of `tracing/exporter.ts`.

use opentelemetry::KeyValue;
// Use aliases to distinguish between OTel's SpanData and our own.
use opentelemetry::trace::Status as OtelStatus;
use opentelemetry_sdk::error::{OTelSdkError, OTelSdkResult};
use opentelemetry_sdk::trace::SpanData as OtelSpanData;
use opentelemetry_sdk::trace::SpanExporter;

use super::types as genkit_trace; // Alias our internal types
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// A static, mutable holder for the telemetry server URL.
static TELEMETRY_SERVER_URL: Lazy<Arc<Mutex<Option<String>>>> =
    Lazy::new(|| Arc::new(Mutex::new(None)));

/// Sets the global URL for the telemetry server.
pub fn set_telemetry_server_url(url: String) {
    let mut guard = TELEMETRY_SERVER_URL.lock().unwrap();
    *guard = Some(url);
}

/// An exporter that sends spans to a Genkit telemetry server.
#[derive(Debug, Clone)]
pub struct TraceServerExporter {
    client: reqwest::Client,
}

impl Default for TraceServerExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceServerExporter {
    /// Creates a new `TraceServerExporter`.
    pub fn new() -> Self {
        TraceServerExporter {
            client: reqwest::Client::new(),
        }
    }

    /// Converts an OpenTelemetry `SpanData` into a Genkit-serializable `SpanData`.
    fn span_kind_to_string(&self, kind: &opentelemetry::trace::SpanKind) -> String {
        match kind {
            opentelemetry::trace::SpanKind::Client => "CLIENT".to_string(),
            opentelemetry::trace::SpanKind::Server => "SERVER".to_string(),
            opentelemetry::trace::SpanKind::Producer => "PRODUCER".to_string(),
            opentelemetry::trace::SpanKind::Consumer => "CONSUMER".to_string(),
            opentelemetry::trace::SpanKind::Internal => "INTERNAL".to_string(),
        }
    }

    fn convert_span(&self, span: &OtelSpanData) -> genkit_trace::SpanData {
        genkit_trace::SpanData {
            span_id: span.span_context.span_id().to_string(),
            trace_id: span.span_context.trace_id().to_string(),
            parent_span_id: if span.parent_span_id.to_string() == "0000000000000000" {
                None
            } else {
                Some(span.parent_span_id.to_string())
            },
            start_time: span
                .start_time
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            end_time: span
                .end_time
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            attributes: span
                .attributes
                .iter()
                .map(|kv: &KeyValue| {
                    (
                        kv.key.to_string(),
                        serde_json::to_value(kv.value.to_string()).unwrap(),
                    )
                })
                .collect(),
            display_name: span.name.to_string(),
            links: Some(
                span.links
                    .iter()
                    .map(|link| genkit_trace::Link {
                        context: Some(genkit_trace::SpanContext {
                            trace_id: link.span_context.trace_id().to_string(),
                            span_id: link.span_context.span_id().to_string(),
                            is_remote: Some(link.span_context.is_remote()),
                            trace_flags: link.span_context.trace_flags().to_u8(),
                        }),
                        attributes: Some(
                            link.attributes
                                .iter()
                                .map(|kv: &KeyValue| {
                                    (
                                        kv.key.to_string(),
                                        serde_json::to_value(kv.value.to_string()).unwrap(),
                                    )
                                })
                                .collect(),
                        ),
                        dropped_attributes_count: Some(link.dropped_attributes_count),
                    })
                    .collect(),
            ),
            instrumentation_library: genkit_trace::InstrumentationLibrary {
                name: span.instrumentation_scope.name().to_string(),
                version: span.instrumentation_scope.version().map(|v| v.to_string()),
                schema_url: span
                    .instrumentation_scope
                    .schema_url()
                    .map(|v| v.to_string()),
            },
            span_kind: self.span_kind_to_string(&span.span_kind),
            status: match &span.status {
                OtelStatus::Ok => Some(genkit_trace::SpanStatus {
                    code: 0,
                    message: None,
                }),
                OtelStatus::Error { description } => Some(genkit_trace::SpanStatus {
                    code: 2,
                    message: Some(description.to_string()),
                }),
                OtelStatus::Unset => None,
            },
            time_events: Some(genkit_trace::TimeEvents {
                time_event: span
                    .events
                    .iter()
                    .map(|e| genkit_trace::TimeEvent {
                        time: e
                            .timestamp
                            .duration_since(SystemTime::UNIX_EPOCH)
                            .unwrap()
                            .as_millis() as u64,
                        annotation: genkit_trace::Annotation {
                            description: e.name.to_string(),
                            attributes: e
                                .attributes
                                .iter()
                                .map(|kv: &KeyValue| {
                                    (
                                        kv.key.to_string(),
                                        serde_json::to_value(kv.value.to_string()).unwrap(),
                                    )
                                })
                                .collect(),
                        },
                    })
                    .collect(),
            }),
            ..Default::default()
        }
    }

    /// Saves a batch of spans belonging to a single trace to the telemetry server.
    async fn save(
        &self,
        trace_id: String,
        spans: Vec<&OtelSpanData>,
    ) -> Result<(), reqwest::Error> {
        let base_url = if let Some(url) = TELEMETRY_SERVER_URL.lock().unwrap().as_ref() {
            url.clone()
        } else {
            return Ok(());
        };

        let mut genkit_spans: HashMap<String, genkit_trace::SpanData> = HashMap::new();
        let mut root_span: Option<&OtelSpanData> = None;

        for span in &spans {
            if span.parent_span_id.to_string() == "0000000000000000" {
                root_span = Some(span);
            }
            genkit_spans.insert(
                span.span_context.span_id().to_string(),
                self.convert_span(span),
            );
        }

        let trace_data = if let Some(root) = root_span {
            genkit_trace::TraceData {
                trace_id,
                display_name: Some(root.name.to_string()),
                start_time: Some(
                    root.start_time
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                ),
                end_time: Some(
                    root.end_time
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                ),
                spans: genkit_spans,
            }
        } else {
            genkit_trace::TraceData {
                trace_id,
                spans: genkit_spans,
                ..Default::default()
            }
        };

        let target_url = format!("{}/api/traces", base_url);
        self.client
            .post(target_url)
            .json(&trace_data)
            .send()
            .await?
            .error_for_status()?;

        Ok(())
    }
}

impl SpanExporter for TraceServerExporter {
    fn export(
        &self,
        batch: Vec<OtelSpanData>,
    ) -> impl futures::Future<Output = OTelSdkResult> + std::marker::Send {
        let exporter = self.clone();
        Box::pin(async move {
            let mut traces: HashMap<String, Vec<&OtelSpanData>> = HashMap::new();
            for span in &batch {
                traces
                    .entry(span.span_context.trace_id().to_string())
                    .or_default()
                    .push(span);
            }

            let futures = traces
                .into_iter()
                .map(|(trace_id, spans)| exporter.save(trace_id, spans));

            let results = futures_util::future::join_all(futures).await;

            if results.iter().any(|res| res.is_err()) {
                return Err(OTelSdkError::InternalFailure(
                    "One or more traces failed to export to telemetry server.".to_owned(),
                ));
            }

            Ok(())
        })
    }

    fn shutdown(&mut self) -> OTelSdkResult {
        todo!()
    }

    fn force_flush(&mut self) -> OTelSdkResult {
        todo!()
    }
}
