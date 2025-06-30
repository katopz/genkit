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

// Re-export key tracing components from `genkit-core` and OpenTelemetry.
pub use genkit_core::tracing::exporter::TraceServerExporter;
pub use genkit_core::tracing::types::PathMetadata;
pub use genkit_core::tracing::{enable_telemetry, flush_tracing, in_new_span as run_in_new_span};
pub use opentelemetry_sdk::export::trace::SpanData;

// Note: The following items from the TypeScript version have different names,
// are not yet available in the Rust version, or are handled by the Rust
// type system and `serde` instead of direct schema exports.
//
// - SPAN_TYPE_ATTR: This constant is not yet public in `genkit-core`.
// - SpanContextSchema
// - SpanDataSchema
// - SpanMetadataSchema
// - SpanStatusSchema
// - TimeEventSchema
// - TraceDataSchema
// - TraceMetadataSchema
// - appendSpan
// - newTrace
// - setCustomMetadataAttribute
// - setCustomMetadataAttributes
// - setTelemetryServerUrl
// - toDisplayPath
// - SpanMetadata
// - TraceData
// - TraceMetadata
