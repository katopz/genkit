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

//! # JSON Output Formatter
//!
//! This module provides a formatter for JSON output. It uses a lenient JSON
//! extractor to find and parse JSON objects or arrays from model-generated text.
//! It is the Rust equivalent of `formats/json.ts`.

use super::types::{Format, Formatter, FormatterConfig};
use crate::extract::extract_json;
use crate::generate::GenerateResponseChunk;
use crate::message::Message;
use schemars::Schema;
use serde_json::{self, Value};
use std::sync::Arc;

/// A struct that implements the `Format` trait for JSON data.
#[derive(Debug)]
struct JsonFormat {
    instructions: Option<String>,
}

impl Format for JsonFormat {
    /// Parses a JSON object from the text content of a complete `Message`.
    fn parse_message(&self, message: &Message) -> Value {
        extract_json(&message.text())
            .unwrap_or(None)
            .unwrap_or(Value::Null)
    }

    /// Parses a JSON object from the accumulated text of a streaming `GenerateResponseChunk`.
    fn parse_chunk(&self, chunk: &GenerateResponseChunk) -> Option<Value> {
        extract_json(&chunk.accumulated_text()).ok().flatten()
    }

    /// Provides model instructions for generating JSON that conforms to a schema.
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }
}

/// Creates and configures the `json` formatter.
pub fn json_formatter() -> Formatter {
    Formatter {
        name: "json".to_string(),
        config: FormatterConfig {
            format: Some("json".to_string()),
            content_type: Some("application/json".to_string()),
            constrained: Some(true),
            default_instructions: Some(false),
        },
        handler: Arc::new(|schema: Option<&Schema>| {
            let instructions = schema.map(|s| {
                format!(
                    "Output should be in JSON format and conform to the following schema:\n\n```\n{}\n```\n",
                    serde_json::to_string_pretty(s).unwrap_or_default()
                )
            });
            Box::new(JsonFormat { instructions })
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{document::Part, generate::chunk::GenerateResponseChunkOptions};
    use serde_json::json;

    #[test]
    fn test_parse_message_with_preamble() {
        let formatter = json_formatter();
        let handler = (formatter.handler)(None);
        let message = Message::new(
            crate::message::MessageData {
                role: crate::message::Role::Model,
                content: vec![Part {
                    text: Some("Here is the JSON:\n```json\n{\"foo\": \"bar\"}\n```".to_string()),
                    ..Default::default()
                }],
                metadata: None,
            },
            None,
        );
        let parsed = handler.parse_message(&message);
        assert_eq!(parsed, json!({"foo": "bar"}));
    }

    #[test]
    fn test_parse_chunk_streaming() {
        let formatter = json_formatter();
        let handler = (formatter.handler)(None);

        let mut chunks = Vec::new();
        let chunk1_data = crate::model::GenerateResponseChunkData {
            index: 0,
            content: vec![Part {
                text: Some("{\"foo\": ".to_string()),
                ..Default::default()
            }],
            usage: None,
            custom: None,
            role: None,
        };
        let chunk1 = GenerateResponseChunk::new(
            chunk1_data.clone(),
            GenerateResponseChunkOptions {
                index: Some(0u32),
                role: Some(crate::message::Role::Model),
                previous_chunks: chunks.clone(),
            },
        );
        chunks.push(chunk1_data);
        let parsed1 = handler.parse_chunk(&chunk1);
        assert_eq!(parsed1, None); // json5 does not parse `{"foo": ` as `{"foo": null}`

        let chunk2_data = crate::model::GenerateResponseChunkData {
            index: 0,
            content: vec![Part {
                text: Some("\"bar\"}".to_string()),
                ..Default::default()
            }],
            usage: None,
            custom: None,
            role: None,
        };
        let chunk2 = GenerateResponseChunk::new(
            chunk2_data.clone(),
            GenerateResponseChunkOptions {
                index: Some(0u32),
                role: Some(crate::message::Role::Model),
                previous_chunks: chunks.clone(),
            },
        );
        let parsed2 = handler.parse_chunk(&chunk2);
        assert_eq!(parsed2, Some(json!({"foo": "bar"})));
    }

    #[test]
    fn test_instructions_generation() {
        let formatter = json_formatter();
        let schema = schemars::schema_for!(serde_json::Value);
        let handler = (formatter.handler)(Some(&schema));
        let instructions = handler.instructions().unwrap();

        assert!(instructions.contains("Output should be in JSON format"));
        assert!(instructions.contains(&serde_json::to_string_pretty(&schema).unwrap()));
    }
}
