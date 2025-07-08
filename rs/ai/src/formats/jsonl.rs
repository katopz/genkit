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

//! # JSONL Output Formatter
//!
//! This module provides a formatter for JSONL (JSON Lines) output, where each
//! line is a separate JSON object. It is the Rust equivalent of `formats/jsonl.ts`.

use super::types::{Format, Formatter, FormatterConfig};
use crate::extract::extract_json;
use crate::generate::GenerateResponseChunk;
use crate::message::Message;
use schemars::Schema;
use serde_json::{self, Value};
use std::sync::Arc;

/// A struct that implements the `Format` trait for JSONL data.
#[derive(Debug)]
struct JsonlFormat {
    instructions: Option<String>,
}

impl Format for JsonlFormat {
    /// Parses a sequence of JSON objects from the text content of a complete `Message`.
    fn parse_message(&self, message: &Message) -> Value {
        let items: Vec<Value> = message
            .text()
            .lines()
            .filter_map(|line| extract_json(line.trim()).ok().flatten())
            .collect();
        Value::Array(items)
    }

    /// Parses a sequence of JSON objects from the accumulated text of a streaming `GenerateResponseChunk`.
    fn parse_chunk(&self, chunk: &GenerateResponseChunk) -> Option<Value> {
        let mut results: Vec<Value> = Vec::new();

        let text = chunk.accumulated_text();
        let prev_text = chunk.previous_text();

        let start_index = if !prev_text.is_empty() {
            prev_text.rfind('\n').map_or(0, |i| i + 1)
        } else {
            0
        };

        let new_text_segment = &text[start_index..];

        for line in new_text_segment.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('{') {
                if let Ok(result) = json5::from_str(trimmed) {
                    results.push(result);
                } else {
                    // Stop if a line looks like JSON but fails to parse,
                    // assuming it's an incomplete object.
                    break;
                }
            }
        }

        Some(Value::Array(results))
    }

    /// Provides model instructions for generating JSONL that conforms to a schema.
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }
}

/// Creates and configures the `jsonl` formatter.
pub fn jsonl_formatter() -> Formatter {
    Formatter {
        name: "jsonl".to_string(),
        config: FormatterConfig {
            content_type: Some("application/jsonl".to_string()),
            ..Default::default()
        },
        handler: Arc::new(|schema: Option<&Schema>| {
            let mut instructions: Option<String> = None;
            if let Some(s) = schema {
                if let Some(schema_obj) = s.as_object() {
                    let mut is_array_of_objects = false;
                    // Check if the root schema is an array
                    if let Some(Value::String(root_type)) = schema_obj.get("type") {
                        if root_type == "array" {
                            // If it's an array, check its items
                            if let Some(items) = schema_obj.get("items") {
                                if let Some(items_obj) = items.as_object() {
                                    if let Some(Value::String(item_type)) = items_obj.get("type") {
                                        if item_type == "object" {
                                            is_array_of_objects = true;
                                        }
                                    } else if items_obj.get("properties").is_some() {
                                        is_array_of_objects = true;
                                    }
                                }
                            }
                        }
                    }

                    if !is_array_of_objects {
                        // Using eprintln! for diagnostic output as it goes to stderr.
                        // A more robust implementation might return a Result.
                        eprintln!("A 'jsonl' format was requested, but the provided schema is not an array of objects. Instructions may be incorrect.");
                    }

                    // Generate instructions if we can get the item schema
                    if let Some(items_schema) = schema_obj.get("items") {
                        instructions = Some(format!(
                            "Output should be JSONL format, a sequence of JSON objects (one per line) separated by a newline `\\n` character. Each line should be a JSON object conforming to the following schema:\n\n```\n{}\n```\n",
                            serde_json::to_string_pretty(items_schema).unwrap_or_default()
                        ));
                    }
                }
            }
            Box::new(JsonlFormat { instructions })
        }),
    }
}
