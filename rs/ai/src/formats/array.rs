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

//! # JSON Array Output Formatter
//!
//! This module provides a formatter for JSON array output. It is the Rust
//! equivalent of `formats/array.ts`.

use super::types::{Format, Formatter, FormatterConfig};
use crate::extract::extract_items;
use crate::generate::GenerateResponseChunk;
use crate::message::Message;
use schemars::Schema;
use serde_json::Value;
use std::sync::Arc;

/// A struct that implements the `Format` trait for JSON array data.
#[derive(Debug)]
struct ArrayFormat {
    instructions: Option<String>,
}

impl Format for ArrayFormat {
    /// Parses a JSON array from the text content of a complete `Message`.
    fn parse_message(&self, message: &Message) -> Value {
        let result = extract_items(&message.text(), 0);
        Value::Array(result.items)
    }

    /// Parses JSON objects from a streaming `GenerateResponseChunk`.
    fn parse_chunk(&self, chunk: &GenerateResponseChunk) -> Option<Value> {
        // This is a simplified streaming parser. To correctly handle the cursor,
        // we determine the end of the last fully parsed section and start
        // from there.
        let cursor = if chunk
            .previous_chunks
            .as_deref()
            .unwrap_or_default()
            .is_empty()
        {
            0
        } else {
            // Re-run extract_items on the previous text to find out where we left off.
            extract_items(&chunk.previous_text(), 0).cursor
        };

        // Now extract new items from the accumulated text, starting from the determined cursor.
        let result = extract_items(&chunk.accumulated_text(), cursor);
        Some(Value::Array(result.items))
    }

    /// Provides model instructions for generating a JSON array that conforms to a schema.
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }
}

/// Creates and configures the `array` formatter.
pub fn array_formatter() -> Formatter {
    Formatter {
        name: "array".to_string(),
        config: FormatterConfig {
            content_type: Some("application/json".to_string()),
            constrained: Some(true),
            ..Default::default()
        },
        handler: Arc::new(|schema: Option<&Schema>| {
            let mut instructions: Option<String> = None;
            if let Some(s) = schema {
                let is_array = s
                    .as_value()
                    .as_object()
                    .and_then(|obj| obj.get("type"))
                    .and_then(|t| t.as_str())
                    == Some("array");

                if !is_array {
                    // TODO: Replace with proper logging.
                    println!(
                        "Warning: An 'array' format was requested, but the provided schema is not of type 'array'."
                    );
                }

                instructions = Some(format!(
                    "Output should be a JSON array conforming to the following schema:\n\n```\n{}\n```\n",
                    serde_json::to_string_pretty(s).unwrap_or_default()
                ));
            }
            Box::new(ArrayFormat { instructions })
        }),
    }
}
