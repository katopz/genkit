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

//! # Text Output Formatter
//!
//! This module provides a simple formatter for plain text output.
//! It is the Rust equivalent of `formats/text.ts`.

use super::types::{Format, Formatter, FormatterConfig};
use crate::generate::chunk::GenerateResponseChunk;
use crate::message::Message;
use schemars::Schema;
use serde_json::Value;
use std::sync::Arc;

/// A struct that implements the `Format` trait for plain text.
#[derive(Debug)]
struct TextFormat;

impl Format for TextFormat {
    /// Parses the text content from a complete `Message`.
    fn parse_message(&self, message: &Message) -> Value {
        Value::String(message.text())
    }

    /// Parses the text content from an incremental `GenerateResponseChunk`.
    fn parse_chunk(&self, chunk: &GenerateResponseChunk) -> Option<Value> {
        Some(Value::String(chunk.text()))
    }
}

/// Creates and configures the `text` formatter.
pub fn text_formatter() -> Formatter {
    Formatter {
        name: "text".to_string(),
        config: FormatterConfig {
            content_type: Some("text/plain".to_string()),
            ..Default::default()
        },
        handler: Arc::new(|_schema: Option<&Schema>| Box::new(TextFormat)),
    }
}
