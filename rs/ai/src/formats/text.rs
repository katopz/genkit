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
        handler: Box::new(|_schema: Option<&Schema>| Box::new(TextFormat)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Part;
    use crate::generate::chunk::{GenerateResponseChunk, GenerateResponseChunkOptions};
    use crate::message::{Message, MessageData, Role};
    use crate::model::GenerateResponseChunkData;
    use serde_json::Value;

    #[test]
    fn test_streaming_parser() {
        let formatter = text_formatter();
        let handler = (formatter.handler)(None);

        // Test case 1: "emits text chunks as they arrive"
        let chunk1_data = GenerateResponseChunkData {
            index: 0,
            role: Some(Role::Model),
            content: vec![Part {
                text: Some("Hello".to_string()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let chunk1 =
            GenerateResponseChunk::new(chunk1_data, GenerateResponseChunkOptions::default());
        assert_eq!(
            handler.parse_chunk(&chunk1),
            Some(Value::String("Hello".to_string()))
        );

        let chunk2_data = GenerateResponseChunkData {
            index: 0,
            role: Some(Role::Model),
            content: vec![Part {
                text: Some(" world".to_string()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let chunk2 =
            GenerateResponseChunk::new(chunk2_data, GenerateResponseChunkOptions::default());
        assert_eq!(
            handler.parse_chunk(&chunk2),
            Some(Value::String(" world".to_string()))
        );

        // Test case 2: "handles empty chunks"
        let empty_chunk_data = GenerateResponseChunkData {
            index: 0,
            role: Some(Role::Model),
            content: vec![Part {
                text: Some("".to_string()),
                ..Default::default()
            }],
            ..Default::default()
        };
        let empty_chunk =
            GenerateResponseChunk::new(empty_chunk_data, GenerateResponseChunkOptions::default());
        assert_eq!(
            handler.parse_chunk(&empty_chunk),
            Some(Value::String("".to_string()))
        );
    }

    #[test]
    fn test_message_parser() {
        let formatter = text_formatter();
        let handler = (formatter.handler)(None);

        // Test case 1: Parses complete text response
        let message1 = Message::new(
            MessageData {
                role: Role::Model,
                content: vec![Part {
                    text: Some("Hello world".to_string()),
                    ..Default::default()
                }],
                metadata: None,
            },
            None,
        );
        assert_eq!(
            handler.parse_message(&message1),
            Value::String("Hello world".to_string())
        );

        // Test case 2: Handles empty response
        let message2 = Message::new(
            MessageData {
                role: Role::Model,
                content: vec![Part {
                    text: Some("".to_string()),
                    ..Default::default()
                }],
                metadata: None,
            },
            None,
        );
        assert_eq!(
            handler.parse_message(&message2),
            Value::String("".to_string())
        );
    }
}
