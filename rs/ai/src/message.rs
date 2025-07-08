// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may
// obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # AI Message Representation
//!
//! This module provides the `Message` struct, a primary data structure for
//! representing a single turn in a conversation with a generative model. It is
//! the Rust equivalent of `message.ts`.

use crate::document::{Media, Part, ToolRequest, ToolResponse};
use crate::extract::extract_json;
use genkit_core::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use strum_macros::Display;

/// Represents the role of the entity creating a message.
#[derive(Default, Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Display)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    System,
    #[default]
    User,
    Model,
    Tool,
}

/// A serializable representation of a message's data.
#[derive(Default, Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct MessageData {
    pub role: Role,
    pub content: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

impl MessageData {
    /// Creates a new `MessageData` with the `User` role.
    pub fn user(content: Vec<Part>) -> Self {
        Self {
            role: Role::User,
            content,
            ..Default::default()
        }
    }

    /// Creates a new `MessageData` with the `System` role.
    pub fn system(content: Vec<Part>) -> Self {
        Self {
            role: Role::System,
            content,
            ..Default::default()
        }
    }
}

/// A function type for parsing the content of a `Message` into a structured format.
pub type MessageParser<T> = Arc<dyn Fn(&Message<T>) -> Result<T> + Send + Sync>;

/// Represents a single message in a conversation.
///
/// A `Message` consists of a `role` (who is speaking) and `content` (what is being said),
/// which can be made up of multiple `Part`s (e.g., text and images).
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
pub struct Message<O = Value> {
    pub role: Role,
    pub content: Vec<Part>,
    pub metadata: Option<HashMap<String, Value>>,
    #[serde(skip)]
    pub parser: Option<MessageParser<O>>,
}

impl<O> std::fmt::Debug for Message<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Message")
            .field("role", &self.role)
            .field("content", &self.content)
            .field("metadata", &self.metadata)
            .finish_non_exhaustive()
    }
}

impl<O> Message<O>
where
    O: for<'de> Deserialize<'de> + 'static,
{
    /// Creates a new `Message`.
    pub fn new(data: MessageData, parser: Option<MessageParser<O>>) -> Self {
        Self {
            role: data.role,
            content: data.content,
            metadata: data.metadata,
            parser,
        }
    }

    /// Sets the parser for this message.
    pub fn set_parser(&mut self, parser: MessageParser<O>) {
        self.parser = Some(parser);
    }

    /// Normalizes different content formats into a `Vec<Part>`.
    pub fn parse_content(lenient_part: Value) -> Result<Vec<Part>> {
        if let Some(s) = lenient_part.as_str() {
            return Ok(vec![Part {
                text: Some(s.to_string()),
                ..Default::default()
            }]);
        }
        if lenient_part.is_object() {
            return serde_json::from_value(lenient_part)
                .map(|p| vec![p])
                .map_err(|e| Error::new_internal(format!("Failed to parse Part: {}", e)));
        }
        if lenient_part.is_array() {
            return serde_json::from_value(lenient_part)
                .map_err(|e| Error::new_internal(format!("Failed to parse Vec<Part>: {}", e)));
        }
        Err(Error::new_internal(format!(
            "Unsupported content format: {}",
            lenient_part
        )))
    }

    /// Attempts to parse the message content into a structured type `O`.
    ///
    /// The parsing logic is as follows:
    /// 1. If a custom `parser` is provided, it is used.
    /// 2. Otherwise, it looks for a `Part` with a `data` field and returns that.
    /// 3. As a final fallback, it concatenates all `text` parts and tries to parse
    ///    it as JSON. If that fails, it will attempt to use the raw text if the
    ///    requested type is a string.
    pub fn output(&self) -> Result<O> {
        println!(
            "[Message::output] called. Parser is_some: {}",
            self.parser.is_some()
        );
        if let Some(parser) = &self.parser {
            return parser(self);
        }
        if let Some(data) = self.data() {
            return serde_json::from_value(data).map_err(|e| {
                Error::new_internal(format!("Failed to deserialize data part: {}", e))
            });
        }
        let text = self.text();
        match extract_json(&text)? {
            Some(value) => serde_json::from_value(value).map_err(|e| {
                Error::new_internal(format!("Failed to deserialize structured output: {}", e))
            }),
            None => {
                // Attempt to treat the whole text as a string output, if that's what is expected.
                serde_json::from_value(Value::String(text)).map_err(|_| {
                    Error::new_internal("No structured output found in message".to_string())
                })
            }
        }
    }

    /// Extracts and concatenates all `text` from the message's parts.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| part.text.as_deref())
            .collect::<String>()
    }

    /// Extracts and concatenates all `reasoning` from the message's parts.
    pub fn reasoning(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| part.reasoning.as_deref())
            .collect::<String>()
    }

    /// Returns the first `media` part found in the message, if any.
    pub fn media(&self) -> Option<&Media> {
        self.content.iter().find_map(|part| part.media.as_ref())
    }

    /// Extracts all `toolRequest` parts from the message.
    pub fn tool_requests(&self) -> Vec<&ToolRequest> {
        self.content
            .iter()
            .filter_map(|part| part.tool_request.as_ref())
            .collect()
    }

    /// Extracts all `toolResponse` parts from the message.
    pub fn tool_response_parts(&self) -> Vec<&ToolResponse> {
        self.content
            .iter()
            .filter_map(|part| part.tool_response.as_ref())
            .collect()
    }

    /// Returns all tool requests that are marked as interrupts.
    pub fn interrupts(&self) -> Vec<&ToolRequest> {
        self.content
            .iter()
            .filter(|part| {
                part.tool_request.is_some()
                    && part
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("interrupt"))
                        .is_some()
            })
            .map(|part| part.tool_request.as_ref().unwrap())
            .collect()
    }

    /// Returns the first `data` part found in the message, if any.
    pub fn data(&self) -> Option<Value> {
        self.content.iter().find_map(|part| part.data.clone())
    }

    /// Converts the `Message` into its serializable `MessageData` representation.
    pub fn to_json(&self) -> MessageData {
        MessageData {
            role: self.role.clone(),
            content: self.content.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Implementation for `MessageData` to easily extract concatenated text from its parts.
impl MessageData {
    /// Creates a new `MessageData` from a `serde_json::Value`.
    ///
    /// This function handles several flexible input formats:
    /// - A raw JSON string is converted into a `user` role message.
    /// - A JSON object with a `content` field that is a string is normalized
    ///   to have `content` as an array of `Part`s.
    /// - A JSON object that already matches the `MessageData` structure is
    ///   deserialized directly.
    pub fn from_value(value: Value) -> Result<Self> {
        // Case 1: Raw string
        if let Some(s) = value.as_str() {
            return Ok(MessageData {
                role: Role::User,
                content: vec![Part::text(s)],
                ..Default::default()
            });
        }

        // Case 2 & 3: Object
        if let Value::Object(mut map) = value {
            // Check if content is a string
            if let Some(content_val) = map.get("content") {
                if content_val.is_string() {
                    // Normalize content to Vec<Part>
                    let text_content = content_val.as_str().unwrap().to_string();
                    let part_array = serde_json::json!([{"text": text_content}]);
                    map.insert("content".to_string(), part_array);
                }
            }
            // Deserialize from the (potentially modified) map
            return serde_json::from_value(Value::Object(map))
                .map_err(|e| Error::new_internal(e.to_string()));
        }

        Err(Error::new_internal(format!(
            "Unsupported message format: expected string or object, got {:?}",
            value
        )))
    }

    /// Concatenates all text from the `content` parts into a single String.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| part.text.as_deref())
            .collect::<Vec<&str>>()
            .join("")
    }
}
