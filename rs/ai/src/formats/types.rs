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

//! # Output Formatting Types
//!
//! This module defines the core traits and structs for handling model output
//! formatting, which is the Rust equivalent of `formats/types.ts`.

use schemars::{JsonSchema, Schema};
use serde::{Deserialize, Serialize};
use serde_json::Value;

// These will be replaced with actual types from other modules once they are ported.
use crate::generate::GenerateResponseChunk;
use crate::message::Message;

/// Configuration for an output format.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct FormatterConfig {
    /// The format identifier, e.g., "json", "text".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// The IANA media type for the output, e.g., "application/json".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Whether the model should be constrained to produce this format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained: Option<bool>,
    /// Whether to provide default instructions to the model for this format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default_instructions: Option<bool>,
}

/// A trait for an object that can parse model outputs.
///
/// Implementations of this trait handle the logic for parsing full messages
/// and streaming chunks for a specific format. The output types are fixed to
/// `serde_json::Value` to allow for dynamic dispatch in the `Formatter` struct.
pub trait Format {
    /// Parses the content of a complete `Message` into a JSON `Value`.
    fn parse_message(&self, message: &Message) -> Value;

    /// Parses a streaming `GenerateResponseChunk` into a JSON `Value`.
    ///
    /// This method is optional as not all formats support streaming parsing.
    fn parse_chunk(&self, _chunk: &GenerateResponseChunk) -> Option<Value> {
        None
    }

    /// Provides instructions for the model on how to generate the output in this format.
    ///
    /// This is optional and is only used if the user does not provide their own
    /// custom instructions.
    fn instructions(&self) -> Option<String> {
        None
    }
}

/// Represents a named output format handler.
///
/// This struct ties together the formatter's name, its configuration, and the
/// factory function (`handler`) that creates a `Format` trait object.
pub struct Formatter {
    pub name: String,
    pub config: FormatterConfig,
    /// A factory function that, given an optional JSON schema, returns a
    /// boxed trait object that can handle the parsing for this format.
    pub handler: Box<dyn Fn(Option<&Schema>) -> Box<dyn Format + Send + Sync> + Send + Sync>,
}

impl std::fmt::Debug for Formatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Formatter")
            .field("name", &self.name)
            .field("config", &self.config)
            .finish()
    }
}
