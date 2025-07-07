// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Streaming Generation Chunk
//!
//! This module defines the `GenerateResponseChunk` struct, representing a piece
//! of a streaming response from a generative model. It is the Rust equivalent
// of `generate/chunk.ts`.

use crate::document::Part;
use crate::extract::extract_json;
use crate::message::Role;
use crate::model::GenerateResponseChunkData;
use genkit_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::sync::Arc;

/// A function type for parsing a `GenerateResponseChunk` into a structured format.
pub type ChunkParser<O> = Arc<dyn Fn(&GenerateResponseChunk<O>) -> Result<O> + Send + Sync>;

/// Represents a chunk of a streaming response from a model.
#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseChunk<O = Value> {
    pub index: u32,
    #[serde(default)]
    pub role: Option<Role>,
    pub content: Vec<Part>,
    pub custom: Option<Value>,
    #[serde(default)]
    pub previous_chunks: Option<Vec<GenerateResponseChunkData>>,

    #[serde(skip)]
    pub parser: Option<ChunkParser<O>>,
}

impl<O> fmt::Debug for GenerateResponseChunk<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenerateResponseChunk")
            .field("index", &self.index)
            .field("role", &self.role)
            .field("content", &self.content)
            .field("custom", &self.custom)
            .field("previous_chunks", &self.previous_chunks)
            .field("parser", &self.parser.as_ref().map(|_| "Some(<fn>)"))
            .finish()
    }
}

/// Options for creating a `GenerateResponseChunk`.
#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseChunkOptions {
    pub previous_chunks: Vec<GenerateResponseChunkData>,
    pub role: Option<Role>,
    pub index: Option<u32>,
}

impl<O> GenerateResponseChunk<O>
where
    O: for<'de> Deserialize<'de> + 'static,
{
    /// Creates a new `GenerateResponseChunk`.
    pub fn new(data: GenerateResponseChunkData, options: GenerateResponseChunkOptions) -> Self {
        Self {
            index: data.index,
            role: data.role.or(options.role),
            content: data.content,
            custom: data.custom,
            previous_chunks: Some(options.previous_chunks),
            parser: None, // This correctly initializes the skipped field.
        }
    }

    /// Concatenates all `text` parts present in the chunk.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect::<String>()
    }

    /// Concatenates all `reasoning` parts present in the chunk.
    pub fn reasoning(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| p.reasoning.as_deref())
            .collect::<String>()
    }

    /// Concatenates all `text` parts of all preceding chunks.
    pub fn previous_text(&self) -> String {
        self.previous_chunks
            .as_deref()
            .unwrap_or_default()
            .iter()
            .flat_map(|c| &c.content)
            .filter_map(|p| p.text.as_deref())
            .collect::<String>()
    }

    /// Concatenates all `text` parts of all chunks from the response thus far.
    pub fn accumulated_text(&self) -> String {
        let mut text = self.previous_text();
        text.push_str(&self.text());
        text
    }

    /// Returns all tool requests found in this chunk.
    pub fn tool_requests(&self) -> Vec<&Part> {
        self.content
            .iter()
            .filter(|p| p.tool_request.is_some())
            .collect()
    }

    /// Parses the chunk into the desired output format `O`.
    pub fn output(&self) -> Result<O> {
        if let Some(parser) = &self.parser {
            return parser(self);
        }
        // Fallback to naive JSON parsing.
        extract_json(&self.accumulated_text())?
            .ok_or_else(|| Error::new_internal("No structured output found in chunk".to_string()))
    }

    /// Converts the `GenerateResponseChunk` to its serializable data representation.
    pub fn to_json(&self) -> GenerateResponseChunkData {
        GenerateResponseChunkData {
            index: self.index,
            content: self.content.clone(),
            usage: None, // Usage is typically aggregated at the end.
            custom: self.custom.clone(),
            role: self.role.clone(),
        }
    }
}

impl GenerateResponseChunk<Value> {
    /// Creates a new `GenerateResponseChunk` from `serde_json::Value`.
    ///
    /// This is primarily for convenience in testing.
    pub fn from_json(data: Value, options: Value) -> Result<Self> {
        let data: GenerateResponseChunkData = serde_json::from_value(data)?;
        let options: GenerateResponseChunkOptions = serde_json::from_value(options)?;
        Ok(Self::new(data, options))
    }
}
