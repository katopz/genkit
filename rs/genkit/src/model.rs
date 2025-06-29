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

//! # Genkit Model
//!
//! This module defines the core data structures for interacting with generative
//! models in the Genkit framework. It is the Rust equivalent of `@genkit-ai/ai/model`.

use crate::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;

/// Represents the originator of a `Message`.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub enum Role {
    /// The user asking a question or providing an instruction.
    #[default]
    User,
    /// The generative model providing a response.
    Model,
    /// A special role for providing system-level instructions or context to the model.
    System,
    /// A special role for providing the output of a tool back to the model.
    Tool,
}

/// Represents a simple text part of a message.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
pub struct TextPart {
    pub text: String,
}

/// Represents a request for a tool to be executed.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolRequestPart {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#ref: Option<String>,
}

/// Represents the response from a tool execution.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolResponsePart {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#ref: Option<String>,
}

/// Represents a single piece of content within a `Message`.
/// This is the Rust equivalent of the `Part` type in the TypeScript SDK.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
pub enum Part {
    Text(TextPart),
    ToolRequest(ToolRequestPart),
    ToolResponse(ToolResponsePart),
}

/// Represents a single message in a conversation history.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Part>,
}

impl Message {
    /// Creates a new user message.
    pub fn user(content: Vec<Part>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }
}

/// Represents a request to a generative model.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateRequest {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    // TODO: Add tools, output config, etc.
}

/// The reason why a model finished generating a response.
#[derive(Default, Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
#[serde(rename_all = "camelCase")]
pub enum FinishReason {
    /// The model finished generating naturally.
    #[default]
    Stop,
    /// The model generated the maximum number of tokens requested.
    MaxTokens,
    /// The model's response was blocked due to safety settings.
    Safety,
    /// The model recited content from a protected source.
    Recitation,
    /// The model called a tool.
    ToolCode,
    /// An unknown reason.
    Other,
}

/// A single response candidate from a generative model.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    pub message: Message,
    pub finish_reason: FinishReason,
    // TODO: Add index, safety ratings, citations, etc.
}

/// The full response from a `generate` call.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponse {
    pub candidates: Vec<Candidate>,
    // TODO: Add usage stats, custom fields, etc.
}

impl GenerateResponse {
    /// Helper to get the text content of the first candidate, if it exists.
    pub fn text(&self) -> Option<String> {
        self.candidates
            .first()
            .and_then(|c| c.message.content.first())
            .and_then(|p| match p {
                Part::Text(text_part) => Some(text_part.text.clone()),
                _ => None,
            })
    }
}

/// A trait representing a generative model that can be used with Genkit.
pub trait Model: Send + Sync {
    /// Get information about the model.
    fn info(&self) -> &ModelInfo;
    /// Generate a response based on a request.
    fn generate<'a>(
        &'a self,
        request: GenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<GenerateResponse>> + Send + 'a>>;
}

/// Information about a generative model.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub supports_system_role: bool,
}

/// A high-level function to generate content with a model.
///
/// Note: This is a placeholder. A real implementation would involve a registry
/// to look up the model by name and invoke its `generate` method.
pub async fn generate(request: GenerateRequest) -> Result<GenerateResponse> {
    // Placeholder logic
    println!(
        "Simulating generate call with {} messages...",
        request.messages.len()
    );

    // For now, return a mock response.
    let response_message = Message {
        role: Role::Model,
        content: vec![Part::Text(TextPart {
            text: "This is a mock response.".to_string(),
        })],
    };

    Ok(GenerateResponse {
        candidates: vec![Candidate {
            message: response_message,
            finish_reason: FinishReason::Stop,
        }],
    })
}
