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

//! # AI Content Generation
//!
//! This module provides the primary `generate` function for interacting with
//! generative models. It is the Rust equivalent of `generate.ts`.

// Declare sub-modules first.
pub mod action;
pub mod chunk;
pub mod resolve_tool_requests;
pub mod response;

// Re-export key structs.
pub use self::chunk::GenerateResponseChunk;
pub use self::response::GenerateResponse;

use crate::document::{Document, Part};
use crate::formats::FormatterConfig;
use crate::message::{MessageData, Role};
use crate::model::GenerateRequest; // Assuming Model is a struct/enum representing a model reference
use crate::tool::{self, ToolArgument};
use futures_util::stream::Stream;
use genkit_core::error::{Error, Result};
use genkit_core::registry::Registry;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

/// Specifies how tools should be called by the model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ToolChoice {
    Auto,
    Required,
    None,
}

/// Configuration for the desired output of a generation request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct OutputOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Value>, // Can be bool or string
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>, // Represents Zod schema or JSON Schema
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained: Option<bool>,
}

/// Options for a `generate` call.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateOptions {
    pub model: Option<crate::model::Model>,
    pub system: Option<Vec<Part>>,
    pub prompt: Option<Vec<Part>>,
    pub docs: Option<Vec<Document>>,
    pub messages: Option<Vec<MessageData>>,
    pub tools: Option<Vec<ToolArgument>>,
    pub tool_choice: Option<ToolChoice>,
    pub config: Option<Value>,
    pub output: Option<OutputOptions>,
    pub max_turns: Option<u32>,
    pub return_tool_requests: Option<bool>,
}

/// Base generate options used by `Chat`.
#[derive(Debug, Clone, Default)]
pub struct BaseGenerateOptions {
    pub model: Option<crate::model::Model>,
    pub docs: Option<Vec<Document>>,
    pub messages: Vec<MessageData>,
    pub tools: Option<Vec<ToolArgument>>,
    pub tool_choice: Option<ToolChoice>,
    pub config: Option<Value>,
    pub output: Option<OutputOptions>,
}

/// The response from a `generate_stream` call.
pub struct GenerateStreamResponse<O = serde_json::Value> {
    /// A stream of response chunks.
    pub stream: Pin<Box<dyn Stream<Item = Result<GenerateResponseChunk<O>>> + Send>>,
    /// A handle to the final, complete response.
    pub response: tokio::task::JoinHandle<Result<GenerateResponse<O>>>,
}

/// An error that occurs during generation.
#[derive(Debug)]
pub struct GenerationResponseError {
    pub response: GenerateResponse,
    pub source_error: Error,
}

impl fmt::Display for GenerationResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Generation failed: {}. Response: {:?}",
            self.source_error, self.response
        )
    }
}

impl std::error::Error for GenerationResponseError {}

/// An error indicating the generation was blocked.
#[derive(Debug)]
pub struct GenerationBlockedError(pub GenerationResponseError);
impl fmt::Display for GenerationBlockedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Generation blocked: {}", self.0)
    }
}
impl std::error::Error for GenerationBlockedError {}

/// Generates content using a model.
pub async fn generate<O>(
    registry: &Registry,
    options: GenerateOptions,
) -> Result<GenerateResponse<O>>
where
    O: for<'de> DeserializeOwned + Send + Sync + 'static,
{
    let helper_options = action::GenerateHelperOptions {
        raw_request: options.clone(),
        current_turn: 0,
        message_index: 0,
    };
    action::generate_helper(Arc::new(registry.clone()), helper_options).await
}

/// Generates content and streams the response.
pub fn generate_stream<O>(
    _registry: &Registry,
    _options: GenerateOptions,
) -> GenerateStreamResponse<O>
where
    O: for<'de> DeserializeOwned + Send + Sync + 'static,
{
    unimplemented!();
}

/// Converts `GenerateOptions` to a `GenerateRequest`.
pub async fn to_generate_request(
    registry: &Registry,
    options: &GenerateOptions,
) -> Result<GenerateRequest> {
    let mut messages: Vec<MessageData> = Vec::new();
    if let Some(system_parts) = &options.system {
        messages.push(MessageData {
            role: Role::System,
            content: system_parts.clone(),
            metadata: None,
        });
    }
    if let Some(user_messages) = &options.messages {
        messages.extend(user_messages.clone());
    }
    if let Some(prompt_parts) = &options.prompt {
        messages.push(MessageData {
            role: Role::User,
            content: prompt_parts.clone(),
            metadata: None,
        });
    }

    if messages.is_empty() {
        return Err(Error::new_internal(
            "No messages provided for generation".to_string(),
        ));
    }

    let resolved_tools = tool::resolve_tools(registry, options.tools.as_deref()).await?;
    let tool_definitions = if !resolved_tools.is_empty() {
        Some(
            resolved_tools
                .iter()
                .map(|t| tool::to_tool_definition(t))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };

    // This is a simplified version of the TS implementation that handles instruction injection.
    Ok(GenerateRequest {
        messages,
        config: options.config.clone(),
        tools: tool_definitions,
        output: options.output.as_ref().map(|opts| FormatterConfig {
            format: opts.format.clone(),
            content_type: opts.content_type.clone(),
            constrained: opts.constrained,
            ..Default::default()
        }),
        docs: options.docs.clone(),
        tool_choice: options.tool_choice.clone(),
        max_turns: options.max_turns,
        return_tool_requests: options.return_tool_requests,
    })
}
