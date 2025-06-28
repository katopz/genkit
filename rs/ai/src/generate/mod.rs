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

use crate::document::{Document, Part, ToolRequestPart, ToolResponsePart};
use crate::generate::action::ModelMiddleware;
use crate::message::{MessageData, Role};
use crate::model::GenerateRequest;
use crate::tool::{self, ToolArgument};
use futures_util::stream::Stream;
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::Registry;
use genkit_core::status::StatusCode;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Specifies how tools should be called by the model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ToolChoice {
    Auto,
    Required,
    None,
}

/// A callback that receives streaming chunks of a specified type.
pub type OnChunkCallback<O> = Arc<dyn Fn(GenerateResponseChunk<O>) -> Result<()> + Send + Sync>;

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

/// Configures how to resume generation after an interrupt.
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ResumeOptions {
    /// A list of `toolResponse` parts corresponding to interrupt `toolRequest` parts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub respond: Option<Vec<ToolResponsePart>>,
    /// A list of `toolRequest` parts to re-run.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restart: Option<Vec<ToolRequestPart>>,
    /// Additional metadata to annotate the created tool message with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

/// Options for a `generate` call. This struct is now generic to support typed `on_chunk` callbacks.
#[derive(Clone, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateOptions<O = Value> {
    pub model: Option<crate::model::Model>,
    pub system: Option<Vec<Part>>,
    pub prompt: Option<Vec<Part>>,
    pub docs: Option<Vec<Document>>,
    pub messages: Option<Vec<MessageData>>,
    pub tools: Option<Vec<ToolArgument>>,
    pub tool_choice: Option<ToolChoice>,
    pub config: Option<Value>,
    pub output: Option<OutputOptions>,
    pub resume: Option<ResumeOptions>,
    pub max_turns: Option<u32>,
    pub return_tool_requests: Option<bool>,
    /// Middleware to be used with this model call.
    #[serde(skip)]
    pub r#use: Option<Vec<ModelMiddleware>>,
    /// Additional context (data, like e.g. auth) to be passed down to tools, prompts and other sub actions.
    #[serde(skip)]
    pub context: Option<ActionContext>,
    /// When provided, models supporting streaming will call this callback with chunks.
    #[serde(skip)]
    pub on_chunk: Option<OnChunkCallback<O>>,
    #[serde(skip)]
    pub _marker: PhantomData<O>,
}

impl<O> Default for GenerateOptions<O> {
    fn default() -> Self {
        Self {
            model: Default::default(),
            system: Default::default(),
            prompt: Default::default(),
            docs: Default::default(),
            messages: Default::default(),
            tools: Default::default(),
            tool_choice: Default::default(),
            config: Default::default(),
            output: Default::default(),
            resume: Default::default(),
            max_turns: Default::default(),
            return_tool_requests: Default::default(),
            r#use: Default::default(),
            context: Default::default(),
            on_chunk: Default::default(),
            _marker: PhantomData,
        }
    }
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

/// An error that occurs during generation, containing the full response for inspection.
#[derive(Debug)]
pub struct GenerationResponseError<O> {
    pub response: GenerateResponse<O>,
    pub message: String,
}

impl<O: fmt::Debug + Clone> fmt::Display for GenerationResponseError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Generation failed: {}. Response: {:?}",
            self.message, self.response
        )
    }
}

impl<O: fmt::Debug + Clone + Send + Sync + 'static> std::error::Error
    for GenerationResponseError<O>
{
}

impl<O: Serialize + Send + Sync + 'static> From<GenerationResponseError<O>> for Error {
    fn from(err: GenerationResponseError<O>) -> Self {
        let details = serde_json::to_value(&err.response).ok();
        Error::new_user_facing(StatusCode::FailedPrecondition, err.message, details)
    }
}

/// An error indicating the generation was blocked due to safety settings or other reasons.
#[derive(Debug)]
pub struct GenerationBlockedError<O>(pub GenerationResponseError<O>);

impl<O: fmt::Debug + Clone> fmt::Display for GenerationBlockedError<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Generation blocked: {}", self.0)
    }
}

impl<O: fmt::Debug + Clone + Send + Sync + 'static> std::error::Error
    for GenerationBlockedError<O>
{
}

impl<O: Serialize + Send + Sync + 'static> From<GenerationBlockedError<O>> for Error {
    fn from(err: GenerationBlockedError<O>) -> Self {
        let details = serde_json::to_value(&err.0.response).ok();
        Error::new_blocked_error(StatusCode::FailedPrecondition, err.0.message, details)
    }
}

/// Generates content using a model.
pub async fn generate<O>(
    registry: &Registry,
    mut options: GenerateOptions<O>,
) -> Result<GenerateResponse<O>>
where
    O: Clone
        + Default
        + for<'de> DeserializeOwned
        + Serialize
        + Send
        + Sync
        + 'static
        + std::fmt::Debug,
{
    // The `on_chunk` callback is handled by `run_with_streaming_callback`, so we take it here.
    let on_chunk_callback = options.on_chunk.take();
    let registry_clone = registry.clone();

    // The `run_with_streaming_callback` function from genkit-core would set the
    // callback in a task-local context, which can then be retrieved by the model action.
    // This is a placeholder for that logic.
    let helper_options = action::GenerateHelperOptions {
        middleware: options.r#use.take().unwrap_or_default(),
        raw_request: options,
        current_turn: 0,
        message_index: 0,
        on_chunk: on_chunk_callback,
    };
    action::generate_helper(Arc::new(registry_clone), helper_options).await
}

/// Generates content and streams the response.
pub fn generate_stream<O>(
    registry: &Registry,
    options: GenerateOptions<O>,
) -> GenerateStreamResponse<O>
where
    O: Clone
        + Default
        + for<'de> DeserializeOwned
        + Serialize
        + Send
        + Sync
        + 'static
        + std::fmt::Debug,
{
    let (tx, rx) = mpsc::channel(128);
    let mut stream_options = options;

    // Create a callback that sends chunks received from the generation logic
    // into the sender part of our channel.
    let on_chunk: OnChunkCallback<O> = Arc::new(move |chunk| {
        // We use try_send to avoid blocking if the receiver is slow.
        // If the receiver has been dropped, this will error, so we ignore it.
        let _ = tx.try_send(Ok(chunk));
        Ok(())
    });

    stream_options.on_chunk = Some(on_chunk);
    let registry_clone = registry.clone();

    let response_handle =
        tokio::spawn(async move { generate(&registry_clone, stream_options).await });

    let stream = ReceiverStream::new(rx);

    GenerateStreamResponse {
        stream: Box::pin(stream),
        response: response_handle,
    }
}

/// Converts `GenerateOptions` to a `GenerateRequest`.
pub async fn to_generate_request<O>(
    registry: &Registry,
    options: &GenerateOptions<O>,
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
                .map(|t| tool::to_tool_definition(t.as_ref()))
                .collect::<Result<Vec<_>>>()?,
        )
    } else {
        None
    };

    Ok(GenerateRequest {
        messages,
        config: options.config.clone(),
        tools: tool_definitions,
        output: options.output.clone(),
        docs: options.docs.clone(),
        tool_choice: options.tool_choice.clone(),
        max_turns: options.max_turns,
        return_tool_requests: options.return_tool_requests,
        // The `resume` field is handled before this stage and is not part of the final model request.
    })
}
