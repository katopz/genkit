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
use crate::formats::{self};
use crate::generate::action::run_with_streaming_callback;
use crate::message::{Message, MessageData, Role};
use crate::model::middleware::ModelMiddleware;
use crate::model::GenerateRequest;
use crate::tool::{self, ToolArgument};
use crate::GenerateResponseData;
use futures_util::stream::Stream;
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::Registry;
use genkit_core::status::StatusCode;
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use strum_macros::EnumString;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Specifies how tools should be called by the model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, EnumString)]
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
#[derive(Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateOptions<O = Value> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<crate::model::Model>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<Part>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Vec<Part>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<Vec<Document>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<MessageData>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolArgument>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume: Option<ResumeOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_tool_requests: Option<bool>,
    /// Middleware to be used with this model call.
    #[serde(skip)]
    pub r#use: Option<Vec<ModelMiddleware>>,
    /// Additional context (data, like e.g. auth) to be passed down to tools, prompts and other sub actions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<ActionContext>,
    /// When provided, models supporting streaming will call this callback with chunks.
    #[serde(skip)]
    pub on_chunk: Option<OnChunkCallback<O>>,
    #[serde(skip)]
    pub _marker: PhantomData<O>,
}

impl<O> fmt::Debug for GenerateOptions<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenerateOptions")
            .field("model", &self.model)
            .field("system", &self.system)
            .field("prompt", &self.prompt)
            .field("docs", &self.docs)
            .field("messages", &self.messages)
            .field("tools", &self.tools)
            .field("tool_choice", &self.tool_choice)
            .field("config", &self.config)
            .field("output", &self.output)
            .field("resume", &self.resume)
            .field("max_turns", &self.max_turns)
            .field("return_tool_requests", &self.return_tool_requests)
            .field("use", &self.r#use.as_ref().map(|v| v.len()))
            .field("context", &self.context)
            .field("on_chunk", &self.on_chunk.as_ref().map(|_| "Some(<fn>)"))
            .field("_marker", &self._marker)
            .finish()
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
    // If no model is specified in the options, try to use the default from the registry.
    if options.model.is_none() {
        if let Some(model_name) = registry.get_default_model() {
            options.model = Some(crate::model::Model::Name(model_name));
        }
    }

    let output_options = options.output.clone();

    let on_chunk_callback = options.on_chunk.take();
    let registry_clone = registry.clone();

    // FIX #1: Add explicit type annotation for the trait object.
    let core_streaming_callback: Option<
        genkit_core::action::StreamingCallback<GenerateResponseChunk<O>>,
    > = on_chunk_callback.map(|user_callback| {
        let new_cb: genkit_core::action::StreamingCallback<GenerateResponseChunk<O>> = Arc::new(
            move |chunk_result: Result<GenerateResponseChunk<O>, genkit_core::Error>| {
                if let Ok(chunk) = chunk_result {
                    let _ = user_callback(chunk);
                }
            },
        );
        new_cb
    });

    // FIX #3: Wrap the async block in a no-argument closure `||`.
    let mut response = run_with_streaming_callback(core_streaming_callback, || async move {
        let helper_options = action::GenerateHelperOptions {
            middleware: options.r#use.take().unwrap_or_default(),
            raw_request: options,
            current_turn: 0,
            message_index: 0,
        };
        action::generate_helper(Arc::new(registry_clone), helper_options).await
    })
    .await?;

    if let Some(formatter) = formats::resolve_format(registry, output_options.as_ref()).await {
        let schema: Option<schemars::Schema> = output_options
            .as_ref()
            .and_then(|o| o.schema.as_ref())
            .and_then(|v| serde_json::from_value(v.clone()).ok());

        let format_impl = (formatter.handler)(schema.as_ref());

        let parser_closure = move |msg: &Message<O>| {
            let value_msg = Message::<Value>::new(msg.to_json(), None);

            let parsed_value = format_impl.parse_message(&value_msg);
            serde_json::from_value(parsed_value)
                .map_err(|e| Error::new_internal(format!("Format parser error: {}", e)))
        };

        let parser_arc = Arc::new(parser_closure);

        response.parser = Some(parser_arc.clone());
        if let Some(message) = response.message.as_mut() {
            message.set_parser(parser_arc);
        }
    }

    Ok(response)
}

/// Generates content and streams the response.
pub async fn generate_stream<O>(
    registry: &Registry,
    mut options: GenerateOptions<O>,
) -> Result<GenerateStreamResponse<O>>
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
    println!(
        "[STREAM] Called. options.on_chunk.is_some(): {}",
        options.on_chunk.is_some()
    );

    let output_options = options.output.clone();
    let chunk_parser =
        if let Some(formatter) = formats::resolve_format(registry, output_options.as_ref()).await {
            println!("[STREAM] Found formatter: {}", formatter.name);
            let schema: Option<schemars::Schema> = output_options
                .as_ref()
                .and_then(|o| o.schema.as_ref())
                .and_then(|v| serde_json::from_value(v.clone()).ok());
            let format_impl = (formatter.handler)(schema.as_ref());

            let parser_closure = move |chunk: &GenerateResponseChunk<O>| {
                // Because the chunk parser in formats/ expects a `Value` chunk,
                // we have to do this conversion.
                let value_chunk_data = chunk.to_json();
                let value_chunk = GenerateResponseChunk::<Value>::new(
                    value_chunk_data,
                    chunk::GenerateResponseChunkOptions {
                        previous_chunks: chunk.previous_chunks.clone().unwrap_or_default(),
                        role: chunk.role.clone(),
                        index: Some(chunk.index),
                    },
                );

                if let Some(parsed_val) = format_impl.parse_chunk(&value_chunk) {
                    serde_json::from_value(parsed_val).map_err(|e| {
                        Error::new_internal(format!(
                            "Failed to deserialize value from chunk parser: {}",
                            e
                        ))
                    })
                } else {
                    Err(Error::new_internal(
                        "Chunk parser returned None".to_string(),
                    ))
                }
            };
            Some(Arc::new(parser_closure) as chunk::ChunkParser<O>)
        } else {
            println!("[STREAM] No formatter found.");
            None
        };

    let (tx, rx) = mpsc::channel(128);

    // Create a callback that sends chunks into the channel.
    let tx_for_callback = tx.clone();
    let on_chunk: OnChunkCallback<O> = Arc::new(move |mut chunk| {
        println!("[STREAM] on_chunk callback invoked. Attaching parser and sending chunk to stream channel.");
        // Attach the parser we just created.
        chunk.parser = chunk_parser.clone();
        // We use try_send to avoid blocking.
        let _ = tx_for_callback.try_send(Ok(chunk));
        Ok(())
    });
    options.on_chunk = Some(on_chunk);

    let registry_clone = registry.clone();

    // Spawn the background generation task.
    println!("[STREAM] Spawning background task to call generate().");
    let response_handle = tokio::spawn(async move {
        println!(
            "[STREAM_TASK] Started. Calling generate() with on_chunk.is_some(): {}",
            options.on_chunk.is_some()
        );
        let final_result = generate(&registry_clone, options).await;
        println!(
            "[STREAM_TASK] generate() finished. Result is_ok(): {}",
            final_result.is_ok()
        );

        if let Err(e) = &final_result {
            let new_error = Error::new_internal(e.to_string());
            let _ = tx.send(Err(new_error)).await;
        }
        final_result
    });

    let stream = ReceiverStream::new(rx);

    Ok(GenerateStreamResponse {
        stream: Box::pin(stream),
        response: response_handle,
    })
}

/// Starts a long-running generation operation.
pub async fn generate_operation<O>(
    registry: &Registry,
    options: GenerateOptions<O>,
) -> Result<genkit_core::background_action::Operation<GenerateResponseData>>
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
    let model_ref = options
        .model
        .as_ref()
        .ok_or_else(|| Error::new_internal("Model not specified".to_string()))?;

    let model_name = match model_ref {
        crate::model::Model::Reference(r) => r.name.clone(),
        crate::model::Model::Name(n) => n.clone(),
    };
    let action_key = format!("/background-model/{}", model_name);

    let action = registry.lookup_action(&action_key).await.ok_or_else(|| {
        Error::new_internal(format!("Background model '{}' not found", model_name))
    })?;

    let request = to_generate_request(registry, &options).await?;
    let request_value =
        serde_json::to_value(request).map_err(|e| Error::new_internal(e.to_string()))?;
    let op_value = action.run_http_json(request_value, None).await?;
    let operation: genkit_core::background_action::Operation<GenerateResponseData> =
        serde_json::from_value(op_value).map_err(|e| Error::new_internal(e.to_string()))?;

    Ok(operation)
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

    let mut merged_config = serde_json::Map::new();

    if let Some(crate::model::Model::Reference(model_ref)) = &options.model {
        // 1. Add base version from ModelRef.version
        if let Some(version) = &model_ref.version {
            merged_config.insert("version".to_string(), json!(version));
        }
        // 2. Merge config from ModelRef.config (overwrites base version if key exists)
        if let Some(Value::Object(map)) = &model_ref.config {
            merged_config.extend(map.clone());
        }
    }

    // 3. Merge config from GenerateOptions (overwrites anything from ModelRef)
    if let Some(Value::Object(map)) = &options.config {
        merged_config.extend(map.clone());
    }

    let final_config = if merged_config.is_empty() {
        None
    } else {
        Some(Value::Object(merged_config))
    };

    let tools = Some(
        tool::resolve_tools(registry, options.tools.as_deref())
            .await?
            .iter()
            .map(|t| tool::to_tool_definition(t.as_ref()))
            .collect::<Result<Vec<_>>>()?,
    );

    Ok(GenerateRequest {
        messages,
        config: final_config, // Use the merged config
        tools,
        output: options
            .output
            .as_ref()
            .and_then(|o| serde_json::to_value(o).ok()),
        docs: options.docs.clone(),
        tool_choice: options
            .tool_choice
            .as_ref()
            .map(|tc| serde_json::to_value(tc).unwrap()),
        max_turns: options.max_turns,
        return_tool_requests: options.return_tool_requests,
        // The `resume` field is handled before this stage and is not part of the final model request.
    })
}
