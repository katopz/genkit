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

//! # Generate Action
//!
//! This module defines the `generate` action and the core implementation logic
//! for content generation, including tool handling and streaming. It is the
// Rust equivalent of `generate/action.ts`.

use super::{response::GenerateResponse, GenerateOptions};
use crate::document::Part;
use crate::generate::{OutputOptions, ToolChoice};
use crate::message::{Message, MessageData, Role};
use crate::model::{self, GenerateRequest, GenerateResponseChunkData, GenerateResponseData};
use crate::{formats, to_generate_request, Document, Model};
use genkit_core::action::{Action, ActionBuilder, ActionFnArg, StreamingCallback};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use genkit_core::tracing;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use tokio::task_local;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::StreamExt;

pub type NextFn = Box<
    dyn FnOnce(
            GenerateRequest,
        ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
        + Send,
>;

pub type ModelMiddleware = Arc<
    dyn Fn(
            GenerateRequest,
            NextFn,
        ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
        + Send
        + Sync,
>;

/// The serializable options for a `generate` action.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateActionOptions {
    pub model: Option<Model>,
    pub system: Option<Vec<Part>>,
    pub prompt: Option<Vec<Part>>,
    pub docs: Option<Vec<Document>>,
    pub messages: Option<Vec<MessageData>>,
    pub tools: Option<Vec<String>>,
    pub tool_choice: Option<ToolChoice>,
    pub config: Option<Value>,
    pub output: Option<OutputOptions>,
    pub max_turns: Option<u32>,
    pub return_tool_requests: Option<bool>,
}

/// A type alias for a `generate` action.
pub type GenerateAction = Action<GenerateOptions, GenerateResponse, GenerateResponseChunkData>;

/// Options for the `generate_helper` function.
#[derive(Clone)]
pub struct GenerateHelperOptions<O: 'static> {
    pub raw_request: GenerateOptions<O>,
    pub middleware: Vec<ModelMiddleware>,
    pub current_turn: u32,
    pub message_index: u32,
}

/// Defines (registers) a utility `generate` action.
pub fn define_generate_action(registry: &mut Registry) -> Arc<GenerateAction> {
    let registry_clone = Arc::new(registry.clone());
    let generate_action = ActionBuilder::new(
        ActionType::Util,
        "generate".to_string(),
        move |req: GenerateOptions, _ctx: ActionFnArg<GenerateResponseChunkData>| {
            let registry = registry_clone.clone();
            async move {
                let mut attrs = HashMap::new();
                attrs.insert(
                    "genkit:spanType".to_string(),
                    Value::String("util".to_string()),
                );

                let raw_request_for_response = req.clone();
                let registry_for_response = registry.clone();

                let (response_data, _telemetry) = tracing::in_new_span(
                    "generate".to_string(),
                    Some(attrs),
                    move |_trace_context| async move {
                        let mut request = req;
                        // The action does not support streaming callbacks directly.
                        let on_chunk = request.on_chunk.take();
                        let mut current_turn = 0;
                        let mut message_index = 0;

                        let response_data_result = loop {
                            let max_turns = request.max_turns.unwrap_or(5);
                            if current_turn >= max_turns {
                                break Err(Error::new_internal(format!(
                                    "Exceeded maximum tool call iterations ({})",
                                    max_turns
                                )));
                            }

                            let resume_result =
                                super::resolve_tool_requests::resolve_resume_option(
                                    registry.as_ref(),
                                    request,
                                )
                                .await?;
                            if let Some(interrupted_response) = resume_result.interrupted_response {
                                break Ok(interrupted_response);
                            }
                            request = resume_result.revised_request.unwrap();
                            if let Some(tool_msg) = resume_result.tool_message {
                                if let Some(cb) = &on_chunk {
                                    let chunk = super::chunk::GenerateResponseChunk::new(
                                        GenerateResponseChunkData {
                                            index: message_index,
                                            content: tool_msg.content,
                                            ..Default::default()
                                        },
                                        super::chunk::GenerateResponseChunkOptions {
                                            role: Some(Role::Tool),
                                            ..Default::default()
                                        },
                                    );
                                    cb(chunk)?;
                                }
                            }

                            let model_ref = request.model.as_ref().ok_or_else(|| {
                                Error::new_internal("Model not specified".to_string())
                            })?;
                            let model_name = match model_ref {
                                model::Model::Reference(r) => r.name.clone(),
                                model::Model::Name(n) => n.clone(),
                            };
                            let erased_action = registry
                                .lookup_action(&format!("/model/{}", model_name))
                                .await
                                .ok_or_else(|| {
                                    Error::new_internal(format!("Model '{}' not found", model_name))
                                })?;
                            let model_action = erased_action;

                            if let Some(format) =
                                formats::resolve_format(&registry, request.output.as_ref()).await
                            {
                                let schema_value = request
                                    .output
                                    .as_ref()
                                    .and_then(|opts| opts.schema.as_ref());
                                let schema: Option<schemars::Schema> = schema_value
                                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                                let instructions_option = request
                                    .output
                                    .as_ref()
                                    .and_then(|opts| opts.instructions.as_ref());

                                let instructions = formats::resolve_instructions(
                                    Some(format.as_ref()),
                                    schema.as_ref(),
                                    instructions_option,
                                );
                                if instructions.is_some() {
                                    let messages = request.messages.get_or_insert_with(Vec::new);
                                    let updated_messages =
                                        formats::inject_instructions(messages, instructions);
                                    request.messages = Some(updated_messages);
                                }

                                if let Some(output_opts) = request.output.as_mut() {
                                    output_opts.format = Some(format.name.clone());
                                    output_opts.content_type = format.config.content_type.clone();
                                    output_opts.constrained = format.config.constrained;
                                }
                            }

                            let generate_request = to_generate_request(&registry, &request).await?;

                            let req_value = serde_json::to_value(generate_request).unwrap();
                            let resp_value = model_action.run_http_json(req_value, None).await?;
                            let response_data: GenerateResponseData =
                                serde_json::from_value(resp_value).map_err(|e| {
                                    Error::new_internal(format!(
                                        "Failed to deserialize model response: {}",
                                        e
                                    ))
                                })?;

                            let generated_message = match response_data.candidates.first() {
                                Some(c) => c.message.clone(),
                                None => {
                                    break Err(Error::new_internal(
                                        "Model did not return a candidate".to_string(),
                                    ))
                                }
                            };

                            let tool_requests: Vec<_> = generated_message
                                .content
                                .iter()
                                .filter(|part| part.tool_request.is_some())
                                .collect();

                            if tool_requests.is_empty()
                                || request.return_tool_requests == Some(true)
                            {
                                break Ok(response_data);
                            }

                            let tool_results = super::resolve_tool_requests::resolve_tool_requests(
                                &registry,
                                &request,
                                &generated_message,
                            )
                            .await?;

                            if let Some(revised_model_message) = tool_results.revised_model_message
                            {
                                let mut final_response = response_data;
                                if let Some(candidate) = final_response.candidates.get_mut(0) {
                                    candidate.message = revised_model_message;
                                    candidate.finish_reason =
                                        Some(model::FinishReason::Interrupted);
                                    candidate.finish_message = Some(
                                        "One or more tool calls resulted in interrupts."
                                            .to_string(),
                                    );
                                }
                                break Ok(final_response);
                            }

                            let mut messages = request.messages.take().unwrap_or_default();
                            messages.push(generated_message);
                            if let Some(tool_message) = tool_results.tool_message {
                                messages.push(tool_message);
                            }
                            request.messages = Some(messages);

                            current_turn += 1;
                            message_index += 1;
                        };
                        response_data_result
                    },
                )
                .await?;

                let request_for_response =
                    to_generate_request(&registry_for_response, &raw_request_for_response).await?;
                let mut response =
                    GenerateResponse::new(&response_data, Some(request_for_response.clone()));

                response.assert_valid()?;

                if let Some(output_options) = raw_request_for_response.output.as_ref() {
                    if let Some(format) =
                        formats::resolve_format(&registry_for_response, Some(output_options)).await
                    {
                        let schema_val = raw_request_for_response
                            .output
                            .as_ref()
                            .and_then(|out| out.schema.as_ref());
                        let schema: Option<schemars::Schema> =
                            schema_val.and_then(|v| serde_json::from_value(v.clone()).ok());
                        let handler = (format.handler)(schema.as_ref());
                        let parser = move |msg: &crate::message::Message<Value>| {
                            let value_msg = Message::new(msg.to_json(), None);
                            let parsed_value = handler.parse_message(&value_msg);
                            serde_json::from_value(parsed_value).map_err(|e| {
                                Error::new_internal(format!(
                                    "Failed to deserialize formatted output: {}",
                                    e
                                ))
                            })
                        };
                        response.parser = Some(Arc::new(parser));
                    }
                }

                Ok(response)
            }
        },
    )
    .build();
    Arc::new(generate_action)
}

/// Encapsulates all generate logic, similar to `generateAction` but callable internally.
pub async fn generate_helper<O>(
    registry: Arc<Registry>,
    options: GenerateHelperOptions<O>,
) -> Result<GenerateResponse<O>>
where
    O: Clone
        + Default
        + for<'de> DeserializeOwned
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + 'static,
{
    let mut attrs = HashMap::new();
    attrs.insert(
        "genkit:spanType".to_string(),
        Value::String("util".to_string()),
    );
    let raw_request_for_response = options.raw_request.clone();
    let registry_for_response = registry.clone();
    let (response_data, _telemetry) = tracing::in_new_span(
        "generate".to_string(),
        Some(attrs),
        move |_trace_context| async move { generate_internal(registry, options).await },
    )
    .await?;

    let request = to_generate_request(&registry_for_response, &raw_request_for_response).await?;
    let mut response = GenerateResponse::new(&response_data, Some(request.clone()));

    response.assert_valid()?;

    // Resolve formatter and attach parser to the response.
    if let Some(output_options) = raw_request_for_response.output.as_ref() {
        if let Some(format) =
            formats::resolve_format(&registry_for_response, Some(output_options)).await
        {
            let schema_val = raw_request_for_response
                .output
                .as_ref()
                .and_then(|out| out.schema.as_ref());
            let schema: Option<schemars::Schema> =
                schema_val.and_then(|v| serde_json::from_value(v.clone()).ok());
            let handler = (format.handler)(schema.as_ref());
            let parser = move |msg: &crate::message::Message<O>| {
                let value_msg = Message::new(msg.to_json(), None);
                let parsed_value = handler.parse_message(&value_msg);
                serde_json::from_value(parsed_value).map_err(|e| {
                    Error::new_internal(format!("Failed to deserialize formatted output: {}", e))
                })
            };
            response.parser = Some(Arc::new(parser));
        }
    }

    Ok(response)
}

/// Builds the middleware chain and calls the underlying model action.
async fn run_model_via_middleware(
    model_action: Arc<dyn ErasedAction>,
    request: GenerateRequest,
    middleware: Vec<ModelMiddleware>,
) -> Result<GenerateResponseData> {
    let mut chain = Box::new(move |req: GenerateRequest| {
        let model_action = model_action.clone();

        Box::pin(async move {
            let req_value = serde_json::to_value(req).unwrap();

            // Get the callback from the context instead of a parameter.
            if let Some(chunk_callback) = get_streaming_callback() {
                // STREAMING PATH
                let mut streaming_response = model_action.stream_http_json(req_value, None)?;

                // Spawn a task to forward stream chunks to the callback.
                tokio::spawn(async move {
                    while let Some(chunk_result) = streaming_response.stream.next().await {
                        chunk_callback(chunk_result);
                    }
                });

                // Wait for the final response.
                let mut final_resp_value = streaming_response.output.await?;
                if let Some(result) = final_resp_value.get_mut("result") {
                    final_resp_value = result.take();
                }
                serde_json::from_value(final_resp_value)
                    .map_err(|e| Error::new_internal(format!("Failed to deserialize: {}", e)))
            } else {
                // NON-STREAMING PATH
                let mut resp_value = model_action.run_http_json(req_value, None).await?;
                if let Some(result) = resp_value.get_mut("result") {
                    resp_value = result.take();
                }
                serde_json::from_value(resp_value)
                    .map_err(|e| Error::new_internal(format!("Failed to deserialize: {}", e)))
            }
        }) as Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
    }) as NextFn;

    // Wrap the chain with each middleware, in reverse order.
    for mw in middleware.into_iter().rev() {
        let next_fn = chain;
        chain = Box::new(move |req| Box::pin(mw(req, next_fn)));
    }

    // Execute the full chain.
    chain(request).await
}

/// The core, private implementation of the generation logic.
fn generate_internal<O>(
    registry: Arc<Registry>,
    options: GenerateHelperOptions<O>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<GenerateResponseData>> + Send>>
where
    O: Default
        + for<'de> DeserializeOwned
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + 'static
        + Clone,
{
    Box::pin(async move {
        // Destructuring no longer includes `on_chunk`
        let GenerateHelperOptions {
            raw_request,
            middleware,
            current_turn,
            message_index,
        } = options;
        let mut request = raw_request;

        // 1. Resolve model.
        let model_ref = request
            .model
            .as_ref()
            .ok_or_else(|| Error::new_internal("Model not specified".to_string()))?;
        let model_name = match model_ref {
            model::Model::Reference(r) => r.name.clone(),
            model::Model::Name(n) => n.clone(),
        };
        let erased_action = registry
            .lookup_action(&format!("/model/{}", model_name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Model '{}' not found", model_name)))?;
        let model_action = erased_action;

        // 2. Resolve and apply format.
        if let Some(format) = formats::resolve_format(&registry, request.output.as_ref()).await {
            let schema_value = request
                .output
                .as_ref()
                .and_then(|opts| opts.schema.as_ref());
            let schema: Option<schemars::Schema> =
                schema_value.and_then(|v| serde_json::from_value(v.clone()).ok());
            let instructions_option = request
                .output
                .as_ref()
                .and_then(|opts| opts.instructions.as_ref());

            let instructions = formats::resolve_instructions(
                Some(format.as_ref()),
                schema.as_ref(),
                instructions_option,
            );
            if instructions.is_some() {
                let messages = request.messages.get_or_insert_with(Vec::new);
                let updated_messages = formats::inject_instructions(messages, instructions);
                request.messages = Some(updated_messages);
            }

            if let Some(output_opts) = request.output.as_mut() {
                output_opts.format = Some(format.name.clone());
                output_opts.content_type = format.config.content_type.clone();
                output_opts.constrained = format.config.constrained;
            }
        }

        // 3. Handle tool loop recursion check.
        let max_turns = request.max_turns.unwrap_or(5);
        if current_turn >= max_turns {
            return Err(Error::new_internal(format!(
                "Exceeded maximum tool call iterations ({})",
                max_turns
            )));
        }

        // 4. Apply resume logic BEFORE the main model call.
        let resume_result =
            super::resolve_tool_requests::resolve_resume_option(registry.as_ref(), request).await?;
        if let Some(interrupted_response) = resume_result.interrupted_response {
            return Ok(interrupted_response);
        }
        let mut request = resume_result.revised_request.unwrap();

        // This block now gets the callback from the context.
        if let Some(tool_msg) = resume_result.tool_message {
            if let Some(cb) = get_streaming_callback() {
                let chunk_data = GenerateResponseChunkData {
                    index: message_index,
                    content: tool_msg.content,
                    ..Default::default()
                };
                let chunk_value = serde_json::to_value(chunk_data)
                    .map_err(|e| Error::new_internal(e.to_string()))?;
                cb(Ok(chunk_value));
            }
        }

        // 5. Convert to GenerateRequest (the type the model action expects).
        let generate_request = to_generate_request(&registry, &request).await?;

        // 6. Call the model action (no longer passing the callback).
        let response_data =
            run_model_via_middleware(model_action, generate_request, middleware.clone()).await?;

        let generated_message = match response_data.candidates.first() {
            Some(c) => c.message.clone(),
            None => {
                return Err(Error::new_internal(
                    "Model did not return a candidate".to_string(),
                ))
            }
        };

        let tool_requests: Vec<_> = generated_message
            .content
            .iter()
            .filter(|part| part.tool_request.is_some())
            .collect();

        if tool_requests.is_empty() || request.return_tool_requests == Some(true) {
            return Ok(response_data);
        }

        // 7. Handle tool requests returned from model
        let tool_results = super::resolve_tool_requests::resolve_tool_requests(
            &registry,
            &request,
            &generated_message,
        )
        .await?;

        if let Some(revised_model_message) = tool_results.revised_model_message {
            let mut final_response = response_data;
            if let Some(candidate) = final_response.candidates.get_mut(0) {
                candidate.message = revised_model_message;
                candidate.finish_reason = Some(model::FinishReason::Interrupted);
                candidate.finish_message =
                    Some("One or more tool calls resulted in interrupts.".to_string());
            }
            return Ok(final_response);
        }

        let mut messages = request.messages.take().unwrap_or_default();
        messages.push(generated_message);
        if let Some(tool_message) = tool_results.tool_message {
            messages.push(tool_message);
        }
        request.messages = Some(messages);

        // 8. Recursive call (no longer passing the callback).
        generate_internal(
            registry,
            GenerateHelperOptions {
                raw_request: request,
                middleware,
                current_turn: current_turn + 1,
                message_index: message_index + 1,
            },
        )
        .await
    })
}

/// A type-erased streaming callback that operates on raw JSON `Value`s.
pub type ErasedStreamingCallback = Arc<dyn Fn(Result<Value>) + Send + Sync>;

task_local! {
    /// The task-local variable that holds the current streaming callback.
    static STREAMING_CALLBACK: Option<ErasedStreamingCallback>;
}

/// Retrieves the type-erased streaming callback from the current task-local context.
pub fn get_streaming_callback() -> Option<ErasedStreamingCallback> {
    STREAMING_CALLBACK.try_with(|cb| cb.clone()).unwrap_or(None)
}

/// Retrieves the type-erased streaming callback from the current task-local context.
pub async fn run_with_streaming_callback<S, F, Fut>(
    callback: Option<StreamingCallback<S>>,
    f: F,
) -> Fut::Output
where
    S: DeserializeOwned + Send + 'static,
    F: FnOnce() -> Fut,
    Fut: Future,
{
    let erased_callback: Option<ErasedStreamingCallback> = callback.map(|cb| {
        // By giving `erased` the explicit type, we tell Rust to create the
        // `dyn Fn` trait object that the task-local variable expects.
        let erased: ErasedStreamingCallback = Arc::new(move |value_result: Result<Value>| {
            let final_result = value_result.and_then(|value| {
                serde_json::from_value(value).map_err(|e| Error::new_internal(e.to_string()))
            });
            cb(final_result);
        });
        erased
    });

    // Use `.scope()` to run the provided future with the task-local set.
    STREAMING_CALLBACK.scope(erased_callback, f()).await
}
