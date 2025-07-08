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
//! for content generation, including tool handling and streaming. It is the Rust
// equivalent of `generate/action.ts`.

use super::{response::GenerateResponse, GenerateOptions};
use crate::document::Part;
use crate::generate::{OutputOptions, ToolChoice};
use crate::message::{Message, MessageData};
use crate::model::{
    self,
    middleware::{ModelMiddleware, ModelMiddlewareNext},
    GenerateRequest, GenerateResponseChunkData, GenerateResponseData,
};
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
    dyn Fn(GenerateRequest) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
        + Send,
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

/// Reconstructs the `GenerateRequest` to reflect user-facing options.
async fn reconstruct_user_facing_request<O: Serialize>(
    model_request: &GenerateRequest,
    raw_request: &GenerateOptions<O>,
    registry: &Registry,
) -> Result<GenerateRequest> {
    let mut final_request = model_request.clone();

    // If the original request didn't have tools, don't show the injected empty array.
    if raw_request.tools.is_none() {
        final_request.tools = None;
    }

    if let Some(user_output_opts) = raw_request.output.as_ref() {
        let mut final_output_opts = user_output_opts.clone();
        if let Some(format) = formats::resolve_format(registry, Some(user_output_opts)).await {
            // Always set format name from resolved format
            final_output_opts.format = Some(format.name.clone());
            // Only set content_type if user didn't specify one
            if final_output_opts.content_type.is_none() {
                final_output_opts.content_type = format.config.content_type.clone();
            }
            // Only set constrained if user didn't specify one
            if final_output_opts.constrained.is_none() {
                final_output_opts.constrained = format.config.constrained;
            }
        }
        final_request.output = Some(serde_json::to_value(final_output_opts).map_err(|e| {
            Error::new_internal(format!("Failed to serialize final output options: {}", e))
        })?);
    }
    Ok(final_request)
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

                let result = tracing::in_new_span(
                    "generate".to_string(),
                    Some(attrs),
                    move |_trace_context| async move {
                        let helper_options = GenerateHelperOptions {
                            raw_request: req,
                            middleware: Vec::new(), // Middleware not supported in this path yet.
                            current_turn: 0,
                            message_index: 0,
                        };
                        generate_internal(registry, helper_options).await
                    },
                )
                .await;

                let ((response_data, request), _telemetry) = result?;

                let final_request = reconstruct_user_facing_request(
                    &request,
                    &raw_request_for_response,
                    &registry_for_response,
                )
                .await?;
                let mut response = GenerateResponse::new(&response_data, Some(final_request));

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
                        let parser = Arc::new(parser);
                        if let Some(message) = response.message.as_mut() {
                            message.set_parser(parser.clone());
                        }
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
    let result = tracing::in_new_span(
        "generate".to_string(),
        Some(attrs),
        move |_trace_context| async move { generate_internal(registry, options).await },
    )
    .await;

    let ((response_data, request), _telemetry) = result?;

    let final_request = reconstruct_user_facing_request(
        &request,
        &raw_request_for_response,
        &registry_for_response,
    )
    .await?;
    let mut response = GenerateResponse::new(&response_data, Some(final_request));

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
            let parser = Arc::new(parser);
            if let Some(message) = response.message.as_mut() {
                message.set_parser(parser.clone());
            }
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
    let mut chain: ModelMiddlewareNext<'static> = Box::new(move |req: GenerateRequest| {
        let model_action = model_action.clone();
        Box::pin(async move {
            let req_value = serde_json::to_value(req)
                .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
            let mut value = if let Some(chunk_callback) = get_streaming_callback() {
                let mut streaming_response = model_action.stream_http_json(req_value, None)?;
                tokio::spawn(async move {
                    while let Some(chunk_result) = streaming_response.stream.next().await {
                        chunk_callback(chunk_result);
                    }
                });
                streaming_response.output.await?
            } else {
                model_action.run_http_json(req_value, None).await?
            };
            if let Some(result) = value.get_mut("result") {
                value = result.take();
            }
            serde_json::from_value(value)
                .map_err(|e| Error::new_internal(format!("Failed to deserialize: {}", e)))
        })
    });

    // Wrap the chain with each middleware, in reverse order.
    for mw in middleware.into_iter().rev() {
        let prev_chain = chain;
        let state = std::sync::Mutex::new(Some(prev_chain));
        chain = Box::new(move |req| {
            let f = state
                .lock()
                .unwrap() // Panics if the mutex is poisoned.
                .take()
                .expect("Middleware chain can only be called once.");
            mw(req, f)
        });
    }

    // Execute the full chain.
    chain(request).await
}

/// The core, private implementation of the generation logic.
#[allow(clippy::type_complexity)]
fn generate_internal<O>(
    registry: Arc<Registry>,
    options: GenerateHelperOptions<O>,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<(GenerateResponseData, GenerateRequest)>> + Send>,
>
where
    O: Default
        + Clone
        + for<'de> DeserializeOwned
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + 'static,
{
    Box::pin(async move {
        let GenerateHelperOptions {
            raw_request,
            middleware,
            current_turn,
            message_index,
        } = options;
        let mut request = raw_request.clone();

        // 1. Pre-process messages (prompt, system, etc.)
        if let Some(prompt_parts) = request.prompt.take() {
            let messages = request.messages.get_or_insert_with(Vec::new);
            messages.push(crate::message::MessageData {
                role: crate::message::Role::User,
                content: prompt_parts,
                ..Default::default()
            });
        }

        // 2. Resolve model.
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

        // 3. Resolve and apply format.
        if let Some(format) = formats::resolve_format(&registry, request.output.as_ref()).await {
            // Get model info from the action's metadata
            let model_info: Option<crate::model::ModelInfo> = model_action
                .metadata()
                .metadata
                .get("metadata")
                .and_then(|v| serde_json::from_value(v.clone()).ok());

            // Check if model supports the format natively
            let model_supports_format = model_info
                .as_ref()
                .and_then(|info| info.supports.as_ref())
                .and_then(|supports| supports.output.as_ref())
                .is_some_and(|supported_formats| supported_formats.contains(&format.name));

            // Only inject instructions if the model does NOT support the format.
            if !model_supports_format {
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
                    println!(
                        "[generate_internal] 🔥 Messages before injection: {:?}",
                        messages
                    );
                    let updated_messages = formats::inject_instructions(messages, instructions);
                    request.messages = Some(updated_messages);
                    println!(
                        "[generate_internal] ✨ Messages after injection: {:?}",
                        request.messages
                    );
                }

                // When simulating, modify the output options for the model request.
                if let Some(output_opts) = request.output.as_mut() {
                    output_opts.constrained = Some(false);
                    if !matches!(
                        output_opts.instructions.as_ref().and_then(|i| i.as_bool()),
                        Some(true)
                    ) {
                        output_opts.format = None;
                        output_opts.schema = None;
                    }
                    output_opts.content_type = None;
                }
            } else {
                // Model supports format, so just pass through the options.
                if let Some(output_opts) = request.output.as_mut() {
                    output_opts.format = Some(format.name.clone());
                    output_opts.content_type = format.config.content_type.clone();
                    output_opts.constrained = format.config.constrained;
                }
            }
        }

        // 4. Handle tool loop recursion check.
        let max_turns = request.max_turns.unwrap_or(5);
        if current_turn >= max_turns {
            return Err(Error::new_internal(format!(
                "Exceeded maximum tool call iterations ({})",
                max_turns
            )));
        }

        // 5. Apply resume logic BEFORE the main model call.
        let resume_result =
            super::resolve_tool_requests::resolve_resume_option(registry.as_ref(), request).await?;
        if let Some(interrupted_response) = resume_result.interrupted_response {
            let final_req = to_generate_request(&registry, &raw_request).await?;
            return Ok((interrupted_response, final_req));
        }
        let mut request = resume_result.revised_request.unwrap();

        if let Some(tool_msg) = resume_result.tool_message {
            if let Some(cb) = get_streaming_callback() {
                let chunk_data = GenerateResponseChunkData {
                    index: message_index,
                    content: tool_msg.content,
                    role: Some(tool_msg.role.clone()),
                    ..Default::default()
                };
                let chunk_value = serde_json::to_value(chunk_data)
                    .map_err(|e| Error::new_internal(e.to_string()))?;
                cb(Ok(chunk_value));
            }
        }

        // 6. Convert to GenerateRequest (the type the model action expects).
        let generate_request = to_generate_request(&registry, &request).await?;

        // 7. Call the model action.
        let response_data =
            run_model_via_middleware(model_action, generate_request.clone(), middleware.clone())
                .await?;

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
            return Ok((response_data, generate_request.clone()));
        }

        // 8. Handle tool requests returned from model
        let tool_results = super::resolve_tool_requests::resolve_tool_requests(
            &registry,
            &request,
            &generated_message,
        )
        .await?;

        // 9. Check for interrupts and return early if found.
        if let Some(revised_with_interrupt) =
            tool_results.revised_model_message.as_ref().filter(|m| {
                m.content.iter().any(|p| {
                    p.metadata
                        .as_ref()
                        .is_some_and(|meta| meta.contains_key("interrupt"))
                })
            })
        {
            let mut final_response = response_data;
            if let Some(candidate) = final_response.candidates.get_mut(0) {
                candidate.message = revised_with_interrupt.clone();
                candidate.finish_reason = Some(model::FinishReason::Interrupted);
                candidate.finish_message =
                    Some("One or more tool calls resulted in interrupts.".to_string());
            }
            return Ok((final_response, generate_request));
        }

        // 10. If no interrupts, update the message history for the next turn.
        let mut messages = request.messages.take().unwrap_or_default();

        if let Some(revised_msg) = tool_results.revised_model_message {
            messages.push(revised_msg);
        } else {
            messages.push(generated_message);
        }

        if let Some(tool_msg) = tool_results.tool_message {
            messages.push(tool_msg);
        }
        request.messages = Some(messages);

        // 11. Recursive call for the next turn.
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

/// Executes a future with a given session set as the current task-local session.
pub async fn run_with_streaming_callback<S, F, Fut>(
    callback: Option<StreamingCallback<S>>,
    f: F,
) -> Fut::Output
where
    S: DeserializeOwned + Send + 'static,
    F: FnOnce() -> Fut,
    Fut: Future,
{
    if let Some(cb) = callback {
        let previous_chunks_state = Arc::new(tokio::sync::Mutex::new(Vec::<
            crate::model::GenerateResponseChunkData,
        >::new()));

        let erased: ErasedStreamingCallback = Arc::new(move |value_result: Result<Value>| {
            let cb = cb.clone();
            let previous_chunks_state = previous_chunks_state.clone();
            tokio::spawn(async move {
                let final_result = async {
                    let value = value_result?;

                    let mut chunk_data_obj = match value {
                        Value::Object(map) => map,
                        _ => {
                            return Err(Error::new_internal("Expected JSON object for chunk data"))
                        }
                    };

                    let mut previous_chunks_guard = previous_chunks_state.lock().await;

                    let current_chunk_data: GenerateResponseChunkData =
                        serde_json::from_value(Value::Object(chunk_data_obj.clone())).map_err(
                            |e| Error::new_internal(format!("Failed to parse chunk data: {}", e)),
                        )?;

                    chunk_data_obj.insert(
                        "previousChunks".to_string(),
                        serde_json::to_value(previous_chunks_guard.clone()).unwrap(),
                    );

                    previous_chunks_guard.push(current_chunk_data);

                    let value_with_history = Value::Object(chunk_data_obj);

                    serde_json::from_value(value_with_history)
                        .map_err(|e| Error::new_internal(e.to_string()))
                }
                .await;
                cb(final_result);
            });
        });
        STREAMING_CALLBACK.scope(Some(erased), f()).await
    } else {
        f().await
    }
}
