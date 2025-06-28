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
use crate::formats::FormatterConfig;
use crate::generate::{OutputOptions, ToolChoice};
use crate::message::{Message, MessageData, Role};
use crate::model::{self, GenerateRequest, GenerateResponseChunkData, GenerateResponseData};
use crate::{formats, Model};
use crate::{tool, Document};
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
use genkit_core::tracing;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The action type for a model.
pub type ModelAction = Action<GenerateRequest, GenerateResponseData, GenerateResponseChunkData>;

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
pub struct GenerateHelperOptions {
    pub raw_request: GenerateOptions,
    // TODO: Middleware support
    // pub middleware: Option<Vec<ModelMiddleware>>,
    pub current_turn: u32,
    pub message_index: u32,
    // TODO: AbortSignal support
    // pub abort_signal: Option<AbortSignal>,
}

/// Defines (registers) a utility `generate` action.
pub fn define_generate_action(registry: &mut Registry) -> Arc<GenerateAction> {
    let registry_clone = Arc::new(registry.clone());
    let generate_action = ActionBuilder::new(
        ActionType::Util,
        "generate".to_string(),
        move |req: GenerateOptions, _ctx| {
            let registry_clone = registry_clone.clone();
            async move {
                let options = GenerateHelperOptions {
                    raw_request: req,
                    current_turn: 0,
                    message_index: 0,
                };
                generate_helper::<Value>(registry_clone, options).await
            }
        },
    )
    .build(registry);
    Arc::new(generate_action)
}

/// Encapsulates all generate logic, similar to `generateAction` but callable internally.
pub async fn generate_helper<O>(
    registry: Arc<Registry>,
    options: GenerateHelperOptions,
) -> Result<GenerateResponse<O>>
where
    O: for<'de> DeserializeOwned + Send + Sync + 'static,
{
    let mut attrs = HashMap::new();
    attrs.insert(
        "genkit:spanType".to_string(),
        Value::String("util".to_string()),
    );
    let raw_request_for_response = options.raw_request.clone();
    let registry_for_response = registry.clone();
    let options_for_internal = options.clone();
    let (response_data, _telemetry) = tracing::in_new_span(
        "generate".to_string(),
        Some(attrs),
        move |_trace_context| async move {
            // TODO: Add input/output to trace metadata
            generate_internal(registry, options_for_internal).await
        },
    )
    .await?;

    let request = to_generate_request(&registry_for_response, &raw_request_for_response).await?;
    let mut response = GenerateResponse::new(&response_data, Some(request.clone()));

    // Resolve formatter and attach parser to the response.
    if let Some(output_options) = options.raw_request.output.as_ref() {
        if let Some(format) =
            formats::resolve_format(&registry_for_response, Some(output_options)).await
        {
            let schema_val = options
                .raw_request
                .output
                .as_ref()
                .and_then(|out| out.schema.as_ref());
            let schema: Option<schemars::Schema> =
                schema_val.and_then(|v| serde_json::from_value(v.clone()).ok());
            let handler = (format.handler)(schema.as_ref());
            let parser = move |msg: &crate::message::Message<O>| {
                let value_msg = Message::new(msg.to_json(), None);
                match serde_json::from_value(handler.parse_message(&value_msg)) {
                    Ok(v) => Ok(v),
                    Err(e) => Err(genkit_core::error::Error::new_internal(format!(
                        "Failed to deserialize output: {}",
                        e
                    ))),
                }
            };
            response.parser = Some(Arc::new(parser));
        }
    }

    Ok(response)
}

/// The core, private implementation of the generation logic.
async fn generate_internal(
    registry: Arc<Registry>,
    options: GenerateHelperOptions,
) -> Result<GenerateResponseData> {
    let mut request = options.raw_request.clone();

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
    let model_action = any_downcast::downcast_arc::<ModelAction>(erased_action)
        .map_err(|_| Error::new_internal(format!("'{}' is not a valid ModelAction", model_name)))?;

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

    // 3. Convert to GenerateRequest (the type the model action expects).
    let generate_request = to_generate_request(&registry, &request).await?;

    // 4. Handle the tool loop.
    let max_turns = request.max_turns.unwrap_or(5);
    if options.current_turn >= max_turns {
        return Err(Error::new_internal(format!(
            "Exceeded maximum tool call iterations ({})",
            max_turns
        )));
    }

    // 5. Call the model action.
    // This is a simplification. The TS code has a middleware dispatch chain.
    let request_value = serde_json::to_value(&generate_request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    let response = model_action.run_http(request_value, None).await?;
    let response_data = response.result;

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

    if tool_requests.is_empty() {
        return Ok(response_data);
    }

    // Only continue the loop if we don't need to return the tool requests.
    if request.return_tool_requests != Some(true) {
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

        let mut next_request = request;
        let mut messages = next_request.messages.take().unwrap_or_default();
        messages.push(generated_message);
        if let Some(tool_message) = tool_results.tool_message {
            messages.push(tool_message);
        }
        next_request.messages = Some(messages);

        // Recursive call for the next turn in the tool loop.
        return Box::pin(generate_internal(
            registry,
            GenerateHelperOptions {
                raw_request: next_request,
                current_turn: options.current_turn + 1,
                message_index: options.message_index + 1, // This may need adjustment
            },
        ))
        .await;
    }

    Ok(response_data)
}

/// Converts high-level `GenerateOptions` to the `GenerateRequest` for a model.
async fn to_generate_request(
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

mod any_downcast {
    use genkit_core::registry::ErasedAction;
    use std::sync::Arc;

    pub fn downcast_arc<T: 'static>(
        arc: Arc<dyn ErasedAction>,
    ) -> std::result::Result<Arc<T>, Arc<dyn ErasedAction>> {
        if arc.as_any().is::<T>() {
            unsafe {
                let raw = Arc::into_raw(arc);
                let ptr = raw as *const T;
                Ok(Arc::from_raw(ptr))
            }
        } else {
            Err(arc)
        }
    }
}
