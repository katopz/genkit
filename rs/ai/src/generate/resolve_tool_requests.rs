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

//! # Tool Request Resolution
//!
//! This module provides functions for processing tool requests from a model,
//! executing the corresponding tools, and handling interrupts. It is the Rust
// equivalent of `generate/resolve-tool-requests.ts`.

use crate::document::{Part, ToolRequest, ToolRequestPart, ToolResponse};
use crate::generate::{GenerateOptions, ResumeOptions};
use crate::message::{MessageData, Role};
use crate::model::GenerateResponseData;
use crate::tool::{self, is_tool_request};
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ErasedAction, Registry};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The result of resolving a single tool request.
pub struct ResolvedToolRequest<O: 'static> {
    pub response: Option<Part>,
    pub interrupt: Option<Part>,
    pub preamble: Option<GenerateOptions<O>>,
}

impl<O: 'static> Default for ResolvedToolRequest<O> {
    fn default() -> Self {
        Self {
            response: None,
            interrupt: None,
            preamble: None,
        }
    }
}

/// The result of resolving all tool requests in a message.
pub struct ResolvedToolRequests<O: 'static> {
    pub revised_model_message: Option<MessageData>,
    pub tool_message: Option<MessageData>,
    pub transfer_preamble: Option<GenerateOptions<O>>,
}

/// The result of applying `ResumeOptions` to a generation request.
#[derive(Default)]
pub struct ResolveResumeOptionResult<O: 'static> {
    pub revised_request: Option<GenerateOptions<O>>,
    pub interrupted_response: Option<GenerateResponseData>,
    pub tool_message: Option<MessageData>,
}

/// Converts a slice of `ErasedAction` trait objects into a map of short names to actions.
/// This is the primary change: we now work with the generic trait object, not the concrete type.
pub fn to_tool_map(
    tools: &[Arc<dyn ErasedAction>],
) -> Result<HashMap<String, Arc<dyn ErasedAction>>> {
    assert_valid_tool_names(tools)?;
    let mut map = HashMap::new();
    for tool in tools {
        let name = tool.name();
        let short_name = name.split('/').next_back().unwrap_or(name).to_string();
        map.insert(short_name, tool.clone());
    }
    Ok(map)
}

/// Ensures that no two tools in a given slice have the same short name.
/// This now accepts a slice of trait objects.
pub fn assert_valid_tool_names(tools: &[Arc<dyn ErasedAction>]) -> Result<()> {
    let mut names = HashMap::new();
    for tool in tools {
        let name = tool.name();
        let short_name = name.split('/').next_back().unwrap_or(name);
        if let Some(existing_name) = names.insert(short_name.to_string(), name.to_string()) {
            return Err(Error::new_internal(format!(
                "Cannot provide two tools with the same short name ('{}'): '{}' and '{}'",
                short_name, name, existing_name
            )));
        }
    }
    Ok(())
}

/// Creates a `ToolRequestPart` that has its output pending in the metadata.
pub fn to_pending_output(part: &Part, response: &Part) -> Part {
    let mut new_part = part.clone();
    if let Some(tool_response) = &response.tool_response {
        let metadata = new_part.metadata.get_or_insert_with(HashMap::new);
        metadata.insert(
            "pendingOutput".to_string(),
            tool_response.output.clone().unwrap_or(Value::Null),
        );
    }
    new_part
}

/// Resolves a single tool request by executing it.
/// This now accepts a map of trait objects.
pub async fn resolve_tool_request<O: 'static>(
    raw_request: &GenerateOptions<O>,
    part: &Part,
    tool_map: &HashMap<String, Arc<dyn ErasedAction>>,
) -> Result<ResolvedToolRequest<O>> {
    let tool_request = part
        .tool_request
        .as_ref()
        .ok_or_else(|| Error::new_internal("resolve_tool_request called on a non-tool part"))?;

    let tool = tool_map
        .get(&tool_request.name)
        .ok_or_else(|| Error::new_internal(format!("Tool '{}' not found", &tool_request.name)))?;

    let request_value = serde_json::to_value(tool_request.input.clone().unwrap_or(Value::Null))
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;

    let context_override = part
        .metadata
        .as_ref()
        .and_then(|m| m.get("restartContext"))
        .and_then(|v| {
            if let Value::Object(map) = v {
                Some(ActionContext {
                    additional_context: map.clone().into_iter().collect(),
                    ..Default::default()
                })
            } else {
                None
            }
        });
    let response_result = tool
        .run_http_json(
            request_value,
            context_override.or(raw_request.context.clone()),
        )
        .await;

    match response_result {
        Ok(response) => {
            let output = response;

            let response_part = Part {
                tool_response: Some(ToolResponse {
                    name: tool_request.name.clone(),
                    ref_id: tool_request.ref_id.clone(),
                    output: Some(output),
                }),
                ..Default::default()
            };
            Ok(ResolvedToolRequest {
                response: Some(response_part),
                ..Default::default()
            })
        }
        Err(e) => {
            if let Error::Internal { message, .. } = &e {
                if message.starts_with("INTERRUPT::") {
                    let mut interrupted_part = part.clone();
                    let metadata = interrupted_part
                        .metadata
                        .get_or_insert_with(Default::default);
                    metadata.insert("interrupt".to_string(), serde_json::Value::Bool(true));

                    return Ok(ResolvedToolRequest {
                        interrupt: Some(interrupted_part),
                        ..Default::default()
                    });
                }
            }
            Err(e)
        }
    }
}

/// Resolves all tool requests within a model-generated message.
pub async fn resolve_tool_requests<O: 'static>(
    registry: &Registry,
    raw_request: &GenerateOptions<O>,
    generated_message: &MessageData,
) -> Result<ResolvedToolRequests<O>> {
    let resolved_tools = tool::resolve_tools(registry, raw_request.tools.as_deref()).await?;

    // The problematic downcast is now removed. We pass the trait objects directly.
    let tool_map = to_tool_map(&resolved_tools)?;

    let mut response_parts: Vec<Part> = Vec::new();
    let mut has_interrupts = false;
    let mut transfer_preamble: Option<GenerateOptions<O>> = None;

    let mut revised_model_message = generated_message.clone();

    let futures = revised_model_message
        .content
        .iter()
        .enumerate()
        .filter(|(_, part)| part.tool_request.is_some())
        .map(|(i, part)| {
            let tool_map = tool_map.clone();
            let part = part.clone();
            async move {
                let result = resolve_tool_request(raw_request, &part, &tool_map).await;
                (i, result)
            }
        });

    let results = futures_util::future::join_all(futures).await;

    for (index, result) in results {
        match result {
            Ok(resolved) => {
                if let Some(preamble) = resolved.preamble {
                    if transfer_preamble.is_some() {
                        return Err(Error::new_internal(
                            "Model attempted to transfer to multiple prompt tools.".to_string(),
                        ));
                    }
                    transfer_preamble = Some(preamble);
                }
                if let Some(response) = resolved.response {
                    let original_part = &revised_model_message.content[index].clone();
                    revised_model_message.content[index] =
                        to_pending_output(original_part, &response);
                    response_parts.push(response);
                }
                if let Some(interrupt) = resolved.interrupt {
                    revised_model_message.content[index] = interrupt;
                    has_interrupts = true;
                }
            }
            Err(e) => return Err(e),
        }
    }

    if has_interrupts {
        Ok(ResolvedToolRequests {
            revised_model_message: Some(revised_model_message),
            tool_message: None,
            transfer_preamble,
        })
    } else {
        let tool_message = if !response_parts.is_empty() {
            Some(MessageData {
                role: Role::Tool,
                content: response_parts,
                metadata: None,
            })
        } else {
            None
        };
        Ok(ResolvedToolRequests {
            revised_model_message: Some(revised_model_message),
            tool_message,
            transfer_preamble,
        })
    }
}

// Helper to find a matching ToolRequestPart in a slice of parts.
fn find_corresponding_tool_request<'a>(
    parts: &'a [ToolRequestPart],
    part: &ToolRequest,
) -> Option<&'a ToolRequestPart> {
    parts.iter().find(|p| {
        p.tool_request
            .as_ref()
            .is_some_and(|tr| tr.name == part.name && tr.ref_id == part.ref_id)
    })
}

// Helper to find a matching ToolResponsePart in a slice of parts.
fn find_corresponding_tool_response<'a>(parts: &'a [Part], part: &ToolRequest) -> Option<&'a Part> {
    parts.iter().find(|p| {
        p.tool_response
            .as_ref()
            .is_some_and(|tr| tr.name == part.name && tr.ref_id == part.ref_id)
    })
}

struct ResumedToolRequestResult {
    tool_request: ToolRequestPart,
    tool_response: Part,
}

// Helper to resolve a single tool request that was interrupted.
async fn resolve_resumed_tool_request<O: 'static>(
    raw_request: &GenerateOptions<O>,
    resume_opts: &ResumeOptions,
    part: ToolRequestPart,
    tool_map: &HashMap<String, Arc<dyn ErasedAction>>,
) -> Result<ResumedToolRequestResult> {
    let tool_request = part.tool_request.as_ref().unwrap(); // Safe due to checks before calling

    // Handle provided responses.
    if let Some(provided_responses) = &resume_opts.respond {
        if let Some(provided_response) =
            find_corresponding_tool_response(provided_responses, tool_request)
        {
            let mut metadata = part.metadata.clone().unwrap_or_default();
            if let Some(interrupt_meta) = metadata.remove("interrupt") {
                metadata.insert("resolvedInterrupt".to_string(), interrupt_meta);
            }
            return Ok(ResumedToolRequestResult {
                tool_request: Part {
                    metadata: Some(metadata),
                    ..part
                },
                tool_response: provided_response.clone(),
            });
        }
    }

    // Handle restarts.
    if let Some(restart_requests) = &resume_opts.restart {
        if let Some(restart_request) =
            find_corresponding_tool_request(restart_requests, tool_request)
        {
            let resolved = resolve_tool_request(raw_request, restart_request, tool_map).await?;
            if resolved.interrupt.is_some() {
                // A restarted tool cannot immediately interrupt again.
                return Err(Error::new_internal(
                    "A restarted tool triggered an interrupt, which is not allowed.",
                ));
            }
            if let Some(response) = resolved.response {
                let mut metadata = part.metadata.clone().unwrap_or_default();
                if let Some(interrupt_meta) = metadata.remove("interrupt") {
                    metadata.insert("resolvedInterrupt".to_string(), interrupt_meta);
                }
                return Ok(ResumedToolRequestResult {
                    tool_request: Part {
                        metadata: Some(metadata),
                        ..part
                    },
                    tool_response: response,
                });
            }
        }
    }

    Err(Error::new_internal(format!(
        "Unresolved tool request '{}' was not handled by the 'resume' argument.",
        tool_request.name
    )))
}

/// Amends message history to handle `resume` arguments.
pub async fn resolve_resume_option<O: Default + Send + Sync + 'static>(
    registry: &Registry,
    mut raw_request: GenerateOptions<O>,
) -> Result<ResolveResumeOptionResult<O>> {
    let resume_opts = match raw_request.resume.take() {
        Some(opts) => opts,
        None => {
            return Ok(ResolveResumeOptionResult {
                revised_request: Some(raw_request),
                ..Default::default()
            })
        }
    };

    let resolved_tools = tool::resolve_tools(registry, raw_request.tools.as_deref()).await?;

    // Again, we remove the downcast and work with the trait objects.
    let tool_map = to_tool_map(&resolved_tools)?;

    let mut messages = raw_request.messages.clone().unwrap_or_default();
    let last_message = match messages.last_mut() {
        Some(msg)
            if msg.role == Role::Model && msg.content.iter().any(|p| p.tool_request.is_some()) =>
        {
            msg
        }
        _ => {
            return Err(Error::new_internal(
                "Cannot 'resume' generation unless the last message is a model message with tool requests.",
            ))
        }
    };

    let mut tool_responses: Vec<Part> = Vec::new();
    let mut new_content: Vec<Part> = Vec::new();
    let mut interrupted = false;

    for part in std::mem::take(&mut last_message.content) {
        if is_tool_request(&part) {
            match resolve_resumed_tool_request(&raw_request, &resume_opts, part.clone(), &tool_map)
                .await
            {
                Ok(resolved) => {
                    new_content.push(resolved.tool_request);
                    tool_responses.push(resolved.tool_response);
                }
                Err(_) => {
                    // Assuming any error here means an interrupt.
                    new_content.push(part);
                    interrupted = true;
                }
            }
        } else {
            new_content.push(part);
        }
    }

    last_message.content = new_content;

    if interrupted {
        // Create a response indicating that an interrupt occurred during resumption.
        let interrupted_response = GenerateResponseData {
            candidates: vec![crate::model::CandidateData {
                index: 0,
                message: last_message.clone(),
                finish_reason: Some(crate::model::FinishReason::Interrupted),
                finish_message: Some("One or more tools triggered interrupts while resuming generation. The model was not called.".to_string()),
            }],
            ..Default::default()
        };
        return Ok(ResolveResumeOptionResult {
            interrupted_response: Some(interrupted_response),
            ..Default::default()
        });
    }

    let mut tool_message_metadata = HashMap::new();
    if let Some(meta) = resume_opts.metadata {
        tool_message_metadata.insert("resumed".to_string(), meta);
    }

    let tool_message = MessageData {
        role: Role::Tool,
        content: tool_responses,
        metadata: if tool_message_metadata.is_empty() {
            None
        } else {
            Some(tool_message_metadata)
        },
    };

    messages.push(tool_message.clone());
    raw_request.messages = Some(messages);

    Ok(ResolveResumeOptionResult {
        revised_request: Some(raw_request),
        tool_message: Some(tool_message),
        ..Default::default()
    })
}

// The `any_downcast` helper module is no longer needed and has been removed.
