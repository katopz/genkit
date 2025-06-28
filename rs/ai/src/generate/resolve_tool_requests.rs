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

use crate::document::{Part, ToolResponse};
use crate::generate::GenerateOptions;
use crate::message::{MessageData, Role};
use crate::tool::{self, ToolAction};
use genkit_core::error::{Error, Result};
use genkit_core::registry::Registry;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// The result of resolving a single tool request.
pub struct ResolvedToolRequest {
    pub response: Option<Part>,
    pub interrupt: Option<Part>,
    pub preamble: Option<GenerateOptions>,
}

/// Converts a slice of `ToolAction`s into a map of short names to actions.
pub fn to_tool_map(tools: &[Arc<ToolAction>]) -> Result<HashMap<String, Arc<ToolAction>>> {
    assert_valid_tool_names(tools)?;
    let mut map = HashMap::new();
    for tool in tools {
        let name = &tool.meta.name;
        let short_name = name.split('/').last().unwrap_or(name).to_string();
        map.insert(short_name, tool.clone());
    }
    Ok(map)
}

/// Ensures that no two tools in a given slice have the same short name.
pub fn assert_valid_tool_names(tools: &[Arc<ToolAction>]) -> Result<()> {
    let mut names = HashMap::new();
    for tool in tools {
        let name = &tool.meta.name;
        let short_name = name.split('/').last().unwrap_or(name);
        if let Some(existing_name) = names.insert(short_name.to_string(), name.clone()) {
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

/// The result of resolving all tool requests in a message.
pub struct ResolvedToolRequests {
    pub revised_model_message: Option<MessageData>,
    pub tool_message: Option<MessageData>,
    pub transfer_preamble: Option<GenerateOptions>,
}

/// Resolves a single tool request.
pub async fn resolve_tool_request(
    _raw_request: &GenerateOptions,
    part: &Part,
    tool_map: &HashMap<String, Arc<ToolAction>>,
) -> Result<ResolvedToolRequest> {
    let tool_request = part
        .tool_request
        .as_ref()
        .ok_or_else(|| Error::new_internal("resolve_tool_request called on a non-tool part"))?;

    let tool = tool_map
        .get(&tool_request.name)
        .ok_or_else(|| Error::new_internal(format!("Tool '{}' not found", &tool_request.name)))?;

    // TODO: Implement is_prompt_action and preamble logic.

    // Execute the tool and handle interrupts.
    // TODO: A proper implementation would catch a specific ToolInterruptError.
    let request_value = serde_json::to_value(tool_request.input.clone().unwrap_or(Value::Null))
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    let response = tool.run_http(request_value, None).await?;
    let output = response.result;

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
        interrupt: None,
        preamble: None,
    })
}

/// Resolves all tool requests within a model-generated message.
///
/// This function iterates through `toolRequest` parts, executes the corresponding
/// tools, and aggregates the responses. It also handles tool interrupts.
pub async fn resolve_tool_requests(
    registry: &Registry,
    raw_request: &GenerateOptions,
    generated_message: &MessageData,
) -> Result<ResolvedToolRequests> {
    let resolved_tools = tool::resolve_tools(registry, raw_request.tools.as_deref()).await?;
    let tool_map = to_tool_map(&resolved_tools)?;

    let mut response_parts: Vec<Part> = Vec::new();
    let mut has_interrupts = false;
    let mut transfer_preamble: Option<GenerateOptions> = None;

    let mut revised_model_message = generated_message.clone();

    // The TS version uses `Promise.all`, so we can do something similar here.
    let futures = revised_model_message
        .content
        .iter()
        .enumerate()
        .filter(|(_, part)| part.tool_request.is_some())
        .map(|(i, part)| {
            let tool_map = tool_map.clone();
            let raw_request = raw_request.clone();
            let part = part.clone();
            async move {
                let result = resolve_tool_request(&raw_request, &part, &tool_map).await;
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
            Err(e) => return Err(e), // Propagate errors
        }
    }

    if has_interrupts {
        Ok(ResolvedToolRequests {
            revised_model_message: Some(revised_model_message),
            tool_message: None,
            transfer_preamble: None,
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
            revised_model_message: None,
            tool_message,
            transfer_preamble,
        })
    }
}

// Note: `resolve_tool_request`, `resolve_resume_option`, and related helpers
// are complex and would be added here in a more complete port.
