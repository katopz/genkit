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

//! # Vertex AI Gemini Models
//!
//! This module provides the implementation for the Gemini family of models
//! on Vertex AI.

use crate::{Error, Result};

use genkit::{
    message::Role,
    model::{FinishReason, GenerateRequest},
    CandidateData, GenerateResponseData, GenerationUsage, ToolRequest,
};
use serde_json::Value;

// Configuration structs for the Gemini model, aligned with the API.
use super::types::*;

pub fn to_vertex_part(part: &genkit::document::Part) -> Result<VertexPart> {
    if let Some(media) = &part.media {
        let (mime_type, data) = media.url.split_once(";base64,").ok_or_else(|| {
            Error::VertexAI("Media URL is not a valid base64 data URI.".to_string())
        })?;
        Ok(VertexPart {
            inline_data: Some(VertexMedia {
                mime_type: mime_type.replace("data:", ""),
                data: data.to_string(),
            }),
            ..Default::default()
        })
    } else if let Some(tool_req) = &part.tool_request {
        Ok(VertexPart {
            function_call: Some(VertexFunctionCall {
                name: tool_req.name.clone(),
                args: tool_req.input.clone().unwrap_or_default(),
            }),
            ..Default::default()
        })
    } else if let Some(tool_resp) = &part.tool_response {
        Ok(VertexPart {
            function_response: Some(VertexFunctionResponse {
                name: tool_resp.name.clone(),
                response: serde_json::json!({
                    "name": tool_resp.name,
                    "content": tool_resp.output
                }),
            }),
            ..Default::default()
        })
    } else if part.reasoning.is_some() {
        let thought_signature = part
            .metadata
            .as_ref()
            .and_then(|m| m.get("thoughtSignature"))
            .and_then(|v| v.as_str())
            .map(String::from);
        Ok(VertexPart {
            thought: Some(true),
            thought_signature,
            ..Default::default()
        })
    } else {
        Ok(VertexPart {
            text: part.text.clone(),
            ..Default::default()
        })
    }
}

/// Converts a Genkit `GenerateRequest` into a `VertexGeminiRequest`.
pub fn to_vertex_request(req: &GenerateRequest) -> Result<VertexGeminiRequest> {
    let mut messages = req.messages.clone();

    // Handle system instructions separately
    let system_instruction = if let Some(pos) = messages.iter().position(|m| m.role == Role::System)
    {
        let system_message = messages.remove(pos);
        let parts = system_message
            .content
            .iter()
            .map(to_vertex_part)
            .collect::<Result<Vec<VertexPart>>>()?;
        Some(VertexContent {
            // System role is not supported directly, use 'user' as per some Gemini patterns
            // or rely on the dedicated `system_instruction` field. Here we populate the dedicated field.
            role: "user".to_string(),
            parts,
        })
    } else {
        None
    };

    let contents = messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                Role::Model => "model".to_string(),
                Role::Tool => "function".to_string(),
                _ => "user".to_string(), // System (if not handled above) and User
            };

            let mut content_parts = msg.content.clone();
            if msg.role == Role::Tool {
                content_parts.sort_by(|a, b| {
                    let a_ref = a.tool_response.as_ref().and_then(|tr| tr.r#ref.as_deref());
                    let b_ref = b.tool_response.as_ref().and_then(|tr| tr.r#ref.as_deref());
                    a_ref.cmp(&b_ref)
                });
            }

            let parts = content_parts
                .iter()
                .map(to_vertex_part)
                .collect::<Result<Vec<VertexPart>>>()?;
            Ok(VertexContent { role, parts })
        })
        .collect::<Result<Vec<VertexContent>>>()?;

    let tools = if let Some(tool_defs) = &req.tools {
        if tool_defs.is_empty() {
            None
        } else {
            let declarations = tool_defs
                .iter()
                .map(|def| {
                    let mut params = def.input_schema.clone();
                    if let Some(Value::Object(map)) = params.as_mut() {
                        map.remove("$schema");
                    }
                    VertexFunctionDeclaration {
                        name: def.name.clone(),
                        description: def.description.clone(),
                        parameters: params,
                    }
                })
                .collect();
            Some(vec![VertexTool {
                function_declarations: declarations,
            }])
        }
    } else {
        None
    };

    let generation_config = req
        .config
        .as_ref()
        .and_then(|config_val| serde_json::from_value::<GeminiConfig>(config_val.clone()).ok());

    let tool_config = generation_config
        .as_ref()
        .and_then(|c| c.function_calling_config.as_ref())
        .map(|fcc| VertexToolConfig {
            function_calling_config: VertexFunctionCallingConfig {
                mode: fcc.mode.clone().unwrap_or_else(|| "AUTO".to_string()),
                allowed_function_names: fcc.allowed_function_names.clone(),
            },
        });

    Ok(VertexGeminiRequest {
        system_instruction,
        contents,
        tools,
        generation_config: generation_config.map(|c| VertexGenerationConfig { common_config: c }),
        tool_config,
        cached_content: None,
    })
}

/// Calculates basic usage statistics like character and media counts.
pub fn get_genkit_usage_stats(
    request_messages: &[genkit::message::MessageData],
    candidate_messages: &[genkit::message::MessageData],
) -> GenerationUsage {
    let mut input_characters = 0;
    let mut input_images = 0;
    let mut input_videos = 0;
    let mut input_audio_files = 0;

    for m in request_messages {
        for p in &m.content {
            if let Some(text) = &p.text {
                input_characters += text.len() as u32;
            }
            if let Some(media) = &p.media {
                if let Some(content_type) = &media.content_type {
                    if content_type.starts_with("image/") {
                        input_images += 1;
                    } else if content_type.starts_with("video/") {
                        input_videos += 1;
                    } else if content_type.starts_with("audio/") {
                        input_audio_files += 1;
                    }
                }
            }
        }
    }

    let mut output_characters = 0;
    let mut output_images = 0;
    let mut output_videos = 0;
    let mut output_audio_files = 0;

    for c in candidate_messages {
        for p in &c.content {
            if let Some(text) = &p.text {
                output_characters += text.len() as u32;
            }
            if let Some(media) = &p.media {
                if let Some(content_type) = &media.content_type {
                    if content_type.starts_with("image/") {
                        output_images += 1;
                    } else if content_type.starts_with("video/") {
                        output_videos += 1;
                    } else if content_type.starts_with("audio/") {
                        output_audio_files += 1;
                    }
                }
            }
        }
    }

    GenerationUsage {
        input_characters: Some(input_characters),
        input_images: Some(input_images),
        input_videos: Some(input_videos),
        input_audio_files: Some(input_audio_files),
        output_characters: Some(output_characters),
        output_images: Some(output_images),
        output_videos: Some(output_videos),
        output_audio_files: Some(output_audio_files),
        ..Default::default()
    }
}

/// Converts a `VertexGeminiResponse` into a Genkit `GenerateResponseData`.
pub fn to_genkit_response(
    req: &GenerateRequest,
    resp: VertexGeminiResponse,
) -> Result<GenerateResponseData> {
    let candidates = resp
        .candidates
        .into_iter()
        .enumerate()
        .map(|(i, candidate)| {
            let content = candidate
                .content
                .parts
                .into_iter()
                .map(|part| {
                    if let Some(fc) = part.function_call {
                        Ok(genkit::document::Part {
                            tool_request: Some(ToolRequest {
                                name: fc.name,
                                input: Some(fc.args),
                                r#ref: Some(i.to_string()),
                            }),
                            ..Default::default()
                        })
                    } else if part.thought.is_some() {
                        let metadata = part.thought_signature.map(|signature| {
                            let mut map = std::collections::HashMap::new();
                            map.insert("thoughtSignature".to_string(), signature.into());
                            map
                        });
                        Ok(genkit::document::Part {
                            reasoning: part.text.or_else(|| Some("".to_string())),
                            metadata,
                            ..Default::default()
                        })
                    } else {
                        Ok(genkit::document::Part {
                            text: part.text,
                            ..Default::default()
                        })
                    }
                })
                .collect::<Result<Vec<genkit::document::Part>>>()?;

            let message = genkit::message::MessageData {
                role: Role::Model,
                content,
                metadata: None,
            };
            let finish_reason = match candidate.finish_reason.as_deref() {
                Some("STOP") | Some("TOOL_CALL") => FinishReason::Stop,
                Some("MAX_TOKENS") => FinishReason::Length,
                Some("SAFETY") => FinishReason::Blocked,
                Some("RECITATION") | Some("OTHER") => FinishReason::Other,
                _ => FinishReason::Unknown,
            };

            let mut custom = serde_json::Map::new();
            if let Some(safety_ratings) = candidate.safety_ratings {
                custom.insert(
                    "safetyRatings".to_string(),
                    serde_json::to_value(safety_ratings)?,
                );
            }
            if let Some(citation_metadata) = candidate.citation_metadata {
                custom.insert(
                    "citationMetadata".to_string(),
                    serde_json::to_value(citation_metadata)?,
                );
            }

            Ok(CandidateData {
                index: i as u32,
                message,
                finish_reason: Some(finish_reason),
                finish_message: None,
                custom: if custom.is_empty() {
                    None
                } else {
                    Some(Value::Object(custom))
                },
            })
        })
        .collect::<Result<Vec<CandidateData>>>()?;

    let usage = resp.usage_metadata.map(|u| {
        let candidate_messages: Vec<genkit::message::MessageData> =
            candidates.iter().map(|c| c.message.clone()).collect();
        let mut usage_stats = get_genkit_usage_stats(&req.messages, &candidate_messages);
        usage_stats.input_tokens = Some(u.prompt_token_count);
        usage_stats.output_tokens = Some(u.candidates_token_count);
        usage_stats.total_tokens = Some(u.total_token_count);
        usage_stats
    });

    Ok(GenerateResponseData {
        candidates,
        usage,
        ..Default::default()
    })
}
