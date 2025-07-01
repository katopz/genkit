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

use crate::common::get_derived_params;
use crate::{context_caching, Error, Result, VertexAIPluginOptions};
use genkit_ai::tool;
use genkit_ai::{
    message::Role,
    model::{
        define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
        GenerationUsage, ModelAction,
    },
    tool::ToolDefinition,
    ToolRequest,
};
use genkit_core::Registry;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Configuration structs for the Gemini model, aligned with the API.

#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SafetySettings {
    pub category: String,
    pub threshold: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct FunctionCallingConfig {
    pub mode: Option<String>,
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GeminiConfig {
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub safety_settings: Option<Vec<SafetySettings>>,
    pub function_calling_config: Option<FunctionCallingConfig>,
}

// Data structures that map to the Vertex AI Gemini API request/response format.

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VertexMedia {
    mime_type: String,
    data: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VertexFunctionCall {
    name: String,
    args: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VertexFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VertexPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<VertexMedia>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_call: Option<VertexFunctionCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    function_response: Option<VertexFunctionResponse>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VertexContent {
    role: String,
    parts: Vec<VertexPart>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexFunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexTool {
    function_declarations: Vec<VertexFunctionDeclaration>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexGenerationConfig {
    #[serde(flatten)]
    common_config: GeminiConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexFunctionCallingConfig {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    allowed_function_names: Option<Vec<String>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexToolConfig {
    function_calling_config: VertexFunctionCallingConfig,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexGeminiRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<VertexContent>,
    contents: Vec<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<VertexTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<VertexGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_config: Option<VertexToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cached_content: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexCandidate {
    content: VertexContent,
    finish_reason: Option<String>,
    // Other fields like index, safetyRatings, citationMetadata are ignored for now.
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexGeminiResponse {
    candidates: Vec<VertexCandidate>,
    usage_metadata: Option<VertexUsageMetadata>,
}

fn to_vertex_part(part: &genkit_ai::document::Part) -> Result<VertexPart> {
    if let Some(media) = &part.media {
        let (mime_type, data) = media.url.split_once(";base64,").ok_or_else(|| {
            Error::VertexAI("Media URL is not a valid base64 data URI.".to_string())
        })?;
        Ok(VertexPart {
            text: None,
            inline_data: Some(VertexMedia {
                mime_type: mime_type.replace("data:", ""),
                data: data.to_string(),
            }),
            function_call: None,
            function_response: None,
        })
    } else if let Some(tool_req) = &part.tool_request {
        Ok(VertexPart {
            text: None,
            inline_data: None,
            function_call: Some(VertexFunctionCall {
                name: tool_req.name.clone(),
                args: tool_req.input.clone().unwrap_or_default(),
            }),
            function_response: None,
        })
    } else if let Some(tool_resp) = &part.tool_response {
        Ok(VertexPart {
            text: None,
            inline_data: None,
            function_call: None,
            function_response: Some(VertexFunctionResponse {
                name: tool_resp.name.clone(),
                response: serde_json::json!({
                    "name": tool_resp.name,
                    "content": tool_resp.output
                }),
            }),
        })
    } else {
        Ok(VertexPart {
            text: part.text.clone(),
            inline_data: None,
            function_call: None,
            function_response: None,
        })
    }
}

/// Converts a Genkit `GenerateRequest` into a `VertexGeminiRequest`.
fn to_vertex_request(req: &GenerateRequest) -> Result<VertexGeminiRequest> {
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
            let parts = msg
                .content
                .iter()
                .map(to_vertex_part)
                .collect::<Result<Vec<VertexPart>>>()?;
            Ok(VertexContent { role, parts })
        })
        .collect::<Result<Vec<VertexContent>>>()?;

    let tools = req.tools.as_ref().map(|tools| {
        vec![VertexTool {
            function_declarations: tools
                .iter()
                .map(|t| tool::to_tool_definition(t.as_ref()))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .map(ToolRequest::from)
                .collect(),
        }]
    });

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

/// Converts a `VertexGeminiResponse` into a Genkit `GenerateResponseData`.
fn to_genkit_response(resp: VertexGeminiResponse) -> Result<GenerateResponseData> {
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
                        Ok(genkit_ai::document::Part {
                            tool_request: Some(ToolRequest {
                                name: fc.name,
                                input: Some(fc.args),
                                ..Default::default()
                            }),
                            ..Default::default()
                        })
                    } else {
                        Ok(genkit_ai::document::Part {
                            text: part.text,
                            ..Default::default()
                        })
                    }
                })
                .collect::<Result<Vec<genkit_ai::document::Part>>>()?;

            let message = genkit_ai::message::MessageData {
                role: Role::Model,
                content,
                metadata: None,
            };
            let finish_reason = match candidate.finish_reason.as_deref() {
                Some("STOP") => FinishReason::Stop,
                Some("MAX_TOKENS") => FinishReason::Length,
                Some("SAFETY") => FinishReason::Blocked,
                Some("TOOL_CALL") => FinishReason::Stop, // Maps to stop as per some conventions
                _ => FinishReason::Unknown,
            };
            Ok(CandidateData {
                index: i as u32,
                message,
                finish_reason: Some(finish_reason),
                finish_message: None,
            })
        })
        .collect::<Result<Vec<CandidateData>>>()?;

    let usage = resp.usage_metadata.map(|u| GenerationUsage {
        input_tokens: Some(u.prompt_token_count),
        output_tokens: Some(u.candidates_token_count),
        total_tokens: Some(u.total_token_count),
    });

    Ok(GenerateResponseData {
        candidates,
        usage,
        ..Default::default()
    })
}

/// The core runner for the Gemini model.
async fn gemini_runner(
    req: GenerateRequest,
    model_id: String,
    options: VertexAIPluginOptions,
) -> Result<GenerateResponseData> {
    let mut vertex_req = to_vertex_request(&req)?;
    let params = get_derived_params(&options).await?;

    if let Some(cache_config_details) = context_caching::utils::extract_cache_config(&req)? {
        if let Some(cache_result) = context_caching::handle_cache_if_needed(
            &params,
            &req,
            &vertex_req.contents,
            &model_id,
            &Some(cache_config_details),
        )
        .await?
        {
            vertex_req.contents = cache_result.remaining_contents;
            vertex_req.cached_content = cache_result.cache.name;
        }
    }

    let url = format!(
        "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent",
        params.location, params.project_id, params.location, model_id
    );

    let token = params
        .token_provider
        .token(&["https://www.googleapis.com/auth/cloud-platform"])
        .await
        .map_err(|e| Error::GcpAuth(format!("Failed to get auth token: {}", e)))?;

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {}", token.as_str()).parse().unwrap(),
    );

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .headers(headers)
        .json(&vertex_req)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        return Err(Error::VertexAI(format!(
            "API request failed with status {}: {}",
            status, error_text
        )));
    }

    // In a real implementation, this would handle the streaming response properly.
    // For now, we will collect the response and parse it as a single object for simplicity.
    // This is a placeholder for true streaming.
    let full_response_text = response.text().await?;
    let aggregated_response: Vec<VertexGeminiResponse> =
        serde_json::from_str(&full_response_text.replace("]\n[", ",")).map_err(|e| {
            Error::VertexAI(format!(
                "Failed to parse streaming response: {}. Body: {}",
                e, full_response_text
            ))
        })?;

    // We need to aggregate the candidates and usage from all chunks.
    let mut all_candidates = Vec::new();
    let mut total_usage = GenerationUsage::default();

    for chunk in aggregated_response {
        all_candidates.extend(chunk.candidates);
        if let Some(usage) = chunk.usage_metadata {
            total_usage.input_tokens = Some(usage.prompt_token_count); // Input tokens are the same in all chunks
            total_usage.output_tokens =
                Some(total_usage.output_tokens.unwrap_or(0) + usage.candidates_token_count);
            total_usage.total_tokens =
                Some(total_usage.total_tokens.unwrap_or(0) + usage.total_token_count);
        }
    }

    // Create a single response from aggregated data.
    let final_resp = VertexGeminiResponse {
        candidates: all_candidates,
        usage_metadata: Some(VertexUsageMetadata {
            prompt_token_count: total_usage.input_tokens.unwrap_or(0),
            candidates_token_count: total_usage.output_tokens.unwrap_or(0),
            total_token_count: total_usage.total_tokens.unwrap_or(0),
        }),
    };

    to_genkit_response(final_resp)
}

/// Defines a Gemini model action.
pub fn define_gemini_model(model_name: &str, options: &VertexAIPluginOptions) -> ModelAction {
    let model_id = model_name.to_string();
    let opts = options.clone();

    let model_options = genkit_ai::model::DefineModelOptions {
        name: format!("vertexai/{}", model_name),
        ..Default::default()
    };

    let mut registry = Registry::new();
    define_model(&mut registry, model_options, move |req, _| {
        let model_id_clone = model_id.clone();
        let opts_clone = opts.clone();
        Box::pin(async move {
            gemini_runner(req, model_id_clone, opts_clone)
                .await
                .map_err(|e| e.into())
        })
    })
}
