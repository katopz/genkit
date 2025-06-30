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

use crate::common::{get_derived_params, VertexAIPluginOptions};
use genkit_ai::{
    model::{
        define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
        GenerationUsage, ModelAction,
    },
    tool::ToolDefinition,
};
use genkit_core::error::{Error, Result};
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
pub struct GeminiConfig {
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
    pub safety_settings: Option<Vec<SafetySettings>>,
}

// Data structures that map to the Vertex AI Gemini API request/response format.

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexPart {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<VertexMedia>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexMedia {
    mime_type: String,
    data: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexContent {
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
struct VertexGeminiRequest {
    contents: Vec<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<VertexTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<VertexGenerationConfig>,
    // TODO: Add tool_config
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

/// Converts a Genkit `GenerateRequest` into a `VertexGeminiRequest`.
fn to_vertex_request(req: &GenerateRequest) -> Result<VertexGeminiRequest> {
    let contents = req
        .messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                genkit_ai::message::Role::Model => "model".to_string(),
                _ => "user".to_string(), // System and Tool roles are mapped to user
            };
            let parts = msg
                .content
                .iter()
                .map(|part| {
                    if let Some(media) = &part.media {
                        let (mime_type, data) =
                            media.url.split_once(";base64,").ok_or_else(|| {
                                Error::new_internal(
                                    "Media URL is not a valid base64 data URI.".to_string(),
                                )
                            })?;
                        Ok(VertexPart {
                            text: None,
                            inline_data: Some(VertexMedia {
                                mime_type: mime_type.replace("data:", ""),
                                data: data.to_string(),
                            }),
                        })
                    } else {
                        Ok(VertexPart {
                            text: part.text.clone(),
                            inline_data: None,
                        })
                    }
                })
                .collect::<Result<Vec<VertexPart>>>()?;
            Ok(VertexContent { role, parts })
        })
        .collect::<Result<Vec<VertexContent>>>()?;

    let tools = req.tools.as_ref().map(|tools| {
        vec![VertexTool {
            function_declarations: tools
                .iter()
                .map(|t: &ToolDefinition| VertexFunctionDeclaration {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                })
                .collect(),
        }]
    });

    let generation_config = req
        .config
        .as_ref()
        .and_then(|config_val| serde_json::from_value::<GeminiConfig>(config_val.clone()).ok());

    Ok(VertexGeminiRequest {
        contents,
        tools,
        generation_config: generation_config.map(|c| VertexGenerationConfig { common_config: c }),
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
                .map(|part| genkit_ai::document::Part {
                    text: part.text,
                    ..Default::default()
                })
                .collect();
            let message = genkit_ai::message::MessageData {
                role: genkit_ai::message::Role::Model,
                content,
                metadata: None,
            };
            let finish_reason = match candidate.finish_reason.as_deref() {
                Some("STOP") => FinishReason::Stop,
                Some("MAX_TOKENS") => FinishReason::Length,
                Some("SAFETY") => FinishReason::Blocked,
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
    let vertex_req = to_vertex_request(&req)?;
    let params = get_derived_params(&options).await?;
    let url = format!(
        "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:streamGenerateContent",
        params.location, params.project_id, params.location, model_id
    );

    let token = params
        .token_provider
        .token(&["https://www.googleapis.com/auth/cloud-platform"])
        .await
        .map_err(|e| Error::new_internal(format!("Failed to get auth token: {}", e)))?;

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
        .await
        .map_err(|e| Error::new_internal(format!("API request failed: {}", e)))?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(Error::new_internal(format!(
            "API request failed with status {}: {}",
            status, error_text
        )));
    }

    // In a real implementation, this would handle the streaming response properly.
    // For now, we will collect the response and parse it as a single object for simplicity.
    // This is a placeholder for true streaming.
    let full_response_text = response
        .text()
        .await
        .map_err(|e| Error::new_internal(format!("Failed to read stream: {}", e)))?;
    let aggregated_response: Vec<VertexGeminiResponse> =
        serde_json::from_str(&full_response_text.replace("]\n[", ",")).map_err(|e| {
            Error::new_internal(format!(
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

    let model_options = genkit_ai::model::DefineModelOptions::<GeminiConfig> {
        name: format!("vertexai/{}", model_name),
        ..Default::default()
    };

    define_model(model_options, move |req, _| {
        let model_id_clone = model_id.clone();
        let opts_clone = opts.clone();
        Box::pin(async move { gemini_runner(req, model_id_clone, opts_clone).await })
    })
}
