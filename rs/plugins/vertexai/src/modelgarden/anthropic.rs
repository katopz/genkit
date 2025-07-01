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

//! # Vertex AI Model Garden - Anthropic Models
//!
//! This module provides an implementation for using Anthropic's Claude models
//! via the Vertex AI Model Garden.

use crate::common::{get_derived_params, VertexAIPluginOptions};
use crate::{Error, Result};
use genkit_ai::model::{
    define_model, model_ref, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
    ModelAction, ModelInfo, ModelInfoSupports, ModelRef,
};
use genkit_ai::{Part, Role};
use genkit_core::Registry;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Configuration for Anthropic models.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct AnthropicConfig {
    pub temperature: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Option<Vec<String>>,
}

// Structs that map to the Anthropic Messages API format on Vertex AI

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum AnthropicContent {
    Text { text: String },
    // Other types like image, tool_use, tool_result would go here if supported.
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct AnthropicMessage {
    role: String, // "user" or "assistant"
    content: Vec<AnthropicContent>,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    anthropic_version: &'static str,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(flatten)]
    config: &'a AnthropicConfig,
    // stream: bool, - handled by endpoint choice
}

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    // id: String,
    // model: String,
    // r#type: String, // e.g., "message"
    // role: String,   // "assistant"
    content: Vec<AnthropicContent>,
    stop_reason: String,
    // stop_sequence: Option<String>,
    // usage: AnthropicUsage,
}

fn to_anthropic_request<'a>(
    req: &'a GenerateRequest,
    config: &'a AnthropicConfig,
) -> Result<AnthropicRequest<'a>> {
    let mut system_prompt: Option<String> = None;
    let messages: Vec<AnthropicMessage> = req
        .messages
        .iter()
        .filter_map(|msg| {
            if msg.role == Role::System {
                system_prompt = Some(
                    msg.content
                        .iter()
                        .map(|p| p.text.as_deref().unwrap_or_default())
                        .collect(),
                );
                None // Remove system message from main message list
            } else {
                let role = match msg.role {
                    Role::User => "user".to_string(),
                    Role::Model => "assistant".to_string(),
                    _ => "user".to_string(), // Default other roles to user
                };
                let content = msg
                    .content
                    .iter()
                    .map(|p| AnthropicContent::Text {
                        text: p.text.as_deref().unwrap_or_default().to_string(),
                    })
                    .collect();
                Some(Ok(AnthropicMessage { role, content }))
            }
        })
        .collect::<Result<Vec<AnthropicMessage>>>()?;

    Ok(AnthropicRequest {
        anthropic_version: "vertex-2023-10-16",
        messages,
        system: system_prompt,
        config,
    })
}

fn from_anthropic_response(resp: AnthropicResponse) -> Result<GenerateResponseData> {
    let content = resp
        .content
        .into_iter()
        .map(|c| match c {
            AnthropicContent::Text { text } => Part::text(text),
        })
        .collect();

    let finish_reason = match resp.stop_reason.as_str() {
        "end_turn" => FinishReason::Stop,
        "max_tokens" => FinishReason::Length,
        "stop_sequence" => FinishReason::Stop,
        _ => FinishReason::Other,
    };

    let candidate = CandidateData {
        index: 0,
        message: genkit_ai::message::MessageData {
            role: Role::Model,
            content,
            ..Default::default()
        },
        finish_reason: Some(finish_reason),
        ..Default::default()
    };

    Ok(GenerateResponseData {
        candidates: vec![candidate],
        ..Default::default()
    })
}

async fn anthropic_runner(
    req: GenerateRequest,
    model_id: String,
    options: VertexAIPluginOptions,
) -> Result<GenerateResponseData> {
    let params = get_derived_params(&options).await?;
    // Note: Anthropic models use a different endpoint format.
    let url = format!(
        "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/anthropic/models/{}:streamRawPredict",
        params.location, params.project_id, params.location, model_id
    );

    let config: AnthropicConfig = req
        .config
        .as_ref()
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()?
        .unwrap_or_default();

    let anthropic_req = to_anthropic_request(&req, &config)?;

    let token = params
        .token_provider
        .token(&["https://www.googleapis.com/auth/cloud-platform"])
        .await?;

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
        .json(&anthropic_req)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(Error::VertexAI(format!(
            "API request failed with status {}: {}",
            response.status(),
            response.text().await?
        )));
    }

    // This is a simplified handling of the streaming response.
    // A proper implementation would need to parse the event stream.
    let full_response_text = response.text().await?;
    let mut last_json = None;
    for line in full_response_text.lines() {
        if line.starts_with("data: ") {
            if let Ok(value) = serde_json::from_str::<Value>(&line[6..]) {
                if value["type"] == "message_stop" {
                    // This event signals the end, but the content is in prior events.
                    // We grab the `message` field from this event as the final one.
                    if let Some(msg) = value.get("message") {
                        last_json = Some(msg.clone());
                    }
                    break;
                }
            }
        }
    }

    let anthropic_response: AnthropicResponse = serde_json::from_value(
        last_json.ok_or_else(|| Error::VertexAI("No valid message found in stream".to_string()))?,
    )?;

    from_anthropic_response(anthropic_response)
}

pub fn define_anthropic_model(
    model_ref: genkit_ai::model::ModelRef<serde_json::Value>,
    options: &VertexAIPluginOptions,
) -> ModelAction {
    let model_id = model_ref.name.split('/').last().unwrap().to_string();
    let opts = options.clone();

    let model_options = genkit_ai::model::DefineModelOptions {
        name: model_ref.name,
        label: Some(model_ref.info.label),
        supports: model_ref.info.supports,
        versions: model_ref.info.versions,
        config_schema: Some(serde_json::from_str("{}").unwrap()),
    };

    let mut registry = Registry::default();
    define_model(&mut registry, model_options, move |req, _| {
        let model_id = model_id.clone();
        let opts = opts.clone();
        async move {
            anthropic_runner(req, model_id, opts)
                .await
                .map_err(|e| e.into())
        }
    })
}

// Helper functions to create model references
pub fn claude_3_5_sonnet() -> ModelRef<AnthropicConfig> {
    model_ref(ModelInfo {
        name: "vertexai/claude-3-5-sonnet-20240620".to_owned(),
        label: "Claude 3.5 Sonnet".to_string(),
        supports: Some(ModelInfoSupports {
            multiturn: Some(true),
            media: Some(true),
            tools: Some(true),
            system_role: Some(true),
            output: Some(vec!["text".to_string()]),
            ..Default::default()
        }),
        ..Default::default()
    })
}
