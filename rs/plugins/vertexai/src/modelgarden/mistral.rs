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

//! # Mistral Models
//!
//! This module provides a client for Mistral's models in the Model Garden.

use crate::common::VertexAIPluginOptions;
use crate::modelgarden::openai_compatibility::openai_types::{
    ChatCompletionResponse, CreateChatCompletionRequest,
};
use crate::modelgarden::openai_compatibility::{
    from_openai_choice, to_openai_messages, to_openai_tool, OpenAIConfig,
};
use crate::Result;
use genkit_ai::model::{
    define_model, GenerateResponse, ModelAction, ModelInfo, ModelInfoSupports, ModelRef,
};
use genkit_core::Registry;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};

// Model references
#[allow(unused)]
pub fn mistral_large() -> ModelRef<OpenAIConfig> {
    todo!();
    // model_ref(ModelInfo {
    //     name: "vertexai/mistral-large".to_string(),
    //     label: "vertexai/mistral-large".to_string(),
    //     ..Default::default()
    // })
}

#[allow(unused)]
pub fn mistral_nemo() -> ModelRef<OpenAIConfig> {
    todo!();
    // model_ref(ModelInfo {
    //     name: "vertexai/mistral-nemo".to_string(),
    //     label: "vertexai/mistral-nemo".to_string(),
    //     ..Default::default()
    // })
}

#[allow(unused)]
pub fn codestral() -> ModelRef<OpenAIConfig> {
    todo!();
    // model_ref(ModelInfo {
    //     name: "vertexai/codestral".to_string(),
    //     label: "vertexai/codestral".to_string(),
    //     ..Default::default()
    // })
}

async fn mistral_runner(
    req: genkit_ai::model::GenerateRequest,
    model_name: String,
    options: VertexAIPluginOptions,
    base_url_template: String,
) -> Result<genkit_ai::model::GenerateResponse> {
    let params = crate::common::get_derived_params(&options).await?;

    // Allow overriding location per-request, similar to other models
    let location = req
        .config
        .as_ref()
        .and_then(|c| c.get("location"))
        .and_then(|v| v.as_str())
        .unwrap_or(&params.location)
        .to_string();

    let base_url = base_url_template
        .replace("{location}", &location)
        .replace("{projectId}", &params.project_id);

    let url = format!("{}/chat/completions", base_url);

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

    let version = req
        .config
        .as_ref()
        .and_then(|c| c.get("version").and_then(|v| v.as_str()))
        .unwrap_or(&model_name)
        .to_string();

    let messages = to_openai_messages(&req.messages);
    let tools = req
        .tools
        .as_ref()
        .map(|requests| requests.iter().map(to_openai_tool).collect());

    let mistral_req = CreateChatCompletionRequest {
        model: version,
        messages,
        tools,
    };

    let response = client
        .post(&url)
        .headers(headers)
        .json(&mistral_req)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(crate::Error::VertexAI(format!(
            "API request failed with status {}: {}",
            response.status(),
            response.text().await?
        )));
    }

    let response_data = response.json::<ChatCompletionResponse>().await?;

    Ok(GenerateResponse {
        candidates: response_data
            .choices
            .into_iter()
            .map(from_openai_choice)
            .collect(),
        ..Default::default()
    })
}

/// Defines a Mistral model from the Model Garden.
pub fn define_mistral_model(
    model_ref: &ModelRef<serde_json::Value>,
    options: &VertexAIPluginOptions,
    base_url_template: String,
) -> ModelAction {
    let model_name = model_ref.name.clone();
    let opts = options.clone();

    let info = match model_name.as_str() {
        "vertexai/mistral-large" => ModelInfo {
            label: "Vertex AI Model Garden - Mistral Large".to_string(),
            versions: Some(vec![
                "mistral-large-2411".to_string(),
                "mistral-large-2407".to_string(),
            ]),
            supports: Some(ModelInfoSupports {
                multiturn: Some(true),
                media: Some(false),
                tools: Some(true),
                system_role: Some(true),
                output: Some(vec!["text".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        },
        "vertexai/mistral-nemo" => ModelInfo {
            label: "Vertex AI Model Garden - Mistral Nemo".to_string(),
            versions: Some(vec!["mistral-nemo-2407".to_string()]),
            supports: Some(ModelInfoSupports {
                multiturn: Some(true),
                media: Some(false),
                tools: Some(false),
                system_role: Some(true),
                output: Some(vec!["text".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        },
        "vertexai/codestral" => ModelInfo {
            label: "Vertex AI Model Garden - Codestral".to_string(),
            versions: Some(vec!["codestral-2405".to_string()]),
            supports: Some(ModelInfoSupports {
                multiturn: Some(true),
                media: Some(false),
                tools: Some(false),
                system_role: Some(true),
                output: Some(vec!["text".to_string()]),
                ..Default::default()
            }),
            ..Default::default()
        },
        _ => panic!("Unsupported Mistral model: {}", model_name),
    };

    let model_options = genkit_ai::model::DefineModelOptions {
        name: model_name.clone(),
        label: Some(info.label),
        supports: info.supports,
        versions: info.versions,
        config_schema: Some(serde_json::from_str("{}").unwrap()),
    };

    let registry = Registry::default();
    define_model(&registry, model_options, move |req, _| {
        let model_name = model_name.clone();
        let opts = opts.clone();
        let base_url_template = base_url_template.clone();
        Box::pin(async move {
            mistral_runner(req, model_name, opts, base_url_template)
                .await
                .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))
        })
    })
}
