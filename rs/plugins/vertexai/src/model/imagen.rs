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

//! # Vertex AI Imagen Models
//!
//! This module provides the implementation for the Imagen family of models
//! on Vertex AI for image generation.

use crate::common::{get_derived_params, VertexAIPluginOptions};
use genkit_ai::{
    model::{
        define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
        ModelAction,
    },
    Part,
};
use genkit_core::error::{Error, Result};
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// Configuration for Imagen models.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ImagenConfig {
    pub num_images: Option<u32>,
    // Add other Imagen-specific parameters here, e.g., seed, aspect_ratio
}

// Data structures that map to the Vertex AI Imagen API request/response format.

#[derive(Serialize)]
struct VertexImagenInstance<'a> {
    prompt: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexImagenParameters {
    sample_count: u32,
}

#[derive(Serialize)]
struct VertexImagenRequest<'a> {
    instances: Vec<VertexImagenInstance<'a>>,
    parameters: VertexImagenParameters,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexImagenPrediction {
    bytes_base64_encoded: String,
    mime_type: String,
}

#[derive(Deserialize)]
struct VertexImagenResponse {
    predictions: Vec<VertexImagenPrediction>,
    // DeployedModelId is ignored
}

/// Converts a Genkit `GenerateRequest` into a `VertexImagenRequest`.
fn to_vertex_request(req: &GenerateRequest) -> Result<VertexImagenRequest> {
    let prompt = req
        .messages
        .last()
        .ok_or_else(|| Error::new_internal("Imagen requires a prompt in the last message."))?
        .content
        .iter()
        .find_map(|p| p.text.as_deref())
        .ok_or_else(|| Error::new_internal("Imagen prompt message must contain text."))?;

    let config: ImagenConfig = req
        .config
        .as_ref()
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|e| Error::new_internal(format!("Failed to parse Imagen config: {}", e)))?
        .unwrap_or_default();

    let parameters = VertexImagenParameters {
        sample_count: config.num_images.unwrap_or(1),
    };

    Ok(VertexImagenRequest {
        instances: vec![VertexImagenInstance { prompt }],
        parameters,
    })
}

/// Converts a `VertexImagenResponse` into a Genkit `GenerateResponseData`.
fn to_genkit_response(resp: VertexImagenResponse) -> Result<GenerateResponseData> {
    let content: Vec<Part> = resp
        .predictions
        .into_iter()
        .map(|pred| {
            Ok(Part {
                media: Some(genkit_ai::Media {
                    content_type: Some(pred.mime_type.clone()),
                    url: format!(
                        "data:{};base64,{}",
                        pred.mime_type, pred.bytes_base64_encoded
                    ),
                }),
                ..Default::default()
            })
        })
        .collect::<Result<Vec<Part>>>()?;

    let candidate = CandidateData {
        index: 0,
        message: genkit_ai::message::MessageData {
            role: genkit_ai::message::Role::Model,
            content,
            metadata: None,
        },
        finish_reason: Some(FinishReason::Stop),
        finish_message: None,
    };

    Ok(GenerateResponseData {
        candidates: vec![candidate],
        ..Default::default()
    })
}

/// The core runner for the Imagen model.
async fn imagen_runner(
    req: GenerateRequest,
    model_id: String,
    options: VertexAIPluginOptions,
) -> Result<GenerateResponseData> {
    let vertex_req = to_vertex_request(&req)?;
    let params = get_derived_params(&options).await?;
    let url = format!(
        "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:predict",
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

    let vertex_resp = response
        .json::<VertexImagenResponse>()
        .await
        .map_err(|e| Error::new_internal(format!("Failed to parse API response: {}", e)))?;

    to_genkit_response(vertex_resp)
}

/// Defines an Imagen model action.
pub fn define_imagen_model(model_name: &str, options: &VertexAIPluginOptions) -> ModelAction {
    let model_id = model_name.to_string();
    let opts = options.clone();

    let info = genkit_ai::model::ModelInfo {
        label: format!("Vertex AI - {}", model_name),
        supports: genkit_ai::model::ModelInfoSupports {
            media: Some(true),
            multiturn: Some(false), // Imagen is not a chat model
            tools: Some(false),
            system_role: Some(false),
            ..Default::default()
        },
        ..Default::default()
    };

    let model_options = genkit_ai::model::DefineModelOptions {
        name: format!("vertexai/{}", model_name),
        label: info.label,
        supports: info.supports,
        config_schema: Some(ImagenConfig::default()),
        versions: info.versions,
    };

    define_model(model_options, move |req, _| {
        let model_id_clone = model_id.clone();
        let opts_clone = opts.clone();
        async move { imagen_runner(req, model_id_clone, opts_clone).await }
    })
}
