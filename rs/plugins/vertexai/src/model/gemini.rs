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
use log;

use genkit_ai::model::{define_model, GenerateRequest, GenerateResponseData, ModelAction};
use genkit_core::Registry;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};

// Configuration structs for the Gemini model, aligned with the API.
pub(crate) use super::helpers::*;
pub use super::types::*;

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
        "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:generateContent",
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

    let request_body_json_str =
        serde_json::to_string_pretty(&vertex_req).unwrap_or_else(|e| e.to_string());
    log::debug!("Vertex AI Request Body: {}", request_body_json_str);

    let response = client
        .post(&url)
        .headers(headers)
        .json(&vertex_req)
        .send()
        .await?;

    let status = response.status();
    let response_text = response.text().await?;

    if !status.is_success() {
        log::error!("Vertex AI API Error: {} - {}", status, &response_text);
        return Err(Error::VertexAI(format!(
            "API request failed with status {}: {}",
            status, response_text
        )));
    }
    log::debug!("Gemini Raw Response: {}", response_text);
    let final_resp: VertexGeminiResponse = serde_json::from_str(&response_text).map_err(|e| {
        log::error!(
            "Failed to decode gemini response body: {}. Body: {}",
            e,
            response_text
        );
        Error::Json(e)
    })?;

    to_genkit_response(&req, final_resp)
}

/// Defines a Gemini model action.
pub fn define_gemini_model(model_name: &str, options: &VertexAIPluginOptions) -> ModelAction {
    let model_id = model_name.to_string();
    let opts = options.clone();

    let model_options = genkit_ai::model::DefineModelOptions {
        name: format!("vertexai/{}", model_name),
        ..Default::default()
    };

    let registry = Registry::new();
    define_model(&registry, model_options, move |req, _| {
        let model_id_clone = model_id.clone();
        let opts_clone = opts.clone();
        Box::pin(async move {
            gemini_runner(req, model_id_clone, opts_clone)
                .await
                .map_err(|e| e.into())
        })
    })
}
