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

//! # List Vertex AI Models
//!
//! This module provides functionality to list available models from Vertex AI.

use crate::common::get_derived_params;
use crate::{Error, Result, VertexAIPluginOptions};
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use serde::Deserialize;

/// Represents a model available on Vertex AI.
#[derive(Debug, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub name: String,
    pub launch_stage: String,
    // Other fields from the API can be added here if needed.
}

/// The response from the list models API.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListModelsResponse {
    publisher_models: Vec<Model>,
}

/// Lists available models from the Vertex AI API.
pub async fn list_models(options: &VertexAIPluginOptions) -> Result<Vec<Model>> {
    let params = get_derived_params(options).await?;
    let url = format!(
        "https://{}-aiplatform.googleapis.com/v1beta1/publishers/google/models",
        params.location
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
    headers.insert("x-goog-user-project", params.project_id.parse().unwrap());

    let client = reqwest::Client::new();
    let response = client.get(&url).headers(headers).send().await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await?;
        return Err(Error::VertexAI(format!(
            "API request to list models failed with status {}: {}",
            status, error_text
        )));
    }

    let list_response = response
        .json::<ListModelsResponse>()
        .await
        .map_err(Error::from)?;

    Ok(list_response.publisher_models)
}
