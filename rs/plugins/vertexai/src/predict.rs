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

//! # Vertex AI Predict API Client
//!
//! This module provides a generic client for making calls to the Vertex AI
//! `predict` endpoint, which is used by models like Imagen and the text embedders.

use crate::{common::get_derived_params, Error, Result, VertexAIPluginOptions};
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use serde::{de::DeserializeOwned, Serialize};

/// The generic request body for the predict API.
#[derive(Serialize)]
struct PredictionRequest<'a, I, P> {
    instances: &'a [I],
    parameters: &'a P,
}

/// The generic response body from the predict API.
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictionResponse<R> {
    pub predictions: Vec<R>,
    // DeployedModelId is ignored
}

/// Calls the Vertex AI `predict` endpoint for a given model.
///
/// This function is a generic helper to interact with models that use the
/// standard `predict` method on Vertex AI, such as Imagen and embedding models.
/// It handles authentication, request serialization, and response deserialization.
pub async fn predict_model<I, P, R>(
    options: &VertexAIPluginOptions,
    model_id: &str,
    instances: &[I],
    parameters: &P,
) -> Result<PredictionResponse<R>>
where
    I: Serialize,
    P: Serialize,
    R: DeserializeOwned,
{
    let params = get_derived_params(options).await?;
    let url = format!(
        "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:predict",
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
    let request_body = PredictionRequest {
        instances,
        parameters,
    };

    let response = client
        .post(&url)
        .headers(headers)
        .json(&request_body)
        .send()
        .await
        .map_err(Error::Request)?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        return Err(Error::VertexAI(format!(
            "API request failed with status {}: {}",
            status, error_text
        )));
    }

    response
        .json::<PredictionResponse<R>>()
        .await
        .map_err(Error::Request)
}
