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

//! # Vertex AI Embedders
//!
//! This module provides the implementation for Vertex AI text embedding models.

use crate::common::{get_derived_params, VertexAIPluginOptions};
use genkit_ai::embedder::{define_embedder, EmbedRequest, EmbedResponse, Embedding};
use genkit_core::error::{Error, Result};
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Configuration for Vertex AI embedding models.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct VertexEmbeddingConfig {
    // Currently no specific options are defined, but this provides a place for them.
}

// Structs for the Vertex AI API request/response format.

#[derive(Serialize)]
struct VertexEmbeddingInstance<'a> {
    content: &'a str,
}

#[derive(Serialize)]
struct VertexEmbeddingRequest<'a> {
    instances: Vec<VertexEmbeddingInstance<'a>>,
}

#[derive(Deserialize)]
struct VertexEmbeddingPrediction {
    embeddings: VertexEmbeddings,
}

#[derive(Deserialize)]
struct VertexEmbeddings {
    values: Vec<f32>,
    // statistics are ignored for now.
}

#[derive(Deserialize)]
struct VertexEmbeddingResponse {
    predictions: Vec<VertexEmbeddingPrediction>,
}

/// Invokes the Vertex AI API to generate embeddings.
async fn invoke_vertex_embedding_api(
    options: &VertexAIPluginOptions,
    model_id: &str,
    request: &VertexEmbeddingRequest<'_>,
) -> Result<VertexEmbeddingResponse> {
    let params = get_derived_params(options).await?;
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
        .json(request)
        .send()
        .await
        .map_err(|e| Error::new_internal(format!("API request failed: {}", e)))?;

    let status = response.status();
    if !status.is_success() {
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        Err(Error::new_internal(format!(
            "API request failed with status {}: {}",
            status, error_text
        )))
    } else {
        response
            .json::<VertexEmbeddingResponse>()
            .await
            .map_err(|e| Error::new_internal(format!("Failed to parse API response: {}", e)))
    }
}

/// Defines a Vertex AI embedder action.
pub fn define_vertex_ai_embedder(
    model_name: &str,
    options: &VertexAIPluginOptions,
) -> genkit_ai::embedder::EmbedderAction<VertexEmbeddingConfig> {
    let model_id = model_name.to_string();
    let opts = options.clone();
    define_embedder(
        &format!("vertexai/{}", model_name),
        move |req: EmbedRequest<VertexEmbeddingConfig>, _| {
            let model_id = model_id.clone();
            let opts = opts.clone();
            async move {
                let contents: Vec<String> = req.input.iter().map(|doc| doc.text()).collect();
                let instances: Vec<VertexEmbeddingInstance> = contents
                    .iter()
                    .map(|content| VertexEmbeddingInstance { content })
                    .collect();

                let vertex_req = VertexEmbeddingRequest { instances };
                let vertex_resp =
                    invoke_vertex_embedding_api(&opts, &model_id, &vertex_req).await?;

                if vertex_resp.predictions.len() != req.input.len() {
                    return Err(Error::new_internal(format!(
                        "Mismatched response count: expected {}, got {}",
                        req.input.len(),
                        vertex_resp.predictions.len()
                    )));
                }

                let embeddings = vertex_resp
                    .predictions
                    .into_iter()
                    .map(|p| Embedding {
                        embedding: p.embeddings.values,
                        metadata: None, // No metadata from Vertex API
                    })
                    .collect();

                Ok(EmbedResponse { embeddings })
            }
        },
    )
}
