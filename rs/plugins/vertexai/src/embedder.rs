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

use crate::{common::VertexAIPluginOptions, predict::predict_model, Error};
use genkit_ai::embedder::{define_embedder, EmbedRequest, EmbedResponse, Embedding};
use genkit_core::Registry;
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

#[derive(Deserialize)]
struct VertexEmbeddingPrediction {
    embeddings: VertexEmbeddings,
}

#[derive(Deserialize)]
struct VertexEmbeddings {
    values: Vec<f32>,
    // statistics are ignored for now.
}

/// Defines a Vertex AI embedder action.
pub fn define_vertex_ai_embedder(
    model_name: &str,
    options: &VertexAIPluginOptions,
) -> genkit_ai::embedder::EmbedderAction<VertexEmbeddingConfig> {
    let model_id = model_name.to_string();
    let opts = options.clone();
    let mut regitry = Registry::new();
    define_embedder(
        &mut regitry,
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

                #[derive(Serialize)]
                struct EmptyParams {}
                let parameters = EmptyParams {};

                let vertex_resp = predict_model(&opts, &model_id, &instances, &parameters).await?;

                if vertex_resp.predictions.len() != req.input.len() {
                    return Err(Error::VertexAI(format!(
                        "Mismatched response count: expected {}, got {}",
                        req.input.len(),
                        vertex_resp.predictions.len()
                    ))
                    .into());
                }

                let embeddings = vertex_resp
                    .predictions
                    .into_iter()
                    .map(|p: VertexEmbeddingPrediction| Embedding {
                        embedding: p.embeddings.values,
                        metadata: None, // No metadata from Vertex API
                    })
                    .collect();

                Ok(EmbedResponse { embeddings })
            }
        },
    )
}
