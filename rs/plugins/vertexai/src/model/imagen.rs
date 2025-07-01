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

use crate::common::VertexAIPluginOptions;
use crate::predict::predict_model;
use crate::Result;
use genkit_ai::{
    model::{
        define_model, CandidateData, FinishReason, GenerateRequest, GenerateResponseData,
        ModelAction,
    },
    Part,
};
use genkit_core::error::Error as CoreError;
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

#[derive(Serialize, Clone)]
struct VertexImagenInstance<'a> {
    prompt: &'a str,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexImagenParameters {
    sample_count: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexImagenPrediction {
    bytes_base64_encoded: String,
    mime_type: String,
}

/// The core runner for the Imagen model.
async fn imagen_runner(
    req: GenerateRequest,
    model_id: String,
    options: VertexAIPluginOptions,
) -> Result<GenerateResponseData> {
    let prompt = req
        .messages
        .last()
        .ok_or_else(|| CoreError::new_internal("Imagen requires a prompt in the last message."))
        .map_err(crate::Error::from)?
        .content
        .iter()
        .find_map(|p| p.text.as_deref())
        .ok_or_else(|| CoreError::new_internal("Imagen prompt message must contain text."))
        .map_err(crate::Error::from)?;

    let config: ImagenConfig = req
        .config
        .as_ref()
        .map(|v| serde_json::from_value(v.clone()))
        .transpose()
        .map_err(|e| {
            crate::Error::from(CoreError::new_internal(format!(
                "Failed to parse Imagen config: {}",
                e
            )))
        })?
        .unwrap_or_default();

    let parameters = VertexImagenParameters {
        sample_count: config.num_images.unwrap_or(1),
    };

    let instances = vec![VertexImagenInstance { prompt }];

    let response =
        predict_model::<_, _, VertexImagenPrediction>(&options, &model_id, &instances, &parameters)
            .await?;

    let content: Vec<Part> = response
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
        .collect::<crate::Result<Vec<Part>>>()?;

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
        async move {
            imagen_runner(req, model_id_clone, opts_clone)
                .await
                .map_err(|e| e.into())
        }
    })
}
