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

//! # Vertex AI Model Garden - OpenAI-Compatible Models
//!
//! This module provides support for models in the Vertex AI Model Garden that
//! expose an OpenAI-compatible API, such as various versions of Llama.

use crate::common::VertexAIPluginOptions;
use genkit_ai::model::{model_ref, ModelAction, ModelInfo, ModelInfoSupports, ModelRef};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::openai_compatibility::OpenAIConfig;

/// Configuration for Model Garden models that use the OpenAI-compatible API.
/// This allows overriding the GCP location per request.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct ModelGardenModelConfig {
    #[serde(flatten)]
    pub open_ai_config: OpenAIConfig,
    pub location: Option<String>,
}

/// A reference to the Llama 3.1 model in the Model Garden.
pub fn llama3_1() -> ModelRef<ModelGardenModelConfig> {
    model_ref(ModelInfo {
        name: "vertexai/llama-3.1".to_string(),
        label: "Llama 3.1".to_string(),
        supports: ModelInfoSupports {
            multiturn: Some(true),
            tools: Some(true),
            media: Some(false),
            system_role: Some(true),
            output: Some(vec!["text".to_string(), "json".to_string()]),
            ..Default::default()
        },
        versions: vec!["meta/llama3-405b-instruct-maas".to_string()],
        ..Default::default()
    })
}

/// A reference to the Llama 3.2 model in the Model Garden.
pub fn llama3_2() -> ModelRef<ModelGardenModelConfig> {
    model_ref(ModelInfo {
        name: "vertexai/llama-3.2".to_string(),
        label: "Llama 3.2".to_string(),
        supports: ModelInfoSupports {
            multiturn: Some(true),
            tools: Some(true),
            media: Some(true),
            system_role: Some(true),
            output: Some(vec!["text".to_string(), "json".to_string()]),
            ..Default::default()
        },
        versions: vec!["meta/llama-3.2-90b-vision-instruct-maas".to_string()],
        ..Default::default()
    })
}

/// A reference to the Llama 3 model in the Model Garden (deprecated in favor of `llama3_1`).
#[deprecated(since = "0.1.0", note = "Please use `llama3_1` instead")]
pub fn llama3() -> ModelRef<ModelGardenModelConfig> {
    model_ref(ModelInfo {
        name: "vertexai/llama3-405b".to_string(),
        label: "Llama 3.1 405b".to_string(),
        supports: ModelInfoSupports {
            multiturn: Some(true),
            tools: Some(true),
            media: Some(false),
            system_role: Some(true),
            output: Some(vec!["text".to_string()]),
            ..Default::default()
        },
        versions: vec!["meta/llama3-405b-instruct-maas".to_string()],
        ..Default::default()
    })
}

/// Defines a `ModelAction` for a Model Garden model that is compatible with the OpenAI API.
///
/// This function constructs the appropriate base URL for the model's endpoint and then
/// uses the `openai_compatible_model` compatibility layer to create the action.
pub fn model_garden_openai_compatible_model(
    model_ref: ModelRef<ModelGardenModelConfig>,
    options: &VertexAIPluginOptions,
    base_url_template: Option<String>,
) -> ModelAction {
    let default_template = "https://{location}-aiplatform.googleapis.com/v1beta1/projects/{projectId}/locations/{location}/endpoints/openapi".to_string();
    let template = base_url_template.unwrap_or(default_template);

    // The openai_compatible_model function will handle the dynamic creation of the client
    // and the fetching of auth tokens for each request.
    super::openai_compatibility::openai_compatible_model(model_ref)
}
