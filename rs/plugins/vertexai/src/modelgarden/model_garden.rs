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
use genkit_ai::model::{GenerateResponseChunkData, ModelAction, ModelInfo, ModelRef};
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
#[allow(unused)]
pub fn llama3_1() -> ModelRef<ModelGardenModelConfig> {
    todo!();
    // model_ref(ModelInfo {
    //     name: "vertexai/llama-3.1".to_string(),
    //     label: "Vertex AI Llama 3.1".to_string(),
    //     ..Default::default()
    // })
}

/// A reference to the Llama 3.2 model in the Model Garden.
#[allow(unused)]
pub fn llama3_2() -> ModelRef<ModelGardenModelConfig> {
    todo!();
    // model_ref(ModelInfo {
    //     name: "vertexai/llama-3.2".to_string(),
    //     label: "Vertex AI Llama 3.2".to_string(),
    //     ..Default::default()
    // })
}

/// A reference to the Llama 3 model in the Model Garden (deprecated in favor of `llama3_1`).
#[deprecated(since = "0.1.0", note = "Please use `llama3_1` instead")]
#[allow(unused)]
pub fn llama3() -> ModelRef<ModelGardenModelConfig> {
    todo!();
    // model_ref(ModelInfo {
    //     name: "vertexai/llama3-405b".to_string(),
    //     label: "Vertex AI Llama3".to_string(),
    //     ..Default::default()
    // })
}

/// Defines a `ModelAction` for a Model Garden model that is compatible with the OpenAI API.
///
/// This function constructs the appropriate base URL for the model's endpoint and then
/// uses the `openai_compatible_model` compatibility layer to create the action.
pub fn model_garden_openai_compatible_model(
    model_ref: &ModelRef<serde_json::Value>,
    options: &VertexAIPluginOptions,
    base_url_template: Option<String>,
) -> ModelAction {
    use super::openai_compatibility::openai_types::{
        ChatCompletionResponse, CreateChatCompletionRequest,
    };
    use super::openai_compatibility::{from_openai_choice, to_openai_messages, to_openai_tool};

    use genkit_ai::model::{define_model, GenerateResponse};
    use genkit_core::Registry;
    use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};

    let model_name = model_ref.name.clone();
    let opts = options.clone();
    let template = base_url_template.unwrap_or_else(|| {
        "https://{location}-aiplatform.googleapis.com/v1beta1/projects/{projectId}/locations/{location}/endpoints/openapi"
            .to_string()
    });

    let runner =
        move |req: genkit_ai::model::GenerateRequest,
              _cb: Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>| {
            let model_name = model_name.clone();
            let opts = opts.clone();
            let template = template.clone();
            async move {
                let config: ModelGardenModelConfig = if let Some(val) = req.config.as_ref() {
                    serde_json::from_value(val.clone())
                        .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))?
                } else {
                    // This case seems to imply we should have a base config from the model_ref,
                    // but ModelRef<T> carries PhantomData, not a value.
                    // For now, we'll default. A proper implementation would need to resolve
                    // the config from the plugin options.
                    Default::default()
                };
                let params = crate::common::get_derived_params(&opts)
                    .await
                    .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))?;
                let location = config
                    .location
                    .as_ref()
                    .or(Some(&params.location))
                    .ok_or_else(|| {
                        genkit_core::error::Error::new_internal(
                            "Model Garden location is required.",
                        )
                    })?
                    .clone();
                let base_url = template
                    .replace("{location}", &location)
                    .replace("{projectId}", &params.project_id);
                let url = format!("{}/chat/completions", base_url);
                let token = params
                    .token_provider
                    .token(&["https://www.googleapis.com/auth/cloud-platform"])
                    .await
                    .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))?;
                let mut headers = HeaderMap::new();
                headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
                headers.insert(
                    AUTHORIZATION,
                    format!("Bearer {}", token.as_str()).parse().unwrap(),
                );
                let client = reqwest::Client::new();
                let messages = to_openai_messages(&req.messages);
                let tools = req
                    .tools
                    .as_ref()
                    .map(|t| t.iter().map(to_openai_tool).collect());
                let openai_req = CreateChatCompletionRequest {
                    model: model_name.clone(),
                    messages,
                    tools,
                };
                let response = client
                    .post(&url)
                    .headers(headers)
                    .json(&openai_req)
                    .send()
                    .await
                    .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))?;
                if !response.status().is_success() {
                    let status = response.status();
                    let err_text = response.text().await.unwrap_or_else(|e| e.to_string());
                    return Err(genkit_core::error::Error::new_internal(format!(
                        "API request failed with status {}: {}",
                        status, err_text
                    )));
                }
                let response_data = response
                    .json::<ChatCompletionResponse>()
                    .await
                    .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))?;
                Ok(GenerateResponse {
                    candidates: response_data
                        .choices
                        .into_iter()
                        .map(from_openai_choice)
                        .collect(),
                    ..Default::default()
                })
            }
        };
    let mut registry = Registry::default();
    define_model(
        &mut registry,
        genkit_ai::model::DefineModelOptions {
            name: model_ref.name.clone(),
            label: Some(model_ref.info.label.clone()),
            supports: model_ref.info.supports.clone(),
            versions: model_ref.info.versions.clone(),
            config_schema: Some(
                serde_json::to_value(schemars::schema_for!(ModelGardenModelConfig)).unwrap(),
            ),
        },
        runner,
    )
}
