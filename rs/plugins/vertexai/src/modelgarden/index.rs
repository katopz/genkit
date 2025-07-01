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

//! # Vertex AI Model Garden Plugin
//!
//! This module provides the main plugin for integrating with models from the
//! Vertex AI Model Garden.

use super::{
    anthropic::define_anthropic_model, mistral::define_mistral_model,
    model_garden::model_garden_openai_compatible_model, types::ModelGardenPluginOptions,
};
use async_trait::async_trait;
use genkit_core::{plugin::Plugin, registry::Registry, Result};
use std::sync::Arc;

fn is_anthropic_model(name: &str) -> bool {
    name.contains("claude")
}

fn is_mistral_model(name: &str) -> bool {
    name.contains("mistral") || name.contains("codestral")
}

fn is_llama_model(name: &str) -> bool {
    name.contains("llama")
}

/// The Vertex AI Model Garden plugin.
#[derive(Debug)]
pub struct VertexAIModelGardenPlugin {
    options: ModelGardenPluginOptions,
}

impl VertexAIModelGardenPlugin {
    /// Creates a new `VertexAIModelGardenPlugin`.
    pub fn new(options: ModelGardenPluginOptions) -> Self {
        Self { options }
    }
}

#[async_trait]
impl Plugin for VertexAIModelGardenPlugin {
    fn name(&self) -> &'static str {
        "vertexai_model_garden"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let base_options = &self.options.common;
        let model_garden_options = &self.options.model_garden;

        for model_ref in &model_garden_options.models {
            let action = if is_anthropic_model(&model_ref.name) {
                let typed_model_ref = genkit_ai::ModelRef {
                    name: model_ref.name.clone(),
                    info: model_ref.info.clone(),
                    config: std::marker::PhantomData,
                };
                define_anthropic_model(typed_model_ref, base_options)
            } else if is_mistral_model(&model_ref.name) {
                let config = match model_ref.config.as_ref() {
                    Some(v) => Some(serde_json::from_value(v.clone())?),
                    None => None,
                };
                let typed_model_ref = genkit_ai::ModelRef {
                    name: model_ref.name.clone(),
                    info: model_ref.info.clone(),
                    config,
                };
                define_mistral_model(
                    typed_model_ref,
                    base_options,
                    model_garden_options
                        .open_ai_base_url_template
                        .unwrap_or_default(),
                )
            } else if is_llama_model(&model_ref.name) {
                let config = match model_ref.config.as_ref() {
                    Some(v) => Some(serde_json::from_value(v.clone())?),
                    None => None,
                };
                let typed_model_ref = genkit_ai::ModelRef {
                    name: model_ref.name.clone(),
                    info: model_ref.info.clone(),
                    config,
                };
                model_garden_openai_compatible_model(
                    typed_model_ref,
                    base_options,
                    model_garden_options.open_ai_base_url_template.clone(),
                )
            } else {
                return Err(genkit_core::error::Error::new_internal(format!(
                    "Unsupported Model Garden model: {}",
                    model_ref.name
                )));
            };
            registry.register_action(model_ref.name.clone(), action)?;
        }
        Ok(())
    }
}

/// Creates an instance of the Vertex AI Model Garden plugin.
pub fn vertex_ai_model_garden(options: ModelGardenPluginOptions) -> Arc<dyn Plugin> {
    Arc::new(VertexAIModelGardenPlugin::new(options))
}
