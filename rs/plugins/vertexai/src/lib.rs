// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Genkit Vertex AI Plugin
//!
//! This crate provides the Vertex AI plugin for the Genkit framework in Rust.

// Declare modules that will be created later.
pub mod common;
pub mod context_caching;
pub mod embedder;
pub mod evaluation;
pub mod list_models;
pub mod model;
pub mod modelgarden;
pub mod predict;

use self::common::VertexAIPluginOptions;
use self::embedder::define_vertex_ai_embedder;
use self::list_models::list_models;
use self::model::gemini::define_gemini_model;
use self::model::imagen::define_imagen_model;
use self::model::SUPPORTED_EMBEDDER_MODELS;
use async_trait::async_trait;
use genkit::{plugin::Plugin, registry::Registry};
use genkit_ai::{embedder_ref, EmbedderRef, ModelRef};
use serde_json::Value;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Genkit core error: {0}")]
    GenkitCore(#[from] genkit_core::error::Error),
    #[error("GCP auth error: {0}")]
    GcpAuth(String),
    #[error("Request error: {0}")]
    Request(#[from] reqwest::Error),
    #[error("Vertex AI error: {0}")]
    VertexAI(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for genkit_core::error::Error {
    fn from(e: Error) -> Self {
        genkit_core::error::Error::new_internal(e.to_string())
    }
}

impl From<gcp_auth::Error> for Error {
    fn from(e: gcp_auth::Error) -> Self {
        Error::GcpAuth(e.to_string())
    }
}

/// The Vertex AI plugin.
#[derive(Debug)]
pub struct VertexAIPlugin {
    options: VertexAIPluginOptions,
}

impl VertexAIPlugin {
    pub fn new(options: VertexAIPluginOptions) -> Self {
        Self { options }
    }

    async fn register_models(&self, registry: &Registry) -> Result<()> {
        println!("🦀 register_models");

        const KNOWN_DECOMISSIONED_MODELS: &[&str] = &[
            "gemini-pro-vision",
            "gemini-pro",
            "gemini-ultra",
            "gemini-ultra-vision",
        ];

        let all_models = list_models(&self.options).await?;

        for model in all_models {
            let short_name = model.name.split('/').next_back().unwrap_or(&model.name);

            if KNOWN_DECOMISSIONED_MODELS.contains(&short_name) {
                continue;
            }

            if short_name.contains("gemini") {
                println!("🦀 call define_gemini_model:{}", short_name);
                let action = define_gemini_model(short_name, &self.options);
                registry.register_action(&format!("vertexai/{}", short_name), action)?;
            } else if short_name.contains("imagen") {
                let action = define_imagen_model(short_name, &self.options);
                registry.register_action(&format!("vertexai/{}", short_name), action)?;
            } else if SUPPORTED_EMBEDDER_MODELS.contains(&short_name) {
                let embedder = define_vertex_ai_embedder(short_name, &self.options);
                registry.register_action(&format!("vertexai/{}", short_name), embedder)?;
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Plugin for VertexAIPlugin {
    fn name(&self) -> &'static str {
        "vertexai"
    }

    async fn initialize(&self, registry: &Registry) -> genkit_core::error::Result<()> {
        self.register_models(registry).await.map_err(|e| e.into())
    }
}

/// Creates an instance of the Vertex AI plugin.
pub fn vertex_ai(options: VertexAIPluginOptions) -> Arc<dyn Plugin> {
    Arc::new(VertexAIPlugin::new(options))
}

/// Helper function to create a `ModelRef` for a Gemini model.
pub fn gemini(_name: &str) -> ModelRef<Value> {
    todo!();
    // model_ref(ModelInfo {
    //     name: format!("vertexai/{}", name).to_owned(),
    //     ..Default::default()
    // })
}

/// Helper function to create a `ModelRef` for an Imagen model.
pub fn imagen(_name: &str) -> ModelRef<Value> {
    todo!();
    // model_ref(ModelInfo {
    //     name: format!("vertexai/{}", name).to_owned(),
    //     ..Default::default()
    // })
}

/// Helper function to create an `EmbedderRef` for a Vertex AI text embedding model.
pub fn text_embedding(name: &str) -> EmbedderRef {
    embedder_ref(&format!("vertexai/{}", name))
}
