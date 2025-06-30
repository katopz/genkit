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
pub mod model;

use self::common::VertexAIPluginOptions;
use self::embedder::define_vertex_ai_embedder;
use self::model::gemini::define_gemini_model;
use self::model::imagen::define_imagen_model;
use self::model::{SUPPORTED_EMBEDDER_MODELS, SUPPORTED_GEMINI_MODELS, SUPPORTED_IMAGEN_MODELS};
use async_trait::async_trait;
use genkit_ai::{embedder_ref, model_ref, EmbedderRef, ModelRef};
use genkit_core::{plugin::Plugin, registry::Registry};
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
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for genkit_core::error::Error {
    fn from(e: Error) -> Self {
        genkit_core::error::Error::new_internal(e.to_string())
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

    async fn register_models(&self, registry: &mut Registry) -> Result<()> {
        for model_name in SUPPORTED_GEMINI_MODELS {
            let model = define_gemini_model(model_name, &self.options);
            registry.register_action(Arc::new(model))?;
        }
        for model_name in SUPPORTED_IMAGEN_MODELS {
            let model = define_imagen_model(model_name, &self.options);
            registry.register_action(Arc::new(model))?;
        }
        for model_name in SUPPORTED_EMBEDDER_MODELS {
            let embedder = define_vertex_ai_embedder(model_name, &self.options);
            registry.register_action(Arc::new(embedder))?;
        }
        Ok(())
    }
}

#[async_trait]
impl Plugin for VertexAIPlugin {
    fn name(&self) -> &'static str {
        "vertexai"
    }

    async fn initialize(&self, registry: &mut Registry) -> genkit_core::error::Result<()> {
        self.register_models(registry).await.map_err(|e| e.into())
    }
}

/// Creates an instance of the Vertex AI plugin.
pub fn vertex_ai(options: VertexAIPluginOptions) -> Arc<dyn Plugin> {
    Arc::new(VertexAIPlugin::new(options))
}

/// Helper function to create a `ModelRef` for a Gemini model.
pub fn gemini(name: &str) -> ModelRef {
    model_ref(&format!("vertexai/{}", name))
}

/// Helper function to create a `ModelRef` for an Imagen model.
pub fn imagen(name: &str) -> ModelRef {
    model_ref(&format!("vertexai/{}", name))
}

/// Helper function to create an `EmbedderRef` for a Vertex AI text embedding model.
pub fn text_embedding(name: &str) -> EmbedderRef {
    embedder_ref(&format!("vertexai/{}", name))
}
