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
pub mod embedder;
pub mod model;

use self::common::VertexAIPluginOptions;
use self::embedder::define_vertex_ai_embedder;
use self::model::gemini::define_gemini_model;
use self::model::imagen::define_imagen_model;
use async_trait::async_trait;
use genkit_ai::{embedder_ref, model_ref, EmbedderRef, ModelRef};
use genkit_core::{error::Result, plugin::Plugin, registry::Registry};
use std::sync::Arc;

// Lists of supported models, similar to the TS implementation.
const SUPPORTED_GEMINI_MODELS: &[&str] = &[
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash-latest",
    "gemini-1.0-pro",
];

const SUPPORTED_IMAGEN_MODELS: &[&str] = &["imagen-3.0-generate"];

const SUPPORTED_EMBEDDER_MODELS: &[&str] = &[
    "text-embedding-004",
    "text-multilingual-embedding-002",
    "text-embedding-gecko@003",
    "text-embedding-gecko-multilingual@001",
    "multimodalembedding@001",
];

/// The Vertex AI plugin.
#[derive(Debug)]
pub struct VertexAIPlugin {
    options: VertexAIPluginOptions,
}

impl VertexAIPlugin {
    pub fn new(options: VertexAIPluginOptions) -> Self {
        Self { options }
    }
}

#[async_trait]
impl Plugin for VertexAIPlugin {
    fn name(&self) -> &'static str {
        "vertexai"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
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
