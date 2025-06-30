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

use async_trait::async_trait;
use genkit_ai::{model_ref, embedder_ref, ModelRef, EmbedderRef};
use genkit_core::{
    error::Result,
    plugin::Plugin,
    registry::Registry,
};
use std::sync::Arc;

/// The Vertex AI plugin.
#[derive(Debug)]
pub struct VertexAIPlugin;

impl VertexAIPlugin {
    pub fn new() -> Arc<dyn Plugin> {
        Arc::new(Self)
    }
}

impl Default for VertexAIPlugin {
    fn default() -> Self {
        Self {}
    }
}

#[async_trait]
impl Plugin for VertexAIPlugin {
    fn name(&self) -> &'static str {
        "vertexai"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        // TODO: Register models and embedders here.
        // For example:
        // let gemini_pro = model::define_gemini_pro();
        // registry.register_action(gemini_pro)?;
        Ok(())
    }
}

/// Helper function to create a `ModelRef` for a Gemini model.
pub fn gemini(name: &str) -> ModelRef {
    model_ref(name)
}

/// Helper function to create an `EmbedderRef` for the text embedding gecko model.
pub fn text_embedding_gecko() -> EmbedderRef {
    embedder_ref("text-embedding-gecko")
}
