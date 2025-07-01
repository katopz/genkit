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

//! # Genkit
//!
//! The main Genkit library for Rust, providing tools for building AI-powered flows.
//!
//! This crate is the Rust equivalent of the main `genkit` npm package. It provides
//! high-level APIs for defining flows, models, tools, and other key components
//! of a Genkit application.

// Silence warnings for missing implementations during development.
#![allow(dead_code)]
#![allow(unused_variables)]

// Public modules that form the core API of the Genkit library.
pub mod client;
pub mod common;
pub mod context;
pub mod embedder;
pub mod error;
pub mod extract;
pub mod flow;
pub mod formats;
pub mod logging;
pub mod middleware;
pub mod model;
pub mod plugin;
pub mod prompt;
pub mod registry;
pub mod retriever;
pub mod schema;
pub mod testing;
pub mod tool;
pub mod tracing;

// Re-export key components for a unified and convenient API.
pub use self::embedder::{
    define_embedder, embed, embedder_ref, EmbedParams, EmbedderAction, EmbedderArgument,
    EmbedderInfo, EmbedderRef, Embedding, EmbeddingBatch,
};
pub use self::error::{Error, Result};
pub use self::flow::{define_flow, run as run_flow_step, Flow};
pub use self::model::{
    FinishReason, GenerateRequest, GenerateResponse, Message, Model, ModelInfo, Part, Role,
};
pub use self::plugin::Plugin;
pub use self::prompt::{
    define_prompt, is_executable_prompt, prompt, ExecutablePrompt, PromptAction, PromptConfig,
    PromptGenerateOptions,
};
pub use self::retriever::{
    define_indexer, define_retriever, index, indexer_ref, retrieve, retriever_ref, Document,
    IndexerAction, IndexerArgument, IndexerInfo, IndexerParams, IndexerRef, RetrieverAction,
    RetrieverArgument, RetrieverInfo, RetrieverParams, RetrieverRef,
};
pub use self::tool::{
    define_interrupt, define_tool, dynamic_tool, to_tool_definition, ToolAction, ToolArgument,
    ToolConfig, ToolDefinition,
};
// Re-export key types directly from the underlying crates for a flat, convenient API.
pub use genkit_ai::chat::Chat;
pub use genkit_ai::session::{Session, SessionStore};
pub use genkit_core::context::ActionContext;
pub use genkit_core::registry::Registry;
use once_cell::sync::OnceCell;
#[cfg(feature = "beta")]
use serde::{de::DeserializeOwned, Serialize};
use std::sync::Arc;

static GENKIT_INSTANCE: OnceCell<Arc<Genkit>> = OnceCell::new();

/// The main entry point for the Genkit framework.
pub struct Genkit {
    registry: Registry,
}

/// Options for creating a new session.
#[cfg(feature = "beta")]
#[derive(Default)]
pub struct CreateSessionOptions<S> {
    pub store: Option<Arc<dyn SessionStore<S>>>,
    pub initial_state: Option<S>,
}

impl Genkit {
    /// Initializes the Genkit framework with a list of plugins.
    pub async fn init(plugins: Vec<Arc<dyn Plugin>>) -> Result<&'static Arc<Self>> {
        let mut registry = Registry::new();
        for plugin in plugins {
            plugin.initialize(&mut registry).await?;
        }
        let instance = Arc::new(Self { registry });
        GENKIT_INSTANCE
            .set(instance.clone())
            .map_err(|_| Error::new_internal("Genkit already initialized."))?;
        Ok(GENKIT_INSTANCE.get().unwrap())
    }

    /// Returns a reference to the global Genkit instance.
    ///
    /// Panics if `init` has not been called.
    pub fn get() -> &'static Arc<Self> {
        GENKIT_INSTANCE
            .get()
            .expect("Genkit has not been initialized. Call Genkit::init() first.")
    }

    /// Returns a reference to the underlying registry.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// Creates a new, empty session.
    #[cfg(feature = "beta")]
    pub async fn create_session<S>(
        &self,
        options: CreateSessionOptions<S>,
    ) -> Result<Arc<Session<S>>>
    where
        S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
    {
        let session = Session::new(
            Arc::new(self.registry.clone()),
            options.store,
            None,
            options.initial_state,
        )
        .await?;
        Ok(Arc::new(session))
    }

    /// Loads an existing session from a store.
    ///
    /// Returns `Ok(None)` if the session is not found in the provided store.
    #[cfg(feature = "beta")]
    pub async fn load_session<S>(
        &self,
        id: String,
        store: Arc<dyn SessionStore<S>>,
    ) -> Result<Option<Arc<Session<S>>>>
    where
        S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
    {
        // First, check if the session exists.
        let session_data = store.get(&id).await?;
        if session_data.is_none() {
            return Ok(None);
        }

        // If it exists, Session::new will load it.
        let session = Session::new(
            Arc::new(self.registry.clone()),
            Some(store),
            Some(id),
            None, // initial_state is not used when loading
        )
        .await?;
        Ok(Some(Arc::new(session)))
    }
}
