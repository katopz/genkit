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
//
// The main Genkit library for Rust, providing tools for building AI-powered flows.
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
pub mod evaluator;
pub mod extract;
pub mod flow;
pub mod formats;
pub mod helpers;
pub mod logging;
pub mod middleware;
pub mod model;
pub mod plugin;
pub mod prompt;
pub mod registry;
pub mod reranker;
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
pub use self::evaluator::{define_evaluator, EvaluatorAction, EvaluatorFn};

pub use self::flow::{define_flow, run as run_flow_step, Flow};

pub use self::model::{
    FinishReason, GenerateRequest, GenerateResponse, Message, Model, ModelInfo, Part, Role,
};
pub use self::plugin::Plugin;

pub use self::prompt::{
    define_prompt, is_executable_prompt, prompt, ExecutablePrompt, PromptAction, PromptConfig,
    PromptGenerateOptions,
};
pub use self::reranker::{define_reranker, RerankerAction, RerankerFn};
pub use self::retriever::{
    define_indexer, define_retriever, index, indexer_ref, retrieve, retriever_ref, Document,
    IndexerAction, IndexerArgument, IndexerInfo, IndexerParams, IndexerRef, RetrieverAction,
    RetrieverArgument, RetrieverInfo, RetrieverParams, RetrieverRef,
};
pub use self::tool::{
    define_interrupt, define_tool, dynamic_tool, to_tool_definition, ToolAction, ToolArgument,
    ToolConfig, ToolDefinition,
};
pub use genkit_ai::generate::GenerateOptions;
use genkit_ai::generate::{generate as _generate, generate_stream as _generate_stream};
// Re-export key types directly from the underlying crates for a flat, convenient API.
pub use genkit_ai::chat::Chat;
use genkit_ai::model::{BackgroundModelAction, DefineBackgroundModelOptions, DefineModelOptions};
use genkit_ai::reranker::{RerankerRequest, RerankerResponse};
use genkit_ai::retriever::{IndexerRequest, RetrieverRequest, RetrieverResponse};
pub use genkit_ai::session::{Session, SessionStore};
use genkit_ai::tool::ToolFnOptions;
pub use genkit_ai::GenerateStreamResponse;
use genkit_ai::{
    define_background_model, define_model, BaseEvalDataPoint, EmbedRequest, EmbedResponse,
    EvalResponse, GenerateResponseChunkData, ModelAction,
};
pub use genkit_core::context::ActionContext;
pub use genkit_core::registry::Registry;
use genkit_core::{Action, ActionFnArg};
use schemars::JsonSchema;
use serde::Deserialize;
#[cfg(feature = "beta")]
use serde::{de::DeserializeOwned, Serialize};
use std::future::Future;
use std::sync::Arc;

/// The main entry point for the Genkit framework.
pub struct Genkit {
    options: GenkitOptions,
    registry: Registry,
    context: Option<ActionContext>,
}

/// Options for Genkit initialization.
#[derive(Default)]
pub struct GenkitOptions {
    pub plugins: Vec<Arc<dyn Plugin>>,
    pub default_model: Option<String>,
    pub context: Option<ActionContext>,
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
    pub async fn init(options: GenkitOptions) -> Result<Arc<Self>> {
        let mut registry = Registry::new();
        if let Some(model_name) = options.default_model.clone() {
            registry.set_default_model(model_name);
        }
        for plugin in &options.plugins {
            plugin.initialize(&mut registry).await?;
        }
        let context = options.context.clone();
        let instance = Arc::new(Self {
            options,
            registry,
            context,
        });
        Ok(instance)
    }

    pub fn from_registry(
        options: GenkitOptions,
        registry: Registry,
        context: Option<ActionContext>,
    ) -> Self {
        Self {
            options,
            registry,
            context,
        }
    }

    /// Returns a reference to the underlying registry.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    pub fn context(&self) -> Option<&ActionContext> {
        self.context.as_ref()
    }

    /// Defines a new tool and registers it with the Genkit registry.
    pub fn define_tool<I, O, F, Fut>(
        &self,
        config: ToolConfig<I, O>,
        runner: F,
    ) -> ToolAction<I, O, ()>
    where
        I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
        O: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
        F: Fn(I, ToolFnOptions) -> Fut + Send + Sync + Clone + 'static,
        Fut: Future<Output = Result<O>> + Send + 'static,
    {
        let mut registry = self.registry.clone();
        define_tool(&mut registry, config, runner)
    }

    /// Defines and registers a flow function.
    pub fn define_flow<I, O, S, F, Fut>(&self, name: impl Into<String>, func: F) -> Flow<I, O, S>
    where
        I: DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
        O: Serialize + JsonSchema + Send + Sync + 'static,
        S: Serialize + JsonSchema + Send + Sync + Clone + 'static,
        F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<O>> + Send,
        Action<I, O, S>: genkit_core::registry::ErasedAction + 'static,
    {
        define_flow(&mut self.registry.clone(), name, func)
    }

    /// Defines a new model and adds it to the registry.
    pub fn define_model<F, Fut>(&self, options: DefineModelOptions, f: F) -> ModelAction
    where
        F: Fn(GenerateRequest, Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<genkit_ai::GenerateResponseData, genkit_core::error::Error>>
            + Send
            + 'static,
    {
        define_model(&mut self.registry.clone(), options, f)
    }

    /// Defines a new background model and adds it to the registry.
    pub fn define_background_model(
        &self,
        registry: &mut genkit_core::Registry,
        options: DefineBackgroundModelOptions,
    ) -> BackgroundModelAction {
        define_background_model(&mut self.registry.clone(), options)
    }

    /// Defines and registers a prompt based on a function.
    pub async fn define_prompt<I, O, C>(
        &self,
        config: PromptConfig<I, O, C>,
    ) -> ExecutablePrompt<I, O, C>
    where
        I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
        O: for<'de> Deserialize<'de>
            + Serialize
            + Send
            + Sync
            + std::fmt::Debug
            + Clone
            + Default
            + 'static,
        C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
    {
        define_prompt(&mut self.registry.clone(), config)
    }

    /// Creates a retriever action for the provided RetrieverFn implementation.
    pub async fn define_retriever<I, F, Fut>(&self, name: &str, runner: F) -> RetrieverAction<I>
    where
        I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
        F: Fn(RetrieverRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<RetrieverResponse>> + Send + 'static,
    {
        define_retriever(&mut self.registry.clone(), name, runner)
    }

    // TODO
    // /// Defines a simple retriever that maps existing data into documents.
    // pub async fn define_simple_retriever<C, R>(
    //     &self,
    //     options: SimpleRetrieverOptions,
    //     handler: R,
    // ) -> Result<RetrieverAction<C, R>>
    // where
    //     C: Serialize + for<'de> DeserializeOwned + Send + Sync + 'static,
    //     R: SimpleRetrieverFn<C> + 'static,
    // {
    //     define_simple_retriever(&mut self.registry, options, handler).await
    // }

    /// Creates an indexer action for the provided IndexerFn implementation.
    pub fn define_indexer<I, F, Fut>(&self, name: &str, runner: F) -> IndexerAction<I>
    where
        I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
        F: Fn(IndexerRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        define_indexer(&mut self.registry.clone(), name, runner)
    }

    /// Creates evaluator action for the provided EvaluatorFn implementation.
    pub fn define_evaluator<I, F, Fut>(&self, name: &str, runner: F) -> EvaluatorAction<I>
    where
        I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
        F: Fn(BaseEvalDataPoint) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EvalResponse>> + Send + 'static,
    {
        define_evaluator(&mut self.registry.clone(), name, runner)
    }

    /// Creates embedder model for the provided EmbedderFn model implementation.
    pub fn define_embedder<I, F, Fut>(&self, name: &str, runner: F) -> EmbedderAction<I>
    where
        I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
        F: Fn(EmbedRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<EmbedResponse>> + Send + 'static,
    {
        define_embedder(&mut self.registry.clone(), name, runner)
    }

    /// Creates reranker action for the provided RerankerFn implementation.
    pub fn define_reranker<I, F, Fut>(&self, name: &str, runner: F) -> RerankerAction<I>
    where
        I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
        F: Fn(RerankerRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut
            + Send
            + Sync
            + 'static,
        Fut: Future<Output = Result<RerankerResponse>> + Send + 'static,
    {
        define_reranker(&mut self.registry.clone(), name, runner)
    }

    /// Generates content using a model.
    pub async fn generate<O>(&self, prompt: &str) -> Result<GenerateResponse<O>>
    where
        O: Clone
            + Default
            + for<'de> DeserializeOwned
            + Serialize
            + Send
            + Sync
            + 'static
            + std::fmt::Debug,
    {
        _generate(
            &self.registry,
            GenerateOptions {
                ..Default::default()
            },
        )
        .await
    }

    /// Generates content using a model with options.
    pub async fn generate_with_options<O>(
        &self,
        options: GenerateOptions<O>,
    ) -> Result<GenerateResponse<O>>
    where
        O: Clone
            + Default
            + for<'de> DeserializeOwned
            + Serialize
            + Send
            + Sync
            + 'static
            + std::fmt::Debug,
    {
        _generate(&self.registry, options).await
    }

    /// Generates content and streams the response.
    pub fn generate_stream<O>(
        &self,
        options: GenerateOptions<O>,
    ) -> genkit_ai::generate::GenerateStreamResponse<O>
    where
        O: Clone
            + Default
            + for<'de> DeserializeOwned
            + Serialize
            + Send
            + Sync
            + 'static
            + std::fmt::Debug,
    {
        _generate_stream(&self.registry, options)
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
