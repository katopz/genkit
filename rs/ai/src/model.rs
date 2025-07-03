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

//! # Generative Models
//!
//! This module defines the core traits and functions for working with generative
//! AI models, including text, chat, and embedding models. It provides a unified
//! interface for interacting with different model providers and allows for
//! middleware-based customization of model behavior.
//!
//! ## Key Concepts
//!
//! - **`ModelAction`**: Represents a model that can be invoked to generate content.
//! - **`ModelMiddleware`**: Allows intercepting and modifying requests and responses.
//! - **`define_model`**: A function for creating new `ModelAction` instances.
//!

use ::core::future::Future;
use ::core::marker::Send;
use ::serde::{Deserialize, Serialize};
use genkit_core::action::ActionMetadata;
use genkit_core::context::ActionContext;
use genkit_core::error::Result;
use genkit_core::{action::StreamingResponse, registry::ErasedAction, Action};
use schemars::JsonSchema;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

use crate::{MessageData, Part, Role, ToolRequest};

pub mod middleware;

/// Represents a loaded model ready for generation, streaming, or other tasks.
pub type LoadedModel<T, E> = dyn Fn(
        GenerateRequest,
        Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>,
    ) -> Pin<Box<dyn Future<Output = Result<T, E>> + Send>>
    + Send
    + Sync;

/// Describes the capabilities of a generative model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfoSupports {
    /// Whether the model supports multi-turn conversations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiturn: Option<bool>,
    /// Whether the model supports image, audio, or other media inputs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<bool>,
    /// Whether the model supports tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<bool>,
    /// Whether the model supports a system role.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_role: Option<bool>,
    /// A list of output formats supported by the model (e.g., "json", "text").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<String>>,
    /// Whether the model supports context augmentation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<bool>,
    /// Whether the model supports long-running operations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_running: Option<bool>,
}

/// Provides information about a generative model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub versions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports: Option<ModelInfoSupports>,
}

/// Common configuration options for generative models.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationCommonConfig {
    /// Controls the randomness of the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// The maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// The number of top-k candidates to consider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// The cumulative probability of top-p candidates to consider.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// A list of sequences that will stop generation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// A request to a generative model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateRequest {
    pub messages: Vec<MessageData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<Vec<Part>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_tool_requests: Option<bool>,
}

/// The reason why a model finished generating a response.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
#[derive(Default)]
pub enum FinishReason {
    /// The model finished generating the response.
    Stop,
    /// The model reached the maximum number of tokens.
    Length,
    /// The model was blocked due to a safety setting.
    Blocked,
    /// The model finished for some other reason.
    Other,
    /// The finish reason is unknown.
    #[default]
    Unknown,
    /// The model was interrupted.
    Interrupted,
}

/// Usage information for a generation request.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationUsage {
    /// The number of tokens in the input.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    /// The number of tokens in the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    /// The total number of tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

/// A candidate response from a generative model.
#[derive(Default, Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CandidateData {
    pub index: u32,
    pub message: MessageData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_message: Option<String>,
}

/// A response from a generative model.
#[derive(Default, Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseData {
    pub candidates: Vec<CandidateData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<Value>,
    // This is set when the response is for a long-running operation.
    // It can be used with `getFlowState` to retrieve the latest state.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<String>,
    // This is set when a request with `maxTurns` is made. It contains all the
    // intermediate responses from the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregated: Option<Vec<GenerateResponseData>>,
}

/// A chunk of a streaming response from a generative model.
#[derive(Default, Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseChunkData {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    pub content: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<Value>,
}

/// The response from a model generation call.
pub type GenerateResponse = GenerateResponseData;

/// A boxed, pinned future.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The `next` function passed to a `ModelMiddleware`.
pub type ModelMiddlewareNext<'a> = Box<
    dyn Fn(
            GenerateRequest,
        ) -> BoxFuture<'a, Result<GenerateResponseData, genkit_core::error::Error>>
        + Send
        + Sync,
>;

/// A middleware function for a `ModelAction`.
pub type ModelMiddleware = Arc<
    dyn for<'a> Fn(
            GenerateRequest,
            ModelMiddlewareNext<'a>,
        )
            -> BoxFuture<'a, Result<GenerateResponseData, genkit_core::error::Error>>
        + Send
        + Sync,
>;

/// An action that can be used to generate content.
#[derive(Clone)]
pub struct ModelAction(Action<GenerateRequest, GenerateResponseData, GenerateResponseChunkData>);

impl Deref for ModelAction {
    type Target = Action<GenerateRequest, GenerateResponseData, GenerateResponseChunkData>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ErasedAction for ModelAction {
    fn run_http_json<'a, 'async_trait>(
        &'a self,
        input: Value,
        context: Option<ActionContext>,
    ) -> ::core::pin::Pin<Box<dyn Future<Output = Result<Value>> + Send + 'async_trait>>
    where
        'a: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.0.run_http_json(input, context).await })
    }

    fn stream_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<StreamingResponse<Value, Value>> {
        self.0.stream_http_json(input, context)
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn metadata(&self) -> &ActionMetadata {
        self.0.metadata()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Clone)]
pub struct BackgroundModelAction(Action<GenerateRequest, GenerateResponseData, ()>);
impl Deref for BackgroundModelAction {
    type Target = Action<GenerateRequest, GenerateResponseData, ()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ErasedAction for BackgroundModelAction {
    fn run_http_json<'a, 'async_trait>(
        &'a self,
        input: Value,
        context: Option<ActionContext>,
    ) -> ::core::pin::Pin<Box<dyn Future<Output = Result<Value>> + Send + 'async_trait>>
    where
        'a: 'async_trait,
        Self: 'async_trait,
    {
        Box::pin(async move { self.0.run_http_json(input, context).await })
    }

    fn stream_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<StreamingResponse<Value, Value>> {
        self.0.stream_http_json(input, context)
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn metadata(&self) -> &ActionMetadata {
        self.0.metadata()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Options for defining a generative model.
#[derive(Default, Serialize, Deserialize, Debug, Clone, JsonSchema)]
pub struct DefineModelOptions {
    pub name: String,
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub versions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports: Option<ModelInfoSupports>,
    pub config_schema: Option<Value>,
}

/// Defines a new generative model.
fn get_model_middleware(options: &DefineModelOptions) -> Vec<ModelMiddleware> {
    let mut middleware: Vec<ModelMiddleware> = Vec::new();
    let supports = options.supports.clone().unwrap_or_default();

    middleware.push(middleware::download_request_media(None, None));
    middleware.push(middleware::validate_support(
        options.name.clone(),
        supports.clone(),
    ));

    if supports.system_role != Some(true) {
        middleware.push(middleware::simulate_system_prompt(None));
    }

    middleware.push(middleware::augment_with_context(None));
    middleware.push(middleware::simulate_constrained_generation(None));

    middleware
}

pub fn define_model<F, Fut>(
    registry: &mut genkit_core::Registry,
    options: DefineModelOptions,
    f: F,
) -> ModelAction
where
    F: Fn(GenerateRequest, Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>) -> Fut
        + Send
        + Sync
        + 'static,
    Fut: Future<Output = Result<GenerateResponseData, genkit_core::error::Error>> + Send + 'static,
{
    let all_middlewares = Arc::new(get_model_middleware(&options));
    let f = Arc::new(f);

    let model_info = ModelInfo {
        name: options.name.clone(),
        label: options.label.clone().unwrap_or_default(),
        versions: options.versions.clone(),
        supports: options.supports.clone(),
    };

    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), json!(options.name));
    metadata.insert("type".to_string(), json!("model"));
    metadata.insert("metadata".to_string(), json!(model_info));
    if let Some(config_schema) = &options.config_schema {
        metadata.insert("configSchema".to_string(), config_schema.clone());
    }
    let metadata_value = json!(metadata);
    let metadata_map: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_value(metadata_value).unwrap();

    let action_f =
        move |req: GenerateRequest,
              args: genkit_core::action::ActionFnArg<GenerateResponseChunkData>| {
            let f = f.clone();
            let middlewares = all_middlewares.clone();

            async move {
                if args.streaming_requested {
                    let callback = Some(Box::new(move |chunk| {
                        let _ = args.chunk_sender.send(chunk);
                    })
                        as Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>);
                    // TODO: Apply middleware to streaming requests.
                    return f(req, callback).await;
                }

                fn apply_middleware<F2, Fut2>(
                    req: GenerateRequest,
                    middlewares: Arc<Vec<ModelMiddleware>>,
                    index: usize,
                    f: Arc<F2>,
                ) -> BoxFuture<'static, Result<GenerateResponseData, genkit_core::error::Error>>
                where
                    F2: Fn(
                            GenerateRequest,
                            Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>,
                        ) -> Fut2
                        + Send
                        + Sync
                        + 'static,
                    Fut2: Future<Output = Result<GenerateResponseData, genkit_core::error::Error>>
                        + Send
                        + 'static,
                {
                    if index < middlewares.len() {
                        let middleware = middlewares[index].clone();
                        let next: ModelMiddlewareNext<'static> = Box::new(move |req| {
                            apply_middleware(req, middlewares.clone(), index + 1, f.clone())
                        });
                        middleware(req, next)
                    } else {
                        Box::pin(f(req, None))
                    }
                }
                apply_middleware(req, middlewares, 0, f).await
            }
        };

    let action = genkit_core::action::ActionBuilder::new(
        genkit_core::registry::ActionType::Model,
        options.name.clone(),
        action_f,
    )
    .with_metadata(metadata_map)
    .build();
    let model_action = ModelAction(action);
    registry
        .register_action(options.name.clone(), model_action.clone())
        .unwrap();
    model_action
}

pub struct DefineBackgroundModelOptions {
    pub name: String,
    pub label: Option<String>,
    pub versions: Option<Vec<String>>,
    pub supports: Option<ModelInfoSupports>,
    pub config_schema: Option<Value>,
    pub start: Arc<dyn Fn(GenerateRequest) -> Result<String, String> + Send + Sync>,
    pub check: Arc<dyn Fn(String) -> Result<GenerateResponseData, String> + Send + Sync>,
    pub cancel: Arc<dyn Fn(String) -> Result<(), String> + Send + Sync>,
}

pub fn define_background_model(
    registry: &mut genkit_core::Registry,
    options: DefineBackgroundModelOptions,
) -> BackgroundModelAction {
    let mut supports = options.supports.clone().unwrap_or_default();
    supports.long_running = Some(true);
    let model_info = ModelInfo {
        name: options.name.clone(),
        label: options.label.clone().unwrap_or_default(),
        versions: options.versions.clone(),
        supports: Some(supports),
    };

    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), json!(options.name));
    metadata.insert("type".to_string(), json!("backgroundModel"));
    metadata.insert("metadata".to_string(), json!(model_info));
    if let Some(config_schema) = &options.config_schema {
        metadata.insert("configSchema".to_string(), config_schema.clone());
    }
    let metadata_value = json!(metadata);

    let start_fn = options.start.clone();
    let check_fn = options.check.clone();

    let action_f = move |req: GenerateRequest, _args: genkit_core::action::ActionFnArg<()>| {
        // This pattern separates the `Fn` closure from the `async` block.
        // The outer closure borrows the `Arc`s and clones them.
        // The `async move` block then takes ownership of the clones,
        // satisfying the `Send` bound for the returned `Future`.
        let start_fn = start_fn.clone();
        let check_fn = check_fn.clone();

        async move {
            let op_id = start_fn(req);
            match op_id {
                Ok(id) => {
                    let mut response = GenerateResponseData {
                        candidates: vec![],
                        usage: None,
                        custom: None,
                        operation: Some(id),
                        aggregated: None,
                    };
                    loop {
                        match check_fn(response.operation.clone().unwrap()) {
                            Ok(check_response) => {
                                response = check_response;
                                if response.candidates.iter().any(|c| {
                                    c.finish_reason != Some(FinishReason::Other)
                                        && c.finish_reason.is_some()
                                }) {
                                    break;
                                }
                            }
                            Err(e) => return Err(genkit_core::error::Error::new_internal(e)),
                        }
                        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                    }
                    Ok(response)
                }
                Err(e) => Err(genkit_core::error::Error::new_internal(e)),
            }
        }
    };

    let metadata_map: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_value(metadata_value).unwrap();

    let action = genkit_core::action::ActionBuilder::new(
        genkit_core::registry::ActionType::BackgroundModel,
        options.name.clone(),
        action_f,
    )
    .with_metadata(metadata_map)
    .build();
    let bg_model_action = BackgroundModelAction(action);
    registry
        .register_action(options.name.clone(), bg_model_action.clone())
        .unwrap();
    bg_model_action
}

/// Validates that a model supports a given feature.
pub fn validate_support(name: String, supports: ModelInfoSupports) -> ModelMiddleware {
    middleware::validate_support(name, supports)
}

pub use middleware::SimulateSystemPromptOptions;

pub fn simulate_system_prompt(options: Option<SimulateSystemPromptOptions>) -> ModelMiddleware {
    middleware::simulate_system_prompt(options)
}

pub use middleware::AugmentWithContextOptions;

pub fn augment_with_context(options: Option<AugmentWithContextOptions>) -> ModelMiddleware {
    middleware::augment_with_context(options)
}

// -- Start of refined code --

/// A serializable reference to a model, often used in plugin configurations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef<T> {
    pub name: String,
    #[serde(default)]
    pub info: ModelInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<T>,
    #[serde(skip)]
    _config_type: PhantomData<T>,
}

impl<T> Default for ModelRef<T> {
    fn default() -> Self {
        ModelRef {
            name: String::new(),
            info: ModelInfo::default(),
            version: None,
            config: None,
            _config_type: PhantomData,
        }
    }
}

impl<T> ModelRef<T>
where
    T: for<'de> serde::Deserialize<'de> + Clone,
{
    /// Creates a new ModelRef from a JSON value.
    /// The JSON must contain a "name" field.
    pub fn new(value: serde_json::Value) -> Self {
        serde_json::from_value(value)
            .expect("Failed to create ModelRef from JSON. 'name' field is required.")
    }

    pub fn with_config(mut self, config: T) -> Self {
        self.config = Some(config);
        self
    }

    pub fn with_version(mut self, version: &str) -> Self {
        self.version = Some(version.to_string());
        self
    }
}

/// Represents a reference to a model, which can hold configuration.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Model {
    /// A reference to a model that includes configuration details.
    Reference(ModelRef<serde_json::Value>),
    /// A reference to a model by name only.
    Name(String),
}

impl From<ModelRef<serde_json::Value>> for Model {
    fn from(model_ref: ModelRef<serde_json::Value>) -> Self {
        Model::Reference(model_ref)
    }
}

impl<T> From<ModelRef<T>> for String {
    fn from(m: ModelRef<T>) -> Self {
        m.name
    }
}
