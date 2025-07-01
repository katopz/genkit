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
use ::serde::de::DeserializeOwned;
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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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

type GenerateRequestPlus = (
    GenerateRequest,
    Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>,
);

/// A function that can be used to chain middleware.
pub type NextFn = Pin<
    Box<
        dyn Fn(
                GenerateRequest,
                Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>,
            )
                -> Pin<Box<dyn Future<Output = Result<GenerateResponseData, Value>> + Send>>
            + Send
            + Sync,
    >,
>;

/// A middleware function for a generative model.
pub type ModelMiddleware = Pin<
    Box<
        dyn Fn(
                GenerateRequestPlus,
                NextFn,
            )
                -> Pin<Box<dyn Future<Output = Result<GenerateResponseData, Value>> + Send>>
            + Send
            + Sync,
    >,
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
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub versions: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supports: Option<ModelInfoSupports>,
    pub config_schema: Option<Value>,
}

fn get_model_middleware(
    f: Pin<
        Box<
            dyn Fn(
                    GenerateRequest,
                    Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>,
                )
                    -> Pin<Box<dyn Future<Output = Result<GenerateResponseData, Value>> + Send>>
                + Send
                + Sync,
        >,
    >,
) -> NextFn {
    Box::pin(move |req, chunk_cb| f(req, chunk_cb))
}

/// Defines a new generative model.
pub fn define_model<F, Fut>(options: DefineModelOptions, f: F) -> ModelAction
where
    F: Fn(GenerateRequest, Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>>) -> Fut
        + Send
        + Sync
        + 'static,
    Fut: Future<Output = Result<GenerateResponseData, Value>> + Send + 'static,
{
    let f = std::sync::Arc::new(f);
    let model_info = ModelInfo {
        name: options.name.clone(),
        label: options.label,
        versions: options.versions,
        supports: options.supports,
    };

    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), json!(options.name));
    metadata.insert("type".to_string(), json!("model"));
    metadata.insert("metadata".to_string(), json!(model_info));
    if let Some(config_schema) = options.config_schema {
        metadata.insert("configSchema".to_string(), config_schema);
    }
    let metadata_value = json!(metadata);
    let metadata_map: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_value(metadata_value).unwrap();

    let action_f =
        move |req: GenerateRequest,
              args: genkit_core::action::ActionFnArg<GenerateResponseChunkData>| {
            let f = f.clone();
            async move {
                let callback: Option<Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>> =
                    if args.streaming_requested {
                        Some(Box::new(move |chunk| args.chunk_sender.send(chunk)))
                    } else {
                        None
                    };
                f(req, callback)
                    .await
                    .map_err(|e| genkit_core::error::Error::new_internal(e.to_string()))
            }
        };

    let action = genkit_core::action::ActionBuilder::new(
        genkit_core::registry::ActionType::Model,
        options.name,
        action_f,
    )
    .with_metadata(metadata_map)
    .build();
    ModelAction(action)
}

pub struct DefineBackgroundModelOptions {
    pub name: String,
    pub label: String,
    pub versions: Option<Vec<String>>,
    pub supports: Option<ModelInfoSupports>,
    pub config_schema: Option<Value>,
    pub start: Arc<dyn Fn(GenerateRequest) -> Result<String, String> + Send + Sync>,
    pub check: Arc<dyn Fn(String) -> Result<GenerateResponseData, String> + Send + Sync>,
    pub cancel: Arc<dyn Fn(String) -> Result<(), String> + Send + Sync>,
}

pub fn define_background_model(options: DefineBackgroundModelOptions) -> BackgroundModelAction {
    let mut supports = options.supports.unwrap_or_default();
    supports.long_running = Some(true);
    let model_info = ModelInfo {
        name: options.name.clone(),
        label: options.label,
        versions: options.versions,
        supports: Some(supports),
    };

    let mut metadata = HashMap::new();
    metadata.insert("name".to_string(), json!(options.name));
    metadata.insert("type".to_string(), json!("backgroundModel"));
    metadata.insert("metadata".to_string(), json!(model_info));
    if let Some(config_schema) = options.config_schema {
        metadata.insert("configSchema".to_string(), config_schema);
    }
    let metadata_value = json!(metadata);

    let start_fn = options.start.clone();
    let check_fn = options.check.clone();

    let action_f = |req: &GenerateRequest, _args: &genkit_core::action::ActionFnArg<()>| {
        // This pattern separates the `Fn` closure from the `async` block.
        // The outer closure borrows the `Arc`s and clones them.
        // The `async move` block then takes ownership of the clones,
        // satisfying the `Send` bound for the returned `Future`.
        let start_fn = start_fn.clone();
        let check_fn = check_fn.clone();
        let req = req.clone();

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

    todo!()
    // let action = genkit_core::action::ActionBuilder::new(
    //     genkit_core::registry::ActionType::Model,
    //     options.name,
    //     action_f,
    // )
    // .with_metadata(metadata_map)
    // .build();
    // BackgroundModelAction(action)
}

/// Validates that a model supports a given feature.
pub fn validate_support(
    model_name: &str,
    _model_supports: &ModelInfoSupports,
    feature_name: &str,
    feature_supported: bool,
) {
    if !feature_supported {
        panic!("Model {} does not support {}.", model_name, feature_name);
    }
}

/// Options for simulating a system prompt.
pub struct SimulateSystemPromptOptions {
    pub preface: Option<String>,
    pub acknowledgement: Option<String>,
}

/// Simulates a system prompt for models that do not support it natively.
pub fn simulate_system_prompt(
    messages: &mut Vec<MessageData>,
    options: Option<SimulateSystemPromptOptions>,
) {
    let options = options.unwrap_or(SimulateSystemPromptOptions {
        preface: None,
        acknowledgement: None,
    });
    let preface = options.preface.unwrap_or_else(|| {
        "The user wants you to act as the following persona. Do not talk about this persona, just embody it. Do not mention that you are an AI model.".to_string()
    });
    let acknowledgement = options
        .acknowledgement
        .unwrap_or_else(|| "Understood.".to_string());

    let mut system_message: Option<MessageData> = None;
    messages.retain(|msg| {
        if msg.role == Role::System {
            system_message = Some(msg.clone());
            false
        } else {
            true
        }
    });

    if let Some(sm) = system_message {
        let new_user_content = format!(
            "{}\n\n{}\n\nBegin.",
            preface,
            sm.content
                .iter()
                .map(|p| p.text.clone().unwrap_or_default())
                .collect::<Vec<_>>()
                .join("")
        );

        messages.insert(
            0,
            MessageData {
                role: Role::User,
                content: vec![Part {
                    text: Some(new_user_content),
                    ..Default::default()
                }],
                metadata: None,
            },
        );

        messages.insert(
            1,
            MessageData {
                role: Role::Model,
                content: vec![Part {
                    text: Some(acknowledgement),
                    ..Default::default()
                }],
                metadata: None,
            },
        );
    }
}

/// Options for augmenting a request with context.
pub struct AugmentWithContextOptions {
    pub preface: Option<String>,
    pub citation_key: Option<String>,
}

pub const CONTEXT_PREFACE: &str = "The user has provided the following documents to provide context for the prompt. This is not part of the prompt, just supporting information.";
fn default_item_template(doc: &Part, citation: usize, key: &str) -> String {
    let mut parts = Vec::new();
    if let Some(metadata) = &doc.metadata {
        if let Some(title) = metadata.get("title") {
            parts.push(format!("title: {}", title.as_str().unwrap_or("")));
        }
        if let Some(url) = metadata.get("url") {
            parts.push(format!("url: {}", url.as_str().unwrap_or("")));
        }
    }
    if let Some(text) = &doc.text {
        parts.push(format!("content: {}", text));
    }
    format!(
        "Document(/{}/{})\n---\n{}\n---",
        key,
        citation,
        parts.join("\n")
    )
}

/// Augments a request with context from a list of documents.
pub fn augment_with_context(
    messages: &mut Vec<MessageData>,
    docs: &[Part],
    options: AugmentWithContextOptions,
) {
    if docs.is_empty() {
        return;
    }
    let preface = options
        .preface
        .unwrap_or_else(|| CONTEXT_PREFACE.to_string());
    let citation_key = options
        .citation_key
        .unwrap_or_else(|| "document".to_string());

    let mut doc_strs = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        doc_strs.push(default_item_template(doc, i + 1, &citation_key));
    }

    let context_str = format!("{}\n\n{}\n\n", preface, doc_strs.join("\n\n"));
    if let Some(last_message) = messages.last_mut() {
        if last_message.role == Role::User {
            if let Some(last_part) = last_message.content.last_mut() {
                if let Some(text) = &mut last_part.text {
                    *text = format!("{}\n\n{}", context_str, text);
                    return;
                }
            }
        }
    }
    messages.push(MessageData {
        role: Role::User,
        content: vec![Part {
            text: Some(context_str),
            ..Default::default()
        }],
        metadata: None,
    });
}

/// A serializable reference to a model, often used in plugin configurations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef<T> {
    pub name: String,
    pub info: ModelInfo,
    pub config: PhantomData<T>,
}

/// Represents a reference to a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Model {
    Reference(String),
    Name(String),
}

impl From<ModelRef<serde_json::Value>> for Model {
    fn from(model_ref: ModelRef<serde_json::Value>) -> Self {
        Model::Reference(model_ref.name)
    }
}

impl<T> From<ModelRef<T>> for String {
    fn from(m: ModelRef<T>) -> Self {
        m.name
    }
}

/// Helper function to create a `ModelRef`.
pub fn model_ref<T: DeserializeOwned>(info: ModelInfo) -> ModelRef<T> {
    ModelRef {
        name: info.name.clone(),
        info,
        config: PhantomData,
    }
}

impl Default for FinishReason {
    fn default() -> Self {
        FinishReason::Unknown
    }
}
