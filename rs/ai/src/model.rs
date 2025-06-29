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
//! models. It is the Rust equivalent of `model.ts` and `model-types.ts`.

pub mod middleware;
use self::middleware::{download_request_media, simulate_constrained_generation};
use crate::document::{Document, Part};
use crate::generate::{OutputOptions, ToolChoice};
use crate::message::{MessageData, Role};

use crate::tool::ToolDefinition;
use async_trait::async_trait;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::background_action::Operation;
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, Value};
use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

//
// SECTION: Model Request & Response Data Structures
// (Ported from model-types.ts)
//

/// Describes the capabilities of a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfoSupports {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multiturn: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_role: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_running: Option<bool>,
}

/// Provides descriptive information about a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelInfo {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub versions: Option<Vec<String>>,
    #[serde(default)]
    pub supports: ModelInfoSupports,
}

/// Common configuration options for generative models.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerationCommonConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
}

/// The request sent to a model action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerateRequest {
    pub messages: Vec<MessageData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>, // This will hold the model-specific config
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<Vec<Document>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_tool_requests: Option<bool>,
}

/// The reason a model stopped generating tokens.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    Stop,
    Length,
    Blocked,
    Other,
    Unknown,
    Interrupted,
}

/// Usage statistics for a generation request.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GenerationUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u32>,
}

/// A single candidate response from a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CandidateData {
    pub index: u32,
    pub message: MessageData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_message: Option<String>,
}

/// The full response from a model action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseData {
    pub candidates: Vec<CandidateData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<Value>,
    /// The operation for a long-running job, if applicable.
    /// Boxed to prevent recursive type error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub operation: Option<Box<Operation<Self>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregated: Option<bool>,
}

/// A chunk of a streaming response from a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseChunkData {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    pub content: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<Value>,
}

//
// SECTION: Model Definition
// (Ported from model.ts)
//

/// A function that represents the next step in a middleware chain.
pub type NextFn = Box<
    dyn FnOnce(
            GenerateRequest,
        ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
        + Send,
>;

/// A middleware for model actions.
pub type ModelMiddleware = Arc<
    dyn Fn(
            GenerateRequest,
            NextFn,
        ) -> Pin<Box<dyn Future<Output = Result<GenerateResponseData>> + Send>>
        + Send
        + Sync,
>;

/// A wrapper for a model `Action`.
#[derive(Clone)]
pub struct ModelAction(
    pub Action<GenerateRequest, GenerateResponseData, GenerateResponseChunkData>,
);

impl Deref for ModelAction {
    type Target = Action<GenerateRequest, GenerateResponseData, GenerateResponseChunkData>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl ErasedAction for ModelAction {
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<Value> {
        self.0.run_http_json(input, context).await
    }

    fn stream_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<genkit_core::action::StreamingResponse<Value, Value>> {
        self.0.stream_http_json(input, context)
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn metadata(&self) -> &genkit_core::action::ActionMetadata {
        self.0.metadata()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A wrapper for a background model `Action`.
#[derive(Clone)]
pub struct BackgroundModelAction(pub Action<GenerateRequest, Operation<GenerateResponseData>, ()>);

impl Deref for BackgroundModelAction {
    type Target = Action<GenerateRequest, Operation<GenerateResponseData>, ()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl ErasedAction for BackgroundModelAction {
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<Value> {
        self.0.run_http_json(input, context).await
    }

    fn stream_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<genkit_core::action::StreamingResponse<Value, Value>> {
        self.0.stream_http_json(input, context)
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    fn metadata(&self) -> &genkit_core::action::ActionMetadata {
        self.0.metadata()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Options for defining a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
pub struct DefineModelOptions<C: JsonSchema + 'static> {
    pub name: String,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub versions: Option<Vec<String>>,
    #[serde(default)]
    pub supports: ModelInfoSupports,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_schema: Option<C>,
}

fn get_model_middleware<C: JsonSchema + Send + Sync + 'static>(
    options: &DefineModelOptions<C>,
) -> Vec<ModelMiddleware> {
    let mut middleware: Vec<ModelMiddleware> = Vec::new();

    middleware.push(download_request_media(None, None));
    middleware.push(validate_support(
        options.name.clone(),
        options.supports.clone(),
    ));

    if options.supports.context != Some(true) {
        middleware.push(augment_with_context(None));
    }

    middleware.push(simulate_constrained_generation(None));

    middleware
}

/// Defines a new model and registers it with the framework.
pub fn define_model<C, F, Fut>(options: DefineModelOptions<C>, runner: F) -> ModelAction
where
    C: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(GenerateRequest, genkit_core::action::ActionFnArg<GenerateResponseChunkData>) -> Fut
        + Send
        + Sync
        + 'static,
    Fut: Future<Output = Result<GenerateResponseData>> + Send,
{
    let middleware = get_model_middleware(&options);
    let runner_arc = Arc::new(runner);

    let wrapped_runner =
        move |req: GenerateRequest,
              args: genkit_core::action::ActionFnArg<GenerateResponseChunkData>| {
            let runner_clone = runner_arc.clone();
            let middleware_clone = middleware.clone();

            async move {
                let final_step: NextFn = Box::new(move |final_req| {
                    // The middleware chain doesn't pass the ActionFnArg, so we have to capture it from the outer scope.
                    let copied_args = genkit_core::action::ActionFnArg {
                        streaming_requested: args.streaming_requested,
                        chunk_sender: args.chunk_sender.clone(),
                        context: args.context.clone(),
                        trace: args.trace.clone(),
                        abort_signal: args.abort_signal.clone(),
                    };
                    Box::pin(async move { (runner_clone)(final_req, copied_args).await })
                });

                let mut chain = final_step;
                for mw in middleware_clone.into_iter().rev() {
                    let next_fn = chain;
                    chain = Box::new(move |req_inner| Box::pin(mw(req_inner, next_fn)));
                }

                chain(req).await
            }
        };

    let mut metadata: HashMap<String, Value> = HashMap::new();
    let model_info = ModelInfo {
        label: options.label.clone(),
        versions: options.versions,
        supports: options.supports.clone(),
    };
    metadata.insert(
        "model".to_string(),
        serde_json::to_value(model_info).unwrap(),
    );

    ModelAction(
        ActionBuilder::new(ActionType::Model, options.name, wrapped_runner)
            .with_description(options.label)
            .with_metadata(metadata)
            .build(),
    )
}

pub struct DefineBackgroundModelOptions<C, StartFn, CheckFn, CancelFn> {
    pub name: String,
    pub label: String,
    pub versions: Option<Vec<String>>,
    pub supports: ModelInfoSupports,
    pub config_schema: Option<C>,
    pub start: StartFn,
    pub check: CheckFn,
    pub cancel: Option<CancelFn>,
}

/// Defines a new background model and registers it with the framework.
pub fn define_background_model<C, StartFn, CheckFn, CancelFn, StartFut, CheckFut, CancelFut>(
    options: DefineBackgroundModelOptions<C, StartFn, CheckFn, CancelFn>,
) -> Arc<BackgroundModelAction>
where
    C: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    StartFn: Fn(GenerateRequest) -> StartFut + Send + Sync + 'static,
    StartFut: Future<Output = Result<Operation<GenerateResponseData>>> + Send,
    CheckFn: Fn(Operation<GenerateResponseData>) -> CheckFut + Send + Sync + 'static,
    CheckFut: Future<Output = Result<Operation<GenerateResponseData>>> + Send,
    CancelFn: Fn(Operation<GenerateResponseData>) -> CancelFut + Send + Sync + 'static,
    CancelFut: Future<Output = Result<Operation<GenerateResponseData>>> + Send,
{
    let mut metadata: HashMap<String, Value> = HashMap::new();
    let mut model_info = ModelInfo {
        label: options.label.clone(),
        versions: options.versions,
        supports: options.supports,
    };
    model_info.supports.long_running = Some(true);
    metadata.insert(
        "model".to_string(),
        serde_json::to_value(model_info).unwrap(),
    );

    let start_action = ActionBuilder::new(
        ActionType::BackgroundModel,
        options.name.clone(),
        move |req, _: genkit_core::action::ActionFnArg<()>| (options.start)(req),
    )
    .with_description(options.label.clone())
    .with_metadata(metadata.clone())
    .build();

    let check_action_name = format!("{}/check", options.name);
    let _check_action = ActionBuilder::new(
        ActionType::CheckOperation,
        check_action_name,
        move |op, _: genkit_core::action::ActionFnArg<()>| (options.check)(op),
    )
    .with_description(format!("Check status for {}", options.label))
    .with_metadata(metadata.clone())
    .build();

    if let Some(cancel_fn) = options.cancel {
        let cancel_action_name = format!("{}/cancel", options.name);
        let _cancel_action = ActionBuilder::new(
            ActionType::CancelOperation,
            cancel_action_name,
            move |op, _: genkit_core::action::ActionFnArg<()>| cancel_fn(op),
        )
        .with_description(format!("Cancel operation for {}", options.label))
        .with_metadata(metadata)
        .build();
    }

    Arc::new(BackgroundModelAction(start_action))
}

/// Validates that a `GenerateRequest` does not include unsupported features.
pub fn validate_support(name: String, supports: ModelInfoSupports) -> ModelMiddleware {
    Arc::new(move |req, next| {
        let name = name.clone();
        let supports = supports.clone();
        Box::pin(async move {
            let invalid = |message: &str| -> Error {
                Error::new_internal(format!(
                    "Model '{}' does not support {}. Request: {}",
                    name,
                    message,
                    serde_json::to_string_pretty(&req)
                        .unwrap_or_else(|_| "Unserializable request".to_string())
                ))
            };

            if supports.media == Some(false)
                && req
                    .messages
                    .iter()
                    .any(|m| m.content.iter().any(|p| p.media.is_some()))
            {
                return Err(invalid("media, but media was provided"));
            }

            if supports.tools == Some(false) && req.tools.as_ref().is_some_and(|t| !t.is_empty()) {
                return Err(invalid("tool use, but tools were provided"));
            }

            if supports.multiturn == Some(false) && req.messages.len() > 1 {
                let len = req.messages.len();
                return Err(invalid(&format!(
                    "multiple messages, but {} were provided",
                    len
                )));
            }

            next(req).await
        })
    })
}

#[derive(Default, Clone)]
pub struct SimulateSystemPromptOptions {
    pub preface: Option<String>,
    pub acknowledgement: Option<String>,
}

/// Provide a simulated system prompt for models that don't support it natively.
pub fn simulate_system_prompt(options: Option<SimulateSystemPromptOptions>) -> ModelMiddleware {
    let opts = options.unwrap_or_default();
    let preface = opts
        .preface
        .unwrap_or_else(|| "SYSTEM INSTRUCTIONS:\n".to_string());
    let acknowledgement = opts
        .acknowledgement
        .unwrap_or_else(|| "Understood.".to_string());

    Arc::new(move |mut req, next| {
        let preface = preface.clone();
        let acknowledgement = acknowledgement.clone();
        Box::pin(async move {
            if let Some(pos) = req.messages.iter().position(|m| m.role == Role::System) {
                let system_message = req.messages.remove(pos);
                let mut user_content = vec![Part {
                    text: Some(preface),
                    ..Default::default()
                }];
                user_content.extend(system_message.content);

                let user_message = MessageData {
                    role: Role::User,
                    content: user_content,
                    metadata: None,
                };
                let model_message = MessageData {
                    role: Role::Model,
                    content: vec![Part {
                        text: Some(acknowledgement),
                        ..Default::default()
                    }],
                    metadata: None,
                };

                req.messages
                    .splice(pos..pos, vec![user_message, model_message]);
            }

            next(req).await
        })
    })
}

#[derive(Default, Clone)]
pub struct AugmentWithContextOptions {
    pub preface: Option<String>,
    pub citation_key: Option<String>,
}

pub const CONTEXT_PREFACE: &str = "\n\nUse the following information to complete your task:\n\n";

fn default_item_template(
    d: &Document,
    index: usize,
    options: &AugmentWithContextOptions,
) -> String {
    let mut out = "- ".to_string();
    let citation = if let Some(key) = &options.citation_key {
        d.metadata
            .as_ref()
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        d.metadata.as_ref().and_then(|m| {
            m.get("ref")
                .or_else(|| m.get("id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
    }
    .unwrap_or_else(|| index.to_string());

    out.push_str(&format!("[{}]: ", citation));
    out.push_str(&d.text());
    out.push('\n');
    out
}

/// Injects retrieved documents as context into the last user message.
pub fn augment_with_context(options: Option<AugmentWithContextOptions>) -> ModelMiddleware {
    let opts = options.unwrap_or_default();
    Arc::new(move |mut req, next| {
        let opts = opts.clone();
        Box::pin(async move {
            let docs = match req.docs.as_ref() {
                Some(d) if !d.is_empty() => d,
                _ => return next(req).await,
            };

            if let Some(user_message) = req.messages.iter_mut().rfind(|m| m.role == Role::User) {
                // Check if context part already exists and is not pending
                let context_part_index = user_message.content.iter().position(|p| {
                    p.metadata
                        .as_ref()
                        .map_or_else(|| false, |m| m.get("purpose") == Some(&"context".into()))
                });

                if let Some(idx) = context_part_index {
                    let is_pending = user_message.content[idx]
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("pending"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if !is_pending {
                        return next(req).await;
                    }
                }

                let preface = opts
                    .preface
                    .clone()
                    .unwrap_or_else(|| CONTEXT_PREFACE.to_string());
                let mut context_text = preface;
                for (i, doc_data) in docs.iter().enumerate() {
                    context_text.push_str(&default_item_template(doc_data, i, &opts));
                }
                context_text.push('\n');

                let mut context_metadata = HashMap::new();
                context_metadata
                    .insert("purpose".to_string(), Value::String("context".to_string()));
                let new_part = Part {
                    text: Some(context_text),
                    metadata: Some(context_metadata),
                    ..Default::default()
                };

                if let Some(idx) = context_part_index {
                    // Replace pending part
                    user_message.content[idx] = new_part;
                } else {
                    // Append new part
                    user_message.content.push(new_part);
                }
            }

            next(req).await
        })
    })
}

/// A serializable reference to a model, often used in plugin configurations.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModelRef<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Represents a reference to a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum Model {
    Reference(ModelRef<Value>),
    Name(String),
}

/// Helper function to create a `ModelRef`.
pub fn model_ref<C>(name: &str) -> ModelRef<C> {
    ModelRef {
        name: name.to_string(),
        config: None,
        version: None,
    }
}
