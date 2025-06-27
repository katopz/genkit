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

use crate::document::{Document, Part};
use crate::message::MessageData;
use crate::tool::ToolDefinition;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::Result;
use genkit_core::registry::{ActionType, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{self, Value};
use std::collections::HashMap;
use std::future::Future;

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
    pub output: Option<super::formats::FormatterConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<Vec<Document>>,
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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseData {
    pub candidates: Vec<CandidateData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<GenerationUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregated: Option<bool>,
}

/// A chunk of a streaming response from a model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponseChunkData {
    pub index: u32,
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

/// A type alias for a model `Action`.
pub type ModelAction = Action<GenerateRequest, GenerateResponseData, GenerateResponseChunkData>;

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

/// Defines a new model and registers it with the framework.
pub fn define_model<C, F, Fut>(
    registry: &mut Registry,
    options: DefineModelOptions<C>,
    runner: F,
) -> ModelAction
where
    C: JsonSchema + Serialize + DeserializeOwned + Send + Sync + 'static,
    F: Fn(GenerateRequest) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<GenerateResponseData>> + Send,
{
    // In a full implementation, middleware would be handled here.
    // For this port, we are directly calling the runner.

    let mut metadata: HashMap<String, Value> = HashMap::new();
    let model_info = ModelInfo {
        label: options.label.clone(),
        versions: options.versions,
        supports: options.supports,
    };
    metadata.insert(
        "model".to_string(),
        serde_json::to_value(model_info).unwrap(),
    );

    let model_action =
        ActionBuilder::new(ActionType::Model, options.name, move |req, _| runner(req))
            .with_description(options.label)
            .with_metadata(metadata)
            .build(registry);

    model_action
}

/// A reference to a model, which can include specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef<C> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// Helper function to create a `ModelRef`.
pub fn model_ref<C>(name: impl Into<String>) -> ModelRef<C> {
    ModelRef {
        name: name.into(),
        config: None,
        version: None,
    }
}
