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

//! # Document Embedders
//!
//! This module provides the core traits and functions for converting documents
//! into vector embeddings. It is the Rust equivalent of `embedder.ts`.

use crate::document::Document;
use async_trait::async_trait;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::any::Any;
use std::collections::HashMap;
use std::future::Future;
use std::ops::Deref;
use std::sync::Arc;

//
// Core Types & Structs
//

/// Represents a single vector embedding, optionally with metadata.
///
/// This is particularly useful when a single document is chunked, and the
/// metadata can store information about the original chunk.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Embedding {
    /// The vector embedding.
    pub embedding: Vec<f32>,
    /// Optional metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

/// A batch of embeddings, which is the typical output of an embedder.
pub type EmbeddingBatch = Vec<Embedding>;

/// Represents the request sent to an embedder action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EmbedRequest<O = Value> {
    pub input: Vec<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

/// Represents the response received from an embedder action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EmbedResponse {
    pub embeddings: EmbeddingBatch,
}

//
// Action & Function Types
//

/// A function that implements the logic for an embedder.
pub type EmbedderFn<I> =
    dyn Fn(EmbedRequest<I>) -> Box<dyn Future<Output = Result<EmbedResponse>> + Send> + Send + Sync;

/// A wrapper for an embedder `Action`.
#[derive(Clone)]
pub struct EmbedderAction<I = Value>(pub Action<EmbedRequest<I>, EmbedResponse, ()>);

impl<I: 'static> Deref for EmbedderAction<I> {
    type Target = Action<EmbedRequest<I>, EmbedResponse, ()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl<I> ErasedAction for EmbedderAction<I>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
{
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

/// Descriptive information about an embedder.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EmbedderInfo {
    pub label: String,
    // `supports` and `dimensions` would be added here in a more complete port.
}

//
// Define Embedder
//

/// Defines a new embedder and registers it.
pub fn define_embedder<I, F, Fut>(registry: &Registry, name: &str, runner: F) -> EmbedderAction<I>
where
    I: JsonSchema + Serialize + DeserializeOwned + Send + Sync + Clone + 'static,
    F: Fn(EmbedRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<EmbedResponse>> + Send + 'static,
{
    let action = ActionBuilder::new(ActionType::Embedder, name.to_string(), runner).build();
    let embedder_action = EmbedderAction(action);
    registry
        .register_action(name, embedder_action.clone())
        .expect("Failed to register embedder");
    embedder_action
}

//
// High-Level API (`embed`)
//

/// A serializable reference to an embedder, often used in plugin configurations.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EmbedderRef<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// A reference to an embedder.
#[derive(Clone)]
pub enum EmbedderArgument<I = Value> {
    Name(String),
    Action(EmbedderAction<I>),
    Ref(EmbedderRef<I>),
}

/// Parameters for the `embed` function.
pub struct EmbedParams<I = Value> {
    pub embedder: EmbedderArgument<I>,
    pub content: Vec<Document>,
    pub options: Option<I>,
}

/// Generates embeddings for a given list of documents using a specified embedder.
pub async fn embed<I>(registry: &Registry, params: EmbedParams<I>) -> Result<EmbeddingBatch>
where
    I: Default + JsonSchema + DeserializeOwned + Serialize + Send + Sync + Clone + 'static,
{
    let (embedder_action, final_options) = match params.embedder {
        EmbedderArgument::Name(name) => {
            let action = registry
                .lookup_action(&format!("/embedder/{}", name))
                .await
                .ok_or_else(|| Error::new_internal(format!("Embedder '{}' not found", name)))?;
            (action, params.options)
        }
        EmbedderArgument::Action(action) => {
            (Arc::new(action) as Arc<dyn ErasedAction>, params.options)
        }
        EmbedderArgument::Ref(embedder_ref) => {
            let action = registry
                .lookup_action(&format!("/embedder/{}", embedder_ref.name))
                .await
                .ok_or_else(|| {
                    Error::new_internal(format!("Embedder '{}' not found", embedder_ref.name))
                })?;

            let mut merged_opts = serde_json::to_value(embedder_ref.config.unwrap_or_default())
                .map_err(|e| Error::new_internal(e.to_string()))?
                .as_object()
                .ok_or_else(|| Error::new_internal("Invalid config format".to_string()))?
                .clone();

            let params_opts_val = serde_json::to_value(params.options.unwrap_or_default())
                .map_err(|e| Error::new_internal(e.to_string()))?;
            if let Some(params_opts) = params_opts_val.as_object() {
                for (k, v) in params_opts {
                    merged_opts.insert(k.clone(), v.clone());
                }
            }

            if let Some(version) = embedder_ref.version {
                merged_opts.insert("version".to_string(), json!(version));
            }

            let final_opts_typed: I = serde_json::from_value(Value::Object(merged_opts))
                .map_err(|e| Error::new_internal(e.to_string()))?;

            (action, Some(final_opts_typed))
        }
    };

    let request = EmbedRequest {
        input: params.content,
        options: final_options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize embed request: {}", e)))?;

    let response_value = embedder_action.run_http_json(request_value, None).await?;

    let mut response_obj = response_value
        .as_object()
        .ok_or_else(|| Error::new_internal("Expected object response".to_string()))?
        .clone();

    let final_response_value = response_obj
        .remove("result")
        .ok_or_else(|| Error::new_internal("Response missing 'result' field".to_string()))?;

    let response: EmbedResponse = serde_json::from_value(final_response_value)
        .map_err(|e| Error::new_internal(format!("Failed to deserialize embed response: {}", e)))?;

    Ok(response.embeddings)
}

//
// Reference Helpers
//

/// Helper to create an `EmbedderRef`.
pub fn embedder_ref<C>(name: &str) -> EmbedderRef<C> {
    EmbedderRef {
        name: name.to_string(),
        config: None,
        version: None,
    }
}
