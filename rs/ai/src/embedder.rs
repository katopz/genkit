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
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

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
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmbedRequest<O = Value> {
    pub input: Vec<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

/// Represents the response received from an embedder action.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct EmbedResponse {
    pub embeddings: EmbeddingBatch,
}

//
// Action & Function Types
//

/// A function that implements the logic for an embedder.
pub type EmbedderFn<I> =
    dyn Fn(EmbedRequest<I>) -> Box<dyn Future<Output = Result<EmbedResponse>> + Send> + Send + Sync;

/// A type alias for an embedder `Action`.
pub type EmbedderAction<I = Value> = Action<EmbedRequest<I>, EmbedResponse, ()>;

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
pub fn define_embedder<I, F, Fut>(
    registry: &mut Registry,
    name: &str,
    runner: F,
) -> Arc<EmbedderAction<I>>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + 'static,
    F: FnMut(EmbedRequest<I>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<EmbedResponse>> + Send + 'static,
{
    let runner = Arc::new(Mutex::new(runner));
    ActionBuilder::new(
        ActionType::Embedder,
        name.to_string(),
        move |req: EmbedRequest<I>, _context| {
            let runner_clone = runner.clone(); // Arc<Mutex<F>>

            // Synchronously acquire the lock, call the inner runner, and release the lock.
            // The `inner_fut` is a `Box<dyn Future + Send>`, which is safe to move.
            let inner_fut = {
                let mut runner_guard = runner_clone.lock().unwrap();
                (runner_guard)(req)
            }; // `runner_guard` is dropped here, releasing the mutex.

            // This async block now only awaits the `inner_fut`, which is `Send`.
            inner_fut
        },
    )
    .build(registry)
    .into()
}

//
// High-Level API (`embed`)
//

/// A reference to an embedder.
#[derive(Clone)]
pub enum EmbedderArgument {
    Name(String),
    Action(Arc<EmbedderAction>),
}

/// Parameters for the `embed` function.
pub struct EmbedParams {
    pub embedder: EmbedderArgument,
    pub content: Vec<Document>,
    pub options: Option<Value>,
}

/// Generates embeddings for a given list of documents using a specified embedder.
pub async fn embed(registry: &Registry, params: EmbedParams) -> Result<EmbeddingBatch> {
    let embedder_action = match params.embedder {
        EmbedderArgument::Name(name) => registry
            .lookup_action(&format!("/embedder/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Embedder '{}' not found", name)))?,
        EmbedderArgument::Action(action) => action,
    };

    let request = EmbedRequest {
        input: params.content,
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize embed request: {}", e)))?;
    let response_value = embedder_action.run_http(request_value).await?;
    let response: EmbedResponse = serde_json::from_value(response_value)
        .map_err(|e| Error::new_internal(format!("Failed to deserialize embed response: {}", e)))?;
    Ok(response.embeddings)
}

//
// Reference Helpers
//

/// A serializable reference to an embedder, often used in plugin configurations.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct EmbedderRef {
    pub name: String,
    // config and info would be here in a full port
}

/// Helper to create an `EmbedderRef`.
pub fn embedder_ref(name: &str) -> EmbedderRef {
    EmbedderRef {
        name: name.to_string(),
    }
}
