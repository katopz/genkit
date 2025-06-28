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

//! # Document Rerankers
//!
//! This module provides the core traits and functions for re-ranking documents
//! based on a query, a common step in Retrieval-Augmented Generation (RAG)
//! pipelines. It is the Rust equivalent of `reranker.ts`.

use crate::document::{Document, Part};
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::sync::{Arc, Mutex};

//
// Core Types & Structs
//

/// Represents the metadata for a reranked document, which must include a score.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RankedDocumentMetadata {
    /// The relevance score assigned by the reranker.
    pub score: f64,
    /// Any other original metadata from the document is preserved here.
    #[serde(flatten)]
    pub other_metadata: Option<Value>,
}

/// Represents a reranked document, combining the original document's content
/// with new metadata that includes a relevance score.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RankedDocument {
    pub content: Vec<Part>,
    pub metadata: RankedDocumentMetadata,
}

impl RankedDocument {
    /// Returns the relevance score of the document.
    pub fn score(&self) -> f64 {
        self.metadata.score
    }
}

/// Represents the request sent to a reranker action.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RerankerRequest<O = Value> {
    pub query: Document,
    pub documents: Vec<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

/// Represents the response received from a reranker action.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RerankerResponse {
    pub documents: Vec<RankedDocument>,
}

//
// Action & Function Types
//

/// A function that implements the logic for a reranker.
pub type RerankerFn<I> = dyn Fn(RerankerRequest<I>) -> Box<dyn Future<Output = Result<RerankerResponse>> + Send>
    + Send
    + Sync;

/// A type alias for a reranker `Action`.
pub type RerankerAction<I = Value> = Action<RerankerRequest<I>, RerankerResponse, ()>;

/// Descriptive information about a reranker.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RerankerInfo {
    pub label: String,
}

//
// Define Reranker
//

/// Defines a new reranker and registers it.
pub fn define_reranker<I, F, Fut>(
    registry: &mut Registry,
    name: &str,
    runner: F,
) -> Arc<RerankerAction<I>>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + 'static,
    F: FnMut(RerankerRequest<I>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<RerankerResponse>> + Send + 'static,
{
    let runner_arc = Arc::new(Mutex::new(runner));
    ActionBuilder::new(
        ActionType::Reranker,
        name.to_string(),
        move |req: RerankerRequest<I>, _context| {
            let runner_clone = runner_arc.clone();
            {
                let mut runner = runner_clone.lock().unwrap();
                runner(req)
            }
        },
    )
    .build(registry)
    .into()
}

//
// High-Level API (`rerank`)
//

/// A reference to a reranker.
#[derive(Clone)]
pub enum RerankerArgument {
    Name(String),
    Action(Arc<RerankerAction>),
}

/// Parameters for the `rerank` function.
pub struct RerankerParams {
    pub reranker: RerankerArgument,
    pub query: Document,
    pub documents: Vec<Document>,
    pub options: Option<Value>,
}

/// Reranks a list of documents based on a query using a specified reranker.
pub async fn rerank(registry: &Registry, params: RerankerParams) -> Result<Vec<RankedDocument>> {
    let reranker_action = match params.reranker {
        RerankerArgument::Name(name) => registry
            .lookup_action(&format!("/reranker/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Reranker '{}' not found", name)))?,
        RerankerArgument::Action(action) => action,
    };

    let request = RerankerRequest {
        query: params.query,
        documents: params.documents,
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    let response_value = reranker_action.run_http(request_value).await?;
    let response: RerankerResponse = serde_json::from_value(response_value)
        .map_err(|e| Error::new_internal(format!("Failed to deserialize response: {}", e)))?;
    Ok(response.documents)
}

//
// Reference Helpers
//

/// A serializable reference to a reranker, often used in plugin configurations.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RerankerRef {
    pub name: String,
    // config and info would be here in a full port
}

/// Helper to create a `RerankerRef`.
pub fn reranker_ref(name: &str) -> RerankerRef {
    RerankerRef {
        name: name.to_string(),
    }
}
