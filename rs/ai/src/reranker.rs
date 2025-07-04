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
use async_trait::async_trait;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::any::Any;
use std::future::Future;
use std::ops::Deref;
use std::sync::Arc;

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
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RerankerRequest<O = Value> {
    pub query: Document,
    pub documents: Vec<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

/// Represents the response received from a reranker action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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

/// A wrapper for a reranker `Action`.
#[derive(Clone)]
pub struct RerankerAction<I = Value>(pub Action<RerankerRequest<I>, RerankerResponse, ()>);

impl<I: 'static> Deref for RerankerAction<I> {
    type Target = Action<RerankerRequest<I>, RerankerResponse, ()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl<I> ErasedAction for RerankerAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
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
) -> RerankerAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + 'static,
    F: Fn(RerankerRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<RerankerResponse>> + Send + 'static,
{
    let action = ActionBuilder::new(ActionType::Reranker, name.to_string(), runner).build();
    let reranker_action = RerankerAction(action);
    registry
        .register_action(name, reranker_action.clone())
        .unwrap();
    reranker_action
}

//
// High-Level API (`rerank`)
//

/// A reference to a reranker.
#[derive(Clone)]
pub enum RerankerArgument<I = Value> {
    Name(String),
    Action(RerankerAction<I>),
}

/// Parameters for the `rerank` function.
pub struct RerankerParams<I = Value> {
    pub reranker: RerankerArgument<I>,
    pub query: Document,
    pub documents: Vec<Document>,
    pub options: Option<I>,
}

/// Reranks a list of documents based on a query using a specified reranker.
pub async fn rerank<I>(
    registry: &Registry,
    params: RerankerParams<I>,
) -> Result<Vec<RankedDocument>>
where
    I: JsonSchema + DeserializeOwned + Serialize + Send + Sync + Clone + 'static,
{
    let reranker_action: Arc<dyn ErasedAction> = match params.reranker {
        RerankerArgument::Name(name) => registry
            .lookup_action(&format!("/reranker/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Reranker '{}' not found", name)))?,
        RerankerArgument::Action(action) => Arc::new(action),
    };

    let request = RerankerRequest {
        query: params.query,
        documents: params.documents,
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    let response_value = reranker_action.run_http_json(request_value, None).await?;
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
pub struct RerankerRef<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
}

/// Helper to create a `RerankerRef`.
pub fn reranker_ref<C>(name: &str) -> RerankerRef<C> {
    RerankerRef {
        name: name.to_string(),
        config: None,
    }
}
