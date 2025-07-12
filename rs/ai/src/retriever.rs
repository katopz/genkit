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

//! # Document Retrievers and Indexers
//!
//! This module provides the core traits and functions for retrieving and
//! indexing documents. It is the Rust equivalent of `retriever.ts`.

// Re-export document types for convenience.
pub use crate::document::{Document, Part};

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
use std::pin::Pin;
use std::sync::Arc;

//
// Core Function Signatures & Request/Response Structs
//

/// A function that implements the logic for a retriever.
pub type RetrieverFn<I> = dyn Fn(RetrieverRequest<I>) -> Box<dyn Future<Output = Result<RetrieverResponse>> + Send>
    + Send
    + Sync;

/// A function that implements the logic for an indexer.
pub type IndexerFn<I> =
    dyn Fn(IndexerRequest<I>) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RetrieverRequest<O = Value> {
    pub query: Document,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RetrieverResponse {
    pub documents: Vec<Document>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexerRequest<O = Value> {
    pub documents: Vec<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

//
// Metadata and Action Types
//

/// Descriptive information about a retriever.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RetrieverInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// Descriptive information about an indexer.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct IndexerInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// A wrapper for a retriever `Action`.
#[derive(Clone)]
pub struct RetrieverAction<I = Value>(pub Action<RetrieverRequest<I>, RetrieverResponse, ()>);

impl<I: 'static> Deref for RetrieverAction<I> {
    type Target = Action<RetrieverRequest<I>, RetrieverResponse, ()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl<I> ErasedAction for RetrieverAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + Serialize + 'static,
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
/// A wrapper for an indexer `Action`.
#[derive(Clone)]
pub struct IndexerAction<I = Value>(pub Action<IndexerRequest<I>, (), ()>);

impl<I: 'static> Deref for IndexerAction<I> {
    type Target = Action<IndexerRequest<I>, (), ()>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[async_trait]
impl<I> ErasedAction for IndexerAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + Serialize + 'static,
{
    async fn run_http_json(
        &self,
        input: Value,
        context: Option<genkit_core::context::ActionContext>,
    ) -> Result<Value> {
        self.0.run_http_json(input, context).await?;
        Ok(serde_json::Value::Null)
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

//
// Define Functions
//

/// Defines a new retriever and registers it.
pub fn define_retriever<I, F, Fut>(registry: &Registry, name: &str, runner: F) -> RetrieverAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + Serialize + 'static,
    F: Fn(RetrieverRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<RetrieverResponse>> + Send + 'static,
{
    let action = ActionBuilder::new(ActionType::Retriever, name.to_string(), runner).build();
    let retriever_action = RetrieverAction(action);
    registry
        .register_action(name, retriever_action.clone())
        .unwrap();
    retriever_action
}

/// Defines a new indexer and registers it.
pub fn define_indexer<I, F, Fut>(registry: &Registry, name: &str, runner: F) -> IndexerAction<I>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + Clone + Serialize + 'static,
    F: Fn(IndexerRequest<I>, genkit_core::action::ActionFnArg<()>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let action = ActionBuilder::new(ActionType::Indexer, name.to_string(), runner).build();
    let indexer_action = IndexerAction(action);
    registry
        .register_action(name, indexer_action.clone())
        .unwrap();
    indexer_action
}

//
// High-level API
//

/// A reference to a retriever.
#[derive(Clone)]
pub enum RetrieverArgument<I = Value> {
    Name(String),
    Action(RetrieverAction<I>),
}

/// Parameters for the `retrieve` function.
pub struct RetrieverParams<I = Value> {
    pub retriever: RetrieverArgument<I>,
    pub query: Document,
    pub options: Option<I>,
}

/// Retrieves documents using a specified retriever.
pub async fn retrieve<I>(registry: &Registry, params: RetrieverParams<I>) -> Result<Vec<Document>>
where
    I: JsonSchema + DeserializeOwned + Serialize + Send + Sync + Clone + 'static,
{
    let retriever_action: Arc<dyn ErasedAction> = match params.retriever {
        RetrieverArgument::Name(name) => registry
            .lookup_action(&format!("/retriever/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Retriever '{}' not found", name)))?,
        RetrieverArgument::Action(action) => Arc::new(action),
    };

    let request = RetrieverRequest {
        query: params.query,
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    let mut response_value = retriever_action.run_http_json(request_value, None).await?;
    if let Some(result) = response_value.get_mut("result") {
        response_value = result.take();
    }
    let response: RetrieverResponse = serde_json::from_value(response_value)
        .map_err(|e| Error::new_internal(format!("Failed to deserialize response: {}", e)))?;
    Ok(response.documents)
}

/// A reference to an indexer.
#[derive(Clone)]
pub enum IndexerArgument<I = Value> {
    Name(String),
    Action(IndexerAction<I>),
}

/// Parameters for the `index` function.
pub struct IndexerParams<I = Value> {
    pub indexer: IndexerArgument<I>,
    pub documents: Vec<Document>,
    pub options: Option<I>,
}

/// Indexes documents using a specified indexer.
pub async fn index<I>(registry: &Registry, params: IndexerParams<I>) -> Result<()>
where
    I: JsonSchema + DeserializeOwned + Serialize + Send + Sync + Clone + 'static,
{
    let indexer_action: Arc<dyn ErasedAction> = match params.indexer {
        IndexerArgument::Name(name) => registry
            .lookup_action(&format!("/indexer/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Indexer '{}' not found", name)))?,
        IndexerArgument::Action(action) => Arc::new(action),
    };

    let request = IndexerRequest {
        documents: params.documents,
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    indexer_action.run_http_json(request_value, None).await?;
    Ok(())
}

/// Common retriever options.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CommonRetrieverOptions {
    /// Number of documents to retrieve.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<u32>,
}

//
// Reference Helpers
//

/// A serializable reference to a retriever, often used in plugin configurations.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RetrieverRef<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub info: Option<RetrieverInfo>,
}

/// A serializable reference to an indexer.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct IndexerRef<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub info: Option<IndexerInfo>,
}

/// Helper to create a `RetrieverRef`.
pub fn retriever_ref<C>(name: &str, info: Option<RetrieverInfo>) -> RetrieverRef<C> {
    RetrieverRef {
        name: name.to_string(),
        config: None,
        info,
    }
}

/// Helper to create an `IndexerRef`.
pub fn indexer_ref<C>(name: &str, info: Option<IndexerInfo>) -> IndexerRef<C> {
    IndexerRef {
        name: name.to_string(),
        config: None,
        info,
    }
}
