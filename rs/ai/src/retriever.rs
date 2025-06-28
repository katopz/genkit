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

use genkit_core::action::{Action, ActionBuilder};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

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

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RetrieverRequest<O = Value> {
    pub query: Document,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<O>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct RetrieverResponse {
    pub documents: Vec<Document>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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
    pub label: String,
}

/// Descriptive information about an indexer.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct IndexerInfo {
    pub label: String,
}

/// A type alias for a retriever `Action`.
pub type RetrieverAction<I = Value> = Action<RetrieverRequest<I>, RetrieverResponse, ()>;

/// A type alias for an indexer `Action`.
pub type IndexerAction<I = Value> = Action<IndexerRequest<I>, (), ()>;

//
// Define Functions
//

/// Defines a new retriever and registers it.
pub fn define_retriever<I, F, Fut>(
    registry: &mut Registry,
    name: &str,
    runner: F,
) -> Arc<RetrieverAction<I>>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + 'static,
    F: FnMut(RetrieverRequest<I>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<RetrieverResponse>> + Send + 'static,
{
    let runner_arc = Arc::new(Mutex::new(runner));
    ActionBuilder::new(
        ActionType::Retriever,
        name.to_string(),
        move |req: RetrieverRequest<I>, _context| {
            let runner_clone = runner_arc.clone();
            async move {
                let fut = {
                    let mut runner = runner_clone.lock().unwrap();
                    runner(req)
                };
                fut.await
            }
        },
    )
    .build(registry)
    .into()
}

/// Defines a new indexer and registers it.
pub fn define_indexer<I, F, Fut>(
    registry: &mut Registry,
    name: &str,
    runner: F,
) -> Arc<IndexerAction<I>>
where
    I: JsonSchema + DeserializeOwned + Send + Sync + 'static,
    F: FnMut(IndexerRequest<I>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let runner_arc = Arc::new(Mutex::new(runner));
    ActionBuilder::new(
        ActionType::Indexer,
        name.to_string(),
        move |req: IndexerRequest<I>, _context| {
            let runner_clone = runner_arc.clone();
            async move {
                let future = {
                    let mut runner = runner_clone.lock().unwrap();
                    runner(req)
                };
                future.await
            }
        },
    )
    .build(registry)
    .into()
}

//
// High-level API
//

/// A reference to a retriever.
#[derive(Clone)]
pub enum RetrieverArgument {
    Name(String),
    Action(Arc<RetrieverAction>),
}

/// Parameters for the `retrieve` function.
pub struct RetrieverParams {
    pub retriever: RetrieverArgument,
    pub query: Document,
    pub options: Option<Value>,
}

/// Retrieves documents using a specified retriever.
pub async fn retrieve(registry: &Registry, params: RetrieverParams) -> Result<Vec<Document>> {
    let retriever_action = match params.retriever {
        RetrieverArgument::Name(name) => registry
            .lookup_action(&format!("/retriever/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Retriever '{}' not found", name)))?,
        RetrieverArgument::Action(action) => action,
    };

    let request = RetrieverRequest {
        query: params.query,
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    let response_value = retriever_action.run_http(request_value).await?;
    let response: RetrieverResponse = serde_json::from_value(response_value)
        .map_err(|e| Error::new_internal(format!("Failed to deserialize response: {}", e)))?;
    Ok(response.documents)
}

/// A reference to an indexer.
#[derive(Clone)]
pub enum IndexerArgument {
    Name(String),
    Action(Arc<IndexerAction>),
}

/// Parameters for the `index` function.
pub struct IndexerParams {
    pub indexer: IndexerArgument,
    pub documents: Vec<Document>,
    pub options: Option<Value>,
}

/// Indexes documents using a specified indexer.
pub async fn index(registry: &Registry, params: IndexerParams) -> Result<()> {
    let indexer_action = match params.indexer {
        IndexerArgument::Name(name) => registry
            .lookup_action(&format!("/indexer/{}", name))
            .await
            .ok_or_else(|| Error::new_internal(format!("Indexer '{}' not found", name)))?,
        IndexerArgument::Action(action) => action,
    };

    let request = IndexerRequest {
        documents: params.documents,
        options: params.options,
    };

    let request_value = serde_json::to_value(request)
        .map_err(|e| Error::new_internal(format!("Failed to serialize request: {}", e)))?;
    indexer_action.run_http(request_value).await?;
    Ok(())
}

//
// Reference Helpers
//

/// A serializable reference to a retriever, often used in plugin configurations.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RetrieverRef {
    pub name: String,
    // config and info would be here in a full port
}

/// A serializable reference to an indexer.
#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
#[serde(rename_all = "camelCase")]
pub struct IndexerRef {
    pub name: String,
    // config and info would be here in a full port
}

/// Helper to create a `RetrieverRef`.
pub fn retriever_ref(name: &str) -> RetrieverRef {
    RetrieverRef {
        name: name.to_string(),
    }
}

/// Helper to create an `IndexerRef`.
pub fn indexer_ref(name: &str) -> IndexerRef {
    IndexerRef {
        name: name.to_string(),
    }
}
