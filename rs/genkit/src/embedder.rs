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

//! # Genkit Embedder
//!
//! This module defines the core data structures and traits for text embedding
//! in the Genkit framework. Embedders convert documents into numerical vectors.
//! This is the Rust equivalent of `@genkit-ai/ai/embedder`.

use crate::error::Result;
use crate::retriever::Document;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a single embedding vector for a piece of content.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct Embedding {
    /// The numerical vector representing the content.
    pub embedding: Vec<f32>,
}

/// A request to an embedder, containing documents to be embedded and options.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct EmbedRequest {
    pub documents: Vec<Document>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<Value>,
}

/// The response from an embedder, containing a batch of embeddings.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct EmbedResponse {
    pub embeddings: Vec<Embedding>,
}

/// Metadata about an embedder.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
pub struct EmbedderInfo {
    pub name: String,
    pub description: String,
}

/// A reference to a configured embedder.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbedderReference<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
}

/// A trait for a component that can generate embeddings for documents.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Returns metadata about the embedder.
    fn info(&self) -> &EmbedderInfo;
    /// Generates embeddings for the given documents.
    async fn embed(&self, req: EmbedRequest) -> Result<EmbedResponse>;
}

/// High-level function to generate embeddings using a registered embedder.
///
/// Note: This is a placeholder for a future implementation that would use a
/// central registry to look up and dispatch to the named embedder.
pub async fn embed(
    embedder: &dyn Embedder,
    documents: Vec<Document>,
    options: Option<Value>,
) -> Result<EmbedResponse> {
    let request = EmbedRequest { documents, options };
    embedder.embed(request).await
}
