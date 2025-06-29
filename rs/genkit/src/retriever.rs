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

//! # Genkit Retriever
//!
//! This module defines the core data structures and traits for document
//! retrieval and indexing in the Genkit framework. Retrievers are used to
// fetch relevant context to augment prompts, while indexers are used to
// populate the data stores that retrievers search.
//!
//! This is the Rust equivalent of `@genkit-ai/ai/retriever`.

use crate::error::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents a simple text part of a document's content.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
pub struct TextPart {
    pub text: String,
}

/// Represents a multimedia part of a document's content.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct MediaPart {
    /// The IANA media type (MIME type) of the content.
    pub content_type: String,
    /// The URI of the media content, which can be a `data:` or `https:` URI.
    pub url: String,
}

/// Represents a single piece of content within a `Document`.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
#[serde(untagged)]
pub enum Part {
    Text(TextPart),
    Media(MediaPart),
}

/// Represents a piece of content that can be indexed and retrieved.
///
/// This is a central data structure used by retrievers and indexers.
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Default)]
pub struct Document {
    /// The content of the document, which can be composed of multiple parts.
    pub content: Vec<Part>,
    /// Optional metadata associated with the document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

impl Document {
    /// Creates a new document from a simple string slice.
    pub fn from_text(text: &str) -> Self {
        Self {
            content: vec![Part::Text(TextPart {
                text: text.to_string(),
            })],
            metadata: None,
        }
    }

    /// Creates a new document from a `serde_json::Value`.
    /// The value is stored in the document's metadata.
    pub fn from_json(json: Value) -> Result<Self> {
        let text = serde_json::to_string(&json)?;
        Ok(Self {
            content: vec![Part::Text(TextPart { text })],
            metadata: Some(json),
        })
    }
}

/// A reference to a configured retriever.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RetrieverReference<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
}

/// A reference to a configured indexer.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IndexerReference<C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
}

/// A trait for a component that can retrieve documents based on a query.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Retrieves a list of documents relevant to the given query.
    async fn retrieve(&self, query: Document, options: Option<Value>) -> Result<Vec<Document>>;
}

/// A trait for a component that can index documents.
#[async_trait]
pub trait Indexer: Send + Sync {
    /// Indexes a list of documents.
    async fn index(&self, docs: Vec<Document>, options: Option<Value>) -> Result<()>;
}

/// High-level function to retrieve documents using a registered retriever.
///
/// Note: This is a placeholder for a future implementation that would use a
/// central registry to look up and dispatch to the named retriever.
pub async fn retrieve(
    retriever: &dyn Retriever,
    query: Document,
    options: Option<Value>,
) -> Result<Vec<Document>> {
    retriever.retrieve(query, options).await
}

/// High-level function to index documents using a registered indexer.
///
/// Note: This is a placeholder for a future implementation that would use a
/// central registry to look up and dispatch to the named indexer.
pub async fn index(
    indexer: &dyn Indexer,
    docs: Vec<Document>,
    options: Option<Value>,
) -> Result<()> {
    indexer.index(docs, options).await
}
