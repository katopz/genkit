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

//! # AI Document and Content Primitives
//!
//! This module defines the core data structures for representing content, such
//! as `Document` and `Part`. It is the Rust equivalent of `document.ts`.

use crate::{embedder::Embedding, ResourceInput};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

/// Represents a media item, typically an image, video, or audio file.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Media {
    /// The IANA media type of the content (e.g., `image/png`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// The URI of the media content. Can be a `data:`, `http:`, `https:`, or `gs:` URI.
    pub url: String,
}

/// Represents a request from a model to invoke a tool.
#[derive(Default, Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ToolRequest {
    /// An identifier for a specific tool call, used to match a `ToolResponse`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#ref: Option<String>,
    /// The name of the tool to be called.
    pub name: String,
    /// The input parameters for the tool, typically a JSON object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
}

/// Represents the output of a tool, to be sent back to the model.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct ToolResponse {
    /// The identifier of the `ToolRequest` this is a response to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#ref: Option<String>,
    /// The name of the tool that was called.
    pub name: String,
    /// The output data from the tool, typically a JSON object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
}

/// Represents a discrete piece of content in a `Document` or `Message`.
///
/// This struct is designed to be flexible, allowing for different types of content
/// to be represented. In Rust, this is modeled with optional fields, where only
/// one primary content field (e.g., `text`, `media`) should be present in a valid `Part`.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct Part {
    /// A text part of the content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// A media part of the content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media: Option<Media>,
    /// A request from the model to call a tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_request: Option<ToolRequest>,
    /// A response from a tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_response: Option<ToolResponse>,
    /// Arbitrary structured data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    /// Reasoning text from the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    /// Additional metadata associated with this part.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
    /// Resources
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource: Option<ResourceInput>,
}

impl Part {
    /// Creates a new `Part` from a text string.
    pub fn text(text: impl Into<String>) -> Self {
        Part {
            text: Some(text.into()),
            ..Default::default()
        }
    }

    /// Creates a new `Part` from a media.
    pub fn media(url: impl Into<String>, content_type: impl Into<String>) -> Self {
        Part {
            media: Some(Media {
                url: url.into(),
                content_type: Some(content_type.into()),
            }),
            ..Default::default()
        }
    }

    /// Creates a new `Part` from a resource.
    pub fn resource(uri: impl Into<String>) -> Self {
        Part {
            resource: Some(ResourceInput { uri: uri.into() }),
            ..Default::default()
        }
    }
}

impl Part {
    /// Creates a new `Part` containing a `tool_request`.
    pub fn tool_request(
        name: impl Into<String>,
        input: Option<Value>,
        r#ref: Option<String>,
    ) -> Self {
        Part {
            tool_request: Some(ToolRequest {
                name: name.into(),
                input,
                r#ref,
            }),
            ..Default::default()
        }
    }

    /// Creates a new `Part` containing a `tool_response`.
    pub fn tool_response(
        name: impl Into<String>,
        output: Option<Value>,
        r#ref: Option<String>,
    ) -> Self {
        Part {
            tool_response: Some(ToolResponse {
                name: name.into(),
                output,
                r#ref,
            }),
            ..Default::default()
        }
    }
}

/// A type alias for a `Part` that is known to contain a `tool_request`.
pub type ToolRequestPart = Part;
pub type ToolResponsePart = Part;

/// Represents a document with content and metadata.
///
/// Documents can be embedded, indexed, or retrieved, and are composed of one or more `Part`s.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Document {
    pub content: Vec<Part>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

impl Document {
    /// Creates a new `Document` with a deep copy of the provided data.
    pub fn new(content: Vec<Part>, metadata: Option<HashMap<String, Value>>) -> Self {
        Document {
            content: content.clone(),
            metadata: metadata.clone(),
        }
    }

    /// Creates a new `Document` from a text string.
    pub fn from_text(text: impl Into<String>, metadata: Option<HashMap<String, Value>>) -> Self {
        Document::new(vec![Part::text(text)], metadata)
    }

    /// Creates a new `Document` from a single part.
    pub fn from_part(part: Part, metadata: Option<HashMap<String, Value>>) -> Self {
        Document::new(vec![part], metadata)
    }

    /// Creates a new `Document` from a vector of parts.
    pub fn from_parts(parts: Vec<Part>, metadata: Option<HashMap<String, Value>>) -> Self {
        Document::new(parts, metadata)
    }

    /// Creates a new `Document` from a single media item.
    pub fn from_media(
        url: impl Into<String>,
        content_type: Option<String>,
        metadata: Option<HashMap<String, Value>>,
    ) -> Self {
        Document::new(
            vec![Part {
                media: Some(Media {
                    url: url.into(),
                    content_type,
                }),
                ..Default::default()
            }],
            metadata,
        )
    }

    /// Creates a new `Document` from raw data and a data type hint.
    pub fn from_data(
        data: String,
        data_type: &str,
        metadata: Option<HashMap<String, Value>>,
    ) -> Self {
        if data_type == "text" {
            Self::from_text(data, metadata)
        } else {
            Self::from_media(data, Some(data_type.to_string()), metadata)
        }
    }

    /// Concatenates all `text` parts in the document.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<&str>>()
            .join("")
    }

    /// Returns all `media` parts in the document.
    pub fn media(&self) -> Vec<&Media> {
        self.content
            .iter()
            .filter_map(|p| p.media.as_ref())
            .collect()
    }

    /// Gets the first item in the document, either text or a media URL.
    pub fn data(&self) -> String {
        let text = self.text();
        if !text.is_empty() {
            return text;
        }
        if let Some(first_media) = self.media().first() {
            return first_media.url.clone();
        }
        String::new()
    }

    /// Gets the content type of the data returned by `data()`.
    pub fn data_type(&self) -> &str {
        if !self.text().is_empty() {
            "text"
        } else if let Some(first_media) = self.media().first() {
            first_media.content_type.as_deref().unwrap_or("")
        } else {
            ""
        }
    }

    /// Returns the document as a `serde_json::Value`, making a deep copy.
    pub fn to_json_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Creates an array of `Document`s from this document, one for each provided embedding.
    ///
    /// This is useful when a single document is chunked into multiple embeddings.
    /// Each returned document will have the same content but different `embedMetadata`.
    pub fn get_embedding_documents(&self, embeddings: &[Embedding]) -> Vec<Document> {
        if embeddings.len() <= 1 {
            if let Some(embedding) = embeddings.first() {
                if embedding.metadata.is_none() {
                    return vec![self.clone()];
                }
            } else {
                return vec![self.clone()];
            }
        }
        let mut documents = Vec::new();
        for embedding in embeddings {
            let mut new_doc = self.clone();
            if let Some(embed_meta) = &embedding.metadata {
                let doc_meta = new_doc.metadata.get_or_insert_with(HashMap::new);
                doc_meta.insert(
                    "embedMetadata".to_string(),
                    serde_json::to_value(embed_meta).unwrap_or(Value::Null),
                );
            }
            documents.push(new_doc);
        }
        check_unique_documents(&documents);
        documents
    }
}

/// Checks if a slice of documents contains duplicates.
///
/// If duplicates are found, a warning is logged. This is important because
/// vector storage is often keyed by a hash of the document, and duplicates
/// can lead to data loss.
pub fn check_unique_documents(documents: &[Document]) -> bool {
    let mut seen = HashSet::new();
    for doc in documents {
        let serialized = serde_json::to_string(doc).unwrap_or_default();
        if !seen.insert(serialized) {
            println!(
                "Warning: embedding documents are not unique. This may cause data loss in vector stores. Are you missing embed metadata?"
            );
            return false;
        }
    }
    true
}
