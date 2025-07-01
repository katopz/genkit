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

//! # Context Caching Utilities
//!
//! This module provides utility functions for the context caching feature.

use super::constants::{invalid_argument_messages, CONTEXT_CACHE_SUPPORTED_MODELS, DEFAULT_TTL};
use super::types::{CacheConfig, CacheConfigDetails};
use crate::model::gemini::VertexContent;
use crate::{Error, Result};
use genkit_ai::model::GenerateRequest;
use genkit_ai::MessageData;
use genkit_core::error::Error as CoreError;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// A request object used for generating a cache key.
///
/// This struct is serialized to a JSON string, which is then hashed to create
/// a unique key for the cached content.
#[derive(Serialize)]
pub struct CacheKeyRequest<'a> {
    pub model: &'a str,
    pub contents: &'a [VertexContent],
}

/// Generates a SHA-256 hash to use as a cache key.
///
/// Hashes the JSON representation of the provided `cache_key_request`.
pub fn generate_cache_key(cache_key_request: &CacheKeyRequest) -> Result<String> {
    let request_string = serde_json::to_string(cache_key_request).map_err(|e| {
        Error::from(CoreError::new_internal(format!(
            "Failed to serialize cache key request: {}",
            e
        )))
    })?;
    let mut hasher = Sha256::new();
    hasher.update(request_string.as_bytes());
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// Splits the chat history into content to be cached and content for the current request.
pub fn get_content_for_cache<'a>(
    history: &'a [VertexContent],
    details: &CacheConfigDetails,
) -> Result<(&'a [VertexContent], &'a [VertexContent])> {
    if history.is_empty() {
        return Err(Error::from(CoreError::new_internal(
            "No history provided for context caching",
        )));
    }

    let split_index = details.end_of_cached_contents + 1;
    if split_index > history.len() {
        return Err(Error::from(CoreError::new_internal(
            "end_of_cached_contents index is out of bounds.",
        )));
    }
    Ok(history.split_at(split_index))
}

/// Extracts the cache configuration from the request metadata.
///
/// It searches for the last message in the request that has a `cache` field
/// in its metadata.
pub fn extract_cache_config(request: &GenerateRequest) -> Result<Option<CacheConfigDetails>> {
    let last_cache_message_index = request.messages.iter().rposition(|message: &MessageData| {
        message
            .metadata
            .as_ref()
            .and_then(|m| m.get("cache"))
            .is_some()
    });

    if let Some(index) = last_cache_message_index {
        let metadata = request.messages[index].metadata.as_ref().unwrap();
        let cache_value = metadata.get("cache").unwrap();
        let cache_config: CacheConfig =
            serde_json::from_value(cache_value.clone()).map_err(|e| {
                Error::from(CoreError::new_internal(format!(
                    "Failed to parse cache config: {}",
                    e
                )))
            })?;
        Ok(Some(CacheConfigDetails {
            cache_config,
            end_of_cached_contents: index,
        }))
    } else {
        Ok(None)
    }
}

/// Validates that a request is compatible with context caching.
///
/// Checks the model version and ensures no unsupported features (like tools) are being used.
pub fn validate_context_cache_request(
    request: &GenerateRequest,
    model_version: &str,
) -> Result<()> {
    if !CONTEXT_CACHE_SUPPORTED_MODELS.contains(&model_version) {
        return Err(Error::from(CoreError::new_user_facing(
            genkit_core::status::StatusCode::InvalidArgument,
            invalid_argument_messages::MODEL_VERSION,
            None,
        )));
    }
    if request.tools.is_some() && !request.tools.as_ref().unwrap().is_empty() {
        return Err(Error::from(CoreError::new_user_facing(
            genkit_core::status::StatusCode::InvalidArgument,
            invalid_argument_messages::TOOLS,
            None,
        )));
    }
    // TODO: Add check for code execution once that's part of the GenerateRequest.
    Ok(())
}

/// Calculates the TTL (Time-To-Live) for the cache in seconds.
pub fn calculate_ttl(cache_config: &CacheConfigDetails) -> u32 {
    match &cache_config.cache_config {
        CacheConfig::Boolean(true) => DEFAULT_TTL,
        CacheConfig::Boolean(false) => 0,
        CacheConfig::Object { ttl_seconds } => ttl_seconds.unwrap_or(DEFAULT_TTL),
    }
}
