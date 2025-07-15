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

//! # Context Caching
//!
//! This module provides the implementation for context caching.

use crate::common::DerivedParams;
use crate::context_caching::utils::{
    calculate_ttl, generate_cache_key, get_content_for_cache, validate_context_cache_request,
    CacheKeyRequest,
};
use crate::model::gemini::VertexContent;
use crate::{Error, Result};
use genkit::model::GenerateRequest;
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::types::CacheConfigDetails;

// Structs for the Vertex AI Caching API

/// Represents the structure of cached content in Vertex AI.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CachedContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contents: Vec<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(Deserialize)]
pub struct ListCachedContentsResponse {
    #[serde(rename = "cachedContents")]
    pub cached_contents: Option<Vec<CachedContent>>,
    #[serde(rename = "nextPageToken")]
    pub next_page_token: Option<String>,
}

async fn create_cached_content(
    params: &DerivedParams,
    cached_content: CachedContent,
) -> Result<CachedContent> {
    let url = format!(
        "https://{}-aiplatform.googleapis.com/v1beta1/projects/{}/locations/{}/cachedContents",
        params.location, params.project_id, params.location
    );

    let token = params
        .token_provider
        .token(&["https://www.googleapis.com/auth/cloud-platform"])
        .await
        .map_err(|e| Error::GcpAuth(format!("Failed to get auth token: {}", e)))?;

    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
    headers.insert(
        AUTHORIZATION,
        format!("Bearer {}", token.as_str()).parse().unwrap(),
    );

    let client = reqwest::Client::new();
    let response = client
        .post(&url)
        .headers(headers)
        .json(&cached_content)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let error_text = response
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".into());
        return Err(Error::VertexAI(format!(
            "Cache creation failed with status {}: {}",
            status, error_text
        )));
    }

    response.json::<CachedContent>().await.map_err(|e| e.into())
}

async fn lookup_context_cache(
    params: &DerivedParams,
    cache_key: &str,
) -> Result<Option<CachedContent>> {
    let mut page_token: Option<String> = None;
    // Limit page traversals to avoid infinite loops
    for _ in 0..100 {
        let mut url = format!(
            "https://{}-aiplatform.googleapis.com/v1beta1/projects/{}/locations/{}/cachedContents?pageSize=100",
            params.location, params.project_id, params.location
        );
        if let Some(token) = &page_token {
            url.push_str(&format!("&pageToken={}", token));
        }

        let token = params
            .token_provider
            .token(&["https://www.googleapis.com/auth/cloud-platform"])
            .await
            .map_err(|e| Error::GcpAuth(format!("Failed to get auth token: {}", e)))?;

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", token.as_str()).parse().unwrap(),
        );

        let client = reqwest::Client::new();
        let response = client.get(&url).headers(headers).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".into());
            return Err(Error::VertexAI(format!(
                "Cache lookup failed with status {}: {}",
                status, error_text
            )));
        }

        let list_response: ListCachedContentsResponse =
            response.json().await.map_err(Error::from)?;

        if let Some(contents) = list_response.cached_contents {
            if let Some(found) = contents
                .into_iter()
                .find(|c| c.display_name.as_deref() == Some(cache_key))
            {
                return Ok(Some(found));
            }
        }

        page_token = list_response.next_page_token;
        if page_token.is_none() {
            break;
        }
    }
    Ok(None)
}

/// The result of a successful cache handling operation.
pub struct CacheResult {
    /// The remaining contents to be sent in the current request.
    pub remaining_contents: Vec<VertexContent>,
    /// The cache entry that was either found or created.
    pub cache: CachedContent,
}

/// Handles cache validation, creation, and usage.
///
/// If caching is not configured or not applicable, returns `Ok(None)`.
/// If caching is applied, returns `Ok(Some(CacheResult))`.
pub async fn handle_cache_if_needed(
    params: &DerivedParams,
    genkit_request: &GenerateRequest,
    all_contents: &[VertexContent],
    model_version: &str,
    cache_config_details: &Option<CacheConfigDetails>,
) -> Result<Option<CacheResult>> {
    let details = match cache_config_details {
        Some(d) => d,
        None => return Ok(None),
    };

    validate_context_cache_request(genkit_request, model_version)?;

    let (to_cache, remaining) = get_content_for_cache(all_contents, details)?;

    let cache_key_request = CacheKeyRequest {
        model: model_version,
        contents: to_cache,
    };
    let cache_key = generate_cache_key(&cache_key_request)?;

    let maybe_cache = lookup_context_cache(params, &cache_key).await?;

    let cache = if let Some(cache) = maybe_cache {
        cache
    } else {
        let ttl = calculate_ttl(details);
        let create_request = CachedContent {
            name: None,
            display_name: Some(cache_key),
            model: model_version.to_string(),
            contents: to_cache.to_vec(),
            ttl: if ttl > 0 {
                Some(format!("{}s", ttl))
            } else {
                None
            },
        };
        create_cached_content(params, create_request).await?
    };

    if cache.name.is_none() {
        return Err(Error::VertexAI(
            "Failed to use context cache feature: cache name is missing".to_string(),
        ));
    }

    Ok(Some(CacheResult {
        remaining_contents: remaining.to_vec(),
        cache,
    }))
}
