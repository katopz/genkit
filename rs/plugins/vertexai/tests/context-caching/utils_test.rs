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

use genkit::Result;
use genkit_vertexai::context_caching::{
    types::{CacheConfig, CacheConfigDetails},
    CachedContent, ListCachedContentsResponse,
};
use genkit_vertexai::model::types::VertexContent;
use rstest::*;
use serde_json::json;
use sha2::{Digest, Sha256};

#[cfg(test)]
/// generateCacheKey
mod generate_cache_key_tests {
    use genkit_vertexai::context_caching::utils::{generate_cache_key, CacheKeyRequest};

    use super::*;

    #[rstest]
    #[tokio::test]
    /// 'should generate a SHA-256 hash for a given request object'
    async fn test_generate_cache_key() {
        let contents: Vec<VertexContent> = serde_json::from_value(json!([
            {
                "role": "user",
                "parts": [{"text": "Hello world"}]
            }
        ]))
        .unwrap();

        let request = CacheKeyRequest {
            model: "gemini-1.5-pro-001",
            contents: &contents,
        };
        let request_string = serde_json::to_string(&request).unwrap();

        let mut hasher = Sha256::new();
        hasher.update(request_string.as_bytes());
        let expected_hash = format!("{:x}", hasher.finalize());

        let result = generate_cache_key(&request).unwrap();
        assert_eq!(result, expected_hash);
    }

    #[rstest]
    #[tokio::test]
    /// 'should generate different hashes for different objects'
    async fn test_generate_different_hashes() {
        let contents1: Vec<VertexContent> = serde_json::from_value(json!([
            {
                "role": "user",
                "parts": [{"text": "Hello"}]
            }
        ]))
        .unwrap();
        let request1 = CacheKeyRequest {
            model: "gemini-1.5-pro-001",
            contents: &contents1,
        };

        let contents2: Vec<VertexContent> = serde_json::from_value(json!([
            {
                "role": "user",
                "parts": [{"text": "Goodbye"}]
            }
        ]))
        .unwrap();
        let request2 = CacheKeyRequest {
            model: "gemini-1.5-pro-001",
            contents: &contents2,
        };

        let hash1 = generate_cache_key(&request1).unwrap();
        let hash2 = generate_cache_key(&request2).unwrap();

        assert_ne!(
            hash1, hash2,
            "Hashes for different objects should not be the same"
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'should be consistent for the same input'
    async fn test_generate_cache_key_consistency() {
        let contents: Vec<VertexContent> = serde_json::from_value(json!([
            {
                "role": "user",
                "parts": [{"text": "Consistent message"}]
            }
        ]))
        .unwrap();

        let request = CacheKeyRequest {
            model: "gemini-1.5-pro-001",
            contents: &contents,
        };

        let hash1 = generate_cache_key(&request).unwrap();
        let hash2 = generate_cache_key(&request).unwrap();

        assert_eq!(hash1, hash2, "Hashes for the same object should match");
    }
}

#[cfg(test)]
/// getContentForCache
mod get_content_for_cache_tests {
    use genkit_vertexai::context_caching::utils::get_content_for_cache;

    use super::*;

    #[rstest]
    #[tokio::test]
    /// 'should correctly retrieve cached content and updated chat request'
    async fn test_should_correctly_split_the_history() {
        let history: Vec<VertexContent> = serde_json::from_value(json!([
            { "role": "system", "parts": [{ "text": "System message" }] },
            { "role": "user", "parts": [{ "text": "Hello!" }] },
        ]))
        .unwrap();

        let details = CacheConfigDetails {
            end_of_cached_contents: 0,
            cache_config: CacheConfig::Object {
                ttl_seconds: Some(300),
            },
        };

        let (to_cache, remaining) = get_content_for_cache(&history, &details).unwrap();

        let expected_cached: Vec<VertexContent> = serde_json::from_value(json!([
            { "role": "system", "parts": [{ "text": "System message" }] },
        ]))
        .unwrap();
        let expected_remaining: Vec<VertexContent> = serde_json::from_value(json!([
            { "role": "user", "parts": [{ "text": "Hello!" }] },
        ]))
        .unwrap();

        assert_eq!(to_cache, expected_cached.as_slice());
        assert_eq!(remaining, expected_remaining.as_slice());
    }
}

/// lookupContextCache
#[cfg(test)]
mod lookup_context_cache_tests {
    use super::*;

    // Note: The following tests are written against a hypothetical, testable
    // version of `lookup_context_cache`. The actual implementation in `index.rs`
    // performs a direct HTTP request and is not easily unit-testable without
    // a mocking library for HTTP calls (e.g., `httpmock`, `wiremock`) and
    // refactoring to allow URL overriding.
    //
    // These tests assume a function `find_cache_in_pages` that contains the
    // core pagination and search logic, and takes a closure to fetch pages,
    // making it testable.

    /// A hypothetical testable function that encapsulates the logic.
    async fn find_cache_in_pages<F, Fut>(
        key: &str,
        mut fetch_page: F,
    ) -> Result<Option<CachedContent>>
    where
        F: FnMut(Option<String>) -> Fut,
        Fut: std::future::Future<Output = Result<ListCachedContentsResponse>>,
    {
        let mut page_token: Option<String> = None;
        for _ in 0..100 {
            // Safety break
            let response = fetch_page(page_token).await?;
            if let Some(contents) = response.cached_contents {
                if let Some(found) = contents
                    .into_iter()
                    .find(|c| c.display_name.as_deref() == Some(key))
                {
                    return Ok(Some(found));
                }
            }
            page_token = response.next_page_token;
            if page_token.is_none() {
                break;
            }
        }
        Ok(None)
    }

    fn create_cached_content(display_name: &str) -> CachedContent {
        serde_json::from_value(json!({
            "name": format!("cachedContents/{}", display_name),
            "displayName": display_name,
            "model": "gemini-1.5-pro-001",
            "contents": [],
        }))
        .unwrap()
    }

    #[rstest]
    #[tokio::test]
    async fn should_return_the_cached_content_if_found() {
        let fetch_page = |_page_token: Option<String>| async {
            Ok(ListCachedContentsResponse {
                cached_contents: Some(vec![
                    create_cached_content("key1"),
                    create_cached_content("key2"),
                ]),
                next_page_token: None,
            })
        };

        let result = find_cache_in_pages("key1", fetch_page).await.unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().display_name.unwrap(), "key1");
    }

    #[rstest]
    #[tokio::test]
    async fn should_return_none_if_the_cached_content_is_not_found() {
        let fetch_page = |_page_token: Option<String>| async {
            Ok(ListCachedContentsResponse {
                cached_contents: Some(vec![
                    create_cached_content("key3"),
                    create_cached_content("key4"),
                ]),
                next_page_token: None,
            })
        };

        let result = find_cache_in_pages("key1", fetch_page).await.unwrap();
        assert!(result.is_none());
    }
}

#[cfg(test)]
/// extractCacheConfig
mod extract_cache_config_tests {
    use std::collections::HashMap;

    use super::*;
    use genkit::model::{GenerateRequest, Role};
    use genkit::MessageData;
    use genkit_vertexai::context_caching::{
        types::{CacheConfig, CacheConfigDetails},
        utils::extract_cache_config,
    };

    #[rstest]
    #[tokio::test]
    /// 'should correctly extract cache config when metadata.cache is present'
    async fn test_extract_cache_config_first_message() {
        let mut metadata_map = HashMap::new();
        metadata_map.insert("cache".to_string(), json!({ "ttlSeconds": 300 }));

        let request = GenerateRequest {
            messages: vec![
                MessageData {
                    role: Role::User,
                    content: vec![],
                    metadata: Some(metadata_map),
                },
                MessageData {
                    role: Role::Model,
                    content: vec![],
                    metadata: Some(HashMap::new()), // empty metadata
                },
            ],
            ..Default::default()
        };

        let result = extract_cache_config(&request).unwrap().unwrap();
        let expected = CacheConfigDetails {
            end_of_cached_contents: 0,
            cache_config: CacheConfig::Object {
                ttl_seconds: Some(300),
            },
        };

        assert_eq!(result, expected);
    }

    #[rstest]
    #[tokio::test]
    /// 'should handle invalid metadata.cache structures gracefully'
    async fn test_extract_invalid_cache_config() {
        let mut metadata_map = HashMap::new();
        metadata_map.insert("cache".to_string(), json!("invalid"));

        let request = GenerateRequest {
            messages: vec![MessageData {
                role: Role::User,
                content: vec![],
                metadata: Some(metadata_map),
            }],
            ..Default::default()
        };

        let result = extract_cache_config(&request);
        assert!(result.is_err());
        let err_string = result.unwrap_err().to_string();
        assert!(
            err_string.contains("Failed to parse cache config"),
            "Error message did not indicate a parsing failure: {}",
            err_string
        );
    }
}

#[cfg(test)]
/// calculateTTL
mod calculate_ttl_tests {
    use super::*;
    use genkit_vertexai::context_caching::{
        constants::DEFAULT_TTL,
        types::{CacheConfig, CacheConfigDetails},
        utils::calculate_ttl,
    };

    #[rstest]
    #[tokio::test]
    /// 'should return the default TTL when cacheConfig is true'
    async fn test_return_default_ttl_when_cache_config_is_true() {
        let cache_config_details = CacheConfigDetails {
            cache_config: CacheConfig::Boolean(true),
            end_of_cached_contents: 0,
        };
        let result = calculate_ttl(&cache_config_details);
        assert_eq!(result, DEFAULT_TTL);
    }
}
