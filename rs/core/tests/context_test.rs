// Copyright 2025 Google LLC
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

//! # Context Provider Tests
//!
//! Integration tests for the context providers, ported from `context_test.ts`.

use genkit_core::context::{ApiKeyPolicy, ApiKeyProvider, ContextProvider, RequestData};
use genkit_core::error::Error;
use genkit_core::status::StatusCode;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Helper function to create `RequestData` for testing purposes.
fn test_request(key: Option<&str>) -> RequestData {
    let mut headers = HashMap::new();
    if let Some(k) = key {
        // Use lowercase "authorization" to mimic real-world header normalization.
        headers.insert("authorization".to_string(), k.to_string());
    }
    RequestData {
        method: "POST".to_string(),
        headers,
        input: Value::Null,
    }
}

#[tokio::test]
async fn test_api_key_extractor_only() {
    let provider = ApiKeyProvider::new(ApiKeyPolicy::ExtractOnly);

    // Test case: No key provided in the request.
    let context_no_key = provider.provide(&test_request(None)).await.unwrap();
    let expected_auth_no_key = json!({ "apiKey": null });
    assert_eq!(context_no_key.auth.unwrap(), expected_auth_no_key);

    // Test case: A key is provided in the request.
    let context_with_key = provider
        .provide(&test_request(Some("my-secret-key")))
        .await
        .unwrap();
    let expected_auth_with_key = json!({ "apiKey": "my-secret-key" });
    assert_eq!(context_with_key.auth.unwrap(), expected_auth_with_key);
}

#[tokio::test]
async fn test_api_key_validator_require() {
    let required_key = "my-secret-key".to_string();
    let provider = ApiKeyProvider::new(ApiKeyPolicy::Require(required_key.clone()));

    // Test case: Correct key is provided.
    let context_correct = provider
        .provide(&test_request(Some(&required_key)))
        .await
        .unwrap();
    let expected_auth_correct = json!({ "apiKey": required_key });
    assert_eq!(context_correct.auth.unwrap(), expected_auth_correct);

    // Test case: An incorrect key is provided.
    let err_wrong_key = provider
        .provide(&test_request(Some("wrong-key")))
        .await
        .unwrap_err();
    match err_wrong_key {
        Error::UserFacing(status) => {
            assert_eq!(status.code, StatusCode::PermissionDenied);
            assert_eq!(status.message, "Permission Denied");
        }
        _ => panic!("Expected a UserFacing error for wrong key"),
    }

    // Test case: No key is provided.
    let err_no_key = provider.provide(&test_request(None)).await.unwrap_err();
    match err_no_key {
        Error::UserFacing(status) => {
            assert_eq!(status.code, StatusCode::Unauthenticated);
            assert_eq!(status.message, "Unauthenticated");
        }
        _ => panic!("Expected a UserFacing error for no key"),
    }
}
