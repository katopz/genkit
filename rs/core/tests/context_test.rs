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

use genkit_core::context::{api_key, ActionContext, ApiKeyPolicy, ContextProvider, RequestData};
use genkit_core::error::Error;
use genkit_core::status::StatusCode;
use rstest::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

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

#[cfg(test)]
/// 'apiKey'
mod api_key_test {
    use super::*;

    #[rstest]
    #[tokio::test]
    /// 'can merely save api keys'
    async fn can_merely_save_api_keys() {
        let provider = api_key(ApiKeyPolicy::ExtractOnly);

        // No key provided
        let context = provider.provide(&test_request(None)).await.unwrap();
        assert_eq!(
            context,
            ActionContext {
                auth: Some(json!({"apiKey": null})),
                ..Default::default()
            }
        );

        // Key provided
        let context_with_key = provider.provide(&test_request(Some("key"))).await.unwrap();
        assert_eq!(
            context_with_key,
            ActionContext {
                auth: Some(json!({"apiKey": "key"})),
                ..Default::default()
            }
        );
    }

    #[rstest]
    #[tokio::test]
    /// 'can expect specific keys'
    async fn can_expect_specific_keys() {
        let provider = api_key(ApiKeyPolicy::Require("key".to_string()));

        // Correct key
        let context = provider.provide(&test_request(Some("key"))).await.unwrap();
        assert_eq!(
            context,
            ActionContext {
                auth: Some(json!({"apiKey": "key"})),
                ..Default::default()
            }
        );

        // Wrong key
        let err = provider
            .provide(&test_request(Some("wrong-key")))
            .await
            .unwrap_err();
        match err {
            Error::UserFacing(status) => {
                assert_eq!(status.code, StatusCode::PermissionDenied);
                assert_eq!(status.message, "Permission Denied");
            }
            other => panic!(
                "Expected UserFacing PermissionDenied error, got {:?}",
                other
            ),
        }

        // No key
        let err_no_key = provider.provide(&test_request(None)).await.unwrap_err();
        match err_no_key {
            Error::UserFacing(status) => {
                assert_eq!(status.code, StatusCode::Unauthenticated);
                assert_eq!(status.message, "Unauthenticated");
            }
            other => panic!("Expected UserFacing Unauthenticated error, got {:?}", other),
        }
    }

    #[rstest]
    #[tokio::test]
    /// 'can use a policy function'
    async fn can_use_a_policy_function() {
        // Test with a key provided
        let policy_with_key = ApiKeyPolicy::Custom(Arc::new(|context: &ActionContext| {
            let expected_context = ActionContext {
                auth: Some(json!({"apiKey": "key"})),
                ..Default::default()
            };
            assert_eq!(context, &expected_context);
            Ok(())
        }));

        let provider_with_key = api_key(policy_with_key);
        // The policy function just asserts; if it doesn't panic or error, the test passes for this part.
        let result = provider_with_key
            .provide(&test_request(Some("key")))
            .await
            .unwrap();
        // We can also assert the provider returns the correct context.
        assert_eq!(
            result.auth,
            Some(json!({
                "apiKey": "key"
            }))
        );

        // Test without a key
        let policy_without_key = ApiKeyPolicy::Custom(Arc::new(|context: &ActionContext| {
            let expected_context = ActionContext {
                auth: Some(json!({"apiKey": null})),
                ..Default::default()
            };
            assert_eq!(context, &expected_context);
            Ok(())
        }));

        let provider_without_key = api_key(policy_without_key);
        let result_no_key = provider_without_key
            .provide(&test_request(None))
            .await
            .unwrap();
        assert_eq!(result_no_key.auth, Some(json!({"apiKey": null})));
    }
}
