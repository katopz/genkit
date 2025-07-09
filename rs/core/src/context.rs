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

//! # Action and Flow Context
//!
//! This module provides mechanisms for managing execution context, including
//! authentication data. It is the Rust equivalent of `context.ts`.
//!
//! In Rust, asynchronous context is managed using `tokio::task_local!`, which is
//! analogous to `AsyncLocalStorage` in Node.js.

use crate::error::{Error, Result};
use crate::status::StatusCode;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

tokio::task_local! {
    /// Task-local storage for the current `ActionContext`.
    pub static CONTEXT: ActionContext;
}

/// Action side channel data, like auth and other invocation context information.
///
/// This is the Rust equivalent of the `ActionContext` interface in TypeScript.
/// `serde_json::Value` is used for the `auth` field to maintain flexibility,
/// and `#[serde(flatten)]` allows for arbitrary additional fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct ActionContext {
    /// Information about the currently authenticated user if provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<Value>,
    /// Additional context data.
    #[serde(flatten)]
    pub additional_context: HashMap<String, Value>,
}

impl ActionContext {
    /// Merges another `ActionContext` into this one.
    ///
    /// The `auth` field from `other` will overwrite the `auth` field in `self` if present.
    /// The `additional_context` maps are merged.
    pub fn extend(&mut self, other: ActionContext) {
        if other.auth.is_some() {
            self.auth = other.auth;
        }
        self.additional_context.extend(other.additional_context);
    }

    /// Retrieves a value from the context by key.
    ///
    /// This is a convenience method that checks for the special "auth" key
    /// before looking in the `additional_context` map.
    pub fn get(&self, key: &str) -> Option<&Value> {
        if key == "auth" {
            self.auth.as_ref()
        } else {
            self.additional_context.get(key)
        }
    }
}

impl From<HashMap<String, Value>> for ActionContext {
    fn from(mut map: HashMap<String, Value>) -> Self {
        let auth = map.remove("auth");
        ActionContext {
            auth,
            additional_context: map,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct FlowContext {
    pub flow_id: String,
}

tokio::task_local! {
    /// Task-local storage for the current `FlowContext`.
    pub static FLOW_CONTEXT: FlowContext;
}

/// Context type for API key-based authentication.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ApiKeyContext {
    pub api_key: String,
}

/// A universal type that request handling extensions (e.g., web frameworks) can
/// map their incoming requests to.
///
/// This allows `ContextProvider` implementations to be portable across different
/// web frameworks. This is the Rust equivalent of `RequestData`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestData {
    /// The request method (e.g., "GET", "POST").
    pub method: String,
    /// Request headers. In practice, keys should be treated case-insensitively.
    pub headers: HashMap<String, String>,
    /// The request body or input.
    pub input: Value,
}

/// A trait for middleware that reads request data and produces an `ActionContext`.
///
/// Implementors of this trait can perform tasks like parsing and validating
/// authentication tokens. This is the Rust equivalent of the `ContextProvider`
// function type.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    async fn provide(&self, request: &RequestData) -> Result<ActionContext>;
}

/// Executes a future within the scope of a given `ActionContext`.
///
/// Any code inside the future (and any functions it calls) can access the
/// context using `get_context()`.
pub async fn run_with_context<F, R>(context: ActionContext, future: F) -> R
where
    F: std::future::Future<Output = R>,
{
    CONTEXT.scope(context, future).await
}

/// Gets a clone of the `ActionContext` for the current task.
///
/// Returns `None` if called outside the scope of `run_with_context`.
pub fn get_context() -> Option<ActionContext> {
    CONTEXT.try_with(|context| context.clone()).ok()
}

/// Executes a future within the scope of a given `FlowContext`.
///
/// Any code inside the future (and any functions it calls) can access the
/// context using `get_flow_context()`.
pub async fn run_with_flow_context<F, R>(context: FlowContext, future: F) -> R
where
    F: std::future::Future<Output = R>,
{
    FLOW_CONTEXT.scope(context, future).await
}

/// Gets a clone of the `FlowContext` for the current task.
///
/// Returns `None` if called outside the scope of `run_with_flow_context`.
pub fn get_flow_context() -> Option<FlowContext> {
    FLOW_CONTEXT.try_with(|context| context.clone()).ok()
}

/// Defines the validation policy for the `ApiKeyProvider`.
pub enum ApiKeyPolicy {
    /// No validation is performed; the provider only extracts the key if present.
    ExtractOnly,
    /// The provider validates that the API key matches a specific required key.
    Require(String),
}

/// A `ContextProvider` that handles API key authentication from the
/// `authorization` header.
pub struct ApiKeyProvider {
    policy: ApiKeyPolicy,
}

impl ApiKeyProvider {
    /// Creates a new `ApiKeyProvider` with the specified policy.
    pub fn new(policy: ApiKeyPolicy) -> Self {
        Self { policy }
    }
}

/// Creates an API key-based authentication provider.
pub fn api_key(policy: ApiKeyPolicy) -> ApiKeyProvider {
    ApiKeyProvider::new(policy)
}

/// Helper to extract the API key from the 'authorization' header.
fn extract_key_from_headers(headers: &HashMap<String, String>) -> Option<String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("authorization"))
        .map(|(_, v)| v.clone())
}

#[async_trait]
impl ContextProvider for ApiKeyProvider {
    async fn provide(&self, request: &RequestData) -> Result<ActionContext> {
        let api_key = extract_key_from_headers(&request.headers);

        match &self.policy {
            ApiKeyPolicy::ExtractOnly => {
                let auth_value = api_key.map_or(Value::Null, Value::from);
                Ok(ActionContext {
                    auth: Some(serde_json::json!({ "apiKey": auth_value })),
                    ..Default::default()
                })
            }
            ApiKeyPolicy::Require(required_key) => match api_key {
                None => Err(Error::new_user_facing(
                    StatusCode::Unauthenticated,
                    "Unauthenticated",
                    None,
                )),
                Some(ref key) if key != required_key => Err(Error::new_user_facing(
                    StatusCode::PermissionDenied,
                    "Permission Denied",
                    None,
                )),
                Some(key) => Ok(ActionContext {
                    auth: Some(serde_json::json!({ "apiKey": key })),
                    ..Default::default()
                }),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_request(key: Option<&str>) -> RequestData {
        let mut headers = HashMap::new();
        if let Some(k) = key {
            headers.insert("authorization".to_string(), k.to_string());
        }
        RequestData {
            method: "POST".to_string(),
            headers,
            input: Value::Null,
        }
    }

    #[tokio::test]
    async fn test_api_key_extractor() {
        let provider = ApiKeyProvider::new(ApiKeyPolicy::ExtractOnly);

        // No key provided
        let context = provider.provide(&test_request(None)).await.unwrap();
        assert_eq!(context.auth, Some(json!({"apiKey": null})));

        // Key provided
        let context = provider
            .provide(&test_request(Some("my-key")))
            .await
            .unwrap();
        assert_eq!(context.auth, Some(json!({"apiKey": "my-key"})));
    }

    #[tokio::test]
    async fn test_api_key_validator() {
        let provider = ApiKeyProvider::new(ApiKeyPolicy::Require("secret-key".to_string()));

        // Correct key
        let context = provider
            .provide(&test_request(Some("secret-key")))
            .await
            .unwrap();
        assert_eq!(context.auth, Some(json!({"apiKey": "secret-key"})));

        // Wrong key
        let err = provider
            .provide(&test_request(Some("wrong-key")))
            .await
            .unwrap_err();
        match err {
            Error::UserFacing(status) => assert_eq!(status.code, StatusCode::PermissionDenied),
            _ => panic!("Expected UserFacing error"),
        }

        // No key
        let err = provider.provide(&test_request(None)).await.unwrap_err();
        match err {
            Error::UserFacing(status) => assert_eq!(status.code, StatusCode::Unauthenticated),
            _ => panic!("Expected UserFacing error"),
        }
    }

    #[tokio::test]
    async fn test_run_and_get_context() {
        assert!(get_context().is_none());

        let my_context = ActionContext {
            auth: Some(json!({"user": "test"})),
            ..Default::default()
        };

        run_with_context(my_context.clone(), async {
            let retrieved_context = get_context().unwrap();
            assert_eq!(retrieved_context, my_context);

            // Nested context
            let nested_context = ActionContext {
                auth: Some(json!({"user": "nested"})),
                ..Default::default()
            };
            run_with_context(nested_context.clone(), async {
                let retrieved_nested = get_context().unwrap();
                assert_eq!(retrieved_nested, nested_context);
            })
            .await;

            // Context is restored after nested scope
            let retrieved_context_after = get_context().unwrap();
            assert_eq!(retrieved_context_after, my_context);
        })
        .await;

        assert!(get_context().is_none());
    }

    #[tokio::test]
    async fn test_run_and_get_flow_context() {
        assert!(get_flow_context().is_none());

        let my_context = FlowContext {
            flow_id: "test".into(),
        };

        run_with_flow_context(my_context.clone(), async {
            let retrieved_context = get_flow_context().unwrap();
            assert_eq!(retrieved_context, my_context);

            // Nested context
            let nested_context = FlowContext {
                flow_id: "nested".into(),
            };
            run_with_flow_context(nested_context.clone(), async {
                let retrieved_nested = get_flow_context().unwrap();
                assert_eq!(retrieved_nested, nested_context);
            })
            .await;

            // Context is restored after nested scope
            let retrieved_context_after = get_flow_context().unwrap();
            assert_eq!(retrieved_context_after, my_context);
        })
        .await;

        assert!(get_flow_context().is_none());
    }
}
