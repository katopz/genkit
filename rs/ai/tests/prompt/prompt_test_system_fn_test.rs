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

use genkit_ai::prompt::{PromptConfig, PromptGenerateOptions};
use genkit_ai::Model;
use genkit_core::context::ActionContext;
use genkit_core::error::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::prompt_test_helpers::{test_runner, TestCase};

#[derive(Serialize, Deserialize, JsonSchema, Debug)]
struct TestInput {
    name: String,
}

#[tokio::test]
async fn test_renders_system_prompt_from_a_function() -> Result<()> {
    let system_fn = Arc::new(
        |input: Value,
         state: Option<Value>,
         _: Option<ActionContext>|
         -> Pin<Box<dyn Future<Output = Result<String>> + Send>> {
            Box::pin(async move {
                let typed_input: TestInput = serde_json::from_value(input).unwrap();
                let state_name = state
                    .as_ref()
                    .and_then(|s| s.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(format!("hello {} ({})", typed_input.name, state_name))
            })
        },
    );

    test_runner(Box::new(TestCase {
        name: "renders system prompt from a function".to_string(),
        config: PromptConfig {
            name: "prompt1".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            config: Some(json!({ "banana": "ripe" })),
            input: Some(serde_json::to_value(schemars::schema_for!(TestInput))?),
            system_fn: Some(system_fn),
            ..Default::default()
        },
        input: json!({ "name": "foo" }),
        state: Some(json!({ "name": "bar" })),
        options: Some(PromptGenerateOptions {
            config: Some(json!({ "temperature": 11 })),
            ..Default::default()
        }),
        context: None,
        want_text:
            "Echo: system: hello foo (bar); config: {\"banana\":\"ripe\",\"temperature\":11}"
                .to_string(),
        want_rendered: json!({
            "model": "echoModel",
            "config": { "banana": "ripe", "temperature": 11 },
            "messages": [{ "role": "system", "content": [{ "text": "hello foo (bar)" }] }],
        }),
    }))
    .await
}

#[tokio::test]
async fn test_renders_system_prompt_from_a_function_with_context() -> Result<()> {
    let system_fn = Arc::new(
        |input: Value,
         state: Option<Value>,
         context: Option<ActionContext>|
         -> Pin<Box<dyn Future<Output = Result<String>> + Send>> {
            Box::pin(async move {
                let typed_input: TestInput = serde_json::from_value(input).unwrap();
                let state_name = state
                    .as_ref()
                    .and_then(|s| s.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let email = context
                    .as_ref()
                    .and_then(|c| c.get("auth"))
                    .and_then(|a| a.get("email"))
                    .and_then(|e| e.as_str())
                    .unwrap_or("");
                Ok(format!(
                    "hello {} ({}, {})",
                    typed_input.name, state_name, email
                ))
            })
        },
    );

    let mut context_map = HashMap::new();
    context_map.insert("auth".to_string(), json!({ "email": "a@b.c" }));

    test_runner(Box::new(TestCase {
        name: "renders system prompt from a function with context".to_string(),
        config: PromptConfig {
            name: "prompt1".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            config: Some(json!({ "banana": "ripe" })),
            input: Some(serde_json::to_value(schemars::schema_for!(TestInput))?),
            system_fn: Some(system_fn),
            ..Default::default()
        },
        input: json!({ "name": "foo" }),
        state: Some(json!({ "name": "bar" })),
        options: Some(PromptGenerateOptions {
            config: Some(json!({ "temperature": 11 })),
            ..Default::default()
        }),
        context: Some(context_map.into()),
        want_text:
            "Echo: system: hello foo (bar, a@b.c); config: {\"banana\":\"ripe\",\"temperature\":11}"
                .to_string(),
        want_rendered: json!({
            "model": "echoModel",
            "config": { "banana": "ripe", "temperature": 11 },
            "messages": [{ "role": "system", "content": [{ "text": "hello foo (bar, a@b.c)" }] }],
        }),
    }))
    .await
}

#[tokio::test]
async fn test_renders_system_prompt_from_a_function_with_context_as_render_option() -> Result<()> {
    let system_fn = Arc::new(
        |input: Value,
         state: Option<Value>,
         context: Option<ActionContext>|
         -> Pin<Box<dyn Future<Output = Result<String>> + Send>> {
            Box::pin(async move {
                let typed_input: TestInput = serde_json::from_value(input).unwrap();
                let state_name = state
                    .as_ref()
                    .and_then(|s| s.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let email = context
                    .as_ref()
                    .and_then(|c| c.get("auth"))
                    .and_then(|a| a.get("email"))
                    .and_then(|e| e.as_str())
                    .unwrap_or("");
                Ok(format!(
                    "hello {} ({}, {})",
                    typed_input.name, state_name, email
                ))
            })
        },
    );

    let mut context_map = HashMap::new();
    context_map.insert("auth".to_string(), json!({ "email": "a@b.c" }));

    test_runner(Box::new(TestCase {
        name: "renders system prompt from a function with context as render option".to_string(),
        config: PromptConfig {
            name: "prompt1".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            config: Some(json!({ "banana": "ripe" })),
            input: Some(serde_json::to_value(schemars::schema_for!(TestInput))?),
            system_fn: Some(system_fn),
            ..Default::default()
        },
        input: json!({ "name": "foo" }),
        state: Some(json!({ "name": "bar" })),
        options: Some(PromptGenerateOptions {
            config: Some(json!({ "temperature": 11 })),
            context: Some(context_map.into()),
            ..Default::default()
        }),
        context: None,
        want_text:
            "Echo: system: hello foo (bar, a@b.c); config: {\"banana\":\"ripe\",\"temperature\":11}"
                .to_string(),
        want_rendered: json!({
            "model": "echoModel",
            "config": { "banana": "ripe", "temperature": 11 },
            "context": { "auth": { "email": "a@b.c" } },
            "messages": [
                { "role": "system", "content": [{ "text": "hello foo (bar, a@b.c)" }] },
            ],
        }),
    }))
    .await
}
