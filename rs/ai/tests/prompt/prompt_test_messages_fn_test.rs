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

use genkit_ai::document::Part;
use genkit_ai::message::{MessageData, Role};
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
async fn test_renders_messages_from_function() -> Result<()> {
    let messages_fn = Arc::new(
        |input: Value,
         state: Option<Value>,
         _: Option<ActionContext>|
         -> Pin<Box<dyn Future<Output = Result<Vec<MessageData>>> + Send>> {
            Box::pin(async move {
                let typed_input: TestInput = serde_json::from_value(input).unwrap();
                let state_name = state
                    .as_ref()
                    .and_then(|s| s.get("name"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                Ok(vec![
                    MessageData {
                        role: Role::System,
                        content: vec![Part::text(format!("system {}", typed_input.name))],
                        ..Default::default()
                    },
                    MessageData {
                        role: Role::User,
                        content: vec![Part::text(format!("user {}", state_name))],
                        ..Default::default()
                    },
                ])
            })
        },
    );

    test_runner(Box::new(TestCase {
        name: "renders messages from function".to_string(),
        config: PromptConfig {
            name: "prompt1".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            config: Some(json!({ "banana": "ripe" })),
            input: Some(serde_json::to_value(schemars::schema_for!(TestInput))?),
            messages_fn: Some(messages_fn),
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
            "Echo: system: system foo,user bar; config: {\"banana\":\"ripe\",\"temperature\":11}"
                .to_string(),
        want_rendered: json!({
            "model": "echoModel",
            "config": { "banana": "ripe", "temperature": 11 },
            "messages": [
                { "role": "system", "content": [{ "text": "system foo" }] },
                { "role": "user", "content": [{ "text": "user bar" }] },
            ],
        }),
    }))
    .await
}

#[tokio::test]
async fn test_renders_messages_from_function_with_context() -> Result<()> {
    let messages_fn = Arc::new(
        |input: Value,
         state: Option<Value>,
         context: Option<ActionContext>|
         -> Pin<Box<dyn Future<Output = Result<Vec<MessageData>>> + Send>> {
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
                Ok(vec![
                    MessageData {
                        role: Role::System,
                        content: vec![Part::text(format!("system {}", typed_input.name))],
                        ..Default::default()
                    },
                    MessageData {
                        role: Role::User,
                        content: vec![Part::text(format!("user {}, {}", state_name, email))],
                        ..Default::default()
                    },
                ])
            })
        },
    );

    let mut context_map = HashMap::new();
    context_map.insert("auth".to_string(), json!({ "email": "a@b.c" }));

    test_runner(Box::new(TestCase {
        name: "renders messages from function with context".to_string(),
        config: PromptConfig {
            name: "prompt1".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            config: Some(json!({ "banana": "ripe" })),
            input: Some(serde_json::to_value(schemars::schema_for!(TestInput))?),
            messages_fn: Some(messages_fn),
            ..Default::default()
        },
        input: json!({ "name": "foo" }),
        state: Some(json!({ "name": "bar" })),
        options: Some(PromptGenerateOptions {
            config: Some(json!({ "temperature": 11 })),
            ..Default::default()
        }),
        context: Some(context_map.into()),
        want_text: "Echo: system: system foo,user bar, a@b.c; config: {\"banana\":\"ripe\",\"temperature\":11}"
            .to_string(),
        want_rendered: json!({
            "model": "echoModel",
            "config": { "banana": "ripe", "temperature": 11 },
            "messages": [
                { "role": "system", "content": [{ "text": "system foo" }] },
                { "role": "user", "content": [{ "text": "user bar, a@b.c" }] },
            ],
        }),
    }))
    .await
}
