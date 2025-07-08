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

use std::sync::Arc;

use genkit_ai::document::Document;
use genkit_ai::prompt::{PromptConfig, PromptGenerateOptions};
use genkit_ai::Model;
use genkit_core::context::ActionContext;
use genkit_core::error::Result;
use serde_json::{json, Value};

use crate::prompt_helpers::{test_runner, TestCase};

#[tokio::test]
async fn test_docs_from_function() -> Result<()> {
    let docs_resolver =
        Arc::new(
            |input: Value,
             state: Option<Value>,
             _: Option<ActionContext>|
             -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<Document>>> + Send>,
            > {
                Box::pin(async move {
                    let input_name = input.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let state_name = state
                        .as_ref()
                        .and_then(|s| s.get("name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    Ok(vec![
                        Document::from_text(format!("doc {}", input_name), None),
                        Document::from_text(format!("doc {}", state_name), None),
                    ])
                })
            },
        );

    test_runner(TestCase {
        name: "includes docs from function".to_string(),
        config: PromptConfig {
            name: "prompt1".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            config: Some(json!({ "banana": "ripe" })),
            prompt: Some("hello {{name}} ({{@state.name}})".to_string()),
            docs_fn: Some(docs_resolver),
            ..Default::default()
        },
        input: json!({ "name": "foo" }),
        state: Some(json!({ "name": "bar" })),
        options: Some(PromptGenerateOptions {
            config: Some(json!({ "temperature": 11 })),
            ..Default::default()
        }),
        want_text: "Echo: hello foo (bar); config: {\"banana\":\"ripe\",\"temperature\":11}"
            .to_string(),
        want_rendered: json!({
            "model": "echoModel",
            "config": { "banana": "ripe", "temperature": 11 },
            "messages": [{ "role": "user", "content": [{ "text": "hello foo (bar)" }] }],
            "docs": [
                { "content": [{ "text": "doc foo" }], "metadata": null },
                { "content": [{ "text": "doc bar" }], "metadata": null }
            ],
            "tools": null,
            "tool_choice": null,
            "output": null,
            "resume": null,
            "max_turns": null,
            "return_tool_requests": null,
            "system": null,
            "prompt": null,
            "context": null,
        }),
    })
    .await
}
