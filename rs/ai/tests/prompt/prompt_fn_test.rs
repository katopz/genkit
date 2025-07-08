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

use genkit_ai::document::Document;
use genkit_ai::prompt::{define_prompt, PromptConfig};
use genkit_ai::session::Session;
use genkit_core::context::ActionContext;
use genkit_core::error::Result;
use serde_json::{json, Value};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

// Import test helpers
#[path = "../helpers.rs"]
mod helpers;

use genkit_ai::Model;

#[tokio::test]
async fn test_docs_from_function() {
    let (registry, _last_request) = helpers::registry_with_echo_model().await;
    let mut mut_registry = (*registry).clone();

    // Define the input and state for the test
    let input = json!({ "name": "foo" });
    let state = json!({ "name": "bar" });

    // Create the async closure for the docs resolver, with an explicit return type.
    // This tells the compiler how to coerce the concrete `async move` block
    // into the `dyn Future` trait object required by `DocsResolver`.
    let docs_resolver = |input: Value,
                         state: Option<Value>,
                         _context: Option<ActionContext>|
     -> Pin<Box<dyn Future<Output = Result<Vec<Document>>> + Send>> {
        Box::pin(async move {
            let input_name = input.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let state_name = state
                .as_ref()
                .and_then(|s| s.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let doc1 = Document::from_text(format!("doc {}", input_name), None);
            let doc2 = Document::from_text(format!("doc {}", state_name), None);

            Ok(vec![doc1, doc2])
        })
    };

    // Manually construct the prompt configuration
    let prompt_config: PromptConfig<Value, Value, Value> = PromptConfig {
        name: "promptFnTest".to_string(),
        model: Some(Model::Name("echoModel".to_string())),
        prompt: Some("hello {{name}} ({{@state.name}})".to_string()),
        docs_fn: Some(Box::new(docs_resolver)),
        ..Default::default()
    };

    // Define the prompt
    let p = define_prompt(&mut mut_registry, prompt_config);

    // Create a session with the specified state
    let session = Session::new(registry.clone(), None, None, Some(state))
        .await
        .unwrap();
    let session_arc = Arc::new(session);

    // Render the prompt within the session context
    let rendered_opts = session_arc
        .run(async move { p.render(input, None).await })
        .await
        .unwrap();

    // Assert that the documents were correctly resolved and added to the options
    let expected_docs = vec![
        Document::from_text("doc foo", None),
        Document::from_text("doc bar", None),
    ];

    assert_eq!(rendered_opts.docs, Some(expected_docs));
}
