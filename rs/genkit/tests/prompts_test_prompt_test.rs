//! Copyright 2024 Google LLC
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

mod helpers;

use genkit::{prompt::PromptConfig, Genkit, Part, Role};
use genkit_ai::{MessageData, OutputOptions};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
struct EmptyInput {}

#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    helpers::genkit_instance_with_echo_model().await
}

#[rstest]
#[tokio::test]
/// 'loads from the folder'
async fn test_loads_from_folder(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // In Rust, we don't have a direct equivalent of `promptDir`.
    // We simulate the file being loaded by defining the prompt directly.
    let test_prompt_config = PromptConfig {
        name: "test".to_string(),
        prompt: Some("Hello from the prompt file".to_string()),
        config: Some(json!({ "temperature": 11 })),
        output: Some(OutputOptions::default()),
        ..Default::default()
    };
    genkit
        .define_prompt::<EmptyInput, Value, Value>(test_prompt_config)
        .await;

    // Look up the prompt from the registry.
    let test_prompt =
        genkit_ai::prompt::prompt::<EmptyInput, Value, Value>(genkit.registry(), "test")
            .await
            .unwrap();

    // First, test generation to ensure the default model is used correctly.
    let response = test_prompt.generate(EmptyInput {}, None).await.unwrap();
    assert_eq!(
        response.text().unwrap(),
        "Echo: Hello from the prompt file; config: {\"temperature\":11}"
    );

    // Second, test the render method.
    let rendered_opts = test_prompt.render(EmptyInput {}, None).await.unwrap();

    let expected_messages = Some(vec![MessageData {
        role: Role::User,
        content: vec![Part::text("Hello from the prompt file")],
        ..Default::default()
    }]);

    assert_eq!(rendered_opts.messages, expected_messages);
    assert_eq!(rendered_opts.config, Some(json!({ "temperature": 11 })));
    assert_eq!(rendered_opts.output, Some(OutputOptions::default()));
    // The render method itself doesn't apply the default model, so this should be None.
    assert!(rendered_opts.model.is_none());
}
