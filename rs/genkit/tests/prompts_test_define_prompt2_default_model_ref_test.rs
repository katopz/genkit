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

use genkit::{prompt::PromptConfig, Genkit, GenkitOptions, Part, Role};
use genkit_ai::MessageData;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
struct TestInput {
    name: String,
}

#[fixture]
async fn genkit_instance_with_ref() -> Arc<Genkit> {
    let echo_plugin = Arc::new(helpers::EchoModelPlugin);
    Genkit::init(GenkitOptions {
        plugins: vec![echo_plugin],
        default_model: Some("echoModel".to_string()),
        context: None,
    })
    .await
    .unwrap()
}

#[rstest]
#[tokio::test]
/// 'calls prompt with default model'
async fn test_calls_prompt_with_default_model_ref(#[future] genkit_instance_with_ref: Arc<Genkit>) {
    let genkit = genkit_instance_with_ref.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_default_model_ref_test".to_string(),
            messages_fn: Some(Arc::new(|input, _state, _context| {
                Box::pin(async move {
                    Ok(vec![MessageData {
                        role: Role::User,
                        content: vec![Part::text(format!("hi {}", input.name))],
                        ..Default::default()
                    }])
                })
            })),
            ..Default::default()
        })
        .await;

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi Genkit; config: {}");
}

#[rstest]
#[tokio::test]
/// 'streams prompt with default model'
async fn test_streams_prompt_with_default_model_ref(
    #[future] genkit_instance_with_ref: Arc<Genkit>,
) {
    let genkit = genkit_instance_with_ref.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_stream_default_model_ref_test".to_string(),
            config: Some(json!({ "temperature": 11 })),
            messages_fn: Some(Arc::new(|input, _state, _context| {
                Box::pin(async move {
                    Ok(vec![MessageData {
                        role: Role::User,
                        content: vec![Part::text(format!("hi {}", input.name))],
                        ..Default::default()
                    }])
                })
            })),
            ..Default::default()
        })
        .await;

    let mut stream_resp = hi_prompt
        .stream(
            TestInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    let mut chunks: Vec<String> = Vec::new();
    while let Some(chunk_result) = stream_resp.stream.next().await {
        let chunk = chunk_result.unwrap();
        chunks.push(chunk.text());
    }

    let response = stream_resp.response.await.unwrap().unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: hi Genkit; config: {\"temperature\":11}"
    );

    // NOTE: This assertion depends on the `EchoModelPlugin` in `helpers.rs`
    // being modified to stream the chunks "3", "2", "1".
    assert_eq!(chunks, vec!["3", "2", "1"]);
}
