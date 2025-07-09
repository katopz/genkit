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
use genkit_ai::MessageData;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Serialize, Deserialize, JsonSchema, Debug, Clone, Default)]
struct TestInput {
    name: String,
}

#[fixture]
async fn genkit_instance() -> Arc<Genkit> {
    helpers::genkit_instance_with_echo_model().await
}

#[rstest]
#[tokio::test]
/// 'calls prompt with default model'
async fn test_calls_prompt_with_default_model(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_default_model2_test".to_string(),
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
/// 'calls legacy prompt with default model'
async fn test_calls_legacy_prompt_with_default_model(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // NOTE: The "legacy" style of `definePrompt(config, runner)` from JS doesn't have
    // a direct equivalent in Rust due to static typing. The idiomatic way to achieve
    // dynamic message generation is to use the `messages_fn` field within the config.
    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_legacy_test".to_string(),
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
/// 'calls legacy prompt with string shorthand'
async fn test_calls_legacy_prompt_with_string_shorthand(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    // NOTE: The "legacy" style of `definePrompt(config, "template")` from JS doesn't have
    // a direct equivalent in Rust. The idiomatic way is to use the `prompt` field within the config.
    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_legacy_string_shorthand_test".to_string(),
            prompt: Some("hi {{name}}".to_string()),
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

use serde_json::json;

#[rstest]
#[tokio::test]
/// 'calls prompt with default model with config'
async fn test_calls_prompt_with_default_model_and_config(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_default_model_with_config_test".to_string(),
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

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: hi Genkit; config: {\"temperature\":11}"
    );
}

#[rstest]
#[tokio::test]
/// 'calls prompt with default model via retrieved prompt'
async fn test_calls_prompt_with_default_model_via_retrieved_prompt(
    #[future] genkit_instance: Arc<Genkit>,
) {
    let genkit = genkit_instance.await;

    genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_retrieved_default_model2_test".to_string(),
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

    let hi_prompt = genkit_ai::prompt::prompt::<TestInput, Value, Value>(
        genkit.registry(),
        "hi_retrieved_default_model2_test",
    )
    .await
    .unwrap();

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
