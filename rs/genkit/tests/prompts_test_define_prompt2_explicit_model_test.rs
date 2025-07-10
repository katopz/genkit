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

use genkit::{
    model::Model,
    prompt::{PromptConfig, PromptGenerateOptions},
    Genkit, Part, Role,
};
use genkit_ai::MessageData;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
/// 'calls prompt with explicit model'
async fn test_calls_prompt_with_explicit_model(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit.define_prompt::<TestInput, Value, Value>(PromptConfig {
        name: "hi_explicit_model2_test".to_string(),
        model: Some(Model::Name("echoModel".to_string())),
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
    });

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
/// 'calls prompt with explicit model with config'
async fn test_calls_prompt_with_explicit_model_and_config(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit.define_prompt::<TestInput, Value, Value>(PromptConfig {
        name: "hi_explicit_model_with_config2_test".to_string(),
        model: Some(Model::Name("echoModel".to_string())),
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
    });

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
/// 'calls prompt with explicit model with call site config'
async fn test_calls_prompt_with_explicit_model_and_call_site_config(
    #[future] genkit_instance: Arc<Genkit>,
) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit.define_prompt::<TestInput, Value, Value>(PromptConfig {
        name: "hi_explicit_model_call_site_config_test".to_string(),
        model: Some(Model::Name("echoModel".to_string())),
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
    });

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                config: Some(json!({"version": "abc"})),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    // Note: The order of keys in the final JSON is not guaranteed.
    // We check for both possible orderings.
    let expected1 = "Echo: hi Genkit; config: {\"temperature\":11,\"version\":\"abc\"}";
    let expected2 = "Echo: hi Genkit; config: {\"version\":\"abc\",\"temperature\":11}";
    let actual = response.text().unwrap();

    assert!(actual == expected1 || actual == expected2);
}
