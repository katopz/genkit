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

use genkit::prompt::PromptGenerateOptions;
use genkit::{model::Model, prompt::PromptConfig, Genkit};
use genkit::{Part, Role};
use genkit_ai::MessageData;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, to_value, Value};
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
/// 'calls dotprompt with default model'
async fn test_calls_prompt_with_default_model(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_explicit_model_test".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
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

#[rstest]
#[tokio::test]
/// 'calls dotprompt with history and places it before user message'
async fn test_calls_prompt_with_history(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_with_history_test".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            prompt: Some("hi {{name}}".to_string()),
            ..Default::default()
        })
        .await;

    let history = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("hi")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("bye")],
            ..Default::default()
        },
    ];

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                messages: Some(history.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    let actual_messages = response.messages().unwrap();

    assert_eq!(
        to_value(&actual_messages).unwrap(),
        json!([
            {
                "role": "user",
                "content": [{"text": "hi"}]
            },
            {
                "role": "model",
                "content": [{"text": "bye"}]
            },
            {
                "role": "user",
                "content": [{"text": "hi Genkit"}]
            },
            {
                "role": "model",
                "content": [{"text": "Echo: hibyehi Genkit; config: {}"}]
            }
        ])
    );
}

use tokio_stream::StreamExt;

#[rstest]
#[tokio::test]
/// 'streams dotprompt with history and places it before user message'
async fn test_streams_prompt_with_history(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_stream_with_history_test".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            prompt: Some("hi {{name}}".to_string()),
            ..Default::default()
        })
        .await;

    let history = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("hi")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part::text("bye")],
            ..Default::default()
        },
    ];

    let mut stream_response = hi_prompt
        .stream(
            TestInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                messages: Some(history.clone()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    // We must drain the stream to allow the final response task to complete.
    while stream_response.stream.next().await.is_some() {
        // In a real application, you would process each chunk here.
    }

    let final_response = stream_response.response.await.unwrap().unwrap();
    let actual_messages = final_response.messages().unwrap();

    assert_eq!(
        to_value(&actual_messages).unwrap(),
        json!([
            {
                "role": "user",
                "content": [{"text": "hi"}]
            },
            {
                "role": "model",
                "content": [{"text": "bye"}]
            },
            {
                "role": "user",
                "content": [{"text": "hi Genkit"}]
            },
            {
                "role": "model",
                "content": [{"text": "Echo: hibyehi Genkit; config: {}"}]
            }
        ])
    );
}

#[rstest]
#[tokio::test]
/// 'calls dotprompt with default model with config'
async fn test_calls_prompt_with_default_model_and_config(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_explicit_model_with_config_test".to_string(),
            model: Some(Model::Name("echoModel".to_string())),
            prompt: Some("hi {{name}}".to_string()),
            config: Some(json!({ "temperature": 11 })),
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
/// 'rejects on invalid model'
async fn test_rejects_on_invalid_model(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_invalid_model_test".to_string(),
            model: Some(Model::Name("modelThatDoesNotExist".to_string())),
            prompt: Some("hi {{name}}".to_string()),
            ..Default::default()
        })
        .await;

    let result = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(
        err.to_string(),
        "INTERNAL: Model 'modelThatDoesNotExist' not found"
    );
}
