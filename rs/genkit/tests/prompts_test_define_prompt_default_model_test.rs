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
mod prompts_helpers;

use genkit::{
    prompt::{PromptConfig, PromptGenerateOptions},
    Genkit,
};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::prompts_helpers::{wrap_request, wrap_response};

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
            name: "hi_default_model_test".to_string(),
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
/// 'calls dotprompt with default model with config'
async fn test_calls_prompt_with_default_model_and_config(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_default_model_config_test".to_string(),
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
/// 'calls dotprompt with default model via retrieved prompt'
async fn test_calls_prompt_with_default_model_via_retrieved_prompt(
    #[future] genkit_instance: Arc<Genkit>,
) {
    let genkit = genkit_instance.await;

    let prompt_config = PromptConfig {
        name: "hi_retrieved".to_string(),
        prompt: Some("hi {{name}}".to_string()),
        ..Default::default()
    };
    genkit
        .define_prompt::<TestInput, Value, Value>(prompt_config)
        .await;

    let hi_prompt =
        genkit_ai::prompt::prompt::<TestInput, Value, Value>(genkit.registry(), "hi_retrieved")
            .await
            .unwrap();

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                r#use: Some(vec![wrap_request(), wrap_response()]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit); config: {}]");
}

#[rstest]
#[tokio::test]
/// 'should apply middleware to a prompt call'
async fn test_should_apply_middleware_to_a_prompt_call(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_middleware_test".to_string(),
            prompt: Some("hi {{name}}".to_string()),
            ..Default::default()
        })
        .await;

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                r#use: Some(vec![wrap_request(), wrap_response()]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit); config: {}]");
}

#[rstest]
#[tokio::test]
/// 'should apply middleware configured on a prompt'
async fn test_should_apply_middleware_configured_on_a_prompt(
    #[future] genkit_instance: Arc<Genkit>,
) {
    let genkit = genkit_instance.await;

    let hi_prompt = genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_configured_middleware_test".to_string(),
            prompt: Some("hi {{name}}".to_string()),
            r#use: Some(vec![wrap_request(), wrap_response()]),
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

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit); config: {}]");
}

#[rstest]
#[tokio::test]
/// 'should apply middleware to a prompt call on a looked up prompt'
async fn test_should_apply_middleware_to_looked_up_prompt(#[future] genkit_instance: Arc<Genkit>) {
    let genkit = genkit_instance.await;

    genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_lookup_with_middleware".to_string(),
            prompt: Some("hi {{name}}".to_string()),
            r#use: Some(vec![wrap_request(), wrap_response()]),
            ..Default::default()
        })
        .await;

    let hi_prompt = genkit_ai::prompt::prompt::<TestInput, Value, Value>(
        genkit.registry(),
        "hi_lookup_with_middleware",
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

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit); config: {}]");
}

#[rstest]
#[tokio::test]
/// 'should apply middleware to a prompt call on a looked up prompt with options'
async fn test_should_apply_middleware_to_looked_up_prompt_with_options(
    #[future] genkit_instance: Arc<Genkit>,
) {
    let genkit = genkit_instance.await;

    genkit
        .define_prompt::<TestInput, Value, Value>(PromptConfig {
            name: "hi_lookup_options_middleware".to_string(),
            prompt: Some("hi {{name}}".to_string()),
            ..Default::default()
        })
        .await;

    let hi_prompt = genkit_ai::prompt::prompt::<TestInput, Value, Value>(
        genkit.registry(),
        "hi_lookup_options_middleware",
    )
    .await
    .unwrap();

    let response = hi_prompt
        .generate(
            TestInput {
                name: "Genkit".to_string(),
            },
            Some(PromptGenerateOptions {
                r#use: Some(vec![wrap_request(), wrap_response()]),
                ..Default::default()
            }),
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit); config: {}]");
}
