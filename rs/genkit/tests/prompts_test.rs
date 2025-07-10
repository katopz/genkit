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

use genkit::Genkit;
use genkit_ai::{
    define_prompt,
    message::{MessageData, Role},
    model::GenerateRequest,
    prompt::{PromptConfig, PromptGenerateOptions},
    Part,
};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Default)]
struct HiInput {
    name: String,
}

#[rstest]
#[tokio::test]
async fn test_should_apply_middleware_to_a_prompt_call(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;

    let prompt_config = PromptConfig {
        name: "hi".to_string(),
        messages_fn: Some(Arc::new(|input: HiInput, _, _| {
            Box::pin(async move {
                Ok(vec![MessageData {
                    role: Role::User,
                    content: vec![Part::text(format!("hi {}", input.name))],
                    ..Default::default()
                }])
            })
        })),
        ..Default::default()
    };
    let prompt = define_prompt::<HiInput, Value, Value>(genkit.registry(), prompt_config);

    let opts = PromptGenerateOptions {
        r#use: Some(vec![
            prompts_helpers::wrap_request(),
            prompts_helpers::wrap_response(),
        ]),
        ..Default::default()
    };

    let response = prompt
        .generate(
            HiInput {
                name: "Genkit".to_string(),
            },
            Some(opts),
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit),; config: {}]");
}

#[rstest]
#[tokio::test]
async fn test_should_apply_middleware_configured_on_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;

    let prompt_config = PromptConfig {
        name: "hi_with_middleware".to_string(),
        r#use: Some(vec![
            prompts_helpers::wrap_request(),
            prompts_helpers::wrap_response(),
        ]),
        messages_fn: Some(Arc::new(|input: HiInput, _, _| {
            Box::pin(async move {
                Ok(vec![MessageData {
                    role: Role::User,
                    content: vec![Part::text(format!("hi {}", input.name))],
                    ..Default::default()
                }])
            })
        })),
        ..Default::default()
    };
    let prompt = define_prompt::<HiInput, Value, Value>(genkit.registry(), prompt_config);

    let response = prompt
        .generate(
            HiInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit),; config: {}]");
}

#[rstest]
#[tokio::test]
async fn test_should_apply_middleware_to_a_looked_up_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;

    let prompt_config = PromptConfig {
        name: "hi_lookup".to_string(),
        r#use: Some(vec![
            prompts_helpers::wrap_request(),
            prompts_helpers::wrap_response(),
        ]),
        messages_fn: Some(Arc::new(|input: HiInput, _, _| {
            Box::pin(async move {
                Ok(vec![MessageData {
                    role: Role::User,
                    content: vec![Part::text(format!("hi {}", input.name))],
                    ..Default::default()
                }])
            })
        })),
        ..Default::default()
    };
    define_prompt::<HiInput, Value, Value>(genkit.registry(), prompt_config);

    let looked_up_prompt =
        genkit_ai::prompt::prompt::<HiInput, Value, Value>(genkit.registry(), "hi_lookup", None)
            .await
            .unwrap();

    let response = looked_up_prompt
        .generate(
            HiInput {
                name: "Genkit".to_string(),
            },
            None,
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit),; config: {}]");
}

#[rstest]
#[tokio::test]
async fn test_should_apply_middleware_to_a_prompt_call_on_a_looked_up_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;

    let prompt_config = PromptConfig {
        name: "hi_lookup_with_call_middleware".to_string(),
        messages_fn: Some(Arc::new(|input: HiInput, _, _| {
            Box::pin(async move {
                Ok(vec![MessageData {
                    role: Role::User,
                    content: vec![Part::text(format!("hi {}", input.name))],
                    ..Default::default()
                }])
            })
        })),
        ..Default::default()
    };
    define_prompt::<HiInput, Value, Value>(genkit.registry(), prompt_config);

    let looked_up_prompt = genkit_ai::prompt::prompt::<HiInput, Value, Value>(
        genkit.registry(),
        "hi_lookup_with_call_middleware",
        None,
    )
    .await
    .unwrap();

    let opts = PromptGenerateOptions {
        r#use: Some(vec![
            prompts_helpers::wrap_request(),
            prompts_helpers::wrap_response(),
        ]),
        ..Default::default()
    };

    let response = looked_up_prompt
        .generate(
            HiInput {
                name: "Genkit".to_string(),
            },
            Some(opts),
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit),; config: {}]");
}
