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

#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use genkit::{
    model::{FinishReason, Part, Role},
    Genkit, Model,
};
use genkit_ai::{
    self as genkit_ai,
    model::{CandidateData, GenerateRequest, GenerateResponseData},
    GenerateOptions, MessageData,
};

use rstest::{fixture, rstest};
use serde_json::json;
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

//
// Default Model Tests
//

#[rstest]
#[tokio::test]
async fn test_calls_default_model(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("short and sweet")]),
            config: Some(json!({ "temperature": 0.5 })),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: short and sweet; config: {\"temperature\":0.5}"
    );
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_with_string_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("short and sweet")]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: short and sweet; config: null"
    );
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_with_parts_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("short and sweet")]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: short and sweet; config: null"
    );
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_system_message(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let messages = vec![
        MessageData {
            role: Role::System,
            content: vec![Part {
                text: Some("You are a cat.".to_string()),
                ..Default::default()
            }],
            metadata: None,
        },
        MessageData {
            role: Role::System,
            content: vec![Part {
                text: Some("What is the meaning of life?".to_string()),
                ..Default::default()
            }],
            metadata: None,
        },
    ];
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            messages: Some(messages),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: SYSTEM INSTRUCTIONS:\nYou are a cat.Understood.system: What is the meaning of life?; config: null"
    );
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_with_tool_choice(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, last_request) = genkit_instance_for_test.await;
    let config = json!({ "tool_choice": "myTool" });
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            config: Some(config.clone()),
            prompt: Some(vec![Part::text("test")]),
            ..Default::default()
        })
        .await
        .unwrap();

    let locked_request = last_request.lock().unwrap();
    let request_config = locked_request.as_ref().unwrap().config.clone();
    assert_eq!(request_config, Some(config.clone()));

    let expected_response = format!("Echo: test; config: {}", config);
    assert_eq!(response.text().unwrap(), expected_response);
}

#[rstest]
#[tokio::test]
async fn test_uses_the_default_model(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, helpers::ProgrammableModel),
) {
    let (genkit, pm) = genkit_with_programmable_model.await;

    let test_prompt = "tell me a joke".to_string();
    {
        let mut handler_mutex_guard = pm.handler.lock().unwrap();
        *handler_mutex_guard = Arc::new(Box::new(move |_req, _| {
            Box::pin(async move {
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        index: 0,
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text("mock response".to_string())],
                            metadata: None,
                        },
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        }));
    }

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text(test_prompt.clone())]),
            ..Default::default()
        })
        .await
        .unwrap();

    let last_request = pm.last_request.lock().unwrap();
    assert!(last_request.is_some());
    let last_message = last_request.as_ref().unwrap().messages.last().unwrap();
    assert_eq!(last_message.text(), test_prompt);
    assert_eq!(response.text().unwrap(), "mock response");
}
