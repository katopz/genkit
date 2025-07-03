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

use async_trait::async_trait;
use genkit::{
    error::Result,
    model::{FinishReason, Part, Role},
    plugin::Plugin,
    registry::Registry,
    Genkit, GenkitOptions, Model,
};
use genkit_ai::{
    self as genkit_ai, define_model,
    model::{
        CandidateData, DefineModelOptions, GenerateRequest, GenerateResponse,
        GenerateResponseChunkData, GenerateResponseData,
    },
    GenerateOptions, MessageData, ModelRef,
};

use rstest::{fixture, rstest};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

// A plugin that defines a model behaving like the TypeScript `echoModel` for testing.
// This is defined locally in the test to allow for inspecting the last request.
struct TsEchoModelPlugin {
    last_request: Arc<Mutex<Option<GenerateRequest>>>,
}

#[async_trait]
impl Plugin for TsEchoModelPlugin {
    fn name(&self) -> &'static str {
        "tsEchoModelPlugin"
    }

    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        let last_request_clone = self.last_request.clone();
        define_model(
            registry,
            DefineModelOptions {
                name: "echoModel".to_string(),
                label: Some("TS-Compatible Echo Model".to_string()),
                ..Default::default()
            },
            move |req: GenerateRequest, streaming_callback| {
                let last_request_clone = last_request_clone.clone();
                async move {
                    // Store the last request for inspection.
                    let mut last_req = last_request_clone.lock().unwrap();
                    *last_req = Some(req.clone());

                    // Handle streaming by sending down a countdown.
                    if let Some(cb) = streaming_callback {
                        for i in (1..=3).rev() {
                            cb(GenerateResponseChunkData {
                                index: 0,
                                content: vec![genkit::model::Part::text(i.to_string())],
                                ..Default::default()
                            });
                        }
                    }

                    // Construct the response text by concatenating messages, similar to the TS version.
                    let concatenated_messages = req
                        .messages
                        .iter()
                        .map(|m| {
                            let prefix = match m.role {
                                Role::User | Role::Model => "".to_string(),
                                _ => format!("{}: ", m.role.to_string().to_lowercase()),
                            };
                            let content = m
                                .content
                                .iter()
                                .filter_map(|p| p.text.clone())
                                .collect::<Vec<_>>()
                                .join("");
                            format!("{}{}", prefix, content)
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    let config_str =
                        serde_json::to_string(&req.config).unwrap_or_else(|_| "null".to_string());

                    let response_text_part = Part::text(format!("Echo: {}", concatenated_messages));

                    let config_part = Part::text(format!("; config: {}", config_str));

                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
                            message: MessageData {
                                role: Role::Model,
                                content: vec![response_text_part, config_part],
                                ..Default::default()
                            },
                            finish_reason: Some(FinishReason::Stop),
                            ..Default::default()
                        }],
                        ..Default::default()
                    })
                }
            },
        );
        Ok(())
    }
}

/// Fixture that creates a Genkit instance with a model that mimics the behavior of the TS echoModel.
#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    let last_request = Arc::new(Mutex::new(None));
    let echo_plugin = Arc::new(TsEchoModelPlugin {
        last_request: last_request.clone(),
    });

    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![echo_plugin as Arc<dyn Plugin>],
        default_model: Some("echoModel".to_string()),
        ..Default::default()
    })
    .await
    .unwrap();
    (genkit, last_request)
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

//
// Explicit Model Tests
//

#[rstest]
#[tokio::test]
async fn test_explicit_model_calls_the_explicitly_passed_in_model(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("echoModel".to_string())),
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
async fn test_explicit_model_rejects_on_invalid_model() {
    let (genkit, _) = genkit_instance_for_test().await;
    let model = Some(Model::Name("modelThatDoesNotExist".to_string()));

    let result = genkit
        .generate_with_options(GenerateOptions::<GenerateResponse> {
            model,
            prompt: Some(vec![Part::text("hi".to_string())]),
            ..Default::default()
        })
        .await;

    // Assert that the operation returned an error
    assert!(result.is_err());

    // Optionally, inspect the error to be more specific
    let err = result.unwrap_err();
    assert!(err
        .to_string()
        .contains("Model 'modelThatDoesNotExist' not found"));
}

//
// Config Tests
//

#[rstest]
#[tokio::test]
async fn test_config_takes_config_passed_to_generate() {
    let (genkit, _) = genkit_instance_for_test().await;

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(
                ModelRef::new(json!({ "name": "echoModel" }))
                    .with_version("bcd")
                    .into(),
            ),
            prompt: Some(vec![Part::text("hi")]),
            config: Some(json!({
                "temperature": 11
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    let left_str = response.text().unwrap();
    let right_str = r#"Echo: hi; config: {"version":"bcd","temperature":11}"#;
    let prefix = "Echo: hi; config: ";

    // Extract the JSON part from each string
    let left_json_str = left_str.strip_prefix(prefix).expect("Prefix not found");
    let right_json_str = right_str.strip_prefix(prefix).expect("Prefix not found");

    // Parse them into serde_json::Value
    let left_value: Value = serde_json::from_str(left_json_str).expect("Failed to parse left JSON");
    let right_value: Value =
        serde_json::from_str(right_json_str).expect("Failed to parse right JSON");

    // Assert the JSON values are equal, ignoring key order
    assert_eq!(left_value, right_value);
}

#[rstest]
#[tokio::test]
async fn test_config_merges_config_from_the_ref() {
    let (genkit, _) = genkit_instance_for_test().await;

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(
                ModelRef::new(json!({
                    "name": "echoModel"
                }))
                .with_config(json!({
                    "version": "abc"
                }))
                .into(),
            ),
            prompt: Some(vec![Part::text("hi")]),
            config: Some(json!({
                "temperature": 11
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    let left_str = response.text().unwrap();
    let right_str = r#"Echo: hi; config: {"version":"abc","temperature":11}"#;
    let prefix = "Echo: hi; config: ";

    // 1. Extract the JSON part from each string
    println!("left_str:{}", left_str);
    let left_json_str = left_str.strip_prefix(prefix).expect("Prefix not found");
    let right_json_str = right_str.strip_prefix(prefix).expect("Prefix not found");

    // 2. Parse them into serde_json::Value
    let left_value: Value = serde_json::from_str(left_json_str).expect("Failed to parse left JSON");
    let right_value: Value =
        serde_json::from_str(right_json_str).expect("Failed to parse right JSON");

    // 3. Assert the JSON values are equal
    assert_eq!(left_value, right_value);
}

#[rstest]
#[tokio::test]
async fn test_config_picks_up_top_level_version_from_the_ref() {
    let (genkit, _) = genkit_instance_for_test().await;

    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(
                ModelRef::new(json!({
                    "name": "echoModel"
                }))
                .with_config(json!({
                    "version": "abc"
                }))
                .into(),
            ),
            prompt: Some(vec![Part::text("hi")]),
            config: Some(json!({
                "temperature": 11
            })),
            ..Default::default()
        })
        .await
        .unwrap();

    let left_str = response.text().unwrap();
    let right_str = r#"Echo: hi; config: {"version":"abc","temperature":11}"#;
    let prefix = "Echo: hi; config: ";

    // 1. Extract the JSON part from each string
    println!("left_str:{}", left_str);
    let left_json_str = left_str.strip_prefix(prefix).expect("Prefix not found");
    let right_json_str = right_str.strip_prefix(prefix).expect("Prefix not found");

    // 2. Parse them into serde_json::Value
    let left_value: Value = serde_json::from_str(left_json_str).expect("Failed to parse left JSON");
    let right_value: Value =
        serde_json::from_str(right_json_str).expect("Failed to parse right JSON");

    // 3. Assert the JSON values are equal
    assert_eq!(left_value, right_value);
}
