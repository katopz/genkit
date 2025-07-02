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

// TODO:
// - tools tests
// - long running tests

#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use async_trait::async_trait;
use genkit::{
    error::{Error, Result},
    model::{FinishReason, Message, Part, Role},
    plugin::Plugin,
    registry::Registry,
    Genkit, GenkitOptions,
};
use genkit_ai::{
    define_model,
    model::{
        CandidateData, DefineModelOptions, GenerateRequest, GenerateResponseChunkData,
        GenerateResponseData, ModelInfoSupports,
    },
    GenerateOptions, MessageData,
};
use serde_json::json;
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

/// Creates a Genkit instance with a model that mimics the behavior of the TS echoModel.
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

//
// Default Model Tests
//

#[tokio::test]
async fn test_calls_default_model() {
    let (genkit, _) = genkit_instance_for_test().await;
    let response = genkit
        .generate(GenerateOptions {
            prompt: Some(vec![Part::text("short and sweet".to_string())]),
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

#[tokio::test]
async fn test_calls_default_model_with_string_prompt() {
    let (genkit, _) = genkit_instance_for_test().await;
    let response = genkit
        .generate(GenerateOptions {
            prompt: Some(vec![Part::text("short and sweet".to_string())]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: short and sweet; config: null"
    );
}

#[tokio::test]
async fn test_calls_default_model_with_parts_prompt() {
    let (genkit, _) = genkit_instance_for_test().await;
    let response = genkit
        .generate(GenerateOptions {
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

#[tokio::test]
async fn test_calls_default_model_system_message() {
    let (genkit, _) = genkit_instance_for_test().await;
    let messages = vec![
        Message::system_text("You are a cat."),
        Message::user_text("What is the meaning of life?"),
    ];
    let response = genkit
        .generate(GenerateOptions {
            messages,
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: system: You are a cat.What is the meaning of life?; config: null"
    );
}

#[tokio::test]
async fn test_calls_default_model_with_tool_choice() {
    let (genkit, last_request) = genkit_instance_for_test().await;
    let config = json!({ "tool_choice": "myTool" });
    let response = genkit
        .generate(GenerateOptions {
            config: Some(config.clone()),
            ..Default::default()
        })
        .await
        .unwrap();

    let locked_request = last_request.lock().unwrap();
    let request_config = locked_request.as_ref().unwrap().config.clone();
    assert_eq!(request_config, Some(config));

    let expected_response = format!("Echo: ; config: {}", config.to_string());
    assert_eq!(response.text().unwrap(), expected_response);
}

#[tokio::test]
async fn test_streams_default_model() {
    let (genkit, _) = genkit_instance_for_test().await;
    let mut chunks = Vec::new();
    let response = genkit
        .generate_stream(
            GenerateOptions {
                prompt: Some(vec![Part::text("unused".to_string())]),
                ..Default::default()
            },
            |chunk| {
                chunks.push(chunk);
            },
        )
        .await
        .unwrap();

    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].text(), "3");
    assert_eq!(chunks[1].text(), "2");
    assert_eq!(chunks[2].text(), "1");

    assert!(response
        .text()
        .unwrap()
        .starts_with("Echo: unused; config: null"));
}

#[tokio::test]
async fn test_uses_the_default_model() {
    let pm_plugin = Arc::new(helpers::ProgrammableModelPlugin::new());
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![pm_plugin.clone() as Arc<dyn Plugin>],
        default_model: Some("programmableModel".to_string()),
        ..Default::default()
    })
    .await
    .unwrap();
    let pm = pm_plugin.get_handle();

    // Set a handler for the programmable model to capture the request
    let test_prompt = "tell me a joke".to_string();
    let mut handler_mutex_guard = pm.handler.lock().unwrap();
    *handler_mutex_guard = Arc::new(Box::new(move |req, _| {
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
    drop(handler_mutex_guard); // Explicitly drop to release the lock

    let response = genkit
        .generate(GenerateOptions {
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

#[tokio::test]
async fn test_explicit_model_calls_the_explicitly_passed_in_model() {
    let (genkit, _) = genkit_instance_for_test().await;
    let model = genkit.model("echoModel").unwrap();
    let response = model
        .generate(GenerateOptions {
            prompt: Some(vec![Part::text("short and sweet".to_string())]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: short and sweet; config: null"
    );
}

#[tokio::test]
async fn test_explicit_model_rejects_on_invalid_model() {
    let (genkit, _) = genkit_instance_for_test().await;
    let model_result = genkit.model("nonExistentModel");
    assert!(model_result.is_err());
    assert!(model_result
        .unwrap_err()
        .to_string()
        .contains("Model not found"));
}

//
// Streaming Tests
//

struct ErrorModelPlugin;
#[async_trait]
impl Plugin for ErrorModelPlugin {
    fn name(&self) -> &'static str {
        "errorModelPlugin"
    }
    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        define_model(
            registry,
            DefineModelOptions {
                name: "errorModel".to_string(),
                ..Default::default()
            },
            |_, _| async { Err(Error::from("foo")) },
        );
        Ok(())
    }
}

#[tokio::test]
async fn test_streaming_rethrows_response_errors() {
    let error_plugin = Arc::new(ErrorModelPlugin);
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![error_plugin as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();

    let model = genkit.model("errorModel").unwrap();
    let result = model.generate_stream(Default::default(), |_| {}).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("foo"));
}

// In TS, initialization can be async and throw. In Rust, we test a model that
// always errors on execution, which is equivalent to the TS test.
#[tokio::test]
async fn test_streaming_rethrows_initialization_errors() {
    let error_plugin = Arc::new(ErrorModelPlugin);
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![error_plugin as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();

    let model = genkit.model("errorModel").unwrap();
    let result = model.generate(Default::default()).await;

    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("foo"));
}

async fn genkit_with_programmable_model() -> (Arc<Genkit>, Arc<helpers::ProgrammableModel>) {
    let programmable_model = Arc::new(helpers::ProgrammableModel::default());

    struct TempPlugin {
        pm: Arc<helpers::ProgrammableModel>,
    }

    #[async_trait]
    impl Plugin for TempPlugin {
        fn name(&self) -> &'static str {
            "programmablePlugin"
        }

        async fn initialize(&self, registry: &mut Registry) -> Result<()> {
            let pm_clone = self.pm.clone();
            define_model(
                DefineModelOptions {
                    name: "programmableModel".to_string(),
                    supports: Some(ModelInfoSupports {
                        tools: Some(true),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                move |req, cb| {
                    let pm_clone2 = pm_clone.clone();
                    async move {
                        *pm_clone2.last_request.lock().unwrap() = Some(req.clone());
                        let handler = pm_clone2.handler.lock().unwrap();
                        let cb_wrapper = cb.map(|c| Box::new(c) as helpers::StreamingCallback);
                        handler(req, cb_wrapper).await
                    }
                },
            );
            Ok(())
        }
    }

    let plugin = Arc::new(TempPlugin {
        pm: programmable_model.clone(),
    });

    let genkit = Genkit::init(&GenkitOptions {
        plugins: vec![plugin as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();

    (genkit, programmable_model)
}

#[tokio::test]
async fn test_streaming_passes_the_streaming_callback_to_the_model() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let model_ref = genkit.model("programmableModel").unwrap();

    let was_called = Arc::new(Mutex::new(false));
    let was_called_clone = was_called.clone();

    let mut handler = pm_handle.handler.lock().unwrap();
    *handler = Box::new(move |_, streaming_callback| {
        let was_called_clone_2 = was_called_clone.clone();
        Box::pin(async move {
            if streaming_callback.is_some() {
                *was_called_clone_2.lock().unwrap() = true;
            }
            Ok(Default::default())
        })
    });
    drop(handler);

    let _ = model_ref.generate_stream(Default::default(), |_| {}).await;

    assert!(*was_called.lock().unwrap());
}

#[tokio::test]
async fn test_streaming_strips_out_noop_streaming_callback() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let model_ref = genkit.model("programmableModel").unwrap();

    let was_called = Arc::new(Mutex::new(false));
    let was_called_clone = was_called.clone();

    let mut handler = pm_handle.handler.lock().unwrap();
    *handler = Box::new(move |_, streaming_callback| {
        let was_called_clone_2 = was_called_clone.clone();
        Box::pin(async move {
            if streaming_callback.is_none() {
                *was_called_clone_2.lock().unwrap() = true;
            }
            Ok(Default::default())
        })
    });
    drop(handler);

    let _ = model_ref.generate(Default::default()).await;

    assert!(*was_called.lock().unwrap());
}

//
// Config Tests
//

#[tokio::test]
async fn test_config_takes_config_passed_to_generate() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let model_ref = genkit.model("programmableModel").unwrap();

    let _ = model_ref
        .generate(GenerateOptions {
            config: Some(json!({"temperature": 0.9})),
            ..Default::default()
        })
        .await;

    let last_req = pm_handle.last_request.lock().unwrap();
    assert_eq!(
        last_req.as_ref().unwrap().config,
        Some(json!({"temperature": 0.9}))
    );
}

struct ConfigurableModelPlugin {
    config: serde_json::Value,
    version: String,
}
#[async_trait]
impl Plugin for ConfigurableModelPlugin {
    fn name(&self) -> &'static str {
        "configurableModelPlugin"
    }
    async fn initialize(&self, registry: &mut Registry) -> Result<()> {
        define_model(
            registry,
            DefineModelOptions {
                name: "configurableModel".to_string(),
                versions: Some(vec![self.version.clone()]),
                config_schema: Some(self.config.clone()),
                ..Default::default()
            },
            |req, _| async {
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text(serde_json::to_string(&req.config)?)],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            },
        );
        Ok(())
    }
}

#[tokio::test]
async fn test_config_merges_config_from_the_ref() {
    let plugin = Arc::new(ConfigurableModelPlugin {
        config: json!({ "a": 1, "b": 1 }),
        version: "test".to_string(),
    });
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![plugin as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();

    let model = genkit.model("configurableModel").unwrap();
    let response = model
        .generate(GenerateOptions {
            config: Some(json!({ "b": 2, "c": 3 })),
            ..Default::default()
        })
        .await
        .unwrap();

    let response_config: serde_json::Value =
        serde_json::from_str(&response.text().unwrap()).unwrap();

    // The core framework should merge the configs.
    assert_eq!(
        response_config,
        json!({
            "a": 1,
            "b": 2,
            "c": 3,
            "version": "test"
        })
    );
}

#[tokio::test]
async fn test_config_picks_up_top_level_version_from_the_ref() {
    let plugin = Arc::new(ConfigurableModelPlugin {
        config: json!({}),
        version: "test-version".to_string(),
    });
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![plugin as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();
    let model = genkit.model("configurableModel").unwrap();
    let response = model
        .generate(GenerateOptions {
            config: Some(json!({ "a": 1 })),
            ..Default::default()
        })
        .await
        .unwrap();

    let response_config: serde_json::Value =
        serde_json::from_str(&response.text().unwrap()).unwrap();
    assert_eq!(
        response_config,
        json!({
            "a": 1,
            "version": "test-version"
        })
    );
}
