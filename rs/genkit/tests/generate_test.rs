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
use futures_util::StreamExt;
use genkit::{
    error::{Error, Result},
    model::{FinishReason, Part, Role},
    plugin::Plugin,
    registry::Registry,
    Genkit, GenkitOptions, Model, ToolConfig,
};
use genkit_ai::{
    self as genkit_ai, // Alias to prevent ambiguity with genkit::define_model
    define_model,
    model::{
        CandidateData, DefineModelOptions, GenerateRequest, GenerateResponseChunkData,
        GenerateResponseData,
    },
    GenerateOptions,
    MessageData,
    ToolRequest,
};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde_json::json;
use std::sync::{Arc, Mutex};

use helpers::StreamingCallback;

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
        .generate(GenerateOptions {
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

#[rstest]
#[tokio::test]
async fn test_calls_default_model_with_parts_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
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
        .generate(GenerateOptions {
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
        .generate(GenerateOptions {
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
async fn test_streams_default_model() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(|_, on_chunk| {
            if let Some(cb) = on_chunk.as_ref() {
                for i in 0..3 {
                    cb(GenerateResponseChunkData {
                        index: i,
                        content: vec![Part::text(format!("chunk{}", i + 1))],
                        ..Default::default()
                    });
                }
            }
            Box::pin(async {
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        index: 0,
                        finish_reason: Some(FinishReason::Stop),
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text("chunk1chunk2chunk3")],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        }));
    }

    let mut response_container: genkit::GenerateStreamResponse =
        genkit.generate_stream(GenerateOptions {
            prompt: Some(vec![Part::text("unused".to_string())]),
            ..Default::default()
        });

    let mut chunks = Vec::new();
    while let Some(chunk_result) = response_container.stream.next().await {
        chunks.push(chunk_result.unwrap());
    }

    assert_eq!(chunks.len(), 3, "Should have received 3 chunks");
    assert_eq!(chunks[0].text(), "3");
    assert_eq!(chunks[1].text(), "2");
    assert_eq!(chunks[2].text(), "1");

    let final_response = response_container
        .response
        .await
        .expect("Tokio task should not panic")
        .expect("Generation process should complete successfully");

    let output_text = final_response
        .text()
        .expect("Final response should contain text");

    assert!(
        output_text.starts_with("Echo: unused; config: null"),
        "Final response text did not match expected output"
    );
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
        .generate(GenerateOptions {
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
        .generate(GenerateOptions {
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
    let response: genkit::GenerateResponse = genkit
        .generate(GenerateOptions {
            model,
            prompt: Some(vec![Part::text("hi".to_string())]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(!response.is_valid());
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
            |_, _| async { Err(Error::new_internal("foo")) },
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

    let model = Some(Model::Name("blockingModel".to_string()));
    let stream_response: genkit::GenerateStreamResponse = genkit.generate_stream(GenerateOptions {
        model,
        prompt: Some(vec![Part::text("short and sweet".to_string())]),
        ..Default::default()
    });

    assert!(stream_response.response.await.unwrap().is_err());
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

    let model = Some(Model::Name("errorModel".to_string()));
    let response: genkit::GenerateResponse = genkit
        .generate(GenerateOptions {
            model,
            prompt: Some(vec![Part::text("hi".to_string())]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(!response.is_valid());
}

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    let pm_plugin = Arc::new(helpers::ProgrammableModelPlugin::new());
    let genkit = Genkit::init(GenkitOptions {
        plugins: vec![pm_plugin.clone() as Arc<dyn Plugin>],
        ..Default::default()
    })
    .await
    .unwrap();
    let handle = pm_plugin.get_handle();
    (genkit, handle)
}

#[rstest]
#[tokio::test]
async fn test_streaming_passes_the_streaming_callback_to_the_model(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, helpers::ProgrammableModel),
) {
    let (genkit, pm_handle) = genkit_with_programmable_model.await;

    let was_called = Arc::new(Mutex::new(false));
    let was_called_clone = was_called.clone();

    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(
            move |_, streaming_callback: Option<StreamingCallback>| {
                let was_called_clone_2 = was_called_clone.clone();
                Box::pin(async move {
                    if streaming_callback.is_some() {
                        *was_called_clone_2.lock().unwrap() = true;
                    }
                    Ok(Default::default())
                })
            },
        ));
    }

    let stream_resp: genkit::GenerateStreamResponse = genkit.generate_stream(GenerateOptions {
        model: Some(Model::Name("error-model".to_string())),
        prompt: Some(vec![Part::text("test")]),
        ..Default::default()
    });
    let _ = stream_resp.response.await;

    assert!(*was_called.lock().unwrap());
}

#[rstest]
#[tokio::test]
async fn test_streaming_strips_out_noop_streaming_callback(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, helpers::ProgrammableModel),
) {
    let (genkit, pm_handle) = genkit_with_programmable_model.await;

    let was_called = Arc::new(Mutex::new(false));
    let was_called_clone = was_called.clone();

    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(
            move |_, streaming_callback: Option<StreamingCallback>| {
                let was_called_clone_2 = was_called_clone.clone();
                Box::pin(async move {
                    if streaming_callback.is_some() {
                        *was_called_clone_2.lock().unwrap() = true;
                    }
                    Ok(Default::default())
                })
            },
        ));
    }

    let _: genkit::GenerateResponse = genkit
        .generate(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(*was_called.lock().unwrap());
}

//
// Config Tests
//

#[rstest]
#[tokio::test]
async fn test_config_takes_config_passed_to_generate(
    #[future] genkit_with_programmable_model: (Arc<Genkit>, helpers::ProgrammableModel),
) {
    let (genkit, pm_handle) = genkit_with_programmable_model.await;

    let _: genkit::GenerateResponse = genkit
        .generate(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            config: Some(json!({"temperature": 0.9})),
            ..Default::default()
        })
        .await
        .unwrap();

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
        let config_schema = self.config.clone();
        let versions = vec![self.version.clone()];
        define_model(
            registry,
            DefineModelOptions {
                name: "configurableModel".to_string(),
                versions: Some(versions),
                config_schema: Some(config_schema),
                ..Default::default()
            },
            |req, _| async move {
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text(serde_json::to_string(&req.config).unwrap())],
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

    let response: genkit::GenerateResponse = genkit
        .generate(GenerateOptions {
            model: Some(Model::Name("configurableModel".to_string())),
            config: Some(json!({ "b": 2, "c": 3 })),
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
            "b": 2,
            "c": 3
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
    let response: genkit::GenerateResponse = genkit
        .generate(GenerateOptions {
            model: Some(Model::Name("configurableModel".to_string())),
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

//
// Tools Tests
//

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
struct TestToolInput {}

#[rstest]
#[tokio::test]
async fn test_tools_call_the_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    let test_tool = genkit::dynamic_tool(
        ToolConfig {
            name: "testTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(TestToolInput {}),
            output_schema: Some(serde_json::json!("tool called")),
            metadata: None,
        },
        |_, _| async { Ok(serde_json::json!("tool called")) },
    );

    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_, _| {
            let mut counter = req_counter.lock().unwrap();
            let response = if *counter == 0 {
                *counter += 1;

                let tool_request = Some(ToolRequest {
                    name: "testTool".to_string(),
                    input: Some(serde_json::to_value(TestToolInput {}).unwrap()),
                    ref_id: Some("ref123".to_string()),
                });

                GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part {
                                tool_request,
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                }
            } else {
                GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text("done")],
                            ..Default::default()
                        },
                        finish_reason: Some(FinishReason::Stop),
                        ..Default::default()
                    }],
                    ..Default::default()
                }
            };
            Box::pin(async { Ok(response) })
        }));
    }

    let response: genkit::GenerateResponse = genkit
        .generate(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![test_tool]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "done");

    let last_request = pm_handle.last_request.lock().unwrap();
    let messages = &last_request.as_ref().unwrap().messages;
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, Role::User);
    assert_eq!(messages[0].text(), "call the tool");
    assert_eq!(messages[1].role, Role::Model);
    assert!(messages[1].content[0].tool_request.is_some());
    assert_eq!(messages[2].role, Role::Tool);
    let tool_response = messages[2].content[0].tool_response.as_ref().unwrap();
    assert_eq!(tool_response.name, "testTool");
    assert_eq!(tool_response.ref_id, Some("ref123".to_string()));
    assert_eq!(
        tool_response.output,
        Some(serde_json::to_value("tool called").unwrap())
    );
}
