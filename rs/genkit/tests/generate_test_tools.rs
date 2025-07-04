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
    Genkit, Model, ToolArgument, ToolConfig,
};
use genkit_ai::{
    dynamic_tool,
    model::{CandidateData, GenerateResponseData},
    GenerateOptions, MessageData, OutputOptions, ToolRequest,
};
use genkit_core::context::ActionContext;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

//
// Tools Tests
//

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
struct TestToolInput {}

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

#[rstest]
#[tokio::test]
async fn test_tools_call_the_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    genkit.define_tool(
        ToolConfig {
            name: "testTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(TestToolInput {}),
            output_schema: Some(json!("tool called")),
            metadata: None,
        },
        |_, _| async { Ok(json!("tool called")) },
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
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![ToolArgument::from("testTool")]),
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

#[rstest]
#[tokio::test]
async fn test_tools_call_the_tool_with_context() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    // Define a tool that stringifies its context.
    genkit.define_tool(
        ToolConfig {
            name: "testTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(TestToolInput {}),
            output_schema: Some(json!("")), // Output is a string
            metadata: None,
        },
        // The tool handler now uses the context.
        |_, ctx| async move {
            // Due to `serde(flatten)`, serializing the whole context object
            // will produce the desired JSON string.
            Ok(json!(serde_json::to_string(&ctx.context).unwrap()))
        },
    );

    // Set up the programmable model's behavior.
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

    // Create the context data.
    let mut context_map = HashMap::new();
    context_map.insert("something".to_string(), json!("extra"));

    // Call generate with context.
    let _response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![ToolArgument::from("testTool")]),
            context: Some(ActionContext {
                auth: None,
                additional_context: context_map,
            }),
            ..Default::default()
        })
        .await
        .unwrap();

    // Assert on the messages sent to the second model call.
    let last_request = pm_handle.last_request.lock().unwrap();
    let messages = &last_request.as_ref().unwrap().messages;

    // The TS test asserts on messages[2], which is the tool response.
    let tool_response_message = &messages[2];

    let expected_tool_message = MessageData {
        role: Role::Tool,
        content: vec![Part {
            tool_response: Some(genkit_ai::ToolResponse {
                name: "testTool".to_string(),
                ref_id: Some("ref123".to_string()),
                output: Some(json!("{\"something\":\"extra\"}")),
            }),
            ..Default::default()
        }],
        metadata: None,
    };

    assert_eq!(tool_response_message, &expected_tool_message);
}

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
struct DynamicToolInput {
    foo: String,
}

#[rstest]
#[tokio::test]
async fn test_calls_the_dynamic_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    let dynamic_test_tool_1 = dynamic_tool(
        ToolConfig {
            name: "dynamicTestTool1".to_string(),
            description: "description".to_string(),
            input_schema: Some(DynamicToolInput {
                foo: "".to_string(),
            }),
            output_schema: Some(json!("")),
            metadata: None,
        },
        |_, _| async { Ok(json!("tool called 1")) },
    );

    let dynamic_test_tool_2 = genkit.dynamic_tool(
        ToolConfig {
            name: "dynamicTestTool2".to_string(),
            description: "description 2".to_string(),
            input_schema: Some(DynamicToolInput {
                foo: "".to_string(),
            }),
            output_schema: Some(json!("")),
            metadata: None,
        },
        |_, _| async { Ok(json!("tool called 2")) },
    );

    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_, _| {
            let mut counter = req_counter.lock().unwrap();
            let response = if *counter == 0 {
                *counter += 1;
                GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![
                                Part {
                                    tool_request: Some(ToolRequest {
                                        name: "dynamicTestTool1".to_string(),
                                        input: Some(json!({"foo": "bar"})),
                                        ref_id: Some("ref123".to_string()),
                                    }),
                                    ..Default::default()
                                },
                                Part {
                                    tool_request: Some(ToolRequest {
                                        name: "dynamicTestTool2".to_string(),
                                        input: Some(json!({"foo": "baz"})),
                                        ref_id: Some("ref234".to_string()),
                                    }),
                                    ..Default::default()
                                },
                            ],
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
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![
                dynamic_test_tool_1.to_tool_argument(),
                dynamic_test_tool_2,
            ]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "done");

    let last_request = pm_handle.last_request.lock().unwrap();
    let messages = &last_request.as_ref().unwrap().messages;
    assert_eq!(messages.len(), 3);

    let tool_responses = &messages[2].content;
    assert_eq!(tool_responses.len(), 2);

    // Find and verify the first tool's response.
    let resp1 = tool_responses
        .iter()
        .find(|p| {
            p.tool_response
                .as_ref()
                .is_some_and(|tr| tr.name == "dynamicTestTool1")
        })
        .unwrap()
        .tool_response
        .as_ref()
        .unwrap();

    assert_eq!(resp1.ref_id, Some("ref123".to_string()));
    assert_eq!(resp1.output, Some(json!("tool called 1")));

    // Find and verify the second tool's response.
    let resp2 = tool_responses
        .iter()
        .find(|p| {
            p.tool_response
                .as_ref()
                .is_some_and(|tr| tr.name == "dynamicTestTool2")
        })
        .unwrap()
        .tool_response
        .as_ref()
        .unwrap();

    assert_eq!(resp2.ref_id, Some("ref234".to_string()));
    assert_eq!(resp2.output, Some(json!("tool called 2")));
}

#[rstest]
#[tokio::test]
async fn test_interrupts_the_dynamic_tool_with_no_impl() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    #[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
    struct InterruptToolInput {
        foo: String,
    }

    // Define a dynamic tool with a schema but no implementation by passing `None`
    // as the handler. This makes the tool's schema available to the model but
    // will cause an interrupt if the model requests to call it.
    let dynamic_test_tool = genkit.dynamic_tool(
        ToolConfig {
            name: "dynamicTestTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(InterruptToolInput {
                foo: "".to_string(),
            }),
            output_schema: None,
            metadata: None,
        },
        |_, _| async { Ok(()) },
    );

    // Configure the programmable model to respond with a tool request.
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_, _| {
            let mut counter = req_counter.lock().unwrap();
            let response = if *counter == 0 {
                *counter += 1;
                GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part {
                                tool_request: Some(ToolRequest {
                                    name: "dynamicTestTool".to_string(),
                                    input: Some(json!({ "foo": "bar" })),
                                    ref_id: Some("ref123".to_string()),
                                }),
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

    // Call generate and expect an interrupt instead of a final answer.
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![dynamic_test_tool]),
            ..Default::default()
        })
        .await
        .unwrap();

    // Verify that the response contains the expected tool request as an interrupt.
    let interrupts = response.interrupts().unwrap();
    assert_eq!(interrupts.len(), 1);
    let interrupt = interrupts[0];

    let mut metadata = HashMap::new();
    metadata.insert("interrupt".to_string(), json!(true));

    let expected_tool_request = json!({
        "name": "dynamicTestTool",
        "input": {"foo": "bar"},
        "ref_id": "ref123",
    });

    // TODO
    // assert_eq!(
    //     serde_json::to_value(interrupt.tool_request.as_ref().unwrap()).unwrap(),
    //     expected_tool_request
    // );
    // assert_eq!(interrupt.metadata.as_ref().unwrap(), &metadata);
}

#[rstest]
#[tokio::test]
async fn test_call_the_tool_with_output_schema() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    // Define a struct that will be used for schema validation.
    #[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone, PartialEq)]
    struct TestSchema {
        foo: String,
    }

    // Define a tool that uses the schema for its input and output.
    genkit.define_tool(
        ToolConfig {
            name: "testTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(TestSchema {
                foo: "".to_string(),
            }),
            output_schema: Some(TestSchema {
                foo: "".to_string(),
            }),
            metadata: None,
        },
        |_, _| async {
            Ok(TestSchema {
                foo: "bar".to_string(),
            })
        },
    );

    // Configure the programmable model's two-step response.
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_, _| {
            let mut counter = req_counter.lock().unwrap();
            let response = if *counter == 0 {
                // On the first call, request to use the tool.
                *counter += 1;
                GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part {
                                tool_request: Some(ToolRequest {
                                    name: "testTool".to_string(),
                                    input: Some(json!({"foo": "fromTool"})),
                                    ref_id: Some("ref123".to_string()),
                                }),
                                ..Default::default()
                            }],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                }
            } else {
                // On the second call, return the final text response,
                // which is a string containing JSON.
                GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text("```json\n{\"foo\": \"fromModel\"}\n```")],
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

    // Call generate with an output schema specified. This instructs the
    // framework to parse the model's final text response.
    let response = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![ToolArgument::from("testTool")]),
            output: Some(OutputOptions {
                schema: Some(serde_json::to_value(schemars::schema_for!(TestSchema)).unwrap()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();

    // Assert that the raw text and the parsed output are both correct.
    assert_eq!(
        response.text().unwrap(),
        "```json\n{\"foo\": \"fromModel\"}\n```"
    );

    let output_value: Value = response.output().unwrap();
    assert_eq!(output_value, json!({"foo": "fromModel"}));
}
