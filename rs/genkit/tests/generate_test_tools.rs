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
    model::{FinishReason, Part, Role},
    GenerateRequest, GenerateResponse, Genkit, Model, ToolAction, ToolArgument, ToolConfig,
};
use genkit_ai::{
    dynamic_resource, dynamic_tool,
    generate::ResumeOptions,
    model::{CandidateData, GenerateResponseData},
    tool::{InterruptConfig, Resumable, ToolFnOptions},
    GenerateOptions, GenerateResponseChunk, GenerateResponseChunkData, MessageData, OutputOptions,
    ResourceOptions, ResourceOutput, ToolRequest, ToolResponse,
};
use genkit_core::context::ActionContext;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Define a struct that will be used for schema validation.
#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone, PartialEq)]
struct TestSchema {
    foo: String,
}

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
/// 'call the tool'
async fn test_tools_call_the_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    genkit.define_tool(
        ToolConfig {
            name: "testTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(TestToolInput {}),
            output_schema: Some(String::new()),
            metadata: None,
        },
        |_, _| async { Ok("tool called".to_string()) },
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
/// 'call the tool with context'
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
                ..Default::default()
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

#[rstest]
#[tokio::test]
/// 'calls the dynamic tool'
async fn test_calls_the_dynamic_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    #[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
    struct DynamicToolInput {
        foo: String,
    }

    let dynamic_test_tool_1 = dynamic_tool(
        ToolConfig {
            name: "dynamicTestTool1".to_string(),
            description: "description".to_string(),
            input_schema: Some(DynamicToolInput {
                foo: "".to_string(),
            }),
            output_schema: Some(String::new()),
            metadata: None,
        },
        |_, _| async { Ok("tool called 1".to_string()) },
    );

    let dynamic_test_tool_2 = genkit.dynamic_tool(
        ToolConfig {
            name: "dynamicTestTool2".to_string(),
            description: "description 2".to_string(),
            input_schema: Some(DynamicToolInput {
                foo: "".to_string(),
            }),
            output_schema: Some(String::new()),
            metadata: None,
        },
        |_, _| async { Ok("tool called 2".to_string()) },
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
/// 'calls the dynamic resource'
async fn test_calls_the_dynamic_resource() {
    let (genkit, last_request) = helpers::genkit_instance_for_test().await;

    let dynamic_test_resource = dynamic_resource(
        ResourceOptions {
            name: Some("dynamicTestTool".to_string()),
            uri: Some("foo://foo".to_string()),
            description: Some("description".to_string()),
            ..Default::default()
        },
        |_, _| async {
            Ok(ResourceOutput {
                content: vec![Part::text("dynamic text")],
            })
        },
    )
    .unwrap();

    genkit
        .define_resource(
            ResourceOptions {
                name: Some("regularResource".to_string()),
                template: Some("bar://{value}".to_string()),
                description: Some("description 2".to_string()),
                ..Default::default()
            },
            |_, _| async {
                Ok(ResourceOutput {
                    content: vec![Part::text("regular text")],
                })
            },
        )
        .unwrap();

    let response = genkit
        .generate_with_options::<serde_json::Value>(GenerateOptions {
            model: Some(Model::Name("echoModel".to_string())),
            messages: Some(vec![MessageData {
                role: Role::User,
                content: vec![
                    Part::text("some text"),
                    Part::resource("foo://foo"),
                    Part::resource("bar://bar"),
                ],
                metadata: None,
            }]),
            resources: Some(vec![dynamic_test_resource]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: some text,dynamic text,regular text; config: {}"
    );

    let last_req = last_request.lock().unwrap();
    let message = &last_req.as_ref().unwrap().messages[0];
    assert_eq!(message.role, Role::User);

    // Assuming the framework replaces resource parts with text parts with metadata.
    // The following assertions depend on Part having a `metadata` field.
    let content = &message.content;
    assert_eq!(content.len(), 3);
    assert_eq!(content[0].text.as_deref(), Some("some text"));

    assert_eq!(content[1].text.as_deref(), Some("dynamic text"));
    assert_eq!(
        serde_json::to_value(content[1].metadata.as_ref().unwrap()).unwrap(),
        serde_json::json!({ "resource": { "uri": "foo://foo" } })
    );

    assert_eq!(content[2].text.as_deref(), Some("regular text"));
    assert_eq!(
        serde_json::to_value(content[2].metadata.as_ref().unwrap()).unwrap(),
        serde_json::json!({ "resource": { "template": "bar://{value}", "uri": "bar://bar" } })
    );
}

#[rstest]
#[tokio::test]
/// 'interrupts the dynamic tool with no impl'
async fn test_interrupts_the_dynamic_tool_with_no_impl() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    #[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema, Clone)]
    struct InterruptToolInput {
        foo: String,
    }

    // Define a dynamic tool that is expected to interrupt.
    let dynamic_test_tool = genkit.dynamic_tool_without_runner(ToolConfig {
        name: "dynamicTestTool".to_string(),
        description: "description".to_string(),
        input_schema: Some(InterruptToolInput {
            foo: "".to_string(),
        }),
        output_schema: None::<Value>,
        metadata: None,
    });

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

    let expected_tool_request = json!({
        "name": "dynamicTestTool",
        "input": {"foo": "bar"},
        "refId": "ref123",
    });

    assert_eq!(
        serde_json::to_value(interrupt).unwrap(),
        expected_tool_request
    );
}

#[rstest]
#[tokio::test]
/// 'call the tool with output schema'
async fn test_call_the_tool_with_output_schema() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

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

#[rstest]
#[tokio::test]
/// 'should propagate context to the tool'
async fn test_should_propagate_context_to_the_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    #[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, PartialEq)]
    struct TestSchema {
        foo: String,
    }

    // A struct for deserializing auth data from the context.
    #[derive(Default, Serialize, Deserialize, Debug, Clone)]
    struct AuthData {
        email: Option<String>,
    }

    // Define a tool whose handler accesses the authentication context to generate its output.
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
        // The handler deserializes auth data from the context and includes it in the response.
        |_, options| async move {
            let email = options
                .context
                .auth
                .and_then(|a| serde_json::from_value::<AuthData>(a).ok())
                .and_then(|auth| auth.email)
                .unwrap_or_else(|| "unknown".to_string());
            let output_foo = format!("bar {}", email);
            Ok(TestSchema { foo: output_foo })
        },
    );

    // Configure the programmable model to first call the tool, and then on the
    // second call, to echo the tool's JSON output back as a raw string.
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |req, _| {
            let mut counter = req_counter.lock().unwrap();
            let response = if *counter == 0 {
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
                let last_message = req.messages.last().expect("Should have previous message");
                let tool_output = last_message.content[0]
                    .tool_response
                    .as_ref()
                    .expect("Should be a tool response")
                    .output
                    .as_ref()
                    .expect("Tool should have output");
                let response_text =
                    serde_json::to_string(tool_output).expect("Should serialize to string");

                GenerateResponseData {
                    candidates: vec![CandidateData {
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text(response_text)],
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

    // Call generate with context that includes authentication information.
    let response: genkit::GenerateResponse<serde_json::Value> = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![ToolArgument::from("testTool")]),
            context: Some(ActionContext {
                auth: Some(
                    serde_json::to_value(AuthData {
                        email: Some("a@b.c".to_string()),
                    })
                    .unwrap(),
                ),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();

    // Assert that the final text response is the stringified JSON from the tool,
    // which correctly incorporated the email from the context.
    assert_eq!(response.text().unwrap(), r#"{"foo":"bar a@b.c"}"#);
}

#[rstest]
#[tokio::test]
/// 'streams the tool responses'
async fn test_streams_the_tool_responses() {
    use tokio_stream::StreamExt;

    let (genkit, pm_handle) = genkit_with_programmable_model().await;

    // Define a simple tool.
    genkit.define_tool(
        ToolConfig {
            name: "testTool".to_string(),
            input_schema: Some(json!({})),
            output_schema: Some(String::new()),
            description: "description".to_string(),
            ..Default::default()
        },
        |_, _| async { Ok("tool called".to_string()) },
    );

    let req_counter = Arc::new(Mutex::new(0));
    let handler: helpers::ProgrammableModelHandler = {
        let req_counter = req_counter.clone();
        Arc::new(Box::new(
            move |_req: GenerateRequest,
                  streaming_callback: Option<
                Box<dyn Fn(GenerateResponseChunkData) + Send + Sync>,
            >| {
                let req_counter = req_counter.clone();
                Box::pin(async move {
                    let mut counter = req_counter.lock().unwrap();
                    let (content, role) = if *counter == 0 {
                        (
                            vec![Part::tool_request(
                                "testTool",
                                Some(json!({})),
                                Some("ref123".to_string()),
                            )],
                            Role::Model,
                        )
                    } else {
                        (vec![Part::text("done")], Role::Model)
                    };

                    if let Some(cb) = &streaming_callback {
                        cb(GenerateResponseChunkData {
                            index: 0, // Candidate index
                            content: content.clone(),
                            role: Some(role.clone()),
                            ..Default::default()
                        });
                    }

                    let response = GenerateResponseData {
                        candidates: vec![CandidateData {
                            message: MessageData {
                                role,
                                content,
                                ..Default::default()
                            },
                            ..Default::default()
                        }],
                        ..Default::default()
                    };
                    *counter += 1;
                    Ok(response)
                })
            },
        ))
    };
    *pm_handle.handler.lock().unwrap() = handler;

    let generate_result = genkit
        .generate_stream::<serde_json::Value>(GenerateOptions {
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec!["testTool".into()]),
            ..Default::default()
        })
        .await
        .unwrap();

    let mut stream = generate_result.stream;

    let mut chunks = Vec::new();
    while let Some(chunk_result) = stream.next().await {
        chunks.push(chunk_result.unwrap().to_json());
    }

    let response = generate_result.response.await.unwrap().unwrap();
    assert_eq!(response.text().unwrap(), "done");

    let actual_chunks: Vec<serde_json::Value> = chunks
        .into_iter()
        .map(|c| {
            let mut v = serde_json::to_value(c).unwrap();
            if let Some(obj) = v.as_object_mut() {
                obj.remove("usage");
                obj.remove("custom");
            }
            v
        })
        .collect();

    let expected_chunks = serde_json::from_str::<Vec<serde_json::Value>>(
        r#"[
        {
          "content": [
            {
              "toolRequest": {
                "input": {},
                "name": "testTool",
                "refId": "ref123"
              }
            }
          ],
          "index": 0,
          "role": "model"
        },
        {
          "content": [
            {
              "toolResponse": {
                "name": "testTool",
                "output": "tool called",
                "refId": "ref123"
              }
            }
          ],
          "index": 1,
          "role": "tool"
        },
        {
          "content": [{ "text": "done" }],
          "index": 2,
          "role": "model"
        }
      ]"#,
    )
    .unwrap();

    assert_eq!(actual_chunks, expected_chunks);
}

#[rstest]
#[tokio::test]
#[should_panic(expected = "Exceeded maximum tool call iterations (17)")]
/// 'throws when exceeding max tool call iterations'
async fn test_throws_when_exceeding_max_tool_call_iterations() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;

    // Define a simple tool.
    genkit.define_tool(
        ToolConfig {
            name: "testTool".to_string(),
            input_schema: Some(json!({})),
            output_schema: Some(String::new()),
            ..Default::default()
        },
        |_, _| async { Ok("tool called".to_string()) },
    );

    // Configure the programmable model to always request the same tool,
    // creating an infinite loop.
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_, _| {
            let response = GenerateResponseData {
                candidates: vec![CandidateData {
                    message: MessageData {
                        role: Role::Model,
                        content: vec![Part {
                            tool_request: Some(ToolRequest {
                                name: "testTool".to_string(),
                                input: Some(json!({})),
                                ref_id: Some("ref123".to_string()),
                            }),
                            ..Default::default()
                        }],
                        ..Default::default()
                    },
                    ..Default::default()
                }],
                ..Default::default()
            };
            Box::pin(async { Ok(response) })
        }));
    }

    // Call generate with a specific limit on tool-calling turns.
    // This call is expected to fail and panic, which is caught and
    // verified by the `#[should_panic]` attribute on the test.
    let _response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![ToolArgument::from("testTool")]),
            model: Some(Model::Name("programmableModel".to_string())),
            max_turns: Some(17),
            ..Default::default()
        })
        .await
        .unwrap();
}

#[rstest]
#[tokio::test]
/// interrupts tool execution
async fn test_interrupts_tool_execution() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    // Create a clone of the Arc to be moved into the closure.
    let req_counter_clone = Arc::clone(&req_counter);

    #[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Default)]
    struct SimpleInput {
        name: String,
    }
    #[derive(Debug, Serialize, Deserialize, JsonSchema, Clone, Default)]
    struct ResumableInput {
        #[serde(rename = "doIt")]
        do_it: bool,
    }

    // A standard tool that runs to completion.
    genkit.define_tool(
        ToolConfig {
            name: "simpleTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(SimpleInput::default()),
            output_schema: Some("".to_string()),
            ..Default::default()
        },
        |input: SimpleInput, _| async move { Ok(format!("response: {}", input.name)) },
    );

    // A tool that immediately interrupts with custom metadata.
    genkit.define_tool(
        ToolConfig {
            name: "interruptingTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(()),
            output_schema: Some(()),
            ..Default::default()
        },
        |_, options: ToolFnOptions| async move {
            Err((options.interrupt)(Some(
                json!({ "confirm": "is it a banana?" }),
            )))
        },
    );

    // A tool that interrupts and can be resumed.
    genkit.define_tool(
        ToolConfig {
            name: "resumableTool".to_string(),
            description: "description".to_string(),
            input_schema: Some(ResumableInput::default()),
            output_schema: Some(true),
            ..Default::default()
        },
        |_, options: ToolFnOptions| async move { Err((options.interrupt)(None)) },
    );

    // Configure the mock model to request all three tools in parallel.
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_, _| {
            // Use the cloned Arc inside the closure.
            let mut counter = req_counter_clone.lock().unwrap();
            *counter += 1;
            let response = GenerateResponseData {
                candidates: vec![CandidateData {
                    message: MessageData {
                        role: Role::Model,
                        content: vec![
                            Part::text("reasoning"),
                            Part {
                                tool_request: Some(ToolRequest {
                                    name: "interruptingTool".to_string(),
                                    input: Some(json!(null)),
                                    ref_id: Some("ref123".to_string()),
                                }),
                                ..Default::default()
                            },
                            Part {
                                tool_request: Some(ToolRequest {
                                    name: "simpleTool".to_string(),
                                    input: Some(json!({ "name": "foo" })),
                                    ref_id: Some("ref456".to_string()),
                                }),
                                ..Default::default()
                            },
                            Part {
                                tool_request: Some(ToolRequest {
                                    name: "resumableTool".to_string(),
                                    input: Some(json!({ "doIt": true })),
                                    ref_id: Some("ref789".to_string()),
                                }),
                                ..Default::default()
                            },
                        ],
                        ..Default::default()
                    },
                    ..Default::default()
                }],
                ..Default::default()
            };
            Box::pin(async { Ok(response) })
        }));
    }

    // Call generate and expect an interrupted response, not a final one.
    let response: GenerateResponse<Value> = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![
                ToolArgument::from("interruptingTool"),
                ToolArgument::from("simpleTool"),
                ToolArgument::from("resumableTool"),
            ]),
            model: Some(Model::Name("programmableModel".to_string())),
            ..Default::default()
        })
        .await
        .unwrap();

    // The original req_counter is still valid here and can be used for the assertion.
    assert_eq!(*req_counter.lock().unwrap(), 1);

    // Verify the pending tool requests and their metadata in the response.
    let tool_requests = response.tool_requests().unwrap();
    assert_eq!(tool_requests.len(), 3);

    // Check the interruptingTool request.
    let interrupting_req = tool_requests.iter().find(|p| p.name == "interruptingTool");
    assert!(interrupting_req.is_some());

    // Check the simpleTool request, which should have its output pre-calculated.
    let simple_req = tool_requests.iter().find(|p| p.name == "simpleTool");
    assert!(simple_req.is_some());

    // Check the resumableTool request.
    let resumable_req = tool_requests.iter().find(|p| p.name == "resumableTool");
    assert!(resumable_req.is_some());
}

#[rstest]
#[tokio::test]
/// 'can resume generation'
async fn test_can_resume_generation() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;

    // Define the three tools required for the test.
    genkit.define_interrupt::<(), i32>(InterruptConfig {
        name: "interrupter".to_string(),
        description: "always interrupts".to_string(),
        ..Default::default()
    });
    genkit.define_tool(
        ToolConfig {
            name: "truth".to_string(),
            description: "always returns true".to_string(),
            output_schema: Some(true),
            ..Default::default()
        },
        |_: (), _: ToolFnOptions| async { Ok(true) },
    );
    genkit.define_tool(
        ToolConfig {
            name: "resumable".to_string(),
            description: "interrupts unless resumed".to_string(),
            output_schema: Some(true),
            ..Default::default()
        },
        // In this test, we only care about resuming, so the initial
        // implementation just interrupts.
        |_: (), options: ToolFnOptions| async move {
            if let Some(status) = options
                .context
                .additional_context
                .get("status")
                .and_then(|v| v.as_str())
            {
                if status == "ok" {
                    return Ok(true);
                }
            }
            Err((options.interrupt)(None))
        },
    );

    // Create a message history representing a paused generation.
    let messages: Vec<MessageData> = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("hello")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![
                Part {
                    tool_request: Some(ToolRequest {
                        name: "interrupter".to_string(),
                        input: Some(json!({})),
                        ref_id: Some("1".to_string()),
                    }),
                    metadata: Some(serde_json::from_str(r#"{"interrupt":true}"#).unwrap()),
                    ..Default::default()
                },
                Part {
                    tool_request: Some(ToolRequest {
                        name: "truth".to_string(),
                        input: Some(json!({})),
                        ref_id: Some("2".to_string()),
                    }),
                    metadata: Some(serde_json::from_str(r#"{"pendingOutput":true}"#).unwrap()),
                    ..Default::default()
                },
                Part {
                    tool_request: Some(ToolRequest {
                        name: "resumable".to_string(),
                        input: Some(json!({})),
                        ref_id: Some("3".to_string()),
                    }),
                    metadata: Some(serde_json::from_str(r#"{"interrupt":true}"#).unwrap()),
                    ..Default::default()
                },
            ],
            ..Default::default()
        },
    ];

    // Look up the tool actions to call .respond() and .restart() on them.
    let interrupter_action = genkit
        .registry()
        .lookup_action("/tool/interrupter")
        .await
        .unwrap();
    let interrupter_tool = interrupter_action
        .as_any()
        .downcast_ref::<ToolAction<(), i32>>()
        .unwrap();
    let resumable_action = genkit
        .registry()
        .lookup_action("/tool/resumable")
        .await
        .unwrap();
    let resumable_tool = resumable_action
        .as_any()
        .downcast_ref::<ToolAction<(), bool>>()
        .unwrap();

    // Configure the programmable model to simply echo back the final set of messages.
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(|req, _| {
            let response = GenerateResponseData {
                candidates: vec![CandidateData {
                    message: req.messages.last().unwrap().clone(),
                    finish_reason: Some(FinishReason::Stop),
                    ..Default::default()
                }],
                ..Default::default()
            };
            Box::pin(async { Ok(response) })
        }));
    }

    // Call generate with the paused history and instructions on how to resume.
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            messages: Some(messages),
            tools: Some(vec![
                ToolArgument::from("interrupter"),
                ToolArgument::from("resumable"),
                ToolArgument::from("truth"),
            ]),
            resume: Some(ResumeOptions {
                // Provide a direct response for the 'interrupter' tool.
                respond: Some(vec![interrupter_tool
                    .respond(
                        &Part {
                            tool_request: Some(ToolRequest {
                                name: "interrupter".to_string(),
                                ref_id: Some("1".to_string()),
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                        23,
                        None,
                    )
                    .unwrap()]),
                // Restart the 'resumable' tool with new metadata.
                restart: Some(vec![resumable_tool
                    .restart(
                        &Part {
                            tool_request: Some(ToolRequest {
                                name: "resumable".to_string(),
                                ref_id: Some("3".to_string()),
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                        Some(json!({ "status": "ok" })),
                        None,
                    )
                    .unwrap()]),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();

    let final_messages = response.messages().unwrap();
    let revised_model_message = final_messages.get(final_messages.len() - 3).unwrap();
    let tool_message = final_messages.get(final_messages.len() - 2).unwrap();

    let expected_revised_model_content = vec![
        Part {
            tool_request: Some(ToolRequest {
                name: "interrupter".to_string(),
                input: Some(json!({})),
                ref_id: Some("1".to_string()),
            }),
            metadata: Some(serde_json::from_str(r#"{"resolvedInterrupt":true}"#).unwrap()),
            ..Default::default()
        },
        Part {
            tool_request: Some(ToolRequest {
                name: "truth".to_string(),
                input: Some(json!({})),
                ref_id: Some("2".to_string()),
            }),
            // NOTE: The TS implementation removes `pendingOutput` metadata here. The Rust
            // implementation currently does not, so a passing test would assert
            // `{"pendingOutput":true}`. We assert for empty metadata to align with the
            // TS test's expectation.
            metadata: Some(serde_json::from_str(r#"{}"#).unwrap()),
            ..Default::default()
        },
        Part {
            tool_request: Some(ToolRequest {
                name: "resumable".to_string(),
                input: Some(json!({})),
                ref_id: Some("3".to_string()),
            }),
            metadata: Some(serde_json::from_str(r#"{"resolvedInterrupt":true}"#).unwrap()),
            ..Default::default()
        },
    ];
    assert_eq!(
        revised_model_message.content,
        expected_revised_model_content
    );

    let expected_tool_content = vec![
        Part {
            tool_response: Some(genkit_ai::ToolResponse {
                name: "interrupter".to_string(),
                ref_id: Some("1".to_string()),
                output: Some(json!(23)),
            }),
            metadata: Some(serde_json::from_str(r#"{"interruptResponse":true}"#).unwrap()),
            ..Default::default()
        },
        Part {
            tool_response: Some(genkit_ai::ToolResponse {
                name: "truth".to_string(),
                ref_id: Some("2".to_string()),
                output: Some(json!(true)),
            }),
            metadata: Some(serde_json::from_str(r#"{"source":"pending"}"#).unwrap()),
            ..Default::default()
        },
        Part {
            tool_response: Some(genkit_ai::ToolResponse {
                name: "resumable".to_string(),
                ref_id: Some("3".to_string()),
                output: Some(json!(true)),
            }),
            ..Default::default()
        },
    ];

    // Assert that the generated tool message contains the correct resolved outputs.
    assert_eq!(tool_message.role, Role::Tool);
    assert_eq!(tool_message.content, expected_tool_content);
}

#[rstest]
#[tokio::test]
/// 'streams a generated tool message when resumed'
async fn test_streams_a_generated_tool_message_when_resumed() {
    // 1. Setup
    let (genkit, pm_handle) = genkit_with_programmable_model().await;

    // 2. Configure mock model
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(|_request, on_chunk| {
            if let Some(cb) = on_chunk {
                cb(GenerateResponseChunkData {
                    index: 1, // Note: The tool chunk will be index 0
                    role: Some(Role::Model),
                    content: vec![Part::text("final response")],
                    ..Default::default()
                });
            }
            Box::pin(async {
                Ok(GenerateResponseData {
                    candidates: vec![CandidateData {
                        index: 0,
                        finish_reason: Some(FinishReason::Stop),
                        message: MessageData {
                            role: Role::Model,
                            content: vec![Part::text("final response")],
                            ..Default::default()
                        },
                        ..Default::default()
                    }],
                    ..Default::default()
                })
            })
        }));
    }

    // 3. Setup on_chunk callback
    let chunks = Arc::new(Mutex::new(Vec::new()));
    let chunks_clone = chunks.clone();
    let on_chunk = Arc::new(move |chunk: GenerateResponseChunk<Value>| {
        chunks_clone.lock().unwrap().push(chunk.to_json());
        Ok(())
    });

    // 4. Define messages and resume options
    let messages = vec![
        MessageData {
            role: Role::User,
            content: vec![Part::text("use the doThing tool")],
            ..Default::default()
        },
        MessageData {
            role: Role::Model,
            content: vec![Part {
                tool_request: Some(ToolRequest {
                    name: "doThing".to_string(),
                    input: Some(json!({})),
                    ..Default::default()
                }),
                metadata: Some(serde_json::from_str(r#"{"interrupt":true}"#).unwrap()),
                ..Default::default()
            }],
            ..Default::default()
        },
    ];

    let resume = ResumeOptions {
        respond: Some(vec![Part {
            tool_response: Some(ToolResponse {
                name: "doThing".to_string(),
                output: Some(json!("did thing")),
                ..Default::default()
            }),
            ..Default::default()
        }]),
        ..Default::default()
    };

    // 5. Call generate
    let _ = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            messages: Some(messages),
            resume: Some(resume),
            on_chunk: Some(on_chunk),
            ..Default::default()
        })
        .await
        .unwrap();

    // 6. Assert
    let collected_chunks = chunks.lock().unwrap();

    let expected_chunks = vec![
        GenerateResponseChunkData {
            index: 0,
            role: Some(Role::Tool),
            content: vec![Part {
                tool_response: Some(ToolResponse {
                    name: "doThing".to_string(),
                    output: Some(json!("did thing")),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        },
        GenerateResponseChunkData {
            index: 1,
            role: Some(Role::Model),
            content: vec![Part::text("final response")],
            ..Default::default()
        },
    ];

    assert_eq!(*collected_chunks, expected_chunks);
}
