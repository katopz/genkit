#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use genkit::{
    model::{FinishReason, Part, Role},
    Genkit, Model, ToolArgument, ToolConfig,
};
use genkit_ai::{
    model::{CandidateData, GenerateResponseData},
    GenerateOptions, MessageData, ToolRequest,
};

use genkit_core::context::ActionContext;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
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
async fn test_calls_the_dynamic_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    // Define the schema for our dynamic tools' input.
    #[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
    struct DynamicToolInput {
        foo: String,
    }

    // Create the first dynamic tool.
    // The `dynamic_tool` function creates a tool definition with its handler,
    // and `.to_tool_argument()` makes it suitable for passing to generate_with_options.
    let dynamic_test_tool_1 = genkit.dynamic_tool(
        ToolConfig {
            name: "dynamicTestTool1".to_string(),
            description: "description".to_string(),
            input_schema: Some(DynamicToolInput {
                foo: "".to_string(),
            }),
            output_schema: Some(json!("")), // Expects a string output
            metadata: None,
        },
        // The handler for the first tool.
        |input: DynamicToolInput, _| async move {
            assert_eq!(input.foo, "bar");
            Ok(json!("tool called 1"))
        },
    );

    // Create the second dynamic tool.
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
        // The handler for the second tool.
        |input: DynamicToolInput, _| async move {
            assert_eq!(input.foo, "baz");
            Ok(json!("tool called 2"))
        },
    );

    // Configure the programmable model to simulate a two-step conversation.
    {
        let mut handler = pm_handle.handler.lock().unwrap();
        *handler = Arc::new(Box::new(move |_, _| {
            let mut counter = req_counter.lock().unwrap();
            let response = if *counter == 0 {
                // On the first call, request both dynamic tools.
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
                // On the second call, provide the final text response.
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

    // Call generate with the dynamic tools.
    let response: genkit::GenerateResponse<serde_json::Value> = genkit
        .generate_with_options(GenerateOptions {
            model: Some(Model::Name("programmableModel".to_string())),
            prompt: Some(vec![Part::text("call the tool")]),
            tools: Some(vec![dynamic_test_tool_1, dynamic_test_tool_2]),
            ..Default::default()
        })
        .await
        .unwrap();

    // Assert the final text is correct.
    assert_eq!(response.text().unwrap(), "done");

    // Assert that the message history sent to the model is correct.
    let last_request = pm_handle.last_request.lock().unwrap();
    let messages = &last_request.as_ref().unwrap().messages;
    assert_eq!(
        messages.len(),
        3,
        "Should be 3 messages in history: user, model (tool_call), tool (tool_response)"
    );

    // Verify the tool response message.
    let tool_response_msg = &messages[2];
    assert_eq!(tool_response_msg.role, Role::Tool);
    let tool_responses = &tool_response_msg.content;
    assert_eq!(tool_responses.len(), 2);

    // Find and verify the first tool's response.
    let resp1 = tool_responses
        .iter()
        .find(|p| {
            p.tool_response
                .as_ref()
                .is_some_and(|tr| tr.name == "dynamicTestTool1")
        })
        .and_then(|p| p.tool_response.as_ref())
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
        .and_then(|p| p.tool_response.as_ref())
        .unwrap();

    assert_eq!(resp2.ref_id, Some("ref234".to_string()));
    assert_eq!(resp2.output, Some(json!("tool called 2")));
}

#[rstest]
#[tokio::test]
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
