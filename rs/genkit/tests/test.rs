#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use genkit::{
    model::{FinishReason, Part, Role},
    Genkit, Model, ToolArgument, ToolConfig,
};
use genkit_ai::{
    generate::ResumeOptions,
    model::{CandidateData, GenerateResponseData},
    tool::{InterruptConfig, Resumable, ToolFnOptions},
    GenerateOptions, MessageData, ToolRequest,
};

use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::{Arc, Mutex};

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
    let dynamic_test_tool_1 = genkit.dynamic_tool(
        ToolConfig {
            name: "dynamicTestTool1".to_string(),
            description: "description".to_string(),
            input_schema: Some(DynamicToolInput {
                foo: "".to_string(),
            }),
            output_schema: Some(String::new()),
            metadata: None,
        },
        // The handler for the first tool.
        |input: DynamicToolInput, _| async move {
            assert_eq!(input.foo, "bar");
            Ok("tool called 1".to_string())
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
            output_schema: Some(String::new()),
            metadata: None,
        },
        // The handler for the second tool.
        |input: DynamicToolInput, _| async move {
            assert_eq!(input.foo, "baz");
            Ok("tool called 2".to_string())
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

use genkit_ai::tool::ToolAction;

#[rstest]
#[tokio::test]
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
            if options.context.additional_context.contains_key("restart") {
                Ok(true)
            } else {
                Err((options.interrupt)(None))
            }
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
                    )
                    .unwrap()]),
                ..Default::default()
            }),
            ..Default::default()
        })
        .await
        .unwrap();

    let final_messages = response.messages().unwrap();
    let tool_message = final_messages.get(final_messages.len() - 2).unwrap();

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
