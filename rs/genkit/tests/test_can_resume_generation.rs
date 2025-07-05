#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use genkit::{
    model::{FinishReason, Part, Role},
    Genkit, Model, ToolAction, ToolArgument, ToolConfig,
};
use genkit_ai::{
    generate::ResumeOptions,
    model::{CandidateData, GenerateResponseData},
    tool::{InterruptConfig, Resumable, ToolFnOptions},
    GenerateOptions, MessageData, ToolRequest,
};

use rstest::{fixture, rstest};
use serde_json::json;
use std::sync::Arc;

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

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
