#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use genkit::{
    model::{Part, Role},
    GenerateResponse, Genkit, Model, ToolArgument, ToolConfig,
};
use genkit_ai::{
    model::{CandidateData, GenerateResponseData},
    tool::ToolFnOptions,
    GenerateOptions, MessageData, ToolRequest,
};

use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

#[rstest]
#[tokio::test]
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
                                    input: Some(json!({})),
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
