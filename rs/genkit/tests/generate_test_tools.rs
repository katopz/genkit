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
    Genkit, Model, ToolConfig,
};
use genkit_ai::{
    model::{CandidateData, GenerateResponseData},
    GenerateOptions, MessageData, ToolRequest,
};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
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
async fn test_tools_call_the_tool() {
    let (genkit, pm_handle) = genkit_with_programmable_model().await;
    let req_counter = Arc::new(Mutex::new(0));

    let test_tool = genkit::dynamic_tool(
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
