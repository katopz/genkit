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
    Genkit, Model,
};
use genkit_ai::{
    self as genkit_ai,
    generate::ResumeOptions,
    model::{CandidateData, GenerateResponseData},
    GenerateOptions, GenerateResponseChunk, GenerateResponseChunkData, MessageData, ToolRequest,
    ToolResponse,
};

use rstest::{fixture, rstest};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
}

#[rstest]
#[tokio::test]
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
