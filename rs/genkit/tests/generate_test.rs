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
    Genkit,
};
use genkit_ai::{
    self as genkit_ai,
    model::{CandidateData, GenerateRequest, GenerateResponseData},
    GenerateOptions, GenerateResponseChunkData, MessageData,
};

use rstest::{fixture, rstest};
use std::sync::{Arc, Mutex};
use tokio_stream::StreamExt;

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

#[fixture]
async fn genkit_with_programmable_model() -> (Arc<Genkit>, helpers::ProgrammableModel) {
    helpers::genkit_with_programmable_model().await
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
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("hi")]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi; config: null");
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_with_string_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("hi")]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi; config: null");
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_with_parts_prompt(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("hi")]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi; config: null");
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_system_message(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, last_request) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            system: Some(vec![Part::text("talk like a pirate")]),
            prompt: Some(vec![Part::text("hi")]),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(
        response.text().unwrap(),
        "Echo: system: talk like a pirate,hi; config: null"
    );

    let locked_request = last_request.lock().unwrap();
    let messages = &locked_request.as_ref().unwrap().messages;
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, Role::System);
    assert_eq!(messages[0].text(), "talk like a pirate");
    assert_eq!(messages[1].role, Role::User);
    assert_eq!(messages[1].text(), "hi");
}

#[rstest]
#[tokio::test]
async fn test_calls_default_model_with_tool_choice(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, last_request) = genkit_instance_for_test.await;
    let response: genkit::GenerateResponse = genkit
        .generate_with_options(GenerateOptions {
            prompt: Some(vec![Part::text("hi")]),
            tool_choice: Some(genkit_ai::ToolChoice::Required),
            ..Default::default()
        })
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "Echo: hi; config: null");

    let locked_request = last_request.lock().unwrap();
    let request_tool_choice = locked_request.as_ref().unwrap().tool_choice.clone();
    assert_eq!(
        request_tool_choice,
        Some(format!("{:?}", genkit_ai::ToolChoice::Required))
    );
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

    let mut response_container: genkit::GenerateStreamResponse = genkit
        .generate_stream(GenerateOptions {
            prompt: Some(vec![Part::text("unused".to_string())]),
            ..Default::default()
        })
        .await
        .unwrap();

    let mut chunks = Vec::new();
    while let Some(chunk_result) = response_container.stream.next().await {
        chunks.push(chunk_result.unwrap());
    }

    assert_eq!(chunks.len(), 3, "Should have received 3 chunks");
    assert_eq!(chunks[0].text(), "chunk1");
    assert_eq!(chunks[1].text(), "chunk2");
    assert_eq!(chunks[2].text(), "chunk3");

    let final_response = response_container
        .response
        .await
        .expect("Tokio task should not panic")
        .expect("Generation process should complete successfully");

    let output_text = final_response
        .text()
        .expect("Final response should contain text");

    assert_eq!(output_text, "chunk1chunk2chunk3");
}
