// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use genkit::{
    action::ActionStreamSender,
    genkit,
    model::{
        define_model, generate, generate_stream, CandidateData, DefineModelOptions,
        GenerateRequest, GenerateResponseChunkData, GenerateResponseData, MessageData, ModelAction,
        ModelInfo, ModelSupport, Part, Role,
    },
    output::{define_format, DefineFormatOptions, FormatDef, FormatParser, OutputStrategy},
    registry::Registry,
    Genkit,
};
use genkit_core::error::Result;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex as StdMutex};
use tokio_stream::StreamExt;

// This module encapsulates the test helpers.
mod helpers {
    use super::*;

    #[derive(Default, Clone)]
    pub struct LastRequest(pub Arc<StdMutex<Option<GenerateRequest>>>);

    // Defines and registers a simple model that echoes its input.
    pub fn define_echo_model(
        registry: &mut Registry,
        model_info: ModelInfo,
        last_request: LastRequest,
    ) {
        let model_action = define_model(
            DefineModelOptions {
                name: "echoModel".to_string(),
                info: model_info,

                ..Default::default()
            },
            move |req: GenerateRequest,
                  mut stream_sender: ActionStreamSender<GenerateResponseChunkData>| {
                let last_request = last_request.clone();
                let req_clone_for_response = req.clone();
                Box::pin(async move {
                    // Store the request for inspection
                    *last_request.0.lock().unwrap() = Some(req.clone());

                    if stream_sender.is_streaming() {
                        for i in (1..=3).rev() {
                            stream_sender
                                .send(GenerateResponseChunkData {
                                    content: vec![Part {
                                        text: Some(i.to_string()),
                                        ..Default::default()
                                    }],
                                    ..Default::default()
                                })
                                .await
                                .unwrap();
                        }
                    }

                    let text_to_echo = req
                        .messages
                        .iter()
                        .map(|m| {
                            m.content
                                .iter()
                                .filter_map(|p| p.text.as_ref())
                                .map(|s| s.as_str())
                                .collect::<Vec<&str>>()
                                .join(",")
                        })
                        .collect::<Vec<String>>()
                        .join(",");

                    let response_message = MessageData {
                        role: Role::Model,
                        content: vec![Part {
                            text: Some(format!("Echo: {}", text_to_echo)),
                            ..Default::default()
                        }],
                        ..Default::default()
                    };

                    Ok(GenerateResponseData {
                        candidates: vec![CandidateData {
                            index: 0,
                            finish_reason: Some(genkit::model::FinishReason::Stop),
                            message: response_message,
                            ..Default::default()
                        }],
                        request: Some(Box::new(req_clone_for_response)),
                        ..Default::default()
                    })
                })
            },
        );

        let action_name = model_action.name().to_string();
        registry.register_action(Arc::new(model_action)).unwrap();
    }

    pub fn define_banana_format(registry: &mut Registry) {
        let format_def = FormatDef {
            parser: Arc::new(BananaFormatParser),
            strategy: OutputStrategy::Json, // This doesn't really matter for this test
            instructions: None,
        };

        let format = define_format(
            DefineFormatOptions {
                name: "banana".to_string(),
                format: "banana".to_string(),
                constrained: Some(true),
            },
            Box::new(move |schema| {
                let instructions = schema.map(|_| "Output should be in banana format".to_string());
                Ok(FormatDef {
                    instructions,
                    ..format_def.clone()
                })
            }),
        );
        registry
            .register_format("banana", Arc::new(StdMutex::new(format)))
            .unwrap();
    }

    #[derive(Debug, Clone)]
    struct BananaFormatParser;

    impl FormatParser for BananaFormatParser {
        fn parse_chunk(&self, chunk: &GenerateResponseChunkData) -> Result<Value> {
            let text = chunk.content[0].text.as_deref().unwrap_or("");
            Ok(json!(format!("banana: {}", text)))
        }

        fn parse_message(&self, message: &MessageData) -> Result<Value> {
            let text = message.content[0].text.as_deref().unwrap_or("");
            Ok(json!(format!("banana: {}", text)))
        }
    }
}

struct TestSetup {
    ai: Genkit,
    last_request: helpers::LastRequest,
}

fn setup(supports: Option<ModelSupport>) -> TestSetup {
    let mut ai = genkit(Default::default());
    let last_request = helpers::LastRequest::default();
    helpers::define_echo_model(
        &mut ai.registry(),
        ModelInfo {
            supports,
            ..Default::default()
        },
        last_request.clone(),
    );
    helpers::define_banana_format(&mut ai.registry());
    TestSetup { ai, last_request }
}

#[tokio::test]
async fn native_constrained_generation() -> Result<()> {
    let setup = setup(Some(ModelSupport {
        constrained_output: Some(true),
        ..Default::default()
    }));

    let mut request = GenerateRequest::new("echoModel", "hi");
    request.output.format = Some("banana".to_string());

    // Non-streaming
    let response = generate(&setup.ai, request.clone()).await?;
    let output: String = serde_json::from_value(response.output.unwrap())?;
    assert_eq!(output, "banana: Echo: hi");

    let last_req = setup.last_request.0.lock().unwrap().clone().unwrap();
    assert_eq!(
        last_req.output.as_ref().unwrap().format.as_deref(),
        Some("banana")
    );
    assert_eq!(last_req.output.as_ref().unwrap().constrained, Some(true));

    // Streaming
    let (mut stream, response_handle) = generate_stream(&setup.ai, request).await?;
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let output: String = serde_json::from_value(chunk.output.unwrap())?;
        chunks.push(output);
    }
    let final_response = response_handle.await??;
    let final_output: String = serde_json::from_value(final_response.output.unwrap())?;

    assert_eq!(chunks, vec!["banana: 3", "banana: 2", "banana: 1"]);
    assert_eq!(final_output, "banana: Echo: hi");

    Ok(())
}

#[tokio::test]
async fn simulated_constrained_generation() -> Result<()> {
    let setup = setup(Some(ModelSupport {
        constrained_output: Some(false),
        ..Default::default()
    }));

    let mut request = GenerateRequest::new("echoModel", "hi");
    request.output.format = Some("banana".to_string());

    // Non-streaming
    let response = generate(&setup.ai, request.clone()).await?;
    let output: String = serde_json::from_value(response.output.unwrap())?;
    assert_eq!(output, "banana: Echo: hi");

    let last_req = setup.last_request.0.lock().unwrap().clone().unwrap();
    // In simulated mode, the format instructions are added to the prompt
    // and the output options are not passed to the model directly.
    assert!(last_req.output.is_none());
    assert_eq!(last_req.messages.last().unwrap().role, Role::User);
    // The format is constrained, so it should add instructions.
    let instruction_part = last_req.messages.last().unwrap().content.iter().find(|p| {
        p.metadata
            .as_ref()
            .map_or(false, |m| m.get("purpose") == Some(&json!("output")))
    });
    assert!(instruction_part.is_some());
    assert_eq!(
        instruction_part.unwrap().text.as_deref(),
        Some("Output should be in banana format")
    );

    // Streaming
    let (mut stream, response_handle) = generate_stream(&setup.ai, request).await?;
    let mut chunks = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let output: String = serde_json::from_value(chunk.output.unwrap())?;
        chunks.push(output);
    }
    let final_response = response_handle.await??;
    let final_output: String = serde_json::from_value(final_response.output.unwrap())?;

    assert_eq!(chunks, vec!["banana: 3", "banana: 2", "banana: 1"]);
    assert_eq!(final_output, "banana: Echo: hi");

    Ok(())
}

#[tokio::test]
async fn explicitly_override_format_options() -> Result<()> {
    let setup = setup(Some(ModelSupport {
        constrained_output: Some(true),
        ..Default::default()
    }));

    let mut request = GenerateRequest::new("echoModel", "hi");
    request.output.format = Some("banana".to_string());
    request.output.constrained = Some(false);
    let schema = json!({"type": "string"});
    request.output.schema = Some(schema.clone());

    generate(&setup.ai, request).await?;

    let last_req = setup.last_request.0.lock().unwrap().clone().unwrap();
    let output_opts = last_req.output.as_ref().unwrap();

    // Overridden options should be respected
    assert_eq!(output_opts.constrained, Some(false));
    assert_eq!(output_opts.schema.as_ref(), Some(&schema));
    // The format is still banana
    assert_eq!(output_opts.format.as_deref(), Some("banana"));

    // Because constrained is now false, instructions should be added to the prompt
    assert_eq!(last_req.messages.last().unwrap().role, Role::User);
    let instruction_part = last_req.messages.last().unwrap().content.iter().find(|p| {
        p.metadata
            .as_ref()
            .map_or(false, |m| m.get("purpose") == Some(&json!("output")))
    });
    assert!(instruction_part.is_some());
    assert_eq!(
        instruction_part.unwrap().text.as_deref(),
        Some("Output should be in banana format")
    );

    Ok(())
}
