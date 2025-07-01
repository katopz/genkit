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

use futures_util::StreamExt;
use genkit::model::{Candidate, GenerateRequest, Message, Role};

use genkit::{FinishReason, Part};
use genkit_ai::formats::types::FormatHandler;
use genkit_ai::generate::{generate, generate_stream, OutputOptions};
use genkit_ai::model::DefineModelOptions;
use genkit_ai::{
    define_model, GenerateOptions, GenerateResponseChunkData, GenerateResponseData, MessageData,
};
use genkit_core::registry::Registry;
use rstest::*;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;

struct BananaFormat;
impl genkit_ai::formats::Format for BananaFormat {
    fn parse_message(&self, message: &Message) -> Value {
        Value::String(format!("banana: {}", message.text()))
    }
    fn parse_chunk(&self, chunk: &genkit_ai::generate::GenerateResponseChunk) -> Option<Value> {
        Some(Value::String(format!("banana: {}", chunk.text())))
    }
    fn instructions(&self) -> Option<String> {
        Some("Output should be in banana format".to_string())
    }
}

// Helper to define an echo model for testing.
fn define_echo_model(registry: &mut Registry, constrained_support: &str) {
    let mut metadata = HashMap::new();
    let supports = json!({ "constrained": constrained_support });
    metadata.insert("supports".to_string(), supports);
    let model_opts = DefineModelOptions {
        name: "echoModel".to_string(),
        label: Some("Echo Model".to_string()),

        ..Default::default()
    };
    define_model(
        registry,
        model_opts,
        |req: GenerateRequest, streaming_callback| async move {
            let last_msg_text = req.messages.last().cloned().unwrap_or_default();

            // If a streaming callback is provided, we send down the countdown chunks.
            if let Some(cb) = streaming_callback {
                let chunks_data = vec![
                    GenerateResponseChunkData {
                        index: 0,
                        content: vec![genkit::model::Part::text("3")],
                        ..Default::default()
                    },
                    GenerateResponseChunkData {
                        index: 0,
                        content: vec![genkit::model::Part::text("2")],
                        ..Default::default()
                    },
                    GenerateResponseChunkData {
                        index: 0,
                        content: vec![genkit::model::Part::text("1")],
                        ..Default::default()
                    },
                ];

                for data in chunks_data {
                    cb(data)
                }
            }

            // Both streaming and non-streaming calls return a final response.
            let text = format!("Echo: {:?}", last_msg_text);
            Ok(GenerateResponseData {
                candidates: vec![Candidate {
                    index: 0,
                    finish_reason: Some(FinishReason::Stop),
                    message: MessageData {
                        role: Role::Model,
                        content: vec![Part::text(text)],
                        metadata: None,
                    },
                    ..Default::default()
                }],
                ..Default::default()
            })
        },
    );
}

#[fixture]
fn registry() -> Registry {
    let mut registry = Registry::new();
    genkit_ai::configure_ai(&mut registry);
    let banana_handler: FormatHandler = Arc::new(|_schema| Box::new(BananaFormat));
    let banana_config = genkit_ai::formats::FormatterConfig {
        constrained: Some(true),
        ..Default::default()
    };
    genkit_ai::formats::define_format(
        &mut registry,
        "banana",
        banana_config.clone(),
        banana_handler.clone(),
    );
    registry
}

#[rstest]
#[tokio::test]
async fn test_custom_format_native_constrained(mut registry: Registry) {
    define_echo_model(&mut registry, "all");

    let response = generate(
        &registry,
        GenerateOptions::<String> {
            model: Some(genkit::model::Model::Name("echoModel".to_string())),
            prompt: Some(vec![genkit::model::Part::text("hi")]),
            output: Some(OutputOptions {
                format: Some("banana".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(response.output().unwrap(), "banana: Echo: hi");

    let mut stream_resp = generate_stream(
        &registry,
        GenerateOptions::<String> {
            model: Some(genkit::model::Model::Name("echoModel".to_string())),
            prompt: Some(vec![genkit::model::Part::text("hi")]),
            output: Some(OutputOptions {
                format: Some("banana".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        },
    );

    let mut chunks = Vec::new();
    while let Some(chunk) = stream_resp.stream.next().await {
        chunks.push(chunk.unwrap().output().unwrap().clone());
    }
    assert_eq!(chunks, vec!["banana: 3", "banana: 2", "banana: 1"]);
    let final_response = stream_resp.response.await.unwrap().unwrap();
    assert_eq!(final_response.output().unwrap(), "banana: Echo: hi");
}

#[rstest]
#[tokio::test]
async fn test_custom_format_simulated_constrained(mut registry: Registry) {
    define_echo_model(&mut registry, "none");

    let response = generate(
        &registry,
        GenerateOptions::<String> {
            model: Some(genkit::model::Model::Name("echoModel".to_string())),
            prompt: Some(vec![genkit::model::Part::text("hi")]),
            output: Some(OutputOptions {
                format: Some("banana".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    assert_eq!(response.output().unwrap(), "banana: Echo: hi");
}

#[rstest]
#[tokio::test]
async fn test_override_format_options(mut registry: Registry) {
    define_echo_model(&mut registry, "all");
    let response = generate(
        &registry,
        GenerateOptions::<serde_json::Value> {
            model: Some(genkit::model::Model::Name("echoModel".to_string())),
            prompt: Some(vec![genkit::model::Part::text("hi")]),
            output: Some(OutputOptions {
                format: Some("banana".to_string()),
                constrained: Some(false),
                schema: Some(json!({ "type": "string" })),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let last_message = response
        .request
        .unwrap()
        .messages
        .last()
        .cloned()
        .unwrap_or_default();
    let context_part = &last_message.content[1];
    let context_text = context_part.text.as_ref().unwrap();

    // In simulated constrained generation, instructions are added.
    // When `constrained: false` is passed, it should simulate and add instructions.
    assert!(context_text.contains("Output should be in banana format"));
}
