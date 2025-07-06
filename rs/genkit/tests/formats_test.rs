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

#[allow(clippy::duplicate_mod)]
#[path = "helpers.rs"]
mod helpers;

use futures_util::StreamExt;
use genkit::model::Message;
use genkit_ai::formats::types::FormatHandler;
use genkit_ai::generate::{generate, generate_stream, OutputOptions};
use genkit_ai::GenerateOptions;
use genkit_core::registry::Registry;
use rstest::*;
use serde_json::{json, Value};
use std::sync::Arc;

struct BananaFormat;
impl genkit_ai::formats::Format for BananaFormat {
    fn parse_message(&self, message: &Message) -> Value {
        println!(
            "[BananaFormat::parse_message] called with message text: {}",
            message.text()
        );
        Value::String(format!("banana: {}", message.text()))
    }
    fn parse_chunk(&self, chunk: &genkit_ai::generate::GenerateResponseChunk) -> Option<Value> {
        Some(Value::String(format!("banana: {}", chunk.text())))
    }
    fn instructions(&self) -> Option<String> {
        Some("Output should be in banana format".to_string())
    }
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
    helpers::define_echo_model(&mut registry, "all");

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

    println!("response output:{:?}", response);

    let output = response.output().unwrap();
    println!("response.output() result: {:?}", output);
    assert_eq!(output, "banana: Echo: hi");

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
    println!("chunks:{:?}", chunks);
    assert_eq!(chunks, vec!["banana: 3", "banana: 2", "banana: 1"]);
    let final_response = stream_resp.response.await.unwrap().unwrap();
    println!("final_response: {:?}", final_response);
    let final_output = final_response.output().unwrap();
    println!("final_response.output() result: {:?}", final_output);
    assert_eq!(final_output, "banana: Echo: hi");
}

#[rstest]
#[tokio::test]
async fn test_custom_format_simulated_constrained(mut registry: Registry) {
    helpers::define_echo_model(&mut registry, "none");

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
    helpers::define_echo_model(&mut registry, "none");
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
