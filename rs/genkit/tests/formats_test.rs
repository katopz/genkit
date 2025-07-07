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
use genkit::formats::{define_format, Format, FormatHandler, Formatter, FormatterConfig};
use genkit::model::Message;
use genkit_ai::generate::{generate, generate_stream, OutputOptions};
use genkit_ai::GenerateOptions;
use genkit_core::registry::Registry;
use rstest::*;
use serde_json::Value;
use std::sync::Arc;

struct BananaFormat;
impl Format for BananaFormat {
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
async fn registry() -> Registry {
    let mut registry = Registry::new();
    genkit_ai::configure_ai(&mut registry);
    let banana_handler: FormatHandler = Arc::new(|_schema| Box::new(BananaFormat));
    let banana_config = FormatterConfig {
        constrained: Some(true),
        ..Default::default()
    };
    define_format(
        &mut registry,
        "banana",
        banana_config.clone(),
        banana_handler.clone(),
    );
    let format_keys = registry.lookup_value::<Formatter>("format", "json").await;
    println!("[fixture] Registered format keys: {:?}", format_keys);
    let format_keys: Vec<String> = registry.list_values("format").keys().cloned().collect();
    println!("[fixture] Registered format keys: {:?}", format_keys);
    registry
}

#[rstest]
#[tokio::test]
async fn test_custom_format_native_constrained(#[future] mut registry: Registry) {
    let mut registry = registry.await;
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
    )
    .await
    .unwrap();

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

    let request = final_response.request.as_ref().unwrap();
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].content[0].text.as_deref(), Some("hi"));
    assert_eq!(request.messages[0].role, genkit_ai::Role::User);
    let expected_output = serde_json::json!({
        "format": "banana",
        "constrained": true,
    });
    assert_eq!(request.output.clone().unwrap(), expected_output);
    assert!(request.tools.is_none());
    assert!(request.config.is_none());
}

#[rstest]
#[tokio::test]
async fn test_custom_format_simulated_constrained(#[future] mut registry: Registry) {
    let mut registry = registry.await;
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
    )
    .await
    .unwrap();

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

    let request = final_response.request.as_ref().unwrap();
    assert_eq!(request.messages.len(), 1);
    assert_eq!(request.messages[0].content[0].text.as_deref(), Some("hi"));
    assert_eq!(request.messages[0].role, genkit_ai::Role::User);
    let expected_output = serde_json::json!({
        "format": "banana",
        "constrained": true,
    });
    assert_eq!(request.output.clone().unwrap(), expected_output);
    assert!(request.tools.is_none());
    assert!(request.config.is_none());
}

#[rstest]
#[tokio::test]
async fn test_override_format_options(#[future] mut registry: Registry) {
    let mut registry = registry.await;
    helpers::define_echo_model(&mut registry, "none");
    let response = generate(
        &registry,
        GenerateOptions::<serde_json::Value> {
            model: Some(genkit::model::Model::Name("echoModel".to_string())),
            prompt: Some(vec![genkit::model::Part::text("hi")]),
            output: Some(OutputOptions {
                format: Some("banana".to_string()),
                constrained: Some(false),
                schema: Some(serde_json::json!({ "type": "string" })),
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let last_message = response
        .request
        .as_ref()
        .unwrap()
        .messages
        .last()
        .cloned()
        .unwrap();

    let instruction_part_found = last_message.content.iter().any(|p| {
        p.metadata
            .as_ref()
            .is_some_and(|m| m.get("purpose") == Some(&"output".into()))
            && p.text
                .as_ref()
                .is_some_and(|t| t.contains("Output should be in banana format"))
    });

    assert!(
        instruction_part_found,
        "Instruction part was not found in the request messages"
    );
}
