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

use genkit_ai::{
    generate::{to_generate_request, GenerateOptions, OutputOptions},
    model::{GenerateRequest, Model},
    tool::{define_tool, ToolArgument, ToolConfig, ToolDefinition},
    Document, MessageData, Part, Role,
};
use genkit_core::registry::Registry;
use rstest::{fixture, rstest};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(JsonSchema, Deserialize, Serialize, Debug, Clone, Default)]
struct JokeInput {
    topic: String,
}

#[derive(JsonSchema, Deserialize, Serialize, Debug, Clone, Default)]
struct AddInput {
    a: i32,
    b: i32,
}

#[fixture]
fn registry() -> Registry {
    let registry = Registry::new();
    define_tool(
            &registry,
            ToolConfig::<JokeInput, String> {
                name: "tellAFunnyJoke".to_string(),
                description:
                    "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke."
                        .to_string(),
                ..Default::default()
            },
            |input, _| async move { Ok(format!("Why did the {} cross the road?", input.topic)) },
        );
    define_tool(
        &registry,
        ToolConfig::<AddInput, i32> {
            name: "namespaced/add".to_string(),
            description: "add two numbers together".to_string(),
            ..Default::default()
        },
        |input, _| async move { Ok(input.a + input.b) },
    );
    registry
}

#[rstest]
#[tokio::test]
async fn test_to_generate_request_simple_prompt(#[from(registry)] registry: Registry) {
    let options = GenerateOptions::<Value> {
        model: Some(Model::Name("vertexai/gemini-1.0-pro".to_string())),
        prompt: Some(vec![Part::text("Tell a joke about dogs.")]),
        ..Default::default()
    };
    let expected = GenerateRequest {
        messages: vec![MessageData {
            role: Role::User,
            content: vec![Part::text("Tell a joke about dogs.")],
            ..Default::default()
        }],
        tools: Some(vec![]),
        ..Default::default()
    };
    let actual = to_generate_request(&registry, &options).await.unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_to_generate_request_with_tools_by_name(#[from(registry)] registry: Registry) {
    let options = GenerateOptions::<Value> {
        model: Some(Model::Name("vertexai/gemini-1.0-pro".to_string())),
        tools: Some(vec![ToolArgument::Name("tellAFunnyJoke".to_string())]),
        prompt: Some(vec![Part::text("Tell a joke about dogs.")]),
        ..Default::default()
    };
    let mut actual = to_generate_request(&registry, &options).await.unwrap();
    let tool = actual.tools.as_mut().unwrap().get_mut(0).unwrap();
    // Schemas are complex and order isn't guaranteed, so we test them separately.
    let input_schema = tool.input_schema.take();
    let output_schema = tool.output_schema.take();

    let expected_input_schema: Value = serde_json::to_value(schema_for!(JokeInput)).unwrap();
    let expected_output_schema: Value = serde_json::to_value(schema_for!(String)).unwrap();
    assert_eq!(input_schema.unwrap(), expected_input_schema);
    assert_eq!(output_schema.unwrap(), expected_output_schema);

    // Test the rest of the struct
    let expected = GenerateRequest {
        messages: vec![MessageData {
            role: Role::User,
            content: vec![Part::text("Tell a joke about dogs.")],
            ..Default::default()
        }],
        tools: Some(vec![ToolDefinition {
            name: "tellAFunnyJoke".to_string(),
            description:
                "Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke."
                    .to_string(),
            input_schema: None,
            output_schema: None,
            metadata: Some(json!({})),
        }]),
        ..Default::default()
    };
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_to_generate_request_strips_tool_namespace(#[from(registry)] registry: Registry) {
    let options = GenerateOptions::<Value> {
        model: Some(Model::Name("vertexai/gemini-1.0-pro".to_string())),
        tools: Some(vec![ToolArgument::Name("namespaced/add".to_string())]),
        prompt: Some(vec![Part::text("Add 10 and 5.")]),
        ..Default::default()
    };
    let mut actual = to_generate_request(&registry, &options).await.unwrap();
    let tool = actual.tools.as_mut().unwrap().get_mut(0).unwrap();
    let input_schema = tool.input_schema.take();
    let output_schema = tool.output_schema.take();

    let expected_input_schema: Value = serde_json::to_value(schema_for!(AddInput)).unwrap();
    let expected_output_schema: Value = serde_json::to_value(schema_for!(i32)).unwrap();

    assert_eq!(input_schema.unwrap(), expected_input_schema);
    assert_eq!(output_schema.unwrap(), expected_output_schema);

    let expected = GenerateRequest {
        messages: vec![MessageData {
            role: Role::User,
            content: vec![Part::text("Add 10 and 5.")],
            ..Default::default()
        }],
        tools: Some(vec![ToolDefinition {
            name: "add".to_string(),
            description: "add two numbers together".to_string(),
            input_schema: None,
            output_schema: None,
            metadata: Some(json!({"originalName": "namespaced/add"})),
        }]),
        ..Default::default()
    };
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_to_generate_request_with_history(#[from(registry)] registry: Registry) {
    let options = GenerateOptions::<Value> {
        model: Some(Model::Name("vertexai/gemini-1.0-pro".to_string())),
        messages: Some(vec![
            MessageData {
                role: Role::User,
                content: vec![Part::text("hi")],
                ..Default::default()
            },
            MessageData {
                role: Role::Model,
                content: vec![Part::text("how can I help you")],
                ..Default::default()
            },
        ]),
        prompt: Some(vec![Part::text("Tell a joke about dogs.")]),
        ..Default::default()
    };
    let expected = GenerateRequest {
        messages: vec![
            MessageData {
                role: Role::User,
                content: vec![Part::text("hi")],
                ..Default::default()
            },
            MessageData {
                role: Role::Model,
                content: vec![Part::text("how can I help you")],
                ..Default::default()
            },
            MessageData {
                role: Role::User,
                content: vec![Part::text("Tell a joke about dogs.")],
                ..Default::default()
            },
        ],
        tools: Some(vec![]),
        ..Default::default()
    };
    let actual = to_generate_request(&registry, &options).await.unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_to_generate_request_with_context(#[from(registry)] registry: Registry) {
    let options = GenerateOptions::<Value> {
        model: Some(Model::Name("vertexai/gemini-1.0-pro".to_string())),
        prompt: Some(vec![Part::text("Tell a joke with context.")]),
        docs: Some(vec![Document::from_text("context here", None)]),
        ..Default::default()
    };
    let expected = GenerateRequest {
        messages: vec![MessageData {
            role: Role::User,
            content: vec![Part::text("Tell a joke with context.")],
            ..Default::default()
        }],
        docs: Some(vec![Document::from_text("context here", None)]),
        tools: Some(vec![]),
        ..Default::default()
    };
    let actual = to_generate_request(&registry, &options).await.unwrap();
    assert_eq!(actual, expected);
}

#[rstest]
#[tokio::test]
async fn test_to_generate_request_passes_output_options(#[from(registry)] registry: Registry) {
    let options = GenerateOptions::<Value> {
        model: Some(Model::Name("vertexai/gemini-1.0-pro".to_string())),
        prompt: Some(vec![Part::text("Tell a joke about dogs.")]),
        output: Some(OutputOptions {
            constrained: Some(true),
            format: Some("banana".to_string()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let expected = GenerateRequest {
        messages: vec![MessageData {
            role: Role::User,
            content: vec![Part::text("Tell a joke about dogs.")],
            ..Default::default()
        }],
        tools: Some(vec![]),
        output: Some(json!({ "constrained": true, "format": "banana" })),
        ..Default::default()
    };
    let actual = to_generate_request(&registry, &options).await.unwrap();
    assert_eq!(actual, expected);
}
