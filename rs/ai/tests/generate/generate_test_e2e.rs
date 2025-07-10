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
    generate::{GenerateOptions, GenerateResponse},
    model::Model,
    tool::{define_tool, ToolConfig},
    MessageData, Part, Role,
};
use genkit_core::registry::Registry;
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::StreamExt;

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

use std::sync::Arc;

use genkit_ai::{
    define_model, generate, generate_stream, model::ModelMiddleware, CandidateData, FinishReason,
    GenerateResponseData, GenerateStreamResponse,
};

#[fixture]
fn echo_model() -> Registry {
    let registry = Registry::new();
    let _ = define_model(
        &registry,
        genkit_ai::model::DefineModelOptions {
            name: "echoModel".to_string(),
            ..Default::default()
        },
        |req, _| async move {
            Ok(GenerateResponseData {
                candidates: vec![CandidateData {
                    message: MessageData {
                        role: Role::Model,
                        content: vec![Part::text(format!(
                            "Echo: {}",
                            req.messages
                                .iter()
                                .map(|m| m
                                    .content
                                    .iter()
                                    .map(|c| c.text.clone().unwrap_or_default())
                                    .collect::<String>())
                                .collect::<String>()
                        ))],
                        ..Default::default()
                    },
                    finish_reason: Some(FinishReason::Stop),
                    ..Default::default()
                }],
                ..Default::default()
            })
        },
    );
    registry
}

#[rstest]
#[tokio::test]
async fn test_applies_middleware(#[from(echo_model)] registry: Registry) {
    let wrap_request: ModelMiddleware = Arc::new(|req, next| {
        Box::pin(async move {
            let text = req
                .messages
                .iter()
                .map(|m| {
                    m.content
                        .iter()
                        .map(|c| c.text.clone().unwrap_or_default())
                        .collect::<String>()
                })
                .collect::<String>();
            let mut new_req = req.clone();
            new_req.messages = vec![MessageData {
                role: Role::User,
                content: vec![Part::text(format!("({})", text))],
                ..Default::default()
            }];
            next(new_req).await
        })
    });

    let wrap_response: ModelMiddleware = Arc::new(|req, next| {
        Box::pin(async move {
            let res = next(req).await?;
            let text = res
                .candidates
                .iter()
                .map(|c| {
                    c.message
                        .content
                        .iter()
                        .map(|p| p.text.clone().unwrap_or_default())
                        .collect::<String>()
                })
                .collect::<String>();

            Ok(GenerateResponseData {
                candidates: vec![CandidateData {
                    message: MessageData {
                        role: Role::Model,
                        content: vec![Part::text(format!("[{}]", text))],
                        ..Default::default()
                    },
                    finish_reason: res.candidates.first().and_then(|c| c.finish_reason.clone()),
                    ..Default::default()
                }],
                ..Default::default()
            })
        })
    });

    let response = generate(
        &registry,
        GenerateOptions::<Value> {
            prompt: Some(vec![Part::text("banana")]),
            model: Some(Model::Name("echoModel".to_string())),
            r#use: Some(vec![wrap_request, wrap_response]),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let want = "[Echo: (banana)]";
    assert_eq!(response.text().unwrap(), want);
}

#[fixture]
fn simple_echo() -> Registry {
    let registry = Registry::new();
    let _ = define_model(
        &registry,
        genkit_ai::model::DefineModelOptions {
            name: "echo".to_string(),
            supports: Some(genkit_ai::model::ModelInfoSupports {
                tools: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        },
        |req, _| async move {
            Ok(GenerateResponseData {
                candidates: vec![CandidateData {
                    message: req.messages.first().unwrap().clone(),
                    finish_reason: Some(FinishReason::Stop),
                    ..Default::default()
                }],
                ..Default::default()
            })
        },
    );
    registry
}

#[rstest]
#[tokio::test]
async fn test_preserves_request_in_response(#[from(simple_echo)] registry: Registry) {
    let response = generate(
        &registry,
        GenerateOptions::<Value> {
            model: Some(Model::Name("echo".to_string())),
            prompt: Some(vec![Part::text("Testing messages")]),
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let texts: Vec<String> = response
        .messages()
        .unwrap()
        .iter()
        .map(|m| m.text())
        .collect();
    assert_eq!(texts, vec!["Testing messages", "Testing messages"]);
}

#[fixture]
fn streaming_model() -> Registry {
    let registry = Registry::new();
    let _ = define_model(
        &registry,
        genkit_ai::model::DefineModelOptions {
            name: "echo-streaming".to_string(),
            ..Default::default()
        },
        |req, streaming_callback| async move {
            if let Some(cb) = streaming_callback {
                cb(genkit_ai::model::GenerateResponseChunkData {
                    index: 0,
                    content: vec![Part::text("hello, ")],
                    ..Default::default()
                });
                cb(genkit_ai::model::GenerateResponseChunkData {
                    index: 0,
                    content: vec![Part::text("world!")],
                    ..Default::default()
                });
            }
            Ok(GenerateResponseData {
                candidates: vec![CandidateData {
                    message: req.messages.first().unwrap().clone(),
                    finish_reason: Some(FinishReason::Stop),
                    ..Default::default()
                }],
                ..Default::default()
            })
        },
    );
    registry
}

#[rstest]
#[tokio::test]
async fn test_generate_stream_chunks(#[from(streaming_model)] registry: Registry) {
    let result: GenerateStreamResponse<Value> = generate_stream(
        &registry,
        GenerateOptions {
            model: Some(Model::Name("echo-streaming".to_string())),
            prompt: Some(vec![Part::text("Testing streaming")]),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    let chunks: Vec<_> = result.stream.map(|c| c.unwrap().to_json()).collect().await;

    assert_eq!(
        chunks,
        vec![
            genkit_ai::model::GenerateResponseChunkData {
                index: 0,
                content: vec![Part::text("hello, ")],
                role: None,
                usage: None,
                custom: None,
            },
            genkit_ai::model::GenerateResponseChunkData {
                index: 0,
                content: vec![Part::text("world!")],
                role: None,
                usage: None,
                custom: None,
            }
        ]
    );

    let final_response: GenerateResponse = result.response.await.unwrap().unwrap();
    let texts: Vec<String> = final_response
        .messages()
        .unwrap()
        .iter()
        .map(|m| m.text())
        .collect();
    assert_eq!(texts, vec!["Testing streaming", "Testing streaming"]);
}
