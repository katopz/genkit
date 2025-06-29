// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Prompts Client Tests
//!
//! Tests for invoking flows that are conceptually defined by prompts.
//! These tests verify that the client can correctly interact with such flows,
//! sending appropriate inputs and handling structured outputs.

use super::helpers::with_mock_server;
use crate::client::{run_flow, RunFlowParams};
use crate::error::Result;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;

// Represents the input for a prompt that takes a subject.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct PromptInput {
    subject: String,
}

// Represents the output of a prompt that returns a simple message.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct PromptOutput {
    message: String,
}

// Represents the output of a prompt with a specific JSON schema.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct StructuredOutput {
    foo: String,
    bar: i32,
}

// Represents the full request sent to a kitchen-sink-style prompt flow.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct KitchenSinkInput {
    subject: String,
    // Assuming other fields like `history` might be part of a more complex type.
}

#[cfg(test)]
mod test {
    use super::*;

    /// Tests invoking a simple prompt-flow that takes no input.
    #[tokio::test]
    async fn test_simple_prompt_flow() -> Result<()> {
        async fn handle(_: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let response_data = json!({
                "result": {
                    "message": "Hello from the prompt file"
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<(), PromptOutput>(RunFlowParams {
            url,
            input: Some(()),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            PromptOutput {
                message: "Hello from the prompt file".to_string()
            }
        );
        Ok(())
    }

    /// Tests a prompt-flow that takes a simple string input.
    #[tokio::test]
    async fn test_prompt_with_input() -> Result<()> {
        async fn handle(req: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: PromptInput = serde_json::from_value(input["data"].clone()).unwrap();

            let response_data = json!({
                "result": {
                    "message": format!("Poem about {}", data.subject)
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, PromptOutput>(RunFlowParams {
            url,
            input: Some(PromptInput {
                subject: "bananas".to_string(),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            PromptOutput {
                message: "Poem about bananas".to_string()
            }
        );
        Ok(())
    }

    /// Tests a prompt-flow that is expected to return a structured JSON output.
    #[tokio::test]
    async fn test_prompt_with_structured_output() -> Result<()> {
        async fn handle(_: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let response_data = json!({
                "result": {
                    "foo": "structured",
                    "bar": 42
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<(), StructuredOutput>(RunFlowParams {
            url,
            input: Some(()),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            StructuredOutput {
                foo: "structured".to_string(),
                bar: 42
            }
        );
        Ok(())
    }

    /// This test simulates calling a complex "kitchen sink" prompt-flow.
    /// The handler validates that it receives the expected input data.
    #[tokio::test]
    async fn test_kitchen_sink_prompt_flow_invocation() -> Result<()> {
        async fn handle(req: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: KitchenSinkInput = serde_json::from_value(input["data"].clone()).unwrap();

            // The main purpose of this test is to ensure the input is received correctly.
            assert_eq!(data.subject, "banana");

            // The response just confirms the interaction.
            let response_data = json!({
                "result": {
                    "text": "Received kitchen sink request for banana"
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response: serde_json::Value = run_flow::<_, _>(RunFlowParams {
            url,
            input: Some(KitchenSinkInput {
                subject: "banana".to_string(),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(response["text"], "Received kitchen sink request for banana");
        Ok(())
    }
}
