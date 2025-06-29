// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Chat Client Tests
//!
//! Tests for chat-like interactions using the flow client. These tests currently
//! use the generic `run_flow` and `stream_flow` functions to interact with a
//! mock server that simulates a chat flow. A higher-level chat client with
//! session management may be added in the future.

use super::helpers::with_mock_server;
use genkit_ai::client::{run_flow, stream_flow, RunFlowParams, StreamFlowParams};
use genkit_ai::error::Result;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::sync::oneshot;
use tokio_stream::StreamExt;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct ChatInput {
    message: String,
    history: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct ChatOutput {
    response: String,
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_single_turn_chat() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input_val: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();

            let chat_input_val = &input_val["data"];
            let message = chat_input_val["message"].as_str().unwrap();

            let response_data = json!({
                "result": {
                    "response": format!("Echo: {}", message)
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, ChatOutput>(RunFlowParams {
            url,
            input: Some(ChatInput {
                message: "hi".to_string(),
                history: vec![],
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            ChatOutput {
                response: "Echo: hi".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_chat() -> Result<()> {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            let body = "data: {\"message\":{\"chunk\":\"Echo:\"}}\n\ndata: {\"message\":{\"chunk\":\" hi\"}}\n\ndata: {\"result\":{\"response\":\"Echo: hi\"}}\n\n";
            let response = Response::builder()
                .header("Content-Type", "text/event-stream")
                .body(Body::from(body))
                .unwrap();
            Ok(response)
        }
        let url = with_mock_server(handle).await;

        #[derive(Deserialize, Debug, PartialEq)]
        struct StreamChunk {
            chunk: String,
        }
        #[derive(Deserialize, Debug, PartialEq)]
        struct FinalOutput {
            response: String,
        }

        let mut response_stream = stream_flow::<ChatInput, FinalOutput, StreamChunk>(StreamFlowParams {
            url,
            input: Some(ChatInput {
                message: "hi".to_string(),
                history: vec![],
            }),
            headers: None,
        });

        let mut chunks = Vec::new();
        while let Some(chunk_result) = response_stream.stream.next().await {
            chunks.push(chunk_result.unwrap());
        }

        assert_eq!(
            chunks,
            vec![
                StreamChunk {
                    chunk: "Echo:".to_string()
                },
                StreamChunk {
                    chunk: " hi".to_string()
                }
            ]
        );

        let final_output = response_stream.output.await??;
        assert_eq!(
            final_output,
            FinalOutput {
                response: "Echo: hi".to_string()
            }
        );

        Ok(())
    }
}
