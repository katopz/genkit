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

//! # Flow Client Tests
//!
//! Integration tests for the flow client, validating `run_flow` and `stream_flow`.

use super::helpers::with_mock_server;
use genkit_ai::client::{run_flow, stream_flow, RunFlowParams, StreamFlowParams};
use genkit_ai::error::Result;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::net::SocketAddr;
use tokio::sync::oneshot;
use tokio_stream::StreamExt;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct TestInput {
    name: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct TestOutput {
    message: String,
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_run_flow_success() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let name = input["data"]["name"].as_str().unwrap();

            let response_data = json!({
                "result": {
                    "message": format!("Hello, {}", name)
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, TestOutput>(RunFlowParams {
            url,
            input: Some(TestInput {
                name: "Genkit".to_string(),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            TestOutput {
                message: "Hello, Genkit".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_run_flow_server_error() {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body("Internal Error".into())
                .unwrap())
        }
        let url = with_mock_server(handle).await;

        let result = run_flow::<_, TestOutput>(RunFlowParams {
            url,
            input: None,
            headers: None,
        })
        .await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Server returned: 500"));
        assert!(err_msg.contains("Internal Error"));
    }

    #[tokio::test]
    async fn test_stream_flow_success() -> Result<()> {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            let body = "data: {\"message\":\"one\"}\n\ndata: {\"message\":\"two\"}\n\ndata: {\"result\":\"done\"}\n\n";
            let response = Response::builder()
                .header("Content-Type", "text/event-stream")
                .body(Body::from(body))
                .unwrap();
            Ok(response)
        }
        let url = with_mock_server(handle).await;

        let mut response_stream = stream_flow::<(), String, String>(StreamFlowParams {
            url,
            input: Some(()),
            headers: None,
        });

        let mut chunks = Vec::new();
        while let Some(chunk_result) = response_stream.stream.next().await {
            chunks.push(chunk_result.unwrap());
        }

        assert_eq!(chunks, vec!["one".to_string(), "two".to_string()]);

        let final_output = response_stream.output.await??;
        assert_eq!(final_output, "done");

        Ok(())
    }

    #[tokio::test]
    async fn test_stream_flow_error_in_stream() {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            let body = "data: {\"message\":\"one\"}\n\ndata: {\"error\":{\"status\":\"FATAL\",\"message\":\"Something broke\"}}\n\n";
            let response = Response::builder()
                .header("Content-Type", "text/event-stream")
                .body(Body::from(body))
                .unwrap();
            Ok(response)
        }
        let url = with_mock_server(handle).await;

        let mut response_stream = stream_flow::<(), String, String>(StreamFlowParams {
            url,
            input: None,
            headers: None,
        });

        // First chunk should be fine
        let first_chunk = response_stream.stream.next().await.unwrap().unwrap();
        assert_eq!(first_chunk, "one");

        // The stream should then end, and the output future should contain the error
        let final_result = response_stream.output.await.unwrap();
        assert!(final_result.is_err());
        let err_msg = final_result.unwrap_err().to_string();
        assert!(err_msg.contains("FATAL: Something broke"));
    }
}
