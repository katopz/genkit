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

//! # Formats Client Tests
//!
//! Tests for flows that might use custom output formats. These tests ensure
//! that the client can correctly serialize requests with format specifications
//! and deserialize the corresponding responses.

use super::helpers::with_mock_server;
use genkit_ai::client::{run_flow, stream_flow, RunFlowParams, StreamFlowParams};
use genkit_ai::error::Result;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use tokio_stream::StreamExt;

// Structs for simulating a request to a flow that uses a model with output formats.

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct OutputOptions {
    format: String,
    constrained: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct GenerateFlowInput {
    prompt: String,
    output: OutputOptions,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct GenerateFlowOutput {
    output: String,
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_run_flow_with_custom_format() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GenerateFlowInput = serde_json::from_value(input["data"].clone()).unwrap();

            assert_eq!(data.prompt, "hi");
            assert_eq!(data.output.format, "banana");
            assert_eq!(data.output.constrained, Some(true));

            let response_data = json!({
                "result": {
                    "output": "banana: Echo: hi"
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateFlowOutput>(RunFlowParams {
            url,
            input: Some(GenerateFlowInput {
                prompt: "hi".to_string(),
                output: OutputOptions {
                    format: "banana".to_string(),
                    constrained: Some(true),
                },
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            GenerateFlowOutput {
                output: "banana: Echo: hi".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_stream_flow_with_custom_format() -> Result<()> {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            let body = "data: {\"message\":\"banana: 3\"}\n\ndata: {\"message\":\"banana: 2\"}\n\ndata: {\"message\":\"banana: 1\"}\n\ndata: {\"result\":{\"output\":\"banana: Echo: hi\"}}\n\n";
            let response = Response::builder()
                .header("Content-Type", "text/event-stream")
                .body(Body::from(body))
                .unwrap();
            Ok(response)
        }
        let url = with_mock_server(handle).await;

        let mut response_stream =
            stream_flow::<GenerateFlowInput, GenerateFlowOutput, String>(StreamFlowParams {
                url,
                input: Some(GenerateFlowInput {
                    prompt: "hi".to_string(),
                    output: OutputOptions {
                        format: "banana".to_string(),
                        constrained: None,
                    },
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
                "banana: 3".to_string(),
                "banana: 2".to_string(),
                "banana: 1".to_string()
            ]
        );

        let final_output = response_stream.output.await??;
        assert_eq!(
            final_output,
            GenerateFlowOutput {
                output: "banana: Echo: hi".to_string()
            }
        );

        Ok(())
    }
}
