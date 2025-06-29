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

//! # Flow Client Tests
//!
//! Tests for basic flow invocation (non-streaming and streaming).

use super::helpers::with_mock_server;
use crate::client::{run_flow, stream_flow, RunFlowParams, StreamFlowParams};
use crate::error::Result;
use hyper::{Body, Request, Response};
use serde_json::json;
use std::collections::HashMap;
use std::convert::Infallible;
use tokio_stream::StreamExt;

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_simple_flow() -> Result<()> {
        async fn handle(_: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let response_data = json!({ "result": "banana" });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<(), String>(RunFlowParams {
            url,
            input: Some(()),
            headers: None,
        })
        .await?;

        assert_eq!(response, "banana");
        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_flow() -> Result<()> {
        async fn handle(_: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let body = "data: {\"message\":\"b\"}\n\ndata: {\"message\":\"a\"}\n\ndata: {\"message\":\"n\"}\n\ndata: {\"message\":\"a\"}\n\ndata: {\"message\":\"n\"}\n\ndata: {\"message\":\"a\"}\n\ndata: {\"result\":\"banana\"}\n\n";
            let response = Response::builder()
                .header("Content-Type", "text/event-stream")
                .body(Body::from(body))
                .unwrap();
            Ok(response)
        }
        let url = with_mock_server(handle).await;

        let mut response_stream = stream_flow::<String, String, String>(StreamFlowParams {
            url,
            input: Some("banana".to_string()),
            headers: None,
        });

        let mut chunks = Vec::new();
        while let Some(chunk_result) = response_stream.stream.next().await {
            chunks.push(chunk_result.unwrap());
        }

        assert_eq!(
            chunks,
            vec!["b", "a", "n", "a", "n", "a"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<String>>()
        );

        let final_output = response_stream.output.await??;
        assert_eq!(final_output, "banana");

        Ok(())
    }

    #[tokio::test]
    async fn test_pass_through_headers() -> Result<()> {
        async fn handle(req: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let auth_header = req
                .headers()
                .get("X-Custom-Header")
                .map(|v| v.to_str().unwrap_or(""))
                .unwrap_or("not_found");
            let response_data = json!({ "result": auth_header });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;
        let mut headers = HashMap::new();
        headers.insert("X-Custom-Header".to_string(), "my-secret-value".to_string());

        let response = run_flow::<(), String>(RunFlowParams {
            url,
            input: Some(()),
            headers: Some(headers),
        })
        .await?;

        assert_eq!(response, "my-secret-value");
        Ok(())
    }
}
