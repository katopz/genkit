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

//! # Embed Client Tests
//!
//! Tests for embedding functionality, simulating calls to an embedding flow.

use super::helpers::with_mock_server;
use genkit_ai::client::{run_flow, RunFlowParams};
use genkit_ai::error::Result;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;

// Simplified local definitions for testing purposes.
// In a real scenario, these would likely be imported from a shared `genkit_ai::ai` module.

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Document {
    pub text: String,
}

impl Document {
    pub fn from_text(text: &str) -> Self {
        Self {
            text: text.to_string(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Embedding {
    pub embedding: Vec<f32>,
}

#[derive(Serialize)]
#[serde(untagged)]
enum EmbedContent {
    Text(String),
    Doc(Document),
}

#[derive(Serialize)]
struct EmbedRequest {
    embedder: String,
    content: EmbedContent,
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_embed_string_content() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data = &input["data"];
            assert_eq!(data["embedder"], "echoEmbedder");
            assert_eq!(data["content"], "hi");

            let response_data = json!({
                "result": [
                    { "embedding": [1.0, 2.0, 3.0, 4.0] }
                ]
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, Vec<Embedding>>(RunFlowParams {
            url,
            input: Some(EmbedRequest {
                embedder: "echoEmbedder".to_string(),
                content: EmbedContent::Text("hi".to_string()),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            vec![Embedding {
                embedding: vec![1.0, 2.0, 3.0, 4.0]
            }]
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_embed_document_content() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data = &input["data"];
            assert_eq!(data["embedder"], "echoEmbedder");
            assert_eq!(data["content"]["text"], "hi doc");

            let response_data = json!({
                "result": [
                    { "embedding": [5.0, 6.0, 7.0, 8.0] }
                ]
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, Vec<Embedding>>(RunFlowParams {
            url,
            input: Some(EmbedRequest {
                embedder: "echoEmbedder".to_string(),
                content: EmbedContent::Doc(Document::from_text("hi doc")),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            vec![Embedding {
                embedding: vec![5.0, 6.0, 7.0, 8.0]
            }]
        );
        Ok(())
    }
}
