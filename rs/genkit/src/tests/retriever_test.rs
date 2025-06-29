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

//! # Retriever Client Tests
//!
//! Tests for retriever and indexer functionality, simulating calls to flows
//! that would use these components.

use super::helpers::with_mock_server;
use crate::client::{run_flow, RunFlowParams};
use crate::error::Result;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;

// Local simplified definition for testing.
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

// Struct for a flow that calls a retriever.
#[derive(Serialize)]
struct RetrieveFlowRequest<'a> {
    retriever: &'a str,
    query: Document,
}

// Struct for a flow that calls an indexer.
#[derive(Serialize)]
struct IndexFlowRequest<'a> {
    indexer: &'a str,
    documents: Vec<Document>,
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_retrieve_flow() -> Result<()> {
        async fn handle(req: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data = &input["data"];

            assert_eq!(data["retriever"], "testRetriever");
            assert_eq!(data["query"]["text"], "some query");

            let response_data = json!({
                "result": [
                    { "text": "retrieved document 1" },
                    { "text": "retrieved document 2" }
                ]
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, Vec<Document>>(RunFlowParams {
            url,
            input: Some(RetrieveFlowRequest {
                retriever: "testRetriever",
                query: Document::from_text("some query"),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(response.len(), 2);
        assert_eq!(response[0].text, "retrieved document 1");
        assert_eq!(response[1].text, "retrieved document 2");

        Ok(())
    }

    #[tokio::test]
    async fn test_index_flow() -> Result<()> {
        async fn handle(req: Request<Body>) -> std::result::Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data = &input["data"];

            assert_eq!(data["indexer"], "testIndexer");
            let docs: Vec<Document> = serde_json::from_value(data["documents"].clone()).unwrap();
            assert_eq!(docs.len(), 2);
            assert_eq!(docs[0].text, "document to index 1");

            // Index flows typically return nothing on success.
            let response_data = json!({ "result": null });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        // The output type is `()` because the flow is expected to return nothing.
        run_flow::<_, ()>(RunFlowParams {
            url,
            input: Some(IndexFlowRequest {
                indexer: "testIndexer",
                documents: vec![
                    Document::from_text("document to index 1"),
                    Document::from_text("document to index 2"),
                ],
            }),
            headers: None,
        })
        .await?;

        // Just assert that the call completed successfully.
        assert_eq!("", "");

        Ok(())
    }
}
