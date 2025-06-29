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

//! # Evaluate Client Tests
//!
//! Tests for evaluation functionality, simulating calls to an evaluator flow.

use super::helpers::with_mock_server;
use genkit_ai::client::{run_flow, RunFlowParams};
use genkit_ai::error::Result;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;

// Simplified local definitions for testing.
// A real implementation would import these from a shared `genkit_ai::ai` module.

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct TestCase {
    #[serde(rename = "testCaseId")]
    test_case_id: String,
    input: String,
    output: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct EvaluateRequest {
    evaluator: String,
    dataset: Vec<TestCase>,
    #[serde(rename = "evalRunId")]
    eval_run_id: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct EvaluationDetails {
    reasoning: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Evaluation {
    score: String,
    details: Option<EvaluationDetails>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct EvaluateResponseItem {
    #[serde(rename = "testCaseId")]
    test_case_id: String,
    evaluation: Evaluation,
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    async fn test_evaluate_flow() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: EvaluateRequest = serde_json::from_value(input["data"].clone()).unwrap();

            assert_eq!(data.evaluator, "bonkness");
            assert_eq!(data.dataset.len(), 1);
            assert_eq!(data.dataset[0].test_case_id, "test-case-1");

            let response_data = json!({
                "result": [
                    {
                        "testCaseId": "test-case-1",
                        "evaluation": {
                            "score": "Much bonk",
                            "details": {
                                "reasoning": "Because I said so!"
                            }
                        }
                    }
                ]
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let request_payload = EvaluateRequest {
            evaluator: "bonkness".to_string(),
            dataset: vec![TestCase {
                test_case_id: "test-case-1".to_string(),
                input: "Why did the chicken cross the road?".to_string(),
                output: "To get to the other side".to_string(),
            }],
            eval_run_id: "my-dog-eval".to_string(),
        };

        let response = run_flow::<_, Vec<EvaluateResponseItem>>(RunFlowParams {
            url,
            input: Some(request_payload),
            headers: None,
        })
        .await?;

        assert_eq!(response.len(), 1);
        assert_eq!(
            response[0],
            EvaluateResponseItem {
                test_case_id: "test-case-1".to_string(),
                evaluation: Evaluation {
                    score: "Much bonk".to_string(),
                    details: Some(EvaluationDetails {
                        reasoning: "Because I said so!".to_string(),
                    }),
                },
            }
        );

        Ok(())
    }
}
