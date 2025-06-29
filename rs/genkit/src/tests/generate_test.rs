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

//! # Generate Client Tests
//!
//! Tests for the generate functionality, simulating calls to a "generate" flow.

use super::helpers::with_mock_server;
use genkit_ai::client::{run_flow, stream_flow, RunFlowParams, StreamFlowParams};
use genkit_ai::error::Result;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use tokio_stream::StreamExt;

// Basic Generate Flow constructs
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct GenerateInput {
    prompt: String,
    system: Option<String>,
    #[serde(rename = "toolChoice")]
    tool_choice: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct GenerateOutput {
    text: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Part {
    text: String,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct GeneratePartsInput {
    prompt: Vec<Part>,
}

#[cfg(test)]
mod default_model_tests {
    use super::*;

    #[tokio::test]
    async fn test_calls_default_model() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GenerateInput = serde_json::from_value(input["data"].clone()).unwrap();

            let response_text = format!("Echo: {}; config: {{}}", data.prompt);

            let response_data = json!({
                "result": {
                    "text": response_text
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some(GenerateInput {
                prompt: "hi".to_string(),
                system: None,
                tool_choice: None,
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            GenerateOutput {
                text: "Echo: hi; config: {}".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_calls_default_model_with_string_prompt() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let prompt_str = input["data"].as_str().unwrap();
            let response_text = format!("Echo: {}; config: {{}}", prompt_str);
            let response_data = json!({
                "result": {
                    "text": response_text
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some("hi".to_string()),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            GenerateOutput {
                text: "Echo: hi; config: {}".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_calls_default_model_with_parts_prompt() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GeneratePartsInput = serde_json::from_value(input["data"].clone()).unwrap();

            let prompt_text = data.prompt.get(0).unwrap().text.clone();
            let response_text = format!("Echo: {}; config: {{}}", prompt_text);

            let response_data = json!({
                "result": {
                    "text": response_text
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some(GeneratePartsInput {
                prompt: vec![Part {
                    text: "hi".to_string(),
                }],
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            GenerateOutput {
                text: "Echo: hi; config: {}".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_calls_default_model_system() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GenerateInput = serde_json::from_value(input["data"].clone()).unwrap();

            let response_text = format!(
                "Echo: system: {},{}; config: {{}}",
                data.system.unwrap(),
                data.prompt
            );

            let response_data = json!({
                "result": {
                    "text": response_text
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some(GenerateInput {
                prompt: "hi".to_string(),
                system: Some("talk like a pirate".to_string()),
                tool_choice: None,
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            GenerateOutput {
                text: "Echo: system: talk like a pirate,hi; config: {}".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_calls_default_model_with_tool_choice() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GenerateInput = serde_json::from_value(input["data"].clone()).unwrap();

            assert_eq!(data.tool_choice, Some("required".to_string()));

            let response_text = format!("Echo: {}; config: {{}}", data.prompt);
            let response_data = json!({
                "result": {
                    "text": response_text
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some(GenerateInput {
                prompt: "hi".to_string(),
                system: None,
                tool_choice: Some("required".to_string()),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            GenerateOutput {
                text: "Echo: hi; config: {}".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_streams_the_default_model() -> Result<()> {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            let body = "data: {\"message\":{\"text\":\"3\"}}\n\ndata: {\"message\":{\"text\":\"2\"}}\n\ndata: {\"message\":{\"text\":\"1\"}}\n\ndata: {\"result\":{\"text\":\"Echo: hi; config: {}\"}}\n\n";
            let response = Response::builder()
                .header("Content-Type", "text/event-stream")
                .body(Body::from(body))
                .unwrap();
            Ok(response)
        }
        let url = with_mock_server(handle).await;

        let mut response_stream = stream_flow::<String, GenerateOutput, Part>(StreamFlowParams {
            url,
            input: Some("hi".to_string()),
            headers: None,
        });

        let mut chunks = Vec::new();
        while let Some(chunk_result) = response_stream.stream.next().await {
            chunks.push(chunk_result.unwrap().text);
        }

        assert_eq!(
            chunks,
            vec!["3".to_string(), "2".to_string(), "1".to_string()]
        );

        let final_output = response_stream.output.await??;
        assert_eq!(
            final_output,
            GenerateOutput {
                text: "Echo: hi; config: {}".to_string()
            }
        );
        Ok(())
    }
}

#[cfg(test)]
mod explicit_model_tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct GenerateWithModelInput {
        model: String,
        prompt: String,
    }

    #[tokio::test]
    async fn test_calls_explicit_model() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GenerateWithModelInput =
                serde_json::from_value(input["data"].clone()).unwrap();

            assert_eq!(data.model, "echoModel");

            let response_text = format!("Echo: {}; config: {{}}", data.prompt);
            let response_data = json!({
                "result": {
                    "text": response_text
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some(GenerateWithModelInput {
                model: "echoModel".to_string(),
                prompt: "hi".to_string(),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response,
            GenerateOutput {
                text: "Echo: hi; config: {}".to_string()
            }
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_rejects_on_invalid_model() {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            Ok(Response::builder()
                .status(404)
                .body(Body::from("Model not found"))
                .unwrap())
        }

        let url = with_mock_server(handle).await;

        let result = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some(GenerateWithModelInput {
                model: "modelThatDoesNotExist".to_string(),
                prompt: "hi".to_string(),
            }),
            headers: None,
        })
        .await;

        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Server returned: 404 Not Found"));
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;

    #[tokio::test]
    async fn test_rethrows_response_errors() {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            let body = "data: {\"message\":{\"text\":\"3\"}}\n\ndata: {\"error\":{\"status\":\"BLOCKED\",\"message\":\"Blocked for some reason\"}}\n\n";
            let response = Response::builder()
                .header("Content-Type", "text/event-stream")
                .body(Body::from(body))
                .unwrap();
            Ok(response)
        }
        let url = with_mock_server(handle).await;

        let mut response_stream = stream_flow::<String, GenerateOutput, Part>(StreamFlowParams {
            url,
            input: Some("hi".to_string()),
            headers: None,
        });

        // The first chunk should be fine
        let chunk = response_stream.stream.next().await.unwrap();
        assert_eq!(chunk.unwrap().text, "3");

        // The stream should end, and the final result should be an error.
        let final_result = response_stream.output.await.unwrap();
        assert!(final_result.is_err());
        let err = final_result.unwrap_err();
        assert!(err.to_string().contains("BLOCKED: Blocked for some reason"));
    }

    #[tokio::test]
    async fn test_rethrows_initialization_errors() {
        async fn handle(_: Request<Body>) -> Result<Response<Body>, Infallible> {
            Ok(Response::builder()
                .status(404)
                .body(Body::from("Model not found"))
                .unwrap())
        }
        let url = with_mock_server(handle).await;

        let response_stream = stream_flow::<String, GenerateOutput, Part>(StreamFlowParams {
            url,
            input: Some("hi".to_string()),
            headers: None,
        });

        let final_result = response_stream.output.await.unwrap();
        assert!(final_result.is_err());
        let err = final_result.unwrap_err();
        assert!(err
            .to_string()
            .contains("Server returned non-200 status"));
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Config {
        temperature: Option<f32>,
        version: Option<String>,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct GenerateWithConfigInput {
        prompt: String,
        model: String,
        config: Config,
    }

    #[tokio::test]
    async fn test_takes_config_passed_to_generate() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GenerateWithConfigInput =
                serde_json::from_value(input["data"].clone()).unwrap();

            assert_eq!(data.config.temperature, Some(11.0));

            let response_text = format!(
                "Echo: {}; config: {}",
                data.prompt,
                serde_json::to_string(&data.config).unwrap()
            );

            let response_data = json!({
                "result": {
                    "text": response_text
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response = run_flow::<_, GenerateOutput>(RunFlowParams {
            url,
            input: Some(GenerateWithConfigInput {
                prompt: "hi".to_string(),
                model: "echoModel".to_string(),
                config: Config {
                    temperature: Some(11.0),
                    version: None,
                },
            }),
            headers: None,
        })
        .await?;

        assert_eq!(
            response.text,
            "Echo: hi; config: {\"temperature\":11.0,\"version\":null}"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tool_tests {
    use super::*;
    use serde_json::Value;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct ToolRequest {
        name: String,
        input: Value,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
    #[serde(untagged)]
    enum MessageContent {
        Text(Part),
        ToolRequest {
            #[serde(rename = "toolRequest")]
            tool_request: ToolRequest,
        },
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Message {
        role: String,
        content: Vec<MessageContent>,
    }

    // This test simulates a multi-turn conversation where a tool is called.
    // The actual tool-calling logic resides within the flow, not the client.
    // The client just sends requests and receives responses.
    #[tokio::test]
    async fn test_simulated_tool_call_conversation() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let req_val: Value = serde_json::from_slice(&whole_body).unwrap();
            let messages: Vec<Message> =
                serde_json::from_value(req_val["data"]["messages"].clone()).unwrap_or_default();

            let response_body = if messages.is_empty() {
                // First request from the user, flow returns a tool request.
                json!({
                    "result": {
                        "message": {
                            "role": "model",
                            "content": [{ "toolRequest": { "name": "testTool", "input": {} } }]
                        }
                    }
                })
            } else {
                // Second request, user has provided the tool response. Flow returns final text.
                assert_eq!(messages.len(), 3); // user, model(tool_request), tool(tool_response)
                json!({
                    "result": {
                         "message": {
                            "role": "model",
                            "content": [{ "text": "done" }]
                        }
                    }
                })
            };
            Ok(Response::new(Body::from(response_body.to_string())))
        }

        let url = with_mock_server(handle).await;

        // 1. Initial call returns a tool request
        let initial_response: Value = run_flow::<_, _>(RunFlowParams {
            url: url.clone(),
            input: Some(json!({ "prompt": "call the tool", "tools": ["testTool"] })),
            headers: None,
        })
        .await?;

        let tool_request_val = initial_response["message"]["content"][0]["toolRequest"].clone();
        let tool_request: ToolRequest = serde_json::from_value(tool_request_val).unwrap();
        assert_eq!(tool_request.name, "testTool");

        // 2. Second call with the tool's response
        let final_response: Value = run_flow::<_, _>(RunFlowParams {
            url: url.clone(),
            input: Some(json!({
                "prompt": "call the tool",
                "tools": ["testTool"],
                "messages": [
                    { "role": "user", "content": [{ "text": "call the tool" }] },
                    { "role": "model", "content": [{ "toolRequest": tool_request }] },
                    { "role": "tool", "content": [{ "toolResponse": { "name": "testTool", "output": "tool called" } }] }
                ]
            })),
            headers: None,
        })
        .await?;

        assert_eq!(
            final_response["message"]["content"][0]["text"]
                .as_str()
                .unwrap(),
            "done"
        );

        Ok(())
    }
}

#[cfg(test)]
mod long_running_tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Operation {
        id: String,
        done: bool,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct GenerateWithModelInput {
        model: String,
        prompt: String,
    }

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct CheckOperationInput {
        action: String,
        id: String,
    }

    #[tokio::test]
    async fn test_starts_the_operation() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: GenerateWithModelInput =
                serde_json::from_value(input["data"].clone()).unwrap();

            assert_eq!(data.model, "bkg-model");

            let response_data = json!({
                "result": {
                    "operation": { "id": "123", "done": false }
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response: serde_json::Value = run_flow::<_, _>(RunFlowParams {
            url,
            input: Some(GenerateWithModelInput {
                model: "bkg-model".to_string(),
                prompt: "generate video".to_string(),
            }),
            headers: None,
        })
        .await?;

        let operation: Operation = serde_json::from_value(response["operation"].clone()).unwrap();

        assert_eq!(operation.id, "123");
        assert!(!operation.done);

        Ok(())
    }

    #[tokio::test]
    async fn test_checks_operation_status() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let data: CheckOperationInput =
                serde_json::from_value(input["data"].clone()).unwrap();

            assert_eq!(data.action, "/background-model/bkg-model");
            assert_eq!(data.id, "123");

            let response_data = json!({
                "result": {
                    "operation": {
                        "id": "123",
                        "done": true,
                        "result": { "message": "done" }
                    }
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;

        let response: serde_json::Value = run_flow::<_, _>(RunFlowParams {
            url,
            input: Some(CheckOperationInput {
                action: "/background-model/bkg-model".to_string(),
                id: "123".to_string(),
            }),
            headers: None,
        })
        .await?;

        #[derive(Deserialize)]
        struct OpResult {
            operation: OperationWithResult,
        }
        #[derive(Deserialize)]
        struct OperationWithResult {
            id: String,
            done: bool,
            result: serde_json::Value,
        }

        let op_result: OpResult = serde_json::from_value(response).unwrap();

        assert_eq!(op_result.operation.id, "123");
        assert!(op_result.operation.done);
        assert_eq!(
            op_result.operation.result,
            json!({ "message": "done" })
        );

        Ok(())
    }
}
