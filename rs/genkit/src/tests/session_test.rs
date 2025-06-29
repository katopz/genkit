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

//! # Session Client Tests
//!
//! Tests for session-based interactions, simulating a stateful conversation
//! with a flow. These tests demonstrate how a client would manage history
//! manually when using the generic `run_flow` function.

use super::helpers::with_mock_server;
use genkit_ai::client::{run_flow, RunFlowParams};
use genkit_ai::error::Result;
use hyper::{Body, Request, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;

// Represents a single message in a conversation.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
struct Message {
    role: String,
    content: String,
}

// Represents the input to a session-aware flow.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SessionInput {
    messages: Vec<Message>,
}

// Represents the output from a session-aware flow.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SessionOutput {
    reply: Message,
}

#[cfg(test)]
mod test {
    use super::*;

    /// This test simulates a multi-turn conversation by manually maintaining
    /// the message history on the client side.
    #[tokio::test]
    async fn test_maintains_history_in_session() -> Result<()> {
        async fn handle(req: Request<Body>) -> Result<Response<Body>, Infallible> {
            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let session_input: SessionInput = serde_json::from_value(input["data"].clone()).unwrap();

            // The mock "model" just echoes the content of all previous messages.
            let combined_history = session_input
                .messages
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<String>>()
                .join(", ");

            let response_data = json!({
                "result": {
                    "reply": {
                        "role": "model",
                        "content": format!("Echo: {}", combined_history)
                    }
                }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let url = with_mock_server(handle).await;
        let mut history: Vec<Message> = Vec::new();

        // Turn 1
        let user_message1 = Message {
            role: "user".to_string(),
            content: "hi".to_string(),
        };
        history.push(user_message1);

        let response1 = run_flow::<_, SessionOutput>(RunFlowParams {
            url: url.clone(),
            input: Some(SessionInput {
                messages: history.clone(),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(response1.reply.content, "Echo: hi");
        history.push(response1.reply); // Add model's reply to history

        // Turn 2
        let user_message2 = Message {
            role: "user".to_string(),
            content: "bye".to_string(),
        };
        history.push(user_message2);

        let response2 = run_flow::<_, SessionOutput>(RunFlowParams {
            url: url.clone(),
            input: Some(SessionInput {
                messages: history.clone(),
            }),
            headers: None,
        })
        .await?;

        assert_eq!(response2.reply.content, "Echo: hi, Echo: hi, bye");

        Ok(())
    }

    /// This test simulates loading a session by sending the server a session ID.
    /// The mock server uses this ID to retrieve pre-existing history.
    #[tokio::test]
    async fn test_load_session_from_store() -> Result<()> {
        use std::collections::HashMap;
        use std::sync::{Arc, Mutex};

        // Mock a server-side session store.
        let store = Arc::new(Mutex::new(HashMap::<String, Vec<Message>>::new()));
        store.lock().unwrap().insert(
            "session123".to_string(),
            vec![
                Message {
                    role: "user".to_string(),
                    content: "previous hi".to_string(),
                },
                Message {
                    role: "model".to_string(),
                    content: "previous bye".to_string(),
                },
            ],
        );

        async fn handle(
            req: Request<Body>,
            store: Arc<Mutex<HashMap<String, Vec<Message>>>>,
        ) -> Result<Response<Body>, Infallible> {
            let session_id = req
                .headers()
                .get("x-session-id")
                .map(|v| v.to_str().unwrap_or(""))
                .unwrap_or("");

            let mut history = if !session_id.is_empty() {
                store
                    .lock()
                    .unwrap()
                    .get(session_id)
                    .cloned()
                    .unwrap_or_default()
            } else {
                vec![]
            };

            let whole_body = hyper::body::to_bytes(req.into_body()).await.unwrap();
            let input: serde_json::Value = serde_json::from_slice(&whole_body).unwrap();
            let current_message: Message =
                serde_json::from_value(input["data"]["message"].clone()).unwrap();

            history.push(current_message);

            let combined_history = history
                .iter()
                .map(|m| m.content.clone())
                .collect::<Vec<String>>()
                .join(", ");

            let response_data = json!({
                "result": { "content": format!("Echo: {}", combined_history) }
            });
            Ok(Response::new(Body::from(response_data.to_string())))
        }

        let store_clone = store.clone();
        let url = with_mock_server(move |req| handle(req, store_clone.clone())).await;

        let mut headers = HashMap::new();
        headers.insert("x-session-id".to_string(), "session123".to_string());

        #[derive(Serialize)]
        struct LoadSessionRequest {
            message: Message,
        }
        #[derive(Deserialize, Debug, PartialEq)]
        struct LoadSessionResponse {
            content: String,
        }

        let response = run_flow::<_, LoadSessionResponse>(RunFlowParams {
            url,
            input: Some(LoadSessionRequest {
                message: Message {
                    role: "user".to_string(),
                    content: "new message".to_string(),
                },
            }),
            headers: Some(headers),
        })
        .await?;

        assert_eq!(
            response.content,
            "Echo: previous hi, previous bye, new message"
        );

        Ok(())
    }
}
