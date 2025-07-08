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

//! # Model Middleware Tests

use genkit_ai::model::{
    middleware::simulate_system_prompt, BoxFuture, GenerateRequest, GenerateResponseData,
};
use genkit_core::error::Result;
use rstest::rstest;
use serde_json::{from_value, json};
use std::sync::{Arc, Mutex};

// Helper to invoke the middleware and capture the modified request.
async fn test_request(req: GenerateRequest) -> GenerateRequest {
    let captured_req = Arc::new(Mutex::new(None));
    let captured_req_clone = captured_req.clone();

    let next = move |req: GenerateRequest| -> BoxFuture<'static, Result<GenerateResponseData>> {
        let mut guard = captured_req_clone.lock().unwrap();
        *guard = Some(req);
        Box::pin(async { Ok(Default::default()) })
    };

    let middleware = simulate_system_prompt(None);
    middleware(req, Box::new(next)).await.unwrap();

    let took_captured_req = captured_req.lock().unwrap().take().unwrap();
    took_captured_req
}

#[rstest]
#[tokio::test]
async fn test_does_not_modify_request_with_no_system_prompt() {
    let req: GenerateRequest = from_value(json!({
        "messages": [{ "role": "user", "content": [{ "text": "hello" }] }],
    }))
    .unwrap();

    let modified_req = test_request(req.clone()).await;
    assert_eq!(modified_req, req);
}

#[rstest]
#[tokio::test]
async fn test_keeps_other_messages_in_place() {
    let req: GenerateRequest = from_value(json!({
        "messages": [
            { "role": "system", "content": [{ "text": "I am a system message" }] },
            { "role": "user", "content": [{ "text": "hello" }] },
        ],
    }))
    .unwrap();

    let expected_req: GenerateRequest = from_value(json!({
        "messages": [
            {
                "role": "user",
                "content": [
                    { "text": "SYSTEM INSTRUCTIONS:\n" },
                    { "text": "I am a system message" },
                ],
            },
            {
                "role": "model",
                "content": [{ "text": "Understood." }],
            },
            {
                "role": "user",
                "content": [{ "text": "hello" }],
            },
        ],
    }))
    .unwrap();

    let modified_req = test_request(req).await;
    assert_eq!(modified_req, expected_req);
}
