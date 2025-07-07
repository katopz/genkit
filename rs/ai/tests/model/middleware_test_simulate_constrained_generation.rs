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

use genkit_ai::model::{BoxFuture, GenerateRequest, GenerateResponseData};
use genkit_core::error::Result;
use rstest::rstest;
use serde_json::{from_value, json};
use std::sync::{Arc, Mutex};

#[cfg(test)]
mod test {
    use genkit_ai::model::middleware::simulate_constrained_generation;

    use super::*;

    // Helper to invoke the middleware and capture the modified request.
    async fn test_constrained_request(req: GenerateRequest) -> GenerateRequest {
        let captured_req = Arc::new(Mutex::new(None));
        let captured_req_clone = captured_req.clone();

        let next = move |req: GenerateRequest| -> BoxFuture<'static, Result<GenerateResponseData>> {
            let mut guard = captured_req_clone.lock().unwrap();
            *guard = Some(req);
            Box::pin(async { Ok(Default::default()) })
        };

        let middleware = simulate_constrained_generation(None);
        middleware(req, Box::new(next)).await.unwrap();

        let took_captured_req = captured_req.lock().unwrap().take().unwrap();
        took_captured_req
    }

    #[rstest]
    #[tokio::test]
    async fn test_simulates_constrained_generation() {
        let output_json = json!({
            "constrained": true,
            "schema": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" }
                }
            }
        });
        let req: GenerateRequest = from_value(json!({
            "messages": [{ "role": "user", "content": [{ "text": "hello" }] }],
            "output": output_json.to_string(),
        }))
        .unwrap();

        let _modified_req = test_constrained_request(req).await;
        // Just running this should trigger the log output.
    }
}
