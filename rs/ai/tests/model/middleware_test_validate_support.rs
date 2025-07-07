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
    middleware::validate_support, BoxFuture, GenerateRequest, GenerateResponseData,
    ModelInfoSupports,
};
use genkit_core::error::Result;
use rstest::rstest;
use serde_json::{from_value, json};

#[cfg(test)]
mod test {
    use super::*;

    #[rstest]
    #[tokio::test]
    async fn test_validate_support_accepts_anything_by_default() {
        async fn test_validation_run(req: GenerateRequest, supports: ModelInfoSupports) {
            let next = |_req: GenerateRequest| -> BoxFuture<'static, Result<GenerateResponseData>> {
                Box::pin(async { Ok(Default::default()) })
            };
            let middleware = validate_support("test-model".to_string(), supports);
            middleware(req, Box::new(next)).await.unwrap();
        }

        let examples_json = json!([
            {
                "messages": [
                    { "role": "user", "content": [{ "text": "hello" }] },
                    { "role": "model", "content": [{ "text": "hi" }] },
                    { "role": "user", "content": [{ "text": "how are you" }] }
                ]
            },
            {
                "messages": [
                    { "role": "user", "content": [{ "media": { "url": "https://example.com/image.png", "content_type": "image/png" } }] }
                ]
            },
            {
                "messages": [
                    { "role": "user", "content": [{ "media": { "url": "https://example.com/image.png", "content_type": "image/png" } }] }
                ],
                "tools": [
                    { "name": "someTool", "description": "hello world", "input_schema": { "type": "object" } }
                ]
            },
            {
                "messages": [
                    { "role": "user", "content": [{ "text": "hello world" }] }
                ],
                "output": "{\"format\":\"json\"}"
            }
        ]);

        let examples: Vec<GenerateRequest> = from_value(examples_json).unwrap();

        let supports = ModelInfoSupports::default();
        for req in examples {
            test_validation_run(req, supports.clone()).await;
        }
    }

    #[rstest]
    #[tokio::test]
    async fn test_validate_support_throws_when_media_not_supported() {
        let media_req: GenerateRequest = from_value(json!({
            "messages": [{ "role": "user", "content": [{ "media": { "url": "bar.jpg", "content_type": "image/jpeg" } }] }]
        }))
        .unwrap();
        let supports = ModelInfoSupports {
            media: Some(false),
            ..Default::default()
        };
        let next = |_req: GenerateRequest| -> BoxFuture<'static, Result<GenerateResponseData>> {
            Box::pin(async { Ok(Default::default()) })
        };
        let middleware = validate_support("test-model".to_string(), supports);
        let result = middleware(media_req, Box::new(next)).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not support media"));
    }

    #[rstest]
    #[tokio::test]
    async fn test_validate_support_throws_when_tools_not_supported() {
        let tools_req: GenerateRequest = from_value(json!({
            "messages": [{ "role": "user", "content": [{ "text": "hello" }] }],
            "tools": [{ "name": "foo" }]
        }))
        .unwrap();
        let supports = ModelInfoSupports {
            tools: Some(false),
            ..Default::default()
        };
        let next = |_req: GenerateRequest| -> BoxFuture<'static, Result<GenerateResponseData>> {
            Box::pin(async { Ok(Default::default()) })
        };
        let middleware = validate_support("test-model".to_string(), supports);
        let result = middleware(tools_req, Box::new(next)).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not support tool use"));
    }

    #[rstest]
    #[tokio::test]
    async fn test_validate_support_throws_when_multiturn_not_supported() {
        let multiturn_req: GenerateRequest = from_value(json!({
            "messages": [
                { "role": "user", "content": [{ "text": "hello" }] },
                { "role": "model", "content": [{ "text": "hi" }] }
            ]
        }))
        .unwrap();
        let supports = ModelInfoSupports {
            multiturn: Some(false),
            ..Default::default()
        };
        let next = |_req: GenerateRequest| -> BoxFuture<'static, Result<GenerateResponseData>> {
            Box::pin(async { Ok(Default::default()) })
        };
        let middleware = validate_support("test-model".to_string(), supports);
        let result = middleware(multiturn_req, Box::new(next)).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("does not support multiple messages"));
    }
}
