//! Copyright 2024 Google LLC
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.

mod helpers;

use genkit::Genkit;
use genkit_ai::{
    define_prompt,
    message::{MessageData, Role},
    model::{
        middleware::{BoxFuture, ModelMiddleware, ModelMiddlewareNext},
        GenerateRequest,
    },
    prompt::{PromptConfig, PromptGenerateOptions},
    Part,
};
use rstest::{fixture, rstest};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};

#[fixture]
async fn genkit_instance_for_test() -> (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>) {
    helpers::genkit_instance_for_test().await
}

// Middleware that wraps the request message text in parentheses.
fn wrap_request() -> ModelMiddleware {
    Arc::new(
        |req: GenerateRequest,
         next: ModelMiddlewareNext<'_>|
         -> BoxFuture<'_, genkit::error::Result<genkit_ai::GenerateResponseData>> {
            Box::pin(async move {
                let all_text = req
                    .messages
                    .iter()
                    .flat_map(|m| &m.content)
                    .filter_map(|p| p.text.as_deref())
                    .collect::<Vec<_>>()
                    .join(",");

                let mut new_req = req;
                new_req.messages = vec![MessageData::user(vec![Part::text(format!(
                    "({})",
                    all_text
                ))])];

                next(new_req).await
            })
        },
    )
}

// Middleware that wraps the response message text in square brackets.
fn wrap_response() -> ModelMiddleware {
    Arc::new(
        |req: GenerateRequest,
         next: ModelMiddlewareNext<'_>|
         -> BoxFuture<'_, genkit::error::Result<genkit_ai::GenerateResponseData>> {
            Box::pin(async move {
                let mut res = next(req).await?;

                if let Some(candidate) = res.candidates.get_mut(0) {
                    let all_text = candidate
                        .message
                        .content
                        .iter()
                        .filter_map(|p| p.text.as_deref())
                        .collect::<Vec<_>>()
                        .join(",");
                    candidate.message.content = vec![Part::text(format!("[{}]", all_text))];
                }

                Ok(res)
            })
        },
    )
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Default)]
struct HiInput {
    name: String,
}

#[rstest]
#[tokio::test]
async fn test_should_apply_middleware_to_a_prompt_call(
    #[future] genkit_instance_for_test: (Arc<Genkit>, Arc<Mutex<Option<GenerateRequest>>>),
) {
    let (genkit, _) = genkit_instance_for_test.await;

    let prompt_config = PromptConfig {
        name: "hi".to_string(),
        messages_fn: Some(Arc::new(|input: HiInput, _, _| {
            Box::pin(async move {
                Ok(vec![MessageData {
                    role: Role::User,
                    content: vec![Part::text(format!("hi {}", input.name))],
                    ..Default::default()
                }])
            })
        })),
        ..Default::default()
    };
    let prompt =
        define_prompt::<HiInput, Value, Value>(&mut genkit.registry().clone(), prompt_config);

    let opts = PromptGenerateOptions {
        r#use: Some(vec![wrap_request(), wrap_response()]),
        ..Default::default()
    };

    let response = prompt
        .generate(
            HiInput {
                name: "Genkit".to_string(),
            },
            Some(opts),
        )
        .await
        .unwrap();

    assert_eq!(response.text().unwrap(), "[Echo: (hi Genkit),; config: {}]");
}
