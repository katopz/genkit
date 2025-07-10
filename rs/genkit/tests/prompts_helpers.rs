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

#![allow(dead_code)] // Allow unused code in a helper module

use genkit_ai::{
    message::MessageData,
    model::{
        middleware::{BoxFuture, ModelMiddleware, ModelMiddlewareNext},
        GenerateRequest,
    },
    Part,
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Default, Debug, PartialEq)]
pub struct HiInput {
    pub name: String,
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Default, Debug, PartialEq)]
pub struct HiOutput {
    pub message: String,
}

// Middleware that wraps the request message text in parentheses.
pub fn wrap_request() -> ModelMiddleware {
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
                    .join("");

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
pub fn wrap_response() -> ModelMiddleware {
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
                        .join("");
                    candidate.message.content = vec![Part::text(format!("[{}]", all_text))];
                }

                Ok(res)
            })
        },
    )
}
