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

use crate::message::{Message, MessageData, MessageParser};
use crate::model::{FinishReason, GenerateRequest, GenerateResponseData, GenerationUsage};
use genkit_core::error::{Error, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Represents the result from a `generate()` call.
#[derive(Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct GenerateResponse<O = serde_json::Value> {
    pub message: Option<Message<O>>,
    pub finish_reason: Option<FinishReason>,
    pub finish_message: Option<String>,
    pub usage: Option<GenerationUsage>,
    pub custom: Option<serde_json::Value>,
    pub request: Option<GenerateRequest>,
    // In Rust, we'll handle operations differently, likely not as a direct field.
    // pub operation: Option<Operation<GenerateResponseData>>,
    pub model: Option<String>,
    #[serde(skip)]
    pub parser: Option<MessageParser<O>>,
}

impl<O: fmt::Debug> fmt::Debug for GenerateResponse<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GenerateResponse")
            .field("message", &self.message)
            .field("finish_reason", &self.finish_reason)
            .field("finish_message", &self.finish_message)
            .field("usage", &self.usage)
            .field("custom", &self.custom)
            .field("request", &self.request)
            .field("model", &self.model)
            .field(
                "parser",
                &if self.parser.is_some() {
                    "Some(<fn>)"
                } else {
                    "None"
                },
            )
            .finish()
    }
}

impl<O> GenerateResponse<O>
where
    O: for<'de> Deserialize<'de> + 'static,
{
    /// Creates a new `GenerateResponse` from the raw data and options.
    pub fn new(response: &GenerateResponseData, request: Option<GenerateRequest>) -> Self {
        let generated_message = response.candidates.first().map(|c| c.message.clone());

        Self {
            message: generated_message.map(|msg| Message::new(msg, None)),
            finish_reason: response
                .candidates
                .first()
                .and_then(|c| c.finish_reason.clone()),
            finish_message: response
                .candidates
                .first()
                .and_then(|c| c.finish_message.clone()),
            usage: response.usage.clone(),
            custom: response.custom.clone(),
            request,
            model: None,  // This would be populated from the request or model action.
            parser: None, // A parser can be attached later.
        }
    }

    /// Returns the full message history, including the new response.
    pub fn messages(&self) -> Result<Vec<MessageData>> {
        let req = self
            .request
            .as_ref()
            .ok_or_else(|| Error::new_internal("Request not found in response".to_string()))?;
        let msg = self
            .message
            .as_ref()
            .ok_or_else(|| Error::new_internal("Message not found in response".to_string()))?;

        let mut history = req.messages.clone();
        history.push(msg.to_json());
        Ok(history)
    }

    /// Returns the structured output from the message.
    pub fn output(&self) -> Result<O> {
        self.message
            .as_ref()
            .ok_or_else(|| Error::new_internal("Message not found in response".to_string()))?
            .output()
    }

    /// Returns the concatenated text from the message.
    pub fn text(&self) -> Result<String> {
        Ok(self
            .message
            .as_ref()
            .ok_or_else(|| Error::new_internal("Message not found in response".to_string()))?
            .text())
    }
}
