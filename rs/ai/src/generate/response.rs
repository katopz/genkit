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

use crate::document::ToolRequest;
use crate::generate::{GenerationBlockedError, GenerationResponseError};
use crate::message::{Message, MessageData, MessageParser};
use crate::model::{FinishReason, GenerateRequest, GenerateResponseData, GenerationUsage};
use genkit_core::error::{Error, Result};
use schemars::JsonSchema;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Operation {
    pub id: String,
    pub done: bool,
    // The result of a "done" operation is a GenerateResponseData.
    // Boxed to avoid recursive type.
    pub result: Option<Box<GenerateResponseData>>,
}

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
    pub operation: Option<Operation>,
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
            .field("operation", &self.operation)
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
    O: Clone + for<'de> DeserializeOwned + Serialize + fmt::Debug + Send + Sync + 'static,
{
    /// Creates a new `GenerateResponse` from the raw data and options.
    pub fn new(response: &GenerateResponseData, request: Option<GenerateRequest>) -> Self {
        let first_candidate = response.candidates.first();
        let generated_message = first_candidate.and_then(|c| {
            if c.message.content.is_empty() && c.message.metadata.is_none() {
                None
            } else {
                Some(c.message.clone())
            }
        });

        Self {
            message: generated_message.map(|msg| Message::new(msg, None)),
            finish_reason: first_candidate.and_then(|c| c.finish_reason.clone()),
            finish_message: first_candidate.and_then(|c| c.finish_message.clone()),
            usage: response.usage.clone(),
            custom: response.custom.clone(),
            request,
            operation: response.operation.as_ref().map(|id| Operation {
                id: id.clone(),
                done: false,
                result: None,
            }),
            model: None,  // This would be populated from the request or model action.
            parser: None, // A parser can be attached later.
        }
    }

    /// Throws an error if the response does not contain valid output.
    pub fn assert_valid(&self) -> Result<()> {
        if self.finish_reason == Some(FinishReason::Blocked) {
            let msg = self
                .finish_message
                .as_deref()
                .unwrap_or("Generation blocked.");
            return Err(GenerationBlockedError(GenerationResponseError {
                response: self.clone(),
                message: msg.to_string(),
            })
            .into());
        }

        if self.message.is_none() {
            let reason = self
                .finish_reason
                .as_ref()
                .map(|r| format!("{:?}", r))
                .unwrap_or("Unknown".to_string());
            let msg = format!(
                "Model did not generate a message. Finish reason: '{}': {}",
                reason,
                self.finish_message.as_deref().unwrap_or("")
            );
            return Err(GenerationResponseError {
                response: self.clone(),
                message: msg,
            }
            .into());
        }
        Ok(())
    }

    /// Throws an error if the response does not conform to expected schema.
    pub fn assert_valid_schema(&self) -> Result<()> {
        if let Some(options) = self.request.as_ref().and_then(|req| req.output.as_ref()) {
            if let Some(schema) = options.get("schema") {
                let output = self.output()?;
                let output_value = serde_json::to_value(&output).map_err(|e| {
                    Error::new_internal(format!(
                        "Failed to serialize output for schema validation: {}",
                        e
                    ))
                })?;

                if let Err(error) = jsonschema::validate(&output_value, schema) {
                    return Err(Error::new_internal(format!(
                        "Schema validation failed: {}",
                        error
                    )));
                }
            }
        }
        Ok(())
    }

    /// Checks if the response is valid and conforms to the schema without panicking.
    pub fn is_valid(&self) -> bool {
        self.assert_valid().is_ok() && self.assert_valid_schema().is_ok()
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

    /// Concatenates all `reasoning` parts present in the generated message.
    pub fn reasoning(&self) -> Result<String> {
        Ok(self
            .message
            .as_ref()
            .ok_or_else(|| Error::new_internal("Message not found in response".to_string()))?
            .reasoning())
    }

    /// Returns the first detected media part in the generated message.
    pub fn media(&self) -> Result<Option<&crate::document::Media>> {
        self.message
            .as_ref()
            .map(|m| m.media())
            .ok_or_else(|| Error::new_internal("Message not found in response".to_string()))
    }

    /// Returns all tool requests found in the generated message.
    pub fn tool_requests(&self) -> Result<Vec<&ToolRequest>> {
        Ok(self
            .message
            .as_ref()
            .ok_or_else(|| Error::new_internal("Message not found in response".to_string()))?
            .tool_requests())
    }

    /// Returns all tool requests annotated as interrupts found in the generated message.
    pub fn interrupts(&self) -> Result<Vec<&ToolRequest>> {
        Ok(self
            .message
            .as_ref()
            .ok_or_else(|| Error::new_internal("Message not found in response".to_string()))?
            .interrupts())
    }

    /// Converts the response to its serializable data representation.
    pub fn to_json_data(&self) -> Result<GenerateResponseData> {
        let candidates = if let Some(msg) = &self.message {
            vec![crate::model::CandidateData {
                index: 0,
                message: msg.to_json(),
                finish_reason: self.finish_reason.clone(),
                finish_message: self.finish_message.clone(),
            }]
        } else if self.finish_reason.is_some() {
            // If there's no message, we might still have a finish reason (e.g., Blocked)
            // So we construct a candidate with an empty message.
            vec![crate::model::CandidateData {
                index: 0,
                message: MessageData::default(),
                finish_reason: self.finish_reason.clone(),
                finish_message: self.finish_message.clone(),
            }]
        } else {
            vec![]
        };

        Ok(GenerateResponseData {
            candidates,
            usage: self.usage.clone(),
            custom: self.custom.clone(),
            operation: self.operation.as_ref().map(|o| o.id.clone()),
            aggregated: None,
        })
    }
}
