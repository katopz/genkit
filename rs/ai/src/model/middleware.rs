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

//! # Model Middleware
//!
//! This module provides middleware for `ModelAction`s to preprocess requests
//! or post-process responses.

use crate::{
    document::Document,
    model::{GenerateRequest, GenerateResponseData, MessageData, ModelInfoSupports, Part},
    Role,
};
use base64::{engine::general_purpose, Engine};
use genkit_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{future::Future, pin::Pin, sync::Arc};

/// A boxed, pinned future.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The `next` function passed to a `ModelMiddleware`.
pub type ModelMiddlewareNext<'a> =
    Box<dyn Fn(GenerateRequest) -> BoxFuture<'a, Result<GenerateResponseData>> + Send + Sync>;

/// A middleware function for a model.
pub type ModelMiddleware = Arc<
    dyn for<'a> Fn(
            GenerateRequest,
            ModelMiddlewareNext<'a>,
        ) -> BoxFuture<'a, Result<GenerateResponseData>>
        + Send
        + Sync,
>;

/// Preprocesses a `GenerateRequest` to download referenced http(s) media URLs and
/// inline them as data URIs.
pub fn download_request_media(
    max_bytes: Option<usize>,
    filter: Option<fn(&Part) -> bool>,
) -> ModelMiddleware {
    Arc::new(move |req: GenerateRequest, next: ModelMiddlewareNext<'_>| {
        let max_bytes = max_bytes;
        let filter = filter;
        Box::pin(async move {
            let mut new_req = req.clone();
            let mut new_messages = Vec::new();

            for message in std::mem::take(&mut new_req.messages) {
                let mut new_content: Vec<Part> = Vec::new();
                for part in message.content {
                    if let Some(media) = &part.media {
                        if media.url.starts_with("http") && filter.is_none_or(|f| f(&part)) {
                            let response = reqwest::get(&media.url).await.map_err(|e| {
                                Error::new_internal(format!(
                                    "HTTP error downloading media '{}': {}",
                                    &media.url, e
                                ))
                            })?;

                            if !response.status().is_success() {
                                return Err(Error::new_internal(format!(
                                    "HTTP error downloading media '{}': {}",
                                    &media.url,
                                    response.text().await.unwrap_or_default()
                                )));
                            }

                            let content_type = response
                                .headers()
                                .get("content-type")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("")
                                .to_string();

                            let body_bytes = response.bytes().await.map_err(|e| {
                                Error::new_internal(format!("Failed to read media bytes: {}", e))
                            })?;

                            if let Some(limit) = max_bytes {
                                if body_bytes.len() > limit {
                                    return Err(Error::new_internal(format!(
                                        "Downloaded media exceeds size limit of {} bytes",
                                        limit
                                    )));
                                }
                            }
                            let encoded = general_purpose::STANDARD.encode(&body_bytes);
                            let new_url = format!("data:{};base64,{}", content_type, encoded);
                            new_content.push(Part {
                                media: Some(crate::Media {
                                    url: new_url,
                                    content_type: Some(content_type),
                                }),
                                ..Default::default()
                            });
                        } else {
                            new_content.push(part);
                        }
                    } else {
                        new_content.push(part);
                    }
                }
                new_messages.push(MessageData {
                    content: new_content,
                    ..message
                });
            }
            new_req.messages = new_messages;
            next(new_req).await
        })
    })
}

/// Validates that a `GenerateRequest` does not include unsupported features.
pub fn validate_support(name: String, supports: ModelInfoSupports) -> ModelMiddleware {
    Arc::new(move |req: GenerateRequest, next: ModelMiddlewareNext<'_>| {
        let name = name.clone();
        let supports = supports.clone();
        Box::pin(async move {
            let invalid = |message: &str| -> Error {
                Error::new_internal(format!(
                    "Model '{}' does not support {}. Request: {}",
                    name,
                    message,
                    serde_json::to_string_pretty(&req)
                        .unwrap_or_else(|_| "Unserializable request".to_string())
                ))
            };

            if supports.media == Some(false)
                && req
                    .messages
                    .iter()
                    .any(|m| m.content.iter().any(|p| p.media.is_some()))
            {
                return Err(invalid("media, but media was provided"));
            }

            if supports.tools == Some(false) && req.tools.as_ref().is_some_and(|t| !t.is_empty()) {
                return Err(invalid("tool use, but tools were provided"));
            }

            if supports.multiturn == Some(false) && req.messages.len() > 1 {
                let len = req.messages.len();
                return Err(invalid(&format!(
                    "multiple messages, but {} were provided",
                    len
                )));
            }
            next(req).await
        })
    })
}

#[derive(Default, Clone)]
pub struct SystemPromptSimulateOptions {
    pub preface: Option<String>,
    pub acknowledgement: Option<String>,
}

/// Provide a simulated system prompt for models that don't support it natively.
pub fn simulate_system_prompt(options: Option<SystemPromptSimulateOptions>) -> ModelMiddleware {
    let opts = options.unwrap_or_default();
    let preface = opts
        .preface
        .unwrap_or_else(|| "SYSTEM INSTRUCTIONS:\n".to_string());
    let acknowledgement = opts
        .acknowledgement
        .unwrap_or_else(|| "Understood.".to_string());

    Arc::new(
        move |mut req: GenerateRequest, next: ModelMiddlewareNext<'_>| {
            let preface = preface.clone();
            let acknowledgement = acknowledgement.clone();
            Box::pin(async move {
                if let Some(pos) = req
                    .messages
                    .iter()
                    .position(|m| m.role == crate::model::Role::System)
                {
                    let system_message = req.messages.remove(pos);
                    let mut user_content = vec![Part {
                        text: Some(preface),
                        ..Default::default()
                    }];
                    user_content.extend(system_message.content);

                    let user_message = MessageData {
                        role: crate::model::Role::User,
                        content: user_content,
                        metadata: None,
                    };
                    let model_message = MessageData {
                        role: crate::model::Role::Model,
                        content: vec![Part {
                            text: Some(acknowledgement),
                            ..Default::default()
                        }],
                        metadata: None,
                    };

                    req.messages
                        .splice(pos..pos, vec![user_message, model_message]);
                }
                next(req).await
            })
        },
    )
}

#[derive(Default, Clone)]
pub struct AugmentWithContextOptions {
    pub preface: Option<String>,
    pub citation_key: Option<String>,
}

pub const CONTEXT_PREFACE: &str = "\n\nUse the following information to complete your task:\n\n";

fn default_item_template(
    d: &Document,
    index: usize,
    options: &AugmentWithContextOptions,
) -> String {
    let mut out = "- ".to_string();
    let citation = if let Some(key) = &options.citation_key {
        d.metadata
            .as_ref()
            .and_then(|m| m.get(key))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        d.metadata.as_ref().and_then(|m| {
            m.get("ref")
                .or_else(|| m.get("id"))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
    }
    .unwrap_or_else(|| index.to_string());

    out.push_str(&format!("[{}]: ", citation));
    out.push_str(&d.text());
    out.push('\n');
    out
}

/// Injects retrieved documents as context into the last user message.
pub fn augment_with_context(options: Option<AugmentWithContextOptions>) -> ModelMiddleware {
    let opts = options.unwrap_or_default();
    Arc::new(
        move |mut req: GenerateRequest, next: ModelMiddlewareNext<'_>| {
            let opts = opts.clone();
            Box::pin(async move {
                let docs = match req.docs.as_ref() {
                    Some(d) if !d.is_empty() => d,
                    _ => return next(req).await,
                };

                if let Some(user_message) = req
                    .messages
                    .iter_mut()
                    .rfind(|m| m.role == crate::model::Role::User)
                {
                    // Check if context part already exists and is not pending
                    let context_part_index = user_message.content.iter().position(|p| {
                        p.metadata
                            .as_ref()
                            .is_some_and(|m| m.get("purpose") == Some(&"context".into()))
                    });

                    if let Some(idx) = context_part_index {
                        let is_pending = user_message.content[idx]
                            .metadata
                            .as_ref()
                            .and_then(|m| m.get("pending"))
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false);
                        if !is_pending {
                            return next(req).await;
                        }
                    }

                    let preface = opts
                        .preface
                        .clone()
                        .unwrap_or_else(|| CONTEXT_PREFACE.to_string());
                    let mut context_text = preface;
                    for (i, doc_data) in docs.iter().enumerate() {
                        let doc = Document {
                            content: vec![doc_data.clone()],
                            metadata: doc_data.metadata.clone(),
                        };
                        context_text.push_str(&default_item_template(&doc, i, &opts));
                    }
                    context_text.push('\n');

                    let mut context_metadata = std::collections::HashMap::new();
                    context_metadata.insert(
                        "purpose".to_string(),
                        serde_json::Value::String("context".to_string()),
                    );
                    let new_part = Part {
                        text: Some(context_text),
                        metadata: Some(context_metadata),
                        ..Default::default()
                    };

                    if let Some(idx) = context_part_index {
                        // Replace pending part
                        user_message.content[idx] = new_part;
                    } else {
                        // Append new part
                        user_message.content.push(new_part);
                    }
                }
                next(req).await
            })
        },
    )
}

#[derive(Serialize, Deserialize, Debug, Default, Clone)]
#[serde(rename_all = "camelCase")]
struct OutputMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    constrained: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_type: Option<String>,
}

pub type InstructionsRenderer = Box<dyn Fn(&serde_json::Value) -> String + Send + Sync>;

#[derive(Default)]
pub struct SimulatedConstrainedGenerationOptions {
    pub instructions_renderer: Option<InstructionsRenderer>,
}

#[allow(unused)]
fn default_constrained_generation_instructions(schema: &serde_json::Value) -> String {
    format!(
        "Output should be in JSON format and conform to the following schema:\n\n```\n{}\n```\n",
        serde_json::to_string_pretty(schema).unwrap_or_default()
    )
}

/// Model middleware that simulates constrained generation by injecting generation
/// instructions into the user message.
pub fn simulate_constrained_generation(
    options: Option<SimulatedConstrainedGenerationOptions>,
) -> ModelMiddleware {
    let options = Arc::new(options.unwrap_or_default());
    Arc::new(
        move |mut req: GenerateRequest, next: ModelMiddlewareNext<'_>| {
            let options = Arc::clone(&options);
            Box::pin(async move {
                let mut instructions: Option<String> = None;
                if let Some(output_str) = &req.output {
                    if let Ok(mut output_meta) = serde_json::from_str::<OutputMetadata>(output_str)
                    {
                        if output_meta.constrained == Some(true) {
                            if let Some(schema) = &output_meta.schema {
                                if let Some(renderer) = &options.instructions_renderer {
                                    instructions = Some(renderer(schema));
                                } else {
                                    instructions =
                                        Some(default_constrained_generation_instructions(schema));
                                }

                                output_meta.constrained = Some(false);
                                output_meta.format = None;
                                output_meta.content_type = None;
                                output_meta.schema = None;
                                req.output = serde_json::to_string(&output_meta).ok();
                            }
                        }
                    }
                }

                if let Some(instr) = instructions {
                    if let Some(user_message) =
                        req.messages.iter_mut().rfind(|m| m.role == Role::User)
                    {
                        let mut metadata = std::collections::HashMap::new();
                        metadata.insert("purpose".to_string(), "output".into());
                        user_message.content.push(Part {
                            text: Some(instr),
                            metadata: Some(metadata),
                            ..Default::default()
                        });
                    }
                }
                next(req).await
            })
        },
    )
}
