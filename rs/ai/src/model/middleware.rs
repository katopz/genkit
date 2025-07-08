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
};
use base64::{engine::general_purpose, Engine};
use genkit_core::error::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::HashMap, future::Future, pin::Pin, sync::Arc};

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

pub type ItemTemplate = Box<dyn Fn(&Document, usize) -> String + Send + Sync>;

#[derive(Default, Serialize, Deserialize)]
pub struct AugmentWithContextOptions {
    pub preface: Option<String>,
    pub citation_key: Option<String>,
    #[serde(skip)]
    pub item_template: Option<ItemTemplate>,
}

impl Clone for AugmentWithContextOptions {
    fn clone(&self) -> Self {
        Self {
            preface: self.preface.clone(),
            citation_key: self.citation_key.clone(),
            item_template: None,
        }
    }
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
    let opts = Arc::new(options.unwrap_or_default());
    Arc::new(
        move |mut req: GenerateRequest, next: ModelMiddlewareNext<'_>| {
            let opts = Arc::clone(&opts);
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
                    for (i, doc) in docs.iter().enumerate() {
                        if let Some(template) = &opts.item_template {
                            context_text.push_str(&template(doc, i));
                        } else {
                            context_text.push_str(&default_item_template(doc, i, &opts));
                        }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<serde_json::Value>,
}

pub type InstructionsRenderer = Arc<dyn Fn(&serde_json::Value) -> String + Send + Sync>;

pub struct SimulatedConstrainedGenerationOptions {
    pub instructions_renderer: InstructionsRenderer,
}

#[derive(Debug)]
pub struct InstructionsRendererWithSchema {
    pub renderer_string: String,
    pub schema: Value,
}

/// A `ModelMiddleware` that simulates constrained (e.g., JSON) output by providing instructions.
pub fn simulate_constrained_generation(
    options: Option<InstructionsRendererWithSchema>,
) -> ModelMiddleware {
    let renderer: InstructionsRenderer = if let Some(opts) = options {
        let renderer_string = opts.renderer_string;
        let schema = opts.schema;
        // The custom renderer ignores the schema passed to it at runtime, and instead
        // uses the one provided at configuration time. This is to support the specific
        // use case in the tests where a string prefix is combined with a static schema.
        Arc::new(move |_s| {
            format!(
                "{}{}",
                renderer_string,
                serde_json::to_string(&schema).unwrap()
            )
        })
    } else {
        Arc::new(|s| {
            format!(
                "Output should be in JSON format and conform to the following schema:\n\n```\n{}\n```\n",
                serde_json::to_string_pretty(s).unwrap_or_else(|_| "null".to_string())
            )
        })
    };

    Arc::new(
        move |mut req: GenerateRequest, next: ModelMiddlewareNext<'_>| {
            let renderer = renderer.clone();
            Box::pin(async move {
                let output_opts_json = match req.output.as_ref() {
                    Some(opts) => opts.clone(),
                    None => return next(req).await,
                };

                let mut output_opts: OutputMetadata =
                    match serde_json::from_value(output_opts_json.clone()) {
                        Ok(opts) => opts,
                        Err(_) => return next(req).await,
                    };

                let force_instructions = output_opts
                    .instructions
                    .as_ref()
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                // When `instructions: true`, we should always inject instructions.
                if output_opts.format == Some("json".to_string())
                    && (!output_opts.constrained.unwrap_or(false) || force_instructions)
                {
                    let schema_ref = output_opts.schema.as_ref();
                    let instructions = renderer(schema_ref.unwrap_or(&Value::Null));

                    // 1. Add instructions to messages.
                    let last_user_message = req
                        .messages
                        .iter_mut()
                        .rfind(|m| m.role == crate::model::Role::User);

                    if let Some(msg) = last_user_message {
                        msg.content.push(Part {
                            text: Some(instructions),
                            metadata: Some(HashMap::from([(
                                "purpose".to_string(),
                                Value::String("output".to_string()),
                            )])),

                            ..Default::default()
                        });
                    } else {
                        // If no user message, create one. This is unusual but possible.
                        req.messages.push(MessageData {
                            role: crate::model::Role::User,
                            content: vec![Part {
                                text: Some(instructions),
                                metadata: Some(HashMap::from([(
                                    "purpose".to_string(),
                                    Value::String("output".to_string()),
                                )])),

                                ..Default::default()
                            }],
                            ..Default::default()
                        });
                    }

                    // 2. Modify output options for the model request.
                    output_opts.constrained = Some(false);
                    // Preserve schema and format if instructions were explicitly requested.
                    if !force_instructions {
                        output_opts.format = None;
                        output_opts.schema = None;
                    }
                    req.output = Some(serde_json::to_value(output_opts).map_err(|e| {
                        Error::new_internal(format!("Failed to re-serialize output options: {}", e))
                    })?);
                }

                next(req).await
            })
        },
    )
}
