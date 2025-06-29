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

//! # Genkit Prompt
//!
//! This module provides structures and traits for defining and working with
//! prompts. A prompt is a template that can be rendered with input to produce
// a `GenerateRequest` for a model. This is the Rust equivalent of prompt
// functionality in `@genkit-ai/ai`.

use crate::error::{Error, Result};
use crate::model::{self, GenerateRequest, GenerateResponse, Message};
use crate::tool::ToolDefinition;
use async_trait::async_trait;
use handlebars::Handlebars;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Represents the configuration for a prompt, including model, tools, and output format.
#[derive(Serialize, Deserialize, Debug, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct PromptConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>, // TODO: Define OutputConfig struct
}

/// The definition of a prompt, including its template and configuration.
#[derive(Clone)]
pub struct PromptDefinition<I>
where
    I: Serialize,
{
    pub name: String,
    pub description: String,
    pub config: PromptConfig,
    /// The function that renders the prompt's messages.
    /// It takes serializable input and returns a vector of `Message`s.
    pub messages_fn: std::sync::Arc<dyn Fn(I) -> Result<Vec<Message>> + Send + Sync>,
}

/// A trait for an executable prompt.
///
/// Implementations of this trait can take input data, render a prompt,
/// and then execute it by calling a model, returning the model's response.
#[async_trait]
pub trait Prompt<I>: Send + Sync
where
    I: Serialize + Send,
{
    /// Renders the prompt into a `GenerateRequest` suitable for sending to a model.
    async fn render(&self, input: I) -> Result<GenerateRequest>;

    /// Renders and then executes the prompt, returning a `GenerateResponse`.
    async fn generate(&self, input: I) -> Result<GenerateResponse>;
}

#[async_trait]
impl<I> Prompt<I> for PromptDefinition<I>
where
    I: Serialize + Send,
{
    async fn render(&self, input: I) -> Result<GenerateRequest> {
        let messages = (self.messages_fn)(input)?;
        Ok(GenerateRequest {
            messages,
            config: self.config.config.clone(),
            // TODO: Handle tools and output config from self.config
        })
    }

    async fn generate(&self, input: I) -> Result<GenerateResponse> {
        let request = self.render(input).await?;
        // In a real implementation, this would look up the model
        // from the request/registry and call it.
        model::generate(request).await
    }
}

/// A simple prompt that uses a Handlebars template string for its messages.
pub struct TemplatePrompt {
    pub definition: PromptDefinition<Value>,
    pub template: String,
}

impl TemplatePrompt {
    /// Renders the Handlebars template string with the given JSON value.
    fn render_template(&self, input: &Value) -> Result<String> {
        let reg = Handlebars::new();
        let rendered_str = reg
            .render_template(&self.template, input)
            .map_err(|e| Error::new_internal(format!("Failed to render template: {}", e)))?;
        Ok(rendered_str)
    }
}

#[async_trait]
impl Prompt<Value> for TemplatePrompt
where
    Value: Send,
{
    async fn render(&self, input: Value) -> Result<GenerateRequest> {
        let rendered_text = self.render_template(&input)?;
        let messages = vec![model::Message {
            role: model::Role::User,
            content: vec![model::Part::Text(model::TextPart {
                text: rendered_text,
            })],
        }];
        Ok(GenerateRequest {
            messages,
            config: self.definition.config.config.clone(),
        })
    }

    async fn generate(&self, input: Value) -> Result<GenerateResponse> {
        let request = self.render(input).await?;
        model::generate(request).await
    }
}

/// Defines a new prompt from a template string and configuration.
pub fn define_prompt<I: Serialize + Send + 'static>(
    def: PromptDefinition<I>,
) -> Box<dyn Prompt<I> + 'static> {
    Box::new(def)
}
