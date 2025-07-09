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

//! # Prompts
//!
//! This module provides the `define_prompt` function and related structures for
//! creating and executing type-safe, templated prompts. It is the Rust
// equivalent of `prompt.ts`.

use crate::document::{Document, Part};
use crate::generate::{
    generate, generate_stream, to_generate_request, GenerateOptions, GenerateResponse,
    GenerateStreamResponse, OutputOptions,
};
use crate::message::MessageData;
use crate::model::{middleware::ModelMiddleware, GenerateRequest, Model};

use crate::tool::ToolArgument;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::context::{get_context, ActionContext};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, ErasedAction, Registry};
use handlebars::Handlebars;
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// A type alias for an action that renders a prompt.
pub type PromptAction<I = Value> = Action<I, GenerateRequest, ()>;

/// A type alias for a function that resolves documents dynamically for a prompt.
pub type DocsResolver<I> = Arc<
    dyn Fn(
            I,
            Option<Value>,
            Option<ActionContext>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<Document>>> + Send>>
        + Send
        + Sync,
>;

/// A type alias for a function that resolves a string dynamically for a prompt.
pub type StringResolver<I> = Arc<
    dyn Fn(
            I,
            Option<Value>,
            Option<ActionContext>,
        ) -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

/// A type alias for a function that resolves messages dynamically for a prompt.
pub type MessagesResolver<I> = Arc<
    dyn Fn(
            I,
            Option<Value>,
            Option<ActionContext>,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<MessageData>>> + Send>>
        + Send
        + Sync,
>;

/// Configuration for a prompt action.
#[derive(Default, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PromptConfig<I = Value, O = Value, C = Value> {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<Model>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<C>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<I>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(skip, default)]
    #[schemars(skip)]
    pub system_fn: Option<StringResolver<I>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(skip, default)]
    #[schemars(skip)]
    pub prompt_fn: Option<StringResolver<I>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<MessageData>>,
    #[serde(skip, default)]
    #[schemars(skip)]
    pub messages_fn: Option<MessagesResolver<I>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<Vec<Document>>,
    #[serde(skip, default)]
    #[schemars(skip)]
    pub docs_fn: Option<DocsResolver<I>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolArgument>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub _marker: Option<std::marker::PhantomData<O>>,
}

// Manual Debug implementation because Box<dyn Fn(...)> is not Debug.
impl<I, O, C> fmt::Debug for PromptConfig<I, O, C>
where
    I: fmt::Debug,
    O: fmt::Debug,
    C: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptConfig")
            .field("name", &self.name)
            .field("variant", &self.variant)
            .field("model", &self.model)
            .field("config", &self.config)
            .field("description", &self.description)
            .field("input", &self.input)
            .field("system", &self.system)
            .field("system_fn", &self.system_fn.as_ref().map(|_| "Some(<fn>)"))
            .field("prompt", &self.prompt)
            .field("prompt_fn", &self.prompt_fn.as_ref().map(|_| "Some(<fn>)"))
            .field("messages", &self.messages)
            .field(
                "messages_fn",
                &self.messages_fn.as_ref().map(|_| "Some(<fn>)"),
            )
            .field("docs", &self.docs)
            .field("docs_fn", &self.docs_fn.as_ref().map(|_| "Some(<fn>)"))
            .field("output", &self.output)
            .field("tools", &self.tools)
            .field("_marker", &self._marker)
            .finish()
    }
}

/// Options for generating a response from a prompt.
#[derive(Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptGenerateOptions<O = Value> {
    pub config: Option<Value>,
    pub messages: Option<Vec<MessageData>>,
    pub output: Option<OutputOptions>,
    pub context: Option<ActionContext>,
    /// Middleware to be used with this model call.
    #[serde(skip)]
    pub r#use: Option<Vec<ModelMiddleware>>,
    #[serde(skip)]
    pub _marker: std::marker::PhantomData<O>,
}

// Manual Clone implementation because ModelMiddleware is not derived.
impl<O> Clone for PromptGenerateOptions<O> {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            messages: self.messages.clone(),
            output: self.output.clone(),
            context: self.context.clone(),
            r#use: self.r#use.clone(),
            _marker: self._marker,
        }
    }
}

// Manual Debug implementation because ModelMiddleware (Arc<dyn Fn(...)>) is not Debug.
impl<O> fmt::Debug for PromptGenerateOptions<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PromptGenerateOptions")
            .field("config", &self.config)
            .field("messages", &self.messages)
            .field("output", &self.output)
            .field("context", &self.context)
            .field("use", &self.r#use.as_ref().map(|v| v.len())) // Print length instead of the functions
            .field("_marker", &self._marker)
            .finish()
    }
}

/// A prompt that can be executed as a function.
#[derive(Clone)]
pub struct ExecutablePrompt<I = Value, O = Value, C = Value> {
    config: Arc<PromptConfig<I, O, C>>,
    registry: Arc<Registry>,
}

impl<I, O, C> ExecutablePrompt<I, O, C>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: for<'de> Deserialize<'de>
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + Clone
        + Default
        + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    /// Generates a response by rendering the prompt and calling the model.
    pub async fn generate(
        &self,
        input: I,
        opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateResponse<O>> {
        let render_opts = self.render(input, opts).await?;
        generate::<O>(&self.registry, render_opts).await
    }

    /// Generates a streaming response.
    pub async fn stream(
        &self,
        input: I,
        opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateStreamResponse<O>> {
        let render_opts = self.render(input, opts).await?;
        generate_stream::<O>(&self.registry, render_opts).await
    }

    /// Renders the prompt template into a `GenerateOptions` struct.
    pub async fn render(
        &self,
        input: I,
        opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateOptions<O>> {
        let mut handlebars = Handlebars::new();
        // Allow missing fields to match JS behavior (e.g. {{foo}} is ok if foo is not provided)
        handlebars.set_strict_mode(false);

        // 1. Prepare render data from input, session state, and context
        let mut render_data = serde_json::to_value(input.clone())
            .map_err(|e| Error::new_internal(format!("Failed to serialize input: {}", e)))?;

        let session = crate::session::get_current_session().ok();
        let state = if let Some(s) = &session {
            s.state().await
        } else {
            None
        };

        let global_context = get_context();
        let options_context = opts.as_ref().and_then(|o| o.context.clone());

        let mut resolver_context = global_context;
        if let Some(opts_ctx) = options_context.as_ref() {
            if let Some(res_ctx) = &mut resolver_context {
                res_ctx.extend(opts_ctx.clone());
            } else {
                resolver_context = Some(opts_ctx.clone());
            }
        }

        if let Some(data_obj) = render_data.as_object_mut() {
            if let Some(state_val) = state.clone() {
                data_obj.insert("state".to_string(), state_val.clone());
                data_obj.insert("@state".to_string(), state_val);
            }
            if let Some(context) = &resolver_context {
                if let Some(auth_val) = context.get("auth") {
                    data_obj.insert("auth".to_string(), auth_val.clone());
                }
            }
        }

        // 2. Build the message list in the correct order
        let mut messages = Vec::new();

        // System Prompt
        if let Some(resolver) = &self.config.system_fn {
            let text = resolver(input.clone(), state.clone(), resolver_context.clone()).await?;
            messages.push(MessageData::system(vec![Part::text(text)]));
        } else if let Some(system_template) = &self.config.system {
            let system_text = handlebars
                .render_template(system_template, &render_data)
                .map_err(|e| {
                    Error::new_internal(format!("Failed to render system template: {}", e))
                })?;
            messages.push(MessageData::system(vec![Part::text(system_text)]));
        }

        // Messages
        if let Some(resolver) = &self.config.messages_fn {
            let resolved_messages =
                resolver(input.clone(), state.clone(), resolver_context.clone()).await?;
            messages.extend(resolved_messages);
        } else if let Some(config_messages) = &self.config.messages {
            messages.extend(config_messages.clone());
        }
        if let Some(opts_messages) = opts.as_ref().and_then(|o| o.messages.as_ref()) {
            messages.extend(opts_messages.clone());
        }

        // Main User Prompt
        if let Some(resolver) = &self.config.prompt_fn {
            let text = resolver(input.clone(), state.clone(), resolver_context.clone()).await?;
            messages.push(MessageData::user(vec![Part::text(text)]));
        } else if let Some(prompt_template) = &self.config.prompt {
            let prompt_text = handlebars
                .render_template(prompt_template, &render_data)
                .map_err(|e| {
                    Error::new_internal(format!("Failed to render prompt template: {}", e))
                })?;
            messages.push(MessageData::user(vec![Part::text(prompt_text)]));
        }

        // 3. Resolve docs
        let docs = if let Some(docs_fn) = &self.config.docs_fn {
            docs_fn(input, state, resolver_context).await?
        } else {
            self.config.docs.clone().unwrap_or_default()
        };

        // 4. Merge configs
        let mut final_config = self
            .config
            .config
            .as_ref()
            .and_then(|c| serde_json::to_value(c).ok())
            .unwrap_or_else(|| json!({}));

        if let Some(opts_config) = opts.as_ref().and_then(|o| o.config.as_ref()) {
            if let (Some(final_obj), Some(opts_obj)) =
                (final_config.as_object_mut(), opts_config.as_object())
            {
                final_obj.extend(opts_obj.clone());
            }
        }
        let final_config = if final_config.as_object().is_some_and(|m| !m.is_empty()) {
            Some(final_config)
        } else {
            None
        };

        // 5. Construct final GenerateOptions
        let gen_opts = GenerateOptions {
            model: self.config.model.clone(),
            messages: Some(messages),
            tools: self.config.tools.clone(),
            docs: if docs.is_empty() { None } else { Some(docs) },
            config: final_config,
            output: self.config.output.clone(),
            context: opts.as_ref().and_then(|o| o.context.clone()),
            r#use: opts.and_then(|o| o.r#use),
            ..Default::default()
        };

        Ok(gen_opts)
    }
}

/// Defines a prompt which can be used to generate content or render a request.
pub fn define_prompt<I, O, C>(
    registry: &mut Registry,
    config: PromptConfig<I, O, C>,
) -> ExecutablePrompt<I, O, C>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: for<'de> Deserialize<'de>
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + Clone
        + Default
        + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    let config_arc = Arc::new(config);
    let registry_arc = Arc::new(registry.clone());

    let prompt_action = {
        let config_clone = config_arc.clone();
        let registry_clone = registry_arc.clone();
        let name = config_clone.name.clone();

        ActionBuilder::new(
            ActionType::Prompt,
            name,
            move |input: I, _ctx: genkit_core::action::ActionFnArg<()>| {
                let config = config_clone.clone();
                let registry = registry_clone.clone();
                async move {
                    let prompt = ExecutablePrompt { config, registry };
                    let gen_opts = prompt.render(input, None).await?;
                    to_generate_request(&prompt.registry, &gen_opts).await
                }
            },
        )
        .build()
    };

    let _ = registry.register_action(prompt_action.clone().name(), prompt_action);

    ExecutablePrompt {
        config: config_arc,
        registry: registry_arc,
    }
}

/// Helper to check if a value is an `ExecutablePrompt`.
pub fn is_executable_prompt<I, O, C>(_: &ExecutablePrompt<I, O, C>) -> bool {
    true
}

/// A placeholder function for `prompt()`, which would look up a defined prompt.
pub async fn prompt<I, O, C>(registry: &Registry, name: &str) -> Result<ExecutablePrompt<I, O, C>>
where
    I: Serialize + DeserializeOwned + JsonSchema + Send + Sync + Clone + 'static,
    O: for<'de> Deserialize<'de>
        + Serialize
        + Send
        + Sync
        + std::fmt::Debug
        + Clone
        + Default
        + 'static,
    C: Serialize + DeserializeOwned + JsonSchema + Send + Sync + 'static,
{
    let _action = registry
        .lookup_action(&format!("/prompt/{}", name))
        .await
        .ok_or_else(|| Error::new_internal(format!("Prompt '{}' not found", name)))?;
    unimplemented!(
        "Looking up prompts by name is not fully supported yet without a way to retrieve the original config from the registry."
    )
}
