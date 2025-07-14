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
//! equivalent of `prompt.ts`.

use crate::generate::{
    generate, generate_stream, to_generate_request, GenerateOptions, GenerateResponse,
    GenerateStreamResponse, OutputOptions,
};
use crate::message::MessageData;
use crate::model::{middleware::ModelMiddleware, GenerateRequest, Model};

use crate::tool::ToolArgument;
use crate::{Document, Part, ToolChoice};
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::context::{get_context, ActionContext};
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
use handlebars::Handlebars;
use schemars::{self, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::any::Any;
use std::collections::HashMap;
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
#[derive(Default, Serialize, Deserialize, JsonSchema, Clone)]
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
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolArgument>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_tool_requests: Option<bool>,
    #[serde(skip, default)]
    #[schemars(skip)]
    pub r#use: Option<Vec<ModelMiddleware>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
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
            .field("metadata", &self.metadata)
            .field("tools", &self.tools)
            .field("max_turns", &self.max_turns)
            .field("return_tool_requests", &self.return_tool_requests)
            .field("use", &self.r#use.as_ref().map(|u| u.len()))
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptRef {
    pub name: String,
}

/// A prompt that can be executed as a function.
#[derive(Clone)]
pub struct ExecutablePrompt<I = Value, O = Value, C = Value> {
    pub config: Arc<PromptConfig<I, O, C>>,
    pub registry: Registry,
    pub r#ref: Option<PromptRef>,
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
        mut opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateResponse<O>> {
        if opts.as_ref().and_then(|o| o.context.as_ref()).is_none() {
            if let Some(global_context) = get_context() {
                let mut new_opts = opts.take().unwrap_or_default();
                new_opts.context = Some(global_context);
                opts = Some(new_opts);
            }
        }
        let render_opts = self.render(input, opts).await?;
        generate::<O>(&self.registry, render_opts).await
    }

    /// Generates a streaming response.
    pub async fn stream(
        &self,
        input: I,
        mut opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateStreamResponse<O>> {
        if opts.as_ref().and_then(|o| o.context.as_ref()).is_none() {
            if let Some(global_context) = get_context() {
                let mut new_opts = opts.take().unwrap_or_default();
                new_opts.context = Some(global_context);
                opts = Some(new_opts);
            }
        }
        let render_opts = self.render(input, opts).await?;
        generate_stream::<O>(&self.registry, render_opts).await
    }

    /// Renders the prompt template into a `GenerateOptions` struct.
    pub async fn render(
        &self,
        input: I,
        opts: Option<PromptGenerateOptions<O>>,
    ) -> Result<GenerateOptions<O>> {
        if (self.config.prompt.is_some() || self.config.prompt_fn.is_some())
            && (self.config.messages.is_some() || self.config.messages_fn.is_some())
        {
            return Err(Error::new_internal(
                "Prompt cannot have both `prompt` and `messages` fields defined.",
            ));
        }

        let mut handlebars = Handlebars::new();
        // Allow missing fields to match JS behavior (e.g. {{foo}} is ok if foo is not provided)
        handlebars.set_strict_mode(false);

        // 1. Prepare render data from input, session state, and context
        let mut render_data = serde_json::to_value(input.clone())
            .map_err(|e| Error::new_internal(format!("Failed to serialize input: {}", e)))?;

        if !render_data.is_object() {
            render_data = json!({ "input": render_data });
        }

        let session = crate::session::get_current_session().ok();
        let state = if let Some(s) = session {
            s.state().await
        } else {
            None
        };

        let global_context = get_context();
        println!("[LOG] global_context: {:?}", global_context);
        let options_context = opts.as_ref().and_then(|o| o.context.clone());
        println!("[LOG] options_context: {:?}", options_context);

        let resolver_context = match (global_context, options_context) {
            (Some(mut global), Some(options)) => {
                global.extend(options);
                Some(global)
            }
            (Some(global), None) => Some(global),
            (None, Some(options)) => Some(options),
            (None, None) => None,
        };
        println!("[LOG] final resolver_context: {:?}", resolver_context);

        if let Some(data_obj) = render_data.as_object_mut() {
            if let Some(state_val) = state.clone() {
                data_obj.insert("state".to_string(), state_val);
            }
            if let Some(context) = &resolver_context {
                if let Some(Value::Object(auth_map)) = &context.auth {
                    println!("[LOG] Extending render_data with auth_map: {:?}", auth_map);
                    data_obj.extend(auth_map.clone());
                }
            }
        }
        println!(
            "[LOG] Final render_data before templates: {}",
            serde_json::to_string_pretty(&render_data).unwrap_or_default()
        );

        // 2. Build the message list in the correct order
        let mut messages = Vec::new();

        // System Prompt
        if let Some(resolver) = &self.config.system_fn {
            let text = resolver(input.clone(), state.clone(), resolver_context.clone()).await?;
            messages.push(MessageData::system(vec![Part::text(text)]));
        } else if let Some(system_template) = &self.config.system {
            println!("[LOG] Rendering system_template: '{}'", system_template);
            let template = system_template.replace("@state", "state");
            let system_text = handlebars
                .render_template(&template, &render_data)
                .map_err(|e| {
                    Error::new_internal(format!("Failed to render system template: {}", e))
                })?;
            messages.push(MessageData::system(vec![Part::text(system_text)]));
        }

        // Messages
        if let Some(opts_messages) = opts.as_ref().and_then(|o| o.messages.as_ref()) {
            messages.extend(opts_messages.clone());
        }
        if let Some(resolver) = &self.config.messages_fn {
            let resolved_messages =
                resolver(input.clone(), state.clone(), resolver_context.clone()).await?;
            messages.extend(resolved_messages);
        } else if let Some(config_messages) = &self.config.messages {
            let mut rendered_messages = Vec::new();
            for msg_template in config_messages {
                let mut new_msg = msg_template.clone();
                let mut rendered_content = Vec::new();
                for part_template in &msg_template.content {
                    let mut new_part = part_template.clone();
                    if let Some(text_template) = &part_template.text {
                        println!("[LOG] Rendering message part template: '{}'", text_template);
                        let template = text_template.replace("@state", "state");
                        let rendered_text = handlebars
                            .render_template(&template, &render_data)
                            .map_err(|e| {
                                Error::new_internal(format!(
                                    "Failed to render message template: {}",
                                    e
                                ))
                            })?;
                        new_part.text = Some(rendered_text);
                    }
                    rendered_content.push(new_part);
                }
                new_msg.content = rendered_content;
                rendered_messages.push(new_msg);
            }
            messages.extend(rendered_messages);
        }

        // Main User Prompt
        if let Some(resolver) = &self.config.prompt_fn {
            let text = resolver(input.clone(), state.clone(), resolver_context.clone()).await?;
            messages.push(MessageData::user(vec![Part::text(text)]));
        } else if let Some(prompt_template) = &self.config.prompt {
            println!(
                "[LOG] Rendering main prompt template: '{}'",
                prompt_template
            );
            let template = prompt_template.replace("@state", "state");
            let prompt_text = handlebars
                .render_template(&template, &render_data)
                .map_err(|e| {
                    Error::new_internal(format!("Failed to render prompt template: {}", e))
                })?;
            messages.push(MessageData::user(vec![Part::text(prompt_text)]));
        }

        // 3. Resolve docs
        let docs = if let Some(docs_fn) = &self.config.docs_fn {
            docs_fn(input.clone(), state, resolver_context.clone()).await?
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
        let mut gen_opts = GenerateOptions {
            model: self.config.model.clone(),
            messages: Some(messages),
            tools: self.config.tools.clone(),
            docs: if docs.is_empty() { None } else { Some(docs) },
            config: final_config,
            output: self.config.output.clone(),
            context: resolver_context,
            max_turns: self.config.max_turns,
            return_tool_requests: self.config.return_tool_requests,
            r#use: {
                let mut middleware = self.config.r#use.clone().unwrap_or_default();
                if let Some(opts_middleware) = opts.as_ref().and_then(|o| o.r#use.as_ref()) {
                    middleware.extend(opts_middleware.iter().cloned());
                }
                if middleware.is_empty() {
                    None
                } else {
                    Some(middleware)
                }
            },
            ..Default::default()
        };

        if let Some(output_opts) = &mut gen_opts.output {
            if let Some(json_schema) = &output_opts.json_schema {
                if let Some(obj) = json_schema.as_object() {
                    if let Some(Value::String(ref_path)) = obj.get("$ref") {
                        let schema_name = ref_path.strip_prefix("schema/").unwrap_or(ref_path);
                        let looked_up_schema =
                            self.registry.lookup_any("schema", schema_name).await;
                        if let Some(schema_any) = looked_up_schema {
                            if let Ok(schema_arc) = schema_any.downcast::<schemars::Schema>() {
                                let schema_value = serde_json::to_value(&*schema_arc)?;
                                output_opts.json_schema = Some(schema_value);
                            }
                        } else {
                            return Err(Error::new_internal(format!(
                                "NOT_FOUND: Schema '{}' not found",
                                schema_name
                            )));
                        }
                    }
                }
            }
        }

        Ok(gen_opts)
    }
}

/// Defines a prompt which can be used to generate content or render a request.
pub fn define_prompt<I, O, C>(
    registry: &Registry,
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
    let registry_clone = registry.clone();

    let key_name = if let Some(variant) = &config_arc.variant {
        format!("{}.{}", &config_arc.name, variant)
    } else {
        config_arc.name.clone()
    };

    let mut action_metadata = config_arc
        .metadata
        .clone()
        .unwrap_or(Value::Object(Default::default()));

    let mut prompt_metadata_map =
        if let Some(Value::Object(map)) = action_metadata.get("prompt").cloned() {
            map
        } else {
            serde_json::Map::new()
        };

    if let Some(config_val) = &config_arc.config {
        prompt_metadata_map.insert(
            "config".to_string(),
            serde_json::to_value(config_val).unwrap_or(Value::Null),
        );
    }
    if let Some(desc) = &config_arc.description {
        prompt_metadata_map.insert("description".to_string(), json!(desc));
    }

    let input_schema = schemars::schema_for!(I);
    prompt_metadata_map.insert("input".to_string(), json!({ "schema": input_schema }));

    prompt_metadata_map
        .entry("metadata".to_string())
        .or_insert_with(|| json!({}));

    let model_name = match &config_arc.model {
        Some(Model::Name(n)) => Some(n.clone()),
        Some(Model::Reference(r)) => Some(r.name.clone()),
        None => None,
    };
    prompt_metadata_map.insert("model".to_string(), json!(model_name));

    let base_name = if let Some(idx) = config_arc.name.find('.') {
        &config_arc.name[..idx]
    } else {
        &config_arc.name
    };
    prompt_metadata_map.insert("name".to_string(), json!(base_name));

    let raw_config = json!({
        "config": config_arc.config,
        "description": config_arc.description,
    });
    prompt_metadata_map.insert("raw".to_string(), raw_config);

    if let Some(prompt_str) = &config_arc.prompt {
        prompt_metadata_map.insert("template".to_string(), json!(prompt_str));
    }

    if let Some(variant) = &config_arc.variant {
        prompt_metadata_map.insert("variant".to_string(), json!(variant));
    }

    if let Some(map) = action_metadata.as_object_mut() {
        map.insert("prompt".to_string(), Value::Object(prompt_metadata_map));
        map.insert("type".to_string(), json!("prompt"));
    }

    let final_metadata_map: HashMap<String, Value> =
        serde_json::from_value(action_metadata).unwrap_or_default();

    let prompt_action = {
        let config_clone = config_arc.clone();
        let registry_clone = registry_clone.clone();

        ActionBuilder::new(
            ActionType::Prompt,
            key_name.clone(),
            move |input: I, _ctx: genkit_core::action::ActionFnArg<()>| {
                let config = config_clone.clone();
                let registry = registry_clone.clone();
                async move {
                    let prompt = ExecutablePrompt {
                        config,
                        registry,
                        r#ref: None,
                    };
                    let gen_opts = prompt.render(input, None).await?;
                    to_generate_request(&prompt.registry, &gen_opts).await
                }
            },
        )
        .with_metadata(final_metadata_map)
        .build()
    };

    let _ = registry.register_action(prompt_action.meta.action_type, prompt_action);

    registry.register_any(
        "prompt",
        &key_name,
        config_arc.clone() as Arc<dyn Any + Send + Sync>,
    );

    ExecutablePrompt {
        config: config_arc,
        registry: registry.clone(),
        r#ref: Some(PromptRef { name: key_name }),
    }
}

/// Helper to check if a value is an `ExecutablePrompt`.
pub fn is_executable_prompt<I, O, C>(_: &ExecutablePrompt<I, O, C>) -> bool {
    true
}

/// Options for looking up a prompt.
#[derive(Default)]
pub struct PromptLookupOptions<'a> {
    pub variant: Option<&'a str>,
}

/// A placeholder function for `prompt()`, which would look up a defined prompt.
pub async fn prompt<I, O, C>(
    registry: &Registry,
    name: &str,
    options: Option<PromptLookupOptions<'_>>,
) -> Result<ExecutablePrompt<I, O, C>>
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
    let options = options.unwrap_or_default();
    let key_name = if let Some(v) = options.variant {
        format!("{}.{}", name, v)
    } else {
        name.to_string()
    };
    let config_any = registry
        .lookup_any("prompt", &key_name)
        .await
        .ok_or_else(|| {
            Error::new_internal(format!("Prompt '{}' not found in registry", key_name))
        })?;

    let config = config_any
        .downcast::<PromptConfig<I, O, C>>()
        .map_err(|_| {
            Error::new_internal(format!(
                "Type mismatch for prompt '{}'. Could not downcast to the expected type.",
                key_name
            ))
        })?;

    Ok(ExecutablePrompt {
        config,
        registry: registry.clone(),
        r#ref: Some(PromptRef { name: key_name }),
    })
}
