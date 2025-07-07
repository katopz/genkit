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

//! # Data Resources
//!
//! This module provides the functionality to define and resolve data resources
//! that can be passed to models. It is the Rust equivalent of `resource.ts`.

use crate::document::Part;
use genkit_core::action::{Action, ActionBuilder};
use genkit_core::context::ActionContext;
use genkit_core::error::{Error, Result};
use genkit_core::registry::{ActionType, Registry};
use regex::Regex;
use schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

//
// Core Types & Structs
//

/// Options for defining a resource.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ResourceOptions {
    /// Resource name. If not specified, uri or template will be used as name.
    pub name: Option<String>,
    /// The URI of the resource. Can be a literal string.
    pub uri: Option<String>,
    /// The URI template (ex. `my://resource/{id}`). See RFC6570 for specification.
    pub template: Option<String>,
    /// A description of the resource.
    pub description: Option<String>,
    /// Resource metadata.
    pub metadata: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
pub struct ResourceInput {
    pub uri: String,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, Clone)]
pub struct ResourceOutput {
    pub content: Vec<Part>,
}

/// A function that returns parts for a given resource.
pub type ResourceFn<Fut> = dyn Fn(ResourceInput, ActionContext) -> Fut + Send + Sync + 'static;

/// A resource action.
pub type ResourceAction = Action<ResourceInput, ResourceOutput, ()>;

//
// Define Resource
//

/// Defines a resource.
pub fn define_resource<F, Fut>(
    registry: &mut Registry,
    opts: ResourceOptions,
    runner: F,
) -> Result<Arc<ResourceAction>>
where
    F: Fn(ResourceInput, ActionContext) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<ResourceOutput>> + Send + 'static,
{
    let uri = opts
        .uri
        .clone()
        .or_else(|| opts.template.clone())
        .ok_or_else(|| Error::new_internal("must specify either uri or template options"))?;

    let name = opts.name.clone().unwrap_or_else(|| uri.clone());

    let mut metadata = HashMap::new();
    if let Some(opts_meta) = opts.metadata.as_ref().and_then(|m| m.as_object()) {
        for (key, value) in opts_meta {
            metadata.insert(key.clone(), value.clone());
        }
    }
    let resource_info = json!({
        "uri": opts.uri,
        "template": opts.template,
    });
    metadata.insert("resource".to_string(), resource_info);

    let runner_arc = Arc::new(runner);
    let opts_template = opts.template.clone();
    let runner_closure = move |input: ResourceInput, args: genkit_core::action::ActionFnArg<()>| {
        let runner_clone = runner_arc.clone();
        let input_clone = input.clone();
        let template_clone = opts_template.clone();
        let context = args.context.unwrap_or_default();

        async move {
            let mut output = runner_clone(input_clone, context).await?;

            let mut parent_info = serde_json::Map::new();
            parent_info.insert("uri".to_string(), input.uri.into());
            if let Some(template) = &template_clone {
                parent_info.insert("template".to_string(), template.clone().into());
            }
            let parent_value = Value::Object(parent_info);

            for part in &mut output.content {
                let metadata_map = part.metadata.get_or_insert_with(HashMap::new);

                if let Some(resource_value) = metadata_map.get_mut("resource") {
                    if let Some(resource_obj) = resource_value.as_object_mut() {
                        if !resource_obj.contains_key("parent") {
                            resource_obj.insert("parent".to_string(), parent_value.clone());
                        }
                    }
                } else {
                    metadata_map.insert(
                        "resource".to_string(),
                        json!({ "parent": parent_value.clone() }),
                    );
                }
            }
            Ok(output)
        }
    };

    let mut builder = ActionBuilder::new(ActionType::Resource, name.clone(), runner_closure)
        .with_metadata(metadata);

    if let Some(desc) = opts.description {
        builder = builder.with_description(desc);
    }

    let action = builder.build();

    let resource_action = Arc::new(action);
    registry.register_action(name.as_str(), (*resource_action).clone())?;
    Ok(resource_action)
}

//
// Find Resource
//

/// Converts a URI template string to a Regex for matching.
fn template_to_regex(template_str: &str) -> Result<Regex> {
    let re = Regex::new(r"\{[^}]+\}")
        .map_err(|e| Error::new_internal(format!("Failed to create regex parser: {}", e)))?;
    let pattern = re.replace_all(template_str, "[^/]+");
    let full_pattern = format!("^{}$", pattern);
    Regex::new(&full_pattern)
        .map_err(|e| Error::new_internal(format!("Invalid regex from template: {}", e)))
}

/// Finds a matching resource in the registry. If not found returns `None`.
pub async fn find_matching_resource(
    registry: &Registry,
    input: &ResourceInput,
) -> Result<Option<Arc<ResourceAction>>> {
    let all_actions = registry.list_actions().await;
    for (key, action) in all_actions {
        if key.starts_with("/resource/") {
            let metadata = action.metadata();
            if let Some(resource_meta) = metadata.metadata.get("resource") {
                let uri = resource_meta.get("uri").and_then(|v| v.as_str());
                let template_str = resource_meta.get("template").and_then(|v| v.as_str());

                let is_match = if let Some(t_str) = template_str {
                    if let Ok(re) = template_to_regex(t_str) {
                        re.is_match(&input.uri)
                    } else {
                        false
                    }
                } else if let Some(u) = uri {
                    u == input.uri
                } else {
                    false
                };

                if is_match {
                    let name = action.name();
                    let concrete_action_ref = action
                        .as_any()
                        .downcast_ref::<ResourceAction>()
                        .ok_or_else(|| {
                            Error::new_internal(format!(
                                "Mismatched type for resource action '{}'.",
                                name
                            ))
                        })?;
                    return Ok(Some(Arc::new(concrete_action_ref.clone())));
                }
            }
        }
    }
    Ok(None)
}
