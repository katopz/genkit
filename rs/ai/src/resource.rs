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
use schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;
use uritemplate::UriTemplate;

//
// Core Types & Structs
//

/// Options for defining a resource.
#[derive(Debug, Clone, Default)]
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
pub type ResourceFn<Fut> = dyn Fn(ResourceInput, ActionContext) -> Fut + Send + Sync;

/// A resource action.
pub type ResourceAction = Action<ResourceInput, ResourceOutput, ()>;

//
// Define Resource
//

/// Defines a resource.
pub fn define_resource<F, Fut>(opts: ResourceOptions, runner: F) -> Result<Arc<ResourceAction>>
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
    metadata.insert("genkit:resource".to_string(), serde_json::json!(true));
    let resource_info = serde_json::json!({
        "uri": opts.uri,
        "template": opts.template,
        "description": opts.description,
    });
    metadata.insert("resource".to_string(), resource_info);

    let action = ActionBuilder::new(
        ActionType::Util, // Changed from Resource to Util
        name,
        move |input: ResourceInput, args: genkit_core::action::ActionFnArg<()>| {
            // Handle optional context
            runner(input, args.context.unwrap_or_default())
        },
    )
    .with_metadata(metadata)
    .build();

    Ok(Arc::new(action))
}

//
// Find Resource
//

/// Finds a matching resource in the registry. If not found returns `None`.
pub async fn find_matching_resource(
    registry: &Registry,
    input: &ResourceInput,
) -> Result<Option<Arc<ResourceAction>>> {
    let all_actions = registry.list_actions().await;
    for (key, action) in all_actions {
        if key.starts_with("/util/") {
            // Changed from /resource/ to /util/
            let metadata = action.metadata();
            // Check for the resource marker
            if !metadata
                .metadata
                .get("genkit:resource")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
            {
                continue;
            }
            if let Some(resource_meta) = metadata.metadata.get("resource") {
                let uri = resource_meta.get("uri").and_then(|v| v.as_str());
                let template_str = resource_meta.get("template").and_then(|v| v.as_str());

                let is_match = if let Some(t_str) = template_str {
                    UriTemplate::new(t_str).build() == input.uri
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
