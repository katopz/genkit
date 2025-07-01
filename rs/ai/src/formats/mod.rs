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

//! # Model Output Formatters
//!
//! This module provides tools for defining and handling various output formats
//! from generative models, such as JSON, JSONL, and plain text. It is the Rust
//! equivalent of the `js/ai/src/formats` directory.

pub mod array;
pub mod r#enum;
pub mod json;
pub mod jsonl;
pub mod text;
pub mod types;

use crate::document::Part;
use crate::generate::OutputOptions;
use crate::message::{MessageData, Role};
use genkit_core::registry::Registry;
use schemars::Schema;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use types::{FormatHandler, Formatter};

// Re-export key public types for easier access.
pub use self::types::{Format, FormatterConfig};

/// Defines a new format and registers it with the given registry.
pub fn define_format(
    registry: &mut Registry,
    name: impl Into<String>,
    config: FormatterConfig,
    handler: FormatHandler,
) -> Formatter {
    let name = name.into();
    let formatter = Formatter {
        name: name.clone(),
        config,
        handler,
    };
    registry.register_value("format", &name, Box::new(formatter.clone()));
    formatter
}

/// Returns a vector containing the default formatters.
pub fn default_formatters() -> Vec<Formatter> {
    vec![
        self::json::json_formatter(),
        self::array::array_formatter(),
        self::text::text_formatter(),
        self::r#enum::enum_formatter(),
        self::jsonl::jsonl_formatter(),
    ]
}

/// Registers the default formatters on a given registry.
///
/// In a real application, this would be part of the framework's initialization.
pub fn configure_formats(registry: &mut Registry) {
    for formatter in default_formatters() {
        // Clone the name before moving the formatter to resolve the borrow checker error.
        let name = formatter.name.clone();
        // Note: The `Registry` will need a method to store these formatters.
        // This is a placeholder for that logic.
        registry.register_value("format", &name, Box::new(formatter));
    }
}

/// Resolves the formatter to use based on output options.
pub async fn resolve_format(
    registry: &Registry,
    output_opts: Option<&OutputOptions>,
) -> Option<Arc<Formatter>> {
    let output_opts = output_opts?;
    // If schema is set but no explicit format is set we default to json.
    if output_opts.schema.is_some() && output_opts.format.is_none() {
        return registry.lookup_value::<Formatter>("format", "json").await;
    }
    if let Some(format) = &output_opts.format {
        return registry.lookup_value::<Formatter>("format", format).await;
    }
    None
}

/// Resolves the instructions to provide to the model.
pub fn resolve_instructions(
    format: Option<&Formatter>,
    schema: Option<&Schema>,
    instructions_option: Option<&Value>,
) -> Option<String> {
    println!(
        "[resolve_instructions] format: {:?}",
        format.map(|f| &f.name)
    );
    println!("[resolve_instructions] schema: {:?}", schema.is_some());
    println!(
        "[resolve_instructions] instructions_option: {:?}",
        instructions_option
    );
    if let Some(Value::String(s)) = instructions_option {
        println!("[resolve_instructions] user provided instructions");
        return Some(s.clone()); // user provided instructions
    }
    if let Some(Value::Bool(false)) = instructions_option {
        println!("[resolve_instructions] user says no instructions");
        return None; // user says no instructions
    }
    let format = format?;
    let handler = (format.handler)(schema);
    let instructions = handler.instructions();
    println!(
        "[resolve_instructions] resolved instructions: {:?}",
        instructions
    );
    instructions
}

/// Injects formatting instructions into the message list.
pub fn inject_instructions(
    messages: &[MessageData],
    instructions: Option<String>,
) -> Vec<MessageData> {
    let instructions = match instructions {
        Some(i) => i,
        None => return messages.to_vec(),
    };

    // bail out if a non-pending output part is already present
    if messages.iter().any(|m| {
        m.content.iter().any(|p| {
            if let Some(meta) = p.metadata.as_ref() {
                if meta.get("purpose") == Some(&Value::String("output".to_string())) {
                    return meta.get("pending") != Some(&Value::Bool(true));
                }
            }
            false
        })
    }) {
        return messages.to_vec();
    }

    let mut new_part_metadata = HashMap::new();
    new_part_metadata.insert("purpose".to_string(), json!("output"));

    let new_part = Part {
        text: Some(instructions),
        metadata: Some(new_part_metadata),
        ..Default::default()
    };

    // find the system message or the last user message
    let target_index = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .or_else(|| messages.iter().position(|m| m.role == Role::System));

    if let Some(target_index) = target_index {
        let mut out_messages = messages.to_vec();
        let m = &mut out_messages[target_index];

        if let Some(part_index) = m.content.iter().position(|p| {
            if let Some(meta) = p.metadata.as_ref() {
                return meta.get("purpose") == Some(&Value::String("output".to_string()))
                    && meta.get("pending") == Some(&Value::Bool(true));
            }
            false
        }) {
            m.content[part_index] = new_part;
        } else {
            m.content.push(new_part);
        }
        return out_messages;
    }

    messages.to_vec()
}
