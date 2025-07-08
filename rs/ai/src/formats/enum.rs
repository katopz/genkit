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

//! # Enum Output Formatter
//!
//! This module provides a formatter for string-based enum outputs.
//! It is the Rust equivalent of `formats/enum.ts`.

use super::types::{Format, Formatter, FormatterConfig};
use crate::message::Message;
use schemars::Schema;
use serde_json::Value;
use std::sync::Arc;

/// A struct that implements the `Format` trait for enum-like strings.
#[derive(Debug)]
struct EnumFormat {
    instructions: Option<String>,
}

impl Format for EnumFormat {
    /// Parses the enum value from a complete `Message`.
    ///
    /// This implementation trims whitespace and surrounding quotes from the text.
    fn parse_message(&self, message: &Message) -> Value {
        Value::String(message.text().trim().trim_matches('"').to_string())
    }

    /// Provides model instructions based on the schema's enum values.
    fn instructions(&self) -> Option<String> {
        self.instructions.clone()
    }
}

/// Creates and configures the `enum` formatter.
pub fn enum_formatter() -> Formatter {
    Formatter {
        name: "enum".to_string(),
        config: FormatterConfig {
            content_type: Some("text/enum".to_string()),
            constrained: Some(true),
            ..Default::default()
        },
        handler: Arc::new(|schema: Option<&Schema>| {
            let mut instructions: Option<String> = None;
            if let Some(s) = schema {
                if let Some(const_value) = s.as_object().and_then(|o| o.get("const")) {
                    if let Some(enum_values) = const_value.as_array() {
                        let values_str = enum_values
                            .iter()
                            .map(|v| v.to_string().trim_matches('"').to_string())
                            .collect::<Vec<_>>()
                            .join("\n");
                        instructions = Some(format!(
                            "Output should be ONLY one of the following enum values. Do not output any additional information or add quotes.\n\n{}",
                            values_str
                        ));
                    }
                }
            }
            Box::new(EnumFormat { instructions })
        }),
    }
}
