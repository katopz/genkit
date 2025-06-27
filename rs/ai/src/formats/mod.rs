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

use genkit_core::registry::Registry;
use types::Formatter;

// Re-export key public types for easier access.
pub use self::types::{Format, FormatterConfig};

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
