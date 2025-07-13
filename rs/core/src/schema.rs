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

//! # Schema Definition and Validation
//!
//! This module provides tools for defining, validating, and parsing data schemas.
//! It is the Rust equivalent of `schema.ts`, leveraging `schemars` for generating
//! JSON schemas from Rust types and `jsonschema` for validation.

use crate::error::{Error, Result};
use jsonschema::Draft;
use schemars::{JsonSchema, Schema};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::fmt::{self, Debug, Display};
use thiserror::Error as ThisError;
/// An error that occurs during schema validation.
///
/// Contains detailed information about the validation failures, the data that
/// was being validated, and the schema it was validated against.
#[derive(ThisError, Debug, Serialize)]
pub struct ValidationError {
    errors: Vec<ValidationErrorDetail>,
    data: Value,
    schema: Schema,
}

impl Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error_list: String = self
            .errors
            .iter()
            .map(|e| format!("- {}", e))
            .collect::<Vec<_>>()
            .join("\n");

        write!(
            f,
            "Schema validation failed. Parse Errors:\n\n{}\n\nProvided data:\n\n{}\n\nRequired JSON schema:\n\n{}",
            error_list,
            serde_json::to_string_pretty(&self.data).unwrap_or_else(|_| "Invalid JSON data".to_string()),
            serde_json::to_string_pretty(&self.schema).unwrap_or_else(|_| "Invalid JSON schema".to_string())
        )
    }
}

impl ValidationError {
    pub fn new(errors: Vec<ValidationErrorDetail>, data: Value, schema: Schema) -> Self {
        Self {
            errors,
            data,
            schema,
        }
    }

    /// Returns a slice of the detailed validation errors.
    pub fn errors(&self) -> &[ValidationErrorDetail] {
        &self.errors
    }
}

/// Contains details for a single schema validation failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationErrorDetail {
    /// The JSON pointer path to the field that failed validation.
    pub path: String,
    /// A message describing the validation failure.
    pub message: String,
}

impl Display for ValidationErrorDetail {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "path: `{}`, message: `{}`", self.path, self.message)
    }
}

/// A wrapper for different ways a schema can be provided.
/// In Rust, this is simplified as we primarily derive schemas from types.
#[allow(clippy::large_enum_variant)]
#[derive(Clone)]
pub enum ProvidedSchema {
    /// A schema derived from a Rust type implementing `JsonSchema`.
    FromType(Schema),
    /// A raw JSON schema value.
    Raw(Value),
}

/// Generates a `Schema` for a given type that implements `JsonSchema`.
pub fn schema_for<T: JsonSchema>() -> Schema {
    schemars::schema_for!(T)
}

/// Parses a raw JSON value against a given schema.
///
/// This function first validates the data against the provided schema and, if
/// successful, deserializes the data into an instance of type `T`.
///
/// # Arguments
///
/// * `data` - The JSON value to parse and deserialize.
/// * `schema` - The schema to validate against.
///
/// # Returns
///
/// A `Result` containing the deserialized instance of `T` on success, or an
/// `Error` on failure.
pub fn parse_schema<T: DeserializeOwned>(data: Value, schema: ProvidedSchema) -> Result<T> {
    let schema_value = match &schema {
        ProvidedSchema::FromType(root_schema) => serde_json::to_value(root_schema)
            .map_err(|e| Error::new_internal(format!("Failed to serialize schema: {}", e)))?,
        ProvidedSchema::Raw(value) => value.clone(),
    };

    let validator = jsonschema::options()
        .with_draft(Draft::Draft202012)
        .build(&schema_value)
        .map_err(|e| Error::new_internal(format!("Invalid schema: {}", e)))?;

    let errors: Vec<_> = validator.iter_errors(&data).collect();
    if !errors.is_empty() {
        let details = errors
            .into_iter()
            .map(|e| ValidationErrorDetail {
                path: e.instance_path.to_string(),
                message: e.to_string(),
            })
            .collect();
        let root_schema = match schema {
            ProvidedSchema::FromType(rs) => rs,
            ProvidedSchema::Raw(val) => serde_json::from_value(val).unwrap_or_default(),
        };
        return Err(Error::Validation(Box::new(ValidationError::new(
            details,
            data.clone(),
            root_schema,
        ))));
    }

    serde_json::from_value(data)
        .map_err(|e| Error::new_internal(format!("Failed to deserialize after validation: {}", e)))
}

/// Registers a Rust type as a named schema object in the Genkit registry.
///
/// This is a placeholder for functionality that will be implemented within the `Registry`.
#[doc(hidden)]
pub fn define_schema<T: JsonSchema + 'static>(_name: &str) {
    unimplemented!("define_schema is not yet implemented");
}

/// Registers a raw JSON schema as a named schema object in the Genkit registry.
///
/// This is a placeholder for functionality that will be implemented within the `Registry`.
#[doc(hidden)]
pub fn define_json_schema(_name: &str, _json_schema: Schema) {
    unimplemented!("define_json_schema is not yet implemented");
}
