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

// Re-export `JsonSchema` from the schemars crate, as it's a fundamental part of the API.
pub use schemars::JsonSchema;
// Re-export schema-related types and functions from `genkit-core`.
pub use genkit_core::schema::{
    parse_schema, ProvidedSchema, ValidationError, ValidationErrorDetail,
};

// Note: The following items from the TypeScript version are not yet available in Rust:
// - toJsonSchema
// - validateSchema
// - ValidationResponse
