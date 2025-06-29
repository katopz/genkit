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

//! # Genkit Model Facade
//!
//! This module re-exports model-related types and functions from the
//! `genkit-ai` crate, providing a consistent public API for the `genkit` crate.

// Main generation function
pub use genkit_ai::generate::generate;

// Core data structures
pub use genkit_ai::document::Part;
pub use genkit_ai::generate::GenerateResponse;
pub use genkit_ai::message::{Message, Role};
pub use genkit_ai::model::{
    CandidateData as Candidate, FinishReason, GenerateRequest, Model, ModelInfo,
};
