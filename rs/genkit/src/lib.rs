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

//! # Genkit
//!
//! The main Genkit library for Rust, providing tools for building AI-powered flows.
//!
//! This crate is the Rust equivalent of the main `genkit` npm package. It provides
//! high-level APIs for defining flows, models, tools, and other key components
//! of a Genkit application.

// Silence warnings for missing implementations during development.
#![allow(dead_code)]
#![allow(unused_variables)]

// Public modules that form the core API of the Genkit library.
pub mod client;
pub mod embedder;
pub mod error;
pub mod flow;
pub mod model;
pub mod prompt;
pub mod retriever;
pub mod tool;

// Re-export key components for a unified and convenient API.
pub use self::embedder::{embed, Embedder, EmbedderInfo, EmbedderReference};
pub use self::error::{Error, Result};
pub use self::flow::{define_flow, Flow, FlowInfo};
pub use self::model::{
    generate, Candidate, FinishReason, GenerateRequest, GenerateResponse, Message, Model,
    ModelInfo, Part, Role,
};
pub use self::prompt::{define_prompt, Prompt, PromptConfig, PromptDefinition};
pub use self::retriever::{
    index, retrieve, Document, Indexer, IndexerReference, Retriever, RetrieverReference,
};
pub use self::tool::{Tool, ToolDefinition};
pub use genkit_ai::retriever::{IndexerInfo, RetrieverInfo};

// The `tests` module is compiled only when running tests.
#[cfg(test)]
mod tests;
