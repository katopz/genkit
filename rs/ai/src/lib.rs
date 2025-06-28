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

//! # Genkit AI
//!
//! This crate provides the generative AI APIs for the Genkit framework in Rust.
//! It includes definitions for models, embedders, retrievers, and other AI components.

// Declare modules corresponding to the files we will create.
pub mod chat;
pub mod check_operation;
pub mod document;
pub mod embedder;
pub mod evaluator;
pub mod extract;
pub mod formats;
pub mod generate;
pub mod message;
pub mod model;
pub mod prompt;
pub mod reranker;
pub mod resource;
pub mod retriever;
pub mod session;
#[cfg(test)]
pub mod testing;
pub mod tool;
pub mod types;

// Re-export key components for easier access.
pub use document::Document;
pub use embedder::{embed, EmbedderAction as Embedder, EmbedderRef};
pub use generate::{
    generate, generate_stream, GenerateOptions, GenerateResponse, GenerateStreamResponse,
};
pub use model::{Model, ModelRef};
pub use prompt::{define_prompt, prompt, ExecutablePrompt};
pub use retriever::{retrieve, RetrieverAction as Retriever, RetrieverRef};
pub use tool::{define_tool, ToolAction as Tool};
