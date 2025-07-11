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

// Declare all modules that make up the library.
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

// Re-export the public API, combining exports from the old lib.rs and index.rs.

pub use self::chat::{Chat, MAIN_THREAD, SESSION_ID_ATTR, THREAD_NAME_ATTR};
pub use self::check_operation::check_operation;
pub use self::document::{
    Document, Media, Part, ToolRequest, ToolRequestPart, ToolResponse, ToolResponsePart,
};
pub use self::embedder::{
    define_embedder, embed, embedder_ref, EmbedParams, EmbedRequest, EmbedResponse, EmbedderAction,
    EmbedderArgument, EmbedderInfo, EmbedderRef, Embedding, EmbeddingBatch,
};
pub use self::evaluator::{
    define_evaluator, evaluate, evaluator_ref, BaseDataPoint, BaseEvalDataPoint, Dataset,
    EvalResponse, EvalResponses, EvalStatusEnum, EvaluatorAction, EvaluatorArgument, EvaluatorInfo,
    EvaluatorParams, EvaluatorRef, Score,
};
pub use self::extract::{extract_items, extract_json, parse_partial_json};
pub use self::generate::{
    generate, generate_operation, generate_stream, to_generate_request, GenerateOptions,
    GenerateResponse, GenerateResponseChunk, GenerateStreamResponse, GenerationBlockedError,
    GenerationResponseError, OutputOptions, ToolChoice,
};
pub use self::message::{Message, MessageData, MessageParser, Role};
pub use self::model::{
    define_background_model, define_model, CandidateData, FinishReason, GenerateRequest,
    GenerateResponseChunkData, GenerateResponseData, GenerationCommonConfig, GenerationUsage,
    Model, ModelAction, ModelInfo, ModelRef,
};
pub use self::prompt::{
    define_prompt, is_executable_prompt, prompt, ExecutablePrompt, PromptAction, PromptConfig,
    PromptGenerateOptions,
};
pub use self::reranker::{
    define_reranker, rerank, reranker_ref, RankedDocument, RerankerAction, RerankerArgument,
    RerankerInfo, RerankerParams, RerankerRef,
};
pub use self::resource::{
    define_resource, dynamic_resource, find_matching_resource, is_dynamic_resource_action,
    DynamicResource, ResourceAction, ResourceInput, ResourceOptions, ResourceOutput,
};
pub use self::retriever::{
    define_indexer, define_retriever, index, indexer_ref, retrieve, retriever_ref, IndexerAction,
    IndexerArgument, IndexerInfo, IndexerParams, IndexerRef, RetrieverAction, RetrieverArgument,
    RetrieverInfo, RetrieverParams, RetrieverRef,
};
pub use self::session::{get_current_session, run_with_session, Session, SessionStore};
pub use self::tool::{
    define_interrupt, define_tool, dynamic_tool, to_tool_definition, ToolAction, ToolArgument,
    ToolConfig, ToolDefinition,
};
pub use self::types::{to_tool_wire_format, LlmResponse, LlmStats, Tool, ToolCall};
use genkit_core::registry::Registry;

/// Configures the AI-related components on the given registry.
///
/// This function should be called to initialize the default AI plugins and
/// components, such as formatters.
pub fn configure_ai(registry: &mut Registry) {
    self::formats::configure_formats(registry);
}
