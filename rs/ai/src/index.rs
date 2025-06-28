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

// Note: This file acts as a facade, re-exporting the public API of the crate.
// In Rust, `lib.rs` is the conventional crate root for a library.

pub use crate::check_operation::check_operation;
pub use crate::document::{Document, Media, Part, ToolRequest, ToolResponse};
pub use crate::embedder::{
    define_embedder, embed, embedder_ref, EmbedParams, EmbedRequest, EmbedResponse, EmbedderAction,
    EmbedderArgument, EmbedderInfo, EmbedderRef, Embedding, EmbeddingBatch,
};
pub use crate::evaluator::{
    define_evaluator, evaluate, evaluator_ref, BaseDataPoint, BaseEvalDataPoint, Dataset,
    EvalResponse, EvalResponses, EvalStatusEnum, EvaluatorAction, EvaluatorArgument, EvaluatorInfo,
    EvaluatorParams, EvaluatorRef, Score,
};
pub use crate::generate::{
    generate, generate_stream, to_generate_request, GenerateOptions, GenerateResponse,
    GenerateResponseChunk, GenerateStreamResponse, GenerationBlockedError, GenerationResponseError,
    OutputOptions, ToolChoice,
};
pub use crate::message::{Message, MessageData, Role};
pub use crate::model::{
    define_background_model, define_model, model_ref, CandidateData, FinishReason, GenerateRequest,
    GenerateResponseChunkData, GenerateResponseData, GenerationCommonConfig, GenerationUsage,
    Model, ModelAction, ModelInfo, ModelRef,
};
pub use crate::prompt::{
    define_prompt, is_executable_prompt, prompt, ExecutablePrompt, PromptAction, PromptConfig,
    PromptGenerateOptions,
};
pub use crate::reranker::{
    define_reranker, rerank, reranker_ref, RankedDocument, RerankerAction, RerankerArgument,
    RerankerInfo, RerankerParams, RerankerRef,
};
pub use crate::resource::{
    define_resource, find_matching_resource, ResourceAction, ResourceInput, ResourceOptions,
    ResourceOutput,
};
pub use crate::retriever::{
    define_indexer, define_retriever, index, indexer_ref, retrieve, retriever_ref, IndexerAction,
    IndexerArgument, IndexerInfo, IndexerParams, IndexerRef, RetrieverAction, RetrieverArgument,
    RetrieverInfo, RetrieverParams, RetrieverRef,
};
pub use crate::tool::{
    define_tool, to_tool_definition, ToolAction, ToolArgument, ToolConfig, ToolDefinition,
};
pub use crate::types::{to_tool_wire_format, LlmResponse, LlmStats, Tool, ToolCall};
