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

//! # Genkit Public API Facade
//!
//! This module re-exports the primary public-facing types, functions, and
//! structs from the underlying `genkit-ai` and `genkit-core` crates, providing
//! a single, convenient entry point for developers using the Genkit framework.

//
// Re-exports from `genkit-ai`
//

pub use genkit_ai::{
    self,
    chat::{Chat, MAIN_THREAD, SESSION_ID_ATTR, THREAD_NAME_ATTR},
    check_operation::check_operation,
    document::{
        Document, Media, Part, ToolRequest, ToolRequestPart, ToolResponse, ToolResponsePart,
    },
    embedder::{
        define_embedder, embed, embedder_ref, EmbedParams, EmbedRequest, EmbedResponse,
        EmbedderAction, EmbedderArgument, EmbedderInfo, EmbedderRef, Embedding, EmbeddingBatch,
    },
    evaluator::{
        define_evaluator, evaluate, evaluator_ref, BaseDataPoint, BaseEvalDataPoint, Dataset,
        EvalResponse, EvalResponses, EvalStatusEnum, EvaluatorAction, EvaluatorArgument,
        EvaluatorInfo, EvaluatorParams, EvaluatorRef, Score,
    },
    extract::{extract_items, extract_json, parse_partial_json},
    generate::{
        action::run_with_streaming_callback, generate, generate_operation, generate_stream,
        to_generate_request, GenerateOptions, GenerateResponse, GenerateResponseChunk,
        GenerateStreamResponse, GenerationBlockedError, GenerationResponseError, OutputOptions,
        ResumeOptions, ToolChoice,
    },
    message::{Message, MessageData, MessageParser, Role},
    model::{
        define_background_model, define_model, middleware::ModelMiddleware, CandidateData,
        FinishReason, GenerateRequest, GenerateResponseChunkData, GenerateResponseData,
        GenerationCommonConfig, GenerationUsage, Model, ModelAction, ModelInfo, ModelRef,
    },
    prompt::{
        define_prompt, is_executable_prompt, prompt, ExecutablePrompt, PromptAction, PromptConfig,
        PromptGenerateOptions,
    },
    reranker::{
        define_reranker, rerank, reranker_ref, RankedDocument, RerankerAction, RerankerArgument,
        RerankerInfo, RerankerParams, RerankerRef,
    },
    resource::{
        define_resource, dynamic_resource, find_matching_resource, is_dynamic_resource_action,
        DynamicResource, ResourceAction, ResourceFn, ResourceInput, ResourceOptions,
        ResourceOutput,
    },
    retriever::{
        define_indexer, define_retriever, index, indexer_ref, retrieve, retriever_ref,
        IndexerAction, IndexerArgument, IndexerInfo, IndexerParams, IndexerRef, RetrieverAction,
        RetrieverArgument, RetrieverInfo, RetrieverParams, RetrieverRef,
    },
    session::{get_current_session, run_with_session, Session, SessionData, SessionStore},
    tool::{
        define_interrupt, define_tool, dynamic_tool, to_tool_definition, InterruptConfig,
        ToolAction, ToolArgument, ToolConfig, ToolDefinition, ToolInterruptError,
    },
    types::{to_tool_wire_format, LlmResponse, LlmStats, Tool, ToolCall},
};

//
// Re-exports from `genkit-core`
//
pub use genkit_core::{
    action::{
        Action, ActionBuilder, ActionFn, ActionFnArg, ActionResult, StreamingCallback,
        StreamingResponse,
    },
    background_action::Operation,
    context::{
        api_key, get_context, get_flow_context, run_with_context, run_with_flow_context,
        ActionContext, ApiKeyPolicy, ContextProvider, RequestData,
    },
    error::{get_error_message, Error as GenkitError},
    flow::{define_flow, run, Flow},
    logging::{debug, error, info, log_structured, log_structured_error, trace, warn},
    plugin::Plugin,
    reflection::{start as start_reflection_server, ReflectionServerOptions},
    registry::{ActionType, Registry},
    schema::{define_json_schema, define_schema, schema_for},
    status::{Status, StatusCode},
    telemetry::TelemetryConfig,
    tracing::{enable_telemetry, flush_tracing},
    utils::{get_current_env, is_dev_env},
};
