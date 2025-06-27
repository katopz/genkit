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

//! # Genkit Actions
//!
//! This module defines the core `Action` abstraction for the Genkit framework,
//! providing a Rust-idiomatic equivalent of `action.ts`. An `Action` is a
//! self-describing, callable unit of work that forms the building block
//! for more complex flows and tool integrations.

use crate::async_utils::{channel, Channel};
use crate::context::{self, ActionContext};
use crate::error::{Error, Result};
use crate::registry::{ActionType, Registry};
use crate::schema::{parse_schema, ProvidedSchema};
use crate::status::StatusCode;

use crate::schema::schema_for;
use crate::tracing::{self, TraceContext};
use async_trait::async_trait;
use futures::{Future, Stream, StreamExt};
use schemars::{schema::RootSchema, JsonSchema};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Debug;
use std::pin::Pin;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Metadata describing a Genkit `Action`.
#[derive(Debug, Clone)]
pub struct ActionMetadata {
    pub action_type: ActionType,
    pub name: String,
    pub description: Option<String>,
    pub input_schema: Option<RootSchema>,
    pub output_schema: Option<RootSchema>,
    pub stream_schema: Option<RootSchema>,
    pub metadata: HashMap<String, Value>,
}

/// The result of a non-streaming action execution, including telemetry data.
#[derive(Debug, Serialize)]
pub struct ActionResult<O> {
    pub result: O,
    pub telemetry: TelemetryInfo,
}

/// Telemetry information associated with a single action execution.
#[derive(Debug, Serialize, Clone)]
pub struct TelemetryInfo {
    pub trace_id: String,
    pub span_id: String,
}

/// The response from a streaming action.
pub struct StreamingResponse<O, S> {
    /// A stream that yields chunks of type `S`.
    pub stream: Pin<Box<dyn Stream<Item = S> + Send>>,
    /// A future that resolves to the final output of type `O` when the stream is complete.
    pub output: Pin<Box<dyn Future<Output = Result<O>> + Send>>,
}

/// Arguments passed to the function that implements an action's logic.
pub struct ActionFnArg<S: Send + 'static> {
    /// Indicates whether the caller requested a streaming response.
    pub streaming_requested: bool,
    /// A channel sender for pushing streaming chunks back to the caller.
    pub chunk_sender: Channel<S>,
    /// Additional runtime context, such as authentication information.
    pub context: Option<ActionContext>,
    /// OpenTelemetry trace context.
    pub trace: TraceContext,
    /// A token that signals when the action should be cancelled.
    pub abort_signal: CancellationToken,
}

/// A trait that defines the executable logic of a Genkit action.
///
/// This allows for different action functions (e.g., closures) to be stored
/// and called polymorphically.
#[async_trait]
pub trait ActionFn<I, O, S: Send + 'static>: Send + Sync {
    async fn run(&self, input: I, args: ActionFnArg<S>) -> Result<O>;
}

/// Blanket implementation of `ActionFn` for any suitable async closure.
#[async_trait]
impl<F, Fut, I, O, S> ActionFn<I, O, S> for F
where
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
    I: Send + 'static,
    O: Send + 'static,
    S: Send + 'static,
{
    async fn run(&self, input: I, args: ActionFnArg<S>) -> Result<O> {
        (self)(input, args).await
    }
}

/// A self-describing, callable unit of work.
///
/// `Action` is the central abstraction in Genkit. It encapsulates metadata
/// and an executable function, and is managed by a `Registry`.
pub struct Action<I, O, S> {
    pub meta: Arc<ActionMetadata>,
    pub func: Arc<dyn ActionFn<I, O, S>>,
}

impl<I, O, S> Clone for Action<I, O, S> {
    fn clone(&self) -> Self {
        Self {
            meta: self.meta.clone(),
            func: self.func.clone(),
        }
    }
}

impl<I, O, S> Action<I, O, S>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + 'static,
{
    /// Executes the action with a raw JSON value and returns the final result.
    ///
    /// This is the primary method for non-streaming invocation. It handles
    /// input parsing, context management, tracing, and output validation.
    pub async fn run_http(
        &self,
        input: Value,
        context: Option<ActionContext>,
    ) -> Result<ActionResult<O>> {
        let parsed_input: I = match self.meta.input_schema.clone() {
            Some(schema_def) => parse_schema(input, ProvidedSchema::FromType(schema_def))?,
            None => serde_json::from_value(input).map_err(|e| {
                Error::new_user_facing(
                    StatusCode::InvalidArgument,
                    format!("Failed to parse input: {}", e),
                    None,
                )
            })?,
        };

        let (result, telemetry) =
            tracing::in_new_span(self.meta.name.clone(), |trace_context| async {
                let (chunk_tx, _chunk_rx) = channel();
                let args = ActionFnArg {
                    streaming_requested: false,
                    chunk_sender: chunk_tx,
                    context: context.clone(),
                    trace: trace_context,
                    abort_signal: CancellationToken::new(),
                };

                let fut = self.func.run(parsed_input, args);

                if let Some(ctx) = context {
                    context::run_with_context(ctx, fut).await
                } else {
                    fut.await
                }
            })
            .await?;

        // TODO: Add output schema validation.

        Ok(ActionResult { result, telemetry })
    }

    /// Executes the action and provides a streaming response.
    pub fn stream(&self, input: I, context: Option<ActionContext>) -> StreamingResponse<O, S> {
        let (chunk_tx, chunk_rx) = channel::<S>();
        let func = self.func.clone();
        let meta_name = self.meta.name.clone();

        let future = Box::pin(async move {
            let (result, _telemetry) = tracing::in_new_span(meta_name, |trace_context| async {
                let args = ActionFnArg {
                    streaming_requested: true,
                    chunk_sender: chunk_tx,
                    context: context.clone(),
                    trace: trace_context,
                    abort_signal: CancellationToken::new(),
                };

                let fut = func.run(input, args);

                if let Some(ctx) = context {
                    context::run_with_context(ctx, fut).await
                } else {
                    fut.await
                }
            })
            .await?;

            // TODO: Add output schema validation.
            Ok(result)
        });

        StreamingResponse {
            stream: Box::pin(chunk_rx.filter_map(|item| async { item.ok() })),
            output: future,
        }
    }
}

/// Builder for creating a new `Action`.
pub struct ActionBuilder<I, O, S, F> {
    action_type: ActionType,
    name: String,
    description: Option<String>,
    metadata: Option<HashMap<String, Value>>,
    func: F,
    _marker: std::marker::PhantomData<(I, O, S)>,
}

impl<I, O, S, F, Fut> ActionBuilder<I, O, S, F>
where
    I: DeserializeOwned + JsonSchema + Send + Sync + 'static,
    O: Serialize + JsonSchema + Send + Sync + 'static,
    S: Serialize + JsonSchema + Send + Sync + 'static,
    F: Fn(I, ActionFnArg<S>) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<O>> + Send,
{
    pub fn new(action_type: ActionType, name: impl Into<String>, func: F) -> Self {
        Self {
            action_type,
            name: name.into(),
            description: None,
            metadata: None,
            func,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Finalizes the build and registers the action.
    pub fn build(self, _registry: &mut Registry) -> Action<I, O, S> {
        let meta = Arc::new(ActionMetadata {
            action_type: self.action_type,
            name: self.name,
            description: self.description,
            input_schema: Some(schema_for::<I>()),
            output_schema: Some(schema_for::<O>()),
            stream_schema: Some(schema_for::<S>()),
            metadata: self.metadata.unwrap_or_default(),
        });

        Action {
            meta,
            func: Arc::new(self.func),
        }
    }
}
